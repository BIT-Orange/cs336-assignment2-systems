"""FlashAttention-2 implementation using pure PyTorch operations."""

import torch
import torch.nn.functional as F
from einops import einsum


class FlashAttentionPyTorch(torch.autograd.Function):
    """FlashAttention-2 implemented with standard PyTorch operations (no Triton).

    This implements the tiled forward pass with online softmax and recomputation
    in the backward pass.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Args:
            Q: (batch_size, n_queries, d)
            K: (batch_size, n_keys, d)
            V: (batch_size, n_keys, d)
            is_causal: if True, apply causal masking
        Returns:
            O: (batch_size, n_queries, d)
        """
        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[1]
        scale = 1.0 / (d ** 0.5)

        # Tile sizes
        Q_TILE_SIZE = min(64, n_queries)
        K_TILE_SIZE = min(64, n_keys)

        # Number of tiles
        Tq = (n_queries + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        Tk = (n_keys + K_TILE_SIZE - 1) // K_TILE_SIZE

        # Initialize output and running statistics
        O = torch.zeros_like(Q)
        l = torch.zeros(batch_size, n_queries, device=Q.device, dtype=torch.float32)
        m = torch.full((batch_size, n_queries), float('-inf'), device=Q.device, dtype=torch.float32)

        # Process each query tile
        for i in range(Tq):
            qi_start = i * Q_TILE_SIZE
            qi_end = min(qi_start + Q_TILE_SIZE, n_queries)
            Qi = Q[:, qi_start:qi_end, :]  # (batch, Q_TILE_SIZE, d)

            # Initialize per-tile running values
            Oi = torch.zeros(batch_size, qi_end - qi_start, d, device=Q.device, dtype=torch.float32)
            li = torch.zeros(batch_size, qi_end - qi_start, device=Q.device, dtype=torch.float32)
            mi = torch.full((batch_size, qi_end - qi_start), float('-inf'), device=Q.device, dtype=torch.float32)

            for j in range(Tk):
                kj_start = j * K_TILE_SIZE
                kj_end = min(kj_start + K_TILE_SIZE, n_keys)
                Kj = K[:, kj_start:kj_end, :]  # (batch, K_TILE_SIZE, d)
                Vj = V[:, kj_start:kj_end, :]  # (batch, K_TILE_SIZE, d)

                # Compute attention scores tile
                Sij = einsum(Qi, Kj, "b q d, b k d -> b q k") * scale  # (batch, Q_TILE_SIZE, K_TILE_SIZE)

                # Apply causal mask if needed
                if is_causal:
                    q_indices = torch.arange(qi_start, qi_end, device=Q.device)[:, None]
                    k_indices = torch.arange(kj_start, kj_end, device=Q.device)[None, :]
                    causal_mask = q_indices >= k_indices
                    Sij = Sij.masked_fill(~causal_mask, float('-inf'))

                # Update running maximum
                mi_new = torch.maximum(mi, Sij.max(dim=-1).values)

                # Compute exp(S - m_new) for current tile
                Pij_tilde = torch.exp(Sij - mi_new.unsqueeze(-1))

                # Update running sum
                correction = torch.exp(mi - mi_new)
                li_new = correction * li + Pij_tilde.sum(dim=-1)

                # Update output
                Oi = Oi * correction.unsqueeze(-1) + einsum(Pij_tilde, Vj, "b q k, b k d -> b q d")

                # Update running stats
                mi = mi_new
                li = li_new

            # Final normalization for this query tile
            Oi = Oi / li.unsqueeze(-1)
            O[:, qi_start:qi_end, :] = Oi

            # Compute logsumexp for this tile
            L_tile = mi + torch.log(li)
            l[:, qi_start:qi_end] = L_tile

        # Save for backward
        ctx.save_for_backward(Q, K, V, O, l)
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale

        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[1]

        # Compute D = rowsum(O * dO)
        D = einsum(O, dO, "b q d, b q d -> b q")

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        Q_TILE_SIZE = ctx.Q_TILE_SIZE
        K_TILE_SIZE = ctx.K_TILE_SIZE
        Tq = (n_queries + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        Tk = (n_keys + K_TILE_SIZE - 1) // K_TILE_SIZE

        # Outer loop over key tiles
        for j in range(Tk):
            kj_start = j * K_TILE_SIZE
            kj_end = min(kj_start + K_TILE_SIZE, n_keys)
            Kj = K[:, kj_start:kj_end, :]
            Vj = V[:, kj_start:kj_end, :]

            # Initialize dK and dV for this key tile
            dKj = torch.zeros(batch_size, kj_end - kj_start, d, device=Q.device, dtype=torch.float32)
            dVj = torch.zeros(batch_size, kj_end - kj_start, d, device=Q.device, dtype=torch.float32)

            for i in range(Tq):
                qi_start = i * Q_TILE_SIZE
                qi_end = min(qi_start + Q_TILE_SIZE, n_queries)
                Qi = Q[:, qi_start:qi_end, :]
                dOi = dO[:, qi_start:qi_end, :]
                Di = D[:, qi_start:qi_end]
                Li = L[:, qi_start:qi_end]

                # Recompute attention scores
                Sij = einsum(Qi, Kj, "b q d, b k d -> b q k") * scale

                if is_causal:
                    q_indices = torch.arange(qi_start, qi_end, device=Q.device)[:, None]
                    k_indices = torch.arange(kj_start, kj_end, device=Q.device)[None, :]
                    causal_mask = q_indices >= k_indices
                    Sij = Sij.masked_fill(~causal_mask, float('-inf'))

                # Recompute attention probabilities
                Pij = torch.exp(Sij - Li.unsqueeze(-1))

                # Compute dV
                dVj = dVj + einsum(Pij, dOi, "b q k, b q d -> b k d")

                # Compute dP
                dPij = einsum(dOi, Vj, "b q d, b k d -> b q k")

                # Compute dS
                dSij = Pij * (dPij - Di.unsqueeze(-1))

                # Accumulate dQ
                dQ[:, qi_start:qi_end, :] += einsum(dSij, Kj, "b q k, b k d -> b q d") * scale

                # Accumulate dK
                dKj = dKj + einsum(dSij, Qi, "b q k, b q d -> b k d") * scale

            dK[:, kj_start:kj_end, :] = dKj
            dV[:, kj_start:kj_end, :] = dVj

        return dQ, dK, dV, None
