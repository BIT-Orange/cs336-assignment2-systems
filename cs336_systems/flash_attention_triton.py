"""FlashAttention-2 implementation using Triton kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index multiplied with the batch stride
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Q tile
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Initialize running values
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), value=float('-inf'), dtype=tl.float32)

    # Create query indices for causal masking
    if IS_CAUSAL:
        q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    # Iterate over key tiles
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load K and V tiles
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Compute attention scores: S = Q @ K^T * scale
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        # Apply causal mask if needed
        if IS_CAUSAL:
            k_indices = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            Sij = tl.where(causal_mask, Sij, -1e6)

        # Update running maximum
        mi_new = tl.maximum(mi, tl.max(Sij, axis=1))

        # Compute exp(S - m_new)
        Pij_tilde = tl.exp(Sij - mi_new[:, None])

        # Update running sum with correction
        correction = tl.exp(mi - mi_new)
        li_new = correction * li + tl.sum(Pij_tilde, axis=1)

        # Update output: O = O * correction + P_tilde @ V
        Oi = Oi * correction[:, None] + tl.dot(Pij_tilde.to(Vj.dtype), Vj)

        # Update running stats
        mi = mi_new
        li = li_new

        # Advance block pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Final normalization
    Oi = Oi / li[:, None]

    # Compute logsumexp
    Li = mi + tl.log(li)

    # Write output
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    """FlashAttention-2 implemented with a Triton forward kernel and PyTorch backward."""

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
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Tensors must be contiguous"

        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[1]
        scale = 1.0 / (d ** 0.5)

        # Tile sizes - must be powers of 2 and at least 16
        Q_TILE_SIZE = triton.next_power_of_2(min(64, n_queries))
        K_TILE_SIZE = triton.next_power_of_2(min(64, n_keys))
        if Q_TILE_SIZE < 16:
            Q_TILE_SIZE = 16
        if K_TILE_SIZE < 16:
            K_TILE_SIZE = 16

        # Ensure d is a power of 2 for tl.dot
        D = triton.next_power_of_2(d)

        # Pad Q, K, V if needed for D dimension
        if D != d:
            Q_pad = torch.zeros(batch_size, n_queries, D, device=Q.device, dtype=Q.dtype)
            K_pad = torch.zeros(batch_size, n_keys, D, device=K.device, dtype=K.dtype)
            V_pad = torch.zeros(batch_size, n_keys, D, device=V.device, dtype=V.dtype)
            Q_pad[:, :, :d] = Q
            K_pad[:, :, :d] = K
            V_pad[:, :, :d] = V
        else:
            Q_pad, K_pad, V_pad = Q, K, V

        O = torch.empty_like(Q_pad)
        L = torch.empty(batch_size, n_queries, device=Q.device, dtype=torch.float32)

        # Launch grid: (Tq, batch_size)
        Tq = triton.cdiv(n_queries, Q_TILE_SIZE)
        grid = (Tq, batch_size)

        flash_fwd_kernel[grid](
            Q_pad, K_pad, V_pad,
            O, L,
            Q_pad.stride(0), Q_pad.stride(1), Q_pad.stride(2),
            K_pad.stride(0), K_pad.stride(1), K_pad.stride(2),
            V_pad.stride(0), V_pad.stride(1), V_pad.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL=is_causal,
        )

        # Remove padding if needed
        if D != d:
            O = O[:, :, :d]

        # Save for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale

        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[1]

        # Compute D = rowsum(O * dO)
        D_vec = (O * dO).sum(dim=-1)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        Q_TILE_SIZE = min(64, n_queries)
        K_TILE_SIZE = min(64, n_keys)

        # Outer loop over key tiles
        Tk = (n_keys + K_TILE_SIZE - 1) // K_TILE_SIZE
        Tq = (n_queries + Q_TILE_SIZE - 1) // Q_TILE_SIZE

        for j in range(Tk):
            kj_start = j * K_TILE_SIZE
            kj_end = min(kj_start + K_TILE_SIZE, n_keys)
            Kj = K[:, kj_start:kj_end, :]
            Vj = V[:, kj_start:kj_end, :]

            dKj = torch.zeros(batch_size, kj_end - kj_start, d, device=Q.device, dtype=torch.float32)
            dVj = torch.zeros(batch_size, kj_end - kj_start, d, device=Q.device, dtype=torch.float32)

            for i in range(Tq):
                qi_start = i * Q_TILE_SIZE
                qi_end = min(qi_start + Q_TILE_SIZE, n_queries)
                Qi = Q[:, qi_start:qi_end, :]
                dOi = dO[:, qi_start:qi_end, :]
                Di = D_vec[:, qi_start:qi_end]
                Li = L[:, qi_start:qi_end]

                # Recompute attention scores
                Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale

                if is_causal:
                    q_indices = torch.arange(qi_start, qi_end, device=Q.device)[:, None]
                    k_indices = torch.arange(kj_start, kj_end, device=Q.device)[None, :]
                    causal_mask = q_indices >= k_indices
                    Sij = Sij.masked_fill(~causal_mask, float('-inf'))

                # Recompute attention probabilities
                Pij = torch.exp(Sij - Li.unsqueeze(-1))

                # Compute dV
                dVj = dVj + torch.matmul(Pij.transpose(-2, -1), dOi)

                # Compute dP
                dPij = torch.matmul(dOi, Vj.transpose(-2, -1))

                # Compute dS
                dSij = Pij * (dPij - Di.unsqueeze(-1))

                # Accumulate dQ
                dQ[:, qi_start:qi_end, :] += torch.matmul(dSij, Kj) * scale

                # Accumulate dK
                dKj = dKj + torch.matmul(dSij.transpose(-2, -1), Qi) * scale

            dK[:, kj_start:kj_end, :] = dKj
            dV[:, kj_start:kj_end, :] = dVj

        return dQ, dK, dV, None
