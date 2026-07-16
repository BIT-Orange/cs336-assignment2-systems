"""Fully Sharded Data Parallel (FSDP) implementation."""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Optional


class _AllGather(torch.autograd.Function):
    """Differentiable all-gather for weight reconstruction.

    Forward: gather weight shards -> full weight tensor
    Backward: all-reduce the gradient (average) -> return this rank's shard
    """

    @staticmethod
    def forward(ctx, input, dim, world_size):
        ctx.dim = dim
        ctx.world_size = world_size
        ctx.input_shape = input.shape

        gathered = [torch.empty_like(input) for _ in range(world_size)]
        dist.all_gather(gathered, input.contiguous())
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        world_size = ctx.world_size
        shard_size = ctx.input_shape[dim]

        # Cast to fp32 for more precise all-reduce, then cast back
        grad_fp32 = grad_output.float()
        dist.all_reduce(grad_fp32, op=dist.ReduceOp.AVG)

        rank = dist.get_rank()
        result = grad_fp32.split(shard_size, dim=dim)[rank].contiguous()
        # Return in the same dtype as the input to match autograd expectations
        return result.to(grad_output.dtype), None, None


def _is_linear(module):
    if isinstance(module, nn.Linear):
        return True
    if type(module).__name__ == 'Linear' and hasattr(module, 'weight') and module.weight.dim() == 2:
        return True
    return False


def _is_embedding(module):
    if isinstance(module, nn.Embedding):
        return True
    if type(module).__name__ == 'Embedding' and hasattr(module, 'weight') and module.weight.dim() == 2:
        return True
    return False


class FSDPLinear(nn.Module):
    """Wrapper for Linear layers that shards weights across FSDP ranks."""

    def __init__(self, weight_data, bias_data=None, compute_dtype=None, original_cls=nn.Module):
        nn.Module.__init__(self)
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._original_cls = original_cls

        self.out_features = weight_data.shape[0]
        self.in_features = weight_data.shape[1]
        self.has_bias = bias_data is not None

        self.shard_size = self.out_features // self.world_size
        assert self.out_features % self.world_size == 0

        self.register_buffer("_full_weight", weight_data.clone().to(torch.float32))

        start = self.rank * self.shard_size
        end = start + self.shard_size
        self.weight = nn.Parameter(
            self._full_weight[start:end, :].clone(), requires_grad=True
        )

        if self.has_bias:
            self.bias = nn.Parameter(bias_data.clone().to(torch.float32), requires_grad=True)
        else:
            self.register_parameter("bias", None)

        if self.compute_dtype is not None:
            self.weight.register_post_accumulate_grad_hook(self._cast_grad_to_fp32)

    @staticmethod
    def _cast_grad_to_fp32(param):
        if param.grad is not None:
            param.grad = param.grad.to(torch.float32)

    def forward(self, x):
        if self.compute_dtype is not None:
            w_shard = self.weight.to(self.compute_dtype)
        else:
            w_shard = self.weight

        full_weight = _AllGather.apply(w_shard, 0, self.world_size)

        from einops import einsum
        output = einsum(x, full_weight, "... d_in, d_out d_in -> ... d_out")

        if self.bias is not None:
            output = output + self.bias

        return output

    def sync_master_weight(self):
        start = self.rank * self.shard_size
        end = start + self.shard_size
        self._full_weight[start:end, :] = self.weight.data


class FSDPEmbedding(nn.Module):
    """Wrapper for Embedding layers that shards weights across FSDP ranks."""

    def __init__(self, weight_data, compute_dtype=None, original_cls=nn.Module):
        nn.Module.__init__(self)
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._original_cls = original_cls

        self.vocab_size = weight_data.shape[0]
        self.d_model = weight_data.shape[1]

        self.shard_size = self.d_model // self.world_size
        assert self.d_model % self.world_size == 0

        self.register_buffer("_full_weight", weight_data.clone().to(torch.float32))

        start = self.rank * self.shard_size
        end = start + self.shard_size
        self.weight = nn.Parameter(
            self._full_weight[:, start:end].clone(), requires_grad=True
        )

        if self.compute_dtype is not None:
            self.weight.register_post_accumulate_grad_hook(self._cast_grad_to_fp32)

    @staticmethod
    def _cast_grad_to_fp32(param):
        if param.grad is not None:
            param.grad = param.grad.to(torch.float32)

    def forward(self, token_ids):
        if self.compute_dtype is not None:
            w_shard = self.weight.to(self.compute_dtype)
        else:
            w_shard = self.weight

        full_weight = _AllGather.apply(w_shard, 1, self.world_size)

        return full_weight[token_ids, :]

    def sync_master_weight(self):
        start = self.rank * self.shard_size
        end = start + self.shard_size
        self._full_weight[:, start:end] = self.weight.data


def _make_fsdp_linear_subclass(original_cls):
    class FSDPLinearCompat(FSDPLinear, original_cls):
        def __init__(self, weight_data, bias_data=None, compute_dtype=None):
            FSDPLinear.__init__(self, weight_data, bias_data, compute_dtype, original_cls)

        def extra_repr(self):
            return f"d_out={self.out_features}, d_in={self.in_features}"

    FSDPLinearCompat.__name__ = f"FSDPLinear_{original_cls.__name__}"
    FSDPLinearCompat.__qualname__ = FSDPLinearCompat.__name__
    return FSDPLinearCompat


def _make_fsdp_embedding_subclass(original_cls):
    class FSDPEmbeddingCompat(FSDPEmbedding, original_cls):
        def __init__(self, weight_data, compute_dtype=None):
            FSDPEmbedding.__init__(self, weight_data, compute_dtype, original_cls)

        def extra_repr(self):
            return f"vocab_size={self.vocab_size}, d={self.d_model}"

    FSDPEmbeddingCompat.__name__ = f"FSDPEmbedding_{original_cls.__name__}"
    FSDPEmbeddingCompat.__qualname__ = FSDPEmbeddingCompat.__name__
    return FSDPEmbeddingCompat


class FSDP(nn.Module):
    """Fully Sharded Data Parallel wrapper."""

    def __init__(self, module, compute_dtype=None):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.module = module
        self.world_size = dist.get_world_size()
        self._fsdp_layers = []
        self._hooks = []
        self._communication_handles = []

        self._replace_layers()
        self._broadcast_parameters()
        self._register_grad_hooks()

    def _replace_layers(self):
        for name, child in self.module.named_children():
            if _is_linear(child):
                bias = child.bias if hasattr(child, 'bias') and child.bias is not None else None
                cls = _make_fsdp_linear_subclass(type(child))
                fsdp_layer = cls(child.weight.data, bias, self.compute_dtype)
                setattr(self.module, name, fsdp_layer)
                self._fsdp_layers.append(fsdp_layer)
            elif _is_embedding(child):
                cls = _make_fsdp_embedding_subclass(type(child))
                fsdp_layer = cls(child.weight.data, self.compute_dtype)
                setattr(self.module, name, fsdp_layer)
                self._fsdp_layers.append(fsdp_layer)
            else:
                self._replace_layers_in(child)

    def _replace_layers_in(self, module):
        for name, child in module.named_children():
            if _is_linear(child):
                bias = child.bias if hasattr(child, 'bias') and child.bias is not None else None
                cls = _make_fsdp_linear_subclass(type(child))
                fsdp_layer = cls(child.weight.data, bias, self.compute_dtype)
                setattr(module, name, fsdp_layer)
                self._fsdp_layers.append(fsdp_layer)
            elif _is_embedding(child):
                cls = _make_fsdp_embedding_subclass(type(child))
                fsdp_layer = cls(child.weight.data, self.compute_dtype)
                setattr(module, name, fsdp_layer)
                self._fsdp_layers.append(fsdp_layer)
            else:
                self._replace_layers_in(child)

    def _broadcast_parameters(self):
        """Broadcast full weights from rank 0, then re-derive shards."""
        for layer in self._fsdp_layers:
            dist.broadcast(layer._full_weight.data, src=0)

        fsdp_param_ids = set()
        for layer in self._fsdp_layers:
            fsdp_param_ids.add(id(layer.weight))
            if hasattr(layer, 'bias') and layer.bias is not None:
                fsdp_param_ids.add(id(layer.bias))

        for param in self.module.parameters():
            if id(param) not in fsdp_param_ids:
                dist.broadcast(param.data, src=0)

        for layer in self._fsdp_layers:
            start = layer.rank * layer.shard_size
            end = start + layer.shard_size
            if isinstance(layer, FSDPLinear):
                layer.weight.data.copy_(layer._full_weight[start:end, :])
            elif isinstance(layer, FSDPEmbedding):
                layer.weight.data.copy_(layer._full_weight[:, start:end])

    def _register_grad_hooks(self):
        """Register gradient all-reduce hooks for replicated (non-FSDP) parameters only.

        FSDP layer gradients are handled by the _AllGather backward.
        """
        fsdp_param_ids = set()
        for layer in self._fsdp_layers:
            fsdp_param_ids.add(id(layer.weight))
            if hasattr(layer, 'bias') and layer.bias is not None:
                fsdp_param_ids.add(id(layer.bias))

        for param in self.module.parameters():
            if param.requires_grad and id(param) not in fsdp_param_ids:
                hook = param.register_post_accumulate_grad_hook(
                    self._make_allreduce_hook(param)
                )
                self._hooks.append(hook)

    def _make_allreduce_hook(self, param):
        def hook(tensor):
            if param.grad is not None:
                handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
                self._communication_handles.append(handle)
        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self._communication_handles:
            handle.wait()
        self._communication_handles.clear()

        for layer in self._fsdp_layers:
            layer.sync_master_weight()

    def gather_full_params(self):
        state_dict = {}
        for name, param in self.module.named_parameters():
            is_fsdp_shard = False
            shard_layer = None
            for layer in self._fsdp_layers:
                if param is layer.weight:
                    is_fsdp_shard = True
                    shard_layer = layer
                    break

            if is_fsdp_shard:
                if self.compute_dtype is not None:
                    shard = param.data.to(self.compute_dtype)
                else:
                    shard = param.data

                gathered = [torch.empty_like(shard) for _ in range(self.world_size)]
                dist.all_gather(gathered, shard.contiguous())

                if isinstance(shard_layer, FSDPLinear):
                    full_param = torch.cat(gathered, dim=0)
                elif isinstance(shard_layer, FSDPEmbedding):
                    full_param = torch.cat(gathered, dim=1)
                else:
                    full_param = torch.cat(gathered, dim=0)

                state_dict[name] = full_param.to(param.data.dtype)
            else:
                state_dict[name] = param.data.clone()

        return state_dict
