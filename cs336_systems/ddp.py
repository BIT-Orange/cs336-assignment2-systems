"""Distributed Data Parallel (DDP) implementation with overlapping communication."""

import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    """Distributed Data Parallel wrapper that overlaps gradient communication with backward pass.

    This module wraps an arbitrary PyTorch nn.Module and handles:
    1. Broadcasting parameters from rank 0 to all other ranks at initialization
    2. Asynchronously all-reducing individual parameter gradients as they become ready
       during the backward pass
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._communication_handles = []
        self._hooks = []

        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()

        # Register hooks for gradient synchronization
        self._register_hooks()

    def _broadcast_parameters(self):
        """Broadcast all parameters from rank 0 to ensure identical initial weights."""
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def _register_hooks(self):
        """Register post-accumulate grad hooks to trigger async all-reduce on each parameter."""
        for param in self.module.parameters():
            if param.requires_grad:
                hook = param.register_post_accumulate_grad_hook(self._make_allreduce_hook(param))
                self._hooks.append(hook)

    def _make_allreduce_hook(self, param):
        """Create a hook function that async all-reduces (averages) the gradient for the given parameter."""
        def hook(tensor):
            if param.grad is not None:
                handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
                self._communication_handles.append(handle)
        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """Wait for all async all-reduce operations to complete."""
        for handle in self._communication_handles:
            handle.wait()
        self._communication_handles.clear()
