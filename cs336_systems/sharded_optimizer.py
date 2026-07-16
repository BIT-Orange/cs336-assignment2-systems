"""Sharded Optimizer implementation for optimizer state sharding."""

from typing import Any, Type

import torch
import torch.distributed as dist
import torch.optim as optim


class ShardedOptimizer(optim.Optimizer):
    """Optimizer that shards optimizer states across ranks.

    Each rank only maintains optimizer states for a subset of parameters
    (approximately 1/world_size of total parameters). After each step,
    parameters are synchronized across ranks via broadcast.
    """

    def __init__(self, params, optimizer_cls: Type[optim.Optimizer], **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Collect all parameters
        param_list = list(params)
        if len(param_list) == 0:
            raise ValueError("Empty parameter list")

        # Flatten parameter groups if needed
        all_params = []
        for group in param_list:
            if isinstance(group, dict):
                all_params.extend(group.get("params", []))
            else:
                all_params.append(group)

        self._all_params = all_params

        # Assign parameters to shards based on rank
        self._param_to_shard = {}
        self._shard_to_params = {r: [] for r in range(self.world_size)}

        # Distribute parameters across ranks (round-robin)
        for idx, param in enumerate(all_params):
            assigned_rank = idx % self.world_size
            self._param_to_shard[id(param)] = assigned_rank
            self._shard_to_params[assigned_rank].append(param)

        # Build parameter groups for the local optimizer (only this rank's shard)
        # We need to preserve parameter groups from the original optimizer
        local_param_groups = []
        for group in param_list:
            if isinstance(group, dict):
                local_params = [p for p in group.get("params", [])
                               if self._param_to_shard.get(id(p)) == self.rank]
                if local_params:
                    local_group = {k: v for k, v in group.items() if k != "params"}
                    local_group["params"] = local_params
                    local_param_groups.append(local_group)
            else:
                if self._param_to_shard.get(id(group)) == self.rank:
                    local_param_groups.append(group)

        # Create local optimizer with only this rank's parameters
        if local_param_groups:
            self._local_optimizer = optimizer_cls(local_param_groups, **kwargs)
        else:
            # Create optimizer with empty param list to avoid errors
            dummy_param = torch.zeros(1, requires_grad=True)
            self._local_optimizer = optimizer_cls([dummy_param], **kwargs)
            self._local_optimizer.param_groups[0]["params"] = []

        # Initialize parent class with all parameters
        super().__init__(param_list, kwargs)

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        """Perform optimizer step and synchronize parameters across ranks."""
        # Let the local optimizer update its shard of parameters
        loss = self._local_optimizer.step(closure=closure, **kwargs)

        # Broadcast updated parameters from each rank to all other ranks
        for rank in range(self.world_size):
            for param in self._shard_to_params[rank]:
                dist.broadcast(param.data, src=rank)

        return loss

    def zero_grad(self, set_to_none=True):
        """Zero gradients for all parameters."""
        for param in self._all_params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def add_param_group(self, param_group: dict[str, Any]):
        """Add a parameter group, assigning parameters to the appropriate shard."""
        super().add_param_group(param_group)

        # Assign new parameters to shards
        new_params = param_group.get("params", [])
        for param in new_params:
            if id(param) not in self._param_to_shard:
                # Assign to the rank with fewest parameters
                min_rank = min(self._shard_to_params.keys(),
                              key=lambda r: len(self._shard_to_params[r]))
                self._param_to_shard[id(param)] = min_rank
                self._shard_to_params[min_rank].append(param)

                # Add to local optimizer if this rank owns it
                if min_rank == self.rank:
                    self._local_optimizer.add_param_group(param_group)
