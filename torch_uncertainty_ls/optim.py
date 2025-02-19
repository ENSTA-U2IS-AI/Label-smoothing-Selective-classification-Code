"""Linted and typed from https://raw.githubusercontent.com/davda54/sam/main/sam.py.

Copyright belongs to the original author.
"""

from collections.abc import Callable
from functools import partial

import torch
from torch.optim import SGD, Optimizer


class SAM(Optimizer):
    def __init__(
        self,
        params,
        base_optimizer: type[Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        optimizer_kwargs: dict | None = None,
    ) -> None:
        """Sharpness-aware minimization optimizer."""
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **optimizer_kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **optimizer_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def prestep(self, zero_grad: bool = False) -> None:
        # First step
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Callable) -> None:
        # Actual sharpness aware step, which must be called step to update trainer.global step
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if closure is not None:
            closure()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        return torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class SAMSGD(SAM):
    def __init__(
        self,
        params,
        lr=0.001,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ) -> None:
        base_optimizer = partial(SGD, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, base_optimizer, rho, adaptive)
