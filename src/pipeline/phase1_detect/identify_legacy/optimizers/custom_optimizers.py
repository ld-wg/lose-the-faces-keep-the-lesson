import torch
import torch.nn as nn
import torch.optim as optim

# SAM
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho: {rho}")
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.rho = rho
        defaults = dict(rho=rho, **kwargs)
        super().__init__(self.base_optimizer.param_groups, defaults)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2) for group in self.param_groups for p in group["params"] if p.grad is not None
            ])
        )
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])
        if zero_grad:
            self.zero_grad()

    @property
    def base(self):
        return self.base_optimizer

    def zero_grad(self):
        self.base_optimizer.zero_grad()

class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, (b1, b2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg.mul_(b1).add_(grad, alpha=1 - b1)

                update = exp_avg.sign().add(p, alpha=wd)
                p.add_(update, alpha=-lr)
                exp_avg.mul_(b2).add_(grad, alpha=1 - b2)