"""
Memory-Efficient AdamW optimizer - Mac Compatible.
From https://github.com/policy-gradient/GRPO-Zero
Changes: Disabled pin_memory for non-CUDA devices.
"""
import math

import torch
from torch.optim import AdamW


class MemoryEfficientAdamW(AdamW):
    """
    Memory Efficient AdamW optimizer that keeps parameters and gradients on GPU
    but optimizer states on CPU when enabled.
    When disabled, behaves exactly like standard AdamW.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        pin_memory=True,
        enabled=True,
    ):
        super(MemoryEfficientAdamW, self).__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        # pin_memory only works with CUDA
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.enabled = enabled

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        if not self.enabled:
            return super(MemoryEfficientAdamW, self).step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    device = "cpu"
                    pin_memory = self.pin_memory
                    dtype = torch.float32

                    state["exp_avg"] = torch.zeros_like(
                        p.data, device=device, pin_memory=pin_memory, dtype=dtype
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, device=device, pin_memory=pin_memory, dtype=dtype
                    )
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p.data, device=device, pin_memory=pin_memory, dtype=dtype
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                state["step"] += 1
                state_steps.append(state["step"])

            self._memory_efficient_update(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

        return loss

    def _memory_efficient_update(
        self,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad,
        beta1,
        beta2,
        lr,
        weight_decay,
        eps,
    ):
        for i, param in enumerate(params):
            grad = grads[i]
            param_device = param.device

            exp_avg = exp_avgs[i].to(param_device, non_blocking=True)
            exp_avg_sq = exp_avg_sqs[i].to(param_device, non_blocking=True)

            step = state_steps[i]

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            if amsgrad:
                max_exp_avg_sq = max_exp_avg_sqs[i].to(param_device, non_blocking=True)
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = max_exp_avg_sq.sqrt().add_(eps)
                max_exp_avg_sqs[i].copy_(max_exp_avg_sq, non_blocking=True)
            else:
                denom = exp_avg_sq.sqrt().add_(eps)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

            if weight_decay != 0:
                param.mul_(1 - lr * weight_decay)

            param.addcdiv_(exp_avg, denom, value=-step_size)

            exp_avgs[i].copy_(exp_avg, non_blocking=True)
            exp_avg_sqs[i].copy_(exp_avg_sq, non_blocking=True)
