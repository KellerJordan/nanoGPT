import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, _get_value, _dispatch_sqrt,
                                   _default_to_fused_or_foreach, _stack_if_compiling)
from typing import List, Optional, Tuple, Union
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

__all__ = ["StripedAdamW", "striped_adamw"]


class StripedAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = (
                    torch.tensor(0.0)
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values,
                # averaged across rows and columns for >= 2D tensors
                if len(p.shape) >= 2:
                    shape = p.shape[:-2]+torch.Size([p.shape[-2]+p.shape[-1]])
                else:
                    shape = p.shape
                state['exp_avg_sq'] = torch.zeros(shape).to(p)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            state_steps.append(state["step"])


    def step(self):
        """Performs a single optimization step."""

        self._cuda_graph_capture_health_check()

        loss = None

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )

            striped_adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

        return loss

def striped_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    *,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """

    _, foreach = _default_to_fused_or_foreach(params, False, use_fused=False)
    assert foreach, 'StripedAdam only supports multi-tensor optimization for now'

    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    func = _multi_tensor_striped_adamw

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
    )

def _multi_tensor_striped_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor):
        raise RuntimeError("lr as a Tensor is not supported")

    grouped_tensors = _group_tensors_by_device_and_dtype([
        params, grads, exp_avgs, exp_avg_sqs, state_steps])
    for (
        device_params,
        device_grads,
        device_exp_avgs,
        device_exp_avg_sqs,
        device_state_steps,
    ) in grouped_tensors.values():

        # update steps
        torch._foreach_add_(device_state_steps, 1)

        # Perform stepweight decay
        if weight_decay != 0:
            torch._foreach_mul_(device_params, 1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

        for p_grads, p_exp_avgs_sqs in zip(device_grads, device_exp_avg_sqs):
            p_grads_sq = p_grads.square()
            if len(p_grads.shape) >= 2:
                p_row = p_grads_sq.mean(-1)
                p_col = p_grads_sq.mean(-2)
                p_rowcol = torch.cat([p_row, p_col], dim=-1)
                p_exp_avgs_sqs.lerp_(p_rowcol, 1 - beta2)
            else:
                p_exp_avgs_sqs.lerp_(p_grads_sq, 1 - beta2)
        #torch._foreach_mul_(device_exp_avg_sqs, beta2)
        #torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
        bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]

        step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

        bias_correction2_sqrt = [_dispatch_sqrt(bc) for bc in bias_correction2]

        exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

        torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)

        approx_exp_avg_sq_sqrt = []
        shapes = [x.shape for x in device_exp_avgs]
        for shape, p_tensor in zip(shapes, exp_avg_sq_sqrt):
            if len(shape) >= 2:
                # Expand row/col averages into the full matrix of approximated sqrt avg second moments
                p_rowcol = p_tensor
                assert len(p_rowcol.shape) == (len(shape) - 1)
                assert p_rowcol.shape[-1] == (shape[-2]+shape[-1])
                p_row = p_rowcol[..., :shape[-2]][..., :, None]
                p_col = p_rowcol[..., shape[-2]:][..., None, :]
                p_mu = p_rowcol.mean(-1)[..., None,None]
                approx_p_exp_avg_sq_sqrt = ((p_row + eps).log() + (p_col + eps).log() - (p_mu + eps).log()).exp()
                assert approx_p_exp_avg_sq_sqrt.shape == shape
                approx_exp_avg_sq_sqrt.append(approx_p_exp_avg_sq_sqrt)
            else:
                p_exp_avg_sq_sqrt = p_tensor
                approx_exp_avg_sq_sqrt.append(p_exp_avg_sq_sqrt)

        torch._foreach_add_(approx_exp_avg_sq_sqrt, eps)
        torch._foreach_addcdiv_(device_params, device_exp_avgs, approx_exp_avg_sq_sqrt, step_size)

