import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _get_value,
                        _stack_if_compiling, _default_to_fused_or_foreach, params_t)
from typing import List, Optional, Tuple, Union

__all__ = ["StripedAdamW", "striped_adamw"]


class StripedAdamW(Optimizer):
    def __init__(
        self,
        params: params_t,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if isinstance(lr, Tensor) and foreach:
            raise ValueError("lr as a Tensor is not supported for foreach=True")
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
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
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
                # note(crcrpar): Deliberately host `step` on CPU if fused is off.
                # This is because kernel launches are costly on CUDA and XLA.
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

            if group['differentiable'] and state['step'].requires_grad:
                raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')

            # Foreach does not support a tensor lr
            if group['foreach'] and isinstance(group['lr'], Tensor):
                raise RuntimeError('lr as a Tensor is not supported for foreach=True')

            state_steps.append(state["step"])

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

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
                maximize=group["maximize"],
                foreach=group["foreach"],
                differentiable=group["differentiable"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
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
    foreach: Optional[bool] = None,
    differentiable: bool = False,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """

    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
        # Do not flip on foreach for the unsupported case where lr is a Tensor.
        if foreach and isinstance(lr, Tensor):
            foreach = False
    if foreach is None:
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_striped_adamw
    else:
        raise RuntimeError("single tensor StripedAdamW not currently supported")

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
        maximize=maximize,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )

def _multi_tensor_striped_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
    maximize: bool,
    differentiable: bool,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor):
        raise RuntimeError("lr as a Tensor is not supported for foreach=True")

    assert not differentiable, "_foreach ops don't support autograd"

    assert grad_scale is None and found_inf is None

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([
        params, grads, exp_avgs, exp_avg_sqs, state_steps])
    for ((
        device_params,
        device_grads,
        device_exp_avgs,
        device_exp_avg_sqs,
        device_state_steps,
    ), _) in grouped_tensors.values():
        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        device_grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_grads]
        device_exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avgs]
        device_exp_avg_sqs = [
            torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avg_sqs
        ]
        device_params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_params]

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

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
        bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]

        step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])
        
        correct_expavg_sqs = torch._foreach_div(device_exp_avg_sqs, bias_correction2)

        approx_sqrt_expavg_sqs = []
        shapes = [x.shape for x in device_exp_avgs]
        for shape, p_expavg_sq in zip(shapes, correct_expavg_sqs):
            if len(shape) >= 2:
                # For 2D+ parameters, approximate the full matrix of second moments using
                # averages across rows and columns.
                # p_rowcol = cat(rowavg(expavg(grads_sq)), colavg(expavg(grads_sq)))
                p_rowcol = p_expavg_sq
                assert len(p_rowcol.shape) == (len(shape) - 1)
                assert p_rowcol.shape[-1] == (shape[-2] + shape[-1])
                p_row = p_rowcol[..., :shape[-2]][..., :, None]
                p_col = p_rowcol[..., shape[-2]:][..., None, :]
                p_mu = p_rowcol.mean(-1)[..., None, None]
                p_approx_sqrt_expavg_sq = (0.5 * ((p_row + eps).log() + (p_col + eps).log() - (p_mu + eps).log())).exp()
                assert p_approx_sqrt_expavg_sq.shape == shape
                approx_sqrt_expavg_sqs.append(p_approx_sqrt_expavg_sq)
            else:
                # For 1D parameters use the exact sqrt of second moments (the standard AdamW update).
                approx_sqrt_expavg_sqs.append(p_expavg_sq.sqrt())

        torch._foreach_add_(approx_sqrt_expavg_sqs, eps)
        torch._foreach_addcdiv_(device_params, device_exp_avgs, approx_sqrt_expavg_sqs, step_size)

