"""Task-loss deletion scores for a shared, unbounded input gate."""

import torch
from torch import nn
from torch.func import functional_call, grad, vmap
import torch.nn.functional as F


def classification_loss(logits, target, reduction="mean"):
    if logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(
            logits.reshape(-1), target.reshape(-1).to(logits.dtype), reduction=reduction
        )
    return F.cross_entropy(logits, target.long(), reduction=reduction)


def gate_score_moments(model, inputs, targets, batch_size=32):
    """Return mean(t) and mean(t**2), t_nj = g_j * d loss_n / d g_j.

    Evaluation mode removes stochastic feature/dropout masks. Parameter gradients,
    per-module train/eval flags, and RNG state are untouched. Only task loss is
    differentiated; no ADMM term or optimizer weight decay enters the score.
    """
    if len(inputs) == 0 or len(inputs) != len(targets) or batch_size < 1:
        raise ValueError("Scoring requires nonempty aligned inputs and positive batch_size")
    if getattr(model, "bounded_gate", False):
        raise ValueError("Deletion score requires a direct, unbounded gate")
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) and not m.track_running_stats
           for m in model.modules()):
        raise ValueError("BatchNorm without running statistics couples scoring samples")
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    gate = model.gate.detach()
    first = torch.zeros_like(gate)
    second = torch.zeros_like(gate)

    def sample_loss(g, x, y):
        prediction = functional_call(model, {"gate": g}, (x.unsqueeze(0),))
        return classification_loss(prediction, y.unsqueeze(0))

    try:
        for start in range(0, len(inputs), batch_size):
            derivatives = vmap(grad(sample_loss), in_dims=(None, 0, 0))(
                gate, inputs[start:start + batch_size], targets[start:start + batch_size]
            )
            contribution = (derivatives * gate).detach()
            first += contribution.sum(0)
            second += contribution.square().sum(0)
    finally:
        for module, training in flags:
            module.training = training
    return first / len(inputs), second / len(inputs)


def score_from_moments(first, second, mode):
    """Aggregate signed terms before abs: abs(mean(-t + 0.5*t**2))."""
    if mode == "taylor_first":
        return -first
    if mode == "taylor_second":
        return 0.5 * second
    if mode == "taylor_fisher":
        return -first + 0.5 * second
    raise ValueError(f"Unknown score mode: {mode}")


def ranking_score(parameters, mode, first=None, second=None):
    """Evaluate the threshold score at the final model for feature ranking."""
    if mode == "column_norm":
        return parameters["layers.1.weight"].detach().norm(dim=0) + 1e-8
    base_mode = mode.removeprefix("column_norm_")
    if base_mode in ("gate", "gate_only", "gate_adaptive"):
        score = parameters["gate"].detach().abs()
    else:
        score = score_from_moments(first, second, base_mode).detach().abs()
    if mode.startswith("column_norm_"):
        score = score * parameters["layers.1.weight"].detach().norm(dim=0)
    return score


def importance_scaling(signed_score):
    """Detached mean-one magnitude; an all-zero score gives uniform scaling."""
    magnitude = signed_score.detach().abs()
    if not torch.isfinite(magnitude).all():
        raise FloatingPointError("Non-finite task importance")
    if magnitude.mean() == 0:
        return torch.ones_like(magnitude)
    return (magnitude / magnitude.mean()).clamp_min(1e-8)
