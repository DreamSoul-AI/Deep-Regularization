"""Bounded port of the historical shared-gate, two-variable Ratio ADMM path.

Source: feature-selection@555ba11e8872bd6618cd721bd3b08f7128db9e6b,
src/admm_input_group_wrapper.py. Only the unbounded gate path is supported.
The retained shrink-plus-cubic update is a historical algorithm, not a claim of
an exact weighted L1/L2 proximal solution. New scores only change its scaling.
"""

from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .feature_scores import (
    classification_loss, gate_score_moments, importance_scaling, score_from_moments,
)


@dataclass(frozen=True)
class RankingConfig:
    epochs: int = 120
    warmup_epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 0.005
    weight_decay: float = 0.001
    penalty: float = 0.05
    rho: float = 50.0
    score_interval: int = 10
    score_samples: int = 128
    score_batch_size: int = 32
    hidden_size: int = 32
    dropout: float = 0.04308691548552568
    feature_dropout: float = 0.6
    protocol: str = "pilot"


class GatedMLP(nn.Module):
    """Historical two-hidden-layer Mish backbone, shared signed input gate."""

    def __init__(self, features, classes=2, config=None):
        super().__init__()
        config = config or RankingConfig()
        self.gate = nn.Parameter(torch.ones(features))
        self.feature_dropout = config.feature_dropout
        self.layers = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(features, config.hidden_size), nn.Mish(),
            nn.Dropout(config.dropout), nn.Linear(config.hidden_size, config.hidden_size), nn.Mish(),
            nn.Linear(config.hidden_size, 1 if classes <= 2 else classes),
        )

    def forward(self, inputs):
        gate = self.gate
        if self.training and self.feature_dropout:
            mask = (torch.rand_like(gate) > self.feature_dropout).float()
            gate = gate * mask / (1.0 - self.feature_dropout + 1e-8)
        return self.layers(inputs * gate)


def cubic_scale(coefficient):
    """Largest positive root of tau**3 - tau - D = 0 for nonnegative D."""
    d = coefficient.double()
    discriminant = (d / 2).square() - 1 / 27
    # Rationalizing the second Cardano term avoids cancellation at large D.
    a = (d / 2 + discriminant.clamp_min(0).sqrt()).pow(1 / 3)
    cardano = a + 1 / (3 * a.clamp_min(torch.finfo(a.dtype).tiny))
    trigonometric = 2 / (3 ** 0.5) * torch.cos(
        torch.acos((3 * (3 ** 0.5) * d / 2).clamp(-1, 1)) / 3
    )
    return torch.where(discriminant >= 0, cardano, trigonometric).to(coefficient.dtype)


def legacy_cubic_scale(coefficient):
    """Original float32 Cardano arithmetic, retained only for source reproduction."""
    d = coefficient.reshape(1)
    discriminant = (d / 2.0) ** 2 - 1.0 / 27.0
    if discriminant.item() >= 0:
        root = discriminant.sqrt()
        a, b = d / 2.0 + root, d / 2.0 - root
        return a.sign() * a.abs() ** (1 / 3) + b.sign() * b.abs() ** (1 / 3)
    sqrt3 = torch.sqrt(torch.tensor(3.0, device=d.device))
    return 2.0 * torch.cos(torch.acos((3.0 * sqrt3 * d / 2.0).clamp(-1, 1)) / 3.0) / sqrt3


def fit_ranking(inputs, targets, *, mode, seed, config=None):
    """Fit only on supplied training rows; never consume support truth or test labels.

    No validation-based checkpoint selection, hard pruning, or downstream refit.
    The calibration subset is fixed by evenly spaced training-row indices.
    """
    config = config or RankingConfig()
    if config.protocol not in ("pilot", "legacy"):
        raise ValueError("Unknown training protocol")
    if mode not in ("column_norm", "taylor_first", "taylor_fisher"):
        raise ValueError(f"Unknown score mode: {mode}")
    if not (0 <= config.warmup_epochs < config.epochs) or config.score_interval < 1:
        raise ValueError("Require 0 <= warmup < epochs and positive score interval")
    if config.score_samples < 1 or config.batch_size < 1 or config.rho <= 0:
        raise ValueError("Invalid calibration, batch size, or rho")
    if inputs.ndim != 2 or targets.ndim != 1 or len(inputs) != len(targets):
        raise ValueError("Expected aligned [samples, features] inputs and [samples] labels")
    if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
        raise ValueError("Non-finite training data")
    labels = torch.unique(targets).long()
    if not torch.equal(labels, torch.arange(len(labels), device=labels.device)):
        raise ValueError("Classification labels must be contiguous starting at zero")
    torch.manual_seed(seed)
    if config.protocol == "legacy":
        if inputs.device.type != "cpu":
            raise ValueError("The legacy reproduction protocol is CPU-only")
        # The historical scaler runs NumPy float64 arithmetic before float32 loading.
        array = inputs.detach().numpy()
        mean_array, std_array = array.mean(0), array.std(0) + 1e-8
        mean, std = torch.from_numpy(mean_array), torch.from_numpy(std_array)
        x = torch.as_tensor((array - mean_array) / std_array, dtype=torch.float32)
    else:
        mean = inputs.mean(0)
        std = inputs.std(0, correction=0) + 1e-8
        x = (inputs - mean) / std
    model = GatedMLP(x.shape[1], len(labels), config).to(x.device)
    calibration_ids = torch.linspace(
        0, len(x) - 1, min(config.score_samples, len(x)), device=x.device
    ).long()
    calibration_x, calibration_y = x[calibration_ids], targets[calibration_ids]
    generator = torch.Generator().manual_seed(seed + 1)
    legacy_loader = DataLoader(TensorDataset(torch.arange(len(x))), batch_size=config.batch_size,
                               shuffle=True, num_workers=0) if config.protocol == "legacy" else None
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    z = model.gate.detach().clone()
    dual = torch.zeros_like(z)
    rho = config.rho
    trace = []
    scale = torch.ones_like(z)
    for epoch in range(config.epochs):
        admm_epoch = epoch - config.warmup_epochs
        if admm_epoch == 0:
            # Preserve the historical warmup/ADMM optimizer reset.
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                         weight_decay=config.weight_decay)
            z = model.gate.detach().clone()
        if admm_epoch >= 0:
            if mode == "column_norm":
                scale = model.layers[1].weight.detach().norm(dim=0) + 1e-8
            elif admm_epoch % config.score_interval == 0:
                first, second = gate_score_moments(model, calibration_x, calibration_y,
                                                   config.score_batch_size)
                scale = importance_scaling(score_from_moments(first, second, mode))
        model.train()
        if legacy_loader is not None:
            batches = (batch[0] for batch in legacy_loader)
        else:
            batches = torch.randperm(len(x), generator=generator).to(x.device).split(config.batch_size)
        task_sum = 0.0
        for ids in batches:
            optimizer.zero_grad(set_to_none=True)
            task_loss = classification_loss(model(x[ids]), targets[ids])
            total = task_loss
            if admm_epoch >= 0:
                total = total + 0.5 * rho * (model.gate - z + dual).square().sum()
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite objective at epoch {epoch}")
            total.backward()
            optimizer.step()
            task_sum += task_loss.detach().item() * len(ids)
        primal = dual_residual = 0.0
        if admm_epoch >= 0:
            with torch.no_grad():
                old_z = z.clone()
                v = model.gate + dual
                thresholds = (config.penalty / scale).clamp(1e-8, 0.5) / rho
                z = v.sign() * (v.abs() - thresholds).clamp_min(0)
                l2 = z.norm() + 1e-8
                l1 = torch.norm(z, p=1)
                if l1 > 1e-8:
                    coefficient = (config.penalty / rho) * l1 / l2.pow(3).clamp_min(1e-10)
                    solver = legacy_cubic_scale if config.protocol == "legacy" else cubic_scale
                    z = solver(coefficient) * z
                if config.protocol == "legacy":
                    # Match the original left-associated float32 arithmetic exactly.
                    dual = dual + model.gate - z
                else:
                    dual += model.gate - z
                primal = (model.gate - z).norm().item()
                dual_residual = rho * (z - old_z).norm().item()
                if admm_epoch > 0 and admm_epoch % 5 == 0:
                    old_rho = rho
                    if max(primal, 1e-12) > 10 * max(dual_residual, 1e-12):
                        rho = min(rho * 2, 1e4)
                    elif max(dual_residual, 1e-12) > 10 * max(primal, 1e-12):
                        floor = 5.0 if x.shape[1] < 64 else (20.0 if x.shape[1] < 256 else 50.0)
                        rho = max(rho / 2, floor)
                    dual *= old_rho / rho
        trace.append(dict(epoch=epoch, task_loss=task_sum / len(x), primal=primal,
                          dual=dual_residual, rho=rho))
    model.eval()
    first, second = gate_score_moments(model, calibration_x, calibration_y,
                                      config.score_batch_size)
    scores = {"gate": model.gate.detach().abs(),
              "taylor_first": score_from_moments(first, second, "taylor_first").abs(),
              "taylor_fisher": score_from_moments(first, second, "taylor_fisher").abs()}
    state = dict(model=model.state_dict(), z=z.detach(), dual=dual.detach(), rho=rho,
                 optimizer=optimizer.state_dict(), mean=mean, std=std,
                 calibration_ids=calibration_ids, first_moment=first, second_moment=second,
                 config=asdict(config), mode=mode, seed=seed, trace=trace)
    return scores, state
