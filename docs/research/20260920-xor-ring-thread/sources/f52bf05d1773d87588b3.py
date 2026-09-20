# -*- coding: utf-8 -*-
"""
Clean ADMM / Lasso feature-selection wrapper for admm_input_group only.

This is a minimal version that only supports the admm_input_group method
with its internal implementation, without dependencies on old optimizers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import from local src directory
from src.nn_wrapper import GaussianNoise
from src.utils import TestSet, TrainingSet

# Import ADMM utilities from sibling module
from .admm_utils import (
    canonical_signed_ratio_prox,
    deterministic_topk,
    safe_cbrt,
    safe_norm,
    soft_thresholding,
    solve_cubic_denominator_copy,
    solve_cubic_paper,
    solve_cubic_ratio_norm,
    strict_ratio_weighted_y_update,
    strict_ratio_weighted_z_update,
    strict_ratio_y_update,
    strict_ratio_z_update,
)

FIRST_LAYER_SPECTRAL_SCALE_MODES = {"none", "fan_ratio", "exact_spectral"}
SCORE_PERMUTATION_SEED = 20260714
STRICT_STATIONARITY_RELATIVE_TOL = 1e-2


def _column_normalized_spectral_scale(
    normalized_weight: torch.Tensor,
    mode: str,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """Return a scalar multiplier for a column-normalized first layer."""
    if mode == "none":
        return normalized_weight.new_tensor(1.0)
    if mode not in FIRST_LAYER_SPECTRAL_SCALE_MODES:
        raise ValueError(
            f"Unknown first_layer_spectral_scale={mode!r}; "
            f"expected one of {sorted(FIRST_LAYER_SPECTRAL_SCALE_MODES)}"
        )

    fan_in = max(1.0, float(in_features))
    fan_out = max(1.0, float(out_features))
    target = normalized_weight.new_tensor(np.sqrt(fan_out / fan_in))
    if mode == "fan_ratio":
        return normalized_weight.new_tensor(fan_out / fan_in)

    spectral_norm = torch.linalg.matrix_norm(
        normalized_weight.float(), ord=2
    ).to(dtype=normalized_weight.dtype, device=normalized_weight.device)
    return target / spectral_norm.detach().clamp_min(1e-8)

# Import adaptive architectures (optional, for comparison experiments)
try:
    from .adaptive_architecture import (
        AdaptiveFeatureSelectionMLP,
        AdaptiveFeatureSelector,
        DimensionAdaptiveGate,
        create_adaptive_model,
    )

    ADAPTIVE_MODELS_AVAILABLE = True
except ImportError:
    ADAPTIVE_MODELS_AVAILABLE = False

# ---------------------------------------------------------------------------
# MLP model (mirrors the benchmark's ``Model`` but simplified for FS)
# ---------------------------------------------------------------------------

DEFAULT_SADMM_LATENT_SIZE = 32
DEFAULT_SADMM_HIDDEN_LAYERS = 2
DEFAULT_SADMM_DROPOUT = 0.04308691548552568
DEFAULT_SADMM_ACTIVATION = "mish"


class ColumnNormalizedLinear(nn.Module):
    """Linear layer that uses unit-norm input columns during forward passes."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        first_layer_spectral_scale: str = "none",
    ):
        super().__init__()
        if first_layer_spectral_scale not in FIRST_LAYER_SPECTRAL_SCALE_MODES:
            raise ValueError(
                f"Unknown first_layer_spectral_scale={first_layer_spectral_scale!r}; "
                f"expected one of {sorted(FIRST_LAYER_SPECTRAL_SCALE_MODES)}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.first_layer_spectral_scale = first_layer_spectral_scale
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.raw_weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.raw_weight)
            bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> torch.Tensor:
        normalized = F.normalize(self.raw_weight, p=2, dim=0, eps=1e-8)
        scale = _column_normalized_spectral_scale(
            normalized,
            self.first_layer_spectral_scale,
            self.in_features,
            self.out_features,
        )
        return scale * normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class FeatureSelectionMLP(nn.Module):
    """Simple MLP for tabular classification, compatible with sparsity optimizers."""

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        latent_size: int = DEFAULT_SADMM_LATENT_SIZE,
        n_hidden_layers: int = DEFAULT_SADMM_HIDDEN_LAYERS,
        gaussian_noise: float = 0.0,
        dropout: float = DEFAULT_SADMM_DROPOUT,
        activation: str = DEFAULT_SADMM_ACTIVATION,
    ):
        super().__init__()
        n_out = 1 if n_classes <= 2 else n_classes

        layers: list[nn.Module] = []
        if gaussian_noise > 0:
            layers.append(GaussianNoise(gaussian_noise))

        for k in range(n_hidden_layers):
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = input_size if k == 0 else latent_size
            layers.append(nn.Linear(in_dim, latent_size))

            act: nn.Module
            if activation == "relu":
                act = nn.ReLU()
            elif activation == "leakyrelu":
                act = nn.LeakyReLU(0.2)
            elif activation == "mish":
                act = nn.Mish()
            elif activation == "selu":
                act = nn.SELU()
            else:
                act = nn.Mish()
            layers.append(act)

        layers.append(nn.Linear(latent_size, n_out))
        self.layers = nn.Sequential(*layers)
        self.apply(init_weights)

    # Convenience: find the first Linear layer (for feature importance)
    @property
    def first_linear(self) -> nn.Module:
        for m in self.layers:
            if isinstance(m, (nn.Linear, ColumnNormalizedLinear)):
                return m
        raise RuntimeError("No Linear layer found")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GatedFeatureSelectionMLP(nn.Module):
    """MLP with learnable per-feature gates and input feature dropout.

    During training, each input feature is randomly masked with probability
    ``feat_drop``.  This prevents the model from memorising via irrelevant
    features (critical when n_features >> n_samples).  The learnable gate
    ``theta`` is pruned via Linearized ADMM + Ratio Norm to identify
    informative features.

    The paper's main benchmark uses one fixed compact predictor
    (2 hidden layers, width 32) without per-dataset retuning.
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        latent_size: int = DEFAULT_SADMM_LATENT_SIZE,
        n_hidden_layers: int = DEFAULT_SADMM_HIDDEN_LAYERS,
        gaussian_noise: float = 0.0,  # Disabled by default; set >0 to enable
        dropout: float = DEFAULT_SADMM_DROPOUT,
        feat_drop: float = 0.6,  # Tuned value (was 0.7); see tune_feat_drop.py ablation
        activation: str = DEFAULT_SADMM_ACTIVATION,
        bounded_gate: bool = False,
        layer_norm: int = 0,
        column_normalize_first_layer: bool = False,
        first_layer_spectral_scale: str = "none",
    ):
        super().__init__()
        if first_layer_spectral_scale not in FIRST_LAYER_SPECTRAL_SCALE_MODES:
            raise ValueError(
                f"Unknown first_layer_spectral_scale={first_layer_spectral_scale!r}; "
                f"expected one of {sorted(FIRST_LAYER_SPECTRAL_SCALE_MODES)}"
            )
        self.gate = nn.Parameter(torch.ones(input_size))
        self.feat_drop = feat_drop
        self.bounded_gate = bounded_gate
        self.column_normalize_first_layer = column_normalize_first_layer
        self.first_layer_spectral_scale = first_layer_spectral_scale
        n_out = 1 if n_classes <= 2 else n_classes

        layers: list[nn.Module] = []
        if gaussian_noise > 0:
            layers.append(GaussianNoise(gaussian_noise))

        inplace = False
        for k in range(n_hidden_layers):
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout, inplace=inplace))

            in_dim = input_size if k == 0 else latent_size
            if k == 0 and column_normalize_first_layer:
                layers.append(
                    ColumnNormalizedLinear(
                        in_dim,
                        latent_size,
                        first_layer_spectral_scale=first_layer_spectral_scale,
                    )
                )
            else:
                layers.append(nn.Linear(in_dim, latent_size))

            if layer_norm:
                layers.append(nn.LayerNorm(latent_size))

            act: nn.Module
            if activation == "relu":
                act = nn.ReLU(inplace=inplace)
            elif activation == "leakyrelu":
                act = nn.LeakyReLU(0.2, inplace=inplace)
            elif activation == "prelu":
                act = nn.PReLU(latent_size)
            elif activation == "tanh":
                act = nn.Tanh()
            elif activation == "sigmoid":
                act = nn.Sigmoid()
            elif activation == "mish":
                act = nn.Mish(inplace=inplace)
            elif activation == "selu":
                act = nn.SELU(inplace=inplace)
            else:
                act = nn.Hardswish(inplace=inplace)
            layers.append(act)

        layers.append(nn.Linear(latent_size, n_out))
        self.layers = nn.Sequential(*layers)
        # Use PyTorch default init (kaiming_uniform_ a=sqrt(5)); empirically
        # better for gate-based feature selection than the custom init_weights.

    @property
    def first_linear(self) -> nn.Module:
        for m in self.layers:
            if isinstance(m, (nn.Linear, ColumnNormalizedLinear)):
                return m
        raise RuntimeError("No Linear layer found")

    def gate_from_parameter(self, gate_param: torch.Tensor) -> torch.Tensor:
        if self.bounded_gate:
            return torch.sigmoid(gate_param)
        return gate_param

    def parameter_from_gate(self, gate_value: torch.Tensor) -> torch.Tensor:
        if self.bounded_gate:
            gate_value = torch.clamp(gate_value, min=1e-6, max=1.0 - 1e-6)
            return torch.logit(gate_value)
        return gate_value

    def get_gate_values(self, training_drop: bool = False) -> torch.Tensor:
        g = self.gate_from_parameter(self.gate)
        if self.training and training_drop and self.feat_drop > 0:
            mask = (torch.rand(g.shape, device=g.device) > self.feat_drop).float()
            g = g * mask / (1.0 - self.feat_drop + 1e-8)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x * self.get_gate_values(training_drop=True))


class ParentGatedFeatureSelectionMLP(nn.Module):
    """MLP with one gate per original parent feature.

    The input is an expanded polynomial design matrix. ADMM operates on the
    original parent gates, while each expanded term receives a deterministic
    multiplier from the parents that generated it:
      x_j      -> g_j
      x_j^2    -> g_j
      x_j x_k  -> g_j * g_k
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        parent_powers: np.ndarray,
        latent_size: int = DEFAULT_SADMM_LATENT_SIZE,
        n_hidden_layers: int = DEFAULT_SADMM_HIDDEN_LAYERS,
        gaussian_noise: float = 0.0,
        dropout: float = DEFAULT_SADMM_DROPOUT,
        feat_drop: float = 0.6,
        activation: str = DEFAULT_SADMM_ACTIVATION,
        bounded_gate: bool = False,
        layer_norm: int = 0,
        column_normalize_first_layer: bool = False,
        first_layer_spectral_scale: str = "none",
    ):
        super().__init__()
        if first_layer_spectral_scale not in FIRST_LAYER_SPECTRAL_SCALE_MODES:
            raise ValueError(
                f"Unknown first_layer_spectral_scale={first_layer_spectral_scale!r}; "
                f"expected one of {sorted(FIRST_LAYER_SPECTRAL_SCALE_MODES)}"
            )
        powers_np = np.asarray(parent_powers)
        if powers_np.ndim != 2 or powers_np.shape[0] != input_size:
            raise ValueError(
                "parent_powers must have shape "
                f"(input_size, n_parent_features); got {powers_np.shape}, input_size={input_size}"
            )

        self.feat_drop = feat_drop
        self.bounded_gate = bounded_gate
        self.column_normalize_first_layer = column_normalize_first_layer
        self.first_layer_spectral_scale = first_layer_spectral_scale
        self.input_size = input_size
        self.parent_input_size = int(powers_np.shape[1])
        self.gate = nn.Parameter(torch.ones(self.parent_input_size))
        n_out = 1 if n_classes <= 2 else n_classes

        parent_mask_np = powers_np > 0
        parent_counts = parent_mask_np.sum(axis=1)
        max_parents = max(1, int(parent_counts.max()) if parent_counts.size else 1)
        parent_index = np.zeros((input_size, max_parents), dtype=np.int64)
        parent_mask = np.zeros((input_size, max_parents), dtype=np.bool_)
        for term_idx, parents in enumerate(np.flatnonzero(row) for row in parent_mask_np):
            if len(parents) == 0:
                continue
            parent_index[term_idx, : len(parents)] = parents
            parent_mask[term_idx, : len(parents)] = True
        self.register_buffer(
            "term_parent_index",
            torch.as_tensor(parent_index, dtype=torch.long),
        )
        self.register_buffer(
            "term_parent_mask",
            torch.as_tensor(parent_mask, dtype=torch.bool),
        )
        parent_count = torch.as_tensor(
            parent_mask_np.sum(axis=0), dtype=torch.float32
        ).clamp_min(1.0)
        self.register_buffer("parent_term_count", parent_count)

        layers: list[nn.Module] = []
        if gaussian_noise > 0:
            layers.append(GaussianNoise(gaussian_noise))

        inplace = False
        for k in range(n_hidden_layers):
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout, inplace=inplace))

            in_dim = input_size if k == 0 else latent_size
            if k == 0 and column_normalize_first_layer:
                layers.append(
                    ColumnNormalizedLinear(
                        in_dim,
                        latent_size,
                        first_layer_spectral_scale=first_layer_spectral_scale,
                    )
                )
            else:
                layers.append(nn.Linear(in_dim, latent_size))

            if layer_norm:
                layers.append(nn.LayerNorm(latent_size))

            if activation == "relu":
                act: nn.Module = nn.ReLU(inplace=inplace)
            elif activation == "leakyrelu":
                act = nn.LeakyReLU(0.2, inplace=inplace)
            elif activation == "prelu":
                act = nn.PReLU(latent_size)
            elif activation == "tanh":
                act = nn.Tanh()
            elif activation == "sigmoid":
                act = nn.Sigmoid()
            elif activation == "mish":
                act = nn.Mish(inplace=inplace)
            elif activation == "selu":
                act = nn.SELU(inplace=inplace)
            else:
                act = nn.Hardswish(inplace=inplace)
            layers.append(act)

        layers.append(nn.Linear(latent_size, n_out))
        self.layers = nn.Sequential(*layers)

    @property
    def first_linear(self) -> nn.Module:
        for m in self.layers:
            if isinstance(m, (nn.Linear, ColumnNormalizedLinear)):
                return m
        raise RuntimeError("No Linear layer found")

    def get_feature_scores(self) -> torch.Tensor:
        term_norm = torch.norm(self.first_linear.weight, p=2, dim=0).clamp_min(1e-8)
        parent_sum = torch.zeros(
            self.parent_input_size,
            device=term_norm.device,
            dtype=term_norm.dtype,
        )
        parent_sum.scatter_add_(
            0,
            self.term_parent_index[self.term_parent_mask],
            term_norm.repeat_interleave(self.term_parent_mask.sum(dim=1)),
        )
        return (parent_sum / self.parent_term_count.to(term_norm.device)).clamp_min(1e-8)

    def gate_from_parameter(self, gate_param: torch.Tensor) -> torch.Tensor:
        if self.bounded_gate:
            return torch.sigmoid(gate_param)
        return gate_param

    def parameter_from_gate(self, gate_value: torch.Tensor) -> torch.Tensor:
        if self.bounded_gate:
            gate_value = torch.clamp(gate_value, min=1e-6, max=1.0 - 1e-6)
            return torch.logit(gate_value)
        return gate_value

    def get_gate_values(self, training_drop: bool = False) -> torch.Tensor:
        g = self.gate_from_parameter(self.gate)
        if self.training and training_drop and self.feat_drop > 0:
            mask = (torch.rand(g.shape, device=g.device) > self.feat_drop).float()
            g = g * mask / (1.0 - self.feat_drop + 1e-8)
        return g

    def get_expanded_gate_values(self, training_drop: bool = False) -> torch.Tensor:
        parent_gate = self.get_gate_values(training_drop=training_drop)
        gathered = parent_gate[self.term_parent_index]
        factors = torch.where(
            self.term_parent_mask,
            gathered,
            torch.ones_like(gathered),
        )
        return torch.prod(factors, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x * self.get_expanded_gate_values(training_drop=True))


class GatedWeightNormMLP(nn.Module):
    """MLP whose feature gate is the weight-normalization scale.

    Key design:
    - each first-layer input-feature column is normalized over hidden units
    - implementation path: x * gate -> column-normalized first layer -> rest
    - strict WN does not multiply raw feature norms back into the forward pass
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        latent_size: int = DEFAULT_SADMM_LATENT_SIZE,
        n_hidden_layers: int = DEFAULT_SADMM_HIDDEN_LAYERS,
        gaussian_noise: float = 0.0,
        dropout: float = DEFAULT_SADMM_DROPOUT,
        feat_drop: float = 0.6,
        activation: str = DEFAULT_SADMM_ACTIVATION,
        bounded_gate: bool = False,
        layer_norm: int = 0,
        legacy_weight_norm: bool = False,
        first_layer_spectral_scale: str = "none",
        feature_score_mode: str = "auto",
    ):
        super().__init__()
        if first_layer_spectral_scale not in FIRST_LAYER_SPECTRAL_SCALE_MODES:
            raise ValueError(
                f"Unknown first_layer_spectral_scale={first_layer_spectral_scale!r}; "
                f"expected one of {sorted(FIRST_LAYER_SPECTRAL_SCALE_MODES)}"
            )
        if feature_score_mode not in {"auto", "ones", "raw_column_norm"}:
            raise ValueError(
                f"Unknown feature_score_mode={feature_score_mode!r}; "
                "expected one of ['auto', 'ones', 'raw_column_norm']"
            )
        self.gate = nn.Parameter(torch.ones(input_size))
        self.feat_drop = feat_drop
        self.bounded_gate = bounded_gate
        self.legacy_weight_norm = legacy_weight_norm
        self.first_layer_spectral_scale = first_layer_spectral_scale
        self.feature_score_mode = feature_score_mode
        self.input_size = input_size
        self.latent_size = latent_size
        n_out = 1 if n_classes <= 2 else n_classes

        # First layer: raw weight (will be normalized in forward)
        self.first_raw_weight = nn.Parameter(torch.empty(latent_size, input_size))
        self.first_bias = nn.Parameter(torch.empty(latent_size))
        nn.init.kaiming_uniform_(self.first_raw_weight, a=np.sqrt(5))
        fan_in = input_size
        bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.first_bias, -bound, bound)

        # Build remaining layers (after first layer)
        layers: list[nn.Module] = []
        if gaussian_noise > 0:
            layers.append(GaussianNoise(gaussian_noise))

        inplace = False
        for k in range(n_hidden_layers - 1):  # -1 because first layer is separate
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout, inplace=inplace))

            layers.append(nn.Linear(latent_size, latent_size))

            if layer_norm:
                layers.append(nn.LayerNorm(latent_size))

            act: nn.Module
            if activation == "relu":
                act = nn.ReLU(inplace=inplace)
            elif activation == "leakyrelu":
                act = nn.LeakyReLU(0.2, inplace=inplace)
            elif activation == "prelu":
                act = nn.PReLU(latent_size)
            elif activation == "tanh":
                act = nn.Tanh()
            elif activation == "sigmoid":
                act = nn.Sigmoid()
            elif activation == "mish":
                act = nn.Mish(inplace=inplace)
            elif activation == "selu":
                act = nn.SELU(inplace=inplace)
            else:
                act = nn.Hardswish(inplace=inplace)
            layers.append(act)

        layers.append(nn.Linear(latent_size, n_out))
        self.layers = nn.Sequential(*layers)

    @property
    def first_weight_norm(self) -> torch.Tensor:
        """Legacy clamp-safe denominator for normalizing raw columns."""
        return self.first_raw_weight_norm.clamp_min(1e-8)

    @property
    def first_raw_weight_norm(self) -> torch.Tensor:
        """Exact per-feature raw-weight column norm, without a clamp."""
        return torch.linalg.vector_norm(self.first_raw_weight, ord=2, dim=0)

    @property
    def first_normalized_weight(self) -> torch.Tensor:
        """First layer weight with unit-norm columns, one column per input feature."""
        return F.normalize(self.first_raw_weight, p=2, dim=0, eps=1e-8)

    @property
    def first_spectral_scale_multiplier(self) -> torch.Tensor:
        return _column_normalized_spectral_scale(
            self.first_normalized_weight,
            self.first_layer_spectral_scale,
            self.input_size,
            self.latent_size,
        )

    @property
    def first_effective_weight(self) -> torch.Tensor:
        """Column-normalized first layer after the optional global scale."""
        return self.first_spectral_scale_multiplier * self.first_normalized_weight

    @property
    def weight(self) -> torch.Tensor:
        """Compatibility alias for code that expects first_linear.weight."""
        return self.first_effective_weight

    @property
    def bias(self) -> torch.Tensor:
        """Compatibility alias for code that expects first_linear.bias."""
        return self.first_bias

    @property
    def first_linear(self) -> nn.Module:
        """Compatibility property - returns a proxy for first layer."""
        return self

    def get_feature_scores(self) -> torch.Tensor:
        """Return feature importance scores for penalty computation.

        gate_j is the feature scale. The first-layer input-feature columns are
        normalized during forward, so strict WN's default ADMM feature penalty
        must not be scaled by raw weight norms.
        This is called by _get_feature_penalty_scores() for ADMM penalty.
        """
        if self.feature_score_mode == "raw_column_norm":
            return self.first_weight_norm
        if self.feature_score_mode == "ones":
            return torch.ones_like(self.gate)
        if self.legacy_weight_norm:
            return self.first_weight_norm
        return torch.ones_like(self.gate)

    def first_layer_spectral_diagnostics(self) -> dict[str, float | str]:
        """Return detached first-layer operator-scale diagnostics."""
        with torch.no_grad():
            normalized = self.first_normalized_weight.detach()
            raw_spectral = torch.linalg.matrix_norm(normalized.float(), ord=2).item()
            scale = self.first_spectral_scale_multiplier.detach().float().item()
            target = float(
                np.sqrt(
                    max(1.0, float(self.latent_size))
                    / max(1.0, float(self.input_size))
                )
            )
        return {
            "mode": self.first_layer_spectral_scale,
            "target": target,
            "normalized_spectral_norm": float(raw_spectral),
            "multiplier": float(scale),
            "effective_spectral_norm": float(scale * raw_spectral),
        }

    def gate_from_parameter(self, gate_param: torch.Tensor) -> torch.Tensor:
        if self.bounded_gate:
            return torch.sigmoid(gate_param)
        return gate_param

    def parameter_from_gate(self, gate_value: torch.Tensor) -> torch.Tensor:
        if self.bounded_gate:
            gate_value = torch.clamp(gate_value, min=1e-6, max=1.0 - 1e-6)
            return torch.logit(gate_value)
        return gate_value

    def get_gate_values(self, training_drop: bool = False) -> torch.Tensor:
        g = self.gate_from_parameter(self.gate)
        if self.training and training_drop and self.feat_drop > 0:
            mask = (torch.rand(g.shape, device=g.device) > self.feat_drop).float()
            g = g * mask / (1.0 - self.feat_drop + 1e-8)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.get_gate_values(training_drop=True)
        if self.legacy_weight_norm:
            gate = gate * self.first_weight_norm

        x_scaled = x * gate
        h = F.linear(x_scaled, self.first_effective_weight, self.first_bias)

        h = F.mish(h)
        return self.layers(h)


class GatedUnitGateMLP(GatedWeightNormMLP):
    """Theory-aligned direct effective-gate reference with ``a == 1``.

    The raw and effective gates are the same tensor.  Raw first-layer columns
    only parameterize unit directions and must remain nonzero; their norms are
    never used as pruning scores.
    """

    admm_constraint_mode = "gate"
    training_variant = "unit_gate"

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        latent_size: int = DEFAULT_SADMM_LATENT_SIZE,
        n_hidden_layers: int = DEFAULT_SADMM_HIDDEN_LAYERS,
        gaussian_noise: float = 0.0,
        dropout: float = DEFAULT_SADMM_DROPOUT,
        feat_drop: float = 0.6,
        activation: str = DEFAULT_SADMM_ACTIVATION,
        bounded_gate: bool = False,
        layer_norm: int = 0,
        first_layer_spectral_scale: str = "none",
    ):
        if bounded_gate:
            raise ValueError("GatedUnitGateMLP requires bounded_gate=False")
        if first_layer_spectral_scale != "none":
            raise ValueError(
                "GatedUnitGateMLP requires first_layer_spectral_scale='none'"
            )
        super().__init__(
            input_size=input_size,
            n_classes=n_classes,
            latent_size=latent_size,
            n_hidden_layers=n_hidden_layers,
            gaussian_noise=gaussian_noise,
            dropout=dropout,
            feat_drop=feat_drop,
            activation=activation,
            bounded_gate=False,
            layer_norm=layer_norm,
            legacy_weight_norm=False,
            first_layer_spectral_scale="none",
            feature_score_mode="ones",
        )

    def _validate_unit_directions(self) -> None:
        if torch.any(self.first_raw_weight_norm <= 1e-8):
            raise RuntimeError(
                "GatedUnitGateMLP requires every raw direction column to have "
                "norm greater than 1e-8"
            )

    def get_raw_gate(self, training_drop: bool = False) -> torch.Tensor:
        return self.get_gate_values(training_drop=training_drop)

    def get_effective_gate(self, training_drop: bool = False) -> torch.Tensor:
        self._validate_unit_directions()
        return self.get_raw_gate(training_drop=training_drop)

    def get_constraint_coordinate(self) -> torch.Tensor:
        return self.get_effective_gate(training_drop=False)

    def get_selection_scores(
        self,
        readout: str = "effective_gate",
    ) -> torch.Tensor:
        if readout not in {"effective_gate", "raw_gate"}:
            raise ValueError("readout must be 'effective_gate' or 'raw_gate'")
        return self.get_effective_gate(training_drop=False).abs()

    def get_feature_selection_scores(self) -> torch.Tensor:
        """Compatibility alias for the primary effective-gate readout."""
        return self.get_selection_scores("effective_gate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_unit_directions()
        return super().forward(x)


class GatedExplicitScaleMLP(GatedWeightNormMLP):
    """Gated MLP with an explicit, functional first-layer feature scale.

    Each raw first-layer column is decomposed as ``W[:, j] = a_j * v_j`` with
    ``a_j = ||W[:, j]||_2`` and ``||v_j||_2 = 1``.  The forward pass uses
    ``x * (a * s)`` followed by the normalized directions ``v``.  Therefore
    ``a`` is part of the predictor rather than a detached pruning saliency,
    and the sparse ADMM variable is the effective gate ``g = a * s``.

    Choosing this class is an explicit opt-in to the joint constraint
    ``z = g = a * s``.  Legacy score-weighted ``C / score_j`` behavior remains
    in the older model/mode combinations and is not used by this architecture.
    """

    admm_constraint_mode = "joint_explicit_scale"
    training_variant = "effective_factorized"

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        latent_size: int = DEFAULT_SADMM_LATENT_SIZE,
        n_hidden_layers: int = DEFAULT_SADMM_HIDDEN_LAYERS,
        gaussian_noise: float = 0.0,
        dropout: float = DEFAULT_SADMM_DROPOUT,
        feat_drop: float = 0.6,
        activation: str = DEFAULT_SADMM_ACTIVATION,
        bounded_gate: bool = False,
        layer_norm: int = 0,
        first_layer_spectral_scale: str = "none",
    ):
        if bounded_gate:
            raise ValueError(
                "GatedExplicitScaleMLP currently requires bounded_gate=False "
                "so the ADMM variable is the same gate used in forward"
            )
        if first_layer_spectral_scale != "none":
            raise ValueError(
                "GatedExplicitScaleMLP requires "
                "first_layer_spectral_scale='none' so effective-gate norms "
                "match actual first-layer column norms"
            )
        super().__init__(
            input_size=input_size,
            n_classes=n_classes,
            latent_size=latent_size,
            n_hidden_layers=n_hidden_layers,
            gaussian_noise=gaussian_noise,
            dropout=dropout,
            feat_drop=feat_drop,
            activation=activation,
            bounded_gate=False,
            layer_norm=layer_norm,
            legacy_weight_norm=False,
            first_layer_spectral_scale=first_layer_spectral_scale,
            feature_score_mode="ones",
        )

        # Canonical initialization fixes the starting gauge at a_j=1 while
        # preserving the sampled column directions.  Subsequent training may
        # change a dynamically.
        with torch.no_grad():
            initial_norm = self.first_raw_weight_norm.clamp_min(1e-8)
            self.first_raw_weight.div_(initial_norm.unsqueeze(0))

    def get_admm_feature_scale(self) -> torch.Tensor:
        """Compatibility accessor for the exact, unclamped scale ``a``."""
        return self.first_raw_weight_norm

    def get_raw_gate(self, training_drop: bool = False) -> torch.Tensor:
        """Return the raw factor ``s`` before multiplication by ``a``."""
        return self.get_gate_values(training_drop=training_drop)

    def get_effective_gate(
        self,
        training_drop: bool = False,
    ) -> torch.Tensor:
        """Return the predictor-aligned gate ``g = a * s``."""
        return self.first_raw_weight_norm * self.get_raw_gate(
            training_drop=training_drop
        )

    def get_constraint_coordinate(self) -> torch.Tensor:
        """Return the differentiable coordinate constrained to the ADMM copy."""
        return self.get_effective_gate(training_drop=False)

    def get_functional_gate_values(
        self,
        training_drop: bool = False,
    ) -> torch.Tensor:
        """Compatibility alias for :meth:`get_effective_gate`."""
        return self.get_effective_gate(training_drop=training_drop)

    def get_feature_scores(self) -> torch.Tensor:
        """Disable all external ``C / score_j`` weighting for this model."""
        return torch.ones_like(self.gate)

    def get_selection_scores(
        self,
        readout: str = "effective_gate",
    ) -> torch.Tensor:
        """Return effective-gate scores or the raw-factor diagnostic."""
        if readout == "effective_gate":
            return self.get_effective_gate(training_drop=False).abs()
        if readout == "raw_gate":
            return self.get_raw_gate(training_drop=False).abs()
        raise ValueError("readout must be 'effective_gate' or 'raw_gate'")

    def get_feature_selection_scores(self) -> torch.Tensor:
        """Compatibility alias for the primary effective-gate readout."""
        return self.get_selection_scores("effective_gate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The clamp is only the safe denominator used to form a direction.
        # Multiplying by the same safe scale here makes it cancel exactly:
        #   (safe_a * s) * (W / safe_a) == s * W.
        # Constraint and readout coordinates use the true a=||W||, not safe_a.
        raw_gate = self.get_raw_gate(training_drop=True)
        safe_scale = self.first_weight_norm
        x_scaled = x * (safe_scale * raw_gate)
        h = F.linear(x_scaled, self.first_effective_weight, self.first_bias)
        h = F.mish(h)
        return self.layers(h)


class ParentGatedWeightNormMLP(nn.Module):
    """Strict-WN MLP with one gate per original parent feature.

    The input is still the expanded polynomial design matrix. For each expanded
    term, its input multiplier is shared from the original features that
    generated that term. For degree-2 data this means:
      x_j      -> g_j
      x_j^2    -> g_j
      x_j x_k  -> g_j * g_k

    ADMM therefore operates on the original feature gates, not on every
    expanded polynomial term separately.
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        parent_powers: np.ndarray,
        latent_size: int = DEFAULT_SADMM_LATENT_SIZE,
        n_hidden_layers: int = DEFAULT_SADMM_HIDDEN_LAYERS,
        gaussian_noise: float = 0.0,
        dropout: float = DEFAULT_SADMM_DROPOUT,
        feat_drop: float = 0.6,
        activation: str = DEFAULT_SADMM_ACTIVATION,
        bounded_gate: bool = False,
        layer_norm: int = 0,
        legacy_weight_norm: bool = False,
        first_layer_spectral_scale: str = "none",
        feature_score_mode: str = "auto",
    ):
        super().__init__()
        if first_layer_spectral_scale not in FIRST_LAYER_SPECTRAL_SCALE_MODES:
            raise ValueError(
                f"Unknown first_layer_spectral_scale={first_layer_spectral_scale!r}; "
                f"expected one of {sorted(FIRST_LAYER_SPECTRAL_SCALE_MODES)}"
            )
        if feature_score_mode not in {"auto", "ones", "raw_column_norm"}:
            raise ValueError(
                f"Unknown feature_score_mode={feature_score_mode!r}; "
                "expected one of ['auto', 'ones', 'raw_column_norm']"
            )
        powers_np = np.asarray(parent_powers)
        if powers_np.ndim != 2 or powers_np.shape[0] != input_size:
            raise ValueError(
                "parent_powers must have shape "
                f"(input_size, n_parent_features); got {powers_np.shape}, input_size={input_size}"
            )

        self.feat_drop = feat_drop
        self.bounded_gate = bounded_gate
        self.legacy_weight_norm = legacy_weight_norm
        self.first_layer_spectral_scale = first_layer_spectral_scale
        self.feature_score_mode = feature_score_mode
        self.input_size = input_size
        self.parent_input_size = int(powers_np.shape[1])
        self.latent_size = latent_size
        self.gate = nn.Parameter(torch.ones(self.parent_input_size))
        n_out = 1 if n_classes <= 2 else n_classes

        powers_float = torch.as_tensor(powers_np, dtype=torch.float32)
        degree = powers_float.sum(dim=1)
        degree_safe = torch.clamp(degree, min=1.0)
        parent_share = powers_float / degree_safe.unsqueeze(1)
        parent_share = torch.where(
            degree.unsqueeze(1) > 0,
            parent_share,
            torch.zeros_like(parent_share),
        )
        self.register_buffer("parent_share", parent_share)

        parent_mask_np = powers_np > 0
        parent_counts = parent_mask_np.sum(axis=1)
        max_parents = max(1, int(parent_counts.max()) if parent_counts.size else 1)
        parent_index = np.zeros((input_size, max_parents), dtype=np.int64)
        parent_mask = np.zeros((input_size, max_parents), dtype=np.bool_)
        for term_idx, parents in enumerate(np.flatnonzero(row) for row in parent_mask_np):
            if len(parents) == 0:
                continue
            parent_index[term_idx, : len(parents)] = parents
            parent_mask[term_idx, : len(parents)] = True
        self.register_buffer(
            "term_parent_index",
            torch.as_tensor(parent_index, dtype=torch.long),
        )
        self.register_buffer(
            "term_parent_mask",
            torch.as_tensor(parent_mask, dtype=torch.bool),
        )

        self.first_raw_weight = nn.Parameter(torch.empty(latent_size, input_size))
        self.first_bias = nn.Parameter(torch.empty(latent_size))
        nn.init.kaiming_uniform_(self.first_raw_weight, a=np.sqrt(5))
        bound = 1.0 / np.sqrt(input_size) if input_size > 0 else 0.0
        nn.init.uniform_(self.first_bias, -bound, bound)

        layers: list[nn.Module] = []
        if gaussian_noise > 0:
            layers.append(GaussianNoise(gaussian_noise))

        inplace = False
        for _ in range(n_hidden_layers - 1):
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout, inplace=inplace))
            layers.append(nn.Linear(latent_size, latent_size))
            if layer_norm:
                layers.append(nn.LayerNorm(latent_size))

            if activation == "relu":
                act: nn.Module = nn.ReLU(inplace=inplace)
            elif activation == "leakyrelu":
                act = nn.LeakyReLU(0.2, inplace=inplace)
            elif activation == "prelu":
                act = nn.PReLU(latent_size)
            elif activation == "tanh":
                act = nn.Tanh()
            elif activation == "sigmoid":
                act = nn.Sigmoid()
            elif activation == "mish":
                act = nn.Mish(inplace=inplace)
            elif activation == "selu":
                act = nn.SELU(inplace=inplace)
            else:
                act = nn.Hardswish(inplace=inplace)
            layers.append(act)

        layers.append(nn.Linear(latent_size, n_out))
        self.layers = nn.Sequential(*layers)

    @property
    def first_weight_norm(self) -> torch.Tensor:
        """Parent-feature raw-weight scale aggregated from expanded terms."""
        term_norm = torch.norm(self.first_raw_weight, p=2, dim=0).clamp_min(1e-8)
        parent_norm = torch.matmul(term_norm, self.parent_share)
        return parent_norm.clamp_min(1e-8)

    @property
    def first_normalized_weight(self) -> torch.Tensor:
        return F.normalize(self.first_raw_weight, p=2, dim=0, eps=1e-8)

    @property
    def first_spectral_scale_multiplier(self) -> torch.Tensor:
        return _column_normalized_spectral_scale(
            self.first_normalized_weight,
            self.first_layer_spectral_scale,
            self.input_size,
            self.latent_size,
        )

    @property
    def first_effective_weight(self) -> torch.Tensor:
        return self.first_spectral_scale_multiplier * self.first_normalized_weight

    @property
    def weight(self) -> torch.Tensor:
        return self.first_effective_weight

    @property
    def bias(self) -> torch.Tensor:
        return self.first_bias

    @property
    def first_linear(self) -> nn.Module:
        return self

    def get_feature_scores(self) -> torch.Tensor:
        if self.feature_score_mode == "raw_column_norm":
            return self.first_weight_norm
        if self.feature_score_mode == "ones":
            return torch.ones_like(self.gate)
        if self.legacy_weight_norm:
            return self.first_weight_norm
        return torch.ones_like(self.gate)

    def first_layer_spectral_diagnostics(self) -> dict[str, float | str]:
        """Return detached first-layer operator-scale diagnostics."""
        with torch.no_grad():
            normalized = self.first_normalized_weight.detach()
            raw_spectral = torch.linalg.matrix_norm(normalized.float(), ord=2).item()
            scale = self.first_spectral_scale_multiplier.detach().float().item()
            target = float(
                np.sqrt(
                    max(1.0, float(self.latent_size))
                    / max(1.0, float(self.input_size))
                )
            )
        return {
            "mode": self.first_layer_spectral_scale,
            "target": target,
            "normalized_spectral_norm": float(raw_spectral),
            "multiplier": float(scale),
            "effective_spectral_norm": float(scale * raw_spectral),
        }

    def gate_from_parameter(self, gate_param: torch.Tensor) -> torch.Tensor:
        if self.bounded_gate:
            return torch.sigmoid(gate_param)
        return gate_param

    def parameter_from_gate(self, gate_value: torch.Tensor) -> torch.Tensor:
        if self.bounded_gate:
            gate_value = torch.clamp(gate_value, min=1e-6, max=1.0 - 1e-6)
            return torch.logit(gate_value)
        return gate_value

    def get_gate_values(self, training_drop: bool = False) -> torch.Tensor:
        g = self.gate_from_parameter(self.gate)
        if self.training and training_drop and self.feat_drop > 0:
            mask = (torch.rand(g.shape, device=g.device) > self.feat_drop).float()
            g = g * mask / (1.0 - self.feat_drop + 1e-8)
        return g

    def get_expanded_gate_values(self, training_drop: bool = False) -> torch.Tensor:
        parent_gate = self.get_gate_values(training_drop=training_drop)
        gathered = parent_gate[self.term_parent_index]
        factors = torch.where(
            self.term_parent_mask,
            gathered,
            torch.ones_like(gathered),
        )
        return torch.prod(factors, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term_gate = self.get_expanded_gate_values(training_drop=True)
        if self.legacy_weight_norm:
            term_gate = term_gate * torch.norm(self.first_raw_weight, p=2, dim=0).clamp_min(1e-8)

        x_scaled = x * term_gate
        h = F.linear(x_scaled, self.first_effective_weight, self.first_bias)
        h = F.mish(h)
        return self.layers(h)


# ---------------------------------------------------------------------------
# Strategy 2: z-score data standardisation
# ---------------------------------------------------------------------------


class _Scaler:
    """Z-score scaler that remembers train statistics."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return (X - self.mean_) / self.std_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


# ---------------------------------------------------------------------------
# WANDA-like score computation for tabular MLP
# ---------------------------------------------------------------------------


def compute_mlp_wanda_scores(
    model: nn.Module, X_sample: torch.Tensor
) -> list[torch.Tensor]:
    """Compute per-parameter WANDA-inspired importance scores.

    For each Linear layer:  score = |W| * ||activation_input||_2
    For other param types:    score = ones (no pruning importance info).

    We run a forward pass on ``X_sample`` to capture activations, then
    combine them with weight magnitudes.

    Returns a list of tensors aligned with ``list(model.parameters())``.
    """
    activations: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(module, inp, out):
            activations[name] = inp[0].detach()

        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(X_sample)

    for h in hooks:
        h.remove()

    # Build scores list aligned with model.parameters()
    param_to_module: dict[int, str] = {}
    for name, module in model.named_modules():
        for pname, p in module.named_parameters(recurse=False):
            param_to_module[id(p)] = name

    scores: list[torch.Tensor] = []
    for p in model.parameters():
        mod_name = param_to_module.get(id(p))
        if mod_name and mod_name in activations:
            act = activations[mod_name]  # (batch, in_features)
            act_norm = torch.norm(act, p=2, dim=0)  # (in_features,)
            if p.dim() == 2:
                # Weight matrix: score_ij = |W_ij| * ||a_j||_2
                scores.append(torch.abs(p.data) * act_norm.unsqueeze(0))
            elif p.dim() == 1 and p.shape[0] == act.shape[1]:
                # Bias: just use magnitude
                scores.append(torch.abs(p.data) + 1e-8)
            else:
                scores.append(torch.ones_like(p.data))
        else:
            scores.append(torch.ones_like(p.data))

    return scores


# ---------------------------------------------------------------------------
# Feature score helper for ADMM thresholds
# ---------------------------------------------------------------------------


def _get_feature_penalty_scores(model: nn.Module) -> torch.Tensor:
    """Return one positive score per original input feature.

    Standard MLP variants use the first dense layer column norms.
    Custom backbones can expose `get_feature_scores()` when there is no
    feature-aligned dense first layer.
    """
    device = next(model.parameters()).device
    if hasattr(model, "get_feature_scores"):
        score = model.get_feature_scores()
        if not torch.is_tensor(score):
            score = torch.as_tensor(score, dtype=torch.float32, device=device)
        else:
            score = score.to(device=device, dtype=torch.float32)
        return torch.clamp(score.detach(), min=1e-8)

    W1 = model.first_linear.weight
    return torch.norm(W1, p=2, dim=0) + 1e-8


def _get_transformed_feature_penalty_scores(
    model: nn.Module,
    score_update_mode: str,
) -> torch.Tensor:
    """Return score diagnostics that isolate scale, mapping, and feedback."""
    raw_score = _get_feature_penalty_scores(model)

    if score_update_mode == "epoch_median_normalized":
        median = torch.median(raw_score).clamp_min(1e-8)
        return torch.clamp(raw_score / median, min=1e-8)

    if score_update_mode == "epoch_bounded_normalized":
        # The weighted ADMM code uses C / score_j.  Returning this transformed
        # score therefore gives the bounded, scale-free penalty multiplier
        #
        #     q_j = 1 / transformed_score_j = 2 m / (score_j + m),
        #
        # where m is the epoch median.  Thus q_j is in (0, 2], q_j=1 at the
        # median, and multiplying every raw score by a constant changes nothing.
        median = torch.median(raw_score).clamp_min(1e-8)
        return torch.clamp((raw_score + median) / (2.0 * median), min=0.5)

    if score_update_mode == "epoch_permuted":
        permutation = getattr(model, "_penalty_score_permutation", None)
        if not torch.is_tensor(permutation) or permutation.numel() != raw_score.numel():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(SCORE_PERMUTATION_SEED)
            permutation = torch.randperm(raw_score.numel(), generator=generator)
            model._penalty_score_permutation = permutation
        return raw_score[permutation.to(device=raw_score.device)]

    if score_update_mode == "freeze_run_initial":
        frozen = getattr(model, "_run_initial_penalty_score", None)
        if not torch.is_tensor(frozen) or frozen.numel() != raw_score.numel():
            frozen = raw_score.detach().cpu().clone()
            model._run_initial_penalty_score = frozen
        return frozen.to(device=raw_score.device, dtype=raw_score.dtype)

    return raw_score


def _strict_two_copy_state_residuals(
    gate: torch.Tensor,
    z: torch.Tensor,
    y: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    data_gate_grad: torch.Tensor,
    C: float,
    rho: float,
    zero_tol: float = 1e-8,
) -> Dict[str, Any]:
    """Evaluate KKT and block fixed-point residuals for scalar two-copy ADMM.

    The constrained objective is

        L(theta, g) + C ||z||_1 / ||y||_2,
        subject to g = z and g = y.

    The unscaled multipliers are ``rho*u`` and ``rho*w``.  This helper is
    intentionally independent of the training loop so the multiplier signs
    and exact equal-gate fixed point can be unit tested directly.
    """

    def _norm(value: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(value).item())

    def _summary(value: torch.Tensor, denominator: float) -> Dict[str, float]:
        flat = value.detach().reshape(-1)
        l2 = _norm(flat)
        rms = l2 / float(np.sqrt(max(int(flat.numel()), 1)))
        max_abs = float(flat.abs().max().item()) if flat.numel() else 0.0
        return {
            "l2": l2,
            "rms": rms,
            "max_abs": max_abs,
            "numel": int(flat.numel()),
            "scale": float(denominator),
            "relative": l2 / max(float(denominator), 1e-12),
        }

    gate = gate.detach()
    z = z.detach()
    y = y.detach()
    u = u.detach()
    w = w.detach()
    data_gate_grad = data_gate_grad.detach()

    lambda_z = float(rho) * u
    lambda_y = float(rho) * w
    primal_z = gate - z
    primal_y = gate - y

    y_norm_tensor = torch.linalg.vector_norm(y)
    y_norm = float(y_norm_tensor.item())
    y_norm_safe = torch.clamp(y_norm_tensor, min=1e-12)
    z_l1 = torch.linalg.vector_norm(z, ord=1)
    alpha = gate.new_tensor(float(C)) / y_norm_safe

    gate_stationarity = data_gate_grad + lambda_z + lambda_y

    z_active = z.abs() > float(zero_tol)
    z_stationarity = torch.zeros_like(z)
    z_stationarity[z_active] = (
        alpha * torch.sign(z[z_active]) - lambda_z[z_active]
    )
    z_inactive = ~z_active
    z_stationarity[z_inactive] = torch.sign(lambda_z[z_inactive]) * torch.relu(
        lambda_z[z_inactive].abs() - alpha
    )

    ratio_y_grad = -float(C) * z_l1 * y / (y_norm_safe**3)
    y_stationarity = ratio_y_grad - lambda_y

    augmented_gate_stationarity = (
        data_gate_grad
        + float(rho) * (gate - z + u)
        + float(rho) * (gate - y + w)
    )

    with torch.no_grad():
        z_expected, _ = strict_ratio_z_update(
            gate,
            y,
            u,
            C=float(C),
            rho=float(rho),
        )
        y_expected, _, _ = strict_ratio_y_update(
            gate,
            z,
            w,
            C=float(C),
            rho=float(rho),
            fallback_direction=y,
        )
    z_fixed_point = z - z_expected
    y_fixed_point = y - y_expected

    gate_norm = _norm(gate)
    z_norm = _norm(z)
    lambda_z_norm = _norm(lambda_z)
    lambda_y_norm = _norm(lambda_y)
    data_gate_grad_norm = _norm(data_gate_grad)
    dual_gate_norm = _norm(lambda_z + lambda_y)
    ratio_y_grad_norm = _norm(ratio_y_grad)
    data_dual_cosine = float(
        torch.dot(data_gate_grad.reshape(-1), (lambda_z + lambda_y).reshape(-1)).item()
        / max(data_gate_grad_norm * dual_gate_norm, 1e-12)
    )
    gate_stationarity_radial = float(
        torch.dot(gate_stationarity.reshape(-1), gate.reshape(-1)).item()
        / max(gate_norm, 1e-12)
    )
    gate_stationarity_tangential = float(
        np.sqrt(
            max(
                _norm(gate_stationarity) ** 2 - gate_stationarity_radial**2,
                0.0,
            )
        )
    )
    active_scale = float(alpha.item()) * float(
        np.sqrt(max(int(z_active.sum().item()), 1))
    )

    residuals = {
        "primal_z": _summary(primal_z, max(gate_norm, z_norm)),
        "primal_y": _summary(primal_y, max(gate_norm, y_norm)),
        "gate_kkt": _summary(
            gate_stationarity,
            max(data_gate_grad_norm, dual_gate_norm),
        ),
        "gate_augmented_current": _summary(
            augmented_gate_stationarity,
            max(
                data_gate_grad_norm,
                _norm(float(rho) * (gate - z + u)),
                _norm(float(rho) * (gate - y + w)),
            ),
        ),
        "z_kkt": _summary(
            z_stationarity,
            max(lambda_z_norm, active_scale),
        ),
        "y_kkt": _summary(
            y_stationarity,
            max(lambda_y_norm, ratio_y_grad_norm),
        ),
        "z_block_fixed_point": _summary(
            z_fixed_point,
            max(z_norm, _norm(z_expected)),
        ),
        "y_block_fixed_point": _summary(
            y_fixed_point,
            max(y_norm, _norm(y_expected)),
        ),
    }
    core_keys = (
        "primal_z",
        "primal_y",
        "gate_kkt",
        "z_kkt",
        "y_kkt",
        "z_block_fixed_point",
        "y_block_fixed_point",
    )
    maximum_core_relative = max(
        float(residuals[key]["relative"]) for key in core_keys
    )
    return {
        "objective": "L(theta,g)+C*||z||1/||y||2; g=z; g=y",
        "multiplier_convention": "lambda_z=rho*u; lambda_y=rho*w",
        "C": float(C),
        "rho": float(rho),
        "z_zero_tol": float(zero_tol),
        "z_active": int(z_active.sum().item()),
        "y_norm": y_norm,
        "y_norm_was_clamped": bool(y_norm < 1e-12),
        "z_l1": float(z_l1.item()),
        "z_subgradient_radius": float(alpha.item()),
        "component_norms": {
            "data_gate_grad_l2": data_gate_grad_norm,
            "lambda_z_l2": lambda_z_norm,
            "lambda_y_l2": lambda_y_norm,
            "lambda_z_plus_lambda_y_l2": dual_gate_norm,
            "ratio_y_grad_l2": ratio_y_grad_norm,
            "data_vs_lambda_sum_cosine": data_dual_cosine,
            "gate_kkt_radial_signed": gate_stationarity_radial,
            "gate_kkt_tangential_l2": gate_stationarity_tangential,
        },
        "residuals": residuals,
        "maximum_core_relative": maximum_core_relative,
        "near_stationary_core_at_1e-2": bool(
            maximum_core_relative <= STRICT_STATIONARITY_RELATIVE_TOL
        ),
    }


def _strict_two_copy_stationarity_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    n_classes: int,
    z: torch.Tensor,
    y: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    C: float,
    rho: float,
    weight_decay: float = 1e-3,
) -> Dict[str, Any]:
    """Compute deterministic full-train KKT diagnostics without updating state."""

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    n_total = max(int(len(loader.dataset)), 1)
    data_loss_sum = 0.0

    with torch.enable_grad():
        for x_batch, y_batch in loader:
            device = model.gate.device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_hat = model(x_batch)
            if n_classes > 2:
                y_hat = torch.log_softmax(y_hat, dim=1)
            else:
                y_hat = y_hat.reshape(len(y_hat))
            try:
                batch_loss = criterion(y_hat, y_batch)
            except RuntimeError:
                batch_loss = criterion(y_hat, y_batch.float())
            batch_n = int(x_batch.shape[0])
            data_loss_sum += float(batch_loss.detach().item()) * batch_n
            (batch_loss * (float(batch_n) / float(n_total))).backward()

        decay_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and name != "gate"
            and not name.endswith(".gate")
        ]
        if decay_parameters and weight_decay > 0:
            decay_loss = 0.5 * float(weight_decay) * sum(
                torch.sum(parameter**2) for parameter in decay_parameters
            )
            decay_loss.backward()
            decay_loss_value = float(decay_loss.detach().item())
        else:
            decay_loss_value = 0.0

    gate_grad = (
        torch.zeros_like(model.gate)
        if model.gate.grad is None
        else model.gate.grad.detach().clone()
    )
    theta_grad_sq = 0.0
    theta_param_sq = 0.0
    theta_grad_max = 0.0
    theta_numel = 0
    for name, parameter in model.named_parameters():
        if name == "gate" or name.endswith(".gate") or not parameter.requires_grad:
            continue
        grad = (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach()
        )
        theta_grad_sq += float(torch.sum(grad**2).item())
        theta_param_sq += float(torch.sum(parameter.detach() ** 2).item())
        theta_grad_max = max(theta_grad_max, float(grad.abs().max().item()))
        theta_numel += int(parameter.numel())

    theta_grad_l2 = float(np.sqrt(theta_grad_sq))
    theta_param_l2 = float(np.sqrt(theta_param_sq))
    theta_relative = theta_grad_l2 / max(theta_param_l2, 1e-12)
    state_residuals = _strict_two_copy_state_residuals(
        model.gate.detach(),
        z,
        y,
        u,
        w,
        gate_grad,
        C=float(C),
        rho=float(rho),
    )
    state_residuals["gradient_evaluation"] = {
        "mode": "model.eval full training fold; dropout disabled",
        "n_samples": int(n_total),
        "mean_data_loss": data_loss_sum / float(n_total),
        "weight_decay": float(weight_decay),
        "weight_decay_objective": decay_loss_value,
    }
    state_residuals["theta_kkt"] = {
        "l2": theta_grad_l2,
        "rms": theta_grad_l2 / float(np.sqrt(max(theta_numel, 1))),
        "max_abs": theta_grad_max,
        "parameter_l2": theta_param_l2,
        "relative": theta_relative,
        "numel": int(theta_numel),
    }
    maximum_all_relative = max(
        float(state_residuals["maximum_core_relative"]),
        float(theta_relative),
    )
    state_residuals["maximum_all_relative"] = maximum_all_relative
    state_residuals["near_stationary_all_at_1e-2"] = bool(
        maximum_all_relative <= STRICT_STATIONARITY_RELATIVE_TOL
    )

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return state_residuals


def _strict_stationarity_combined_criterion(
    stationarity: Dict[str, Any],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> Dict[str, Any]:
    """Apply a scale-aware absolute-plus-relative stationarity threshold."""

    checks: Dict[str, Dict[str, Any]] = {}
    for name, residual in stationarity["residuals"].items():
        l2 = float(residual["l2"])
        numel = int(residual.get("numel", 1))
        scale = float(
            residual.get(
                "scale",
                l2 / max(float(residual.get("relative", 0.0)), 1e-12),
            )
        )
        threshold = float(absolute_tolerance) * float(np.sqrt(max(numel, 1)))
        threshold += float(relative_tolerance) * max(scale, 0.0)
        checks[name] = {
            "l2": l2,
            "numel": numel,
            "scale": scale,
            "threshold": threshold,
            "normalized_violation": l2 / max(threshold, 1e-30),
            "passed": bool(l2 <= threshold),
        }

    theta = stationarity["theta_kkt"]
    theta_l2 = float(theta["l2"])
    theta_numel = int(theta["numel"])
    theta_scale = float(theta["parameter_l2"])
    theta_threshold = float(absolute_tolerance) * float(
        np.sqrt(max(theta_numel, 1))
    ) + float(relative_tolerance) * max(theta_scale, 0.0)
    checks["theta_kkt"] = {
        "l2": theta_l2,
        "numel": theta_numel,
        "scale": theta_scale,
        "threshold": theta_threshold,
        "normalized_violation": theta_l2 / max(theta_threshold, 1e-30),
        "passed": bool(theta_l2 <= theta_threshold),
    }
    maximum_violation = max(
        float(check["normalized_violation"]) for check in checks.values()
    )
    return {
        "relative_tolerance": float(relative_tolerance),
        "absolute_tolerance": float(absolute_tolerance),
        "checks": checks,
        "maximum_normalized_violation": maximum_violation,
        "passed": bool(maximum_violation <= 1.0),
    }


# ---------------------------------------------------------------------------
# ADMM-Gate training loop for Input-Group feature selection
# ---------------------------------------------------------------------------


def _train_input_group(
    model,  # GatedFeatureSelectionMLP
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    lr: float = 0.005,
    weight_decay: float = 1e-3,
    C: float = 0.1,  # Ratio Norm sparsity coefficient
    C_schedule_per_epoch: Optional[list[float]] = None,
    epochs: int = 500,
    warmup_epochs: int = 120,
    batch_size: int = 64,
    g_inner_epochs: int = 1,
    rho_init: float = 200.0,
    rho_update_interval: int = 5,
    score_refresh_interval: int = 10,
    score_update_mode: str = "epoch",
    device: Optional[str] = None,
    use_ratio_norm: bool = True,  # False -> plain L1 proximal (ablation)
    ratio_norm_update: str = "formula_z_den",  # canonical_signed is the theory-aligned one-copy solver
    ratio_norm_dim_scale: str = "none",  # "none" or "pqi"
    ratio_norm_p: float = 1.0,
    ratio_norm_q: float = 2.0,
    use_admm: bool = True,  # False -> proximal gradient (ablation)
    uniform_penalty: bool = False,  # True -> lambda_j = C; False -> lambda_j = C / s_j
    three_variable: bool = False,  # True -> paper's 3-variable ADMM (y,z with duals v,w)
    write_z_to_gate: bool = False,  # Experimental: copy z-step prox result back into model.gate
    n_features: int = None,  # Feature dimension for rho scaling
    # Phase 1 experimental improvements:
    optimizer_type: str = "adam",  # "adam" or "adagrad"
    use_early_stopping: bool = False,
    patience: int = 66,
    val_split: float = 0.2,  # Validation split for early stopping
    initial_admm_state: Optional[dict] = None,
    min_z_support: int = 0,
    admm_weighted_gate: bool = False,
    adaptive_rho: bool = True,
    adaptive_rho_mu: float = 10.0,
    adaptive_rho_tau: float = 2.0,
    adaptive_rho_min: Optional[float] = None,
    adaptive_rho_max: float = 1e4,
    auto_C_gamma: Optional[float] = None,
    auto_C_support: int = 0,
    adaptive_C: bool = False,
    adaptive_C_target: int = 0,
    adaptive_C_update_interval: int = 5,
    adaptive_C_low_ratio: float = 0.75,
    adaptive_C_high_ratio: float = 1.5,
    adaptive_C_mult: float = 1.25,
    adaptive_C_min: float = 1e-8,
    adaptive_C_max: float = 1e3,
    dynamic_C_mode: str = "none",
    dynamic_C_update_interval: int = 5,
    dynamic_C_ema: float = 0.8,
    dynamic_C_min_factor: float = 0.1,
    dynamic_C_max_factor: float = 10.0,
    dynamic_C_kappa: float = 1.0,
    dynamic_C_quantile: float = 0.9,
    dynamic_C_continuation_gamma: float = 1.5,
    dynamic_C_stability_patience: int = 2,
    dynamic_C_support_tol: float = 0.02,
    dynamic_C_residual_tol: float = 0.10,
    dynamic_C_loss_tol: float = 0.02,
    dynamic_C_cap_seed: Optional[float] = None,
    dynamic_C_floor: Optional[float] = None,
    dynamic_C_support_parent_low: int = 0,
    dynamic_C_support_parent_high: int = 0,
    dynamic_C_support_term_low: int = 0,
    dynamic_C_support_term_high: int = 0,
    dynamic_C_support_up_mult: float = 1.25,
    dynamic_C_support_down_mult: float = 1.5,
    group_sparse_powers: Optional[np.ndarray] = None,
    group_sparse_mode: str = "none",
    group_parent_support: int = 0,
    diagnostic_ground_truth: Optional[list[int]] = None,
    diagnostic_k: int = 0,
    lr_scheduler: str = "none",
    cosine_eta_min: Optional[float] = None,
    cosine_decay_fraction: float = 1.0,
    rho_follow_lr: bool = False,
    strict_neural_block_mode: str = "joint_minibatch",
    strict_kkt_check_interval: int = 0,
    strict_kkt_tolerance: float = STRICT_STATIONARITY_RELATIVE_TOL,
    strict_kkt_absolute_tolerance: float = 0.0,
    strict_kkt_patience: int = 3,
    epoch_end_callback: Optional[
        Callable[[nn.Module, Dict[str, Any]], None]
    ] = None,
    epoch_callback_include_optimizer_moments: bool = False,
) -> None:
    """Train a GatedFeatureSelectionMLP with Linearized ADMM + Ratio Norm.

    **Phase 1 - Warm-up** (epoch 0 to warmup_epochs - 1):
        All parameters trained with Adam.  Feature dropout (in the model)
        prevents memorisation, letting the MLP learn the true signal.

    **Phase 2 - Linearized ADMM pruning** (epoch warmup_epochs to epochs - 1):
        Uses ADMM-style splitting: min L(theta, g) + lambda * R(z), s.t. g = z
        where R(z) = ||z||_1 / ||z||_2 is the Ratio Norm (scale-invariant).
        In the current effective-coordinate contract, ``g`` is the functional
        gate. ``GatedExplicitScaleMLP`` parameterizes it as ``g = a * s``;
        ``GatedUnitGateMLP`` fixes ``a = 1`` and stores ``g`` directly.
        With admm_weighted_gate=True, the legacy diagnostic constraint becomes
        A*g = z, where A is a detached per-feature first-layer column norm.
        GatedExplicitScaleMLP instead declares the joint constraint
        ``z = g = a * s``, where ``a`` is recomputed differentiably inside
        every neural-block step and is also part of that model's forward pass.
        The default ratio_norm_update="formula_z_den" uses a denominator-copy
        block update whose z-subproblem is an exact soft-thresholding step for
        a fixed denominator copy. ratio_norm_update="formula_center_den" uses
        the current ADMM center norm ||g + u||_2 as a detached denominator,
        removing z_den trajectory dependence from the support threshold.
        ratio_norm_update="formula_center_den_weighted" uses the same center
        denominator with legacy score weights lambda_j = C / s_j when
        uniform_penalty=False.
        ratio_norm_update="formula_z_den_weighted" uses the denominator-copy
        block with legacy score weights lambda_j = C / s_j when
        uniform_penalty=False. The legacy
        ratio_norm_update="current" branch
        applies a sequential soft-threshold-then-cubic-rescale approximation and
        should not be described as an exact RatioNorm proximal operator.

        Per epoch:
          1. **neural block**: by default, Adam on ALL params (theta AND g)
             with minibatch augmented loss. The diagnostic
             strict_neural_block_mode="theta_minibatch_gate_fullfold" instead
             updates theta for one minibatch pass with gate fixed, then takes
             one deterministic full-training-fold Adam gate step with theta
             fixed and both strict augmented penalties included. The
             "theta_minibatch_gate_fullfold_inner" diagnostic repeats that
             full-fold gate step, with a fresh gradient each time, until the
             declared mixed augmented-gradient check passes or the matched
             minibatch-count safety cap is reached.
          2. **z-step**: denominator-copy soft-thresholding by default; legacy
             current mode uses cubic tau^3 - tau - D as an approximate rescale
          3. **Dual update**: u <- u + g - z
          4. **Adaptive rho**: Boyd section 3.4.1 with proper dual rescaling

        Some legacy and weighted configurations use score
        s_j = ||W1[:,j]||_2 to form importance-adaptive thresholds. The default
        denominator-copy update uses the scalar z-space threshold implied by
        its fixed-denominator subproblem.
        score_update_mode="freeze_after_warmup" fixes these scores after the
        warm-up pass, giving a fixed weighted objective within the ADMM phase.

    **Phase 1 Experimental Improvements:**
        - optimizer_type: "adam" (default) or "adagrad" (benchmark default)
        - use_early_stopping: enable validation-based early stopping
        - patience: number of epochs without improvement before stopping
    """
    N = len(X_train)
    n_features = X_train.shape[1]  # FIX: define n_features for rho_min calculation
    if int(g_inner_epochs) != g_inner_epochs or int(g_inner_epochs) < 1:
        raise ValueError("g_inner_epochs must be a positive integer")
    g_inner_epochs = int(g_inner_epochs)
    model_constraint_mode = getattr(model, "admm_constraint_mode", "gate")
    training_variant = getattr(model, "training_variant", "legacy")
    bounded_gate = getattr(model, "bounded_gate", False)
    if not np.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative")
    valid_model_constraint_modes = {"gate", "joint_explicit_scale"}
    if model_constraint_mode not in valid_model_constraint_modes:
        raise ValueError(
            f"Unknown model admm_constraint_mode={model_constraint_mode!r}; "
            f"expected one of {sorted(valid_model_constraint_modes)}"
        )
    joint_explicit_scale_mode = model_constraint_mode == "joint_explicit_scale"
    if joint_explicit_scale_mode and admm_weighted_gate:
        raise ValueError(
            "GatedExplicitScaleMLP already defines joint z=g=a*s; do not combine "
            "it with legacy detached admm_weighted_gate=True"
        )
    uses_scaled_gate_constraint = bool(
        admm_weighted_gate or joint_explicit_scale_mode
    )
    effective_admm_constraint_mode = (
        "joint_explicit_scale"
        if joint_explicit_scale_mode
        else ("detached_weighted_gate" if admm_weighted_gate else "gate")
    )
    valid_neural_block_modes = {
        "joint_minibatch",
        "theta_minibatch_gate_fullfold",
        "theta_minibatch_gate_fullfold_inner",
    }
    fullfold_neural_block_modes = valid_neural_block_modes - {"joint_minibatch"}
    if strict_neural_block_mode not in valid_neural_block_modes:
        raise ValueError(
            "strict_neural_block_mode must be one of "
            f"{sorted(valid_neural_block_modes)}"
        )
    if (
        strict_neural_block_mode in fullfold_neural_block_modes
        and ratio_norm_update != "strict_two_copy"
    ):
        raise ValueError(
            "full-fold gate block modes are defined only for strict_two_copy"
        )
    if (
        strict_neural_block_mode in fullfold_neural_block_modes
        and dynamic_C_mode != "none"
    ):
        raise ValueError(
            "full-fold gate block modes currently require fixed dynamic_C_mode=none"
        )
    if (
        strict_neural_block_mode == "theta_minibatch_gate_fullfold_inner"
        and g_inner_epochs != 1
    ):
        raise ValueError(
            "theta_minibatch_gate_fullfold_inner requires g_inner_epochs=1"
        )
    if adaptive_rho_mu <= 1.0:
        raise ValueError("adaptive_rho_mu must be greater than one")
    if adaptive_rho_tau <= 1.0:
        raise ValueError("adaptive_rho_tau must be greater than one")
    if adaptive_rho_min is not None and adaptive_rho_min <= 0.0:
        raise ValueError("adaptive_rho_min must be positive when provided")
    if adaptive_rho_max <= 0.0:
        raise ValueError("adaptive_rho_max must be positive")
    if adaptive_rho_min is not None and adaptive_rho_min > adaptive_rho_max:
        raise ValueError("adaptive_rho_min cannot exceed adaptive_rho_max")
    if strict_kkt_check_interval < 0:
        raise ValueError("strict_kkt_check_interval must be nonnegative")
    if strict_kkt_tolerance <= 0.0:
        raise ValueError("strict_kkt_tolerance must be positive")
    if strict_kkt_absolute_tolerance < 0.0:
        raise ValueError("strict_kkt_absolute_tolerance must be nonnegative")
    if strict_kkt_patience < 1:
        raise ValueError("strict_kkt_patience must be positive")
    if not 0.0 < cosine_decay_fraction <= 1.0:
        raise ValueError("cosine_decay_fraction must be in (0, 1]")

    valid_ratio_norm_updates = {
        "canonical_signed",
        "current",
        "formula_center_den",
        "formula_center_den_weighted",
        "formula_z_den",
        "formula_z_den_weighted",
        "strict_two_copy",
        "strict_two_copy_weighted",
        "direct_ratio_gradient",
    }
    if ratio_norm_update not in valid_ratio_norm_updates:
        raise ValueError(
            f"Unknown ratio_norm_update={ratio_norm_update!r}. "
            f"Expected one of {sorted(valid_ratio_norm_updates)}"
        )
    strict_two_copy_modes = {"strict_two_copy", "strict_two_copy_weighted"}
    strict_two_copy_mode = ratio_norm_update in strict_two_copy_modes
    canonical_signed_mode = ratio_norm_update == "canonical_signed"
    if joint_explicit_scale_mode:
        supported_explicit_scale_updates = {
            "canonical_signed",
            "current",
            "formula_center_den",
            "formula_z_den",
        }
        if not use_ratio_norm or not use_admm:
            raise ValueError(
                "joint explicit-scale mode requires use_ratio_norm=True and use_admm=True"
            )
        if use_early_stopping:
            raise ValueError(
                "joint explicit-scale mode currently requires "
                "use_early_stopping=False so restored model parameters cannot "
                "diverge from the final z/u state"
            )
        if three_variable:
            raise ValueError(
                "joint explicit-scale mode defines the single constraint z=g=a*s "
                "and cannot use the legacy three_variable path"
            )
        if ratio_norm_update not in supported_explicit_scale_updates:
            raise ValueError(
                "joint explicit-scale mode requires an unweighted single-dual "
                f"update from {sorted(supported_explicit_scale_updates)}; "
                f"got {ratio_norm_update!r}"
            )
    if canonical_signed_mode:
        if training_variant not in {"effective_factorized", "unit_gate"}:
            raise ValueError(
                "canonical_signed requires a model with the explicit "
                "effective-gate coordinate contract"
            )
        if not use_ratio_norm or not use_admm:
            raise ValueError(
                "canonical_signed requires use_ratio_norm=True and use_admm=True"
            )
        if bounded_gate or admm_weighted_gate or three_variable:
            raise ValueError(
                "canonical_signed requires an unbounded, unweighted, one-copy "
                "effective-gate constraint"
            )
        if not uniform_penalty:
            raise ValueError(
                "canonical_signed uses one global C and requires "
                "uniform_penalty=True"
            )
        if ratio_norm_dim_scale != "none":
            raise ValueError("canonical_signed uses C directly")
        if min_z_support != 0 or group_sparse_mode != "none":
            raise ValueError(
                "canonical_signed rejects post-prox support modification"
            )
        if weight_decay != 0.0:
            raise ValueError(
                "canonical_signed theory-aligned training requires "
                "weight_decay=0"
            )
        if score_update_mode != "epoch":
            raise ValueError(
                "canonical_signed does not use feature scores; keep "
                "score_update_mode='epoch'"
            )
        if adaptive_C or dynamic_C_mode != "none" or auto_C_gamma is not None:
            raise ValueError(
                "canonical_signed currently requires a directly specified "
                "global C without adaptive or dynamic C controls"
            )
    if strict_kkt_check_interval > 0 and ratio_norm_update != "strict_two_copy":
        raise ValueError(
            "strict KKT stopping currently requires unweighted strict_two_copy"
        )
    if strict_two_copy_mode:
        if not use_ratio_norm or not use_admm:
            raise ValueError(
                f"{ratio_norm_update} requires use_ratio_norm=True and use_admm=True"
            )
        if three_variable:
            raise ValueError(
                f"{ratio_norm_update} already defines the z/y two-copy splitting; "
                "do not combine it with the legacy three_variable path"
            )
        if uses_scaled_gate_constraint:
            raise ValueError(f"{ratio_norm_update} currently fixes s(g)=g")
        if ratio_norm_dim_scale != "none":
            raise ValueError(
                f"{ratio_norm_update} uses C directly without dimension scaling"
            )
        if min_z_support != 0 or group_sparse_mode != "none":
            raise ValueError(
                f"{ratio_norm_update} requires unmodified z subproblems "
                "(min_z_support=0 and group_sparse_mode='none')"
            )
    if ratio_norm_update == "strict_two_copy_weighted" and uniform_penalty:
        raise ValueError(
            "strict_two_copy_weighted requires uniform_penalty=False so score weights are active"
        )
    if ratio_norm_update == "direct_ratio_gradient":
        if not use_ratio_norm or use_admm:
            raise ValueError(
                "direct_ratio_gradient requires use_ratio_norm=True and use_admm=False"
            )
        if uses_scaled_gate_constraint:
            raise ValueError("direct_ratio_gradient currently fixes s(g)=g")
        if ratio_norm_dim_scale != "none":
            raise ValueError("direct_ratio_gradient uses C directly")
    valid_score_update_modes = {
        "epoch",
        "epoch_bounded_normalized",
        "epoch_median_normalized",
        "epoch_permuted",
        "freeze_after_warmup",
        "freeze_run_initial",
    }
    if score_update_mode not in valid_score_update_modes:
        raise ValueError(
            f"Unknown score_update_mode={score_update_mode!r}. "
            f"Expected one of {sorted(valid_score_update_modes)}"
        )
    strict_weighted_diagnostic_score_modes = {
        "epoch_bounded_normalized",
        "epoch_median_normalized",
        "epoch_permuted",
        "freeze_run_initial",
    }
    if (
        score_update_mode in strict_weighted_diagnostic_score_modes
        and ratio_norm_update != "strict_two_copy_weighted"
    ):
        raise ValueError(
            f"score_update_mode={score_update_mode!r} is restricted to "
            "strict_two_copy_weighted diagnostics"
        )
    valid_ratio_norm_dim_scales = {"none", "pqi"}
    if ratio_norm_dim_scale not in valid_ratio_norm_dim_scales:
        raise ValueError(
            f"Unknown ratio_norm_dim_scale={ratio_norm_dim_scale!r}. "
            f"Expected one of {sorted(valid_ratio_norm_dim_scales)}"
        )
    if ratio_norm_dim_scale == "pqi" and not (0.0 < ratio_norm_p <= 1.0 < ratio_norm_q):
        raise ValueError(
            "PQI RatioNorm scaling expects 0 < ratio_norm_p <= 1 < ratio_norm_q"
        )
    valid_lr_schedulers = {"none", "cosine"}
    if lr_scheduler not in valid_lr_schedulers:
        raise ValueError(
            f"Unknown lr_scheduler={lr_scheduler!r}. "
            f"Expected one of {sorted(valid_lr_schedulers)}"
        )
    valid_dynamic_C_modes = {
        "none",
        "z_norm",
        "loss_scale",
        "grad_scale",
        "l1_l2",
        "kkt_grad",
        "sqrt_loss",
        "event_continuation",
        "strict_relative_continuation",
        "lambda_max_const",
        "lambda_max_phase_const",
        "root_n_lambda_max_const",
        "root_n_lambda_max_phase_const",
        "root_n_lambda_max_phase_monotone_after2",
        "score_quantile_lambda_phase_const",
        "score_quantile_lambda_dynamic",
        "parent_k_lambda_const",
        "parent_k_phase_lambda_const",
        "support_band",
        "pre_threshold_band",
    }
    if dynamic_C_mode not in valid_dynamic_C_modes:
        raise ValueError(
            f"Unknown dynamic_C_mode={dynamic_C_mode!r}. "
            f"Expected one of {sorted(valid_dynamic_C_modes)}"
        )
    if dynamic_C_mode == "strict_relative_continuation" and not strict_two_copy_mode:
        raise ValueError(
            "strict_relative_continuation requires a strict two-copy ratio mode"
        )
    if dynamic_C_update_interval <= 0:
        raise ValueError("dynamic_C_update_interval must be positive")
    if not (0.0 <= dynamic_C_ema < 1.0):
        raise ValueError("dynamic_C_ema must be in [0, 1)")
    if dynamic_C_min_factor <= 0.0 or dynamic_C_max_factor < dynamic_C_min_factor:
        raise ValueError("dynamic_C factors must satisfy 0 < min_factor <= max_factor")
    if dynamic_C_kappa <= 0.0:
        raise ValueError("dynamic_C_kappa must be positive")
    if not (0.0 < dynamic_C_quantile <= 1.0):
        raise ValueError("dynamic_C_quantile must be in (0, 1]")
    if dynamic_C_continuation_gamma <= 1.0:
        raise ValueError("dynamic_C_continuation_gamma must be > 1")
    if dynamic_C_stability_patience <= 0:
        raise ValueError("dynamic_C_stability_patience must be positive")
    for name, value in {
        "dynamic_C_support_tol": dynamic_C_support_tol,
        "dynamic_C_residual_tol": dynamic_C_residual_tol,
        "dynamic_C_loss_tol": dynamic_C_loss_tol,
    }.items():
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    for name, value in {
        "dynamic_C_support_parent_low": dynamic_C_support_parent_low,
        "dynamic_C_support_parent_high": dynamic_C_support_parent_high,
        "dynamic_C_support_term_low": dynamic_C_support_term_low,
        "dynamic_C_support_term_high": dynamic_C_support_term_high,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if dynamic_C_support_up_mult <= 1.0 or dynamic_C_support_down_mult <= 1.0:
        raise ValueError("dynamic_C support multipliers must be > 1")
    valid_group_sparse_modes = {"none", "parent_subset"}
    if group_sparse_mode not in valid_group_sparse_modes:
        raise ValueError(
            f"Unknown group_sparse_mode={group_sparse_mode!r}. "
            f"Expected one of {sorted(valid_group_sparse_modes)}"
        )

    # Validation split for early stopping
    if use_early_stopping and val_split > 0:
        n_val = int(len(X_train) * val_split)
        idx = np.random.permutation(len(X_train))
        X_val, y_val = X_train[idx[:n_val]], y_train[idx[:n_val]]
        X_train, y_train = X_train[idx[n_val:]], y_train[idx[n_val:]]
    else:
        X_val, y_val = None, None

    dataset = TrainingSet(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if X_val is not None:
        val_dataset = TrainingSet(X_val, y_val)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
    else:
        val_loader = None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if n_classes <= 2:
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
    else:
        criterion = nn.NLLLoss(reduction="mean")

    def _optimizer_groups(
        group_weight_decay: Optional[float] = None,
        parameter_role: str = "all",
    ):
        if parameter_role not in {"all", "theta", "gate"}:
            raise ValueError(
                "parameter_role must be one of {'all', 'theta', 'gate'}"
            )
        active_weight_decay = (
            float(weight_decay)
            if group_weight_decay is None
            else float(group_weight_decay)
        )
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_gate = name == "gate" or name.endswith(".gate")
            if parameter_role == "theta" and is_gate:
                continue
            if parameter_role == "gate" and not is_gate:
                continue
            if is_gate:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        groups = []
        if decay_params:
            groups.append(
                {"params": decay_params, "weight_decay": active_weight_decay}
            )
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        return groups

    # ==================================================================
    # Phase 1: Warm-up optimizer selection (Adam/Adagrad)
    # ==================================================================
    if optimizer_type == "adam":
        opt_warmup = torch.optim.Adam(_optimizer_groups(), lr=lr)
    elif optimizer_type == "adagrad":
        opt_warmup = torch.optim.Adagrad(_optimizer_groups(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    model.train()
    for epoch in range(warmup_epochs):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            opt_warmup.zero_grad()
            y_hat = model(x_batch)
            if n_classes > 2:
                y_hat = torch.log_softmax(y_hat, dim=1)
            else:
                y_hat = y_hat.reshape(len(y_hat))

            try:
                loss = criterion(y_hat, y_batch)
            except RuntimeError:
                loss = criterion(y_hat, y_batch.float())

            loss.backward()
            opt_warmup.step()

    # ==================================================================
    # Phase 2: Linearized ADMM; gate in Adam + RatioNorm z-step
    #
    #   min_{g, theta} L(theta, g) + lambda * R(z)    (R = Ratio Norm)
    #   s.t.  g = z
    #
    #   Augmented Lagrangian (scaled form):
    #     L_rho = L(theta, g) + (rho / 2) * ||g - z + u||^2
    #   Per epoch:
    #     (1) g-step:  Adam on (theta, g) with augmented loss
    #                  gate gets full gradient dynamics
    #     (2) z-step:  formula_z_den uses a denominator-copy soft-threshold
    #                  update; formula_center_den uses the current center norm
    #                  ||g + u||_2 as a detached denominator; *_weighted variants
    #                  additionally use legacy score weights lambda_j = C/s_j;
    #                  legacy current uses a sequential approximate soft-threshold
    #                  plus cubic rescale.
    #     (3) dual:    u <- u + g - z
    #
    #   Score s_j = ||W1[:, j]||_2 gives importance-adaptive thresholds.
    # ==================================================================
    gate_param = model.gate  # shape (m,)
    m = gate_param.shape[0]
    bounded_gate = getattr(model, "bounded_gate", False)
    group_parent_share = None
    group_parent_mask = None
    group_term_has_parent = None
    group_parent_support_value = 0
    diagnostic_parent_terms = None
    diagnostic_parent_share = None
    if group_sparse_powers is not None:
        powers_np_diag = np.asarray(group_sparse_powers)
        if powers_np_diag.ndim != 2 or powers_np_diag.shape[0] != m:
            raise ValueError(
                "group_sparse_powers must have shape "
                f"(n_expanded_features, n_original_features); got {powers_np_diag.shape}, expected first dim {m}"
            )
        diagnostic_parent_terms = [
            torch.as_tensor(
                np.flatnonzero(powers_np_diag[:, parent]),
                device=device,
                dtype=torch.long,
            )
            for parent in range(int(powers_np_diag.shape[1]))
        ]
        powers_t_diag = torch.as_tensor(
            powers_np_diag, device=device, dtype=gate_param.dtype
        )
        degree_diag = powers_t_diag.sum(dim=1)
        degree_safe_diag = torch.clamp(degree_diag, min=1.0)
        diagnostic_parent_share = powers_t_diag / degree_safe_diag.unsqueeze(1)
        diagnostic_parent_share = torch.where(
            degree_diag.unsqueeze(1) > 0,
            diagnostic_parent_share,
            torch.zeros_like(diagnostic_parent_share),
        )
        # Read-only original-parent support diagnostics. This intentionally does
        # not enable group projection when group_sparse_mode == "none".
        group_parent_share = diagnostic_parent_share
    if group_sparse_mode != "none":
        if group_sparse_powers is None:
            raise ValueError("group_sparse_powers is required when group_sparse_mode is enabled")
        powers_np = np.asarray(group_sparse_powers)
        if powers_np.ndim != 2 or powers_np.shape[0] != m:
            raise ValueError(
                "group_sparse_powers must have shape "
                f"(n_expanded_features, n_original_features); got {powers_np.shape}, expected first dim {m}"
            )
        powers_t = torch.as_tensor(powers_np, device=device, dtype=gate_param.dtype)
        degree_t = powers_t.sum(dim=1)
        degree_safe = torch.clamp(degree_t, min=1.0)
        group_parent_share = powers_t / degree_safe.unsqueeze(1)
        group_parent_share = torch.where(
            degree_t.unsqueeze(1) > 0,
            group_parent_share,
            torch.zeros_like(group_parent_share),
        )
        group_parent_mask = powers_t > 0
        group_term_has_parent = degree_t > 0
        n_parent_features = int(powers_np.shape[1])
        group_parent_support_value = int(group_parent_support) if group_parent_support else int(min_z_support)
        group_parent_support_value = max(1, min(group_parent_support_value, n_parent_features))

    def _effective_gate(param: torch.Tensor) -> torch.Tensor:
        """Convert raw parameter to effective gate value (apply sigmoid if bounded)."""
        if hasattr(model, "gate_from_parameter"):
            return model.gate_from_parameter(param)
        return param

    def _parameter_from_effective(gate_value: torch.Tensor) -> torch.Tensor:
        """Convert effective gate value to raw parameter (inverse sigmoid if bounded)."""
        if hasattr(model, "parameter_from_gate"):
            return model.parameter_from_gate(gate_value)
        return gate_value

    def _project_effective_gate(gate_value: torch.Tensor) -> torch.Tensor:
        """Project effective gate to valid range. Only used for effective-space ADMM."""
        if bounded_gate:
            return torch.clamp(gate_value, min=0.0, max=1.0)
        return gate_value

    def _enforce_min_support(
        z_value: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """Keep ADMM z from collapsing below the target support size."""
        if min_z_support <= 0 or z_value.numel() == 0:
            return z_value
        n_keep = min(int(min_z_support), z_value.numel())
        if int((z_value.abs() > 0).sum().item()) >= n_keep:
            return z_value
        topk_idx = reference.abs().topk(n_keep).indices
        z_value = z_value.clone()
        z_value[topk_idx] = reference[topk_idx]
        return z_value

    def _group_parent_scores(term_score: torch.Tensor) -> torch.Tensor:
        if group_parent_share is None:
            return torch.zeros(0, device=term_score.device, dtype=term_score.dtype)
        return torch.matmul(term_score.detach().abs(), group_parent_share)

    def _group_parent_support(z_value: torch.Tensor) -> int:
        if group_parent_share is None:
            return int((z_value.abs() > 1e-12).sum().item())
        return int((_group_parent_scores(z_value) > 1e-12).sum().item())

    def _parent_max_score(term_score: torch.Tensor) -> Optional[torch.Tensor]:
        """Original-feature diagnostic for expanded polynomial gates."""
        if diagnostic_parent_terms is None or term_score.numel() != m:
            return None
        parent_score = torch.zeros(
            len(diagnostic_parent_terms),
            device=term_score.device,
            dtype=term_score.dtype,
        )
        abs_score = term_score.detach().abs()
        for parent, term_idx in enumerate(diagnostic_parent_terms):
            if term_idx.numel() > 0:
                parent_score[parent] = abs_score[term_idx].max()
        return parent_score

    def _support_band_defaults() -> Dict[str, Optional[int]]:
        """Default unsupervised support band for pre-threshold C control."""
        support_k = int(diagnostic_k) if diagnostic_k else 1
        support_k = max(1, support_k)
        parent_n = (
            len(diagnostic_parent_terms)
            if diagnostic_parent_terms is not None
            else 0
        )
        if parent_n > 0:
            default_parent_low = max(
                8 * support_k,
                int(np.ceil(np.sqrt(float(parent_n)))),
            )
            default_parent_low = min(parent_n, default_parent_low)
            default_parent_high = max(default_parent_low + 1, 16 * support_k)
            default_parent_high = min(parent_n, default_parent_high)
        else:
            default_parent_low = None
            default_parent_high = None

        parent_low_for_terms = default_parent_low if default_parent_low else support_k
        parent_high_for_terms = default_parent_high if default_parent_high else 16 * support_k
        default_term_low = max(16 * support_k, 4 * parent_low_for_terms)
        default_term_low = min(m, default_term_low)
        default_term_high = max(default_term_low + 1, 16 * parent_high_for_terms, 32)
        default_term_high = min(m, default_term_high)

        return {
            "support_k": support_k,
            "parent_low": (
                int(dynamic_C_support_parent_low)
                if dynamic_C_support_parent_low > 0
                else default_parent_low
            ),
            "parent_high": (
                int(dynamic_C_support_parent_high)
                if dynamic_C_support_parent_high > 0
                else default_parent_high
            ),
            "term_low": (
                int(dynamic_C_support_term_low)
                if dynamic_C_support_term_low > 0
                else default_term_low
            ),
            "term_high": (
                int(dynamic_C_support_term_high)
                if dynamic_C_support_term_high > 0
                else default_term_high
            ),
        }

    def _critical_lower_bound(values: torch.Tensor, high: Optional[int]) -> Optional[float]:
        """Minimum C that gives at most `high` active coordinates."""
        if high is None or values.numel() == 0:
            return None
        high = max(0, min(int(high), int(values.numel())))
        if high >= int(values.numel()):
            return 0.0
        sorted_values = torch.sort(values.detach().reshape(-1), descending=True).values
        return float(sorted_values[high].item())

    def _critical_upper_bound(values: torch.Tensor, low: Optional[int]) -> Optional[float]:
        """Maximum C that keeps at least `low` active coordinates."""
        if low is None or values.numel() == 0:
            return None
        positive = values.detach().reshape(-1)
        positive = positive[positive > 0]
        if positive.numel() == 0:
            return 0.0
        low = max(1, min(int(low), int(positive.numel())))
        sorted_values = torch.sort(positive, descending=True).values
        return float(sorted_values[low - 1].item())

    def _support_from_critical(values: torch.Tensor, C_value: float) -> int:
        if values.numel() == 0:
            return 0
        return int((values.detach().reshape(-1) > float(C_value)).sum().item())

    def _pre_threshold_band_update(
        C_before: float,
        C_anchor: float,
        critical_C: torch.Tensor,
    ) -> tuple[float, Dict[str, Any]]:
        """Choose C before soft-thresholding from predicted support bands."""
        eps_local = 1e-12
        band = _support_band_defaults()
        term_critical = critical_C.detach().abs()
        parent_critical = _parent_max_score(term_critical)
        parent_available = parent_critical is not None and parent_critical.numel() > 0

        term_low = int(band["term_low"]) if band["term_low"] is not None else None
        term_high = int(band["term_high"]) if band["term_high"] is not None else None
        parent_low = int(band["parent_low"]) if band["parent_low"] is not None else None
        parent_high = int(band["parent_high"]) if band["parent_high"] is not None else None

        lower_bounds = [
            _critical_lower_bound(term_critical, term_high),
        ]
        upper_bounds = [
            _critical_upper_bound(term_critical, term_low),
        ]
        if parent_available:
            lower_bounds.append(_critical_lower_bound(parent_critical, parent_high))
            upper_bounds.append(_critical_upper_bound(parent_critical, parent_low))

        finite_lowers = [v for v in lower_bounds if v is not None]
        finite_uppers = [v for v in upper_bounds if v is not None]
        target_low = max(finite_lowers) if finite_lowers else 0.0
        target_high = min(finite_uppers) if finite_uppers else float("inf")

        min_C = float(dynamic_C_bound_seed) * float(dynamic_C_min_factor)
        max_C = float(dynamic_C_bound_seed) * float(dynamic_C_max_factor)
        target_low = max(target_low, min_C)
        target_high = min(target_high, max_C)

        C_target = float(C_before)
        action = "hold"
        feasible = target_low <= target_high + eps_local
        if feasible:
            if C_before < target_low:
                C_target = target_low
                action = "increase_to_band"
            elif C_before > target_high:
                C_target = target_high
                action = "decrease_to_band"
        else:
            if target_high > 0.0 and C_before > target_high:
                C_target = target_high
                action = "decrease_to_low_support"
            elif C_before < target_low:
                C_target = target_low
                action = "increase_to_high_support"
            else:
                action = "infeasible_hold"

        if C_target > C_before:
            C_target = min(C_target, C_before * float(dynamic_C_support_up_mult))
        elif C_target < C_before:
            C_target = max(C_target, C_before / float(dynamic_C_support_down_mult))
        C_after = min(max(float(C_target), min_C), max_C)
        if abs(C_after - float(C_before)) <= eps_local:
            action = "hold" if action == "hold" else "step_bound_hold"

        row = {
            "pre_threshold_signal": (
                "parent_and_term" if parent_available else "term_only_parent_unavailable"
            ),
            "pre_threshold_feasible": bool(feasible),
            "pre_threshold_target_low": float(target_low),
            "pre_threshold_target_high": (
                None if not np.isfinite(target_high) else float(target_high)
            ),
            "pre_threshold_pred_parent_before": (
                _support_from_critical(parent_critical, C_before)
                if parent_available
                else None
            ),
            "pre_threshold_pred_parent_after": (
                _support_from_critical(parent_critical, C_after)
                if parent_available
                else None
            ),
            "pre_threshold_pred_term_before": _support_from_critical(
                term_critical, C_before
            ),
            "pre_threshold_pred_term_after": _support_from_critical(
                term_critical, C_after
            ),
            "pre_threshold_parent_low": parent_low,
            "pre_threshold_parent_high": parent_high,
            "pre_threshold_term_low": term_low,
            "pre_threshold_term_high": term_high,
            "pre_threshold_min_C": float(min_C),
            "pre_threshold_max_C": float(max_C),
        }
        return float(C_after), {"action": action, **row}

    def _parent_selection_diag(parent_score: torch.Tensor) -> Dict[str, Any]:
        positive = torch.nonzero(parent_score > 0).flatten()
        if positive.numel() == 0:
            selected: List[int] = []
        else:
            n_select = min(
                int(diagnostic_k) if diagnostic_k else int(positive.numel()),
                int(positive.numel()),
            )
            local = parent_score[positive].topk(n_select)
            selected = [int(v) for v in positive[local.indices].detach().cpu().tolist()]
        best_k = None
        if diagnostic_ground_truth is not None and diagnostic_k:
            best_k = float(len(set(selected) & set(diagnostic_ground_truth)) / diagnostic_k)
        gt_ranks: Dict[str, Optional[int]] = {}
        if diagnostic_ground_truth is not None:
            order = torch.argsort(parent_score.detach().abs(), descending=True)
            rank_lookup = {
                int(parent): int(rank + 1)
                for rank, parent in enumerate(order.detach().cpu().tolist())
            }
            gt_ranks = {
                str(int(parent)): rank_lookup.get(int(parent))
                for parent in diagnostic_ground_truth
            }
        return {
            "best_k": best_k,
            "selected": selected,
            "nonzero": int((parent_score > 0).sum().item()),
            "gt_ranks": gt_ranks,
        }

    def _parent_max_selection_diag(term_score: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Original-feature diagnostic using max over expanded terms."""
        parent_score = _parent_max_score(term_score)
        if parent_score is None:
            return None
        return _parent_selection_diag(parent_score)

    def _parent_weight_sum_selection_diag(term_score: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Original-feature diagnostic using degree-shared term aggregation."""
        if diagnostic_parent_share is None or term_score.numel() != m:
            return None
        parent_score = torch.matmul(
            term_score.detach().abs(),
            diagnostic_parent_share,
        )
        return _parent_selection_diag(parent_score)

    def _apply_group_sparse_projection(
        z_value: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project expanded terms onto a top-parent subset.

        Parent ranking is based on the z-step center, not on individual expanded
        terms after shrinkage. A term survives only if all its parents are in
        the selected original-feature set.
        """
        if group_sparse_mode == "none" or group_parent_share is None:
            return z_value, torch.zeros(0, device=z_value.device, dtype=torch.long), _group_parent_scores(z_value)

        parent_score = _group_parent_scores(reference)
        n_keep = min(group_parent_support_value, parent_score.numel())
        parent_idx = parent_score.topk(n_keep).indices
        parent_keep = torch.zeros(parent_score.numel(), device=z_value.device, dtype=torch.bool)
        parent_keep[parent_idx] = True
        outside_parent = group_parent_mask & (~parent_keep).unsqueeze(0)
        term_keep = (outside_parent.sum(dim=1) == 0) & group_term_has_parent
        projected = z_value.clone()
        projected[~term_keep] = 0
        return projected, parent_idx, parent_score

    def _current_gate_for_admm() -> torch.Tensor:
        if bounded_gate:
            return gate_param.data
        return _effective_gate(gate_param.data)

    def _get_admm_feature_scale(*, detach: bool = True) -> torch.Tensor:
        if not uses_scaled_gate_constraint:
            return torch.ones(m, device=device, dtype=gate_param.dtype)
        if joint_explicit_scale_mode:
            if not hasattr(model, "get_admm_feature_scale"):
                raise TypeError(
                    "joint explicit-scale mode requires "
                    "model.get_admm_feature_scale()"
                )
            scale = model.get_admm_feature_scale()
        elif hasattr(model, "first_weight_norm"):
            scale = model.first_weight_norm
        else:
            scale = _get_feature_penalty_scores(model)
        scale = scale.to(device=device, dtype=gate_param.dtype)
        if not joint_explicit_scale_mode:
            scale = torch.clamp(scale, min=1e-8)
        return scale.detach() if detach else scale

    def _gate_to_constraint(gate_value: torch.Tensor, feature_scale: torch.Tensor) -> torch.Tensor:
        return feature_scale * gate_value if uses_scaled_gate_constraint else gate_value

    def _constraint_to_gate(z_value: torch.Tensor, feature_scale: torch.Tensor) -> torch.Tensor:
        if not uses_scaled_gate_constraint:
            return z_value
        safe_scale = torch.where(
            feature_scale.abs() > 0,
            feature_scale,
            torch.ones_like(feature_scale),
        )
        raw_value = z_value / safe_scale
        return torch.where(
            feature_scale.abs() > 0,
            raw_value,
            torch.zeros_like(raw_value),
        )

    def _ratio_norm_coeff(z_value: torch.Tensor) -> float:
        if ratio_norm_dim_scale == "none":
            return 1.0
        d_active = max(1, int(z_value.numel()))
        return float(d_active ** (1.0 / ratio_norm_q - 1.0 / ratio_norm_p))

    def _ratio_norm_C(C_value: float, z_value: torch.Tensor) -> float:
        if not use_ratio_norm:
            return float(C_value)
        return float(C_value) * _ratio_norm_coeff(z_value)

    def _data_gate_grad_mean() -> tuple[torch.Tensor, int]:
        """Mean data-loss gradient with respect to the gate over the full loader."""
        was_training = model.training
        model.eval()
        grad_sum = torch.zeros_like(gate_param, device=device)
        n_seen = 0
        with torch.enable_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_hat = model(x_batch)
                if n_classes > 2:
                    y_hat = torch.log_softmax(y_hat, dim=1)
                else:
                    y_hat = y_hat.reshape(len(y_hat))
                try:
                    data_loss = criterion(y_hat, y_batch)
                except RuntimeError:
                    data_loss = criterion(y_hat, y_batch.float())
                data_gate_grad = torch.autograd.grad(
                    data_loss,
                    gate_param,
                    retain_graph=False,
                    allow_unused=True,
                )[0]
                if data_gate_grad is not None:
                    batch_n = int(x_batch.shape[0])
                    grad_sum += data_gate_grad.detach() * batch_n
                    n_seen += batch_n
        if was_training:
            model.train()
        if n_seen <= 0:
            return grad_sum, 0
        return grad_sum / float(n_seen), int(n_seen)

    def _data_gate_lambda_max() -> tuple[float, int]:
        """Maximum absolute gradient of mean data loss with respect to gate."""
        grad_mean, n_seen = _data_gate_grad_mean()
        if n_seen <= 0:
            return 0.0, 0
        return float(grad_mean.detach().abs().max().item()), int(n_seen)

    feature_scale = _get_admm_feature_scale()

    if joint_explicit_scale_mode and initial_admm_state:
        state_constraint_mode = initial_admm_state.get("admm_constraint_mode")
        if state_constraint_mode not in {None, effective_admm_constraint_mode}:
            raise ValueError(
                "initial_admm_state constraint mode does not match the joint "
                f"explicit-scale model: {state_constraint_mode!r}"
            )
        state_feature_scale = initial_admm_state.get("feature_scale")
        if torch.is_tensor(state_feature_scale):
            state_feature_scale = state_feature_scale.to(
                device=device,
                dtype=feature_scale.dtype,
            )
            scale_shape_matches = tuple(state_feature_scale.shape) == tuple(
                feature_scale.shape
            )
            scale_values_match = scale_shape_matches and torch.allclose(
                state_feature_scale, feature_scale, rtol=1e-5, atol=1e-7
            )
            if not scale_values_match:
                raise ValueError(
                    "initial_admm_state feature_scale does not match the current "
                    "explicit-scale model; restore the matching model checkpoint "
                    "or reset z/u"
                )

    # ADMM buffers are in raw space only for the frozen legacy bounded-gate
    # branch. Current A/B variants keep zk and uk in effective-g space. The
    # detached historical weighting branch continues to use A*g.
    gate_for_admm = _current_gate_for_admm()
    if bounded_gate:
        # Raw space ADMM: gate_for_admm is the raw gate value.
        zk = _gate_to_constraint(gate_for_admm, feature_scale).clone()
    else:
        # Effective space ADMM: gate_for_admm is the effective gate value.
        zk = _gate_to_constraint(gate_for_admm, feature_scale).clone()
    z_den = zk.clone()
    yk = zk.clone()
    uk = torch.zeros(m, device=device)  # scaled dual for s(g) = z
    wk = torch.zeros(m, device=device)  # scaled dual for s(g) = y
    rho = rho_init
    if initial_admm_state:
        state_z = initial_admm_state.get("zk")
        state_z_den = initial_admm_state.get("z_den")
        state_y = initial_admm_state.get("yk")
        state_u = initial_admm_state.get("uk")
        state_w = initial_admm_state.get("wk")
        if torch.is_tensor(state_z) and tuple(state_z.shape) == tuple(zk.shape):
            zk = state_z.detach().to(device=device, dtype=zk.dtype).clone()
        if torch.is_tensor(state_z_den) and tuple(state_z_den.shape) == tuple(z_den.shape):
            z_den = state_z_den.detach().to(device=device, dtype=z_den.dtype).clone()
        if torch.is_tensor(state_y) and tuple(state_y.shape) == tuple(yk.shape):
            yk = state_y.detach().to(device=device, dtype=yk.dtype).clone()
        if torch.is_tensor(state_u) and tuple(state_u.shape) == tuple(uk.shape):
            uk = state_u.detach().to(device=device, dtype=uk.dtype).clone()
        if torch.is_tensor(state_w) and tuple(state_w.shape) == tuple(wk.shape):
            wk = state_w.detach().to(device=device, dtype=wk.dtype).clone()

    auto_C_info = None
    if auto_C_gamma is not None:
        if C_schedule_per_epoch is not None:
            raise ValueError("auto_C_gamma cannot be used with C_schedule_per_epoch")
        with torch.no_grad():
            center = _gate_to_constraint(_current_gate_for_admm(), feature_scale) + uk
            support = int(auto_C_support) if auto_C_support else int(min_z_support)
            support = max(1, min(support, center.numel()))
            abs_center = center.detach().abs()
            top_values = torch.topk(abs_center, support).values
            top_cutoff = top_values[-1]
            z_den_norm = torch.clamp(torch.norm(z_den, p=2), min=1e-10)
            C_tensor = float(auto_C_gamma) * float(rho) * z_den_norm * top_cutoff
            C = float(torch.clamp(C_tensor, min=0.0).item())
            auto_C_info = {
                "gamma": float(auto_C_gamma),
                "support": support,
                "top_cutoff": float(top_cutoff.item()),
                "z_den_norm": float(z_den_norm.item()),
                "center_norm": float(torch.norm(center, p=2).item()),
                "C": C,
            }

    # The strict two-copy branch has no external penalty score. Legacy branches
    # retain their existing score behavior for result reproducibility.
    with torch.no_grad():
        if ratio_norm_update in {"canonical_signed", "strict_two_copy"}:
            score = torch.ones_like(gate_param)
        else:
            score = _get_transformed_feature_penalty_scores(
                model,
                score_update_mode,
            )
        frozen_score = score.clone()

    # Adam on ALL parameters including gate (gate gets gradient dynamics)
    opt_all = torch.optim.Adam(_optimizer_groups(), lr=lr)

    prune_epochs = epochs - warmup_epochs
    if C_schedule_per_epoch is not None and len(C_schedule_per_epoch) == 0:
        raise ValueError("C_schedule_per_epoch must be non-empty when provided")
    if adaptive_C and C_schedule_per_epoch is not None:
        raise ValueError("adaptive_C cannot be combined with C_schedule_per_epoch")
    if dynamic_C_mode != "none":
        if adaptive_C:
            raise ValueError("dynamic_C_mode cannot be combined with adaptive_C")
        if C_schedule_per_epoch is not None:
            raise ValueError("dynamic_C_mode cannot be combined with C_schedule_per_epoch")
    if adaptive_C_update_interval <= 0:
        raise ValueError("adaptive_C_update_interval must be positive")
    if adaptive_C_mult <= 1.0:
        raise ValueError("adaptive_C_mult must be > 1")
    adaptive_C_target_value = int(adaptive_C_target) if adaptive_C_target else int(min_z_support)
    adaptive_C_target_value = max(0, min(adaptive_C_target_value, m))
    if group_sparse_mode != "none" and adaptive_C_target <= 0:
        adaptive_C_target_value = group_parent_support_value
    C_dynamic = float(C)
    C_epoch_history: list[float] = []
    last_C_epoch = float(C)
    adaptive_C_history = []
    dynamic_C_history = []
    dynamic_C_ref_z_norm = None
    dynamic_C_ref_l1_l2 = None
    dynamic_C_ref_loss_scale = None
    dynamic_C_ref_grad_scale = None
    dynamic_C_event_prev_metrics = None
    dynamic_C_event_stable_count = 0
    dynamic_C_bound_seed = (
        float(dynamic_C_cap_seed) if dynamic_C_cap_seed is not None else float(C)
    )
    last_group_parent_idx = torch.zeros(0, device=device, dtype=torch.long)
    last_parent_z_support = 0
    last_term_z_support = int((zk.abs() > 1e-12).sum().item())

    const_lambda_modes = {
        "lambda_max_const",
        "lambda_max_phase_const",
        "root_n_lambda_max_const",
        "root_n_lambda_max_phase_const",
        "root_n_lambda_max_phase_monotone_after2",
        "score_quantile_lambda_phase_const",
        "parent_k_lambda_const",
        "parent_k_phase_lambda_const",
    }
    if dynamic_C_mode in const_lambda_modes:
        eps = 1e-12
        grad_mean, lambda_n = _data_gate_grad_mean()
        grad_abs_mean = grad_mean.detach().abs()
        lambda_source = "expanded_max"
        lambda_loss_convention = "sum_loss"
        parent_k_used = None
        parent_nonzero = None
        source_c = None
        source_gamma = None
        source_quantile = None
        source_p_eff = None
        lambda_eff_mean = float(grad_abs_mean.max().item()) if lambda_n > 0 else 0.0
        if dynamic_C_mode in {
            "root_n_lambda_max_const",
            "root_n_lambda_max_phase_const",
            "root_n_lambda_max_phase_monotone_after2",
        }:
            lambda_eff = lambda_eff_mean * float(np.sqrt(max(int(lambda_n), 1)))
            lambda_loss_convention = (
                "phase_root_n_mean_loss"
                if dynamic_C_mode
                in {"root_n_lambda_max_phase_const", "root_n_lambda_max_phase_monotone_after2"}
                else "root_n_mean_loss"
            )
        elif dynamic_C_mode == "score_quantile_lambda_phase_const":
            source_c = 1.1
            source_gamma = 0.1 / float(np.log(max(int(lambda_n), 3)))
            source_p_eff = max(int(grad_abs_mean.numel()), 2)
            tail_prob = 1.0 - float(source_gamma) / (2.0 * float(source_p_eff))
            tail_prob = min(max(tail_prob, 1e-12), 1.0 - 1e-12)
            source_quantile = NormalDist().inv_cdf(tail_prob)
            lambda_eff = (
                float(source_c)
                * float(source_quantile)
                * float(np.sqrt(max(int(lambda_n), 1)))
                * float(lambda_eff_mean)
            )
            lambda_source = "expanded_max_score_quantile"
            lambda_loss_convention = (
                "phase_root_n_mean_loss_times_source_quantile;"
                "c=1.1,gamma=0.1/log(n),normal_tail=1-gamma/(2p)"
            )
        elif dynamic_C_mode in {"parent_k_lambda_const", "parent_k_phase_lambda_const"}:
            parent_score = _parent_max_score(grad_abs_mean)
            if parent_score is None:
                raise ValueError(
                    f"{dynamic_C_mode} requires group_sparse_powers so expanded terms can be "
                    "mapped back to original parent features"
                )
            positive = parent_score[parent_score > 0]
            if positive.numel() == 0:
                lambda_eff_mean = 0.0
                parent_k_used = 0
                parent_nonzero = 0
            else:
                parent_k_used = min(
                    int(diagnostic_k) if diagnostic_k else int(positive.numel()),
                    int(positive.numel()),
                )
                lambda_eff_mean = float(torch.topk(positive, parent_k_used).values[-1].item())
                parent_nonzero = int(positive.numel())
            lambda_eff = lambda_eff_mean * max(int(lambda_n), 1)
            lambda_source = "parent_max_kth"
            lambda_loss_convention = (
                "phase_parent_k_sum_loss"
                if dynamic_C_mode == "parent_k_phase_lambda_const"
                else "parent_k_sum_loss"
            )
        else:
            lambda_eff = lambda_eff_mean * max(int(lambda_n), 1)
            if dynamic_C_mode == "lambda_max_phase_const":
                lambda_loss_convention = "phase_sum_loss"
        z_norm = float(torch.norm(zk.detach().abs(), p=2).item())
        target_C_unclipped = max(float(lambda_eff) * max(z_norm, eps), eps)
        C_dynamic = target_C_unclipped
        dynamic_C_floor_value = None
        floor_applied = False
        if dynamic_C_floor is not None:
            dynamic_C_floor_value = max(float(dynamic_C_floor), eps)
            if C_dynamic < dynamic_C_floor_value:
                C_dynamic = dynamic_C_floor_value
                floor_applied = True
        history_row = {
            "epoch": -1,
            "mode": dynamic_C_mode,
            "C_epoch_used": float(C_dynamic),
            "C_before": float(C),
            "C_after": float(C_dynamic),
            "phase_C_anchor": float(C),
            "C_seed": float(C),
            "C_bound_seed": float(dynamic_C_bound_seed),
            "lambda_eff": float(lambda_eff),
            "lambda_eff_mean_loss": float(lambda_eff_mean),
            "lambda_source": lambda_source,
            "lambda_max_n": int(lambda_n),
            "lambda_max_loss_convention": lambda_loss_convention,
            "target_factor": float(C_dynamic) / max(float(C), eps),
            "target_C_unclipped": float(target_C_unclipped),
            "target_C_clipped": float(C_dynamic),
            "dynamic_C_floor": dynamic_C_floor_value,
            "dynamic_C_floor_applied": bool(floor_applied),
            "z_l1": float(torch.norm(zk.detach().abs(), p=1).item()),
            "z_norm": float(z_norm),
            "z_l1_l2": float(torch.norm(zk.detach().abs(), p=1).item()) / max(z_norm, eps),
            "rho": float(rho),
            "action": "set_const_floor" if floor_applied else "set_const",
        }
        if parent_k_used is not None:
            history_row["parent_k_used"] = int(parent_k_used)
            history_row["parent_nonzero"] = int(parent_nonzero or 0)
        if source_c is not None:
            history_row["source_c"] = float(source_c)
            history_row["source_gamma"] = float(source_gamma)
            history_row["source_p_eff"] = int(source_p_eff)
            history_row["source_normal_quantile"] = float(source_quantile)
        dynamic_C_history.append(history_row)

    cosine_eta_min_value = (
        float(cosine_eta_min)
        if cosine_eta_min is not None
        else float(lr) * 0.1
    )

    def _scheduled_lr(epoch_idx: int) -> float:
        if lr_scheduler == "none" or prune_epochs <= 1:
            return float(lr)
        decay_end = max(
            int(round(max(prune_epochs - 1, 1) * float(cosine_decay_fraction))),
            1,
        )
        progress = min(epoch_idx / decay_end, 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return cosine_eta_min_value + (float(lr) - cosine_eta_min_value) * cosine

    def _set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = float(value)

    # Direct objective control: minibatch Adam on L + C_t * ||g||_1 / ||g||_2.
    # This is intentionally separate from the legacy approximate proximal path.
    if not use_admm and ratio_norm_update == "direct_ratio_gradient":
        opt_direct = torch.optim.Adam(_optimizer_groups(), lr=lr)
        direct_C_history: list[float] = []
        direct_ratio_history: list[float] = []
        for epoch in range(prune_epochs):
            C_epoch = (
                C_schedule_per_epoch[min(epoch, len(C_schedule_per_epoch) - 1)]
                if C_schedule_per_epoch is not None
                else C_dynamic
            )
            direct_C_history.append(float(C_epoch))

            model.train()
            last_ratio = float("nan")
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                opt_direct.zero_grad()
                y_hat = model(x_batch)
                if n_classes > 2:
                    y_hat = torch.log_softmax(y_hat, dim=1)
                else:
                    y_hat = y_hat.reshape(len(y_hat))
                try:
                    data_loss = criterion(y_hat, y_batch)
                except RuntimeError:
                    data_loss = criterion(y_hat, y_batch.float())

                gate_for_ratio = (
                    gate_param if bounded_gate else _effective_gate(gate_param)
                )
                denominator = torch.linalg.vector_norm(gate_for_ratio, ord=2)
                denominator = denominator.clamp_min(1e-10)
                ratio_value = torch.linalg.vector_norm(
                    gate_for_ratio, ord=1
                ) / denominator
                total_loss = data_loss + float(C_epoch) * ratio_value
                total_loss.backward()
                opt_direct.step()
                last_ratio = float(ratio_value.detach().item())
            direct_ratio_history.append(last_ratio)

        model.eval()
        model.admm_state = {
            "optimization_mode": "direct_ratio_gradient",
            "C_used": float(direct_C_history[-1]) if direct_C_history else float(C),
            "C_schedule_per_epoch": direct_C_history,
            "direct_ratio_history": direct_ratio_history,
            "uses_external_penalty_score": False,
            "ratio_norm_update": "direct_ratio_gradient",
            "score_update_mode": "disabled",
        }
        return

    #  Ablation: legacy proximal gradient (no ADMM)
    if not use_admm:
        opt_prox = torch.optim.Adam(_optimizer_groups(), lr=lr)
        for epoch in range(prune_epochs):
            C_epoch = (
                C_schedule_per_epoch[min(epoch, len(C_schedule_per_epoch) - 1)]
                if C_schedule_per_epoch is not None
                else C_dynamic
            )
            with torch.no_grad():
                if score_update_mode == "epoch":
                    score = _get_feature_penalty_scores(model)
                else:
                    score = frozen_score

            # Forward pass (Adam step)
            model.train()
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                opt_prox.zero_grad()
                y_hat = model(x_batch)
                if n_classes > 2:
                    y_hat = torch.log_softmax(y_hat, dim=1)
                else:
                    y_hat = y_hat.reshape(len(y_hat))
                try:
                    loss = criterion(y_hat, y_batch)
                except RuntimeError:
                    loss = criterion(y_hat, y_batch.float())
                loss.backward()
                opt_prox.step()

            # Proximal step: apply proximal operator directly to gate
            with torch.no_grad():
                # For bounded_gate: use raw gate; for unbounded: use effective gate
                if bounded_gate:
                    g = gate_param.data  # raw space
                else:
                    g = _effective_gate(gate_param.data)
                if uniform_penalty:
                    C_ratio = _ratio_norm_C(C_epoch, g)
                    lam = torch.full_like(score, C_ratio)
                else:
                    C_ratio = _ratio_norm_C(C_epoch, g)
                    lam = C_ratio / score  # per-feature lambda
                lam = torch.clamp(
                    lam, min=1e-8
                )  # FIX: Remove max clamp to allow larger thresholds
                alpha = lam * lr  # proximal step size
                g_shrunk = soft_thresholding(g, alpha)
                if use_ratio_norm and torch.norm(g_shrunk, p=1) > 1e-8:
                    v_l2 = safe_norm(g_shrunk)
                    mu = C_ratio * torch.norm(g_shrunk, p=1) / N
                    D_k = (mu * score * score) / (1.0 * torch.clamp(v_l2**3, min=1e-10))
                    tau_k = solve_cubic_ratio_norm(D_k)
                    g_shrunk = tau_k * g_shrunk
                # For bounded_gate (raw space): no projection needed
                # For unbounded_gate (effective space): project to valid range
                if not bounded_gate:
                    g_shrunk = _project_effective_gate(g_shrunk)
                g_shrunk = _enforce_min_support(g_shrunk, g)
                if bounded_gate:
                    gate_param.data.copy_(g_shrunk)
                else:
                    gate_param.data.copy_(_parameter_from_effective(g_shrunk))
        model.eval()
        return

    #  3-variable ADMM path (paper's formulation)
    if three_variable and use_ratio_norm:
        # For bounded_gate: use raw space; for unbounded: use effective space
        if bounded_gate:
            yk = gate_param.data.clone()
            zk_3 = gate_param.data.clone()
        else:
            yk = _effective_gate(gate_param.data).clone()
            zk_3 = _effective_gate(gate_param.data).clone()
        vk = torch.zeros(m, device=device)  # dual for g=y
        wk = torch.zeros(m, device=device)  # dual for g=z
        convergence_log = []

        for epoch in range(prune_epochs):
            C_epoch = (
                C_schedule_per_epoch[min(epoch, len(C_schedule_per_epoch) - 1)]
                if C_schedule_per_epoch is not None
                else C_dynamic
            )
            with torch.no_grad():
                if score_update_mode == "epoch":
                    score = _get_feature_penalty_scores(model)
                else:
                    score = frozen_score

            # (1) g-step: Adam on augmented loss with TWO penalty terms
            model.train()
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                opt_all.zero_grad()
                y_hat = model(x_batch)
                if n_classes > 2:
                    y_hat = torch.log_softmax(y_hat, dim=1)
                else:
                    y_hat = y_hat.reshape(len(y_hat))
                try:
                    data_loss = criterion(y_hat, y_batch)
                except RuntimeError:
                    data_loss = criterion(y_hat, y_batch.float())

                # For bounded_gate: use raw gate; for unbounded: use effective gate
                if bounded_gate:
                    gate_value_for_penalty = gate_param
                else:
                    gate_value_for_penalty = _effective_gate(gate_param)
                penalty = (rho / 2.0) * (
                    torch.sum((gate_value_for_penalty - yk.detach() + vk.detach()) ** 2)
                    + torch.sum(
                        (gate_value_for_penalty - zk_3.detach() + wk.detach()) ** 2
                    )
                )
                (data_loss + penalty).backward()
                opt_all.step()

            # (2) y-step: cubic update in effective gate space
            with torch.no_grad():
                if uniform_penalty:
                    lam = torch.full_like(score, C_epoch)
                else:
                    lam = C_epoch / score
                lam = torch.clamp(
                    lam, min=1e-6
                )  # FIX: Remove max clamp to allow larger thresholds

                c_t = torch.sum(lam * torch.abs(zk_3))
                if bounded_gate:
                    d_t = gate_param.data + vk / rho
                else:
                    d_t = _effective_gate(gate_param.data) + vk / rho
                d_norm = safe_norm(d_t)

                if d_norm > 1e-8 and c_t > 1e-8:
                    D_scalar = c_t / (N * rho * torch.clamp(d_norm**3, min=1e-10))
                    tau_y = solve_cubic_paper(D_scalar.unsqueeze(0)).squeeze(0)
                    yk = tau_y * d_t
                else:
                    yk = d_t.clone()
                # For bounded_gate (raw space): no projection needed
                if not bounded_gate:
                    yk = _project_effective_gate(yk)

            # (3) z-step: soft-threshold in effective gate space
            # FIX: Use standard soft-threshold threshold = lam/rho (not diluted by N)
            # Previous formula: kappa = lam / (N * rho * y_norm) was WRONG
            # Standard ADMM: z = soft_threshold(v, lambda/rho)
            with torch.no_grad():
                zk_3_old = zk_3.clone()
                if bounded_gate:
                    a = gate_param.data + wk / rho
                else:
                    a = _effective_gate(gate_param.data) + wk / rho
                # CORRECTED: threshold = lam/rho (per-feature adaptive)
                kappa = lam / rho
                zk_3 = soft_thresholding(a, kappa)
                # For bounded_gate (raw space): no projection needed
                if not bounded_gate:
                    zk_3 = _project_effective_gate(zk_3)
                zk_3 = _enforce_min_support(zk_3, a)

                if write_z_to_gate:
                    if bounded_gate:
                        gate_param.data.copy_(zk_3)
                    else:
                        gate_param.data.copy_(_parameter_from_effective(zk_3))

            # (4) dual updates
            with torch.no_grad():
                if bounded_gate:
                    gate_value = gate_param.data
                else:
                    gate_value = _effective_gate(gate_param.data)
                vk = vk + rho * (gate_value - yk)
                wk = wk + rho * (gate_value - zk_3)

            with torch.no_grad():
                if bounded_gate:
                    gate_value = gate_param.data
                else:
                    gate_value = _effective_gate(gate_param.data)
                primal_resid = torch.norm(gate_value - zk_3).item()
                dual_resid = rho * torch.norm(zk_3 - zk_3_old).item()
                convergence_log.append((epoch, primal_resid, dual_resid))

            if adaptive_rho and epoch > 0 and epoch % rho_update_interval == 0:
                with torch.no_grad():
                    gate_value = _effective_gate(gate_param.data)
                    r_norm = max(torch.norm(gate_value - zk_3).item(), 1e-12)
                    s_norm = max(rho * torch.norm(zk_3 - zk_3_old).item(), 1e-12)
                    rho_old = rho
                    if r_norm > 10.0 * s_norm:
                        rho = min(rho * 2.0, 1e4)
                    elif s_norm > 10.0 * r_norm:
                        rho = max(rho / 2.0, 50.0)
                    if rho != rho_old:
                        vk = vk * (rho_old / rho)
                        wk = wk * (rho_old / rho)

        model.eval()
        model.convergence_log = convergence_log
        return

    convergence_log = []  # stores combined (epoch, primal_resid, dual_resid)
    relative_convergence_log: list[dict[str, Any]] = []
    canonical_prox_log: list[dict[str, Any]] = []
    strict_two_copy_log: list[dict[str, Any]] = []
    strict_kkt_history: list[dict[str, Any]] = []
    strict_kkt_consecutive = 0
    strict_kkt_previous_support: Optional[tuple[int, ...]] = None
    strict_kkt_converged = False
    strict_kkt_stop_epoch: Optional[int] = None
    completed_outer_iterations = 0
    strict_gate_inner_history: list[dict[str, Any]] = []
    strict_gate_inner_total_steps = 0
    strict_gate_inner_passes = 0
    strict_gate_inner_cap_hits = 0

    # Early stopping variables
    best_val_loss = float("inf")
    best_state_dict = None
    no_improve_count = 0

    # ADMM optimizer selection. The historical mode uses one optimizer over
    # (theta, g). The full-fold gate diagnostic keeps independent optimizer
    # state because theta and g are solved as separate neural blocks.
    split_neural_block_mode = strict_neural_block_mode in fullfold_neural_block_modes
    gate_inner_solve_mode = (
        strict_neural_block_mode == "theta_minibatch_gate_fullfold_inner"
    )
    strict_gate_inner_max_steps = len(loader) if gate_inner_solve_mode else 1
    if optimizer_type == "adam":
        optimizer_class = torch.optim.Adam
    elif optimizer_type == "adagrad":
        optimizer_class = torch.optim.Adagrad
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    opt_all = None
    opt_theta = None
    opt_gate = None
    if split_neural_block_mode:
        opt_theta = optimizer_class(
            _optimizer_groups(parameter_role="theta"), lr=lr
        )
        opt_gate = optimizer_class(
            _optimizer_groups(parameter_role="gate"), lr=lr
        )
        if initial_admm_state:
            for optimizer, state_key in (
                (opt_theta, "theta_optimizer_state_dict"),
                (opt_gate, "gate_optimizer_state_dict"),
            ):
                state_dict = initial_admm_state.get(state_key)
                if state_dict is None:
                    continue
                try:
                    optimizer.load_state_dict(state_dict)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                        if group.get("weight_decay", 0.0) != 0.0:
                            group["weight_decay"] = float(weight_decay)
                except (ValueError, RuntimeError, KeyError):
                    pass
        full_train_x = dataset.X.to(device)
        full_train_y = dataset.Y.to(device)
    else:
        opt_all = optimizer_class(_optimizer_groups(), lr=lr)
        if (
            initial_admm_state
            and initial_admm_state.get("optimizer_state_dict") is not None
        ):
            try:
                opt_all.load_state_dict(initial_admm_state["optimizer_state_dict"])
                for group in opt_all.param_groups:
                    group["lr"] = lr
                    if group.get("weight_decay", 0.0) != 0.0:
                        group["weight_decay"] = float(weight_decay)
            except (ValueError, RuntimeError, KeyError):
                pass

    for epoch in range(prune_epochs):
        completed_outer_iterations = epoch + 1
        current_lr = _scheduled_lr(epoch)
        if split_neural_block_mode:
            _set_optimizer_lr(opt_theta, current_lr)
            _set_optimizer_lr(opt_gate, current_lr)
        else:
            _set_optimizer_lr(opt_all, current_lr)
        if rho_follow_lr:
            rho_old = rho
            rho = 1.0 / max(float(current_lr), 1e-12)
            if rho != rho_old:
                uk = uk * (rho_old / rho)
                if strict_two_copy_mode:
                    wk = wk * (rho_old / rho)

        C_epoch = (
            C_schedule_per_epoch[min(epoch, len(C_schedule_per_epoch) - 1)]
            if C_schedule_per_epoch is not None
            else C_dynamic
        )
        last_C_epoch = float(C_epoch)
        C_epoch_history.append(last_C_epoch)
        epoch_data_loss_sum = 0.0
        epoch_data_loss_n = 0
        epoch_grad_scale_sum = 0.0
        epoch_grad_scale_n = 0
        # Refresh score only in adaptive-score mode. In freeze_after_warmup mode
        # the weighted penalty is fixed for this ADMM phase.
        with torch.no_grad():
            if ratio_norm_update == "strict_two_copy":
                score = torch.ones_like(gate_param)
            elif score_update_mode in {
                "epoch",
                "epoch_bounded_normalized",
                "epoch_median_normalized",
                "epoch_permuted",
            }:
                score = _get_transformed_feature_penalty_scores(
                    model,
                    score_update_mode,
                )
            else:
                score = frozen_score
            feature_scale = _get_admm_feature_scale()

        # (1) Neural block update with z/y/u/w frozen. The default preserves
        # the historical joint minibatch update. The diagnostic split mode
        # gives theta the same minibatch data exposure, then freezes theta and
        # updates the stored gate parameter once from the deterministic
        # full-fold mean objective (raw s for A, direct g for B/legacy).
        if split_neural_block_mode:
            for _g_inner_epoch in range(g_inner_epochs):
                model.train()
                gate_requires_grad = gate_param.requires_grad
                gate_param.requires_grad_(False)
                try:
                    for x_batch, y_batch in loader:
                        x_batch = x_batch.to(device)
                        y_batch = y_batch.to(device)

                        opt_theta.zero_grad()
                        y_hat = model(x_batch)
                        if n_classes > 2:
                            y_hat = torch.log_softmax(y_hat, dim=1)
                        else:
                            y_hat = y_hat.reshape(len(y_hat))
                        try:
                            data_loss = criterion(y_hat, y_batch)
                        except RuntimeError:
                            data_loss = criterion(y_hat, y_batch.float())
                        batch_n = int(x_batch.shape[0])
                        epoch_data_loss_sum += (
                            float(data_loss.detach().item()) * batch_n
                        )
                        epoch_data_loss_n += batch_n
                        data_loss.backward()
                        opt_theta.step()
                finally:
                    gate_param.requires_grad_(gate_requires_grad)

                if gate_inner_solve_mode:
                    inner_steps = 0
                    inner_passed = False
                    initial_normalized_violation = None
                    final_normalized_violation = float("inf")
                    final_gradient_l2 = float("inf")
                    final_threshold = 0.0
                    final_scale = 0.0
                    final_data_gradient_l2 = 0.0
                    final_z_penalty_gradient_l2 = 0.0
                    final_y_penalty_gradient_l2 = 0.0

                    # There are max_steps optimizer updates and max_steps + 1
                    # gradient evaluations so the post-final-step residual is
                    # recorded rather than inferred from a stale gradient.
                    for _gate_inner_evaluation in range(
                        strict_gate_inner_max_steps + 1
                    ):
                        model.eval()
                        opt_gate.zero_grad()
                        y_hat_full = model(full_train_x)
                        if n_classes > 2:
                            y_hat_full = torch.log_softmax(y_hat_full, dim=1)
                        else:
                            y_hat_full = y_hat_full.reshape(len(y_hat_full))
                        try:
                            full_data_loss = criterion(y_hat_full, full_train_y)
                        except RuntimeError:
                            full_data_loss = criterion(
                                y_hat_full, full_train_y.float()
                            )

                        gate_value_for_penalty = (
                            gate_param
                            if bounded_gate
                            else _effective_gate(gate_param)
                        )
                        constraint_gate = _gate_to_constraint(
                            gate_value_for_penalty, feature_scale
                        )
                        z_penalty = (rho / 2.0) * torch.sum(
                            (constraint_gate - zk.detach() + uk.detach()) ** 2
                        )
                        y_penalty = (rho / 2.0) * torch.sum(
                            (constraint_gate - yk.detach() + wk.detach()) ** 2
                        )

                        data_gate_grad = torch.autograd.grad(
                            full_data_loss,
                            gate_param,
                            retain_graph=True,
                        )[0]
                        z_penalty_grad = torch.autograd.grad(
                            z_penalty,
                            gate_param,
                            retain_graph=True,
                        )[0]
                        y_penalty_grad = torch.autograd.grad(
                            y_penalty,
                            gate_param,
                        )[0]
                        gate_grad = (
                            data_gate_grad + z_penalty_grad + y_penalty_grad
                        )

                        final_gradient_l2 = float(
                            torch.linalg.vector_norm(gate_grad.detach()).item()
                        )
                        final_data_gradient_l2 = float(
                            torch.linalg.vector_norm(
                                data_gate_grad.detach()
                            ).item()
                        )
                        final_z_penalty_gradient_l2 = float(
                            torch.linalg.vector_norm(
                                z_penalty_grad.detach()
                            ).item()
                        )
                        final_y_penalty_gradient_l2 = float(
                            torch.linalg.vector_norm(
                                y_penalty_grad.detach()
                            ).item()
                        )
                        final_scale = max(
                            final_data_gradient_l2,
                            final_z_penalty_gradient_l2,
                            final_y_penalty_gradient_l2,
                        )
                        final_threshold = (
                            float(strict_kkt_absolute_tolerance)
                            * float(np.sqrt(max(int(gate_param.numel()), 1)))
                            + float(strict_kkt_tolerance) * final_scale
                        )
                        final_normalized_violation = final_gradient_l2 / max(
                            final_threshold,
                            1e-30,
                        )
                        if initial_normalized_violation is None:
                            initial_normalized_violation = (
                                final_normalized_violation
                            )

                        if final_gradient_l2 <= final_threshold:
                            inner_passed = True
                            break
                        if inner_steps >= strict_gate_inner_max_steps:
                            break

                        gate_param.grad = gate_grad.detach()
                        opt_gate.step()
                        inner_steps += 1

                    strict_gate_inner_total_steps += int(inner_steps)
                    strict_gate_inner_passes += int(inner_passed)
                    strict_gate_inner_cap_hits += int(
                        not inner_passed
                        and inner_steps >= strict_gate_inner_max_steps
                    )
                    strict_gate_inner_history.append(
                        {
                            "epoch": int(epoch),
                            "max_steps": int(strict_gate_inner_max_steps),
                            "steps": int(inner_steps),
                            "passed": bool(inner_passed),
                            "termination": (
                                "mixed_augmented_gradient"
                                if inner_passed
                                else "matched_step_cap"
                            ),
                            "initial_normalized_violation": float(
                                initial_normalized_violation
                            ),
                            "final_normalized_violation": float(
                                final_normalized_violation
                            ),
                            "final_gradient_l2": float(final_gradient_l2),
                            "final_threshold": float(final_threshold),
                            "final_scale": float(final_scale),
                            "final_data_gradient_l2": float(
                                final_data_gradient_l2
                            ),
                            "final_z_penalty_gradient_l2": float(
                                final_z_penalty_gradient_l2
                            ),
                            "final_y_penalty_gradient_l2": float(
                                final_y_penalty_gradient_l2
                            ),
                        }
                    )
                else:
                    model.eval()
                    opt_gate.zero_grad()
                    y_hat_full = model(full_train_x)
                    if n_classes > 2:
                        y_hat_full = torch.log_softmax(y_hat_full, dim=1)
                    else:
                        y_hat_full = y_hat_full.reshape(len(y_hat_full))
                    try:
                        full_data_loss = criterion(y_hat_full, full_train_y)
                    except RuntimeError:
                        full_data_loss = criterion(
                            y_hat_full, full_train_y.float()
                        )

                    gate_value_for_penalty = (
                        gate_param if bounded_gate else _effective_gate(gate_param)
                    )
                    constraint_feature_scale = (
                        _get_admm_feature_scale(detach=False)
                        if joint_explicit_scale_mode
                        else feature_scale
                    )
                    constraint_gate = _gate_to_constraint(
                        gate_value_for_penalty, constraint_feature_scale
                    )
                    admm_penalty = (rho / 2.0) * torch.sum(
                        (constraint_gate - zk.detach() + uk.detach()) ** 2
                    )
                    if strict_two_copy_mode:
                        admm_penalty = admm_penalty + (rho / 2.0) * torch.sum(
                            (constraint_gate - yk.detach() + wk.detach()) ** 2
                        )
                    total_loss = full_data_loss + admm_penalty
                    gate_grad = torch.autograd.grad(total_loss, gate_param)[0]
                    gate_param.grad = gate_grad
                    opt_gate.step()
        else:
            model.train()
            for _g_inner_epoch in range(g_inner_epochs):
                for x_batch, y_batch in loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    opt_all.zero_grad()
                    y_hat = model(x_batch)
                    if n_classes > 2:
                        y_hat = torch.log_softmax(y_hat, dim=1)
                    else:
                        y_hat = y_hat.reshape(len(y_hat))

                    try:
                        data_loss = criterion(y_hat, y_batch)
                    except RuntimeError:
                        data_loss = criterion(y_hat, y_batch.float())
                    batch_n = int(x_batch.shape[0])
                    epoch_data_loss_sum += float(data_loss.detach().item()) * batch_n
                    epoch_data_loss_n += batch_n
                    if dynamic_C_mode in {"grad_scale", "kkt_grad"}:
                        data_gate_grad = torch.autograd.grad(
                            data_loss,
                            gate_param,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if data_gate_grad is not None:
                            grad_abs = data_gate_grad.detach().abs().reshape(-1)
                            if dynamic_C_mode == "kkt_grad":
                                grad_scale = torch.quantile(
                                    grad_abs,
                                    float(dynamic_C_quantile),
                                ).item()
                            else:
                                grad_scale = torch.median(grad_abs).item()
                            epoch_grad_scale_sum += float(grad_scale) * batch_n
                            epoch_grad_scale_n += batch_n

                    # ADMM augmented Lagrangian penalty on gate
                    # For bounded_gate: ADMM operates in RAW SPACE (pre-sigmoid)
                    # For unbounded_gate: ADMM operates in EFFECTIVE SPACE
                    gate_value_for_penalty = (
                        gate_param if bounded_gate else _effective_gate(gate_param)
                    )
                    constraint_feature_scale = (
                        _get_admm_feature_scale(detach=False)
                        if joint_explicit_scale_mode
                        else feature_scale
                    )
                    constraint_gate = _gate_to_constraint(
                        gate_value_for_penalty, constraint_feature_scale
                    )
                    admm_penalty = (rho / 2.0) * torch.sum(
                        (constraint_gate - zk.detach() + uk.detach()) ** 2
                    )
                    if strict_two_copy_mode:
                        admm_penalty = admm_penalty + (rho / 2.0) * torch.sum(
                            (constraint_gate - yk.detach() + wk.detach()) ** 2
                        )
                    total_loss = data_loss + admm_penalty

                    total_loss.backward()
                    opt_all.step()

        # The joint explicit scale is part of theta and may have changed during
        # the neural block.  Refresh it before forming the z-step center so the
        # center uses f_{theta^{k+1}}(g^{k+1}), not the pre-block scale.
        if joint_explicit_scale_mode:
            with torch.no_grad():
                feature_scale = _get_admm_feature_scale()

        #  (2) z-step: RatioNorm update (once per epoch)
        pre_threshold_band_row = None
        with torch.no_grad():
            zk_old = zk.clone()
            yk_old = yk.clone()
            gate_for_admm = _current_gate_for_admm()
            Ag = _gate_to_constraint(gate_for_admm, feature_scale)
            Ag_plus_u = Ag + uk

            if canonical_signed_mode:
                zk, prox_diagnostics = canonical_signed_ratio_prox(
                    Ag_plus_u,
                    float(C_epoch) / float(rho),
                )
                canonical_prox_log.append(
                    {
                        "epoch": int(epoch),
                        "C": float(C_epoch),
                        "rho": float(rho),
                        "lambda": float(C_epoch) / float(rho),
                        **prox_diagnostics,
                    }
                )
            elif strict_two_copy_mode:
                if ratio_norm_update == "strict_two_copy_weighted":
                    strict_penalty_weights = torch.clamp(
                        float(C_epoch) / score,
                        min=1e-8,
                    )
                    zk_new, strict_z_threshold = strict_ratio_weighted_z_update(
                        Ag,
                        yk,
                        uk,
                        penalty_weights=strict_penalty_weights,
                        rho=float(rho),
                    )
                else:
                    strict_penalty_weights = None
                    zk_new, strict_z_threshold = strict_ratio_z_update(
                        Ag,
                        yk,
                        uk,
                        C=float(C_epoch),
                        rho=float(rho),
                    )
                if not bounded_gate:
                    zk_new = _project_effective_gate(zk_new)
                if strict_penalty_weights is not None:
                    yk_new, strict_y_gamma, strict_y_delta = (
                        strict_ratio_weighted_y_update(
                            Ag,
                            zk_new,
                            wk,
                            penalty_weights=strict_penalty_weights,
                            rho=float(rho),
                            fallback_direction=yk,
                        )
                    )
                else:
                    yk_new, strict_y_gamma, strict_y_delta = strict_ratio_y_update(
                        Ag,
                        zk_new,
                        wk,
                        C=float(C_epoch),
                        rho=float(rho),
                        fallback_direction=yk,
                    )
                if not bounded_gate:
                    yk_new = _project_effective_gate(yk_new)
                if not torch.isfinite(zk_new).all() or not torch.isfinite(yk_new).all():
                    raise FloatingPointError(
                        f"{ratio_norm_update} produced a non-finite z or y update"
                    )
                zk = zk_new
                yk = yk_new
            elif use_ratio_norm and ratio_norm_update in {
                "formula_center_den",
                "formula_center_den_weighted",
                "formula_z_den",
                "formula_z_den_weighted",
            }:
                eps = 1e-10
                center_norm = torch.clamp(torch.norm(Ag_plus_u, p=2), min=eps)
                z_den_norm = torch.clamp(torch.norm(z_den, p=2), min=eps)
                ratio_coeff = _ratio_norm_coeff(Ag_plus_u)
                if (
                    dynamic_C_mode == "pre_threshold_band"
                    and C_schedule_per_epoch is None
                    and epoch % dynamic_C_update_interval == 0
                ):
                    if (
                        ratio_norm_update
                        in {"formula_center_den_weighted", "formula_z_den_weighted"}
                        and not uniform_penalty
                        and not uses_scaled_gate_constraint
                    ):
                        critical_denom = (
                            center_norm
                            if ratio_norm_update == "formula_center_den_weighted"
                            else z_den_norm
                        )
                        critical_scale = critical_denom * torch.clamp(score, min=1e-8)
                    elif ratio_norm_update == "formula_center_den":
                        critical_scale = torch.ones_like(score) * center_norm
                    else:
                        critical_scale = torch.ones_like(score) * z_den_norm

                    critical_C = (
                        float(rho)
                        * critical_scale
                        * Ag_plus_u.detach().abs()
                        / max(float(ratio_coeff), eps)
                    )
                    C_before = float(C_dynamic)
                    C_dynamic, pre_row = _pre_threshold_band_update(
                        C_before,
                        float(C),
                        critical_C,
                    )
                    C_epoch = float(C_dynamic)
                    pre_threshold_band_row = {
                        "epoch": int(epoch),
                        "mode": dynamic_C_mode,
                        "C_epoch_used": float(C_epoch),
                        "C_before": float(C_before),
                        "C_after": float(C_dynamic),
                        "phase_C_anchor": float(C),
                        "C_seed": float(C),
                        "C_bound_seed": float(dynamic_C_bound_seed),
                        "rho": float(rho),
                        "ratio_norm_coeff": float(ratio_coeff),
                        "lambda_eff": float(C_dynamic) / max(float(z_den_norm.item()), eps),
                        "target_factor": float(C_dynamic) / max(float(C), eps),
                        "min_factor": float(dynamic_C_min_factor),
                        "max_factor": float(dynamic_C_max_factor),
                        "support_band_up_mult": float(dynamic_C_support_up_mult),
                        "support_band_down_mult": float(dynamic_C_support_down_mult),
                        **pre_row,
                    }

                C_ratio = _ratio_norm_C(C_epoch, Ag_plus_u)
                if (
                    ratio_norm_update
                    in {"formula_center_den_weighted", "formula_z_den_weighted"}
                    and not uniform_penalty
                    and not uses_scaled_gate_constraint
                ):
                    lam = torch.clamp(C_ratio / score, min=1e-8)
                    denom_norm = (
                        center_norm
                        if ratio_norm_update == "formula_center_den_weighted"
                        else z_den_norm
                    )
                    threshold = lam / (rho * denom_norm)
                elif ratio_norm_update == "formula_center_den":
                    lam = torch.full_like(score, C_ratio)
                    threshold = C_ratio / (rho * center_norm)
                else:
                    lam = torch.full_like(score, C_ratio)
                    threshold = C_ratio / (rho * z_den_norm)

                # Exact sparse-copy minimizer for a detached denominator:
                # min_z sum_j lambda_j |z_j|/||z_den||_2
                #       + (rho/2)||z - (Ag+u)||_2^2.
                # formula_center_den replaces ||z_den||_2 by ||Ag+u||_2,
                # removing dependence on the denominator-copy trajectory.
                zk_new = soft_thresholding(Ag_plus_u, threshold)
                if not bounded_gate:
                    zk_new = _project_effective_gate(zk_new)
                if group_sparse_mode != "none":
                    zk_new, last_group_parent_idx, _ = _apply_group_sparse_projection(
                        zk_new, Ag_plus_u
                    )
                else:
                    zk_new = _enforce_min_support(zk_new, Ag_plus_u)

                if ratio_norm_update in {
                    "formula_center_den",
                    "formula_center_den_weighted",
                }:
                    z_den_new = Ag_plus_u.clone()
                else:
                    alpha = torch.sum(lam * torch.abs(zk_new))
                    r = torch.clamp(torch.norm(Ag_plus_u, p=2), min=eps)
                    if alpha > eps and r > eps:
                        # Exact radial minimizer for fixed numerator copy:
                        # min_y alpha/||y||_2 + (rho/2)||y - (Ag+u)||_2^2.
                        delta = alpha / (rho * r**3)
                        lambda_star = solve_cubic_denominator_copy(delta)
                        z_den_new = lambda_star * Ag_plus_u
                    else:
                        z_den_new = Ag_plus_u.clone()

                zk = zk_new
                z_den = z_den_new
            else:
                v = Ag_plus_u

                # --- Legacy sequential RatioNorm approximation on v ---
                # This branch first applies a fixed-threshold L1 shrinkage and
                # then rescales the shrunk vector by solving tau^3 - tau - D = 0.
                # It is useful as a historical/diagnostic mode, but it is not an
                # exact proximal update for ||z||_1 / ||z||_2.
                #
                # Optional importance-adaptive lambda_j = C / s_j heuristic.
                if uniform_penalty:
                    C_ratio = _ratio_norm_C(C_epoch, v)
                    lam = torch.full_like(score, C_ratio)
                elif uses_scaled_gate_constraint:
                    # Formula path: A enters through z = A*g, so the z-space
                    # RatioNorm threshold is scalar rather than divided by A.
                    C_ratio = _ratio_norm_C(C_epoch, v)
                    lam = torch.full_like(score, C_ratio)
                else:
                    C_ratio = _ratio_norm_C(C_epoch, v)
                    lam = C_ratio / score  # per-feature lambda
                lam = torch.clamp(
                    lam, min=1e-8
                )  # FIX: Remove max clamp to allow larger thresholds

                # Fixed-threshold L1 shrinkage used by the legacy approximation.
                v_shrunk = soft_thresholding(v, lam / rho)

                # Cubic rescale used by the legacy approximation (skip if ablation: plain L1).
                if use_ratio_norm:
                    v_norm_l1 = torch.norm(v_shrunk, p=1)
                    v_norm_l2 = safe_norm(v_shrunk)

                    if v_norm_l1 > 1e-8 and v_norm_l2 > 1e-8:
                        # D is scalar because this is a global rescale correction.
                        D_scalar = (
                            (C_ratio / rho)
                            * v_norm_l1
                            / torch.clamp(v_norm_l2**3, min=1e-10)
                        )

                        # Solve for scalar tau
                        tau_k = solve_cubic_ratio_norm(
                            torch.tensor([D_scalar], device=v.device)
                        )

                        # Apply Ratio Norm scaling to shrunk v
                        zk = tau_k * v_shrunk
                    else:
                        # v is approximately zero: all features pruned, keep zeros
                        zk = v_shrunk
                else:
                    # Plain L1: just soft-thresholding, no Ratio Norm scaling
                    zk = v_shrunk
                # For bounded_gate (raw space ADMM): no projection needed
                # The sigmoid naturally constrains effective gate to [0,1]
                # For unbounded_gate (effective space ADMM): project to valid range
                if not bounded_gate:
                    zk = _project_effective_gate(zk)
                if group_sparse_mode != "none":
                    zk, last_group_parent_idx, _ = _apply_group_sparse_projection(zk, v)
                else:
                    zk = _enforce_min_support(zk, v)

            if write_z_to_gate:
                z_as_gate = _constraint_to_gate(zk, feature_scale)
                if bounded_gate:
                    gate_param.data.copy_(z_as_gate)
                else:
                    gate_param.data.copy_(_parameter_from_effective(z_as_gate))

            last_term_z_support = int((zk.abs() > 1e-12).sum().item())
            last_parent_z_support = _group_parent_support(zk)

            if pre_threshold_band_row is not None:
                z_abs_for_history = zk.detach().abs()
                z_norm_for_history = float(torch.norm(z_abs_for_history, p=2).item())
                z_l1_for_history = float(torch.norm(z_abs_for_history, p=1).item())
                pre_threshold_band_row.update(
                    {
                        "actual_parent_support": int(last_parent_z_support),
                        "actual_term_support": int(last_term_z_support),
                        "z_l1": z_l1_for_history,
                        "z_norm": z_norm_for_history,
                        "z_l1_l2": z_l1_for_history
                        / max(z_norm_for_history, 1e-12),
                    }
                )
                dynamic_C_history.append(pre_threshold_band_row)

            if adaptive_C and adaptive_C_target_value > 0 and epoch % adaptive_C_update_interval == 0:
                z_support = (
                    last_parent_z_support
                    if group_sparse_mode != "none"
                    else last_term_z_support
                )
                low = max(1, int(np.floor(adaptive_C_low_ratio * adaptive_C_target_value)))
                high = max(low + 1, int(np.ceil(adaptive_C_high_ratio * adaptive_C_target_value)))
                C_old = C_dynamic
                action = "hold"
                if z_support == 0 or z_support < low:
                    C_dynamic = max(float(adaptive_C_min), C_dynamic / float(adaptive_C_mult))
                    action = "decrease"
                elif z_support > high:
                    C_dynamic = min(float(adaptive_C_max), C_dynamic * float(adaptive_C_mult))
                    action = "increase"
                adaptive_C_history.append(
                    {
                        "epoch": int(epoch),
                        "z_support": z_support,
                        "term_z_support": int(last_term_z_support),
                        "parent_z_support": int(last_parent_z_support),
                        "selected_parents": [
                            int(v) for v in last_group_parent_idx.detach().cpu().tolist()
                        ],
                        "target": int(adaptive_C_target_value),
                        "low": int(low),
                        "high": int(high),
                        "C_before": float(C_old),
                        "C_after": float(C_dynamic),
                        "rho": float(rho),
                        "action": action,
                    }
                )

            if (
                dynamic_C_mode
                not in {
                    "none",
                    "lambda_max_const",
                    "lambda_max_phase_const",
                    "root_n_lambda_max_const",
                    "root_n_lambda_max_phase_const",
                    "root_n_lambda_max_phase_monotone_after2",
                    "score_quantile_lambda_phase_const",
                    "parent_k_lambda_const",
                    "parent_k_phase_lambda_const",
                    "pre_threshold_band",
                }
                and epoch % dynamic_C_update_interval == 0
            ):
                eps = 1e-12
                z_abs = zk.detach().abs()
                z_l1 = float(torch.norm(z_abs, p=1).item())
                z_norm = float(torch.norm(z_abs, p=2).item())
                z_l1_l2 = z_l1 / max(z_norm, eps)
                data_loss_mean = epoch_data_loss_sum / max(epoch_data_loss_n, 1)
                loss_scale = float(np.sqrt(max(data_loss_mean, 0.0) + eps))
                grad_scale = (
                    epoch_grad_scale_sum / max(epoch_grad_scale_n, 1)
                    if epoch_grad_scale_n > 0
                    else 0.0
                )
                with torch.no_grad():
                    if bounded_gate:
                        gate_constraint_for_C = _gate_to_constraint(
                            gate_param.data, feature_scale
                        )
                    else:
                        gate_constraint_for_C = _gate_to_constraint(
                            _effective_gate(gate_param.data), feature_scale
                        )
                    primal_resid_for_C = float(
                        torch.norm(gate_constraint_for_C - zk).item()
                    )
                    dual_resid_for_C = float(rho * torch.norm(zk - zk_old).item())
                z_support_for_C = (
                    int(last_parent_z_support)
                    if group_sparse_mode != "none"
                    else int(last_term_z_support)
                )
                C_old = float(C_dynamic)
                action = "init"
                target_factor = 1.0
                lambda_eff = 0.0
                lambda_eff_mean = 0.0
                lambda_source = None
                lambda_n = 0
                lambda_loss_convention = None
                source_c = None
                source_gamma = None
                source_p_eff = None
                source_quantile = None
                target_C_unclipped = float(C)
                target_C_clipped = float(C)
                support_rel_change = 0.0
                primal_resid_rel_change = 0.0
                dual_resid_rel_change = 0.0
                loss_rel_change = 0.0
                event_stable = False
                strict_primal_z_relative = None
                strict_primal_y_relative = None
                strict_dual_z_step_relative = None
                strict_dual_y_step_relative = None
                support_band_parent_support = None
                support_band_term_support = None
                support_band_parent_low = None
                support_band_parent_high = None
                support_band_term_high = None
                support_band_signal = None
                refs_were_empty = dynamic_C_ref_z_norm is None
                if refs_were_empty:
                    dynamic_C_ref_z_norm = max(z_norm, eps)
                    dynamic_C_ref_l1_l2 = max(z_l1_l2, eps)
                    dynamic_C_ref_loss_scale = max(loss_scale, eps)
                    dynamic_C_ref_grad_scale = max(grad_scale, eps)

                if dynamic_C_mode == "strict_relative_continuation":
                    min_C = float(dynamic_C_bound_seed) * float(dynamic_C_min_factor)
                    max_C = float(dynamic_C_bound_seed) * float(dynamic_C_max_factor)
                    with torch.no_grad():
                        gate_norm_for_C = float(torch.norm(gate_constraint_for_C).item())
                        y_norm_for_C = float(torch.norm(yk).item())
                        z_old_norm_for_C = float(torch.norm(zk_old).item())
                        y_old_norm_for_C = float(torch.norm(yk_old).item())
                        primal_y_for_C = float(
                            torch.norm(gate_constraint_for_C - yk).item()
                        )
                        dual_y_for_C = float(rho * torch.norm(yk - yk_old).item())
                    strict_primal_z_relative = float(
                        primal_resid_for_C
                        / max(gate_norm_for_C, z_norm, eps)
                    )
                    strict_primal_y_relative = float(
                        primal_y_for_C
                        / max(gate_norm_for_C, y_norm_for_C, eps)
                    )
                    strict_dual_z_step_relative = float(
                        (dual_resid_for_C / rho) / max(z_old_norm_for_C, eps)
                    )
                    strict_dual_y_step_relative = float(
                        (dual_y_for_C / rho) / max(y_old_norm_for_C, eps)
                    )
                    event_stable = max(
                        strict_primal_z_relative,
                        strict_primal_y_relative,
                        strict_dual_z_step_relative,
                        strict_dual_y_step_relative,
                    ) <= float(dynamic_C_residual_tol)
                    if event_stable:
                        dynamic_C_event_stable_count += 1
                        action = "stable_hold"
                    else:
                        dynamic_C_event_stable_count = 0
                        action = "hold"
                    target_C_unclipped = float(C_dynamic)
                    target_C_clipped = min(max(target_C_unclipped, min_C), max_C)
                    if dynamic_C_event_stable_count >= int(
                        dynamic_C_stability_patience
                    ):
                        target_C_unclipped = float(C_dynamic) * float(
                            dynamic_C_continuation_gamma
                        )
                        target_C_clipped = min(
                            max(target_C_unclipped, min_C), max_C
                        )
                        if target_C_clipped > float(C_dynamic) + eps:
                            C_dynamic = target_C_clipped
                            action = "increase"
                        else:
                            action = "max_hold"
                        dynamic_C_event_stable_count = 0
                    lambda_eff = float(C_dynamic) / max(z_norm, eps)
                    target_factor = float(C_dynamic) / max(float(C), eps)
                elif dynamic_C_mode == "support_band":
                    support_k = int(diagnostic_k) if diagnostic_k else 1
                    support_k = max(1, support_k)
                    support_band_parent_support = int(last_parent_z_support)
                    support_band_term_support = int(last_term_z_support)
                    support_band_parent_low = (
                        int(dynamic_C_support_parent_low)
                        if dynamic_C_support_parent_low > 0
                        else support_k
                    )
                    support_band_parent_high = (
                        int(dynamic_C_support_parent_high)
                        if dynamic_C_support_parent_high > 0
                        else max(support_band_parent_low + 1, 4 * support_k, support_k + 4)
                    )
                    support_band_term_high = (
                        int(dynamic_C_support_term_high)
                        if dynamic_C_support_term_high > 0
                        else max(32, 16 * support_k)
                    )
                    support_band_signal = (
                        "parent_and_term"
                        if group_parent_share is not None
                        else "term_only_parent_unavailable"
                    )
                    min_C = float(dynamic_C_bound_seed) * float(dynamic_C_min_factor)
                    max_C = float(dynamic_C_bound_seed) * float(dynamic_C_max_factor)
                    target_C_unclipped = float(C_dynamic)
                    action = "hold"
                    if support_band_term_support == 0:
                        target_C_unclipped = float(C_dynamic) / float(dynamic_C_support_down_mult)
                        action = "decrease_zero_support"
                    elif (
                        support_band_term_support > support_band_term_high
                        or support_band_parent_support > support_band_parent_high
                    ):
                        target_C_unclipped = float(C_dynamic) * float(dynamic_C_support_up_mult)
                        action = "increase_dense_support"
                    elif support_band_parent_support < support_band_parent_low:
                        target_C_unclipped = float(C_dynamic) / float(dynamic_C_support_down_mult)
                        action = "decrease_low_parent_support"
                    target_C_clipped = min(max(target_C_unclipped, min_C), max_C)
                    if action != "hold":
                        if abs(target_C_clipped - float(C_dynamic)) <= eps:
                            action = "bound_hold"
                        else:
                            C_dynamic = target_C_clipped
                    lambda_eff = float(C_dynamic) / max(z_norm, eps)
                    target_factor = float(C_dynamic) / max(float(C), eps)
                elif dynamic_C_mode == "event_continuation":
                    min_C = float(dynamic_C_bound_seed) * float(dynamic_C_min_factor)
                    max_C = float(dynamic_C_bound_seed) * float(dynamic_C_max_factor)
                    target_C_unclipped = float(C_dynamic)
                    target_C_clipped = min(max(target_C_unclipped, min_C), max_C)
                    action = "init"
                    if dynamic_C_event_prev_metrics is not None:
                        prev = dynamic_C_event_prev_metrics
                        support_rel_change = abs(
                            float(z_support_for_C) - float(prev["z_support"])
                        ) / max(float(prev["z_support"]), 1.0)
                        primal_resid_rel_change = abs(
                            primal_resid_for_C - float(prev["primal_resid"])
                        ) / max(float(prev["primal_resid"]), eps)
                        dual_resid_rel_change = abs(
                            dual_resid_for_C - float(prev["dual_resid"])
                        ) / max(float(prev["dual_resid"]), eps)
                        loss_rel_change = abs(
                            data_loss_mean - float(prev["loss"])
                        ) / max(float(prev["loss"]), eps)
                        event_stable = (
                            support_rel_change <= float(dynamic_C_support_tol)
                            and loss_rel_change <= float(dynamic_C_loss_tol)
                            and max(primal_resid_rel_change, dual_resid_rel_change)
                            <= float(dynamic_C_residual_tol)
                        )
                        if event_stable:
                            dynamic_C_event_stable_count += 1
                            action = "stable_hold"
                        else:
                            dynamic_C_event_stable_count = 0
                            action = "hold"
                        if dynamic_C_event_stable_count >= int(
                            dynamic_C_stability_patience
                        ):
                            target_C_unclipped = float(C_dynamic) * float(
                                dynamic_C_continuation_gamma
                            )
                            target_C_clipped = min(
                                max(target_C_unclipped, min_C), max_C
                            )
                            if target_C_clipped > float(C_dynamic) + eps:
                                C_dynamic = target_C_clipped
                                action = "increase"
                            else:
                                action = "max_hold"
                            dynamic_C_event_stable_count = 0
                    dynamic_C_event_prev_metrics = {
                        "loss": float(data_loss_mean),
                        "primal_resid": float(primal_resid_for_C),
                        "dual_resid": float(dual_resid_for_C),
                        "z_support": int(z_support_for_C),
                    }
                    lambda_eff = float(C_dynamic) / max(z_norm, eps)
                    target_factor = float(C_dynamic) / max(float(C), eps)
                elif dynamic_C_mode == "score_quantile_lambda_dynamic":
                    grad_mean_for_C, lambda_n = _data_gate_grad_mean()
                    grad_abs_mean = grad_mean_for_C.detach().abs()
                    lambda_eff_mean = (
                        float(grad_abs_mean.max().item()) if lambda_n > 0 else 0.0
                    )
                    source_c = 1.1
                    source_gamma = 0.1 / float(np.log(max(int(lambda_n), 3)))
                    source_p_eff = max(int(grad_abs_mean.numel()), 2)
                    tail_prob = 1.0 - float(source_gamma) / (2.0 * float(source_p_eff))
                    tail_prob = min(max(tail_prob, 1e-12), 1.0 - 1e-12)
                    source_quantile = NormalDist().inv_cdf(tail_prob)
                    lambda_eff = (
                        float(source_c)
                        * float(source_quantile)
                        * float(np.sqrt(max(int(lambda_n), 1)))
                        * float(lambda_eff_mean)
                    )
                    target_C_unclipped = lambda_eff * max(z_norm, eps)
                    target_C_clipped = max(float(target_C_unclipped), eps)
                    C_dynamic = target_C_clipped
                    target_factor = float(C_dynamic) / max(float(C), eps)
                    lambda_source = "expanded_max_score_quantile"
                    lambda_loss_convention = (
                        "dynamic_root_n_mean_loss_times_source_quantile;"
                        "c=1.1,gamma=0.1/log(n),normal_tail=1-gamma/(2p)"
                    )
                    action = "set_score_quantile"
                elif dynamic_C_mode == "kkt_grad":
                    lambda_eff = float(dynamic_C_kappa) * max(float(grad_scale), eps)
                    target_C_unclipped = lambda_eff * max(z_norm, eps)
                    min_C = float(dynamic_C_bound_seed) * float(dynamic_C_min_factor)
                    max_C = float(dynamic_C_bound_seed) * float(dynamic_C_max_factor)
                    target_C_clipped = min(max(target_C_unclipped, min_C), max_C)
                    C_dynamic = (
                        float(dynamic_C_ema) * float(C_dynamic)
                        + (1.0 - float(dynamic_C_ema)) * target_C_clipped
                    )
                    action = "update"
                elif dynamic_C_mode == "sqrt_loss":
                    lambda_eff = (
                        float(dynamic_C_kappa)
                        * max(loss_scale, eps)
                        * float(np.sqrt(np.log(max(int(m), 2)) / max(int(N), 1)))
                    )
                    target_C_unclipped = lambda_eff * max(z_norm, eps)
                    min_C = float(dynamic_C_bound_seed) * float(dynamic_C_min_factor)
                    max_C = float(dynamic_C_bound_seed) * float(dynamic_C_max_factor)
                    target_C_clipped = min(max(target_C_unclipped, min_C), max_C)
                    C_dynamic = (
                        float(dynamic_C_ema) * float(C_dynamic)
                        + (1.0 - float(dynamic_C_ema)) * target_C_clipped
                    )
                    action = "update"
                elif not refs_were_empty:
                    z_factor = max(z_norm, eps) / max(float(dynamic_C_ref_z_norm), eps)
                    target_factor = z_factor
                    if dynamic_C_mode == "l1_l2":
                        target_factor = max(z_l1_l2, eps) / max(float(dynamic_C_ref_l1_l2), eps)
                    elif dynamic_C_mode == "loss_scale":
                        loss_factor = max(loss_scale, eps) / max(float(dynamic_C_ref_loss_scale), eps)
                        target_factor *= loss_factor
                    elif dynamic_C_mode == "grad_scale":
                        grad_factor = max(grad_scale, eps) / max(float(dynamic_C_ref_grad_scale), eps)
                        target_factor *= grad_factor
                    target_C_unclipped = float(C) * float(target_factor)
                    min_C = float(dynamic_C_bound_seed) * float(dynamic_C_min_factor)
                    max_C = float(dynamic_C_bound_seed) * float(dynamic_C_max_factor)
                    target_C_clipped = min(max(target_C_unclipped, min_C), max_C)
                    C_dynamic = (
                        float(dynamic_C_ema) * float(C_dynamic)
                        + (1.0 - float(dynamic_C_ema)) * target_C_clipped
                    )
                    action = "update"
                    lambda_eff = float(target_C_clipped) / max(z_norm, eps)
                dynamic_C_history.append(
                    {
                        "epoch": int(epoch),
                        "mode": dynamic_C_mode,
                        "C_epoch_used": float(C_epoch),
                        "C_before": C_old,
                        "C_after": float(C_dynamic),
                        "phase_C_anchor": float(C),
                        "C_seed": float(C),
                        "C_bound_seed": float(dynamic_C_bound_seed),
                        "lambda_eff": float(lambda_eff),
                        "lambda_eff_mean_loss": float(lambda_eff_mean),
                        "lambda_source": lambda_source,
                        "lambda_max_n": int(lambda_n),
                        "lambda_max_loss_convention": lambda_loss_convention,
                        "source_c": None if source_c is None else float(source_c),
                        "source_gamma": None if source_gamma is None else float(source_gamma),
                        "source_p_eff": None if source_p_eff is None else int(source_p_eff),
                        "source_normal_quantile": None
                        if source_quantile is None
                        else float(source_quantile),
                        "kappa": float(dynamic_C_kappa),
                        "quantile": float(dynamic_C_quantile),
                        "target_factor": float(target_factor),
                        "target_C_unclipped": float(target_C_unclipped),
                        "target_C_clipped": float(target_C_clipped),
                        "z_l1": float(z_l1),
                        "z_norm": float(z_norm),
                        "z_l1_l2": float(z_l1_l2),
                        "ref_z_norm": float(dynamic_C_ref_z_norm),
                        "ref_z_l1_l2": float(dynamic_C_ref_l1_l2),
                        "loss_scale": float(loss_scale),
                        "ref_loss_scale": float(dynamic_C_ref_loss_scale),
                        "grad_scale": float(grad_scale),
                        "ref_grad_scale": float(dynamic_C_ref_grad_scale),
                        "event_stable": bool(event_stable),
                        "event_stable_count": int(dynamic_C_event_stable_count),
                        "event_support": int(z_support_for_C),
                        "event_support_rel_change": float(support_rel_change),
                        "event_primal_resid": float(primal_resid_for_C),
                        "event_dual_resid": float(dual_resid_for_C),
                        "event_primal_resid_rel_change": float(primal_resid_rel_change),
                        "event_dual_resid_rel_change": float(dual_resid_rel_change),
                        "event_loss_rel_change": float(loss_rel_change),
                        "strict_primal_z_relative": strict_primal_z_relative,
                        "strict_primal_y_relative": strict_primal_y_relative,
                        "strict_dual_z_step_relative": strict_dual_z_step_relative,
                        "strict_dual_y_step_relative": strict_dual_y_step_relative,
                        "support_band_signal": support_band_signal,
                        "support_band_parent_support": support_band_parent_support,
                        "support_band_term_support": support_band_term_support,
                        "support_band_parent_low": support_band_parent_low,
                        "support_band_parent_high": support_band_parent_high,
                        "support_band_term_high": support_band_term_high,
                        "support_band_up_mult": float(dynamic_C_support_up_mult),
                        "support_band_down_mult": float(dynamic_C_support_down_mult),
                        "continuation_gamma": float(dynamic_C_continuation_gamma),
                        "stability_patience": int(dynamic_C_stability_patience),
                        "support_tol": float(dynamic_C_support_tol),
                        "residual_tol": float(dynamic_C_residual_tol),
                        "loss_tol": float(dynamic_C_loss_tol),
                        "ema": float(dynamic_C_ema),
                        "min_factor": float(dynamic_C_min_factor),
                        "max_factor": float(dynamic_C_max_factor),
                        "rho": float(rho),
                        "action": action,
                    }
                )

        #  (3) Dual update (once per epoch)
        with torch.no_grad():
            gate_value = gate_param.data if bounded_gate else _effective_gate(gate_param.data)
            constraint_gate = _gate_to_constraint(gate_value, feature_scale)
            uk_previous = uk.detach().clone()
            uk = uk + constraint_gate - zk
            if strict_two_copy_mode:
                wk_previous = wk.detach().clone()
                wk = wk + constraint_gate - yk
        # Log convergence diagnostics
        with torch.no_grad():
            gate_value = gate_param.data if bounded_gate else _effective_gate(gate_param.data)
            constraint_gate = _gate_to_constraint(gate_value, feature_scale)
            primal_z = torch.norm(constraint_gate - zk).item()
            dual_z = rho * torch.norm(zk - zk_old).item()
            gate_norm = torch.norm(constraint_gate).item()
            z_norm = torch.norm(zk).item()
            scaled_dual_norm = rho * torch.norm(uk).item()
            scaled_dual_previous_norm = rho * torch.norm(uk_previous).item()
            if strict_two_copy_mode:
                primal_y = torch.norm(constraint_gate - yk).item()
                dual_y = rho * torch.norm(yk - yk_old).item()
                y_norm = torch.norm(yk).item()
                z_old_norm = torch.norm(zk_old).item()
                y_old_norm = torch.norm(yk_old).item()
                primal_resid = float(np.hypot(primal_z, primal_y))
                dual_resid = float(np.hypot(dual_z, dual_y))
                strict_two_copy_log.append(
                    {
                        "epoch": int(epoch),
                        "C": float(C_epoch),
                        "rho": float(rho),
                        "z_threshold": float(
                            torch.median(strict_z_threshold.reshape(-1)).item()
                        ),
                        "z_threshold_min": float(strict_z_threshold.min().item()),
                        "z_threshold_max": float(strict_z_threshold.max().item()),
                        "y_gamma": (
                            float(strict_y_gamma.item())
                            if torch.isfinite(strict_y_gamma)
                            else None
                        ),
                        "y_delta": (
                            float(strict_y_delta.item())
                            if torch.isfinite(strict_y_delta)
                            else None
                        ),
                        "primal_z": float(primal_z),
                        "primal_y": float(primal_y),
                        "dual_z": float(dual_z),
                        "dual_y": float(dual_y),
                        "primal_z_relative": float(
                            primal_z / max(gate_norm, z_norm, 1e-12)
                        ),
                        "primal_y_relative": float(
                            primal_y / max(gate_norm, y_norm, 1e-12)
                        ),
                        "dual_z_step_relative": float(
                            (dual_z / rho) / max(z_old_norm, 1e-12)
                        ),
                        "dual_y_step_relative": float(
                            (dual_y / rho) / max(y_old_norm, 1e-12)
                        ),
                        "gate_norm": float(gate_norm),
                        "z_norm": float(z_norm),
                        "y_norm": float(y_norm),
                        "z_support": int((zk.abs() > 1e-12).sum().item()),
                    }
                )
            else:
                primal_resid = float(primal_z)
                dual_resid = float(dual_z)
            relative_record = {
                "epoch": int(epoch),
                "C": float(C_epoch),
                "rho": float(rho),
                "primal": float(primal_resid),
                "dual": float(dual_resid),
                "primal_relative": float(
                    primal_z / max(1.0, gate_norm, z_norm)
                ),
                "dual_relative": float(
                    dual_z
                    / max(
                        1.0,
                        scaled_dual_norm,
                        scaled_dual_previous_norm,
                    )
                ),
                "gate_norm": float(gate_norm),
                "z_norm": float(z_norm),
                "scaled_dual_norm": float(scaled_dual_norm),
                "scaled_dual_previous_norm": float(
                    scaled_dual_previous_norm
                ),
                "z_support": int((zk.abs() > 1e-12).sum().item()),
            }
            if strict_two_copy_mode:
                scaled_w_dual_norm = rho * torch.norm(wk).item()
                scaled_w_dual_previous_norm = rho * torch.norm(
                    wk_previous
                ).item()
                relative_record.update(
                    {
                        "y_norm": float(y_norm),
                        "scaled_w_dual_norm": float(scaled_w_dual_norm),
                        "scaled_w_dual_previous_norm": float(
                            scaled_w_dual_previous_norm
                        ),
                    }
                )
            relative_convergence_log.append(relative_record)
            convergence_log.append((epoch, primal_resid, dual_resid))

            if epoch_end_callback is not None:
                gate_l1 = torch.linalg.vector_norm(constraint_gate, ord=1)
                gate_l2 = torch.linalg.vector_norm(constraint_gate, ord=2)
                ratio_norm = (
                    float(gate_l1.item() / gate_l2.item())
                    if float(gate_l2.item()) > 0.0
                    else 1.0
                )
                epoch_snapshot = {
                    "epoch": int(epoch),
                    "C": float(C_epoch),
                    "rho": float(rho),
                    "train_supervised_loss": (
                        float(epoch_data_loss_sum / epoch_data_loss_n)
                        if epoch_data_loss_n > 0
                        else None
                    ),
                    "ratio_norm_effective_gate": ratio_norm,
                    "raw_gate": (
                        model.get_raw_gate(training_drop=False)
                        if hasattr(model, "get_raw_gate")
                        else gate_value
                    )
                    .detach()
                    .cpu()
                    .clone(),
                    "effective_gate": constraint_gate.detach().cpu().clone(),
                    "z": zk.detach().cpu().clone(),
                    "u": uk.detach().cpu().clone(),
                    "residual": dict(relative_record),
                }
                if epoch_callback_include_optimizer_moments:
                    if split_neural_block_mode:
                        optimizer_by_parameter = {
                            **{
                                id(parameter): opt_theta
                                for group in opt_theta.param_groups
                                for parameter in group["params"]
                            },
                            **{
                                id(parameter): opt_gate
                                for group in opt_gate.param_groups
                                for parameter in group["params"]
                            },
                        }
                    else:
                        optimizer_by_parameter = {
                            id(parameter): opt_all
                            for group in opt_all.param_groups
                            for parameter in group["params"]
                        }
                    first_moments: Dict[str, Optional[torch.Tensor]] = {}
                    optimizer_steps: Dict[str, Optional[float]] = {}
                    for name, parameter in model.named_parameters():
                        if name not in {"gate", "first_raw_weight"}:
                            continue
                        optimizer = optimizer_by_parameter.get(id(parameter))
                        state = {} if optimizer is None else optimizer.state.get(parameter, {})
                        exp_avg = state.get("exp_avg")
                        step = state.get("step")
                        first_moments[name] = (
                            None
                            if exp_avg is None
                            else exp_avg.detach().cpu().clone()
                        )
                        optimizer_steps[name] = (
                            None
                            if step is None
                            else float(step.detach().cpu().item())
                            if torch.is_tensor(step)
                            else float(step)
                        )
                    epoch_snapshot["optimizer_first_moments"] = first_moments
                    epoch_snapshot["optimizer_steps"] = optimizer_steps
                    epoch_snapshot["optimizer_moment_semantics"] = (
                        "Adam exp_avg after the final neural-block minibatch update "
                        "of this ADMM epoch; it includes the supervised and augmented "
                        "terms used by the optimizer and is not a supervised gradient."
                    )

        if epoch_end_callback is not None:
            epoch_end_callback(model, epoch_snapshot)

        #  Early Stopping Check
        if use_early_stopping and val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    y_val_hat = model(x_val)
                    if n_classes > 2:
                        y_val_hat = torch.log_softmax(y_val_hat, dim=1)
                    else:
                        y_val_hat = y_val_hat.reshape(len(y_val_hat))
                    val_loss += criterion(y_val_hat, y_val.float()).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best state
                best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(
                        f"Early stopping at epoch {warmup_epochs + epoch}, best val loss: {best_val_loss:.4f}"
                    )
                    break

        #  Adaptive rho (Boyd section 3.4.1) with proper dual rescaling
        if (
            adaptive_rho
            and not rho_follow_lr
            and epoch > 0
            and epoch % rho_update_interval == 0
        ):
            with torch.no_grad():
                gate_value = _effective_gate(gate_param.data)
                constraint_gate = _gate_to_constraint(gate_value, feature_scale)
                r_z = torch.norm(constraint_gate - zk).item()
                s_z = rho * torch.norm(zk - zk_old).item()
                if strict_two_copy_mode:
                    r_y = torch.norm(constraint_gate - yk).item()
                    s_y = rho * torch.norm(yk - yk_old).item()
                    r_norm = float(np.hypot(r_z, r_y))
                    s_norm = float(np.hypot(s_z, s_y))
                else:
                    r_norm = r_z
                    s_norm = s_z
                mu_bal = float(adaptive_rho_mu)
                r_n = max(r_norm, 1e-12)
                s_n = max(s_norm, 1e-12)
                rho_old = rho
                if r_n > mu_bal * s_n:
                    rho = min(rho * float(adaptive_rho_tau), float(adaptive_rho_max))
                elif s_n > mu_bal * r_n:
                    rho_min_value = (
                        float(adaptive_rho_min)
                        if adaptive_rho_min is not None
                        else (
                            5.0
                            if n_features < 64
                            else (20.0 if n_features < 256 else 50.0)
                        )
                    )
                    rho = max(rho / float(adaptive_rho_tau), rho_min_value)
                # Rescale dual variable when rho changes (Boyd section 3.4.1)
                if rho != rho_old:
                    uk = uk * (rho_old / rho)
                    if strict_two_copy_mode:
                        wk = wk * (rho_old / rho)

        if strict_kkt_check_interval > 0 and (
            (epoch + 1) % int(strict_kkt_check_interval) == 0
            or epoch == prune_epochs - 1
        ):
            stationarity_check = _strict_two_copy_stationarity_diagnostics(
                model,
                loader,
                criterion,
                n_classes,
                zk,
                yk,
                uk,
                wk,
                C=float(C_epoch),
                rho=float(rho),
            )
            support_tuple = tuple(
                torch.nonzero(zk.detach().abs() > 1e-8, as_tuple=False)
                .reshape(-1)
                .cpu()
                .tolist()
            )
            support_unchanged = (
                strict_kkt_previous_support is not None
                and support_tuple == strict_kkt_previous_support
            )
            combined_criterion = _strict_stationarity_combined_criterion(
                stationarity_check,
                relative_tolerance=float(strict_kkt_tolerance),
                absolute_tolerance=float(strict_kkt_absolute_tolerance),
            )
            stationarity_ok = bool(combined_criterion["passed"])
            if stationarity_ok:
                strict_kkt_consecutive = (
                    strict_kkt_consecutive + 1 if support_unchanged else 1
                )
            else:
                strict_kkt_consecutive = 0
            strict_kkt_previous_support = support_tuple
            strict_kkt_history.append(
                {
                    "epoch": int(epoch),
                    "outer_iteration": int(epoch + 1),
                    "C": float(C_epoch),
                    "rho": float(rho),
                    "z_support": int(len(support_tuple)),
                    "support_unchanged_from_previous_check": bool(
                        support_unchanged
                    ),
                    "stationarity_within_tolerance": stationarity_ok,
                    "qualifying_consecutive_checks": int(strict_kkt_consecutive),
                    "combined_criterion": combined_criterion,
                    "stationarity": stationarity_check,
                }
            )
            if strict_kkt_consecutive >= int(strict_kkt_patience):
                strict_kkt_converged = True
                strict_kkt_stop_epoch = int(epoch)
                break

    # Restore best state if early stopping was triggered
    if use_early_stopping and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Restored best model (val loss: {best_val_loss:.4f})")

    model.eval()
    with torch.no_grad():
        final_feature_scale = _get_admm_feature_scale()
        final_gate_for_admm = _current_gate_for_admm()
        if hasattr(model, "get_raw_gate"):
            final_raw_gate = model.get_raw_gate(training_drop=False)
        else:
            final_raw_gate = final_gate_for_admm
        if hasattr(model, "get_effective_gate"):
            final_effective_gate = model.get_effective_gate(training_drop=False)
        else:
            final_effective_gate = _gate_to_constraint(
                final_gate_for_admm, final_feature_scale
            )
        if hasattr(model, "get_constraint_coordinate"):
            final_constraint_coordinate = model.get_constraint_coordinate()
        else:
            final_constraint_coordinate = final_effective_gate
        zk_gate = _constraint_to_gate(zk, final_feature_scale)
        final_group_parent_score = _group_parent_scores(zk)
        final_parent_z_support = _group_parent_support(zk)
        parent_max_g_diag = _parent_max_selection_diag(final_gate_for_admm)
        parent_max_A_times_g_diag = _parent_max_selection_diag(
            final_feature_scale * final_gate_for_admm
        )
        parent_max_z_diag = _parent_max_selection_diag(zk)
        parent_weight_sum_g_diag = _parent_weight_sum_selection_diag(final_gate_for_admm)
        parent_weight_sum_A_times_g_diag = _parent_weight_sum_selection_diag(
            final_feature_scale * final_gate_for_admm
        )
        parent_weight_sum_z_diag = _parent_weight_sum_selection_diag(zk)
    strict_stationarity = None
    if ratio_norm_update == "strict_two_copy":
        strict_stationarity = _strict_two_copy_stationarity_diagnostics(
            model,
            loader,
            criterion,
            n_classes,
            zk,
            yk,
            uk,
            wk,
            C=float(last_C_epoch),
            rho=float(rho),
        )
        strict_stationarity["combined_criterion"] = (
            _strict_stationarity_combined_criterion(
                strict_stationarity,
                relative_tolerance=float(strict_kkt_tolerance),
                absolute_tolerance=float(strict_kkt_absolute_tolerance),
            )
        )
    model.convergence_log = convergence_log
    model.admm_state = {
        "zk": zk.detach().cpu().clone(),
        "z_den": z_den.detach().cpu().clone(),
        "yk": yk.detach().cpu().clone(),
        "zk_gate": zk_gate.detach().cpu().clone(),
        "uk": uk.detach().cpu().clone(),
        "wk": wk.detach().cpu().clone(),
        "feature_scale": final_feature_scale.detach().cpu().clone(),
        "raw_gate": final_raw_gate.detach().cpu().clone(),
        "effective_gate": final_effective_gate.detach().cpu().clone(),
        "constraint_coordinate": final_constraint_coordinate.detach().cpu().clone(),
        "training_variant": training_variant,
        "selection_readout": "effective_gate",
        "weight_decay": float(weight_decay),
        "last_z_step_score": score.detach().cpu().clone(),
        "frozen_score": frozen_score.detach().cpu().clone(),
        "uses_external_penalty_score": bool(
            not uniform_penalty
            and not uses_scaled_gate_constraint
            and ratio_norm_update
            in {
                "current",
                "formula_center_den_weighted",
                "formula_z_den_weighted",
                "strict_two_copy_weighted",
            }
        ),
        "score_update_mode": score_update_mode,
        "score_permutation_seed": (
            SCORE_PERMUTATION_SEED
            if score_update_mode == "epoch_permuted"
            else None
        ),
        "admm_weighted_gate": admm_weighted_gate,
        "admm_constraint_mode": effective_admm_constraint_mode,
        "feature_scale_joint_gradient": bool(joint_explicit_scale_mode),
        "ratio_norm_update": ratio_norm_update,
        "canonical_prox_log": canonical_prox_log,
        "relative_convergence_log": relative_convergence_log,
        "ratio_norm_dim_scale": ratio_norm_dim_scale,
        "ratio_norm_p": float(ratio_norm_p),
        "ratio_norm_q": float(ratio_norm_q),
        "ratio_norm_d_active": int(zk.numel()),
        "ratio_norm_coeff": (
            1.0
            if strict_two_copy_mode
            else float(_ratio_norm_coeff(zk))
        ),
        "adaptive_rho": adaptive_rho,
        "adaptive_rho_mu": float(adaptive_rho_mu),
        "adaptive_rho_tau": float(adaptive_rho_tau),
        "adaptive_rho_min": (
            None if adaptive_rho_min is None else float(adaptive_rho_min)
        ),
        "adaptive_rho_max": float(adaptive_rho_max),
        "lr_scheduler": lr_scheduler,
        "cosine_eta_min": cosine_eta_min_value if lr_scheduler == "cosine" else None,
        "cosine_decay_fraction": (
            float(cosine_decay_fraction) if lr_scheduler == "cosine" else None
        ),
        "g_inner_epochs": int(g_inner_epochs),
        "requested_outer_iterations": int(prune_epochs),
        "completed_outer_iterations": int(completed_outer_iterations),
        "total_admm_optimizer_passes": int(
            completed_outer_iterations * g_inner_epochs
        ),
        "strict_kkt_check_interval": int(strict_kkt_check_interval),
        "strict_kkt_tolerance": float(strict_kkt_tolerance),
        "strict_kkt_absolute_tolerance": float(strict_kkt_absolute_tolerance),
        "strict_kkt_patience": int(strict_kkt_patience),
        "strict_kkt_converged": bool(strict_kkt_converged),
        "strict_kkt_stop_epoch": strict_kkt_stop_epoch,
        "strict_kkt_stop_reason": (
            "full_kkt_and_exact_support_stable"
            if strict_kkt_converged
            else (
                "disabled"
                if strict_kkt_check_interval == 0
                else "safety_cap_reached_without_full_kkt"
            )
        ),
        "strict_kkt_history": strict_kkt_history,
        "rho_follow_lr": rho_follow_lr,
        "strict_neural_block_mode": strict_neural_block_mode,
        "strict_gate_inner_enabled": bool(gate_inner_solve_mode),
        "strict_gate_inner_max_steps": int(strict_gate_inner_max_steps),
        "strict_gate_inner_total_steps": int(strict_gate_inner_total_steps),
        "strict_gate_inner_passes": int(strict_gate_inner_passes),
        "strict_gate_inner_cap_hits": int(strict_gate_inner_cap_hits),
        "strict_gate_inner_mean_steps": (
            float(strict_gate_inner_total_steps)
            / max(float(len(strict_gate_inner_history)), 1.0)
        ),
        "strict_gate_inner_history": strict_gate_inner_history,
        "final_lr": float(
            (opt_gate if split_neural_block_mode else opt_all).param_groups[0]["lr"]
        ),
        "rho": float(rho),
        "rho_init": float(rho_init),
        "C_used": float(last_C_epoch),
        "C_initial": float(C),
        "C_schedule_per_epoch": [float(v) for v in C_epoch_history],
        "adaptive_C": adaptive_C,
        "adaptive_C_target": adaptive_C_target_value,
        "adaptive_C_history": adaptive_C_history,
        "dynamic_C_mode": dynamic_C_mode,
        "dynamic_C_update_interval": int(dynamic_C_update_interval),
        "dynamic_C_ema": float(dynamic_C_ema),
        "dynamic_C_min_factor": float(dynamic_C_min_factor),
        "dynamic_C_max_factor": float(dynamic_C_max_factor),
        "dynamic_C_kappa": float(dynamic_C_kappa),
        "dynamic_C_quantile": float(dynamic_C_quantile),
        "dynamic_C_continuation_gamma": float(dynamic_C_continuation_gamma),
        "dynamic_C_stability_patience": int(dynamic_C_stability_patience),
        "dynamic_C_support_tol": float(dynamic_C_support_tol),
        "dynamic_C_residual_tol": float(dynamic_C_residual_tol),
        "dynamic_C_loss_tol": float(dynamic_C_loss_tol),
        "dynamic_C_cap_seed": float(dynamic_C_bound_seed),
        "dynamic_C_floor": None if dynamic_C_floor is None else float(dynamic_C_floor),
        "dynamic_C_support_parent_low": int(dynamic_C_support_parent_low),
        "dynamic_C_support_parent_high": int(dynamic_C_support_parent_high),
        "dynamic_C_support_term_low": int(dynamic_C_support_term_low),
        "dynamic_C_support_term_high": int(dynamic_C_support_term_high),
        "dynamic_C_support_up_mult": float(dynamic_C_support_up_mult),
        "dynamic_C_support_down_mult": float(dynamic_C_support_down_mult),
        "dynamic_C_history": dynamic_C_history,
        "group_sparse_mode": group_sparse_mode,
        "group_parent_support": int(group_parent_support_value),
        "group_selected_parents": [
            int(v) for v in last_group_parent_idx.detach().cpu().tolist()
        ],
        "group_parent_score": final_group_parent_score.detach().cpu().clone(),
        "parent_z_support": int(final_parent_z_support),
        "term_z_support": int((zk.detach().abs() > 1e-12).sum().item()),
        "term_y_support": int((yk.detach().abs() > 1e-12).sum().item()),
        "strict_two_copy_log": strict_two_copy_log,
        "strict_stationarity": strict_stationarity,
        "parent_max_g_diag": parent_max_g_diag,
        "parent_max_A_times_g_diag": parent_max_A_times_g_diag,
        "parent_max_z_diag": parent_max_z_diag,
        "parent_weight_sum_g_diag": parent_weight_sum_g_diag,
        "parent_weight_sum_A_times_g_diag": parent_weight_sum_A_times_g_diag,
        "parent_weight_sum_z_diag": parent_weight_sum_z_diag,
        "auto_C_info": auto_C_info,
        "optimizer_state_dict": (
            opt_theta.state_dict()
            if split_neural_block_mode
            else opt_all.state_dict()
        ),
        "theta_optimizer_state_dict": (
            opt_theta.state_dict() if split_neural_block_mode else None
        ),
        "gate_optimizer_state_dict": (
            opt_gate.state_dict() if split_neural_block_mode else None
        ),
    }
    if joint_explicit_scale_mode:
        model.admm_state.update(
            {
                "final_gate": final_gate_for_admm.detach().cpu().clone(),
                "functional_gate": final_effective_gate.detach().cpu().clone(),
                "scale_gate_identifiability": (
                    "g=a_times_s_is_functional; a_and_s_have_reciprocal_gauge"
                ),
            }
        )
    if training_variant in {"effective_factorized", "unit_gate"}:
        model.feature_selection_score = final_effective_gate.detach().abs().clone()
        model.feature_selection_snapshot = {
            "readout": "effective_gate",
            "timepoint": "end_of_train_input_group",
        }


def _train_adaptive_input_group(
    model,  # AdaptiveFeatureSelectionMLP or AdaptiveFeatureSelector
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    lr: float = 0.005,
    C: float = 0.05,
    epochs: int = 500,
    warmup_epochs: int = 120,
    batch_size: int = 64,
    rho_init: float = 200.0,
    device: Optional[str] = None,
    use_ratio_norm: bool = True,
    use_admm: bool = True,
    optimizer_type: str = "adam",
    use_early_stopping: bool = False,
    patience: int = 66,
    val_split: float = 0.2,
) -> None:
    """Train an adaptive model with standard training (simplified for adaptive architectures).

    This is a simplified training function for adaptive models that don't have
    the same structure as GatedFeatureSelectionMLP.

    Args:
        model: AdaptiveFeatureSelectionMLP or AdaptiveFeatureSelector
        X_train: Training features
        y_train: Training labels
        n_classes: Number of classes
        lr: Learning rate
        C: Sparsity coefficient (not used in simplified version)
        epochs: Total training epochs
        warmup_epochs: Warmup epochs before ADMM (not used in simplified version)
        batch_size: Batch size
        rho_init: Initial ADMM penalty parameter (not used in simplified version)
        device: Training device
        use_ratio_norm: Use Ratio Norm (not used in simplified version)
        use_admm: Use ADMM (not used in simplified version)
        optimizer_type: 'adam' or 'adagrad'
        use_early_stopping: Enable early stopping
        patience: Early stopping patience
        val_split: Validation split
    """
    # Reuse the full ADMM path when the adaptive model still exposes a
    # global per-feature gate vector compatible with the original solver.
    gate_attr = getattr(model, "gate", None)
    if isinstance(gate_attr, nn.Parameter) and hasattr(model, "first_linear"):
        _train_input_group(
            model,
            X_train,
            y_train,
            n_classes,
            lr=lr,
            C=C,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            batch_size=batch_size,
            rho_init=rho_init,
            device=device,
            use_ratio_norm=use_ratio_norm,
            use_admm=use_admm,
            optimizer_type=optimizer_type,
            use_early_stopping=use_early_stopping,
            patience=patience,
            val_split=val_split,
            n_features=X_train.shape[1],
        )
        return

    N = len(X_train)
    n_features = X_train.shape[1]

    # Validation split for early stopping
    if use_early_stopping and val_split > 0:
        n_val = int(len(X_train) * val_split)
        idx = np.random.permutation(len(X_train))
        X_val, y_val = X_train[idx[:n_val]], y_train[idx[:n_val]]
        X_train, y_train = X_train[idx[n_val:]], y_train[idx[n_val:]]
    else:
        X_val, y_val = None, None

    dataset = TrainingSet(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if X_val is not None:
        val_dataset = TrainingSet(X_val, y_val)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
    else:
        val_loader = None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Optimizer choice
    if optimizer_type == "adagrad":
        opt_class = torch.optim.Adagrad
    else:
        opt_class = torch.optim.Adam

    optimizer = opt_class(
        list(model.parameters()),
        lr=lr,
        weight_decay=1e-3,
    )

    criterion = (
        torch.nn.BCEWithLogitsLoss() if n_classes <= 2 else torch.nn.CrossEntropyLoss()
    )

    convergence_log = []
    best_val_loss = float("inf")
    best_state_dict = None
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            # Use .float()/.long() methods for proper dtype conversion
            batch_X = batch_X.float().to(device)
            batch_y = (
                batch_y.float().to(device)
                if n_classes <= 2
                else batch_y.long().to(device)
            )

            optimizer.zero_grad()
            logits = model(batch_X)

            if n_classes <= 2:
                batch_y = batch_y.view(-1, 1)
                loss = criterion(logits, batch_y)
            else:
                loss = criterion(logits, batch_y.squeeze().long())

            loss.backward()
            optimizer.step()

        # Validation and early stopping
        if use_early_stopping and val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.float().to(device)
                    batch_y = (
                        batch_y.float().to(device)
                        if n_classes <= 2
                        else batch_y.long().to(device)
                    )
                    logits = model(batch_X)
                    if n_classes <= 2:
                        batch_y = batch_y.view(-1, 1)
                    val_loss += criterion(logits, batch_y).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    break

    # Restore best model
    if use_early_stopping and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    model.convergence_log = convergence_log


def _extract_feature_importance(
    model,
    X_train: np.ndarray,
    readout: str = "effective_gate",
) -> np.ndarray:
    """Feature importance extraction.

    For GatedFeatureSelectionMLP: returns |gate| values directly.
    For theory-aligned models, ``effective_gate`` is the primary readout and
    ``raw_gate`` is available only as a factorization diagnostic.
    For AdaptiveFeatureSelector/MLP: uses get_gate_values() method.
    For FeatureSelectionMLP: WANDA-style score[j] = sum_i |W1[i, j]| * ||a_j||_2"""
    if hasattr(model, "get_selection_scores"):
        scores = model.get_selection_scores(readout)
        return scores.detach().abs().cpu().numpy()

    if readout != "effective_gate":
        raise ValueError(
            "raw_gate readout requires a model with get_selection_scores()"
        )

    if hasattr(model, "get_feature_selection_scores"):
        scores = model.get_feature_selection_scores()
        return scores.detach().abs().cpu().numpy()

    # If model has get_gate_values method (AdaptiveFeatureSelector, AdaptiveFeatureSelectionMLP)
    if hasattr(model, "get_gate_values"):
        try:
            x_sample = torch.as_tensor(
                X_train[: min(1024, len(X_train))], dtype=torch.float32
            )
            gate_values = model.get_gate_values(x_sample)
        except TypeError:
            gate_values = model.get_gate_values()
        return gate_values.detach().abs().cpu().numpy()

    # If model has a learnable gate, use it directly
    if hasattr(model, "gate"):
        if getattr(model, "bounded_gate", False):
            return torch.sigmoid(model.gate.data).cpu().numpy()
        return model.gate.data.abs().cpu().numpy()
    # Capture activation at the first Linear layer
    first_linear = model.first_linear
    activation = None

    def hook_fn(module, inp, out):
        nonlocal activation
        activation = inp[0].detach()

    handle = first_linear.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        x_sample = torch.FloatTensor(X_train[: min(512, len(X_train))])
        # Move to same device as model
        device = next(model.parameters()).device
        model(x_sample.to(device))
    handle.remove()

    W = first_linear.weight.data  # (out_features, in_features)
    act_norm = torch.norm(activation, p=2, dim=0)  # (in_features,)

    # WANDA score per input feature: sum over output neurons of |W_ij| * ||a_j||
    wanda_per_feature = (torch.abs(W) * act_norm.unsqueeze(0)).sum(
        dim=0
    )  # (in_features,)
    return wanda_per_feature.cpu().numpy()


def _select_model_features(
    model: nn.Module,
    k: int,
    *,
    readout: str = "effective_gate",
    zero_tol: float = 0.0,
) -> tuple[list[int], dict[str, Any]]:
    """Apply the shared deterministic selection contract to a model readout."""
    if hasattr(model, "get_selection_scores"):
        scores = model.get_selection_scores(readout)
    else:
        if readout != "effective_gate":
            raise ValueError(
                "raw_gate readout requires a model with get_selection_scores()"
            )
        if not hasattr(model, "get_feature_selection_scores"):
            raise TypeError("model does not expose feature-selection scores")
        scores = model.get_feature_selection_scores()
    selected, metadata = deterministic_topk(
        scores,
        k,
        zero_tol=zero_tol,
        readout=readout,
    )
    return [int(index) for index in selected.detach().cpu().tolist()], metadata


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------


def _predict_proba(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_t = torch.FloatTensor(X).to(device)
        logits = model(x_t)
        if n_classes <= 2:
            proba = torch.sigmoid(logits).squeeze().cpu().numpy()
        else:
            proba = torch.softmax(logits, dim=1).cpu().numpy()
    return proba


# ---------------------------------------------------------------------------
# Public API: adaptive input-group framework
# ---------------------------------------------------------------------------


def run_adaptive_input_group(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    *,
    model_type: str = "adaptive_gate",
    hp_overrides: Optional[dict] = None,
    seed: Optional[int] = None,
):
    """Train an adaptive feature-selection model and return FS results.

    Returns:
        (y_train_hat, y_hat, scores, scores2)
    """
    if not ADAPTIVE_MODELS_AVAILABLE:
        raise RuntimeError(
            "Adaptive models are unavailable. Ensure adaptive_architecture.py is importable."
        )

    n_features = X_train.shape[1]
    hp = {
        "lr": 0.005,
        "C": 0.05,
        "epochs": 500,
        "warmup_epochs": 120,
        "batch_size": 64,
        "rho_init": 200.0,
        "optimizer_type": "adam",
        "use_early_stopping": True,
        "patience": 66,
        "val_split": 0.2,
        # model kwargs
        "feat_drop": 0.6,
        "dropout": 0.043 if model_type == "adaptive_mlp" else 0.0,
        "gaussian_noise": 0.0,
    }
    if hp_overrides:
        hp.update(hp_overrides)

    scaler = _Scaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    _seed = seed if seed is not None else hash(tuple(y_train[:20].tolist())) % 2**31
    torch.manual_seed(_seed)

    model_kwargs = {
        "feat_drop": hp.get("feat_drop", 0.6),
        "dropout": hp.get("dropout", 0.0),
        "gaussian_noise": hp.get("gaussian_noise", 0.0),
    }
    model = create_adaptive_model(
        n_features=n_features,
        n_classes=n_classes,
        model_type=model_type,
        **model_kwargs,
    )

    _train_adaptive_input_group(
        model,
        X_train_s,
        y_train,
        n_classes,
        lr=hp["lr"],
        C=hp["C"],
        epochs=hp["epochs"],
        warmup_epochs=hp["warmup_epochs"],
        batch_size=hp["batch_size"],
        rho_init=hp["rho_init"],
        optimizer_type=hp["optimizer_type"],
        use_early_stopping=hp["use_early_stopping"],
        patience=hp["patience"],
        val_split=hp["val_split"],
    )

    scores = _extract_feature_importance(model, X_train_s)
    scores2 = scores
    y_train_hat = _predict_proba(model, X_train_s, n_classes)
    y_hat = _predict_proba(model, X_test_s, n_classes)
    return y_train_hat, y_hat, scores, scores2


# ---------------------------------------------------------------------------
# Public API: admm_input_group only
# ---------------------------------------------------------------------------


def run_admm_input_group(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    *,
    use_ratio_norm: bool = True,
    use_admm: bool = True,
    bounded_gate: bool = False,
    hp_overrides: Optional[dict] = None,
    seed: Optional[int] = None,
):
    """Train an MLP with admm_input_group and return FS results.

    Implements three strategies:
      1. Warm start (ADMM only): initialise first layer from L1-logistic
      2. Standardisation: z-score input features
      3. Adaptive rho: Boyd's primal-dual balancing (inside training loop)

    Returns:
        (y_train_hat, y_hat, scores, scores2)
    """
    # Default hyperparameters for admm_input_group
    # Updated 2026-03-11:
    #   - feat_drop tuned from 0.7 to 0.6 (improves ring avg_best_k from 0.12 to 0.31)
    #   - gaussian_noise disabled by default (can be enabled via hp_overrides)
    #   - rho_init scaled by feature dimension (avoids over-constraint in low-dim tasks)
    n_features = X_train.shape[1]

    # FIX: Scale rho_init with feature dimension
    # Low-dimensional tasks (m < 256) need smaller rho to avoid over-constraint
    # High-dimensional tasks (m >= 256) can use larger rho for faster convergence
    if n_features < 64:
        rho_init_default = 20.0  # Very small for very low dimensions
    elif n_features < 256:
        rho_init_default = 50.0  # Small for low dimensions
    elif n_features < 512:
        rho_init_default = 100.0  # Medium for medium dimensions
    else:
        rho_init_default = 200.0  # Original value for high dimensions

    hp = {
        "lr": 0.005,
        "C": 0.05,
        "epochs": 500,
        "warmup_epochs": 120,
        "batch_size": 64,
        "latent_size": DEFAULT_SADMM_LATENT_SIZE,
        "n_hidden_layers": DEFAULT_SADMM_HIDDEN_LAYERS,
        "dropout": DEFAULT_SADMM_DROPOUT,
        "activation": DEFAULT_SADMM_ACTIVATION,
        "feat_drop": 0.6,  # Tuned from 0.7
        "gaussian_noise": 0.0,  # Optional; set >0 to enable input noise
        "column_normalize_first_layer": False,
        "rho_init": rho_init_default,  # Scaled by dimension
        "warm_start": False,
        "optimizer_type": "adam",
        "use_early_stopping": False,
        "patience": 66,
        "val_split": 0.2,
        "write_z_to_gate": False,
        "ratio_norm_update": "formula_z_den",
    }
    if hp_overrides:
        hp.update(hp_overrides)

    # ---- Strategy 2: Standardise features ----
    scaler = _Scaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---- Model selection ----
    # Fix torch seed per call for reproducibility across CV folds
    _seed = seed if seed is not None else hash(tuple(y_train[:20].tolist())) % 2**31
    torch.manual_seed(_seed)
    _use_ratio = use_ratio_norm
    model = GatedFeatureSelectionMLP(
        input_size=n_features,
        n_classes=n_classes,
        latent_size=hp["latent_size"],
        n_hidden_layers=hp["n_hidden_layers"],
        gaussian_noise=hp.get("gaussian_noise", 0.0),
        dropout=hp["dropout"],
        feat_drop=hp.get("feat_drop", 0.6),
        activation=hp["activation"],
        bounded_gate=bounded_gate,
        column_normalize_first_layer=hp.get("column_normalize_first_layer", False),
    )

    _train_input_group(
        model,
        X_train_s,
        y_train,
        n_classes,
        lr=hp["lr"],
        C=hp["C"],
        epochs=hp["epochs"],
        warmup_epochs=hp.get("warmup_epochs", 120),
        batch_size=hp.get("batch_size", 64),
        rho_init=hp.get("rho_init", 200.0),
        use_ratio_norm=_use_ratio,
        ratio_norm_update=hp.get("ratio_norm_update", "formula_z_den"),
        use_admm=use_admm,
        write_z_to_gate=hp.get("write_z_to_gate", False),
        n_features=n_features,  # Pass for rho scaling
        optimizer_type=hp.get("optimizer_type", "adam"),
        use_early_stopping=hp.get("use_early_stopping", False),
        patience=hp.get("patience", 66),
        val_split=hp.get("val_split", 0.2),
    )

    scores = _extract_feature_importance(model, X_train_s)
    scores2 = scores

    y_train_hat = _predict_proba(model, X_train_s, n_classes)
    y_hat = _predict_proba(model, X_test_s, n_classes)

    return y_train_hat, y_hat, scores, scores2


# ---------------------------------------------------------------------------
# Default hyperparameters (for backward compatibility with old _HPARAMS)
# ---------------------------------------------------------------------------

DEFAULT_HPARAMS = {
    "admm_input_group": {
        "lr": 0.005,
        "C": 0.05,
        "epochs": 500,
        "warmup_epochs": 120,
        "feat_drop": 0.6,
        "rho_init": 200.0,  # Will be scaled by dimension in run_admm_input_group
        "warm_start": False,
    },
    # Aliases for backward compatibility
    "lasso_input_group": {
        "lr": 0.005,
        "C": 0.05,
        "epochs": 500,
        "warmup_epochs": 120,
        "feat_drop": 0.6,
        "rho_init": 200.0,
        "warm_start": False,
    },
}

# Alias for backward compatibility with old admm_lasso_wrapper.py
_HPARAMS = DEFAULT_HPARAMS


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------

__all__ = [
    "FeatureSelectionMLP",
    "GatedFeatureSelectionMLP",
    "GatedWeightNormMLP",
    "GatedUnitGateMLP",
    "GatedExplicitScaleMLP",
    "ColumnNormalizedLinear",
    "_Scaler",
    "_train_input_group",
    "_train_adaptive_input_group",  # For adaptive models
    "_extract_feature_importance",
    "_select_model_features",
    "_predict_proba",
    "run_adaptive_input_group",
    "run_admm_input_group",
    "DEFAULT_HPARAMS",
    "_HPARAMS",  # Backward compatibility alias
    # Adaptive models (if available)
    "ADAPTIVE_MODELS_AVAILABLE",
    # Re-export utilities for convenience
    "soft_thresholding",
    "safe_norm",
    "safe_cbrt",
    "solve_cubic_ratio_norm",
    "solve_cubic_denominator_copy",
    "solve_cubic_paper",
    "canonical_signed_ratio_prox",
    "deterministic_topk",
]
