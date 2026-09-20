# -*- coding: utf-8 -*-
"""
Gradual ADMM with Pruning + Re-weighting.

Implements fair comparison between soft and hard pruning by:
1. Gradually increasing ADMM penalty C
2. Optionally pruning weak features (soft mask or hard delete)
3. Optionally re-weighting surviving gates to maintain energy

Variants:
- gradual_none: Only increase C, no pruning
- gradual_soft: Soft mask + re-weight
- gradual_soft_no_rw: Soft mask, no re-weight
- gradual_hard: Hard delete + re-weight
- gradual_hard_no_rw: Hard delete, no re-weight
"""

from __future__ import annotations

import copy
from typing import Callable, Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .admm_utils import deterministic_topk


def compute_adaptive_C_schedule(n_features: int, k: int) -> List[float]:
    """
    Compute C_schedule scaled to n_features for consistent per-feature pressure.

    Key insight: Per-feature ADMM pressure = C / n_features.
    For low-dim (128) vs high-dim (8384), we need proportionally larger C
    to maintain the same pressure per feature.

    Baseline (n=128): C_schedule = [0.1, 0.2, 0.3, 0.4, 0.5] (verified working)
    Scale factor = n_features / 128

    Args:
        n_features: Number of input features
        k: Number of features to select (unused but kept for future extensions)

    Returns:
        C_schedule: Scaled penalty values
    """
    # Baseline: verified working for 128 features (Ring best-k=1.00)
    baseline_n = 128
    baseline_C = [0.1, 0.2, 0.3, 0.4, 0.5]  # Updated: higher initial C

    # Scale factor: maintain same per-feature pressure
    scale_factor = n_features / baseline_n

    # Apply scaling
    scaled_C = [c * scale_factor for c in baseline_C]

    # Cap at reasonable max to prevent explosion
    # For very high dimensions, we don't want C > 10 (would kill all gates)
    max_C = 10.0
    scaled_C = [min(c, max_C) for c in scaled_C]

    return scaled_C


# ---------------------------------------------------------------------------
# Re-weighting Functions
# ---------------------------------------------------------------------------


def reweight_gates(
    gate: torch.Tensor,
    alive_mask: torch.Tensor,
    target_sum: float,
    mode: str = "scale",
) -> torch.Tensor:
    """
    Re-weight surviving gates to maintain total gate energy.

    Args:
        gate: Current gate values
        alive_mask: Boolean mask for alive features
        target_sum: Target sum of absolute gate values
        mode: Re-weighting mode
            - "scale": Simple scaling to target
            - "preserve_ratio": Preserve relative importance ratios
            - "gradual": Gradual scaling (cap at max_scale)

    Returns:
        new_gate: Re-weighted gate values
    """
    current_sum = (gate.abs() * alive_mask).sum().item()

    if current_sum < 1e-8:
        # All gates are zero, nothing to re-weight
        return gate * alive_mask

    if mode == "scale":
        scale = target_sum / current_sum
        new_gate = gate * alive_mask * scale

    elif mode == "preserve_ratio":
        # Preserve relative importance among alive features
        alive_gate = gate * alive_mask
        ratios = alive_gate.abs() / alive_gate.abs().sum()
        new_gate = ratios * target_sum * (alive_gate.sign())

    elif mode == "gradual":
        # Gradual scaling to avoid sudden jumps
        max_scale = 1.5  # Cap at 1.5x increase
        scale = min(max_scale, target_sum / current_sum)
        new_gate = gate * alive_mask * scale

    else:
        raise ValueError(f"Unknown reweight mode: {mode}")

    return new_gate


def compute_gate_energy(gate: torch.Tensor) -> Dict[str, float]:
    """Compute gate energy statistics."""
    gate_abs = gate.abs()
    return {
        "total_sum": gate_abs.sum().item(),
        "max_val": gate_abs.max().item(),
        "min_val": gate_abs.min().item(),
        "mean_val": gate_abs.mean().item(),
        "alive_count": (gate_abs > 1e-4).sum().item(),
    }


# ---------------------------------------------------------------------------
# Gradual ADMM Training with Pruning
# ---------------------------------------------------------------------------


def train_one_phase_admm(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    C: float,
    C_schedule_per_epoch: Optional[List[float]] = None,
    epochs: int = 50,
    warmup_epochs: int = 0,
    lr: float = 0.005,
    weight_decay: float = 1e-3,
    batch_size: int = 64,
    g_inner_epochs: int = 1,
    device: str = "cuda",
    use_ratio_norm: bool = True,
    use_admm: bool = True,
    uniform_penalty: bool = False,
    rho_init: float = 200.0,
    write_z_to_gate: bool = False,
    initial_admm_state: Optional[dict] = None,
    min_z_support: int = 0,
    admm_weighted_gate: bool = False,
    ratio_norm_update: str = "formula_z_den",
    score_update_mode: str = "epoch",
    ratio_norm_dim_scale: str = "none",
    ratio_norm_p: float = 1.0,
    ratio_norm_q: float = 2.0,
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
    diagnostic_ground_truth: Optional[List[int]] = None,
    diagnostic_k: int = 0,
    lr_scheduler: str = "none",
    cosine_eta_min: Optional[float] = None,
    cosine_decay_fraction: float = 1.0,
    rho_follow_lr: bool = False,
    strict_neural_block_mode: str = "joint_minibatch",
    strict_kkt_check_interval: int = 0,
    strict_kkt_tolerance: float = 1e-2,
    strict_kkt_absolute_tolerance: float = 0.0,
    strict_kkt_patience: int = 3,
    legacy_observer=None,
) -> Dict:
    """
    Train one phase with proper ADMM using existing _train_input_group.

    Uses the correct ADMM implementation from admm_input_group_wrapper.py.

    Args:
        epochs: Number of ADMM epochs AFTER warmup (not total epochs)
        warmup_epochs: Number of warmup epochs before ADMM
        rho_init: Initial ADMM penalty parameter (default 200.0)
    """
    # Import the correct ADMM training function
    from .admm_input_group_wrapper import _train_input_group

    model.to(device)

    # Total epochs = warmup + ADMM epochs
    # _train_input_group expects epochs to be total, not just ADMM phase
    total_epochs = warmup_epochs + epochs

    # Call the proper ADMM training
    _train_input_group(
        model,
        X_train,
        y_train,
        n_classes,
        lr=lr,
        weight_decay=weight_decay,
        C=C,
        C_schedule_per_epoch=C_schedule_per_epoch,
        epochs=total_epochs,  # Total epochs (warmup + ADMM)
        warmup_epochs=warmup_epochs,  # Warmup phase length
        batch_size=batch_size,
        g_inner_epochs=g_inner_epochs,
        device=device,
        use_ratio_norm=use_ratio_norm,
        ratio_norm_update=ratio_norm_update,
        ratio_norm_dim_scale=ratio_norm_dim_scale,
        ratio_norm_p=ratio_norm_p,
        ratio_norm_q=ratio_norm_q,
        use_admm=use_admm,
        uniform_penalty=uniform_penalty,
        write_z_to_gate=write_z_to_gate,
        optimizer_type="adam",  # Adam for better convergence
        rho_init=rho_init,
        initial_admm_state=initial_admm_state,
        min_z_support=min_z_support,
        admm_weighted_gate=admm_weighted_gate,
        adaptive_rho=adaptive_rho,
        adaptive_rho_mu=adaptive_rho_mu,
        adaptive_rho_tau=adaptive_rho_tau,
        adaptive_rho_min=adaptive_rho_min,
        adaptive_rho_max=adaptive_rho_max,
        score_update_mode=score_update_mode,
        auto_C_gamma=auto_C_gamma,
        auto_C_support=auto_C_support,
        adaptive_C=adaptive_C,
        adaptive_C_target=adaptive_C_target,
        adaptive_C_update_interval=adaptive_C_update_interval,
        adaptive_C_low_ratio=adaptive_C_low_ratio,
        adaptive_C_high_ratio=adaptive_C_high_ratio,
        adaptive_C_mult=adaptive_C_mult,
        adaptive_C_min=adaptive_C_min,
        adaptive_C_max=adaptive_C_max,
        dynamic_C_mode=dynamic_C_mode,
        dynamic_C_update_interval=dynamic_C_update_interval,
        dynamic_C_ema=dynamic_C_ema,
        dynamic_C_min_factor=dynamic_C_min_factor,
        dynamic_C_max_factor=dynamic_C_max_factor,
        dynamic_C_kappa=dynamic_C_kappa,
        dynamic_C_quantile=dynamic_C_quantile,
        dynamic_C_continuation_gamma=dynamic_C_continuation_gamma,
        dynamic_C_stability_patience=dynamic_C_stability_patience,
        dynamic_C_support_tol=dynamic_C_support_tol,
        dynamic_C_residual_tol=dynamic_C_residual_tol,
        dynamic_C_loss_tol=dynamic_C_loss_tol,
        dynamic_C_cap_seed=dynamic_C_cap_seed,
        dynamic_C_floor=dynamic_C_floor,
        dynamic_C_support_parent_low=dynamic_C_support_parent_low,
        dynamic_C_support_parent_high=dynamic_C_support_parent_high,
        dynamic_C_support_term_low=dynamic_C_support_term_low,
        dynamic_C_support_term_high=dynamic_C_support_term_high,
        dynamic_C_support_up_mult=dynamic_C_support_up_mult,
        dynamic_C_support_down_mult=dynamic_C_support_down_mult,
        group_sparse_powers=group_sparse_powers,
        group_sparse_mode=group_sparse_mode,
        group_parent_support=group_parent_support,
        diagnostic_ground_truth=diagnostic_ground_truth,
        diagnostic_k=diagnostic_k,
        lr_scheduler=lr_scheduler,
        cosine_eta_min=cosine_eta_min,
        cosine_decay_fraction=cosine_decay_fraction,
        rho_follow_lr=rho_follow_lr,
        strict_neural_block_mode=strict_neural_block_mode,
        strict_kkt_check_interval=strict_kkt_check_interval,
        strict_kkt_tolerance=strict_kkt_tolerance,
        strict_kkt_absolute_tolerance=strict_kkt_absolute_tolerance,
        strict_kkt_patience=strict_kkt_patience,
        legacy_observer=legacy_observer,
    )

    # Collect gate statistics
    history = {"losses": [], "gate_stats": []}

    if hasattr(model, "gate"):
        gate_stats = compute_gate_energy(model.gate.detach())
        history["gate_stats"].append(gate_stats)

    return history


def gradual_admm_with_pruning(
    model_class: Callable,
    model_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    ground_truth: List[int],
    k: int,
    C_schedule: Optional[List[float]] = None,  # None = auto-compute
    global_C_schedule_per_epoch: Optional[List[float]] = None,
    prune_threshold: float = 0.01,
    prune_mode: str = "soft",
    reweight: bool = True,
    reweight_mode: str = "scale",
    epochs_per_phase: int = 50,
    warmup_epochs: int = 20,
    warmup_per_phase: bool = True,  # NEW: warmup in all phases (default True after ablation)
    auto_scale_C: bool = True,  # Auto-scale C based on n_features
    use_ratio_norm: bool = True,
    use_admm: bool = True,
    uniform_penalty: bool = False,
    rho_init: float = 200.0,  # Initial ADMM penalty parameter
    write_z_to_gate: bool = False,  # Experimental: write ADMM z-step result back into gate
    soft_prune_update: str = "threshold_gate",
    cross_phase_state: Optional[Dict[str, bool]] = None,
    min_z_support: Optional[int] = None,
    admm_weighted_gate: bool = False,
    ratio_norm_update: str = "formula_z_den",
    score_update_mode: str = "epoch",
    ratio_norm_dim_scale: str = "none",
    ratio_norm_p: float = 1.0,
    ratio_norm_q: float = 2.0,
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
    dynamic_C_support_parent_low: int = 0,
    dynamic_C_support_parent_high: int = 0,
    dynamic_C_support_term_low: int = 0,
    dynamic_C_support_term_high: int = 0,
    dynamic_C_support_up_mult: float = 1.25,
    dynamic_C_support_down_mult: float = 1.5,
    group_sparse_powers: Optional[np.ndarray] = None,
    group_sparse_mode: str = "none",
    group_parent_support: int = 0,
    lr: float = 0.005,
    weight_decay: float = 1e-3,
    g_inner_epochs: int = 1,
    lr_scheduler: str = "none",
    cosine_eta_min: Optional[float] = None,
    cosine_decay_fraction: float = 1.0,
    rho_follow_lr: bool = False,
    strict_neural_block_mode: str = "joint_minibatch",
    strict_kkt_check_interval: int = 0,
    strict_kkt_tolerance: float = 1e-2,
    strict_kkt_absolute_tolerance: float = 0.0,
    strict_kkt_patience: int = 3,
    device: str = "cuda",
    verbose: bool = True,
    legacy_observer=None,
) -> Tuple[List[int], Dict]:
    """
    Gradual ADMM with optional pruning and re-weighting.

    Args:
        model_class: Class to instantiate model
        model_kwargs: kwargs for model constructor
        ground_truth: List of true relevant feature indices
        k: Number of features to select
        C_schedule: Gradually increasing penalty values (None=auto-compute)
        prune_threshold: Threshold for pruning weak gates
        prune_mode: "none", "soft", or "hard"
        reweight: Whether to re-weight surviving gates
        reweight_mode: "scale", "preserve_ratio", or "gradual"
        epochs_per_phase: ADMM epochs per phase
        warmup_epochs: Warmup epochs (per phase if warmup_per_phase=True)
        warmup_per_phase: If True, warmup before each phase; if False, only Phase 0
        auto_scale_C: If True and C_schedule=None, auto-compute based on n_features
        soft_prune_update: Phase-end soft-pruning update. "threshold_gate" is
            the current method. "replace_with_z" copies final ADMM z into g.
            "mask_by_z" keeps g values but zeros coordinates where z == 0.
            "mask_by_z_final" uses "threshold_gate" before the last phase and
            "mask_by_z" at the last phase only.

    Returns:
        selected_features: List of selected feature indices
        history: Dict with per-phase statistics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_features = X_train.shape[1]

    if adaptive_C:
        if C_schedule is None:
            C_schedule = [0.1]
        auto_scale_C = False
    # Auto-compute C_schedule if not provided
    elif C_schedule is None:
        C_schedule = compute_adaptive_C_schedule(n_features, k)
        if verbose:
            print(f"Auto C_schedule for n_features={n_features}, k={k}: {C_schedule}")
    elif auto_scale_C:
        # Scale provided C_schedule based on n_features
        baseline_n = 128
        scale_factor = n_features / baseline_n
        max_C = 5.0
        C_schedule = [min(c * scale_factor, max_C) for c in C_schedule]
        if verbose:
            print(f"Scaled C_schedule: {C_schedule}")
    if global_C_schedule_per_epoch is not None:
        expected_global_epochs = len(C_schedule) * int(epochs_per_phase)
        if len(global_C_schedule_per_epoch) != expected_global_epochs:
            raise ValueError(
                "global_C_schedule_per_epoch must contain exactly one value per "
                f"ADMM epoch across all phases; expected {expected_global_epochs}, "
                f"got {len(global_C_schedule_per_epoch)}"
            )
    if (
        dynamic_C_cap_seed is None
        and C_schedule
        and dynamic_C_mode not in {"support_band", "pre_threshold_band"}
    ):
        dynamic_C_cap_seed = float(C_schedule[0])

    current_features = list(range(n_features))
    cross_phase_state = cross_phase_state or {}
    keep_gate = cross_phase_state.get("keep_gate", True)
    keep_weights = cross_phase_state.get("keep_weights", True)
    keep_z = cross_phase_state.get("keep_z", False)
    keep_u = cross_phase_state.get("keep_u", False)
    keep_adam = cross_phase_state.get("keep_adam", False)
    reset_nonzero_gate = cross_phase_state.get("reset_nonzero_gate", False)

    valid_soft_prune_updates = {
        "threshold_gate",
        "replace_with_z",
        "mask_by_z",
        "mask_by_z_final",
    }
    if soft_prune_update not in valid_soft_prune_updates:
        raise ValueError(
            f"Unknown soft_prune_update={soft_prune_update!r}. "
            f"Expected one of {sorted(valid_soft_prune_updates)}"
        )
    if min_z_support is None:
        min_z_support = 0

    # Initialize model
    model = model_class(input_size=n_features, n_classes=n_classes, **model_kwargs)
    model.to(device)

    # Track initial gate energy for re-weighting
    if hasattr(model, "gate"):
        initial_gate_sum = model.gate.abs().sum().item()
    else:
        initial_gate_sum = 1.0

    history = {
        "phases": [],
        "prune_mode": prune_mode,
        "reweight": reweight,
        "use_ratio_norm": use_ratio_norm,
        "use_admm": use_admm,
        "uniform_penalty": uniform_penalty,
        "global_C_schedule_per_epoch": (
            None
            if global_C_schedule_per_epoch is None
            else [float(v) for v in global_C_schedule_per_epoch]
        ),
        "write_z_to_gate": write_z_to_gate,
        "soft_prune_update": soft_prune_update,
        "min_z_support": min_z_support,
        "admm_weighted_gate": admm_weighted_gate,
        "admm_constraint_mode": getattr(model, "admm_constraint_mode", "gate"),
        "ratio_norm_update": ratio_norm_update,
        "score_update_mode": score_update_mode,
        "ratio_norm_dim_scale": ratio_norm_dim_scale,
        "ratio_norm_p": float(ratio_norm_p),
        "ratio_norm_q": float(ratio_norm_q),
        "adaptive_rho": adaptive_rho,
        "weight_decay": float(weight_decay),
        "auto_C_gamma": auto_C_gamma,
        "auto_C_support": auto_C_support,
        "adaptive_C": adaptive_C,
        "adaptive_C_target": adaptive_C_target,
        "adaptive_C_update_interval": adaptive_C_update_interval,
        "adaptive_C_low_ratio": adaptive_C_low_ratio,
        "adaptive_C_high_ratio": adaptive_C_high_ratio,
        "adaptive_C_mult": adaptive_C_mult,
        "adaptive_C_min": adaptive_C_min,
        "adaptive_C_max": adaptive_C_max,
        "dynamic_C_mode": dynamic_C_mode,
        "dynamic_C_update_interval": dynamic_C_update_interval,
        "dynamic_C_ema": dynamic_C_ema,
        "dynamic_C_min_factor": dynamic_C_min_factor,
        "dynamic_C_max_factor": dynamic_C_max_factor,
        "dynamic_C_kappa": dynamic_C_kappa,
        "dynamic_C_quantile": dynamic_C_quantile,
        "dynamic_C_continuation_gamma": dynamic_C_continuation_gamma,
        "dynamic_C_stability_patience": dynamic_C_stability_patience,
        "dynamic_C_support_tol": dynamic_C_support_tol,
        "dynamic_C_residual_tol": dynamic_C_residual_tol,
        "dynamic_C_loss_tol": dynamic_C_loss_tol,
        "dynamic_C_cap_seed": dynamic_C_cap_seed,
        "dynamic_C_support_parent_low": dynamic_C_support_parent_low,
        "dynamic_C_support_parent_high": dynamic_C_support_parent_high,
        "dynamic_C_support_term_low": dynamic_C_support_term_low,
        "dynamic_C_support_term_high": dynamic_C_support_term_high,
        "dynamic_C_support_up_mult": dynamic_C_support_up_mult,
        "dynamic_C_support_down_mult": dynamic_C_support_down_mult,
        "dynamic_C_replaces_C_schedule": dynamic_C_mode in {
            "kkt_grad",
            "sqrt_loss",
            "event_continuation",
            "lambda_max_const",
            "lambda_max_phase_const",
            "root_n_lambda_max_const",
            "root_n_lambda_max_phase_const",
            "root_n_lambda_max_phase_monotone_after2",
            "score_quantile_lambda_phase_const",
            "score_quantile_lambda_dynamic",
            "parent_k_lambda_const",
            "parent_k_phase_lambda_const",
        },
        "group_sparse_mode": group_sparse_mode,
        "group_parent_support": group_parent_support,
        "lr": lr,
        "lr_scheduler": lr_scheduler,
        "cosine_eta_min": cosine_eta_min,
        "rho_follow_lr": rho_follow_lr,
        "cross_phase_state": {
            "keep_gate": keep_gate,
            "keep_weights": keep_weights,
            "keep_z": keep_z,
            "keep_u": keep_u,
            "keep_adam": keep_adam,
            "reset_nonzero_gate": reset_nonzero_gate,
        },
        "initial_gate_sum": initial_gate_sum,
    }

    # Target for re-weighting: maintain small proportion of original energy
    # Lower target (0.2) prevents pushing gates back up after pruning
    target_ratio = 0.2  # Target 20% of original energy
    last_admm_state = None

    def _add_topk_floor(mask: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """Ensure the final soft mask has at least k ranked coordinates."""
        if k <= 0 or score.numel() == 0:
            return mask
        n_keep = min(k, score.numel())
        topk_idx, _ = deterministic_topk(
            score,
            n_keep,
            readout="effective_gate",
        )
        floored = mask.clone()
        floored[topk_idx] = True
        return floored

    def _gate_selection_score(current_model: nn.Module) -> torch.Tensor:
        if hasattr(current_model, "get_selection_scores"):
            return current_model.get_selection_scores("effective_gate").detach().to(
                device=current_model.gate.device,
                dtype=current_model.gate.dtype,
            )
        if hasattr(current_model, "get_feature_selection_scores"):
            return current_model.get_feature_selection_scores().detach().to(
                device=current_model.gate.device,
                dtype=current_model.gate.dtype,
            )
        gate_param = current_model.gate.detach()
        if hasattr(current_model, "gate_from_parameter"):
            gate_value = current_model.gate_from_parameter(gate_param)
        else:
            gate_value = gate_param
        if admm_weighted_gate and hasattr(current_model, "first_weight_norm"):
            scale = current_model.first_weight_norm.detach().to(
                device=gate_value.device, dtype=gate_value.dtype
            )
            return (scale * gate_value).abs()
        return gate_value.abs()

    schedule_free_dynamic_C_modes = {
        "kkt_grad",
        "sqrt_loss",
        "event_continuation",
        "lambda_max_const",
        "root_n_lambda_max_const",
        "score_quantile_lambda_dynamic",
        "parent_k_lambda_const",
    }
    single_init_dynamic_C_modes = {
        "lambda_max_const",
        "root_n_lambda_max_const",
        "parent_k_lambda_const",
    }
    phase_monotone_dynamic_C_modes = {
        "root_n_lambda_max_phase_monotone_after2",
    }
    last_dynamic_C_used: Optional[float] = None

    for phase_idx, C in enumerate(C_schedule):
        C_phase = float(C)
        if (
            dynamic_C_mode in schedule_free_dynamic_C_modes
            and last_dynamic_C_used is not None
        ):
            C_phase = float(last_dynamic_C_used)
        if verbose:
            print(f"\n=== Phase {phase_idx + 1}/{len(C_schedule)}: C={C_phase:.4f} ===")
            print(f"Features: {len(current_features)}")

        # Phase 1: Train with current C
        # For hard pruning mode, X_train is already subsetted, so use it directly
        # For other modes, subset using current_features
        if prune_mode == "hard" and phase_idx > 0:
            X_train_phase = X_train  # Already subsetted in previous phase
            X_test_phase = X_test
        else:
            X_train_phase = X_train[:, current_features]
            X_test_phase = X_test[:, current_features]

        if phase_idx > 0 and not keep_weights:
            preserved_gate = None
            if hasattr(model, "gate") and keep_gate:
                preserved_gate = model.gate.detach().clone()
            model = model_class(
                input_size=X_train_phase.shape[1],
                n_classes=n_classes,
                **model_kwargs,
            )
            model.to(device)
            if preserved_gate is not None and hasattr(model, "gate") and tuple(model.gate.shape) == tuple(preserved_gate.shape):
                model.gate.data.copy_(preserved_gate.to(device=device, dtype=model.gate.dtype))

        if phase_idx > 0 and hasattr(model, "gate") and not keep_gate:
            model.gate.data.fill_(1.0)

        if phase_idx > 0 and hasattr(model, "gate") and reset_nonzero_gate:
            nonzero_mask = model.gate.data != 0
            model.gate.data[nonzero_mask] = 1.0

        initial_admm_state = None
        if last_admm_state is not None:
            initial_admm_state = {}
            if keep_z and "zk" in last_admm_state:
                initial_admm_state["zk"] = last_admm_state["zk"]
            if keep_z and "z_den" in last_admm_state:
                initial_admm_state["z_den"] = last_admm_state["z_den"]
            if keep_z and "yk" in last_admm_state:
                initial_admm_state["yk"] = last_admm_state["yk"]
            if keep_u and "uk" in last_admm_state:
                initial_admm_state["uk"] = last_admm_state["uk"]
            if keep_u and "wk" in last_admm_state:
                initial_admm_state["wk"] = last_admm_state["wk"]
            if keep_adam and "optimizer_state_dict" in last_admm_state:
                initial_admm_state["optimizer_state_dict"] = last_admm_state["optimizer_state_dict"]
            if keep_adam:
                for state_key in (
                    "theta_optimizer_state_dict",
                    "gate_optimizer_state_dict",
                ):
                    if last_admm_state.get(state_key) is not None:
                        initial_admm_state[state_key] = last_admm_state[state_key]
            if not initial_admm_state:
                initial_admm_state = None

        phase_dynamic_C_mode = dynamic_C_mode
        if (
            dynamic_C_mode in single_init_dynamic_C_modes
            and last_dynamic_C_used is not None
        ):
            phase_dynamic_C_mode = "none"
        phase_dynamic_C_floor = None
        if (
            dynamic_C_mode in phase_monotone_dynamic_C_modes
            and phase_idx >= 2
            and last_dynamic_C_used is not None
        ):
            phase_dynamic_C_floor = float(last_dynamic_C_used)

        phase_C_schedule_per_epoch = None
        if global_C_schedule_per_epoch is not None:
            phase_start = phase_idx * int(epochs_per_phase)
            phase_end = phase_start + int(epochs_per_phase)
            phase_C_schedule_per_epoch = global_C_schedule_per_epoch[
                phase_start:phase_end
            ]
            C_phase = float(phase_C_schedule_per_epoch[0])

        phase_history = train_one_phase_admm(
            model,
            X_train_phase,
            y_train,
            n_classes,
            C=C_phase,
            C_schedule_per_epoch=phase_C_schedule_per_epoch,
            epochs=epochs_per_phase,
            warmup_epochs=warmup_epochs if (warmup_per_phase or phase_idx == 0) else 0,
            lr=lr,
            weight_decay=weight_decay,
            g_inner_epochs=g_inner_epochs,
            device=device,
            use_ratio_norm=use_ratio_norm,
            use_admm=use_admm,
            uniform_penalty=uniform_penalty,
            rho_init=rho_init,
            write_z_to_gate=write_z_to_gate,
            initial_admm_state=initial_admm_state,
            min_z_support=min_z_support,
            admm_weighted_gate=admm_weighted_gate,
            ratio_norm_update=ratio_norm_update,
            score_update_mode=score_update_mode,
            ratio_norm_dim_scale=ratio_norm_dim_scale,
            ratio_norm_p=ratio_norm_p,
            ratio_norm_q=ratio_norm_q,
            adaptive_rho=adaptive_rho,
            adaptive_rho_mu=adaptive_rho_mu,
            adaptive_rho_tau=adaptive_rho_tau,
            adaptive_rho_min=adaptive_rho_min,
            adaptive_rho_max=adaptive_rho_max,
            auto_C_gamma=auto_C_gamma,
            auto_C_support=auto_C_support,
            adaptive_C=adaptive_C,
            adaptive_C_target=adaptive_C_target,
            adaptive_C_update_interval=adaptive_C_update_interval,
            adaptive_C_low_ratio=adaptive_C_low_ratio,
            adaptive_C_high_ratio=adaptive_C_high_ratio,
            adaptive_C_mult=adaptive_C_mult,
            adaptive_C_min=adaptive_C_min,
            adaptive_C_max=adaptive_C_max,
            dynamic_C_mode=phase_dynamic_C_mode,
            dynamic_C_update_interval=dynamic_C_update_interval,
            dynamic_C_ema=dynamic_C_ema,
            dynamic_C_min_factor=dynamic_C_min_factor,
            dynamic_C_max_factor=dynamic_C_max_factor,
            dynamic_C_kappa=dynamic_C_kappa,
            dynamic_C_quantile=dynamic_C_quantile,
            dynamic_C_continuation_gamma=dynamic_C_continuation_gamma,
            dynamic_C_stability_patience=dynamic_C_stability_patience,
            dynamic_C_support_tol=dynamic_C_support_tol,
            dynamic_C_residual_tol=dynamic_C_residual_tol,
            dynamic_C_loss_tol=dynamic_C_loss_tol,
            dynamic_C_cap_seed=dynamic_C_cap_seed,
            dynamic_C_floor=phase_dynamic_C_floor,
            dynamic_C_support_parent_low=dynamic_C_support_parent_low,
            dynamic_C_support_parent_high=dynamic_C_support_parent_high,
            dynamic_C_support_term_low=dynamic_C_support_term_low,
            dynamic_C_support_term_high=dynamic_C_support_term_high,
            dynamic_C_support_up_mult=dynamic_C_support_up_mult,
            dynamic_C_support_down_mult=dynamic_C_support_down_mult,
            group_sparse_powers=group_sparse_powers,
            group_sparse_mode=group_sparse_mode,
            group_parent_support=group_parent_support,
            diagnostic_ground_truth=ground_truth,
            diagnostic_k=k,
            lr_scheduler=lr_scheduler,
            cosine_eta_min=cosine_eta_min,
            cosine_decay_fraction=cosine_decay_fraction,
            rho_follow_lr=rho_follow_lr,
            strict_neural_block_mode=strict_neural_block_mode,
            strict_kkt_check_interval=strict_kkt_check_interval,
            strict_kkt_tolerance=strict_kkt_tolerance,
            strict_kkt_absolute_tolerance=strict_kkt_absolute_tolerance,
            strict_kkt_patience=strict_kkt_patience,
            legacy_observer=legacy_observer,
        )
        last_admm_state = getattr(model, "admm_state", None)
        if (
            (
                dynamic_C_mode in schedule_free_dynamic_C_modes
                or dynamic_C_mode in phase_monotone_dynamic_C_modes
            )
            and isinstance(last_admm_state, dict)
            and "C_used" in last_admm_state
        ):
            last_dynamic_C_used = float(last_admm_state["C_used"])

        # Phase 2: Get gate values and identify weak features
        if hasattr(model, "gate"):
            gate = model.gate.detach().clone()
            gate_score = _gate_selection_score(model)
            gate_abs = gate_score

            alive_mask = gate_abs >= prune_threshold
            alive_indices = alive_mask.nonzero().squeeze().tolist()

            if isinstance(alive_indices, int):
                alive_indices = [alive_indices]

            gate_energy = compute_gate_energy(gate)

            if verbose:
                print(f"Gate energy: sum={gate_energy['total_sum']:.4f}, "
                      f"alive={gate_energy['alive_count']}")

        else:
            # No gate, use first layer weights
            w = model.first_linear.weight.detach()
            scores = w.abs().sum(dim=0)
            alive_mask = scores >= prune_threshold
            alive_indices = alive_mask.nonzero().squeeze().tolist()
            gate_energy = {"total_sum": scores.sum().item(), "alive_count": len(alive_indices)}

        # Phase 3: Prune + Re-weight (if enabled)
        phase_selection_score = None
        phase_selection_mask = None
        if prune_mode != "none" and len(alive_indices) > 0:
            if prune_mode == "soft":
                # Soft: keep all features, but write selected gates to exact zero.
                if hasattr(model, "gate"):
                    is_final_phase = phase_idx == len(C_schedule) - 1
                    phase_soft_prune_update = soft_prune_update
                    if soft_prune_update == "mask_by_z_final":
                        phase_soft_prune_update = (
                            "mask_by_z"
                            if phase_idx == len(C_schedule) - 1
                            else "threshold_gate"
                        )

                    if phase_soft_prune_update == "threshold_gate":
                        phase_end_mask = alive_mask
                        phase_selection_score = gate_abs * phase_end_mask.float()
                        phase_selection_mask = phase_end_mask
                        model.gate.data = gate * phase_end_mask.float()

                    elif phase_soft_prune_update == "replace_with_z":
                        if last_admm_state is None or "zk" not in last_admm_state:
                            raise RuntimeError("replace_with_z requires model.admm_state['zk']")
                        scaled_constraint = (
                            last_admm_state.get("admm_constraint_mode", "gate")
                            != "gate"
                        )
                        z_key = (
                            "zk_gate"
                            if scaled_constraint and "zk_gate" in last_admm_state
                            else "zk"
                        )
                        z_gate = last_admm_state[z_key].to(
                            device=model.gate.device,
                            dtype=model.gate.dtype,
                        )
                        if getattr(model, "bounded_gate", False):
                            model.gate.data.copy_(z_gate)
                        elif hasattr(model, "parameter_from_gate"):
                            model.gate.data.copy_(model.parameter_from_gate(z_gate))
                        else:
                            model.gate.data.copy_(z_gate)
                        phase_end_mask = z_gate != 0
                        phase_selection_score = _gate_selection_score(model) * phase_end_mask.float()
                        phase_selection_mask = phase_end_mask

                    elif phase_soft_prune_update == "mask_by_z":
                        if last_admm_state is None or "zk" not in last_admm_state:
                            raise RuntimeError("mask_by_z requires model.admm_state['zk']")
                        z_gate = last_admm_state["zk"].to(
                            device=gate.device,
                            dtype=gate.dtype,
                        )
                        phase_end_mask = z_gate != 0
                        phase_selection_score = gate_abs * phase_end_mask.float()
                        phase_selection_mask = phase_end_mask
                        model.gate.data = gate * phase_end_mask.float()

                    if reweight:
                        target_sum = initial_gate_sum * target_ratio
                        model.gate.data = reweight_gates(
                            model.gate.data,
                            phase_end_mask.float(),
                            target_sum,
                            mode=reweight_mode,
                        )
                    if phase_selection_score is not None:
                        model.feature_selection_score = phase_selection_score.detach().clone()
                        model.feature_selection_mask = phase_selection_mask.detach().clone()

            elif prune_mode == "hard":
                # Hard: GRADUAL pruning - delete 10% per phase
                # More gentle than one-shot 90% deletion
                if hasattr(model, "gate"):
                    n_current = len(current_features)
                    # Delete 10% each phase, but keep at least k
                    n_keep = max(k, int(n_current * 0.9))  # Keep 90% (delete 10%)

                    gate_abs_sorted = gate_abs.sort(descending=True)
                    percentile_threshold = gate_abs_sorted.values[n_keep - 1].item()
                    hard_alive_mask = gate_abs >= percentile_threshold
                    hard_alive_indices = hard_alive_mask.nonzero().squeeze().tolist()

                    if isinstance(hard_alive_indices, int):
                        hard_alive_indices = [hard_alive_indices]

                    if len(hard_alive_indices) < k:
                        # Don't prune below k
                        if verbose:
                            print(f"Skipping hard prune: {len(hard_alive_indices)} < k={k}")
                    else:
                        # Map alive_indices (local) to original feature indices
                        original_alive = [current_features[i] for i in hard_alive_indices]

                        # Create new model with reduced input size
                        new_model = model_class(
                            input_size=len(hard_alive_indices),
                            n_classes=n_classes,
                            **model_kwargs
                        )
                        new_model.to(device)

                        # Copy weights (handle shape mismatch)
                        _copy_weights_subset(model, new_model, hard_alive_indices)

                        if hasattr(new_model, "gate") and reweight:
                            target_sum = initial_gate_sum * target_ratio
                            new_model.gate.data = reweight_gates(
                                new_model.gate.data,
                                torch.ones_like(new_model.gate.data),
                                target_sum,
                                mode=reweight_mode,
                            )

                        model = new_model

                        # Update data: subset to alive features
                        X_train = X_train[:, hard_alive_indices]
                        X_test = X_test[:, hard_alive_indices]

                        # Update current_features to track original indices for evaluation
                        current_features = original_alive

                        if verbose:
                            print(f"Gradual hard prune: {n_current} -> {len(hard_alive_indices)} features (deleted 10%)")

        if hasattr(model, "gate") and phase_selection_score is None:
            model.feature_selection_score = _gate_selection_score(model).detach().clone()
            model.feature_selection_mask = model.feature_selection_score > 0

        # Compute best-k for current phase
        selection_metadata = None
        if hasattr(model, "gate"):
            gate_abs = _gate_selection_score(model)
            if prune_mode == "hard":
                # Map back to original indices
                top_k_tensor, selection_metadata = deterministic_topk(
                    gate_abs,
                    min(k, len(current_features)),
                    readout="effective_gate",
                )
                top_k_local = top_k_tensor.tolist()
                selected = [current_features[i] for i in top_k_local]
            else:
                top_k_tensor, selection_metadata = deterministic_topk(
                    gate_abs,
                    k,
                    readout="effective_gate",
                )
                selected = top_k_tensor.tolist()
        else:
            selected = current_features[:k]

        best_k = (
            len(set(selected) & set(ground_truth)) / k if k > 0 else 0.0
        )

        # Record phase results
        phase_result = {
            "phase": phase_idx,
            "C": C_phase,
            "C_schedule_entry": float(C),
            "C_epoch_start": (
                float(phase_C_schedule_per_epoch[0])
                if phase_C_schedule_per_epoch is not None
                else C_phase
            ),
            "C_epoch_end": (
                float(phase_C_schedule_per_epoch[-1])
                if phase_C_schedule_per_epoch is not None
                else float(last_admm_state.get("C_used", C_phase))
                if isinstance(last_admm_state, dict)
                else C_phase
            ),
            "C_used": float(last_admm_state.get("C_used", C_phase)) if isinstance(last_admm_state, dict) else C_phase,
            "C_schedule_per_epoch": (
                [float(v) for v in last_admm_state.get("C_schedule_per_epoch", [])]
                if isinstance(last_admm_state, dict)
                else []
            ),
            "rho": float(last_admm_state.get("rho", rho_init)) if isinstance(last_admm_state, dict) else rho_init,
            "z_support": int((last_admm_state["zk"].abs() > 0).sum().item()) if isinstance(last_admm_state, dict) and "zk" in last_admm_state else None,
            "parent_z_support": int(last_admm_state.get("parent_z_support", 0)) if isinstance(last_admm_state, dict) else None,
            "group_selected_parents": last_admm_state.get("group_selected_parents", []) if isinstance(last_admm_state, dict) else [],
            "adaptive_C_history": last_admm_state.get("adaptive_C_history", []) if isinstance(last_admm_state, dict) else [],
            "dynamic_C_history": last_admm_state.get("dynamic_C_history", []) if isinstance(last_admm_state, dict) else [],
            "n_features": len(current_features),
            "alive_count": gate_energy["alive_count"],
            "gate_sum": gate_energy["total_sum"],
            "post_prune_gate_sum": float(model.gate.detach().abs().sum().item()) if hasattr(model, "gate") else gate_energy["total_sum"],
            "post_prune_strict_zeros": int((model.gate.detach().abs() == 0).sum().item()) if hasattr(model, "gate") else 0,
            "selection_score_nonzero": int((getattr(model, "feature_selection_score", torch.zeros(0, device=device)).detach().abs() > 0).sum().item()) if hasattr(model, "gate") else 0,
            "best_k": best_k,
            "selected": selected,
            "selection": selection_metadata,
        }
        if isinstance(last_admm_state, dict):
            parent_diag_map = {
                "parent_max_g": last_admm_state.get("parent_max_g_diag"),
                "parent_max_A_times_g": last_admm_state.get("parent_max_A_times_g_diag"),
                "parent_max_z": last_admm_state.get("parent_max_z_diag"),
                "parent_weight_sum_g": last_admm_state.get("parent_weight_sum_g_diag"),
                "parent_weight_sum_A_times_g": last_admm_state.get("parent_weight_sum_A_times_g_diag"),
                "parent_weight_sum_z": last_admm_state.get("parent_weight_sum_z_diag"),
            }
            for diag_name, diag in parent_diag_map.items():
                if isinstance(diag, dict):
                    phase_result[f"{diag_name}_best_k"] = float(diag.get("best_k", 0.0))
                    phase_result[f"{diag_name}_selected"] = [
                        int(v) for v in diag.get("selected", [])
                    ]
                    phase_result[f"{diag_name}_nonzero"] = int(diag.get("nonzero", 0))
                    phase_result[f"{diag_name}_gt_ranks"] = diag.get("gt_ranks", {})
        history["phases"].append(phase_result)

        if verbose:
            print(f"best-k: {best_k:.4f}")

    # Final selection
    if hasattr(model, "gate"):
        gate_abs = _gate_selection_score(model)
        if prune_mode == "hard":
            top_k_tensor, final_selection_metadata = deterministic_topk(
                gate_abs,
                min(k, len(current_features)),
                readout="effective_gate",
            )
            top_k_local = top_k_tensor.tolist()
            selected_features = [current_features[i] for i in top_k_local]
        else:
            top_k_tensor, final_selection_metadata = deterministic_topk(
                gate_abs,
                k,
                readout="effective_gate",
            )
            selected_features = top_k_tensor.tolist()
        history["final_selection"] = final_selection_metadata
    else:
        selected_features = current_features[:k]

    return selected_features, history, model


def _copy_weights_subset(
    old_model: nn.Module,
    new_model: nn.Module,
    alive_indices: List[int],
):
    """Copy weights from old model to new model, handling shape mismatch."""
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()

    # Identify first linear layer by checking for input dimension match
    first_linear_name = None
    for name, param in old_state.items():
        if param.dim() == 2 and param.shape[1] == len(old_state.get('gate', torch.zeros(1))):
            first_linear_name = name
            break

    for name, param in new_state.items():
        if name in old_state:
            old_param = old_state[name]

            # Gate: subset of 1D tensor
            if "gate" in name and old_param.dim() == 1:
                param.copy_(old_param[alive_indices])

            # First linear layer: subset of columns (2D tensor)
            # Check by comparing input dimension to gate size
            elif name == first_linear_name and param.dim() == 2 and old_param.dim() == 2:
                # old_param shape: (out_features, in_features=old_gate_size)
                # new_param shape: (out_features, in_features=len(alive_indices))
                # Copy subset of columns corresponding to alive features
                param.copy_(old_param[:, alive_indices])

            # Other params: copy if shapes match
            elif old_param.shape == param.shape:
                param.copy_(old_param)


# ---------------------------------------------------------------------------
# Quick Run Function
# ---------------------------------------------------------------------------


def run_gradual_admm_ablation(
    datasets: List[Dict],
    model_class: Callable,
    model_kwargs: dict,
    variants: List[Dict] = None,
    seeds: List[int] = [42, 43, 44],
    device: str = "cuda",
    verbose: bool = True,
    # Training parameters
    C_schedule: List[float] = [0.01, 0.05, 0.1, 0.3, 0.5],
    epochs_per_phase: int = 50,
    warmup_epochs: int = 120,
    prune_threshold: float = 0.1,
) -> Dict:
    """
    Run ablation study on all datasets and variants.

    Args:
        datasets: List of dicts with 'name', 'X_train', 'y_train', 'X_test',
                  'y_test', 'n_classes', 'ground_truth', 'k'
        variants: List of dicts with 'name', 'prune_mode', 'reweight'
        seeds: Random seeds for reproducibility
        C_schedule: Gradually increasing penalty values
        epochs_per_phase: ADMM epochs per phase
        warmup_epochs: Warmup epochs before ADMM
        prune_threshold: Threshold for pruning weak gates

    Returns:
        results: Dict with all experiment results
    """
    if variants is None:
        variants = [
            {"name": "gradual_none", "prune_mode": "none", "reweight": False},
            {"name": "gradual_soft", "prune_mode": "soft", "reweight": True},
            {"name": "gradual_soft_no_rw", "prune_mode": "soft", "reweight": False},
            {"name": "gradual_hard", "prune_mode": "hard", "reweight": True},
            {"name": "gradual_hard_no_rw", "prune_mode": "hard", "reweight": False},
        ]

    results = {"variants": variants, "datasets": [], "runs": []}

    for dataset in datasets:
        dataset_name = dataset["name"]
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*60}")

        for variant in variants:
            variant_name = variant["name"]
            if verbose:
                print(f"\n--- Variant: {variant_name} ---")

            run_results = []

            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                selected, history = gradual_admm_with_pruning(
                    model_class=model_class,
                    model_kwargs=model_kwargs,
                    X_train=dataset["X_train"],
                    y_train=dataset["y_train"],
                    X_test=dataset["X_test"],
                    y_test=dataset["y_test"],
                    n_classes=dataset["n_classes"],
                    ground_truth=dataset["ground_truth"],
                    k=dataset["k"],
                    C_schedule=C_schedule,
                    prune_threshold=prune_threshold,
                    prune_mode=variant["prune_mode"],
                    reweight=variant["reweight"],
                    epochs_per_phase=epochs_per_phase,
                    warmup_epochs=warmup_epochs,
                    device=device,
                    verbose=verbose,
                )

                best_k = len(set(selected) & set(dataset["ground_truth"])) / dataset["k"]

                run_results.append({
                    "seed": seed,
                    "best_k": best_k,
                    "selected": selected,
                    "history": history,
                })

            # Aggregate
            mean_best_k = np.mean([r["best_k"] for r in run_results])
            std_best_k = np.std([r["best_k"] for r in run_results])

            results["runs"].append({
                "dataset": dataset_name,
                "variant": variant_name,
                "mean_best_k": mean_best_k,
                "std_best_k": std_best_k,
                "run_results": run_results,
            })

            if verbose:
                print(f"Result: best_k = {mean_best_k:.4f} ± {std_best_k:.4f}")

    # Summary table
    summary = {}
    for dataset in datasets:
        dataset_name = dataset["name"]
        summary[dataset_name] = {}
        for variant in variants:
            variant_name = variant["name"]
            for run in results["runs"]:
                if run["dataset"] == dataset_name and run["variant"] == variant_name:
                    summary[dataset_name][variant_name] = {
                        "mean": run["mean_best_k"],
                        "std": run["std_best_k"],
                    }

    results["summary"] = summary

    return results
