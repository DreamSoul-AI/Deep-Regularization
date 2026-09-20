# -*- coding: utf-8 -*-
"""Multi-phase fixed-C/fixed-rho m=128 run with only polynomial expansion added.

This is the raw input-gate ADMM baseline plus exactly one modeling change:
  - raw scaled features expanded with PolynomialFeatures(degree=2)
  - GatedFeatureSelectionMLP; no weight-normalized first-layer path
  - fixed multi-phase C schedule; no auto/adaptive C
  - fixed rho; no adaptive rho
  - no phase-boundary pruning, z masking, hard pruning, or gate reweighting
  - per-phase warmup retained

Because the model gate is now in expanded polynomial space, final feature
selection is evaluated by multiple original-space aggregation readouts,
including parent_weight_sum.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path = [str(ROOT), str(ROOT / "Feature-Selection-Benchmark")] + sys.path

from src.admm_input_group_wrapper import (  # noqa: E402
    GatedFeatureSelectionMLP,
    ParentGatedFeatureSelectionMLP,
    _Scaler,
)
from src.c_schedules import build_global_c_schedule  # noqa: E402
from src.data import generate_dataset  # noqa: E402
from src.gradual_admm_with_pruning import gradual_admm_with_pruning  # noqa: E402


DATASET_GT = {
    "xor": [0, 1],
    "ring": [0, 1],
    "ring+xor": [0, 1, 2, 3],
    "ring+xor+sum": [0, 1, 2, 3, 4, 5],
}

SCHEDULE_REPLACEMENT_DYNAMIC_C_MODES = {
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
}

AGGREGATION_NAMES = [
    "group_product_current",
    "group_product_eff",
    "sum_g",
    "max_g",
    "L2_g",
    "L2_eff",
    "sum_eff",
    "max_eff",
    "degree_share_g",
    "degree_share_eff",
    "linear_only_g",
    "linear_only_eff",
    "parent_weight_sum",
    "parent_weight_sum_eff",
]


def parse_float_list(value: str) -> list[float]:
    return [float(v) for v in value.replace(";", ",").split(",") if v.strip()]


def compute_ratio_norm_coeff(dim_scale: str, p: float, q: float, d_active: int) -> float:
    if dim_scale == "none":
        return 1.0
    return float(max(1, int(d_active)) ** (1.0 / q - 1.0 / p))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_lognormal_gate_initialization(
    shape: torch.Size,
    epsilon: float,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a positive, mean-one gate without consuming training RNG state."""
    if epsilon < 0:
        raise ValueError("gate initialization epsilon must be nonnegative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(shape, generator=generator, dtype=dtype, device="cpu")
    gate = torch.exp(float(epsilon) * noise)
    return gate / gate.mean().clamp_min(torch.finfo(gate.dtype).eps)


def spearman_rank_correlation(
    left: torch.Tensor,
    right: torch.Tensor,
) -> float | None:
    """Return average-rank Spearman correlation, or None for a constant input."""
    left_np = left.detach().cpu().numpy().reshape(-1)
    right_np = right.detach().cpu().numpy().reshape(-1)
    if left_np.size != right_np.size:
        raise ValueError("Spearman inputs must have the same number of elements")
    if np.ptp(left_np) == 0 or np.ptp(right_np) == 0:
        return None
    left_rank = rankdata(left_np, method="average")
    right_rank = rankdata(right_np, method="average")
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return value if np.isfinite(value) else None


def format_optional_float(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_poly_maps(powers: np.ndarray) -> dict[str, Any]:
    n_original = powers.shape[1]
    linear_idx = np.full(n_original, -1, dtype=np.int64)
    square_idx = np.full(n_original, -1, dtype=np.int64)
    parent_terms: list[list[int]] = [[] for _ in range(n_original)]

    for term_idx, alpha in enumerate(powers):
        parents = np.flatnonzero(alpha)
        degree = int(alpha.sum())
        for parent in parents:
            parent_terms[int(parent)].append(int(term_idx))
        if degree == 1 and len(parents) == 1:
            linear_idx[int(parents[0])] = term_idx
        elif degree == 2 and len(parents) == 1:
            square_idx[int(parents[0])] = term_idx

    if np.any(linear_idx < 0) or np.any(square_idx < 0):
        raise RuntimeError("Polynomial index map is missing a linear or square term")

    return {
        "linear_idx": linear_idx,
        "square_idx": square_idx,
        "parent_terms": parent_terms,
        "powers": powers,
    }


def aggregate_sum(values: torch.Tensor, parent_terms: list[list[int]]) -> torch.Tensor:
    out = torch.zeros(len(parent_terms), dtype=values.dtype)
    for parent, term_indices in enumerate(parent_terms):
        out[parent] = values[term_indices].sum()
    return out


def aggregate_max(values: torch.Tensor, parent_terms: list[list[int]]) -> torch.Tensor:
    out = torch.zeros(len(parent_terms), dtype=values.dtype)
    for parent, term_indices in enumerate(parent_terms):
        out[parent] = values[term_indices].max()
    return out


def aggregate_l2(values: torch.Tensor, parent_terms: list[list[int]]) -> torch.Tensor:
    out = torch.zeros(len(parent_terms), dtype=values.dtype)
    for parent, term_indices in enumerate(parent_terms):
        out[parent] = torch.linalg.vector_norm(values[term_indices], ord=2)
    return out


def aggregate_linear_only(values: torch.Tensor, linear_idx: np.ndarray) -> torch.Tensor:
    idx = torch.as_tensor(linear_idx, dtype=torch.long)
    return values[idx].clone()


def aggregate_group_product(values: torch.Tensor, maps: dict[str, Any]) -> torch.Tensor:
    linear = torch.as_tensor(maps["linear_idx"], dtype=torch.long)
    square = torch.as_tensor(maps["square_idx"], dtype=torch.long)
    return values[linear].abs() * values[square].abs()


def aggregate_parent_weight_sum(values: torch.Tensor, powers: np.ndarray) -> torch.Tensor:
    """Aggregate a term score to parents using alpha_j / degree(term) weights."""
    out = torch.zeros(powers.shape[1], dtype=values.dtype)
    for term_idx, alpha in enumerate(powers):
        degree = int(alpha.sum())
        if degree <= 0:
            continue
        for parent in np.flatnonzero(alpha):
            out[int(parent)] += values[term_idx] * float(alpha[parent] / degree)
    return out


def topk_from_score(score: torch.Tensor, ground_truth: list[int], k: int) -> dict[str, Any]:
    score = score.detach().abs().cpu()
    positive = torch.nonzero(score > 0).flatten()
    if positive.numel() == 0:
        selected = []
    else:
        n_select = min(k, int(positive.numel()))
        local = score[positive].topk(n_select)
        selected = positive[local.indices].tolist()
    selected = [int(v) for v in selected]
    return {
        "best_k": compute_best_k(selected, ground_truth, k),
        "selected": selected,
        "nonzero": int((score > 0).sum().item()),
        "support_1e_2": int((score > 1e-2).sum().item()),
        "l1": float(score.sum().item()),
        "l2": float(torch.linalg.vector_norm(score, ord=2).item()),
        "max": float(score.max().item()),
    }


def first_column_norm(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "first_weight_norm"):
        return model.first_weight_norm.detach().cpu()
    first_linear = model.first_linear
    weight = first_linear.weight.detach().cpu()
    return torch.linalg.vector_norm(weight, ord=2, dim=0).clamp_min(1e-8)


def compute_aggregation_scores(
    gate: torch.Tensor,
    col_norm: torch.Tensor,
    powers: np.ndarray,
) -> dict[str, torch.Tensor]:
    maps = build_poly_maps(powers)
    parent_terms = maps["parent_terms"]
    g_abs = gate.detach().abs().cpu()
    eff_abs = (gate.detach().cpu() * col_norm.detach().cpu()).abs()

    return {
        "group_product_current": aggregate_group_product(g_abs, maps),
        "group_product_eff": aggregate_group_product(eff_abs, maps),
        "sum_g": aggregate_sum(g_abs, parent_terms),
        "max_g": aggregate_max(g_abs, parent_terms),
        "L2_g": aggregate_l2(g_abs, parent_terms),
        "L2_eff": aggregate_l2(eff_abs, parent_terms),
        "sum_eff": aggregate_sum(eff_abs, parent_terms),
        "max_eff": aggregate_max(eff_abs, parent_terms),
        "degree_share_g": aggregate_parent_weight_sum(g_abs, powers),
        "degree_share_eff": aggregate_parent_weight_sum(eff_abs, powers),
        "linear_only_g": aggregate_linear_only(g_abs, maps["linear_idx"]),
        "linear_only_eff": aggregate_linear_only(eff_abs, maps["linear_idx"]),
        "parent_weight_sum": aggregate_parent_weight_sum(g_abs, powers),
        "parent_weight_sum_eff": aggregate_parent_weight_sum(eff_abs, powers),
    }


def evaluate_all_aggregations(
    gate: torch.Tensor,
    col_norm: torch.Tensor,
    powers: np.ndarray,
    ground_truth: list[int],
    k: int,
) -> dict[str, dict[str, Any]]:
    scores = compute_aggregation_scores(gate, col_norm, powers)
    return {
        name: topk_from_score(score, ground_truth, k)
        for name, score in scores.items()
    }


def create_synthetic_dataset(
    name: str,
    n_samples: int,
    n_features: int,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    X, _, y = generate_dataset(name, n_samples=n_samples, n_features=n_features)
    scaler = _Scaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32, copy=False)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled).astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)

    ground_truth = DATASET_GT[name]
    return {
        "name": name,
        "X": X_poly,
        "y": y,
        "n_classes": 2,
        "ground_truth": ground_truth,
        "k": len(ground_truth),
        "n_features": int(X_poly.shape[1]),
        "n_original_features": n_features,
        "n_samples": n_samples,
        "polynomial_degree": 2,
        "poly_powers": poly.powers_.astype(np.int64, copy=False),
    }


def compute_best_k(selected: list[int], ground_truth: list[int], k: int) -> float:
    return float(len(set(selected) & set(ground_truth)) / k)


def topk_by_gate(model: torch.nn.Module, k: int) -> tuple[list[int], torch.Tensor]:
    gate_abs = model.gate.detach().abs().cpu()
    positive = torch.nonzero(gate_abs > 0).flatten()
    if positive.numel() == 0:
        selected = []
    else:
        n_select = min(k, int(positive.numel()))
        local = gate_abs[positive].topk(n_select)
        selected = positive[local.indices].tolist()
    return [int(v) for v in selected], gate_abs


def predict_proba(model: torch.nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        if logits.ndim == 2 and logits.shape[1] > 1:
            return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return torch.sigmoid(logits.reshape(-1)).cpu().numpy()


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float(np.mean(y_true))


def run_fold(
    dataset: dict[str, Any],
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    device: str,
    c_schedule: list[float],
    global_c_schedule: list[float] | None = None,
    legacy_observer=None,
) -> dict[str, Any]:
    fold_seed = args.seed + 1009 * (fold_idx + 1)
    set_seed(fold_seed)

    X_train = dataset["X"][train_idx]
    y_train = dataset["y"][train_idx]
    X_test = dataset["X"][test_idx]
    y_test = dataset["y"][test_idx]

    model_kwargs = {
        "latent_size": 32,
        "n_hidden_layers": 2,
        "dropout": args.dropout,
        "activation": "mish",
        "feat_drop": args.feat_drop,
        "bounded_gate": False,
        "column_normalize_first_layer": bool(args.column_normalize_first_layer),
    }
    base_model_class = GatedFeatureSelectionMLP
    if args.gate_mode == "parent":
        base_model_class = ParentGatedFeatureSelectionMLP
        model_kwargs["parent_powers"] = dataset["poly_powers"]

    gate_init_seed = fold_seed + int(args.gate_init_seed_offset)
    initial_state: dict[str, torch.Tensor] = {}

    def initialized_model_class(*model_args: Any, **model_call_kwargs: Any) -> torch.nn.Module:
        model = base_model_class(*model_args, **model_call_kwargs)
        initial_gate = build_lognormal_gate_initialization(
            model.gate.shape,
            args.gate_init_epsilon,
            gate_init_seed,
            dtype=model.gate.dtype,
        )
        with torch.no_grad():
            model.gate.copy_(initial_gate.to(model.gate.device))
        if not initial_state:
            initial_state["gate"] = model.gate.detach().cpu().clone()
            initial_state["col_norm"] = first_column_norm(model).clone()
            if legacy_observer is not None:
                legacy_observer.initialization(model)
        return model

    _, history, model = gradual_admm_with_pruning(
        model_class=initialized_model_class,
        model_kwargs=model_kwargs,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_classes=dataset["n_classes"],
        ground_truth=dataset["ground_truth"],
        k=dataset["k"],
        C_schedule=c_schedule,
        global_C_schedule_per_epoch=global_c_schedule,
        prune_threshold=args.prune_threshold,
        prune_mode="none",
        reweight=False,
        epochs_per_phase=args.epochs_per_phase,
        warmup_epochs=args.warmup_epochs,
        warmup_per_phase=len(c_schedule) > 1 and args.warmup_epochs > 0,
        auto_scale_C=False,
        use_ratio_norm=True,
        use_admm=args.ratio_norm_update != "direct_ratio_gradient",
        uniform_penalty=args.uniform_penalty,
        rho_init=args.rho_init,
        write_z_to_gate=False,
        soft_prune_update="threshold_gate",
        min_z_support=0,
        admm_weighted_gate=False,
        ratio_norm_update=args.ratio_norm_update,
        score_update_mode=args.score_update_mode,
        ratio_norm_dim_scale=args.ratio_norm_dim_scale,
        ratio_norm_p=args.ratio_norm_p,
        ratio_norm_q=args.ratio_norm_q,
        adaptive_rho=args.adaptive_rho,
        adaptive_rho_mu=args.adaptive_rho_mu,
        adaptive_rho_tau=args.adaptive_rho_tau,
        adaptive_rho_min=args.adaptive_rho_min,
        adaptive_rho_max=args.adaptive_rho_max,
        adaptive_C=False,
        dynamic_C_mode=args.dynamic_c_mode,
        dynamic_C_update_interval=args.dynamic_c_update_interval,
        dynamic_C_ema=args.dynamic_c_ema,
        dynamic_C_min_factor=args.dynamic_c_min_factor,
        dynamic_C_max_factor=args.dynamic_c_max_factor,
        dynamic_C_kappa=args.dynamic_c_kappa,
        dynamic_C_quantile=args.dynamic_c_quantile,
        dynamic_C_continuation_gamma=args.dynamic_c_continuation_gamma,
        dynamic_C_stability_patience=args.dynamic_c_stability_patience,
        dynamic_C_support_tol=args.dynamic_c_support_tol,
        dynamic_C_residual_tol=args.dynamic_c_residual_tol,
        dynamic_C_loss_tol=args.dynamic_c_loss_tol,
        dynamic_C_support_parent_low=args.dynamic_c_support_parent_low,
        dynamic_C_support_parent_high=args.dynamic_c_support_parent_high,
        dynamic_C_support_term_low=args.dynamic_c_support_term_low,
        dynamic_C_support_term_high=args.dynamic_c_support_term_high,
        dynamic_C_support_up_mult=args.dynamic_c_support_up_mult,
        dynamic_C_support_down_mult=args.dynamic_c_support_down_mult,
        group_sparse_powers=(
            None if args.gate_mode == "parent" else dataset["poly_powers"]
        ),
        group_sparse_mode="none",
        group_parent_support=0,
        lr=args.lr,
        g_inner_epochs=args.g_inner_epochs,
        lr_scheduler=args.lr_scheduler,
        cosine_eta_min=args.cosine_eta_min,
        cosine_decay_fraction=args.cosine_decay_fraction,
        rho_follow_lr=False,
        strict_neural_block_mode=args.strict_neural_block_mode,
        strict_kkt_check_interval=args.strict_kkt_check_interval,
        strict_kkt_tolerance=args.strict_kkt_tolerance,
        strict_kkt_absolute_tolerance=args.strict_kkt_absolute_tolerance,
        strict_kkt_patience=args.strict_kkt_patience,
        device=device,
        verbose=False,
        legacy_observer=legacy_observer,
    )

    gate = model.gate.detach().cpu()
    gate_abs = gate.abs()
    col_norm = first_column_norm(model)
    aggregation_gate = gate
    if args.gate_mode == "parent":
        with torch.no_grad():
            aggregation_gate = model.get_expanded_gate_values(training_drop=False).detach().cpu()
    aggregations = evaluate_all_aggregations(
        gate=aggregation_gate,
        col_norm=col_norm,
        powers=dataset["poly_powers"],
        ground_truth=dataset["ground_truth"],
        k=dataset["k"],
    )
    initial_gate = initial_state["gate"]
    initial_col_norm = initial_state["col_norm"]
    initial_aggregation_gate = initial_gate
    if args.gate_mode == "parent":
        parent_index = model.term_parent_index.detach().cpu()
        parent_mask = model.term_parent_mask.detach().cpu()
        gathered = initial_gate[parent_index]
        initial_aggregation_gate = torch.where(
            parent_mask,
            gathered,
            torch.ones_like(gathered),
        ).prod(dim=1)
    initial_scores = compute_aggregation_scores(
        initial_aggregation_gate,
        initial_col_norm,
        dataset["poly_powers"],
    )
    final_scores = compute_aggregation_scores(
        aggregation_gate,
        col_norm,
        dataset["poly_powers"],
    )
    if args.gate_mode == "parent":
        primary_readout = topk_from_score(
            gate_abs,
            dataset["ground_truth"],
            dataset["k"],
        )
        primary_aggregation = "identity"
    else:
        primary_aggregation = args.primary_aggregation
        primary_readout = aggregations[primary_aggregation]

    initial_primary_score = (
        initial_gate.abs()
        if primary_aggregation == "identity"
        else initial_scores[primary_aggregation]
    )
    final_primary_score = (
        gate_abs
        if primary_aggregation == "identity"
        else final_scores[primary_aggregation]
    )
    gate_init_diagnostics = {
        "mode": "mean_one_lognormal",
        "epsilon": float(args.gate_init_epsilon),
        "seed": int(gate_init_seed),
        "mean": float(initial_gate.mean().item()),
        "std": float(initial_gate.std(unbiased=False).item()),
        "min": float(initial_gate.min().item()),
        "max": float(initial_gate.max().item()),
        "term_gate_initial_to_final_spearman": spearman_rank_correlation(
            initial_aggregation_gate.abs(),
            aggregation_gate.abs(),
        ),
        "primary_score_initial_to_final_spearman": spearman_rank_correlation(
            initial_primary_score,
            final_primary_score,
        ),
        "initial_primary": topk_from_score(
            initial_primary_score,
            dataset["ground_truth"],
            dataset["k"],
        ),
    }

    y_train_score = predict_proba(model, X_train, device)
    y_test_score = predict_proba(model, X_test, device)
    train_auc = safe_auc(y_train, y_train_score)
    test_auc = safe_auc(y_test, y_test_score)
    train_auprc = safe_auprc(y_train, y_train_score)
    test_auprc = safe_auprc(y_test, y_test_score)

    admm_state = getattr(model, "admm_state", {})
    z_support = None
    z_support_thresholds: dict[str, int] = {}
    strict_final_diagnostics: dict[str, Any] = {}
    final_rho = (
        None if args.ratio_norm_update == "direct_ratio_gradient" else args.rho_init
    )
    final_C = c_schedule[-1]
    if isinstance(admm_state, dict):
        if "zk" in admm_state:
            z_abs = admm_state["zk"].abs()
            z_support_thresholds = {
                "0": int((z_abs > 0.0).sum().item()),
                "1e-12": int((z_abs > 1e-12).sum().item()),
                "1e-8": int((z_abs > 1e-8).sum().item()),
                "1e-4": int((z_abs > 1e-4).sum().item()),
                "1e-2": int((z_abs > 1e-2).sum().item()),
            }
            z_support = z_support_thresholds["1e-12"]
            strict_final_diagnostics["z_l1"] = float(z_abs.sum().item())
            strict_final_diagnostics["z_l2"] = float(torch.norm(z_abs, p=2).item())
        if "yk" in admm_state:
            y_abs = admm_state["yk"].abs()
            strict_final_diagnostics["y_l1"] = float(y_abs.sum().item())
            strict_final_diagnostics["y_l2"] = float(torch.norm(y_abs, p=2).item())
        strict_log = admm_state.get("strict_two_copy_log", [])
        if strict_log:
            strict_final_diagnostics["last_update"] = strict_log[-1]
            # Preserve a compact residual trajectory for convergence audits.
            # Epoch 0, every tenth epoch, and the final epoch are retained.
            strict_final_diagnostics["history_stride"] = 10
            strict_final_diagnostics["history"] = [
                row
                for index, row in enumerate(strict_log)
                if index == 0 or index % 10 == 0 or index == len(strict_log) - 1
            ]
        if args.ratio_norm_update != "direct_ratio_gradient":
            final_rho = float(admm_state.get("rho", args.rho_init))
        final_C = float(admm_state.get("C_used", c_schedule[-1]))
        stationarity = admm_state.get("strict_stationarity")
        if stationarity is not None:
            strict_final_diagnostics["stationarity"] = stationarity
        if "strict_kkt_converged" in admm_state:
            strict_final_diagnostics["convergence_stop"] = {
                "converged": bool(admm_state["strict_kkt_converged"]),
                "stop_epoch": admm_state.get("strict_kkt_stop_epoch"),
                "stop_reason": admm_state.get("strict_kkt_stop_reason"),
                "requested_outer_iterations": int(
                    admm_state.get("requested_outer_iterations", 0)
                ),
                "completed_outer_iterations": int(
                    admm_state.get("completed_outer_iterations", 0)
                ),
                "check_interval": int(
                    admm_state.get("strict_kkt_check_interval", 0)
                ),
                "tolerance": float(
                    admm_state.get("strict_kkt_tolerance", 1e-2)
                ),
                "absolute_tolerance": float(
                    admm_state.get("strict_kkt_absolute_tolerance", 0.0)
                ),
                "patience": int(admm_state.get("strict_kkt_patience", 0)),
                "history": admm_state.get("strict_kkt_history", []),
            }
        if admm_state.get("strict_gate_inner_enabled", False):
            gate_inner_history = admm_state.get(
                "strict_gate_inner_history", []
            )
            strict_final_diagnostics["gate_inner_solve"] = {
                "enabled": True,
                "max_steps": int(
                    admm_state.get("strict_gate_inner_max_steps", 0)
                ),
                "total_steps": int(
                    admm_state.get("strict_gate_inner_total_steps", 0)
                ),
                "mean_steps": float(
                    admm_state.get("strict_gate_inner_mean_steps", 0.0)
                ),
                "passes": int(
                    admm_state.get("strict_gate_inner_passes", 0)
                ),
                "cap_hits": int(
                    admm_state.get("strict_gate_inner_cap_hits", 0)
                ),
                "history_stride": 10,
                "history": [
                    row
                    for index, row in enumerate(gate_inner_history)
                    if index == 0
                    or index % 10 == 0
                    or index == len(gate_inner_history) - 1
                ],
            }
    score_update_mode = (
        str(admm_state.get("score_update_mode", args.score_update_mode))
        if isinstance(admm_state, dict)
        else args.score_update_mode
    )

    phase_rows = history.get("phases", [])
    phase_diagnostic_keys = [
        "parent_max_g",
        "parent_max_A_times_g",
        "parent_max_z",
        "parent_weight_sum_g",
        "parent_weight_sum_A_times_g",
        "parent_weight_sum_z",
    ]
    phase_diagnostics = []
    for phase_row in phase_rows:
        diag_row: dict[str, Any] = {
            "phase": int(phase_row.get("phase", 0)),
            "C_used": float(phase_row.get("C_used", 0.0)),
            "rho": float(phase_row.get("rho", 0.0)),
            "z_support": phase_row.get("z_support"),
        }
        for key in phase_diagnostic_keys:
            if f"{key}_best_k" in phase_row:
                diag_row[key] = {
                    "best_k": float(phase_row.get(f"{key}_best_k", 0.0)),
                    "selected": [
                        int(v) for v in phase_row.get(f"{key}_selected", [])
                    ],
                    "nonzero": int(phase_row.get(f"{key}_nonzero", 0)),
                    "gt_ranks": phase_row.get(f"{key}_gt_ranks", {}),
                }
        phase_diagnostics.append(diag_row)
    result = {
        "dataset": dataset["name"],
        "fold": int(fold_idx),
        "seed": int(fold_seed),
        "best_k": primary_readout["best_k"],
        "selected": primary_readout["selected"],
        "primary_aggregation": primary_aggregation,
        "score_update_mode": score_update_mode,
        "aggregations": aggregations,
        "ground_truth": dataset["ground_truth"],
        "k": int(dataset["k"]),
        "n_original_features": int(dataset["n_original_features"]),
        "n_poly_features": int(dataset["n_features"]),
        "gate_initialization": gate_init_diagnostics,
        "train_auc": train_auc,
        "auc": test_auc,
        "train_auprc": train_auprc,
        "auprc": test_auprc,
        "gate_nonzero": int((gate_abs > 0).sum().item()),
        "gate_zeros_strict": int((gate_abs == 0).sum().item()),
        "gate_zeros_loose_1e_2": int((gate_abs < 1e-2).sum().item()),
        "gate_l1": float(gate_abs.sum().item()),
        "gate_l2": float(torch.norm(gate_abs, p=2).item()),
        "gate_max": float(gate_abs.max().item()),
        "z_support": z_support,
        "z_support_thresholds": z_support_thresholds,
        "strict_final_diagnostics": strict_final_diagnostics,
        "final_rho": final_rho,
        "final_C_used": final_C,
        "phase_C_requested": [float(p.get("C", 0.0)) for p in phase_rows],
        "phase_C_schedule_entry": [
            float(p.get("C_schedule_entry", p.get("C", 0.0)))
            for p in phase_rows
        ],
        "phase_C_used": [float(p.get("C_used", 0.0)) for p in phase_rows],
        "global_C_history": [
            float(value)
            for phase in phase_rows
            for value in phase.get("C_schedule_per_epoch", [])
        ],
        "phase_dynamic_C_history": [p.get("dynamic_C_history", []) for p in phase_rows],
        "phase_rho": [float(p.get("rho", 0.0)) for p in phase_rows],
        "phase_z_support": [p.get("z_support") for p in phase_rows],
        "phase_best_k": [float(p.get("best_k", 0.0)) for p in phase_rows],
        "phase_alive_count": [int(p.get("alive_count", 0)) for p in phase_rows],
        "phase_diagnostics": phase_diagnostics,
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    z_values = [f["z_support"] for f in folds if f["z_support"] is not None]
    gate_rank_correlations = [
        f["gate_initialization"]["term_gate_initial_to_final_spearman"]
        for f in folds
        if f["gate_initialization"]["term_gate_initial_to_final_spearman"] is not None
    ]
    primary_rank_correlations = [
        f["gate_initialization"]["primary_score_initial_to_final_spearman"]
        for f in folds
        if f["gate_initialization"]["primary_score_initial_to_final_spearman"] is not None
    ]
    return {
        "n_folds": len(folds),
        "best_k_mean": float(np.mean([f["best_k"] for f in folds])),
        "best_k_std": float(np.std([f["best_k"] for f in folds])),
        "auc_mean": float(np.mean([f["auc"] for f in folds])),
        "auc_std": float(np.std([f["auc"] for f in folds])),
        "auprc_mean": float(np.mean([f["auprc"] for f in folds])),
        "auprc_std": float(np.std([f["auprc"] for f in folds])),
        "gate_zeros_strict_mean": float(np.mean([f["gate_zeros_strict"] for f in folds])),
        "gate_zeros_loose_1e_2_mean": float(np.mean([f["gate_zeros_loose_1e_2"] for f in folds])),
        "z_support_mean": float(np.mean(z_values)) if z_values else None,
        "term_gate_initial_to_final_spearman_mean": (
            float(np.mean(gate_rank_correlations)) if gate_rank_correlations else None
        ),
        "primary_score_initial_to_final_spearman_mean": (
            float(np.mean(primary_rank_correlations))
            if primary_rank_correlations
            else None
        ),
        "aggregations": {
            name: {
                "best_k_mean": float(np.mean([f["aggregations"][name]["best_k"] for f in folds])),
                "best_k_std": float(np.std([f["aggregations"][name]["best_k"] for f in folds])),
                "fold_best_k": [
                    float(f["aggregations"][name]["best_k"])
                    for f in folds
                ],
            }
            for name in AGGREGATION_NAMES
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["xor", "ring", "ring+xor", "ring+xor+sum"],
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-schedule", default="0.1,0.2,0.5,1.0,2.0")
    parser.add_argument(
        "--global-c-schedule",
        choices=["none", "linear", "cosine", "exponential", "exponential_hold"],
        default="none",
        help="Optional monotone C schedule over all ADMM epochs and phases.",
    )
    parser.add_argument("--c-min", type=float, default=0.01)
    parser.add_argument("--c-max", type=float, default=None)
    parser.add_argument("--rho-init", type=float, default=200.0)
    parser.add_argument(
        "--adaptive-rho",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--adaptive-rho-mu", type=float, default=10.0)
    parser.add_argument("--adaptive-rho-tau", type=float, default=2.0)
    parser.add_argument("--adaptive-rho-min", type=float, default=None)
    parser.add_argument("--adaptive-rho-max", type=float, default=1e4)
    parser.add_argument("--epochs-per-phase", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--cosine-eta-min", type=float, default=None)
    parser.add_argument("--cosine-decay-fraction", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.043)
    parser.add_argument("--feat-drop", type=float, default=0.0)
    parser.add_argument(
        "--gate-init-epsilon",
        type=float,
        default=0.0,
        help="Epsilon in positive mean-one log-normal gate initialization.",
    )
    parser.add_argument(
        "--gate-init-seed-offset",
        type=int,
        default=0,
        help="Offset added to the fold seed for the isolated gate-init RNG.",
    )
    parser.add_argument("--prune-threshold", type=float, default=0.01)
    parser.add_argument("--only-fold", type=int, default=None, help="1-based fold index to run")
    parser.add_argument("--tag", default="multiphase_fixed_c_rho_poly2_only_m128")
    parser.add_argument(
        "--primary-aggregation",
        choices=AGGREGATION_NAMES,
        default="parent_weight_sum",
        help="Primary best-k readout for expanded-gate runs.",
    )
    parser.add_argument("--ratio-norm-dim-scale", choices=["none", "pqi"], default="none")
    parser.add_argument(
        "--ratio-norm-update",
        choices=[
            "current",
            "formula_center_den",
            "formula_center_den_weighted",
            "formula_z_den",
            "formula_z_den_weighted",
            "strict_two_copy",
            "strict_two_copy_weighted",
            "direct_ratio_gradient",
        ],
        default="current",
        help="RatioNorm z-step variant; default preserves legacy run5.",
    )
    parser.add_argument(
        "--uniform-penalty",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use one scalar C for every coordinate. The default preserves legacy "
            "behavior: enabled for strict_two_copy and disabled otherwise."
        ),
    )
    parser.add_argument(
        "--score-update-mode",
        choices=[
            "epoch",
            "epoch_bounded_normalized",
            "epoch_median_normalized",
            "epoch_permuted",
            "freeze_after_warmup",
            "freeze_run_initial",
        ],
        default="epoch",
        help="How to update score_j used by score-weighted penalties.",
    )
    parser.add_argument("--ratio-norm-p", type=float, default=1.0)
    parser.add_argument("--ratio-norm-q", type=float, default=2.0)
    parser.add_argument(
        "--dynamic-c-mode",
        choices=[
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
        ],
        default="none",
    )
    parser.add_argument("--dynamic-c-update-interval", type=int, default=5)
    parser.add_argument("--dynamic-c-ema", type=float, default=0.8)
    parser.add_argument("--dynamic-c-min-factor", type=float, default=0.1)
    parser.add_argument("--dynamic-c-max-factor", type=float, default=10.0)
    parser.add_argument("--dynamic-c-kappa", type=float, default=1.0)
    parser.add_argument("--dynamic-c-quantile", type=float, default=0.9)
    parser.add_argument("--dynamic-c-continuation-gamma", type=float, default=1.5)
    parser.add_argument("--dynamic-c-stability-patience", type=int, default=2)
    parser.add_argument("--dynamic-c-support-tol", type=float, default=0.02)
    parser.add_argument("--dynamic-c-residual-tol", type=float, default=0.10)
    parser.add_argument("--dynamic-c-loss-tol", type=float, default=0.02)
    parser.add_argument("--dynamic-c-support-parent-low", type=int, default=0)
    parser.add_argument("--dynamic-c-support-parent-high", type=int, default=0)
    parser.add_argument("--dynamic-c-support-term-low", type=int, default=0)
    parser.add_argument("--dynamic-c-support-term-high", type=int, default=0)
    parser.add_argument("--dynamic-c-support-up-mult", type=float, default=1.25)
    parser.add_argument("--dynamic-c-support-down-mult", type=float, default=1.5)
    parser.add_argument(
        "--gate-mode",
        choices=["expanded", "parent"],
        default="expanded",
        help="expanded = one gate per polynomial term; parent = one gate per original feature",
    )
    parser.add_argument(
        "--column-normalize-first-layer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use unit-norm first-layer input columns inside the forward pass. "
            "This removes the gate/column scale degree of freedom."
        ),
    )
    parser.add_argument(
        "--g-inner-epochs",
        type=int,
        default=1,
        help="Full data passes in each frozen-copy (theta,g) ADMM block update.",
    )
    parser.add_argument(
        "--strict-neural-block-mode",
        choices=[
            "joint_minibatch",
            "theta_minibatch_gate_fullfold",
            "theta_minibatch_gate_fullfold_inner",
        ],
        default="joint_minibatch",
        help=(
            "Neural block solver for strict two-copy ADMM. The diagnostic "
            "full-fold mode updates theta by minibatch, then g once from the "
            "deterministic full training fold."
        ),
    )
    parser.add_argument("--strict-kkt-check-interval", type=int, default=0)
    parser.add_argument("--strict-kkt-tolerance", type=float, default=1e-2)
    parser.add_argument("--strict-kkt-absolute-tolerance", type=float, default=0.0)
    parser.add_argument("--strict-kkt-patience", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main() -> dict[str, Any]:
    args = parse_args()
    if args.gate_init_epsilon < 0:
        raise ValueError("--gate-init-epsilon must be nonnegative")
    if args.g_inner_epochs < 1:
        raise ValueError("--g-inner-epochs must be positive")
    if (
        args.strict_neural_block_mode
        in {
            "theta_minibatch_gate_fullfold",
            "theta_minibatch_gate_fullfold_inner",
        }
        and args.ratio_norm_update != "strict_two_copy"
    ):
        raise ValueError(
            "--strict-neural-block-mode theta_minibatch_gate_fullfold "
            "requires --ratio-norm-update strict_two_copy"
        )
    if (
        args.strict_neural_block_mode
        in {
            "theta_minibatch_gate_fullfold",
            "theta_minibatch_gate_fullfold_inner",
        }
        and args.dynamic_c_mode != "none"
    ):
        raise ValueError(
            "--strict-neural-block-mode theta_minibatch_gate_fullfold "
            "currently requires --dynamic-c-mode none"
        )
    if (
        args.strict_neural_block_mode
        == "theta_minibatch_gate_fullfold_inner"
        and args.g_inner_epochs != 1
    ):
        raise ValueError(
            "--strict-neural-block-mode theta_minibatch_gate_fullfold_inner "
            "requires --g-inner-epochs 1"
        )
    if args.strict_kkt_check_interval < 0:
        raise ValueError("--strict-kkt-check-interval must be nonnegative")
    if (
        args.strict_kkt_check_interval > 0
        and args.ratio_norm_update != "strict_two_copy"
    ):
        raise ValueError("strict KKT stopping requires --ratio-norm-update strict_two_copy")
    if args.uniform_penalty is None:
        args.uniform_penalty = args.ratio_norm_update == "strict_two_copy"
    if args.ratio_norm_update == "strict_two_copy" and not args.uniform_penalty:
        raise ValueError("strict_two_copy has one scalar C and requires --uniform-penalty")
    if args.ratio_norm_update == "strict_two_copy_weighted" and args.uniform_penalty:
        raise ValueError(
            "strict_two_copy_weighted requires --no-uniform-penalty so score is active"
        )
    if args.smoke:
        args.datasets = args.datasets[:1]
        args.folds = min(args.folds, 2)
        args.epochs_per_phase = 1
        args.warmup_epochs = 0

    for dataset_name in args.datasets:
        if dataset_name not in DATASET_GT:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    c_schedule = parse_float_list(args.c_schedule)
    global_c_schedule = None
    if args.global_c_schedule != "none":
        if args.c_max is None:
            raise ValueError("--c-max is required with --global-c-schedule")
        total_admm_epochs = len(c_schedule) * int(args.epochs_per_phase)
        global_c_schedule = build_global_c_schedule(
            args.c_min,
            args.c_max,
            total_admm_epochs,
            args.global_c_schedule,
        )
        # C_schedule now defines phase count only. The per-epoch global values
        # are sliced inside gradual_admm_with_pruning without phase resets.
        c_schedule = [float(args.c_min)] * len(c_schedule)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / f"{args.tag}_{timestamp}.jsonl"
    output_file = output_dir / f"{args.tag}_{timestamp}.json"

    model_name = (
        "ParentGatedFeatureSelectionMLP"
        if args.gate_mode == "parent"
        else "GatedFeatureSelectionMLP"
    )
    primary_aggregation = "identity" if args.gate_mode == "parent" else args.primary_aggregation
    n_poly_features = args.n_features + args.n_features * (args.n_features + 1) // 2
    ratio_norm_d_active = (
        args.n_features if args.gate_mode == "parent" else n_poly_features
    )
    ratio_norm_coeff = compute_ratio_norm_coeff(
        args.ratio_norm_dim_scale,
        args.ratio_norm_p,
        args.ratio_norm_q,
        ratio_norm_d_active,
    )
    metadata = {
        "timestamp": timestamp,
        "experiment": "multiphase_fixed_c_rho_poly2_only_m128",
        "datasets": args.datasets,
        "n_samples": args.n_samples,
        "n_original_features": args.n_features,
        "n_poly_features": n_poly_features,
        "folds": args.folds,
        "seed": args.seed,
        "device": device,
        "model": model_name,
        "gate_mode": args.gate_mode,
        "polynomial_degree": 2,
        "polynomial_include_bias": False,
        "only_change_vs_base": (
            "PolynomialFeatures(degree=2, include_bias=False); fixed-C/fixed-rho "
            "no-extra training unchanged. ratio_norm_update is controlled by "
            "--ratio-norm-update for strict-update diagnostics. "
            "column_normalize_first_layer is an explicit diagnostic axis."
        ),
        "weight_norm": False,
        "column_normalize_first_layer": bool(args.column_normalize_first_layer),
        "selection": (
            "parent_gate_identity"
            if args.gate_mode == "parent"
            else "original_space_aggregation_from_expanded_gate"
        ),
        "primary_aggregation": primary_aggregation,
        "aggregation_methods": AGGREGATION_NAMES,
        "multi_phase": len(c_schedule) > 1,
        "lifecycle": (
            "multi_phase"
            if len(c_schedule) > 1
            else "single_continuous_direct_ratio_gradient"
            if args.ratio_norm_update == "direct_ratio_gradient"
            else "single_continuous_admm"
        ),
        "C_schedule": c_schedule,
        "global_C_schedule_type": args.global_c_schedule,
        "global_C_schedule_per_epoch": global_c_schedule,
        "C_min": float(args.c_min) if global_c_schedule is not None else None,
        "C_max": float(args.c_max) if global_c_schedule is not None else None,
        "C_schedule_role": (
            "global_epoch_schedule; phase list defines phase count only"
            if global_c_schedule is not None
            else
            "phase_count_seed_only; dynamic_C computes C_t; phase modes recompute per phase"
            if args.dynamic_c_mode in SCHEDULE_REPLACEMENT_DYNAMIC_C_MODES
            else "phase_anchor_schedule"
        ),
        "auto_scale_C": False,
        "adaptive_C": False,
        "dynamic_C_mode": args.dynamic_c_mode,
        "dynamic_C_replaces_C_schedule": args.dynamic_c_mode
        in SCHEDULE_REPLACEMENT_DYNAMIC_C_MODES,
        "dynamic_C_update_interval": args.dynamic_c_update_interval,
        "dynamic_C_ema": args.dynamic_c_ema,
        "dynamic_C_min_factor": args.dynamic_c_min_factor,
        "dynamic_C_max_factor": args.dynamic_c_max_factor,
        "dynamic_C_kappa": args.dynamic_c_kappa,
        "dynamic_C_quantile": args.dynamic_c_quantile,
        "dynamic_C_continuation_gamma": args.dynamic_c_continuation_gamma,
        "dynamic_C_stability_patience": args.dynamic_c_stability_patience,
        "dynamic_C_support_tol": args.dynamic_c_support_tol,
        "dynamic_C_residual_tol": args.dynamic_c_residual_tol,
        "dynamic_C_loss_tol": args.dynamic_c_loss_tol,
        "dynamic_C_cap_seed": None,
        "dynamic_C_bound_seed_role": "phase_C_anchor",
        "dynamic_C_support_parent_low": args.dynamic_c_support_parent_low,
        "dynamic_C_support_parent_high": args.dynamic_c_support_parent_high,
        "dynamic_C_support_term_low": args.dynamic_c_support_term_low,
        "dynamic_C_support_term_high": args.dynamic_c_support_term_high,
        "dynamic_C_support_up_mult": args.dynamic_c_support_up_mult,
        "dynamic_C_support_down_mult": args.dynamic_c_support_down_mult,
        "rho_init": (
            None if args.ratio_norm_update == "direct_ratio_gradient" else args.rho_init
        ),
        "rho_argument_record_only": (
            args.rho_init
            if args.ratio_norm_update == "direct_ratio_gradient"
            else None
        ),
        "adaptive_rho": args.adaptive_rho,
        "adaptive_rho_mu": args.adaptive_rho_mu,
        "adaptive_rho_tau": args.adaptive_rho_tau,
        "adaptive_rho_min": args.adaptive_rho_min,
        "adaptive_rho_max": args.adaptive_rho_max,
        "use_admm": args.ratio_norm_update != "direct_ratio_gradient",
        "use_ratio_norm": True,
        "ratio_norm_update": args.ratio_norm_update,
        "uniform_penalty": bool(args.uniform_penalty),
        "uses_external_penalty_score": bool(
            not args.uniform_penalty
            and args.ratio_norm_update
            in {
                "current",
                "formula_center_den_weighted",
                "formula_z_den_weighted",
                "strict_two_copy_weighted",
            }
        ),
        "score_update_mode": (
            "disabled"
            if args.ratio_norm_update == "direct_ratio_gradient"
            else args.score_update_mode
        ),
        "ratio_norm_dim_scale": args.ratio_norm_dim_scale,
        "ratio_norm_p": args.ratio_norm_p,
        "ratio_norm_q": args.ratio_norm_q,
        "ratio_norm_d_active": ratio_norm_d_active,
        "ratio_norm_coeff": ratio_norm_coeff,
        "admm_weighted_gate": False,
        "prune_mode": "none",
        "soft_prune_update": "disabled_by_prune_mode_none",
        "reweight": False,
        "warmup_per_phase": len(c_schedule) > 1 and args.warmup_epochs > 0,
        "C_applied_from_epoch_0": (
            args.warmup_epochs == 0
        ),
        "epochs_per_phase": args.epochs_per_phase,
        "g_inner_epochs": args.g_inner_epochs,
        "strict_neural_block_mode": args.strict_neural_block_mode,
        "total_admm_optimizer_passes": int(
            len(c_schedule) * args.epochs_per_phase * args.g_inner_epochs
        ),
        "warmup_epochs": args.warmup_epochs,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "cosine_eta_min": args.cosine_eta_min,
        "cosine_decay_fraction": args.cosine_decay_fraction,
        "strict_kkt_check_interval": args.strict_kkt_check_interval,
        "strict_kkt_tolerance": args.strict_kkt_tolerance,
        "strict_kkt_absolute_tolerance": args.strict_kkt_absolute_tolerance,
        "strict_kkt_patience": args.strict_kkt_patience,
        "dropout": args.dropout,
        "feat_drop": args.feat_drop,
        "gate_initialization": {
            "mode": "mean_one_lognormal",
            "epsilon": float(args.gate_init_epsilon),
            "seed_rule": "fold_seed + gate_init_seed_offset",
            "gate_init_seed_offset": int(args.gate_init_seed_offset),
            "isolated_rng": True,
            "initial_consensus": (
                "not_applicable_no_admm"
                if args.ratio_norm_update == "direct_ratio_gradient"
                else "z0=y0=g0; u0=w0=0"
            ),
        },
        "prune_threshold_record_only": args.prune_threshold,
        "only_fold": args.only_fold,
        "checkpoint_file": str(checkpoint_file),
    }

    print("=" * 96, flush=True)
    lifecycle_label = "MULTI-PHASE" if len(c_schedule) > 1 else "SINGLE-CONTINUOUS"
    print(
        f"{lifecycle_label} FIXED RHO + POLY2 ONLY, "
        f"COLUMN_NORM={args.column_normalize_first_layer}, NO PRUNING, m=128",
        flush=True,
    )
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Checkpoint: {checkpoint_file}", flush=True)
    print("=" * 96, flush=True)

    results: dict[str, Any] = {}
    with checkpoint_file.open("w", encoding="utf-8") as ckpt:
        for dataset_name in args.datasets:
            dataset = create_synthetic_dataset(
                dataset_name,
                n_samples=args.n_samples,
                n_features=args.n_features,
                seed=args.seed,
            )
            skf = StratifiedKFold(
                n_splits=args.folds,
                shuffle=True,
                random_state=args.seed,
            )
            folds: list[dict[str, Any]] = []

            print(
                f"\n[{dataset_name}] gt={dataset['ground_truth']} k={dataset['k']} "
                f"features={dataset['n_original_features']}->{dataset['n_features']}",
                flush=True,
            )
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dataset["X"], dataset["y"])):
                if args.only_fold is not None and fold_idx != args.only_fold - 1:
                    continue
                fold_result = run_fold(
                    dataset=dataset,
                    fold_idx=fold_idx,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    args=args,
                    device=device,
                    c_schedule=c_schedule,
                    global_c_schedule=global_c_schedule,
                )
                folds.append(fold_result)
                ckpt.write(json.dumps(fold_result) + "\n")
                ckpt.flush()

                agg = fold_result["aggregations"]
                if args.gate_mode == "parent":
                    print(
                        f"  fold {fold_idx + 1}/{args.folds}: "
                        f"identity={fold_result['best_k']:.3f} "
                        f"expanded_parent_weight_sum={agg['parent_weight_sum']['best_k']:.3f} "
                        f"expanded_max_g={agg['max_g']['best_k']:.3f} "
                        f"AUC={fold_result['auc']:.3f} "
                        f"rho={format_optional_float(fold_result['final_rho'], 3)} "
                        f"C={fold_result['final_C_used']:.3f} "
                        f"z={fold_result['z_support']}",
                        flush=True,
                    )
                else:
                    print(
                        f"  fold {fold_idx + 1}/{args.folds}: "
                        f"{fold_result['primary_aggregation']}={fold_result['best_k']:.3f} "
                        f"parent_weight_sum={agg['parent_weight_sum']['best_k']:.3f} "
                        f"parent_weight_sum_eff={agg['parent_weight_sum_eff']['best_k']:.3f} "
                        f"group_product={agg['group_product_current']['best_k']:.3f} "
                        f"sum_g={agg['sum_g']['best_k']:.3f} "
                        f"max_g={agg['max_g']['best_k']:.3f} "
                        f"AUC={fold_result['auc']:.3f} "
                        f"rho={format_optional_float(fold_result['final_rho'], 3)} "
                        f"C={fold_result['final_C_used']:.3f} "
                        f"z={fold_result['z_support']}",
                        flush=True,
                    )

            results[dataset_name] = {
                "folds": folds,
                "summary": summarize_folds(folds),
            }
            s = results[dataset_name]["summary"]
            agg_s = s["aggregations"]
            if args.gate_mode == "parent":
                print(
                    f"  >> {dataset_name}: "
                    f"identity={s['best_k_mean']:.3f}+/-{s['best_k_std']:.3f} "
                    f"AUC={s['auc_mean']:.3f}+/-{s['auc_std']:.3f} "
                    f"z={format_optional_float(s['z_support_mean'])}",
                    flush=True,
                )
            else:
                print(
                    f"  >> {dataset_name}: "
                    f"{args.primary_aggregation}={s['best_k_mean']:.3f}+/-{s['best_k_std']:.3f} "
                    f"parent_weight_sum={agg_s['parent_weight_sum']['best_k_mean']:.3f} "
                    f"parent_weight_sum_eff={agg_s['parent_weight_sum_eff']['best_k_mean']:.3f} "
                    f"group_product={agg_s['group_product_current']['best_k_mean']:.3f} "
                    f"sum_g={agg_s['sum_g']['best_k_mean']:.3f} "
                    f"max_g={agg_s['max_g']['best_k_mean']:.3f} "
                    f"AUC={s['auc_mean']:.3f}+/-{s['auc_std']:.3f} "
                    f"z={format_optional_float(s['z_support_mean'])}",
                    flush=True,
                )

    overall_z_support = [
        results[d]["summary"]["z_support_mean"]
        for d in args.datasets
        if results[d]["summary"]["z_support_mean"] is not None
    ]
    overall = {
        "best_k_mean": float(np.mean([results[d]["summary"]["best_k_mean"] for d in args.datasets])),
        "auc_mean": float(np.mean([results[d]["summary"]["auc_mean"] for d in args.datasets])),
        "auprc_mean": float(np.mean([results[d]["summary"]["auprc_mean"] for d in args.datasets])),
        "z_support_mean": (
            float(np.mean(overall_z_support)) if overall_z_support else None
        ),
        "aggregations": {
            name: float(
                np.mean([
                    results[d]["summary"]["aggregations"][name]["best_k_mean"]
                    for d in args.datasets
                ])
            )
            for name in AGGREGATION_NAMES
        },
    }
    output = {
        "metadata": metadata,
        "overall": overall,
        "results": results,
    }
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nSummary", flush=True)
    if args.gate_mode == "parent":
        print("dataset          identity    AUC         z_support", flush=True)
        print("-" * 64, flush=True)
        for dataset_name in args.datasets:
            s = results[dataset_name]["summary"]
            print(
                f"{dataset_name:<16} "
                f"{s['best_k_mean']:.3f}+/-{s['best_k_std']:.3f}  "
                f"{s['auc_mean']:.3f}+/-{s['auc_std']:.3f}  "
                f"{format_optional_float(s['z_support_mean'])}",
                flush=True,
            )
        print(
            f"overall          {overall['best_k_mean']:.3f}        "
            f"{overall['auc_mean']:.3f}",
            flush=True,
        )
    else:
        print(
            f"dataset          primary({args.primary_aggregation})  parent_eff   "
            "parent_wsum   group_prod   AUC",
            flush=True,
        )
        print("-" * 104, flush=True)
        for dataset_name in args.datasets:
            s = results[dataset_name]["summary"]
            agg_s = s["aggregations"]
            print(
                f"{dataset_name:<16} "
                f"{s['best_k_mean']:.3f}+/-{s['best_k_std']:.3f}  "
                f"{agg_s['parent_weight_sum_eff']['best_k_mean']:.3f}+/-{agg_s['parent_weight_sum_eff']['best_k_std']:.3f}  "
                f"{agg_s['parent_weight_sum']['best_k_mean']:.3f}+/-{agg_s['parent_weight_sum']['best_k_std']:.3f}  "
                f"{agg_s['group_product_current']['best_k_mean']:.3f}+/-{agg_s['group_product_current']['best_k_std']:.3f}  "
                f"{s['auc_mean']:.3f}+/-{s['auc_std']:.3f}",
                flush=True,
            )
        print(
            f"overall          {overall['best_k_mean']:.3f}        "
            f"{overall['aggregations']['parent_weight_sum_eff']:.3f}        "
            f"{overall['aggregations']['parent_weight_sum']:.3f}        "
            f"{overall['aggregations']['group_product_current']:.3f}        "
            f"{overall['auc_mean']:.3f}",
            flush=True,
        )
    print(f"\nSaved: {output_file}", flush=True)
    return output


if __name__ == "__main__":
    main()
