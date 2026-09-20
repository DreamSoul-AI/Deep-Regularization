"""Frozen v3 effective-gate C-path diagnostic runner.

The formal protocol has two distinct training trajectories: F uses a fixed
endpoint C after warm-up, while P follows five increasing C stages ending at
the same endpoint.  P-final and P-path share the exact same P checkpoint and
differ only in readout.  A formal run is fail-closed; ``--smoke`` is explicitly
non-evidence and must write outside the repository results tree.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from scipy.stats import rankdata
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ablation.run_scheme_abc_bridge import (  # noqa: E402
    JsonlWriter,
    _array_hash,
    _assert_smoke_output_is_non_evidence,
    _fit_scaler,
    _jsonable,
    _set_determinism,
    _sha256_bytes,
    _sha256_file,
    _state_dict_hash,
)
from src.admm_input_group_wrapper import (  # noqa: E402
    GatedExplicitScaleMLP,
    _train_input_group,
)
from src.admm_utils import deterministic_topk  # noqa: E402
from src.data import generate_dataset  # noqa: E402


PROTOCOL_ID = "EFFECTIVE-GATE-C-PATH-V3-20260824"
PROTOCOL_VERSION = "v3-frozen"
DEFAULT_TAG = "effective_gate_c_path_v3"
VARIANTS = ("F", "P-final", "P-path")
GROUND_TRUTH = (0, 1)
DATASET_OFFSETS = {"xor": 0, "ring": 100_000}
C_GRID = [0.05, 0.20, 0.80, 3.20, 12.80]
STAGE_MULTIPLIERS = [1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0]
RELEVANT_SOURCE_PATHS = [
    "experiments/ablation/run_effective_gate_c_path_v3.py",
    "src/admm_input_group_wrapper.py",
    "src/admm_utils.py",
    "src/data.py",
    "docs/plans/PLAN_20260824_EFFECTIVE_GATE_C_PATH_PREREGISTRATION_V3.md",
]

FORMAL_CONFIG = {
    "datasets": ["xor", "ring"],
    "n_samples": 1000,
    "n_features": 128,
    "folds": 6,
    "seed": 20260824,
    "c_grid": C_GRID,
    "rho": 50.0,
    "warmup_epochs": 20,
    "admm_epochs": 100,
    "stage_epochs": 20,
    "stage_multipliers": STAGE_MULTIPLIERS,
    "path_epsilon": 1e-12,
    "endpoint_tie_tolerance": 1e-12,
    "inner_validation_fraction": 0.2,
    "batch_size": 64,
    "lr": 0.005,
    "device": "cpu",
    "tag": DEFAULT_TAG,
    "hidden_width": 32,
    "hidden_layers": 2,
    "readout_k": 2,
    "readout_zero_tolerance": 0.0,
    "refit_epochs": 120,
    "refit_batch_size": 64,
    "refit_lr": 0.005,
    "refit_weight_decay": 0.0,
    "fold_limit": None,
    "evidence_status": "formal_frozen_v3",
}


def _parse_c_grid(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("c-grid must be comma-separated floats") from exc
    if not values or not all(math.isfinite(item) and item > 0.0 for item in values):
        raise argparse.ArgumentTypeError("c-grid values must be finite and positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--formal-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["xor", "ring"])
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--c-grid", type=_parse_c_grid, default=C_GRID.copy())
    parser.add_argument("--rho", type=float, default=50.0)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--admm-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "ablation"
    )
    return parser


def normalized_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        **FORMAL_CONFIG,
        "datasets": [item.lower() for item in args.datasets],
        "n_samples": int(args.n_samples),
        "n_features": int(args.n_features),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "c_grid": [float(item) for item in args.c_grid],
        "rho": float(args.rho),
        "warmup_epochs": int(args.warmup_epochs),
        "admm_epochs": int(args.admm_epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "device": str(args.device),
        "tag": str(args.tag),
    }
    if args.smoke:
        config.update(
            {
                "datasets": ["xor"],
                "n_samples": 80,
                "n_features": 8,
                "folds": 2,
                "c_grid": [0.05],
                "warmup_epochs": 0,
                "admm_epochs": 5,
                "stage_epochs": 1,
                "hidden_width": 8,
                "hidden_layers": 1,
                "refit_epochs": 2,
                "fold_limit": 1,
                "tag": f"{args.tag}_smoke",
                "evidence_status": "non_evidence_smoke",
            }
        )
    return config


def validate_frozen_config(config: dict[str, Any]) -> None:
    mismatches = []
    for key, frozen_value in FORMAL_CONFIG.items():
        if config.get(key) != frozen_value:
            mismatches.append(
                f"{key}: received {config.get(key)!r}, frozen {frozen_value!r}"
            )
    if mismatches:
        raise ValueError("formal run differs from frozen v3:\n  " + "\n  ".join(mismatches))
    _validate_schedule_shape(config)


def _validate_schedule_shape(config: dict[str, Any]) -> None:
    stages = len(config["stage_multipliers"])
    expected = int(config["stage_epochs"]) * stages
    if int(config["admm_epochs"]) != expected:
        raise ValueError(
            f"admm_epochs must equal stage_epochs * {stages}; got "
            f"{config['admm_epochs']} versus {expected}"
        )


def build_c_schedule(endpoint: float, trajectory: str, config: dict[str, Any]) -> list[float]:
    if not math.isfinite(endpoint) or endpoint <= 0.0:
        raise ValueError("endpoint must be finite and positive")
    _validate_schedule_shape(config)
    if trajectory == "F":
        schedule = [float(endpoint)] * int(config["admm_epochs"])
    elif trajectory == "P":
        schedule = []
        for multiplier in config["stage_multipliers"]:
            schedule.extend(
                [float(endpoint) * float(multiplier)] * int(config["stage_epochs"])
            )
    else:
        raise ValueError("trajectory must be 'F' or 'P'")
    if len(schedule) != int(config["admm_epochs"]):
        raise RuntimeError("constructed C schedule has the wrong length")
    if schedule[-1] != float(endpoint):
        raise RuntimeError("constructed C schedule does not end at endpoint")
    return schedule


def stage_endpoint_epochs(config: dict[str, Any]) -> list[int]:
    return [
        int(config["stage_epochs"]) * (stage + 1) - 1
        for stage in range(len(config["stage_multipliers"]))
    ]


def _model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_size": config["n_features"],
        "n_classes": 2,
        "latent_size": config["hidden_width"],
        "n_hidden_layers": config["hidden_layers"],
        "gaussian_noise": 0.0,
        "dropout": 0.0,
        "feat_drop": 0.0,
        "activation": "mish",
        "bounded_gate": False,
        "layer_norm": 0,
        "first_layer_spectral_scale": "none",
    }


def build_matched_trajectory_models(
    config: dict[str, Any], seed: int, probe: np.ndarray
) -> tuple[GatedExplicitScaleMLP, GatedExplicitScaleMLP, dict[str, Any]]:
    _set_determinism(seed)
    model_f = GatedExplicitScaleMLP(**_model_kwargs(config))
    model_p = copy.deepcopy(model_f)
    model_f.eval()
    model_p.eval()
    with torch.no_grad():
        probe_tensor = torch.as_tensor(probe, dtype=torch.float32)
        logits_delta = torch.max(
            torch.abs(model_f(probe_tensor) - model_p(probe_tensor))
        ).item()
        gate_delta = torch.max(
            torch.abs(model_f.get_effective_gate() - model_p.get_effective_gate())
        ).item()
    diagnostics = {
        "seed": int(seed),
        "state_dict_sha256_f": _state_dict_hash(model_f.state_dict()),
        "state_dict_sha256_p": _state_dict_hash(model_p.state_dict()),
        "logits_max_abs_delta": float(logits_delta),
        "effective_gate_max_abs_delta": float(gate_delta),
        "byte_identical_state": all(
            torch.equal(model_f.state_dict()[name], model_p.state_dict()[name])
            for name in model_f.state_dict()
        ),
    }
    if (
        not diagnostics["byte_identical_state"]
        or diagnostics["state_dict_sha256_f"] != diagnostics["state_dict_sha256_p"]
        or logits_delta != 0.0
        or gate_delta != 0.0
    ):
        raise RuntimeError(f"F/P initialization mismatch: {diagnostics}")
    return model_f, model_p, diagnostics


def _module_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _supervised_metrics(
    model: nn.Module, X: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    device = _module_device(model)
    with torch.no_grad():
        logits = model(
            torch.as_tensor(X, dtype=torch.float32, device=device)
        ).reshape(-1)
        targets = torch.as_tensor(y, dtype=torch.float32, device=device)
        bce = F.binary_cross_entropy_with_logits(logits, targets).item()
        probability = torch.sigmoid(logits).cpu().numpy()
    model.train(was_training)
    return {
        "bce": float(bce),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
    }


def _supervised_bce(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    was_training = model.training
    model.eval()
    device = _module_device(model)
    with torch.no_grad():
        logits = model(
            torch.as_tensor(X, dtype=torch.float32, device=device)
        ).reshape(-1)
        targets = torch.as_tensor(y, dtype=torch.float32, device=device)
        bce = F.binary_cross_entropy_with_logits(logits, targets).item()
    model.train(was_training)
    return float(bce)


class EpochTraceRecorder:
    def __init__(self, X_validation: np.ndarray | None, y_validation: np.ndarray | None):
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.rows: list[dict[str, Any]] = []

    def __call__(self, model: nn.Module, snapshot: dict[str, Any]) -> None:
        scores = snapshot["effective_gate"].abs()
        selected, selection = deterministic_topk(
            scores,
            min(2, scores.numel()),
            zero_tol=0.0,
            readout="effective_gate_epoch",
        )
        row = {
            **snapshot,
            "topk": [int(item) for item in selected.tolist()],
            "topk_metadata": selection,
            "validation_supervised_loss": None,
            "validation_role": "not_evaluated_on_outer_final_fit",
        }
        if self.X_validation is not None and self.y_validation is not None:
            row["validation_supervised_loss"] = _supervised_bce(
                model, self.X_validation, self.y_validation
            )
            row["validation_role"] = "inner_validation_not_used_for_updates"
        self.rows.append(_jsonable(row))


def train_trajectory(
    model: GatedExplicitScaleMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
    endpoint: float,
    trajectory: str,
    seed: int,
    X_validation: np.ndarray | None = None,
    y_validation: np.ndarray | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    schedule = build_c_schedule(endpoint, trajectory, config)
    recorder = EpochTraceRecorder(X_validation, y_validation)
    _set_determinism(seed)
    start = time.perf_counter()
    _train_input_group(
        model,
        X_train.astype(np.float32, copy=False),
        y_train.astype(np.int64, copy=False),
        2,
        lr=config["lr"],
        weight_decay=0.0,
        C=schedule[0],
        C_schedule_per_epoch=schedule,
        epochs=config["warmup_epochs"] + config["admm_epochs"],
        warmup_epochs=config["warmup_epochs"],
        batch_size=config["batch_size"],
        g_inner_epochs=1,
        rho_init=config["rho"],
        device=str(config.get("device", "cpu")),
        use_ratio_norm=True,
        ratio_norm_update="canonical_signed",
        ratio_norm_dim_scale="none",
        use_admm=True,
        uniform_penalty=True,
        three_variable=False,
        write_z_to_gate=False,
        optimizer_type="adam",
        use_early_stopping=False,
        min_z_support=0,
        admm_weighted_gate=False,
        adaptive_rho=False,
        auto_C_gamma=None,
        adaptive_C=False,
        dynamic_C_mode="none",
        group_sparse_mode="none",
        diagnostic_ground_truth=None,
        diagnostic_k=0,
        lr_scheduler="none",
        rho_follow_lr=False,
        strict_neural_block_mode="joint_minibatch",
        strict_kkt_check_interval=0,
        epoch_end_callback=recorder,
    )
    elapsed = time.perf_counter() - start
    state = getattr(model, "admm_state", None)
    if not isinstance(state, dict):
        raise RuntimeError("training did not publish admm_state")
    if state.get("ratio_norm_update") != "canonical_signed":
        raise RuntimeError("training left canonical_signed")
    if state.get("C_schedule_per_epoch") != schedule:
        raise RuntimeError("recorded C schedule differs from requested schedule")
    if len(recorder.rows) != config["admm_epochs"]:
        raise RuntimeError(
            f"expected {config['admm_epochs']} epoch traces, got {len(recorder.rows)}"
        )
    if float(state["C_used"]) != float(endpoint):
        raise RuntimeError("training did not end at the common endpoint")
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"non-finite parameter after training: {name}")
    requested_device = torch.device(str(config.get("device", "cpu")))
    actual_device = _module_device(model)
    if actual_device != requested_device:
        raise RuntimeError(
            f"training device mismatch: requested {requested_device}, got {actual_device}"
        )
    return float(elapsed), recorder.rows


def select_common_endpoint(
    rows: Sequence[dict[str, Any]], tolerance: float = 1e-12
) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "success"]
    if not successful:
        raise RuntimeError("all endpoint calibration pairs failed")
    minimum_loss = min(float(row["mean_validation_bce"]) for row in successful)
    tied = [
        row
        for row in successful
        if float(row["mean_validation_bce"]) <= minimum_loss + float(tolerance)
    ]
    return min(tied, key=lambda row: float(row["endpoint"]))


def path_scores_from_trace(
    trace: Sequence[dict[str, Any]], config: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    by_epoch = {int(row["epoch"]): row for row in trace}
    endpoints = stage_endpoint_epochs(config)
    missing = [epoch for epoch in endpoints if epoch not in by_epoch]
    if missing:
        raise ValueError(f"path trace is missing stage endpoints: {missing}")
    normalized = []
    endpoint_topk: list[list[int]] = []
    endpoint_ranks: list[np.ndarray] = []
    active_any = None
    for epoch in endpoints:
        gate = torch.as_tensor(by_epoch[epoch]["effective_gate"], dtype=torch.float64).abs()
        active = gate > float(config["readout_zero_tolerance"])
        active_any = active if active_any is None else (active_any | active)
        normalized.append(gate / (gate.sum() + float(config["path_epsilon"])))
        chosen, _ = deterministic_topk(
            gate,
            int(config["readout_k"]),
            zero_tol=float(config["readout_zero_tolerance"]),
            readout="stage_endpoint_effective_gate",
        )
        endpoint_topk.append([int(item) for item in chosen.tolist()])
        endpoint_ranks.append(rankdata(-gate.numpy(), method="average"))
    score = torch.stack(normalized).mean(dim=0)
    score = torch.where(active_any, score, torch.zeros_like(score))
    consecutive_jaccard = []
    for left, right in zip(endpoint_topk, endpoint_topk[1:]):
        left_set = set(left)
        right_set = set(right)
        union = left_set | right_set
        consecutive_jaccard.append(
            1.0 if not union else float(len(left_set & right_set) / len(union))
        )
    rank_correlation: list[list[float | None]] = []
    for left in endpoint_ranks:
        row: list[float | None] = []
        for right in endpoint_ranks:
            if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
                row.append(None)
            else:
                row.append(float(np.corrcoef(left, right)[0, 1]))
        rank_correlation.append(row)
    return score, {
        "stage_endpoint_epochs": endpoints,
        "normalized_endpoint_scores": [row.tolist() for row in normalized],
        "active_any": active_any.tolist(),
        "endpoint_topk": endpoint_topk,
        "consecutive_topk_jaccard": consecutive_jaccard,
        "endpoint_rank_correlation": rank_correlation,
        "path_epsilon": float(config["path_epsilon"]),
    }


def select_scores(
    scores: torch.Tensor, config: dict[str, Any], readout: str
) -> tuple[list[int], dict[str, Any]]:
    selected, metadata = deterministic_topk(
        scores,
        int(config["readout_k"]),
        zero_tol=float(config["readout_zero_tolerance"]),
        readout=readout,
    )
    return [int(item) for item in selected.tolist()], metadata


class PostSelectionMLP(nn.Module):
    """Plain no-gate refit model used identically for every readout."""

    def __init__(self, input_size: int, hidden_width: int, hidden_layers: int):
        super().__init__()
        self.input_size = int(input_size)
        if self.input_size == 0:
            self.bias_only = nn.Parameter(torch.zeros(()))
            self.network = None
            return
        layers: list[nn.Module] = [nn.Linear(self.input_size, hidden_width), nn.Mish()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.Mish()])
        layers.append(nn.Linear(hidden_width, 1))
        self.network = nn.Sequential(*layers)
        self.register_parameter("bias_only", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.network is None:
            return self.bias_only.expand(x.shape[0])
        return self.network(x).reshape(-1)


def fit_post_selection_model(
    selected: Sequence[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    canonical_selected = sorted({int(item) for item in selected})
    X_train_selected = X_train[:, canonical_selected].astype(np.float32, copy=False)
    X_test_selected = X_test[:, canonical_selected].astype(np.float32, copy=False)
    _set_determinism(seed)
    model = PostSelectionMLP(
        len(canonical_selected), config["hidden_width"], config["hidden_layers"]
    )
    requested_device = torch.device(str(config.get("device", "cpu")))
    model.to(requested_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["refit_lr"]),
        weight_decay=float(config["refit_weight_decay"]),
    )
    dataset = TensorDataset(
        torch.as_tensor(X_train_selected, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["refit_batch_size"]),
        shuffle=True,
        generator=generator,
    )
    loss_history = []
    for _epoch in range(int(config["refit_epochs"])):
        model.train()
        total = 0.0
        count = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(requested_device)
            y_batch = y_batch.to(requested_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = F.binary_cross_entropy_with_logits(logits, y_batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item()) * int(x_batch.shape[0])
            count += int(x_batch.shape[0])
        loss_history.append(total / max(count, 1))

    train_metrics = _supervised_metrics(model, X_train_selected, y_train)
    test_metrics = _supervised_metrics(model, X_test_selected, y_test)
    return {
        "selected_sorted": canonical_selected,
        "seed": int(seed),
        "model_kind": "bias_only" if not canonical_selected else "plain_mlp_no_gate",
        "epochs": int(config["refit_epochs"]),
        "batch_size": int(config["refit_batch_size"]),
        "lr": float(config["refit_lr"]),
        "weight_decay": float(config["refit_weight_decay"]),
        "execution_device": str(_module_device(model)),
        "train_loss_history": loss_history,
        "train": train_metrics,
        "test": test_metrics,
        "state_dict_sha256": _state_dict_hash(model.state_dict()),
    }


def _selection_evaluation(
    selected: Sequence[int],
    selection: dict[str, Any],
    scores: torch.Tensor,
    refit: dict[str, Any],
) -> dict[str, Any]:
    selected_set = set(int(item) for item in selected)
    recovered = len(selected_set.intersection(GROUND_TRUTH))
    return {
        "selected": [int(item) for item in selected],
        "selection": selection,
        "scores": scores.detach().cpu().tolist(),
        "recovered_count": int(recovered),
        "recall_at_2": float(recovered / 2.0),
        "exact_recovery": int(selected_set == set(GROUND_TRUTH)),
        "selected_count": len(selected),
        "post_selection_refit": refit,
    }


def _save_checkpoint(
    path: Path,
    model: GatedExplicitScaleMLP,
    dataset: str,
    fold: int,
    trajectory: str,
    endpoint: float,
    trace: Sequence[dict[str, Any]],
) -> tuple[str, str]:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "dataset": dataset,
        "fold": int(fold),
        "trajectory": trajectory,
        "endpoint": float(endpoint),
        "model_class": type(model).__name__,
        "model_state_dict": model.state_dict(),
        "admm_state": model.admm_state,
        "epoch_trace": list(trace),
    }
    torch.save(payload, path)
    return _sha256_file(path), _state_dict_hash(model.state_dict())


def _provenance_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def calibrate_endpoint(
    dataset: str,
    fold: int,
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    config: dict[str, Any],
    seed: int,
    writer: JsonlWriter,
) -> tuple[float, dict[str, Any]]:
    offset = DATASET_OFFSETS[dataset]
    split_seed = config["seed"] + offset + 2003 * (fold + 1)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(config["inner_validation_fraction"]),
        random_state=split_seed,
    )
    inner_train_index, inner_val_index = next(
        splitter.split(X_outer_train, y_outer_train)
    )
    (X_inner, X_val), scaler = _fit_scaler(
        X_outer_train[inner_train_index], X_outer_train[inner_val_index]
    )
    y_inner = y_outer_train[inner_train_index]
    y_val = y_outer_train[inner_val_index]
    split_hash = _array_hash(inner_train_index, inner_val_index)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for endpoint in config["c_grid"]:
        try:
            model_f, model_p, initialization = build_matched_trajectory_models(
                config, seed, X_inner[:16]
            )
            runtime_f, trace_f = train_trajectory(
                model_f,
                X_inner,
                y_inner,
                config,
                endpoint,
                "F",
                seed,
                X_val,
                y_val,
            )
            runtime_p, trace_p = train_trajectory(
                model_p,
                X_inner,
                y_inner,
                config,
                endpoint,
                "P",
                seed,
                X_val,
                y_val,
            )
            validation_f = _supervised_metrics(model_f, X_val, y_val)
            validation_p = _supervised_metrics(model_p, X_val, y_val)
            row = {
                "event": "calibration_endpoint_pair",
                "status": "success",
                "dataset": dataset,
                "fold": int(fold),
                "endpoint": float(endpoint),
                "seed": int(seed),
                "inner_split_seed": int(split_seed),
                "inner_split_sha256": split_hash,
                "inner_scaler_sha256": scaler["statistics_sha256"],
                "initialization": initialization,
                "validation_f": validation_f,
                "validation_p": validation_p,
                "mean_validation_bce": float(
                    (validation_f["bce"] + validation_p["bce"]) / 2.0
                ),
                "runtime_seconds_f": runtime_f,
                "runtime_seconds_p": runtime_p,
                "trace_f": trace_f,
                "trace_p": trace_p,
                "distinct_training_trajectory_count": 2,
                "p_final_and_p_path_counted_once": True,
            }
            rows.append(row)
            writer.write(row)
        except Exception as exc:
            failure = {
                "event": "calibration_endpoint_pair",
                "status": "failure",
                "dataset": dataset,
                "fold": int(fold),
                "endpoint": float(endpoint),
                "seed": int(seed),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            writer.write(failure)
    selected = select_common_endpoint(
        rows, tolerance=float(config["endpoint_tie_tolerance"])
    )
    decision = {
        "event": "calibration_decision",
        "dataset": dataset,
        "fold": int(fold),
        "selected_endpoint": float(selected["endpoint"]),
        "selected_mean_validation_bce": float(selected["mean_validation_bce"]),
        "valid_endpoint_count": len(rows),
        "failed_endpoint_count": len(failures),
        "selection_rule": "mean full supervised BCE across distinct F and P trajectories; smaller endpoint on tolerance tie",
        "tie_tolerance": float(config["endpoint_tie_tolerance"]),
        "inner_split_sha256": split_hash,
        "inner_scaler_sha256": scaler["statistics_sha256"],
    }
    writer.write(decision)
    return float(selected["endpoint"]), decision


def final_fold_rows(
    dataset: str,
    fold: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    config: dict[str, Any],
    endpoint: float,
    seed: int,
    checkpoint_dir: Path,
    split_hash: str,
    writer: JsonlWriter,
) -> list[dict[str, Any]]:
    (X_train, X_test), scaler = _fit_scaler(X_train_raw, X_test_raw)
    model_f, model_p, initialization = build_matched_trajectory_models(
        config, seed, X_train[:16]
    )
    runtime_f, trace_f = train_trajectory(
        model_f, X_train, y_train, config, endpoint, "F", seed
    )
    runtime_p, trace_p = train_trajectory(
        model_p, X_train, y_train, config, endpoint, "P", seed
    )

    final_scores_f = model_f.get_effective_gate(training_drop=False).detach().abs()
    final_scores_p = model_p.get_effective_gate(training_drop=False).detach().abs()
    path_scores, path_details = path_scores_from_trace(trace_p, config)
    selected_f, selection_f = select_scores(final_scores_f, config, "final_effective_gate")
    selected_p_final, selection_p_final = select_scores(
        final_scores_p, config, "final_effective_gate"
    )
    selected_p_path, selection_p_path = select_scores(
        path_scores, config, "five_stage_path_mean"
    )

    checkpoint_f = checkpoint_dir / f"{dataset}_fold{fold}_F.pt"
    checkpoint_p = checkpoint_dir / f"{dataset}_fold{fold}_P.pt"
    sha_f, state_sha_f = _save_checkpoint(
        checkpoint_f, model_f, dataset, fold, "F", endpoint, trace_f
    )
    sha_p, state_sha_p = _save_checkpoint(
        checkpoint_p, model_p, dataset, fold, "P", endpoint, trace_p
    )

    refit_seed = seed + 1_000_000
    refit_f = fit_post_selection_model(
        selected_f, X_train, y_train, X_test, y_test, config, refit_seed
    )
    refit_p_final = fit_post_selection_model(
        selected_p_final, X_train, y_train, X_test, y_test, config, refit_seed
    )
    refit_p_path = fit_post_selection_model(
        selected_p_path, X_train, y_train, X_test, y_test, config, refit_seed
    )

    common = {
        "event": "final_variant",
        "status": "success",
        "dataset": dataset,
        "fold": int(fold),
        "selected_endpoint": float(endpoint),
        "split_sha256": split_hash,
        "outer_scaler_sha256": scaler["statistics_sha256"],
        "initialization": initialization,
        "refit_seed": int(refit_seed),
    }
    rows = [
        {
            **common,
            "variant": "F",
            "training_trajectory": "F",
            "checkpoint": _provenance_path(checkpoint_f),
            "checkpoint_sha256": sha_f,
            "state_dict_sha256": state_sha_f,
            "runtime_seconds": runtime_f,
            "schedule": build_c_schedule(endpoint, "F", config),
            "epoch_trace": trace_f,
            "full_model_train": _supervised_metrics(model_f, X_train, y_train),
            "full_model_test": _supervised_metrics(model_f, X_test, y_test),
            "evaluation": _selection_evaluation(
                selected_f, selection_f, final_scores_f, refit_f
            ),
        },
        {
            **common,
            "variant": "P-final",
            "training_trajectory": "P",
            "checkpoint": _provenance_path(checkpoint_p),
            "checkpoint_sha256": sha_p,
            "state_dict_sha256": state_sha_p,
            "runtime_seconds": runtime_p,
            "schedule": build_c_schedule(endpoint, "P", config),
            "epoch_trace": trace_p,
            "full_model_train": _supervised_metrics(model_p, X_train, y_train),
            "full_model_test": _supervised_metrics(model_p, X_test, y_test),
            "evaluation": _selection_evaluation(
                selected_p_final, selection_p_final, final_scores_p, refit_p_final
            ),
        },
        {
            **common,
            "variant": "P-path",
            "training_trajectory": "P",
            "checkpoint": _provenance_path(checkpoint_p),
            "checkpoint_sha256": sha_p,
            "state_dict_sha256": state_sha_p,
            "runtime_seconds": 0.0,
            "schedule": build_c_schedule(endpoint, "P", config),
            "training_record_ref": {
                "dataset": dataset,
                "fold": int(fold),
                "variant": "P-final",
            },
            "same_checkpoint_as_p_final": True,
            "path_details": path_details,
            "evaluation": _selection_evaluation(
                selected_p_path, selection_p_path, path_scores, refit_p_path
            ),
        },
    ]
    if rows[1]["checkpoint_sha256"] != rows[2]["checkpoint_sha256"]:
        raise RuntimeError("P-final and P-path checkpoint identity failed")
    for row in rows:
        writer.write(row)
    return rows


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_provenance() -> dict[str, Any]:
    missing_sources = [
        relative for relative in RELEVANT_SOURCE_PATHS if not (ROOT / relative).is_file()
    ]
    source_hashes = {
        relative: _sha256_file(ROOT / relative)
        for relative in RELEVANT_SOURCE_PATHS
        if (ROOT / relative).is_file()
    }
    dirty = _git("status", "--porcelain")
    relevant_dirty = _git("status", "--porcelain", "--", *RELEVANT_SOURCE_PATHS)
    diff = _git("diff", "--binary", "--", *RELEVANT_SOURCE_PATHS)
    return {
        "argv": list(sys.argv),
        "command_line": subprocess.list2cmdline(sys.argv),
        "working_directory": str(Path.cwd()),
        "git_commit": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(dirty),
        "relevant_paths_dirty": bool(relevant_dirty),
        "relevant_status": relevant_dirty or "",
        "relevant_tracked_diff_sha256": _sha256_bytes((diff or "").encode()),
        "source_sha256": source_hashes,
        "missing_relevant_sources": missing_sources,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _assert_formal_source_clean(provenance: dict[str, Any]) -> None:
    if provenance["git_commit"] is None:
        raise RuntimeError("formal run requires a Git commit")
    if provenance.get("missing_relevant_sources"):
        raise RuntimeError(
            "formal run requires every frozen source; missing paths:\n"
            + "\n".join(provenance["missing_relevant_sources"])
        )
    if provenance["relevant_paths_dirty"]:
        raise RuntimeError(
            "formal run requires committed relevant sources; dirty paths:\n"
            + provenance["relevant_status"]
        )


def _read_final_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") == "final_variant" and event.get("status") == "success":
                rows.append(event)
    return rows


def _summary_stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "count": int(array.size),
    }


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"by_variant": {}, "comparisons": {}}
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        by_dataset = {}
        for dataset in sorted({row["dataset"] for row in variant_rows}):
            dataset_rows = [row for row in variant_rows if row["dataset"] == dataset]
            by_dataset[dataset] = {
                "recall_at_2": _summary_stats(
                    [row["evaluation"]["recall_at_2"] for row in dataset_rows]
                ),
                "post_selection_test_auroc": _summary_stats(
                    [
                        row["evaluation"]["post_selection_refit"]["test"]["auroc"]
                        for row in dataset_rows
                    ]
                ),
            }
        dataset_recall_means = [
            item["recall_at_2"]["mean"] for item in by_dataset.values()
        ]
        dataset_auc_means = [
            item["post_selection_test_auroc"]["mean"] for item in by_dataset.values()
        ]
        result["by_variant"][variant] = {
            "by_dataset": by_dataset,
            "macro_recall_at_2": float(np.mean(dataset_recall_means)),
            "macro_post_selection_test_auroc": float(np.mean(dataset_auc_means)),
            "total_recovered_coordinates": int(
                sum(row["evaluation"]["recovered_count"] for row in variant_rows)
            ),
            "exact_recovery_count": int(
                sum(row["evaluation"]["exact_recovery"] for row in variant_rows)
            ),
        }
    comparisons = (("F", "P-final"), ("P-final", "P-path"), ("F", "P-path"))
    for left, right in comparisons:
        result["comparisons"][f"{left}_minus_{right}"] = {
            "macro_recall_at_2": float(
                result["by_variant"][left]["macro_recall_at_2"]
                - result["by_variant"][right]["macro_recall_at_2"]
            ),
            "macro_post_selection_test_auroc": float(
                result["by_variant"][left]["macro_post_selection_test_auroc"]
                - result["by_variant"][right]["macro_post_selection_test_auroc"]
            ),
            "single_factor_interpretation": (
                "schedule"
                if (left, right) == ("F", "P-final")
                else ("readout" if (left, right) == ("P-final", "P-path") else None)
            ),
        }
    return result


def run_experiment(config: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    _validate_schedule_shape(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = f"{config['tag']}_{timestamp}"
    config_path = output_dir / f"{prefix}.config.json"
    jsonl_path = output_dir / f"{prefix}.jsonl"
    summary_path = output_dir / f"{prefix}.json"
    log_path = output_dir / f"{prefix}.log"
    checkpoint_dir = output_dir / "checkpoints" / prefix
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    provenance = collect_provenance()
    normalized = {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now().isoformat(),
        "config": config,
        "provenance": provenance,
    }
    config_path.write_text(
        json.dumps(_jsonable(normalized), indent=2, sort_keys=True), encoding="utf-8"
    )
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    logging.info("normalized config: %s", json.dumps(_jsonable(normalized), sort_keys=True))
    writer = JsonlWriter(jsonl_path)
    writer.write({"event": "run_start", **normalized})

    try:
        all_rows: list[dict[str, Any]] = []
        for dataset in config["datasets"]:
            offset = DATASET_OFFSETS[dataset]
            dataset_seed = config["seed"] + offset
            _set_determinism(dataset_seed)
            X, _, y = generate_dataset(
                dataset, config["n_samples"], config["n_features"]
            )
            X = X.astype(np.float32)
            y = y.astype(np.int64)
            dataset_hash = _array_hash(X, y)
            split_seed = config["seed"] + offset + 101
            outer = StratifiedKFold(
                n_splits=config["folds"], shuffle=True, random_state=split_seed
            )
            for fold, (train_index, test_index) in enumerate(outer.split(X, y)):
                if config["fold_limit"] is not None and fold >= config["fold_limit"]:
                    break
                split_hash = _array_hash(train_index, test_index)
                writer.write(
                    {
                        "event": "outer_fold_start",
                        "dataset": dataset,
                        "fold": int(fold),
                        "dataset_seed": int(dataset_seed),
                        "dataset_sha256": dataset_hash,
                        "split_seed": int(split_seed),
                        "split_sha256": split_hash,
                    }
                )
                calibration_seed = config["seed"] + offset + 1009 * (fold + 1)
                selected_endpoint, _decision = calibrate_endpoint(
                    dataset,
                    fold,
                    X[train_index],
                    y[train_index],
                    config,
                    calibration_seed,
                    writer,
                )
                training_seed = calibration_seed + 50_000
                rows = final_fold_rows(
                    dataset,
                    fold,
                    X[train_index],
                    y[train_index],
                    X[test_index],
                    y[test_index],
                    config,
                    selected_endpoint,
                    training_seed,
                    checkpoint_dir,
                    split_hash,
                    writer,
                )
                all_rows.extend(rows)

        durable_rows = _read_final_rows(jsonl_path)
        expected = (
            len(config["datasets"])
            * (config["fold_limit"] or config["folds"])
            * len(VARIANTS)
        )
        if len(durable_rows) != expected:
            raise RuntimeError(f"expected {expected} final rows, found {len(durable_rows)}")
        summary = {
            "protocol_id": PROTOCOL_ID,
            "protocol_version": PROTOCOL_VERSION,
            "status": "success",
            "evidence_status": config["evidence_status"],
            "config_path": str(config_path),
            "jsonl_path": str(jsonl_path),
            "log_path": str(log_path),
            "checkpoint_dir": str(checkpoint_dir),
            "aggregate": aggregate_rows(durable_rows),
        }
        summary_path.write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8"
        )
        writer.write(
            {"event": "run_complete", "status": "success", "summary_path": str(summary_path)}
        )
    except Exception as exc:
        writer.write(
            {
                "event": "run_failure",
                "status": "implementation_failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        failure = {
            "protocol_id": PROTOCOL_ID,
            "protocol_version": PROTOCOL_VERSION,
            "status": "implementation_failure",
            "evidence_status": config["evidence_status"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "config_path": str(config_path),
            "jsonl_path": str(jsonl_path),
            "log_path": str(log_path),
        }
        summary_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise
    return {
        "config": config_path,
        "jsonl": jsonl_path,
        "summary": summary_path,
        "log": log_path,
        "checkpoints": checkpoint_dir,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_default_dtype(torch.float32)
    _set_determinism(args.seed)
    config = normalized_config(args)
    print(
        json.dumps(
            _jsonable({"protocol_id": PROTOCOL_ID, "config": config}),
            indent=2,
            sort_keys=True,
        )
    )
    if args.validate_only:
        validate_frozen_config(config)
        print("Frozen v3 config validation passed; no files written and no training run.")
        return 0
    if args.formal_run:
        validate_frozen_config(config)
        provenance = collect_provenance()
        _assert_formal_source_clean(provenance)
        expected_output = (ROOT / "results" / "ablation").resolve()
        if args.output_dir.resolve() != expected_output:
            raise ValueError(f"formal output-dir is frozen to {expected_output}")
    if args.smoke:
        _assert_smoke_output_is_non_evidence(args.output_dir)
    paths = run_experiment(config, args.output_dir.resolve())
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
