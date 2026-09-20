# -*- coding: utf-8 -*-
"""May 10 run-8 diagnostics: phase stops, single-C, warmup, lottery reset.

Base run: full-poly2 no-WN run 8 settings.
This script keeps the run-8 model/training protocol fixed unless the mode name
states the changed axis.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path = [str(ROOT), str(ROOT / "Feature-Selection-Benchmark")] + sys.path

from experiments.ablation.run_multiphase_fixed_c_rho_poly2_only_m128 import (  # noqa: E402
    AGGREGATION_NAMES,
    DATASET_GT,
    aggregate_max,
    build_poly_maps,
    create_synthetic_dataset,
    evaluate_all_aggregations,
    first_column_norm,
    parse_float_list,
    predict_proba,
    safe_auc,
    safe_auprc,
    set_seed,
    topk_from_score,
)
from src.admm_input_group_wrapper import GatedFeatureSelectionMLP  # noqa: E402
from src.gradual_admm_with_pruning import train_one_phase_admm  # noqa: E402


DEFAULT_MODES = ["phase_stop", "single_c", "warmup0", "lottery_z_reset"]


class MaskedGatedFeatureSelectionMLP(GatedFeatureSelectionMLP):
    """Run-8 model with a fixed expanded-feature mask in the effective gate."""

    def __init__(self, *args: Any, fixed_gate_mask: np.ndarray | torch.Tensor, **kwargs: Any):
        super().__init__(*args, **kwargs)
        mask = torch.as_tensor(fixed_gate_mask, dtype=torch.float32)
        if mask.numel() != self.gate.numel():
            raise ValueError(
                f"fixed_gate_mask has {mask.numel()} entries, expected {self.gate.numel()}"
            )
        self.register_buffer("fixed_gate_mask", mask.reshape_as(self.gate))

    def gate_from_parameter(self, gate_param: torch.Tensor) -> torch.Tensor:
        gate = super().gate_from_parameter(gate_param)
        return gate * self.fixed_gate_mask.to(device=gate.device, dtype=gate.dtype)


def model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "latent_size": 32,
        "n_hidden_layers": 2,
        "dropout": args.dropout,
        "activation": "mish",
        "feat_drop": args.feat_drop,
        "bounded_gate": False,
        "column_normalize_first_layer": False,
    }


def train_phase(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    device: str,
    C: float,
    warmup_epochs: int,
) -> dict[str, Any]:
    phase_history = train_one_phase_admm(
        model,
        X_train,
        y_train,
        dataset["n_classes"],
        C=C,
        epochs=args.epochs_per_phase,
        warmup_epochs=warmup_epochs,
        lr=args.lr,
        device=device,
        use_ratio_norm=True,
        use_admm=True,
        uniform_penalty=False,
        rho_init=args.rho_init,
        write_z_to_gate=False,
        initial_admm_state=None,
        min_z_support=0,
        admm_weighted_gate=False,
        ratio_norm_update="current",
        adaptive_rho=False,
        adaptive_C=False,
        group_sparse_powers=dataset["poly_powers"],
        group_sparse_mode="none",
        group_parent_support=0,
        diagnostic_ground_truth=dataset["ground_truth"],
        diagnostic_k=dataset["k"],
        lr_scheduler="none",
        rho_follow_lr=False,
    )
    return phase_history


def effective_gate(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "get_gate_values"):
        with torch.no_grad():
            return model.get_gate_values(training_drop=False).detach().cpu()
    return model.gate.detach().cpu()


def evaluate_model(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
) -> dict[str, Any]:
    gate = effective_gate(model)
    col_norm = first_column_norm(model)
    aggregations = evaluate_all_aggregations(
        gate=gate,
        col_norm=col_norm,
        powers=dataset["poly_powers"],
        ground_truth=dataset["ground_truth"],
        k=dataset["k"],
    )
    primary = aggregations["parent_weight_sum"]
    admm_state = getattr(model, "admm_state", {})
    if not isinstance(admm_state, dict):
        admm_state = {}

    z_parent_max = None
    z_support = None
    parent_z_support = None
    if torch.is_tensor(admm_state.get("zk")):
        z = admm_state["zk"].detach().cpu()
        z_support = int((z.abs() > 0).sum().item())
        maps = build_poly_maps(dataset["poly_powers"])
        z_parent_max = topk_from_score(
            aggregate_max(z.abs(), maps["parent_terms"]),
            dataset["ground_truth"],
            dataset["k"],
        )
    if "parent_z_support" in admm_state:
        parent_z_support = int(admm_state["parent_z_support"])

    y_train_score = predict_proba(model, X_train, device)
    y_test_score = predict_proba(model, X_test, device)
    return {
        "best_k": primary["best_k"],
        "selected": primary["selected"],
        "primary_aggregation": "parent_weight_sum",
        "aggregations": aggregations,
        "z_parent_max": z_parent_max,
        "train_auc": safe_auc(y_train, y_train_score),
        "auc": safe_auc(y_test, y_test_score),
        "train_auprc": safe_auprc(y_train, y_train_score),
        "auprc": safe_auprc(y_test, y_test_score),
        "gate_nonzero": int((gate.abs() > 0).sum().item()),
        "gate_l1": float(gate.abs().sum().item()),
        "gate_l2": float(torch.linalg.vector_norm(gate.abs(), ord=2).item()),
        "z_support": z_support,
        "parent_z_support": parent_z_support,
        "final_rho": float(admm_state.get("rho", 0.0)) if admm_state else None,
        "final_C_used": float(admm_state.get("C_used", 0.0)) if admm_state else None,
    }


def summarize_evals(evals: list[dict[str, Any]]) -> dict[str, Any]:
    z_values = [e["z_support"] for e in evals if e.get("z_support") is not None]
    summary = {
        "n_folds": len(evals),
        "best_k_mean": float(np.mean([e["best_k"] for e in evals])),
        "best_k_std": float(np.std([e["best_k"] for e in evals])),
        "auc_mean": float(np.mean([e["auc"] for e in evals])),
        "auc_std": float(np.std([e["auc"] for e in evals])),
        "auprc_mean": float(np.mean([e["auprc"] for e in evals])),
        "auprc_std": float(np.std([e["auprc"] for e in evals])),
        "z_support_mean": float(np.mean(z_values)) if z_values else None,
        "z_support_std": float(np.std(z_values)) if z_values else None,
        "aggregations": {},
    }
    for name in AGGREGATION_NAMES:
        values = [e["aggregations"][name]["best_k"] for e in evals]
        summary["aggregations"][name] = {
            "best_k_mean": float(np.mean(values)),
            "best_k_std": float(np.std(values)),
            "fold_best_k": [float(v) for v in values],
        }
    z_parent_values = [
        e["z_parent_max"]["best_k"]
        for e in evals
        if isinstance(e.get("z_parent_max"), dict)
    ]
    if z_parent_values:
        summary["z_parent_max"] = {
            "best_k_mean": float(np.mean(z_parent_values)),
            "best_k_std": float(np.std(z_parent_values)),
            "fold_best_k": [float(v) for v in z_parent_values],
        }
    return summary


def run_phase_stop_fold(
    dataset: dict[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    device: str,
    c_schedule: list[float],
    fold_seed: int,
) -> dict[str, Any]:
    set_seed(fold_seed)
    X_train, y_train = dataset["X"][train_idx], dataset["y"][train_idx]
    X_test, y_test = dataset["X"][test_idx], dataset["y"][test_idx]
    model = GatedFeatureSelectionMLP(
        input_size=dataset["n_features"],
        n_classes=dataset["n_classes"],
        **model_kwargs(args),
    )
    initial_state = copy.deepcopy(model.state_dict())
    phase_evals = []
    for phase_idx, C in enumerate(c_schedule):
        train_phase(
            model=model,
            dataset=dataset,
            X_train=X_train,
            y_train=y_train,
            args=args,
            device=device,
            C=C,
            warmup_epochs=args.warmup_epochs,
        )
        phase_eval = evaluate_model(model, dataset, X_train, y_train, X_test, y_test, device)
        phase_eval.update({"phase": phase_idx, "C": float(C)})
        phase_evals.append(phase_eval)
    final_eval = phase_evals[-1]
    return {
        "phase_evals": phase_evals,
        "final_eval": final_eval,
        "initial_state_dict": initial_state,
        "final_state_dict": copy.deepcopy(model.state_dict()),
        "final_admm_state": copy.deepcopy(getattr(model, "admm_state", {})),
    }


def run_single_c_fold(
    dataset: dict[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    device: str,
    c_values: list[float],
    fold_seed: int,
) -> dict[str, Any]:
    X_train, y_train = dataset["X"][train_idx], dataset["y"][train_idx]
    X_test, y_test = dataset["X"][test_idx], dataset["y"][test_idx]
    out = {}
    for C in c_values:
        set_seed(fold_seed)
        model = GatedFeatureSelectionMLP(
            input_size=dataset["n_features"],
            n_classes=dataset["n_classes"],
            **model_kwargs(args),
        )
        train_phase(model, dataset, X_train, y_train, args, device, C, args.warmup_epochs)
        eval_row = evaluate_model(model, dataset, X_train, y_train, X_test, y_test, device)
        eval_row["C"] = float(C)
        out[str(C)] = eval_row
    return out


def run_warmup0_fold(
    dataset: dict[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    device: str,
    c_schedule: list[float],
    fold_seed: int,
) -> dict[str, Any]:
    set_seed(fold_seed)
    X_train, y_train = dataset["X"][train_idx], dataset["y"][train_idx]
    X_test, y_test = dataset["X"][test_idx], dataset["y"][test_idx]
    model = GatedFeatureSelectionMLP(
        input_size=dataset["n_features"],
        n_classes=dataset["n_classes"],
        **model_kwargs(args),
    )
    phase_evals = []
    for phase_idx, C in enumerate(c_schedule):
        train_phase(model, dataset, X_train, y_train, args, device, C, warmup_epochs=0)
        phase_eval = evaluate_model(model, dataset, X_train, y_train, X_test, y_test, device)
        phase_eval.update({"phase": phase_idx, "C": float(C)})
        phase_evals.append(phase_eval)
    return {"phase_evals": phase_evals, "final_eval": phase_evals[-1]}


def mask_from_admm_state(admm_state: dict[str, Any], fallback_score: torch.Tensor, k: int) -> torch.Tensor:
    if torch.is_tensor(admm_state.get("zk")):
        mask = admm_state["zk"].detach().cpu().abs() > 0
        if int(mask.sum().item()) > 0:
            return mask.float()
    n_keep = min(k, fallback_score.numel())
    mask = torch.zeros_like(fallback_score, dtype=torch.float32)
    mask[fallback_score.abs().topk(n_keep).indices] = 1.0
    return mask


def run_lottery_fold(
    dataset: dict[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    device: str,
    c_schedule: list[float],
    fold_seed: int,
    phase_stop_state: dict[str, Any] | None,
) -> dict[str, Any]:
    X_train, y_train = dataset["X"][train_idx], dataset["y"][train_idx]
    X_test, y_test = dataset["X"][test_idx], dataset["y"][test_idx]
    if phase_stop_state is None:
        phase_stop_state = run_phase_stop_fold(
            dataset, train_idx, test_idx, args, device, c_schedule, fold_seed
        )
    initial_state = phase_stop_state["initial_state_dict"]
    final_gate = phase_stop_state["final_state_dict"]["gate"].detach().cpu()
    mask = mask_from_admm_state(
        phase_stop_state["final_admm_state"],
        fallback_score=final_gate,
        k=dataset["k"],
    )

    set_seed(fold_seed)
    model = MaskedGatedFeatureSelectionMLP(
        input_size=dataset["n_features"],
        n_classes=dataset["n_classes"],
        fixed_gate_mask=mask.numpy(),
        **model_kwargs(args),
    )
    missing, unexpected = model.load_state_dict(initial_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected lottery state keys: {unexpected}")
    allowed_missing = {"fixed_gate_mask"}
    if set(missing) - allowed_missing:
        raise RuntimeError(f"Missing lottery state keys: {missing}")

    phase_evals = []
    for phase_idx, C in enumerate(c_schedule):
        train_phase(
            model=model,
            dataset=dataset,
            X_train=X_train,
            y_train=y_train,
            args=args,
            device=device,
            C=C,
            warmup_epochs=args.warmup_epochs,
        )
        phase_eval = evaluate_model(model, dataset, X_train, y_train, X_test, y_test, device)
        phase_eval.update({"phase": phase_idx, "C": float(C)})
        phase_evals.append(phase_eval)
    return {
        "mask_source": "final_run8_z_nonzero_expanded_support",
        "mask_support": int(mask.sum().item()),
        "phase_evals": phase_evals,
        "final_eval": phase_evals[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="*", default=DEFAULT_MODES)
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
    parser.add_argument("--rho-init", type=float, default=200.0)
    parser.add_argument("--epochs-per-phase", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--dropout", type=float, default=0.043)
    parser.add_argument("--feat-drop", type=float, default=0.0)
    parser.add_argument("--only-fold", type=int, default=None, help="1-based fold index")
    parser.add_argument("--tag", default="run8_diagnostics_20260510")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    valid_modes = set(DEFAULT_MODES)
    unknown = [m for m in args.modes if m not in valid_modes]
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}. Valid={sorted(valid_modes)}")
    if args.smoke:
        args.modes = args.modes[:1]
        args.datasets = args.datasets[:1]
        args.folds = min(args.folds, 2)
        args.epochs_per_phase = 1
        args.warmup_epochs = 0

    c_schedule = parse_float_list(args.c_schedule)
    for dataset_name in args.datasets:
        if dataset_name not in DATASET_GT:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / f"{args.tag}_{timestamp}.jsonl"
    output_file = output_dir / f"{args.tag}_{timestamp}.json"

    metadata = {
        "timestamp": timestamp,
        "experiment": "run8_diagnostics_20260510",
        "modes": args.modes,
        "datasets": args.datasets,
        "n_samples": args.n_samples,
        "n_original_features": args.n_features,
        "n_poly_features": args.n_features + args.n_features * (args.n_features + 1) // 2,
        "folds": args.folds,
        "seed": args.seed,
        "device": device,
        "base_run": "xlsx run 8",
        "model": "GatedFeatureSelectionMLP",
        "polynomial_degree": 2,
        "polynomial_include_bias": False,
        "C_schedule": c_schedule,
        "rho_init": args.rho_init,
        "adaptive_rho": False,
        "adaptive_C": False,
        "prune_mode": "none",
        "reweight": False,
        "warmup_per_phase": True,
        "epochs_per_phase": args.epochs_per_phase,
        "warmup_epochs": args.warmup_epochs,
        "lr": args.lr,
        "dropout": args.dropout,
        "feat_drop": args.feat_drop,
        "lottery_reset": "reset surviving masked model to theta_0 initialization",
        "lottery_mask_source": "final run8 z != 0 in expanded feature space",
        "checkpoint_file": str(checkpoint_file),
    }

    print("=" * 100, flush=True)
    print("MAY 10 RUN-8 DIAGNOSTICS", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Checkpoint: {checkpoint_file}", flush=True)
    print("=" * 100, flush=True)

    results: dict[str, Any] = {mode: {} for mode in args.modes}
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
            mode_folds: dict[str, list[dict[str, Any]]] = {mode: [] for mode in args.modes}
            print(
                f"\n[{dataset_name}] gt={dataset['ground_truth']} "
                f"features={dataset['n_original_features']}->{dataset['n_features']}",
                flush=True,
            )
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dataset["X"], dataset["y"])):
                if args.only_fold is not None and fold_idx != args.only_fold - 1:
                    continue
                fold_seed = args.seed + 1009 * (fold_idx + 1)
                phase_stop_state = None

                if "phase_stop" in args.modes or "lottery_z_reset" in args.modes:
                    phase_stop_state = run_phase_stop_fold(
                        dataset, train_idx, test_idx, args, device, c_schedule, fold_seed
                    )
                    if "phase_stop" in args.modes:
                        row = {
                            "mode": "phase_stop",
                            "dataset": dataset_name,
                            "fold": int(fold_idx),
                            "seed": int(fold_seed),
                            "phase_evals": phase_stop_state["phase_evals"],
                            "final_eval": phase_stop_state["final_eval"],
                        }
                        mode_folds["phase_stop"].append(row)
                        ckpt.write(json.dumps(row) + "\n")
                        ckpt.flush()
                        print(
                            f"  fold {fold_idx + 1}/{args.folds} phase_stop: "
                            f"final={row['final_eval']['best_k']:.3f} "
                            f"z={row['final_eval']['z_support']} "
                            f"AUC={row['final_eval']['auc']:.3f}",
                            flush=True,
                        )

                if "single_c" in args.modes:
                    single = run_single_c_fold(
                        dataset, train_idx, test_idx, args, device, c_schedule, fold_seed
                    )
                    row = {
                        "mode": "single_c",
                        "dataset": dataset_name,
                        "fold": int(fold_idx),
                        "seed": int(fold_seed),
                        "single_c": single,
                    }
                    mode_folds["single_c"].append(row)
                    ckpt.write(json.dumps(row) + "\n")
                    ckpt.flush()
                    readout = " ".join(
                        f"C={C}:{single[str(C)]['best_k']:.3f}" for C in c_schedule
                    )
                    print(f"  fold {fold_idx + 1}/{args.folds} single_c: {readout}", flush=True)

                if "warmup0" in args.modes:
                    warmup0 = run_warmup0_fold(
                        dataset, train_idx, test_idx, args, device, c_schedule, fold_seed
                    )
                    row = {
                        "mode": "warmup0",
                        "dataset": dataset_name,
                        "fold": int(fold_idx),
                        "seed": int(fold_seed),
                        "phase_evals": warmup0["phase_evals"],
                        "final_eval": warmup0["final_eval"],
                    }
                    mode_folds["warmup0"].append(row)
                    ckpt.write(json.dumps(row) + "\n")
                    ckpt.flush()
                    print(
                        f"  fold {fold_idx + 1}/{args.folds} warmup0: "
                        f"final={row['final_eval']['best_k']:.3f} "
                        f"z={row['final_eval']['z_support']} "
                        f"AUC={row['final_eval']['auc']:.3f}",
                        flush=True,
                    )

                if "lottery_z_reset" in args.modes:
                    lottery = run_lottery_fold(
                        dataset,
                        train_idx,
                        test_idx,
                        args,
                        device,
                        c_schedule,
                        fold_seed,
                        phase_stop_state=phase_stop_state,
                    )
                    row = {
                        "mode": "lottery_z_reset",
                        "dataset": dataset_name,
                        "fold": int(fold_idx),
                        "seed": int(fold_seed),
                        **lottery,
                    }
                    mode_folds["lottery_z_reset"].append(row)
                    ckpt.write(json.dumps(row) + "\n")
                    ckpt.flush()
                    print(
                        f"  fold {fold_idx + 1}/{args.folds} lottery_z_reset: "
                        f"mask={row['mask_support']} "
                        f"final={row['final_eval']['best_k']:.3f} "
                        f"z={row['final_eval']['z_support']} "
                        f"AUC={row['final_eval']['auc']:.3f}",
                        flush=True,
                    )

            for mode in args.modes:
                rows = mode_folds[mode]
                if mode == "single_c":
                    c_summary = {}
                    for C in c_schedule:
                        evals = [row["single_c"][str(C)] for row in rows]
                        c_summary[str(C)] = summarize_evals(evals)
                    results[mode][dataset_name] = {"folds": rows, "summary": c_summary}
                    best_text = " ".join(
                        f"C={C}:{c_summary[str(C)]['best_k_mean']:.3f}"
                        for C in c_schedule
                    )
                    print(f"  >> {dataset_name} single_c: {best_text}", flush=True)
                else:
                    evals = [row["final_eval"] for row in rows]
                    summary = summarize_evals(evals)
                    results[mode][dataset_name] = {"folds": rows, "summary": summary}
                    print(
                        f"  >> {dataset_name} {mode}: "
                        f"best_k={summary['best_k_mean']:.3f}+/-{summary['best_k_std']:.3f} "
                        f"AUC={summary['auc_mean']:.3f}+/-{summary['auc_std']:.3f} "
                        f"z={summary['z_support_mean']:.1f}",
                        flush=True,
                    )

    payload = {"metadata": metadata, "results": results}
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_file}", flush=True)
    return payload


if __name__ == "__main__":
    main()
