"""Isolated current-method minimal-change runner; never mutates frozen trainers.

Uses hash-locked integration code. New contracts are explicit diagnostic rounds,
not reruns of v3/v5. Existing source and complete results remain untouched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = ROOT.with_name(ROOT.name + "_integration")
AUDIT = ROOT / "docs/analysis/data/current_method_minimal_change_reuse_audit_20260907.json"
DROPOUT_SOURCE = "experiments/ablation/minimal_dropout_control.py"
ACTIVATION_SOURCE = "experiments/ablation/minimal_square_activation.py"
WARMSTART_SOURCE = "experiments/ablation/minimal_warmstart.py"
NUMERIC_SOURCE = "experiments/ablation/minimal_prox_numeric_candidate.py"
NUMERIC_BRIDGE_SOURCE = "experiments/ablation/minimal_numeric_training_bridge.py"
SQUARE_RELEASE_SOURCE = "experiments/ablation/minimal_square_release.py"
POSTWARM_SOURCE = "experiments/ablation/minimal_postwarm_amplitude.py"
CUE_SOURCE = "experiments/ablation/minimal_marginal_initialization.py"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned_code():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["status"] != "pass":
        raise RuntimeError("Reuse audit must pass")
    for row in audit["source_checks"]:
        if sha(INTEGRATION / row["path"]) != row["recorded_sha256"]:
            raise RuntimeError("Source drift: " + row["path"])
    for row in audit["artifacts"]:
        if sha(row["path"]) != row["sha256"]:
            raise RuntimeError("Artifact drift: " + row["path"])
    sys.path.insert(0, str(INTEGRATION))
    from experiments.ablation import run_effective_gate_c_path_v3 as v3
    from experiments.ablation import run_parent_block_quadratic_v5 as v5
    from experiments.ablation.run_parent_block_quadratic_v5_gpu_v1 import configure_gpu_runtime
    for module in (v3, v5):
        if not Path(module.__file__).resolve().is_relative_to(INTEGRATION):
            raise RuntimeError("Imported wrong worktree")
    return audit, v3, v5, configure_gpu_runtime


def validate_contract(contract):
    if contract["data_identity"] != "frozen_v3_20260824":
        raise ValueError("This runner only implements the matched v3 discovery data")
    if contract["tasks"] != ["xor", "ring"] or contract["folds"] != list(range(6)):
        raise ValueError("Do not select known successful folds")
    changed_c = any(a.get("c_multiplier", 1.0) != 1.0 for a in contract["arms"].values())
    expected_policy = ("reuse_per_fold_v3_endpoint_times_registered_global_multiplier_no_recalibration"
                       if changed_c else "reuse_exact_per_fold_v3_endpoint_no_recalibration")
    if contract["endpoint_policy"] != expected_policy:
        raise ValueError("Unexpected C policy")
    if contract["readout"] != "final_abs_effective_gate_top2" or contract["downstream_refit"]:
        raise ValueError("Selector-only final gate experiment required")
    for arm in contract["arms"].values():
        if arm.get("marginal_initialization", "none") not in ("none", "fixed_v1"):
            raise ValueError("Unregistered marginal initialization")
        if arm.get("marginal_initialization", "none") == "fixed_v1" and arm != {
                "mode": "raw", "warmup_epochs": 20, "stage_epochs": 20,
                "numeric_prox": "isolated_v1", "marginal_initialization": "fixed_v1"}:
            raise ValueError("Marginal initializer is a raw-only fixed-recipe single factor")
        if arm.get("post_warmup_amplitude", 1.0) not in (1.0, .0625):
            raise ValueError("Unregistered postwarm amplitude")
        if arm.get("post_warmup_amplitude", 1.0) != 1.0 and (
            arm.get("gate_amplitude", 1.0) != 1.0 or arm["mode"] not in ("raw", "local_quadratic")
            or arm["warmup_epochs"] != 20 or arm["stage_epochs"] != 20 or arm.get("warmup_starts", 1) != 1
            or arm.get("first_activation", "mish") != "mish" or arm.get("hidden_dropout", 0.0)
            or arm.get("gauge_scale", 1.0) != 1.0 or arm.get("hidden_width", 32) != 32
            or arm.get("lr", .005) != .005 or arm.get("rho", 50.0) != 50.0
            or arm.get("c_multiplier", 1.0) != 1.0 or arm.get("numeric_prox", "frozen") != "isolated_v1"):
            raise ValueError("Postwarm control must use the fixed canonical recipe")
        if arm.get("square_release", "immediate") not in ("immediate", "after_warmup"):
            raise ValueError("Unregistered square release policy")
        if arm.get("square_release", "immediate") == "after_warmup" and (
            arm["mode"] != "local_quadratic" or arm["warmup_epochs"] != 20
            or arm.get("warmup_starts", 1) != 1 or arm.get("first_activation", "mish") != "mish"
            or arm.get("hidden_dropout", 0.0) or arm.get("gauge_scale", 1.0) != 1.0
            or arm.get("hidden_width", 32) != 32 or arm.get("lr", .005) != .005
            or arm.get("rho", 50.0) != 50.0 or arm.get("c_multiplier", 1.0) != 1.0
            or arm.get("numeric_prox", "frozen") != "isolated_v1"):
            raise ValueError("Square release is a single fixed-recipe, explicit-numeric control")
        if arm.get("numeric_prox", "frozen") not in ("frozen", "isolated_v1"):
            raise ValueError("Unregistered numerical implementation")
        if arm.get("numeric_prox", "frozen") != "frozen" and arm.get("warmup_starts", 1) != 1:
            raise ValueError("Numerical interface currently validated for single-start only")
        if arm.get("c_multiplier", 1.0) not in (.0625, 1.0, 4.0):
            raise ValueError("Unregistered global C multiplier")
        if arm.get("c_multiplier", 1.0) != 1.0 and arm.get("warmup_starts", 1) != 1:
            raise ValueError("C-strength control does not include multistart")
        if arm.get("warmup_starts", 1) not in (1, 3):
            raise ValueError("Only single-start or fixed-three-start warm-up is registered")
        if arm.get("warmup_starts", 1) == 3 and (
            arm["warmup_epochs"] != 20 or arm.get("first_activation", "mish") != "mish"
            or arm.get("hidden_dropout", 0.0) or arm.get("gauge_scale", 1.0) != 1.0
            or arm.get("hidden_width", 32) != 32 or arm.get("lr", .005) != .005
            or arm.get("rho", 50.0) != 50.0):
            raise ValueError("Multistart must not silently accumulate withdrawn factors")
        if arm["mode"] not in ("raw", "local_quadratic", "duplicate_linear"):
            raise ValueError("Unknown representation")
        if arm["stage_epochs"] < 1 or arm["warmup_epochs"] < 0:
            raise ValueError("Invalid epoch budget")
        if arm.get("gauge_scale", 1.0) not in (.25, 1.0, 4.0):
            raise ValueError("Only the registered binary-exact gauge values are implemented")
        if arm.get("rho", 50.0) not in (.5, 5.0, 50.0):
            raise ValueError("Unregistered fixed rho")
        if arm.get("lr", .005) not in (.001, .005, .01):
            raise ValueError("Unregistered global Adam learning rate")
        if arm.get("hidden_width", 32) not in (4, 8, 32):
            raise ValueError("Unregistered hidden width")
        if arm.get("gate_amplitude", 1.0) not in (.03125, .0625, .125, .25, 1.0):
            raise ValueError("Unregistered initial effective-gate amplitude")
        if arm.get("hidden_dropout", 0.0) not in (0.0, .043):
            raise ValueError("Unregistered hidden dropout")
        if arm.get("first_activation", "mish") not in ("mish", "square", "mish_plus_square", "relu"):
            raise ValueError("Unregistered first activation")
        if arm.get("first_activation", "mish") != "mish" and arm.get("hidden_dropout", 0.0):
            raise ValueError("Square-first control is dropout-free")
    if any(arm.get("hidden_dropout", 0.0) for arm in contract["arms"].values()):
        if contract.get("local_sources_sha256", {}).get(DROPOUT_SOURCE) != sha(ROOT / DROPOUT_SOURCE):
            raise ValueError("Dropout control source must be hash-locked before launch")
    if any(arm.get("first_activation", "mish") != "mish" for arm in contract["arms"].values()):
        if contract.get("local_sources_sha256", {}).get(ACTIVATION_SOURCE) != sha(ROOT / ACTIVATION_SOURCE):
            raise ValueError("Activation control source must be hash-locked before launch")
    if any(arm.get("warmup_starts", 1) == 3 for arm in contract["arms"].values()):
        if contract.get("local_sources_sha256", {}).get(WARMSTART_SOURCE) != sha(ROOT / WARMSTART_SOURCE):
            raise ValueError("Warmstart adapter must be hash-locked before launch")
        verification = contract.get("phase_verification", {})
        path = ROOT / verification.get("path", "missing")
        if not path.is_file() or sha(path) != verification.get("sha256"):
            raise ValueError("GPU phase verification artifact must be hash-locked")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "non_evidence_pass" or report.get("adapter_sha256") != sha(ROOT / WARMSTART_SOURCE):
            raise ValueError("GPU phase verification does not cover this adapter")
    if any(arm.get("numeric_prox", "frozen") == "isolated_v1" for arm in contract["arms"].values()):
        for source in (NUMERIC_SOURCE, NUMERIC_BRIDGE_SOURCE):
            if contract.get("local_sources_sha256", {}).get(source) != sha(ROOT / source):
                raise ValueError("Numerical kernel and binding must be hash-locked")
        verification = contract.get("numeric_training_verification", {})
        path = ROOT / verification.get("path", "missing")
        if not path.is_file() or sha(path) != verification.get("sha256"):
            raise ValueError("Numerical GPU training verification artifact must be locked")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("status") != "non_evidence_pass" or not report["original_modules_unchanged"]
                or report["bridge_sha256"] != sha(ROOT / NUMERIC_BRIDGE_SOURCE)
                or report["binding"]["candidate_sha256"] != sha(ROOT / NUMERIC_SOURCE)):
            raise ValueError("Numerical verification does not cover current binding")
    if any(a.get("square_release", "immediate") == "after_warmup" and a.get("post_warmup_amplitude", 1.0) == 1.0 for a in contract["arms"].values()):
        if contract.get("local_sources_sha256", {}).get(SQUARE_RELEASE_SOURCE) != sha(ROOT / SQUARE_RELEASE_SOURCE):
            raise ValueError("Square-release source must be hash-locked")
        verification = contract.get("square_release_verification", {})
        path = ROOT / verification.get("path", "missing")
        if not path.is_file() or sha(path) != verification.get("sha256"):
            raise ValueError("Square-release GPU verification must be hash-locked")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "non_evidence_pass" or report["release_source_sha256"] != sha(ROOT / SQUARE_RELEASE_SOURCE):
            raise ValueError("Square-release verification does not cover current source")
    if any(a.get("post_warmup_amplitude", 1.0) != 1.0 for a in contract["arms"].values()):
        for source in (POSTWARM_SOURCE, WARMSTART_SOURCE):
            if contract.get("local_sources_sha256", {}).get(source) != sha(ROOT / source):
                raise ValueError("Postwarm and phase helpers must be hash-locked")
        verification = contract.get("postwarm_verification", {})
        path = ROOT / verification.get("path", "missing")
        if not path.is_file() or sha(path) != verification.get("sha256"):
            raise ValueError("Postwarm GPU verification must be locked")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("status") != "non_evidence_pass"
                or report["postwarm_source_sha256"] != sha(ROOT / POSTWARM_SOURCE)
                or report["phase_source_sha256"] != sha(ROOT / WARMSTART_SOURCE)
                or report["numeric_bridge_sha256"] != sha(ROOT / NUMERIC_BRIDGE_SOURCE)):
            raise ValueError("Postwarm verification does not cover current sources")
    if any(a.get("marginal_initialization", "none") == "fixed_v1" for a in contract["arms"].values()):
        if contract.get("local_sources_sha256", {}).get(CUE_SOURCE) != sha(ROOT / CUE_SOURCE):
            raise ValueError("Marginal initialization source must be locked")
        verification = contract.get("cue_verification", {})
        path = ROOT / verification.get("path", "missing")
        if not path.is_file() or sha(path) != verification.get("sha256"):
            raise ValueError("Marginal initialization verification must be locked")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "non_evidence_pass" or report["cue_source_sha256"] != sha(ROOT / CUE_SOURCE):
            raise ValueError("Cue verification does not cover current source")
    if 12 * len(contract["arms"]) != contract["new_selector_trajectories"]:
        raise ValueError("Trajectory budget mismatch")
    epochs = 12 * sum(a["warmup_epochs"] * a.get("warmup_starts", 1) + 5 * a["stage_epochs"] for a in contract["arms"].values())
    if epochs != contract["new_selector_epochs_including_warmup"]:
        raise ValueError("Epoch budget mismatch")
    return contract


def aggregate(rows, target):
    out = {}
    for arm in sorted({row["arm"] for row in rows}):
        tasks = {}
        for task in ("xor", "ring"):
            subset = sorted((r for r in rows if r["arm"] == arm and r["dataset"] == task), key=lambda r: r["fold"])
            if len({r["fold"] for r in subset}) != len(subset):
                raise ValueError("Duplicate result identity")
            recovered = sum(r["evaluation"]["recovered_count"] for r in subset)
            exact = sum(r["evaluation"]["exact_recovery"] for r in subset)
            tasks[task] = {"folds": len(subset), "recovered_coordinates": recovered, "exact_folds": exact, "mean_recall_at_2": recovered / (2 * len(subset)) if subset else None, "fold_recalls": [r["evaluation"]["recall_at_2"] for r in subset]}
        tasks["target_pass"] = all(tasks[t]["folds"] == 6 and tasks[t]["recovered_coordinates"] >= target[t + "_recovered_coordinates_min"] and tasks[t]["exact_folds"] >= target[t + "_exact_folds_min"] for t in ("xor", "ring"))
        out[arm] = tasks
    return out


def endpoint_for_arm(reference_endpoint, arm):
    """One scalar multiplier for the whole existing path, never coordinatewise."""
    multiplier = arm.get("c_multiplier", 1.0)
    if multiplier not in (.0625, 1.0, 4.0):
        raise ValueError("Unregistered global C multiplier")
    return float(reference_endpoint) * multiplier


def apply_initial_gauge(model, train_input, multiplier, state_hash):
    """One global reciprocal reparameterization; no change to initial Q or g."""
    if multiplier == 1.0:
        return {"multiplier": 1.0, "applied": False}
    import torch
    if multiplier not in (.25, 4.0):
        raise ValueError("Unregistered gauge multiplier")
    before = copy.deepcopy(model).eval()
    probe = torch.as_tensor(train_input[:16], dtype=torch.float32)
    def effective_matrix(network):
        shape = [1, -1] + [1] * (network.first_raw_weight.ndim - 2)
        return network.first_raw_weight * network.gate.reshape(shape)
    with torch.no_grad():
        model.first_raw_weight.mul_(multiplier)
        model.gate.div_(multiplier)
        checks = {
            "Q_float32_exact": torch.equal(effective_matrix(before), effective_matrix(model)),
            "g_float32_exact": torch.equal(before.get_effective_gate(), model.get_effective_gate()),
            "logits_float32_exact": torch.equal(before(probe), model(probe)),
        }
        before64, after64 = before.double(), copy.deepcopy(model).double().eval()
        checks["logits_float64_exact"] = torch.equal(before64(probe.double()), after64(probe.double()))
        checks["g_float64_exact"] = torch.equal(before64.get_effective_gate(), after64.get_effective_gate())
    if not all(checks.values()):
        raise RuntimeError("Initial functional gauge invariance failed: " + str(checks))
    return {"multiplier": multiplier, "applied": True, "checks": checks, "post_gauge_state_sha256": state_hash(model.state_dict()), "coordinate": "first_raw_weight*=t; raw_s/=t; g and Q unchanged; a and s subsequently both trained"}


def apply_initial_amplitude(model, amplitude, state_hash):
    """Scale all initial effective gates equally; not a reciprocal gauge."""
    if amplitude == 1.0:
        return {"amplitude": 1.0, "applied": False}
    import torch
    if amplitude not in (.03125, .0625, .125, .25):
        raise ValueError("Unregistered initial amplitude")
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    shape = [1, -1] + [1] * (model.first_raw_weight.ndim - 2)
    with torch.no_grad():
        old_g = model.get_effective_gate().clone()
        old_q = model.first_raw_weight * model.gate.reshape(shape)
        model.gate.mul_(amplitude)
        checks = {
            "g_scaled_exact": torch.equal(model.get_effective_gate(), old_g * amplitude),
            "Q_scaled_exact": torch.equal(model.first_raw_weight * model.gate.reshape(shape), old_q * amplitude),
            "all_other_state_unchanged": all(torch.equal(tensor, model.state_dict()[name]) for name, tensor in before.items() if name != "gate"),
        }
    if not all(checks.values()):
        raise RuntimeError("Initial amplitude contract failed: " + str(checks))
    return {"amplitude": amplitude, "applied": True, "checks": checks, "post_amplitude_state_sha256": state_hash(model.state_dict()), "coordinate": "raw_s*=amplitude; a unchanged; g and Q scaled; initial function intentionally changes; no truth-dependent initialization"}


def training_config(base_config, arm):
    config = dict(base_config, device="cuda:0", required_device="cuda:0", warmup_epochs=arm["warmup_epochs"], stage_epochs=arm["stage_epochs"], admm_epochs=5 * arm["stage_epochs"], rho=arm.get("rho", base_config["rho"]))
    if "hidden_width" in arm:
        config["hidden_width"] = arm["hidden_width"]
    if "lr" in arm:
        config["lr"] = arm["lr"]
    if "hidden_dropout" in arm:
        config["hidden_dropout"] = arm["hidden_dropout"]
    if "first_activation" in arm:
        config["first_activation"] = arm["first_activation"]
    return config


def initial_dropout(model, probability, train_input, state_hash):
    if probability == 0.0:
        return {"probability": 0.0, "applied": False}
    spec = importlib.util.spec_from_file_location("minimal_dropout_control", ROOT / DROPOUT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.insert_hidden_dropout(model, probability, train_input, state_hash)


def initial_reference(arm, canonical_hash, width_anchors):
    width = arm.get("hidden_width", 32)
    if width == 32:
        return canonical_hash
    if arm["mode"] == "raw":
        return None  # New architecture; never claim byte identity to width32.
    if width not in width_anchors:
        raise ValueError("A new-width block arm requires its same-width raw anchor first")
    return width_anchors[width]


def build_initialized_arm(v3, v5, base_config, arm, X_train, y_train, seed, expected_initial_hash=None):
    config = training_config(base_config, arm)
    if arm["mode"] == "raw":
        model, _, initialization = v3.build_matched_trajectory_models(config, seed, X_train[:16])
        initial_hash = initialization["state_dict_sha256_p"]
        transform = None
        train_input = X_train
    else:
        states = {name: v5.fit_parent_block_transform(X_train, mode, square_std_tolerance=1e-8) for name, mode in v5.MODE_BY_VARIANT.items()}
        inputs = {name: v5.apply_parent_block_transform(X_train, state) for name, state in states.items()}
        models, initialization = v5.build_matched_models(config, seed, X_train[:16], {name: array[:16] for name, array in inputs.items()}, y_train[:16])
        variant = "Local-quadratic" if arm["mode"] == "local_quadratic" else "Duplicate-linear"
        model, transform, train_input = models[variant], states[variant], inputs[variant]
        initial_hash = initialization["raw_state_dict_sha256"]
    if expected_initial_hash is not None and initial_hash != expected_initial_hash:
        raise RuntimeError("Initial model differs from reused square anchor")
    initialization["initial_gauge"] = apply_initial_gauge(model, train_input, arm.get("gauge_scale", 1.0), v5._state_dict_hash)
    initialization["initial_amplitude"] = apply_initial_amplitude(model, arm.get("gate_amplitude", 1.0), v5._state_dict_hash)
    if arm.get("first_activation", "mish") != "mish":
        spec = importlib.util.spec_from_file_location("minimal_square_activation", ROOT / ACTIVATION_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model, initialization["first_activation_change"] = module.replace_first_activation(model, v3._model_kwargs(config), train_input, v5._state_dict_hash, activation=arm["first_activation"])
    initialization["initial_dropout"] = initial_dropout(model, arm.get("hidden_dropout", 0.0), train_input, v5._state_dict_hash)
    if arm.get("marginal_initialization", "none") == "fixed_v1":
        spec = importlib.util.spec_from_file_location("minimal_marginal_initialization", ROOT / CUE_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cue = module.marginal_cues(X_train, y_train)
        initialization["marginal_initialization"] = module.apply_cues(model, cue, v5._state_dict_hash)
    return model, train_input, config, initialization, initial_hash, transform


def train_arm(v3, v5, base_config, arm, X_train, y_train, seed, endpoint, expected_initial_hash=None, warmup_callback=None, prox_failure_callback=None, boundary_callback=None):
    if arm.get("numeric_prox", "frozen") == "isolated_v1":
        spec = importlib.util.spec_from_file_location("minimal_numeric_training_bridge", ROOT / NUMERIC_BRIDGE_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        v3 = module.make_numeric_training_view(v3, prox_failure_callback)
    if arm.get("post_warmup_amplitude", 1.0) != 1.0:
        spec = importlib.util.spec_from_file_location("minimal_postwarm_amplitude", ROOT / POSTWARM_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.train_postwarm(v3, v5, build_initialized_arm, apply_initial_amplitude,
                                     base_config, arm, X_train, y_train, seed, endpoint,
                                     expected_initial_hash, boundary_callback)
    if arm.get("warmup_starts", 1) == 3:
        spec = importlib.util.spec_from_file_location("minimal_warmstart", ROOT / WARMSTART_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.train_multistart(v3, v5, build_initialized_arm, base_config, arm, X_train, y_train, seed, endpoint, expected_initial_hash, warmup_callback)
    model, train_input, config, initialization, initial_hash, transform = build_initialized_arm(v3, v5, base_config, arm, X_train, y_train, seed, expected_initial_hash)
    release = None
    if arm.get("square_release", "immediate") == "after_warmup":
        spec = importlib.util.spec_from_file_location("minimal_square_release", ROOT / SQUARE_RELEASE_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        batches = math.ceil(len(y_train) / config["batch_size"])
        release = module.SquareWarmupRelease(model, batches*config["warmup_epochs"],
                                           batches*(config["warmup_epochs"]+config["admm_epochs"]))
    try:
        runtime, trace = v3.train_trajectory(model, train_input, y_train, config, endpoint, "P", seed)
    finally:
        if release is not None:
            release.remove()
    if release is not None:
        initialization["square_release"] = release.finish(v5._state_dict_hash)
        model.minimal_square_release_warm_state = release.warm_state
    if arm.get("numeric_prox", "frozen") == "isolated_v1":
        initialization["numeric_solver"] = dict(v3.minimal_numeric_binding)
    return model, trace, runtime, config, initialization, initial_hash, transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=ROOT / "experiments/ablation/current_minimal_r1_20260907.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    contract = validate_contract(json.loads(args.contract.read_text(encoding="utf-8")))
    audit, v3, v5, configure_gpu_runtime = load_pinned_code()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    runtime = configure_gpu_runtime()
    if torch.__version__ != "2.5.1+cu121" or runtime["cuda_device_name"] != "NVIDIA GeForce RTX 3060 Ti":
        raise RuntimeError("GPU runtime differs from reusable results")
    base_config = dict(v5.FORMAL_CONFIG)

    def cpu_copy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu_copy(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu_copy(v) for v in value)
        return copy.deepcopy(value)

    if args.smoke:
        rng = np.random.default_rng(731)
        X = rng.normal(size=(40, 8)).astype(np.float32)
        y = rng.integers(0, 2, size=40, dtype=np.int64)
        config = dict(base_config, n_features=8)
        records = []
        smoke_cases = [(mode, multiplier, 50.0, 32) for mode in ("raw", "local_quadratic") for multiplier in (1.0, .25, 4.0)]
        smoke_cases += [(mode, 1.0, rho, 32) for mode in ("raw", "local_quadratic") for rho in (.5, 5.0)]
        smoke_cases += [(mode, 1.0, 50.0, width) for mode in ("raw", "local_quadratic") for width in (4, 8)]
        smoke_cases = [(*case, 1.0) for case in smoke_cases]
        smoke_cases += [(mode, 1.0, 50.0, 32, amplitude) for mode in ("raw", "local_quadratic") for amplitude in (.0625, .25)]
        smoke_cases = [(*case, 0.0) for case in smoke_cases]
        smoke_cases += [(mode, 1.0, 50.0, 32, amplitude, .043) for mode in ("raw", "local_quadratic") for amplitude in (1.0, .0625)]
        smoke_cases += [(mode, .25, 50.0, 32, .0625, 0.0) for mode in ("raw", "local_quadratic")]
        smoke_cases = [(*case, "mish") for case in smoke_cases]
        smoke_cases += [("raw", 1.0, 50.0, 32, amplitude, 0.0, "square") for amplitude in (1.0, .0625)]
        smoke_cases += [("local_quadratic", 1.0, 50.0, 32, amplitude, 0.0, "square") for amplitude in (1.0, .0625)]
        smoke_cases += [("local_quadratic", 1.0, 50.0, 32, .0625, 0.0, "mish_plus_square")]
        smoke_cases += [("local_quadratic", 1.0, 50.0, 32, amplitude, 0.0, "mish") for amplitude in (.03125, .125)]
        smoke_cases = [(*case, .005) for case in smoke_cases]
        smoke_cases += [("local_quadratic", 1.0, 50.0, 32, .0625, 0.0, "mish", lr) for lr in (.001, .01)]
        smoke_cases += [(mode, 4.0, 50.0, 32, .0625, 0.0, "mish", .005) for mode in ("raw", "local_quadratic")]
        for mode, multiplier, rho, width, amplitude, probability, activation, lr in smoke_cases:
            arm = {"mode": mode, "stage_epochs": 1, "warmup_epochs": 1, "gauge_scale": multiplier, "rho": rho, "hidden_width": width, "gate_amplitude": amplitude, "hidden_dropout": probability, "first_activation": activation, "lr": lr}
            hashes = []
            for _ in range(2):
                model, trace, _, _, _, _, _ = train_arm(v3, v5, config, arm, X, y, 811, .2)
                if len(trace) != 5 or str(next(model.parameters()).device) != "cuda:0":
                    raise RuntimeError("Smoke trace/device mismatch")
                if model.admm_state["rho"] != rho or model.admm_state["rho_init"] != rho:
                    raise RuntimeError("Smoke fixed rho mismatch")
                probabilities = [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)]
                if probabilities != ([probability] if probability else []):
                    raise RuntimeError("Smoke dropout graph mismatch")
                if getattr(model, "minimal_first_activation", "mish") != activation:
                    raise RuntimeError("Smoke first-activation mismatch")
                if any(group["lr"] != lr for group in model.admm_state["optimizer_state_dict"]["param_groups"]):
                    raise RuntimeError("Smoke Adam learning rate mismatch")
                hashes.append(v5._state_dict_hash(model.state_dict()))
            if hashes[0] != hashes[1]:
                raise RuntimeError("Smoke same-seed replay failed")
            records.append({"mode": mode, "gauge_scale": multiplier, "rho": rho, "hidden_width": width, "gate_amplitude": amplitude, "hidden_dropout": probability, "first_activation": activation, "lr": lr, "identical_replay": True, "state_sha256": hashes[0]})
        out = Path(tempfile.mkdtemp(prefix="current_minimal_smoke_")) / "smoke.json"
        for multiplier in (.0625, 4.0):
            arm = dict(mode="local_quadratic", warmup_epochs=1, stage_epochs=1,
                       gate_amplitude=.0625, c_multiplier=multiplier)
            hashes = []
            for _ in range(2):
                endpoint = endpoint_for_arm(.2, arm)
                model, trace, _, _, _, _, _ = train_arm(v3, v5, config, arm, X, y, 811, endpoint)
                if model.admm_state["C_used"] != endpoint or trace[-1]["C"] != endpoint:
                    raise RuntimeError("C multiplier did not reach actual trainer")
                hashes.append(v5._state_dict_hash(model.state_dict()))
            if hashes[0] != hashes[1]:
                raise RuntimeError("C multiplier replay failed")
            records.append({"mode": "local_quadratic", "gate_amplitude": .0625,
                            "c_multiplier": multiplier, "actual_endpoint": endpoint,
                            "identical_replay": True, "state_sha256": hashes[0]})
        for activation in ("mish", "relu"):
            for mode in ("raw", "local_quadratic"):
                for amplitude in (1.0, .0625):
                    arm = dict(mode=mode, gate_amplitude=amplitude, warmup_epochs=1, stage_epochs=1,
                               first_activation=activation, numeric_prox="isolated_v1")
                    hashes = []
                    for _ in range(2):
                        model, trace, _, _, initialization, _, _ = train_arm(v3, v5, config, arm, X, y, 811, .2)
                        if (initialization["numeric_solver"]["implementation"] != "isolated_numeric_candidate_v1"
                                or any(r.get("implementation") != "isolated_numeric_candidate_v1" for r in model.admm_state["canonical_prox_log"])):
                            raise RuntimeError("Numerical candidate not called")
                        hashes.append(v5._state_dict_hash(model.state_dict()))
                    if hashes[0] != hashes[1]:
                        raise RuntimeError("Numerical/activation replay failed")
                    if activation == "mish":
                        reference = next(r for r in records if r.get("mode") == mode and r.get("gate_amplitude") == amplitude
                                         and r.get("first_activation") == "mish" and r.get("gauge_scale") == 1.0
                                         and r.get("rho") == 50.0 and r.get("hidden_width") == 32
                                         and r.get("hidden_dropout") == 0.0 and r.get("lr") == .005)
                        if hashes[0] != reference["state_sha256"]:
                            raise RuntimeError("Numerical interface changed matched frozen smoke state")
                    records.append({"mode": mode, "gate_amplitude": amplitude, "first_activation": activation,
                                    "numeric_prox": "isolated_v1", "identical_replay": True, "state_sha256": hashes[0]})
        out.write_text(json.dumps({"status": "non_evidence_pass", "records": records, "runtime": runtime, "script_sha256": sha(__file__), "dropout_source_sha256": sha(ROOT / DROPOUT_SOURCE), "activation_source_sha256": sha(ROOT / ACTIVATION_SOURCE)}, indent=2), encoding="utf-8")
        print(json.dumps({"smoke": str(out), "records": records}), flush=True)
        return 0

    output = (ROOT / contract["output_relative"]).resolve()
    if not output.is_relative_to(ROOT / "results/ablation"):
        raise ValueError("Output must stay in this research result family")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    source_dir = output / "source"
    source_dir.mkdir()
    for path in [Path(__file__), args.contract, *(ROOT / p for p in contract.get("local_sources_sha256", {}))]:
        shutil.copy2(path, source_dir / path.name)
    writer = v3.JsonlWriter(output / "run.jsonl")
    metadata = {"protocol_id": contract["protocol_id"], "contract": contract, "contract_sha256": sha(args.contract), "runner_sha256": sha(__file__), "reuse_audit_sha256": sha(AUDIT), "imported_sources": audit["source_checks"], "runtime": runtime, "created_at": datetime.now().isoformat(), "command": sys.argv, "working_directory": str(Path.cwd()), "evidence_status": "diagnostic", "source_policy": "dirty_workspace_preserved_exact_relevant_sources_hash_locked", "xlsx_row": None, "xlsx_reason": "diagnostic_not_promoted_to_main_table"}
    metadata["local_sources_sha256"] = contract.get("local_sources_sha256", {})
    (output / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    writer.write({"event": "run_start", **metadata})
    all_rows = []
    current_identity = None
    try:
        v3_path = ROOT / "results/ablation/effective_gate_c_path_v3_20260824_162300_648919.jsonl"
        reference = v5.load_v3_reference(v3_path)
        v5_path = INTEGRATION / "results/ablation/parent_block_quadratic_v5_gpu_frozen_v1_20260904_014430_491836.jsonl"
        for line in v5_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["event"] != "final_variant":
                continue
            for arm_id, variant in contract["reuse"].items():
                if row["variant"] == variant:
                    reused = {"event": "reused_result", "arm": arm_id, "dataset": row["dataset"], "fold": row["fold"], "evaluation": row["evaluation"], "source_jsonl": str(v5_path), "source_jsonl_sha256": sha(v5_path), "checkpoint": row["checkpoint"], "checkpoint_sha256": row["checkpoint_sha256"], "execution_device": row["execution_device"], "selected_endpoint": row["selected_endpoint"]}
                    writer.write(reused)
                    all_rows.append(reused)
        for task in contract["tasks"]:
            offset = v3.DATASET_OFFSETS[task]
            dataset_seed = base_config["seed"] + offset
            v5._set_determinism(dataset_seed)
            X, _, y = v3.generate_dataset(task, 1000, 128)
            X, y = X.astype(np.float32), y.astype(np.int64)
            dataset_hash = v3._array_hash(X, y)
            outer = v3.StratifiedKFold(n_splits=6, shuffle=True, random_state=dataset_seed + 101)
            for fold, (train_index, test_index) in enumerate(outer.split(X, y)):
                ref = reference["p_final"][(task, fold)]
                ref_outer = reference["outer"][(task, fold)]
                split_hash = v3._array_hash(train_index, test_index)
                (scaled, scaler) = v3._fit_scaler(X[train_index], X[test_index])
                if dataset_hash != ref_outer["dataset_sha256"] or split_hash != ref["split_sha256"] or scaler["statistics_sha256"] != ref["outer_scaler_sha256"]:
                    raise RuntimeError("Data/split/scaler identity mismatch")
                X_train, X_test = scaled
                seed, reference_endpoint = ref["initialization"]["seed"], ref["selected_endpoint"]
                width_anchors = {}
                for arm_id, arm in contract["arms"].items():
                    endpoint = endpoint_for_arm(reference_endpoint, arm)
                    current_identity = {"arm": arm_id, "dataset": task, "fold": fold}
                    writer.write({"event": "trajectory_start", **current_identity, "seed": seed, "endpoint": endpoint, "arm_config": arm})
                    print(json.dumps({"event": "start", **current_identity}), flush=True)
                    reference_hash = initial_reference(arm, ref["initialization"]["state_dict_sha256_p"], width_anchors)
                    def save_warmup(trial):
                        warm_path = checkpoint_dir / f"{task}_fold{fold}_{arm_id}_warm{trial['start_index']}.pt"
                        with warm_path.open("xb") as handle:
                            torch.save(trial, handle)
                        writer.write({"event": "warmup_result", **current_identity,
                                      "start_index": trial["start_index"], "seed": trial["seed"],
                                      "train_eval_bce": trial["train_eval_bce"],
                                      "checkpoint": str(warm_path), "checkpoint_sha256": sha(warm_path)})
                    def save_prox_failure(failure):
                        failure_path = checkpoint_dir / f"{task}_fold{fold}_{arm_id}_failed_prox.pt"
                        with failure_path.open("xb") as handle:
                            torch.save({**failure, **current_identity}, handle)
                        writer.write({"event": "prox_failure_input", **current_identity,
                                      "checkpoint": str(failure_path), "checkpoint_sha256": sha(failure_path),
                                      "lambda": failure["lambda"], "error": failure["error"]})
                    def save_boundary(boundary):
                        boundary_path = checkpoint_dir / f"{task}_fold{fold}_{arm_id}_postwarm.pt"
                        with boundary_path.open("xb") as handle:
                            torch.save(boundary, handle)
                        writer.write({"event": "postwarm_boundary", **current_identity,
                                      "checkpoint": str(boundary_path), "checkpoint_sha256": sha(boundary_path),
                                      "multiplier": boundary["multiplier"]})
                    model, trace, seconds, config, initialization, initial_hash, transform = train_arm(v3, v5, base_config, arm, X_train, y[train_index], seed, endpoint, reference_hash, save_warmup, save_prox_failure, save_boundary)
                    width_anchors.setdefault(config["hidden_width"], initial_hash)
                    if width_anchors[config["hidden_width"]] != initial_hash:
                        raise RuntimeError("Same-width raw initialization mismatch")
                    initialization["architecture_identity"] = {"hidden_width": config["hidden_width"], "hidden_layers": config["hidden_layers"], "raw_anchor_sha256": initial_hash, "matches_frozen_width32_architecture": config["hidden_width"] == 32, "parameter_count": sum(p.numel() for p in model.parameters())}
                    scores = model.get_effective_gate(training_drop=False).detach().abs()
                    selected, selection = v3.select_scores(scores, config, "final_effective_gate")
                    if selection["status"] != "ok":
                        raise RuntimeError("Invalid final selection")
                    recovered = len(set(selected) & {0, 1})
                    evaluation = {"selected": selected, "scores": scores.cpu().tolist(), "selection": selection, "recovered_count": recovered, "recall_at_2": recovered / 2, "exact_recovery": int(recovered == 2)}
                    checkpoint = checkpoint_dir / f"{task}_fold{fold}_{arm_id}.pt"
                    payload = {"protocol_id": contract["protocol_id"], **current_identity, "config": config, "arm_config": arm, "endpoint": endpoint, "initialization": initialization, "model_state_dict": cpu_copy(model.state_dict()), "admm_state": cpu_copy(model.admm_state), "epoch_trace": trace, "parent_block_transform": transform, "raw_scaler": scaler, "train_indices": train_index, "test_indices": test_index, "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}}
                    payload["hidden_dropout_modules"] = [{"name": name, "p": module.p} for name, module in model.named_modules() if isinstance(module, torch.nn.Dropout)]
                    payload["first_activation"] = getattr(model, "minimal_first_activation", "mish")
                    if hasattr(model, "minimal_square_release_warm_state"):
                        payload["square_release_warm_state"] = cpu_copy(model.minimal_square_release_warm_state)
                    if hasattr(model, "minimal_postwarm_boundary"):
                        payload["postwarm_boundary"] = cpu_copy(model.minimal_postwarm_boundary)
                    if hasattr(model, "minimal_cue_initialization_state"):
                        payload["cue_initialization_state"] = cpu_copy(model.minimal_cue_initialization_state)
                    if arm.get("warmup_starts", 1) == 3:
                        payload["warmup_trials"] = cpu_copy(model.minimal_warmstart_trials)
                    torch.save(payload, checkpoint)
                    test_input = X_test if transform is None else v5.apply_parent_block_transform(X_test, transform)
                    row = {"event": "final_result", **current_identity, "status": "success", "execution_device": str(next(model.parameters()).device), "runtime_seconds": seconds, "seed": seed, "selected_endpoint": endpoint, "schedule": v3.build_c_schedule(endpoint, "P", config), "dataset_sha256": dataset_hash, "split_sha256": split_hash, "outer_scaler_sha256": scaler["statistics_sha256"], "initialization": initialization, "square_scaler_sha256": v5._square_scaler_sha256(transform) if transform else None, "evaluation": evaluation, "full_model_test": v5._supervised_metrics(model, test_input, y[test_index]), "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint), "state_dict_sha256": v5._state_dict_hash(model.state_dict()), "epoch_trace": trace}
                    row["hidden_dropout_modules"] = payload["hidden_dropout_modules"]
                    row["first_activation"] = payload["first_activation"]
                    row["reference_endpoint"] = reference_endpoint
                    row["c_multiplier"] = arm.get("c_multiplier", 1.0)
                    writer.write(row)
                    all_rows.append(row)
                    print(json.dumps({"event": "done", **current_identity, "selected": selected, "recall": recovered / 2, "seconds": round(seconds, 2)}), flush=True)
                    del payload, model
        summary = {"protocol_id": contract["protocol_id"], "status": "completed", "new_trajectories": sum(r["event"] == "final_result" for r in all_rows), "reused_trajectories": sum(r["event"] == "reused_result" for r in all_rows), "aggregate": aggregate(all_rows, contract["primary_target"]), "scope": "exploration_on_previously_inspected_six_folds_not_independent_confirmation"}
        if summary["new_trajectories"] != contract["new_selector_trajectories"]:
            raise RuntimeError("Incomplete run")
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        writer.write({"event": "run_complete", "status": "success", "summary": str(output / "summary.json")})
        print(json.dumps(summary), flush=True)
    except Exception as error:
        writer.write({"event": "failure", "identity": current_identity, "error": repr(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
