"""Explicit non-default numerical kernel binding; never mutates frozen modules."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "experiments/ablation/minimal_prox_numeric_candidate.py"
EXPECTED_CANDIDATE_SHA = "6ed77fc604592abeabf3ab3087ff937a04911c8f87a023279bd63c1c3615b0a6"


def clone_with_globals(function, replacements):
    """Keep bytecode/defaults/closure; copy the global namespace before binding."""
    namespace = dict(function.__globals__)
    if not set(replacements).issubset(namespace):
        raise ValueError("Replacement must bind an existing global")
    namespace.update(replacements)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = dict(function.__annotations__)
    changed = {key for key, value in namespace.items() if value is not function.__globals__[key]}
    if changed != set(replacements):
        raise RuntimeError("Unexpected copied-global changes")
    return result


def make_numeric_training_view(v3, failure_recorder=None):
    actual_sha = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    if actual_sha != EXPECTED_CANDIDATE_SHA:
        raise RuntimeError("Numerical candidate differs from audited version")
    spec = importlib.util.spec_from_file_location("minimal_bound_prox_numeric", CANDIDATE)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    original_trainer = v3._train_input_group
    original_trajectory = v3.train_trajectory
    original_prox = original_trainer.__globals__["canonical_signed_ratio_prox"]

    def recorded_prox(center, lam, **kwargs):
        try:
            value, info = candidate.canonical_signed_ratio_prox(center, lam, **kwargs)
        except (RuntimeError, ValueError) as error:
            if failure_recorder is not None:
                failure_recorder({"center": center.detach().cpu().clone(),
                                  "lambda": float(lam), "kwargs": dict(kwargs),
                                  "error": repr(error), "candidate_sha256": actual_sha,
                                  "scope": "exact failed prox input only,not a training restart"})
            raise
        return value, dict(info, implementation="isolated_numeric_candidate_v1",
                           numeric_candidate_source_sha256=actual_sha)

    trainer = clone_with_globals(original_trainer, {"canonical_signed_ratio_prox": recorded_prox})
    trajectory = clone_with_globals(original_trajectory, {"_train_input_group": trainer})
    view = SimpleNamespace(**vars(v3))
    view._train_input_group = trainer
    view.train_trajectory = trajectory
    view.minimal_numeric_binding = {"implementation": "isolated_numeric_candidate_v1",
                                    "candidate_sha256": actual_sha,
                                    "frozen_trainer_bytecode_unchanged": trainer.__code__ is original_trainer.__code__,
                                    "frozen_trajectory_bytecode_unchanged": trajectory.__code__ is original_trajectory.__code__,
                                    "global_changes": ["trainer.canonical_signed_ratio_prox", "trajectory._train_input_group"]}
    if (v3._train_input_group is not original_trainer or v3.train_trajectory is not original_trajectory
            or original_trainer.__globals__["canonical_signed_ratio_prox"] is not original_prox
            or original_trajectory.__globals__["_train_input_group"] is not original_trainer):
        raise RuntimeError("Frozen module mutated while constructing numerical view")
    return view
