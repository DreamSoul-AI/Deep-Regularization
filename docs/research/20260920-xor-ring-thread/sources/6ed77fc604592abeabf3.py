"""Provisional, isolated numerical correction; not wired into any trainer.

Preserves the frozen objective, candidate formulas and default tolerances.
Only deltas: keep strictly positive tau, certify candidates before selection,
and reserve sparsity tie-breaking for roundoff-scale computed-objective ties.
Other numerical edge cases are not claimed solved; frozen source is untouched.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FROZEN_SOURCE = ROOT.with_name(ROOT.name + "_integration") / "src/admm_utils.py"
EXPECTED_SHA = "c42881da970fe46d5a9ed326faebd4c4a50a9f0f1d88f13edb133368e8eb52ee"
if hashlib.sha256(FROZEN_SOURCE.read_bytes()).hexdigest() != EXPECTED_SHA:
    raise RuntimeError("Frozen prox source changed")
_spec = importlib.util.spec_from_file_location("frozen_prox_numeric_reference", FROZEN_SOURCE)
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)
_ratio_prox_objective_numpy = _reference._ratio_prox_objective_numpy
_ratio_prox_kkt_residual_numpy = _reference._ratio_prox_kkt_residual_numpy


def canonical_signed_ratio_prox(
    center: torch.Tensor,
    lam: float | torch.Tensor,
    *,
    root_tol: float = 1e-7,
    certification_tol: float = 5e-6,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Diagnostic numerical variant of the frozen signed RatioNorm prox.

    The solved objective is

        0.5 * ||z - center||_2^2 + lam * Rbar(z),

    where Rbar(z)=||z||_1/||z||_2 for nonzero z and Rbar(0)=1.  The
    implementation enumerates every ordered support prefix, all real roots of
    its regular-branch quartic, the one-sparse boundary candidates, and the
    singular directions created by a tied maximum center.  Every reconstructed
    candidate is checked against the unsquared equations and the original
    objective.  Numerical certification failure raises instead of falling
    back to a historical surrogate.
    """
    if center.ndim != 1:
        raise ValueError("center must be a one-dimensional tensor")
    if not center.is_floating_point():
        raise TypeError("center must have a floating dtype")
    if not torch.isfinite(center).all():
        raise ValueError("center must be finite")
    if root_tol <= 0.0 or certification_tol <= 0.0:
        raise ValueError("solver tolerances must be positive")

    if torch.is_tensor(lam):
        if lam.numel() != 1:
            raise ValueError("lam must be scalar")
        lam_value = float(lam.detach().cpu().item())
    else:
        lam_value = float(lam)
    if not np.isfinite(lam_value) or lam_value < 0.0:
        raise ValueError("lam must be finite and nonnegative")

    center_np = center.detach().to(device="cpu", dtype=torch.float64).numpy()
    dimension = int(center_np.size)
    if dimension == 0:
        raise ValueError("center must be non-empty")

    if lam_value == 0.0:
        result = center.detach().clone()
        return result, {
            "branch": "identity",
            "certified": True,
            "objective": 0.0,
            "support": int(torch.count_nonzero(result).item()),
            "candidate_count": 1,
            "objective_tie_count": 1,
            "kkt_residual": 0.0,
        }

    center_norm = float(np.linalg.norm(center_np, ord=2))
    if center_norm == 0.0:
        result = torch.zeros_like(center)
        return result, {
            "branch": "zero_center",
            "certified": True,
            "objective": lam_value,
            "support": 0,
            "candidate_count": 1,
            "objective_tie_count": 1,
            "kkt_residual": 0.0,
        }

    alpha = np.abs(center_np)
    original_index = np.arange(dimension, dtype=np.int64)
    order = np.lexsort((original_index, -alpha))
    sorted_alpha = alpha[order]
    max_alpha = float(sorted_alpha[0])
    sign = np.sign(center_np)
    sign[sign == 0.0] = 1.0

    candidates: list[dict[str, Any]] = []

    def _add_candidate(sorted_magnitude: np.ndarray, branch: str) -> None:
        if sorted_magnitude.shape != sorted_alpha.shape:
            raise RuntimeError("internal prox candidate has the wrong shape")
        magnitude = np.zeros_like(sorted_magnitude)
        magnitude[order] = sorted_magnitude
        value = sign * magnitude
        support_indices = tuple(np.flatnonzero(magnitude > root_tol).tolist())
        candidates.append(
            {
                "value": value,
                "branch": branch,
                "support_indices": support_indices,
                "support": len(support_indices),
                "objective": _ratio_prox_objective_numpy(
                    value, center_np, lam_value
                ),
            }
        )

    max_tie_count = int(
        np.sum(np.isclose(sorted_alpha, max_alpha, rtol=0.0, atol=root_tol))
    )
    for tie_position in range(max_tie_count):
        magnitude = np.zeros(dimension, dtype=np.float64)
        magnitude[tie_position] = max_alpha
        _add_candidate(magnitude, "one_sparse")

    prefix_sum = 0.0
    prefix_square_sum = 0.0
    regular_root_count = 0
    rejected_root_count = 0
    for support in range(1, dimension + 1):
        current_alpha = float(sorted_alpha[support - 1])
        prefix_sum += current_alpha
        prefix_square_sum += current_alpha * current_alpha
        if current_alpha <= root_tol:
            break

        polynomial = np.asarray(
            [
                prefix_sum * prefix_sum,
                -2.0 * prefix_sum * prefix_square_sum,
                prefix_square_sum * prefix_square_sum
                - lam_value * lam_value * support,
                2.0 * lam_value * lam_value * prefix_sum,
                -lam_value * lam_value * prefix_square_sum,
            ],
            dtype=np.float64,
        )
        roots = np.roots(polynomial)
        accepted_tau: list[float] = []
        for root in roots:
            scale = max(1.0, abs(float(root.real)))
            if abs(float(root.imag)) > root_tol * scale:
                continue
            tau = float(root.real)
            if tau <= 0.0:  # A positive mathematical root is not zero merely because it is small.
                continue
            if any(abs(tau - previous) <= root_tol * scale for previous in accepted_tau):
                continue

            active_gap = sorted_alpha[:support] - tau
            next_alpha = (
                float(sorted_alpha[support]) if support < dimension else 0.0
            )
            if np.any(active_gap <= root_tol) or next_alpha > tau + root_tol:
                rejected_root_count += 1
                continue

            square_norm = float(np.dot(active_gap, active_gap))
            projection = prefix_square_sum - tau * prefix_sum
            if square_norm <= root_tol or projection <= root_tol:
                rejected_root_count += 1
                continue
            unsquared_residual = abs(
                tau * projection - lam_value * np.sqrt(square_norm)
            )
            unsquared_scale = max(
                1.0,
                abs(tau * projection),
                abs(lam_value * np.sqrt(square_norm)),
            )
            if unsquared_residual > certification_tol * unsquared_scale:
                rejected_root_count += 1
                continue

            direction = active_gap / np.sqrt(square_norm)
            radius = projection / np.sqrt(square_norm)
            sorted_magnitude = np.zeros(dimension, dtype=np.float64)
            sorted_magnitude[:support] = radius * direction
            norm2 = float(np.linalg.norm(sorted_magnitude, ord=2))
            norm1 = float(np.linalg.norm(sorted_magnitude, ord=1))
            denominator = 1.0 - lam_value * norm1 / (norm2**3)
            if denominator <= root_tol:
                rejected_root_count += 1
                continue

            accepted_tau.append(tau)
            regular_root_count += 1
            _add_candidate(sorted_magnitude, f"regular_prefix_{support}")

    singular_candidate_count = 0
    singular_l1 = lam_value / (max_alpha * max_alpha)
    if singular_l1 >= 1.0 - root_tol:
        for support in range(1, max_tie_count + 1):
            if singular_l1 > np.sqrt(float(support)) + root_tol:
                continue
            if support == 1:
                if abs(singular_l1 - 1.0) > root_tol:
                    continue
                direction = np.ones(1, dtype=np.float64)
            else:
                discriminant = (support - 1.0) * (
                    support - singular_l1 * singular_l1
                )
                if discriminant < -root_tol:
                    continue
                first = (
                    singular_l1 + np.sqrt(max(discriminant, 0.0))
                ) / support
                remaining = (singular_l1 - first) / (support - 1.0)
                direction = np.full(support, remaining, dtype=np.float64)
                direction[0] = first
                if np.any(direction <= root_tol):
                    continue
            sorted_magnitude = np.zeros(dimension, dtype=np.float64)
            radius = max_alpha * singular_l1
            sorted_magnitude[:support] = radius * direction
            singular_candidate_count += 1
            _add_candidate(sorted_magnitude, f"singular_max_tie_{support}")

    if not candidates:
        raise RuntimeError("canonical RatioNorm prox produced no candidates")

    # KKT is a feasibility/certification gate, not a final check on a sparsity tie.
    unfiltered_candidate_count = len(candidates)
    certified_candidates = []
    for candidate in candidates:
        residual = _ratio_prox_kkt_residual_numpy(
            candidate["value"], center_np, lam_value, root_tol
        )
        scale = max(1.0, center_norm, float(np.linalg.norm(candidate["value"])))
        if residual <= certification_tol * scale:
            certified_candidates.append(candidate)
    if not certified_candidates:
        raise RuntimeError("isolated numeric prox has no KKT-certified candidate")
    candidates = certified_candidates
    minimum_objective = min(candidate["objective"] for candidate in candidates)
    # Equality bookkeeping must not inherit the much wider KKT stopping tolerance.
    # This is a roundoff-sized band, not an interval-arithmetic global certificate.
    objective_tol = (32.0 * np.finfo(np.float64).eps * (dimension + 1)
                     * max(1.0, abs(minimum_objective)))
    tied = [
        candidate
        for candidate in candidates
        if candidate["objective"] <= minimum_objective + objective_tol
    ]
    tied.sort(
        key=lambda candidate: (
            candidate["support"],
            candidate["support_indices"],
            candidate["branch"],
        )
    )
    winner = tied[0]
    kkt_residual = _ratio_prox_kkt_residual_numpy(
        winner["value"], center_np, lam_value, root_tol
    )
    kkt_scale = max(1.0, center_norm, float(np.linalg.norm(winner["value"])))
    certified = bool(kkt_residual <= certification_tol * kkt_scale)
    if not certified:
        raise RuntimeError(
            "canonical RatioNorm prox failed KKT certification: "
            f"residual={kkt_residual:.6e}, scale={kkt_scale:.6e}"
        )

    result = torch.as_tensor(
        winner["value"],
        device=center.device,
        dtype=center.dtype,
    )
    diagnostics: dict[str, Any] = {
        "branch": winner["branch"],
        "certified": True,
        "objective": float(winner["objective"]),
        "support": int(winner["support"]),
        "candidate_count": len(candidates),
        "regular_root_count": int(regular_root_count),
        "rejected_root_count": int(rejected_root_count),
        "singular_candidate_count": int(singular_candidate_count),
        "objective_tie_count": len(tied),
        "coordinate_max_tie_count": int(max_tie_count),
        "unfiltered_candidate_count": unfiltered_candidate_count,
        "implementation": "isolated_numeric_candidate_v1",
        "objective_roundoff_tie_band": float(objective_tol),
        "kkt_residual": float(kkt_residual),
        "tie_rule": "objective_then_support_then_original_index",
    }
    return result, diagnostics
