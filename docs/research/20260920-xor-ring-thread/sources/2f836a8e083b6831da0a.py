# -*- coding: utf-8 -*-
"""
ADMM utility functions for feature selection.

This module contains mathematical utilities used by the ADMM
feature-selection wrappers:
  - deterministic_topk: nonzero-only selection with auditable tie semantics
  - canonical_signed_ratio_prox: certified global signed RatioNorm prox
  - soft_thresholding: proximal operator for L1 norm
  - safe_norm: numerical stable L2 norm
  - safe_cbrt: numerical stable cube root
  - solve_cubic_ratio_norm: Cardano root for legacy sequential RatioNorm scaling
  - solve_cubic_denominator_copy: Newton root for z_den radial update
  - strict_ratio_z_update: sparse-copy update for strict two-copy RatioNorm
  - strict_ratio_y_update: denominator-copy update with its own dual center
  - solve_cubic_paper: wrapper for paper's cubic equation
"""

from typing import Any

import numpy as np
import torch


def deterministic_topk(
    scores: torch.Tensor,
    k: int,
    *,
    zero_tol: float = 0.0,
    readout: str = "effective_gate",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select up to ``k`` strictly nonzero scores with a stable tie rule.

    Scores are ordered by descending magnitude and then by ascending original
    coordinate.  Zero-score coordinates are never promoted merely to fill the
    requested budget.  The returned metadata makes all-zero, insufficient-
    support, and cutoff-tie cases explicit.
    """
    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional tensor")
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer")
    if k < 0 or k > scores.numel():
        raise ValueError("k must satisfy 0 <= k <= number of scores")
    if zero_tol < 0.0:
        raise ValueError("zero_tol must be nonnegative")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")

    magnitudes = scores.detach().abs().to(device="cpu", dtype=torch.float64)
    order = sorted(
        range(magnitudes.numel()),
        key=lambda index: (-float(magnitudes[index].item()), index),
    )
    positive = [
        index for index in order if float(magnitudes[index].item()) > zero_tol
    ]
    selected_list = positive[:k]
    selected = torch.tensor(
        selected_list,
        device=scores.device,
        dtype=torch.long,
    )

    if k == 0:
        status = "ok"
    elif not positive:
        status = "degenerate_all_zero"
    elif len(positive) < k:
        status = "insufficient_nonzero"
    else:
        status = "ok"

    cutoff = None
    cutoff_tie_count = 0
    cutoff_tie_crosses_budget = False
    if selected_list:
        cutoff = float(magnitudes[selected_list[-1]].item())
        cutoff_tie_count = sum(
            bool(float(value.item()) == cutoff) for value in magnitudes
        )
        if len(selected_list) == k and len(positive) > k:
            next_value = float(magnitudes[positive[k]].item())
            cutoff_tie_crosses_budget = next_value == cutoff

    metadata: dict[str, Any] = {
        "readout": readout,
        "status": status,
        "requested_k": int(k),
        "selected_count": len(selected_list),
        "nonzero_count": len(positive),
        "zero_tol": float(zero_tol),
        "cutoff": cutoff,
        "cutoff_tie_count": int(cutoff_tie_count),
        "cutoff_tie_crosses_budget": bool(cutoff_tie_crosses_budget),
        "tie_rule": "score_desc_then_original_index_asc",
    }
    return selected, metadata


def _ratio_norm_value_numpy(value: np.ndarray) -> float:
    norm2 = float(np.linalg.norm(value, ord=2))
    if norm2 == 0.0:
        return 1.0
    return float(np.linalg.norm(value, ord=1) / norm2)


def _ratio_prox_objective_numpy(
    value: np.ndarray,
    center: np.ndarray,
    lam: float,
) -> float:
    delta = value - center
    return 0.5 * float(np.dot(delta, delta)) + lam * _ratio_norm_value_numpy(
        value
    )


def _ratio_prox_kkt_residual_numpy(
    value: np.ndarray,
    center: np.ndarray,
    lam: float,
    support_tol: float,
) -> float:
    """Return an absolute first-order residual for a nonzero prox candidate."""
    magnitude = np.abs(value)
    center_magnitude = np.abs(center)
    active = magnitude > support_tol
    if not np.any(active):
        return 0.0 if np.linalg.norm(center, ord=2) == 0.0 else float("inf")

    norm2 = float(np.linalg.norm(magnitude, ord=2))
    norm1 = float(np.linalg.norm(magnitude, ord=1))
    gradient = magnitude[active] - center_magnitude[active]
    gradient += lam * (
        1.0 / norm2 - norm1 * magnitude[active] / (norm2**3)
    )
    active_residual = float(np.max(np.abs(gradient), initial=0.0))
    threshold = lam / norm2
    inactive_residual = float(
        np.max(
            np.maximum(center_magnitude[~active] - threshold, 0.0),
            initial=0.0,
        )
    )
    return max(active_residual, inactive_residual)


def canonical_signed_ratio_prox(
    center: torch.Tensor,
    lam: float | torch.Tensor,
    *,
    root_tol: float = 1e-7,
    certification_tol: float = 5e-6,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Globally solve the signed RatioNorm proximal subproblem.

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
            if tau <= root_tol:
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

    minimum_objective = min(candidate["objective"] for candidate in candidates)
    objective_tol = certification_tol * max(1.0, abs(minimum_objective))
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
        "kkt_residual": float(kkt_residual),
        "tie_rule": "objective_then_support_then_original_index",
    }
    return result, diagnostics


def soft_thresholding(v: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Soft thresholding operator: sign(v) * max(|v| - alpha, 0).

    This is the proximal operator for the L1 norm:
        prox_{alpha * ||.||_1}(v) = sign(v) * max(|v| - alpha, 0)

    Args:
        v: Input tensor
        alpha: Threshold (same shape as v or broadcastable)

    Returns:
        Soft-thresholded tensor
    """
    return torch.sign(v) * torch.clamp(torch.abs(v) - alpha, min=0.0)


def safe_norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safe L2 norm with epsilon for numerical stability.

    Args:
        v: Input tensor
        eps: Small constant for numerical stability

    Returns:
        L2 norm of v plus epsilon
    """
    return torch.norm(v, p=2) + eps


def safe_cbrt(x: torch.Tensor) -> torch.Tensor:
    """Safe cube root that handles negative numbers.

    For x < 0: cbrt(x) = -cbrt(|x|)
    For x >= 0: cbrt(x) = x^(1/3)

    Args:
        x: Input tensor

    Returns:
        Element-wise cube root
    """
    return torch.sign(x) * torch.abs(x) ** (1.0 / 3.0)


def solve_cubic_ratio_norm(D: torch.Tensor) -> torch.Tensor:
    """Solve tau^3 - tau - D = 0 for tau >= 0 using Cardano's formula.

    For the equation tau^3 - tau - D = 0:
    - Discriminant: Delta = D^2/4 - 1/27
    - If Delta >= 0: one real root tau = cbrt(D/2 + sqrt(Delta)) + cbrt(D/2 - sqrt(Delta))
    - If Delta < 0: three real roots, use trigonometric method

    This only solves the scalar cubic used by the legacy sequential
    soft-threshold-then-rescale RatioNorm update. It is not, by itself, an
    exact proximal solver for R(z) = ||z||_1 / ||z||_2.

    Args:
        D: Input tensor (should be non-negative for Ratio Norm)

    Returns:
        tau: Positive real root of the cubic equation

    References:
        Cardano's formula for cubic equations.
    """
    tau = torch.zeros_like(D)

    # Discriminant for tau^3 - tau - D = 0: Delta = D^2/4 - 1/27
    discriminant = (D / 2.0) ** 2 - 1.0 / 27.0

    # Case 1: discriminant >= 0 (one real root)
    case1 = discriminant >= 0
    if torch.any(case1):
        sqrt_disc = torch.sqrt(discriminant[case1])
        # tau = cbrt(D/2 + sqrt(Delta)) + cbrt(D/2 - sqrt(Delta))
        # FIX: Use safe_cbrt to handle negative values
        term1 = D[case1] / 2.0 + sqrt_disc
        term2 = D[case1] / 2.0 - sqrt_disc  # FIX: was -D/2 + sqrt_disc (wrong sign)
        tau[case1] = safe_cbrt(term1) + safe_cbrt(term2)

    # Case 2: discriminant < 0 (three real roots, use trigonometric method)
    case2 = ~case1
    if torch.any(case2):
        # For tau^3 - tau - D = 0, the trigonometric solution is:
        # tau = 2/sqrt(3) * cos(theta/3) where theta = arccos(3*sqrt(3)*D / 2)
        sqrt_3 = torch.sqrt(torch.tensor(3.0, device=D.device))
        arg = torch.clamp(3.0 * sqrt_3 * D[case2] / 2.0, min=-1.0, max=1.0)
        theta = torch.acos(arg)
        tau[case2] = 2.0 * torch.cos(theta / 3.0) / sqrt_3

    return tau


def _solve_cubic_denominator_copy_cardano(delta: torch.Tensor) -> torch.Tensor:
    """Cardano reference for lambda^3 - lambda^2 - delta = 0."""
    out_dtype = delta.dtype
    delta = torch.clamp(delta.double(), min=0.0)
    delta_hat = delta + 2.0 / 27.0
    D = torch.sqrt(torch.clamp(delta_hat**2 / 4.0 - 1.0 / 729.0, min=0.0))
    lam = (
        1.0 / 3.0
        + safe_cbrt(delta_hat / 2.0 + D)
        + safe_cbrt(delta_hat / 2.0 - D)
    )
    return lam.to(dtype=out_dtype)


def solve_cubic_denominator_copy(
    delta: torch.Tensor, num_iters: int = 6
) -> torch.Tensor:
    """Solve lambda^3 - lambda^2 - delta = 0 for the denominator-copy update.

    The denominator-copy update only needs the unique root lambda >= 1 for
    delta >= 0. Newton's method is monotone from the right with the initializer
    below because f(lambda)=lambda^2(lambda-1)-delta is increasing and convex
    on [1, infinity).
    """
    out_dtype = delta.dtype
    delta = torch.clamp(delta.double(), min=0.0)

    # This initializer is always on the right side of the lambda >= 1 root:
    # 1 + delta for small delta, and 1 + cbrt(delta) for large delta.
    lam = 1.0 + torch.where(delta <= 1.0, delta, delta ** (1.0 / 3.0))
    lam = torch.where(delta == 0.0, torch.ones_like(lam), lam)

    for _ in range(num_iters):
        f = lam * lam * (lam - 1.0) - delta
        fp = lam * (3.0 * lam - 2.0)
        lam = torch.clamp(lam - f / fp, min=1.0)

    return lam.to(dtype=out_dtype)


def strict_ratio_z_update(
    gate_value: torch.Tensor,
    denominator_copy: torch.Tensor,
    sparse_dual: torch.Tensor,
    C: float,
    rho: float,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update the sparse copy in strict two-copy RatioNorm ADMM.

    This solves, with ``denominator_copy`` fixed,

        C * ||z||_1 / ||y||_2 + (rho / 2) * ||z - (g + u)||_2^2.

    ``C`` is a single scalar. No feature score or coordinate-wise lambda is
    applied.
    """
    if C < 0.0:
        raise ValueError("C must be nonnegative")
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    denominator_norm = torch.linalg.vector_norm(denominator_copy, ord=2).clamp_min(eps)
    threshold = gate_value.new_tensor(float(C) / float(rho)) / denominator_norm
    center = gate_value + sparse_dual
    return soft_thresholding(center, threshold), threshold


def strict_ratio_weighted_z_update(
    gate_value: torch.Tensor,
    denominator_copy: torch.Tensor,
    sparse_dual: torch.Tensor,
    penalty_weights: torch.Tensor,
    rho: float,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update z for a score-weighted strict two-copy diagnostic.

    With ``penalty_weights[j] = C / score[j]``, this solves

        sum_j penalty_weights[j] * |z_j| / ||y||_2
        + (rho / 2) * ||z - (g + u)||_2^2.
    """
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if penalty_weights.shape != gate_value.shape:
        raise ValueError("penalty_weights must have the same shape as gate_value")
    if not torch.isfinite(penalty_weights).all() or torch.any(penalty_weights < 0):
        raise ValueError("penalty_weights must be finite and nonnegative")

    weights = penalty_weights.to(device=gate_value.device, dtype=gate_value.dtype)
    denominator_norm = torch.linalg.vector_norm(denominator_copy, ord=2).clamp_min(eps)
    threshold = weights / (float(rho) * denominator_norm)
    center = gate_value + sparse_dual
    return soft_thresholding(center, threshold), threshold


def _strict_ratio_y_update_from_alpha(
    gate_value: torch.Tensor,
    denominator_dual: torch.Tensor,
    alpha: torch.Tensor,
    rho: float,
    fallback_direction: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared radial y minimizer for scalar and weighted strict updates."""
    center = gate_value + denominator_dual
    center_norm = torch.linalg.vector_norm(center, ord=2)

    if float(alpha.item()) <= eps:
        one = gate_value.new_tensor(1.0)
        zero = gate_value.new_tensor(0.0)
        return center.clone(), one, zero

    if float(center_norm.item()) <= eps:
        direction = fallback_direction
        if direction is None or direction.shape != center.shape:
            direction = torch.zeros_like(center)
            direction.reshape(-1)[0] = 1.0
        else:
            direction = direction.to(device=center.device, dtype=center.dtype).clone()
        direction_norm = torch.linalg.vector_norm(direction, ord=2)
        if float(direction_norm.item()) <= eps:
            direction.zero_()
            direction.reshape(-1)[0] = 1.0
            direction_norm = gate_value.new_tensor(1.0)
        radius = torch.pow(alpha / float(rho), 1.0 / 3.0)
        y_new = radius * direction / direction_norm
        return y_new, gate_value.new_tensor(float("nan")), gate_value.new_tensor(float("inf"))

    delta = alpha / (float(rho) * center_norm**3)
    gamma = solve_cubic_denominator_copy(delta)
    return gamma * center, gamma, delta


def strict_ratio_y_update(
    gate_value: torch.Tensor,
    sparse_copy: torch.Tensor,
    denominator_dual: torch.Tensor,
    C: float,
    rho: float,
    fallback_direction: torch.Tensor | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update the denominator copy in strict two-copy RatioNorm ADMM.

    For ``q = g + w`` and ``alpha = C * ||z||_1``, the nonzero-center
    minimizer is ``y = gamma * q`` where

        gamma^3 - gamma^2 - alpha / (rho * ||q||_2^3) = 0.

    When ``q`` is numerically zero and ``alpha > 0``, the subproblem has a
    non-unique direction. A deterministic fallback direction is used with the
    exact optimal radius ``(alpha / rho)^(1/3)``.
    """
    if C < 0.0:
        raise ValueError("C must be nonnegative")
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    alpha = gate_value.new_tensor(float(C)) * torch.linalg.vector_norm(
        sparse_copy, ord=1
    )
    return _strict_ratio_y_update_from_alpha(
        gate_value,
        denominator_dual,
        alpha,
        rho,
        fallback_direction,
        eps,
    )


def strict_ratio_weighted_y_update(
    gate_value: torch.Tensor,
    sparse_copy: torch.Tensor,
    denominator_dual: torch.Tensor,
    penalty_weights: torch.Tensor,
    rho: float,
    fallback_direction: torch.Tensor | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update y for a score-weighted strict two-copy diagnostic."""
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if penalty_weights.shape != sparse_copy.shape:
        raise ValueError("penalty_weights must have the same shape as sparse_copy")
    if not torch.isfinite(penalty_weights).all() or torch.any(penalty_weights < 0):
        raise ValueError("penalty_weights must be finite and nonnegative")

    weights = penalty_weights.to(device=sparse_copy.device, dtype=sparse_copy.dtype)
    alpha = torch.sum(weights * torch.abs(sparse_copy))
    return _strict_ratio_y_update_from_alpha(
        gate_value,
        denominator_dual,
        alpha,
        rho,
        fallback_direction,
        eps,
    )


def solve_cubic_paper(D: torch.Tensor) -> torch.Tensor:
    """Solve cubic equation from the paper formulation.

    This uses the same Cardano's formula as solve_cubic_ratio_norm,
    adapted for the paper's specific cubic equation format.

    Args:
        D: Input tensor

    Returns:
        Positive real root of the cubic equation
    """
    return solve_cubic_ratio_norm(D)
