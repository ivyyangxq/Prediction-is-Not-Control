"""Finite public-history reference fit for a declared panel prefix."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, log, sqrt
from typing import Any

import numpy as np


_PUBLIC_FIELDS = (
    "household", "period", "decision_time", "outcome_release",
    "features_available_at", "features", "outcome", "noise_loading",
)
_OUTPUT_FIELDS = _PUBLIC_FIELDS + (
    "support_point", "weights", "weights_available_at",
    "source_loadings_available_at", "latent_loading", "mechanism_loading",
)


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _invalid(cutoff: int, reasons: list[str], selected_rows: list[dict[str, Any]], rank: int = 0) -> dict[str, Any]:
    return {"status": "not_ready", "theta_hat": None, "gram_inverse": None,
            "score_bounds": None, "rank": int(rank), "n_rows": len(selected_rows),
            "cutoff": cutoff, "reasons": list(dict.fromkeys(reasons)),
            "selected_rows": selected_rows}


def fit_prefix(
    rows: Sequence[Mapping[str, Any]], *, households: int, periods: int,
    total_prefixes: int, feature_bounds: Sequence[float],
    sigma_common: float, sigma_idiosyncratic: float, loading_bound: float,
    alpha: float, as_of: int,
) -> dict[str, Any]:
    """Fit the fixed ``households × periods`` released public prefix.

    ``feature_bounds`` are nonnegative absolute caps C_j. Future rows are
    filtered by the public release clock before any other field is inspected.
    Malformed selected rows and duplicate selected keys raise ``ValueError``;
    incomplete, domain-exited, and rank-deficient prefixes return
    ``status='not_ready'``.
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("rows must be a sequence of mappings")
    for name, value in (("households", households), ("periods", periods), ("total_prefixes", total_prefixes)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(as_of, int) or isinstance(as_of, bool):
        raise ValueError("as_of must be an integer")
    caps = tuple(_finite(value, "feature absolute cap") for value in feature_bounds)
    if not caps or any(cap < 0 for cap in caps):
        raise ValueError("feature_bounds must be a nonempty list of nonnegative caps")
    sigma_common = _finite(sigma_common, "sigma_common")
    sigma_idiosyncratic = _finite(sigma_idiosyncratic, "sigma_idiosyncratic")
    loading_bound = _finite(loading_bound, "loading_bound")
    alpha = _finite(alpha, "alpha")
    if sigma_common < 0 or sigma_idiosyncratic < 0 or loading_bound < 0:
        raise ValueError("noise proxies and loading_bound must be nonnegative")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")

    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    reasons: list[str] = []
    p = len(caps)
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("every row must be a mapping")
        if "household" not in raw or "period" not in raw:
            raise ValueError("every row must contain household and period")
        household, period = raw["household"], raw["period"]
        if not isinstance(household, int) or isinstance(household, bool) or household < 0:
            raise ValueError("household must be a nonnegative integer")
        if not isinstance(period, int) or isinstance(period, bool) or period < 0:
            raise ValueError("period must be a nonnegative integer")
        if not (0 <= household < households and 0 <= period < periods):
            continue
        # Release is the only field inspected before the row is known public.
        if "outcome_release" not in raw:
            raise ValueError("row must contain outcome_release")
        release = raw["outcome_release"]
        if not isinstance(release, int) or isinstance(release, bool):
            raise ValueError("outcome_release must be an integer")
        if release > as_of:
            continue
        for field in _PUBLIC_FIELDS:
            if field not in raw:
                raise ValueError(f"selected row is missing {field}")
        if (household, period) in seen:
            raise ValueError("duplicate public row for household and period")
        seen.add((household, period))
        decision_time = raw["decision_time"]
        available_at = raw["features_available_at"]
        if (not isinstance(decision_time, int) or isinstance(decision_time, bool)
                or not isinstance(available_at, int) or isinstance(available_at, bool)):
            raise ValueError("decision_time and features_available_at must be integers")
        if decision_time != period or release != period + 1:
            raise ValueError("selected row violates the declared public clock")
        if available_at > decision_time:
            raise ValueError("features are not available at decision_time")
        features = raw["features"]
        if not isinstance(features, Sequence) or isinstance(features, (str, bytes)) or len(features) != p:
            raise ValueError("features must be a vector matching feature_bounds")
        values = [_finite(value, "feature") for value in features]
        outcome = _finite(raw["outcome"], "outcome")
        loading = _finite(raw["noise_loading"], "noise_loading")
        item = {field: raw[field] for field in _OUTPUT_FIELDS if field in raw}
        item["features"] = values
        item["outcome"] = outcome
        item["noise_loading"] = loading
        selected.append(item)
        if any(abs(value) > cap for value, cap in zip(values, caps)):
            reasons.append("feature_domain_exit")
        if abs(loading) > loading_bound:
            reasons.append("noise_loading_domain_exit")

    expected = households * periods
    if len(selected) != expected:
        reasons.append("planned_denominator_mismatch")
    if len(seen) != expected:
        reasons.append("incomplete_full_prefix")
    if reasons:
        return _invalid(periods, reasons, selected)

    design = np.asarray([row["features"] for row in selected], dtype=float)
    outcome = np.asarray([row["outcome"] for row in selected], dtype=float)
    rank = int(np.linalg.matrix_rank(design))
    if rank < p:
        return _invalid(periods, ["rank_insufficient"], selected, rank)
    theta_hat, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    gram_inverse = np.linalg.solve(design.T @ design, np.eye(p))
    if not np.isfinite(theta_hat).all() or not np.isfinite(gram_inverse).all():
        raise ValueError("reference fit returned nonfinite coefficients or inverse")
    common_log = log((2.0 * p * total_prefixes) / alpha)
    scale = (sigma_common ** 2 * households ** 2 * loading_bound ** 2
             + sigma_idiosyncratic ** 2 * households)
    score_bounds = [sqrt(2.0 * periods * cap ** 2 * scale * common_log) for cap in caps]
    if not np.isfinite(np.asarray(score_bounds, dtype=float)).all():
        raise ValueError("score_bounds are nonfinite")
    return {"status": "ready", "theta_hat": [float(v) for v in theta_hat],
            "gram_inverse": [[float(v) for v in row] for row in gram_inverse],
            "score_bounds": [float(v) for v in score_bounds], "rank": rank,
            "n_rows": len(selected), "cutoff": periods, "reasons": [],
            "selected_rows": selected}


__all__ = ["fit_prefix"]
