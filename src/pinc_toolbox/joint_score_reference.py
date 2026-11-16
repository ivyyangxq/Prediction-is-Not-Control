"""Exact reusable joint-score reference and projection construction.

``build_joint_reference`` receives an already released, current prefix and
the unchanged unpenalized center supplied by the caller's existing fit API.
It constructs exact represented-input ``V``, ``c``, predictable block
covariance, and the two predeclared Lambda choices ``k=1,128``.  It does not
fit a penalized model, use truth rows, generate data, or optimize Lambda.

``project_joint_reference`` accepts arbitrary current-window direction rows
``B`` with the same feature width.  It applies the same time-uniform family
budget used for initialization and returns the exact center compensation plus
the square-root support radius for both Lambda choices and their legal
coordinatewise minimum.  Float fields are diagnostics only; exact strings are
authoritative.  The statistical MGF and represented-input premises remain
explicit scope conditions of the returned payload.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
import math
from typing import Any

from .certified_construction import _q, _solve_fraction, ln_upper, sqrt_upper


SCALES = (1, 128)
DEFAULT_BRANCH_COUNT = 4


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _vector(value: Any, name: str) -> list[Fraction]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a finite vector")
    return [_q(item) for item in value]


def _matrix(value: Any, name: str) -> list[list[Fraction]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a finite matrix")
    result = [_vector(row, name) for row in value]
    if result and any(len(row) != len(result[0]) for row in result):
        raise ValueError(f"{name} must be rectangular")
    return result


def _dot(left: Sequence[Any], right: Sequence[Any]) -> Fraction:
    if len(left) != len(right):
        raise ValueError("vector dimensions do not agree")
    return sum((_q(a) * _q(b) for a, b in zip(left, right)), Fraction(0))


def _matrix_vector(matrix: Sequence[Sequence[Any]], vector: Sequence[Any]) -> list[Fraction]:
    return [_dot(row, vector) for row in matrix]


def _transpose(matrix: Sequence[Sequence[Any]]) -> list[list[Any]]:
    if not matrix:
        return []
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix is ragged")
    return [[matrix[row][column] for row in range(len(matrix))] for column in range(width)]


def _outer(vector: Sequence[Any]) -> list[list[Fraction]]:
    return [[_q(left) * _q(right) for right in vector] for left in vector]


def _matrix_strings(matrix: Sequence[Sequence[Any]]) -> list[list[str]]:
    return [[str(_q(value)) for value in row] for row in matrix]


def _vector_strings(vector: Sequence[Any]) -> list[str]:
    return [str(_q(value)) for value in vector]


def _display(value: Any) -> float | None:
    try:
        number = float(_q(value))
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _display_vector(vector: Sequence[Any]) -> list[float | None]:
    return [_display(value) for value in vector]


def _display_matrix(matrix: Sequence[Sequence[Any]]) -> list[list[float | None]]:
    return [[_display(value) for value in row] for row in matrix]


def _ldl_positive(matrix: Sequence[Sequence[Any]]) -> tuple[list[list[Fraction]], list[Fraction], Fraction]:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("LDL matrix must be nonempty and square")
    lower = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for index in range(n):
        lower[index][index] = Fraction(1)
    pivots: list[Fraction] = []
    for index in range(n):
        pivot = _q(matrix[index][index]) - sum(
            ((lower[index][prior] ** 2) * pivots[prior] for prior in range(index)), Fraction(0)
        )
        if pivot <= 0:
            raise ValueError(f"exact LDL pivot {index} is not positive")
        pivots.append(pivot)
        for row in range(index + 1, n):
            numerator = _q(matrix[row][index]) - sum(
                (lower[row][prior] * lower[index][prior] * pivots[prior]
                 for prior in range(index)), Fraction(0)
            )
            lower[row][index] = numerator / pivot
    return lower, pivots, math.prod(pivots, start=Fraction(1))


def _budget(alpha_reference: Any, family_budget: Mapping[str, Any], scale_count: int) -> dict[str, Any]:
    if not isinstance(family_budget, Mapping):
        raise ValueError("family_budget is required and must be explicit")
    alpha = _q(alpha_reference)
    declared_alpha = _q(family_budget.get("alpha_reference", alpha_reference))
    if declared_alpha != alpha:
        raise ValueError("family_budget.alpha_reference must match alpha_reference")
    branches = _positive_int(family_budget.get("branch_count"), "family_budget.branch_count")
    declared_scales = _positive_int(family_budget.get("scale_count", scale_count), "family_budget.scale_count")
    if declared_scales != scale_count:
        raise ValueError("family_budget.scale_count must match the two predeclared scales")
    denominator = branches * declared_scales
    if not 0 < alpha < 1:
        raise ValueError("alpha_reference must lie strictly between zero and one")
    explicit_delta = family_budget.get("delta_per_scale")
    delta = alpha / denominator if explicit_delta is None else _q(explicit_delta)
    if delta <= 0 or denominator * delta > alpha:
        raise ValueError("delta_per_scale must be positive and fit within alpha_reference")
    return {
        "alpha_reference_exact": str(alpha),
        "branch_count": branches,
        "scale_count": declared_scales,
        "allocation_count": denominator,
        "delta_exact": str(delta),
        "delta_source": "alpha_reference/allocation_count" if explicit_delta is None else "family_budget.delta_per_scale",
        "time_uniform": True,
        "declared_t_max": family_budget.get("declared_t_max"),
        "declared_n_count": family_budget.get("declared_n_count", branches),
        "allocation_note": "time-uniform J4 scope; no additional union over dates or task rows",
    }


def _scale_payload(V: Sequence[Sequence[Fraction]], omega: Sequence[Sequence[Fraction]],
                   sigma_common: Fraction, sigma_idiosyncratic: Fraction,
                   loading_bound: Fraction, households: int, caps: Sequence[Fraction],
                   scale: int, budget: Mapping[str, Any]) -> dict[str, Any]:
    p = len(caps)
    variance_scale = (sigma_idiosyncratic ** 2 * households
                      + sigma_common ** 2 * households ** 2 * loading_bound ** 2)
    Lambda = [[Fraction(0) for _ in range(p)] for _ in range(p)]
    for index, cap in enumerate(caps):
        Lambda[index][index] = scale * variance_scale * cap * cap
    H = [[_q(omega[row][column]) + Lambda[row][column]
          for column in range(p)] for row in range(p)]
    _, pivots, detH = _ldl_positive(H)
    detLambda = math.prod((Lambda[index][index] for index in range(p)), start=Fraction(1))
    if detLambda <= 0:
        raise ValueError("Lambda must be positive definite; feature caps and noise scale must be positive")
    ratio = detH / detLambda
    if ratio < 1:
        raise ValueError("determinant ratio must be at least one")
    delta = _q(budget["delta_exact"])
    log_delta = ln_upper(Fraction(1, 1) / delta)
    log_ratio = Fraction(0) if ratio == 1 else ln_upper(ratio)
    q_plus = 2 * log_delta + log_ratio
    return {
        "scale_factor_exact": str(_q(scale)),
        "Lambda_exact": _matrix_strings(Lambda),
        "H_exact": _matrix_strings(H),
        "LDL_pivots_exact": _vector_strings(pivots),
        "detH_exact": str(detH), "detLambda_exact": str(detLambda),
        "det_ratio_exact": str(ratio), "log_delta_upper_exact": str(log_delta),
        "log_det_ratio_upper_exact": str(log_ratio), "q_plus_exact": str(q_plus),
        "float_display": {"det_ratio": _display(ratio), "q_plus": _display(q_plus)},
    }


def build_joint_reference(
    rows: Sequence[Mapping[str, Any]], *, theta_float: Sequence[Any],
    feature_caps: Sequence[Any], sigma_common: Any, sigma_idiosyncratic: Any,
    loading_bound: Any, households: int, periods: int, total_prefixes: int,
    alpha_reference: Any, family_budget: Mapping[str, Any],
    feature_key: str = "features", outcome_key: str = "outcome",
    period_key: str = "period", loading_key: str = "noise_loading",
) -> dict[str, Any]:
    """Build an exact reference payload for one released prefix.

    ``theta_float`` is the unchanged unpenalized center from the caller's
    existing fit API.  Rows are represented exactly as supplied, including
    finite Python floats via ``Fraction.from_float``.  Dates are grouped by
    their sorted public period values; no N/T/P dimension is hard-coded.
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("rows must be a nonempty sequence")
    households = _positive_int(households, "households")
    periods = _positive_int(periods, "periods")
    total_prefixes = _positive_int(total_prefixes, "total_prefixes")
    caps = _vector(feature_caps, "feature_caps")
    if not caps or any(cap < 0 for cap in caps):
        raise ValueError("feature_caps must be nonnegative")
    p = len(caps)
    theta = _vector(theta_float, "theta_float")
    if len(theta) != p:
        raise ValueError("theta_float and feature_caps dimensions do not agree")
    if len(rows) != households * periods:
        raise ValueError("rows must equal households times periods")
    grouped: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or period_key not in row:
            raise ValueError("each row must contain the public period")
        grouped.setdefault(row[period_key], []).append(row)
    dates = sorted(grouped)
    if len(dates) != periods or any(len(grouped[date]) != households for date in dates):
        raise ValueError("each public period must contain exactly households rows")
    sigma_c, sigma_v, loading_cap = map(_q, (sigma_common, sigma_idiosyncratic, loading_bound))
    if any(value < 0 for value in (sigma_c, sigma_v, loading_cap)):
        raise ValueError("noise proxies and loading_bound must be nonnegative")
    V = [[Fraction(0) for _ in range(p)] for _ in range(p)]
    c = [Fraction(0) for _ in range(p)]
    omega_common = [[Fraction(0) for _ in range(p)] for _ in range(p)]
    for date in dates:
        g = [Fraction(0) for _ in range(p)]
        for row in grouped[date]:
            phi = _vector(row.get(feature_key), feature_key)
            if len(phi) != p:
                raise ValueError("public feature width does not match feature_caps")
            if any(abs(phi[index]) > caps[index] for index in range(p)):
                raise ValueError("public feature exceeds feature_caps")
            outcome = _q(row.get(outcome_key))
            loading = _q(row.get(loading_key, 1))
            if abs(loading) > loading_cap:
                raise ValueError("public loading exceeds loading_bound")
            for index in range(p):
                c[index] += phi[index] * outcome
                g[index] += phi[index] * loading
                for column in range(p):
                    V[index][column] += phi[index] * phi[column]
        outer = _outer(g)
        for index in range(p):
            for column in range(p):
                omega_common[index][column] += sigma_c ** 2 * outer[index][column]
    omega_idio = [[sigma_v ** 2 * V[row][column] for column in range(p)] for row in range(p)]
    omega = [[omega_common[row][column] + omega_idio[row][column]
              for column in range(p)] for row in range(p)]
    identity = [[Fraction(1 if row == column else 0) for column in range(p)]
                for row in range(p)]
    _solve_fraction(V, identity)
    d = [c[index] - _dot(V[index], theta) for index in range(p)]
    budget = _budget(alpha_reference, family_budget, len(SCALES))
    scales = [_scale_payload(V, omega, sigma_c, sigma_v, loading_cap, households, caps, scale, budget)
              for scale in SCALES]
    return {
        "schema": "joint_score_reference.v1",
        "reference_region_method": "joint_score",
        "status": "ready", "dimensions": {"features": p, "households": households,
                                              "periods": periods, "rows": len(rows),
                                              "total_prefixes": total_prefixes},
        "periods_exact": [str(_q(date)) for date in dates],
        "feature_caps_exact": _vector_strings(caps),
        "sigma_common_exact": str(sigma_c), "sigma_idiosyncratic_exact": str(sigma_v),
        "loading_bound_exact": str(loading_cap),
        "theta_float": [_display(value) for value in theta],
        "theta_exact_from_float": _vector_strings(theta),
        "V_exact": _matrix_strings(V), "c_exact": _vector_strings(c),
        "score_residual_at_center_exact": _vector_strings(d),
        "Omega_common_exact": _matrix_strings(omega_common),
        "Omega_idiosyncratic_exact": _matrix_strings(omega_idio),
        "Omega_total_exact": _matrix_strings(omega),
        "family_budget": dict(budget),
        "scales": scales,
        "premise_scope": {
            "represented_input_exact": True,
            "unpenalized_center_preserved": True,
            "uniform_over_time": True,
            "future_projection_directions_allowed": True,
            "statistical_mgf_premise": "caller-maintained predictable block-MGF and stopped-prefix assumptions",
            "coverage_claim": "not established by this construction alone",
        },
        "source_fields": {"feature": feature_key, "outcome": outcome_key,
                          "period": period_key, "loading": loading_key},
    }


def project_joint_reference(payload: Mapping[str, Any], B: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """Project one exact reference payload onto arbitrary current-window B rows."""
    if not isinstance(payload, Mapping) or payload.get("schema") != "joint_score_reference.v1":
        raise ValueError("payload is not a joint_score_reference.v1 object")
    V = _matrix(payload.get("V_exact"), "payload.V_exact")
    c = _vector(payload.get("c_exact"), "payload.c_exact")
    theta = _vector(payload.get("theta_exact_from_float"), "payload.theta_exact_from_float")
    d = _vector(payload.get("score_residual_at_center_exact"), "payload.score_residual_at_center_exact")
    directions = _matrix(B, "B")
    p = len(V)
    if len(c) != p or len(theta) != p or len(d) != p or any(len(row) != p for row in directions):
        raise ValueError("B and payload dimensions do not agree")
    if not directions:
        raise ValueError("B must contain at least one direction")
    solved = _solve_fraction(V, _transpose(directions))
    rows = []
    for direction_index, direction in enumerate(directions):
        z = [solved[index][direction_index] for index in range(p)]
        compensation = abs(_dot(z, d))
        candidates = []
        for scale_payload in payload.get("scales", []):
            scale = scale_payload["scale_factor_exact"]
            H = _matrix(scale_payload["H_exact"], "payload.H_exact")
            omega_common = _matrix(payload["Omega_common_exact"], "payload.Omega_common_exact")
            omega_idio = _matrix(payload["Omega_idiosyncratic_exact"], "payload.Omega_idiosyncratic_exact")
            Lambda = _matrix(scale_payload["Lambda_exact"], "payload.Lambda_exact")
            q_plus = _q(scale_payload["q_plus_exact"])
            common_quad = _dot(z, _matrix_vector(omega_common, z))
            idio_quad = _dot(z, _matrix_vector(omega_idio, z))
            lambda_quad = _dot(z, _matrix_vector(Lambda, z))
            total_quad = _dot(z, _matrix_vector(H, z))
            if total_quad != common_quad + idio_quad + lambda_quad:
                raise ValueError("payload covariance decomposition is inconsistent")
            sqrt_argument = q_plus * total_quad
            radius = compensation + sqrt_upper(sqrt_argument)
            candidates.append({
                "scale_factor_exact": scale, "status": "compatible_region",
                "radius_exact": str(radius), "compensation_exact": str(compensation),
                "common_quadratic_exact": str(common_quad),
                "idiosyncratic_quadratic_exact": str(idio_quad),
                "Lambda_quadratic_exact": str(lambda_quad),
                "total_quadratic_exact": str(total_quad),
                "sqrt_argument_exact": str(sqrt_argument), "q_plus_exact": str(q_plus),
                "B_exact": _vector_strings(direction), "z_exact": _vector_strings(z),
                "float_display": {"radius": _display(radius), "compensation": _display(compensation),
                                  "common_quadratic": _display(common_quad),
                                  "idiosyncratic_quadratic": _display(idio_quad),
                                  "Lambda_quadratic": _display(lambda_quad),
                                  "q_plus": _display(q_plus)},
            })
        if not candidates:
            raise ValueError("payload contains no Lambda scales")
        selected = min(candidates, key=lambda item: _q(item["radius_exact"]))
        rows.append({
            "direction_index": direction_index,
            "scales": candidates,
            "minimum": {"status": "compatible_region", "radius_exact": selected["radius_exact"],
                         "selected_scale_factor_exact": selected["scale_factor_exact"],
                         "candidate_radii_exact": {item["scale_factor_exact"]: item["radius_exact"] for item in candidates}},
        })
    return {
        "schema": "joint_score_projection.v1", "status": "ready",
        "reference_region_method": "joint_score",
        "direction_count": len(rows), "feature_dimension": p,
        "rows": rows,
        "premise_scope": payload["premise_scope"],
    }


__all__ = ["build_joint_reference", "project_joint_reference", "SCALES"]
