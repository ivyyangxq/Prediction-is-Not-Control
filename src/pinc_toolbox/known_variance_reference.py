"""Exact single-sequence known-variance task-radius reference.

The caller supplies the already fitted unpenalized center.  This module only
constructs the represented-input normal equations and their conservative
known-variance score radius; it does not fit, tune, simulate, or certify an
arbitrary macro-data law.
"""
from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
import math
from typing import Any

from .certified_construction import _q, _solve_fraction, ln_upper, sqrt_upper
from .joint_score_reference import _ldl_positive


def _vector(value: Any, name: str) -> list[Fraction]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a finite vector")
    return [_q(item) for item in value]


def _matrix(value: Any, name: str) -> list[list[Fraction]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a finite matrix")
    rows = [_vector(row, name) for row in value]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} must be a nonempty rectangular matrix")
    return rows


def _dot(left, right):
    if len(left) != len(right):
        raise ValueError("vector dimensions do not agree")
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def _mat_vec(matrix, vector):
    return [_dot(row, vector) for row in matrix]


def _transpose(matrix):
    return [[matrix[row][column] for row in range(len(matrix))]
            for column in range(len(matrix[0]))]


def _mat_add(left, right):
    return [[a + b for a, b in zip(row_left, row_right)]
            for row_left, row_right in zip(left, right)]


def _matrix_strings(matrix):
    return [[str(value) for value in row] for row in matrix]


def _vector_strings(vector):
    return [str(value) for value in vector]


def _display(value):
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _display_matrix(matrix):
    return [[_display(value) for value in row] for row in matrix]


def build_known_variance_reference(features, outcomes, *, theta_float,
                                   variance_upper, alpha):
    """Build exact ``V,c,d,H,ell`` for one scalar response.

    ``features`` is a nonempty row-major matrix and ``outcomes`` and
    ``theta_float`` have matching lengths.  ``variance_upper`` is the declared
    positive scalar q in ``Lambda=qI`` and ``H=q(V+I)``.  All exact strings in
    the returned payload are authoritative; display floats are diagnostics.
    """
    phi = _matrix(features, "features")
    y = _vector(outcomes, "outcomes")
    theta = _vector(theta_float, "theta_float")
    p = len(phi[0])
    if len(y) != len(phi) or len(theta) != p:
        raise ValueError("features, outcomes, and theta_float dimensions disagree")
    q = _q(variance_upper)
    if q <= 0:
        raise ValueError("variance_upper must be positive")
    a = _q(alpha)
    if not 0 < a < 1:
        raise ValueError("alpha must lie in (0,1)")
    V = [[sum((row[i] * row[j] for row in phi), Fraction(0))
          for j in range(p)] for i in range(p)]
    c = [sum((row[j] * value for row, value in zip(phi, y)), Fraction(0))
         for j in range(p)]
    V_inv = _solve_fraction(V, [[Fraction(int(i == j)) for j in range(p)] for i in range(p)])
    d = [ci - vi for ci, vi in zip(c, _mat_vec(V, theta))]
    Lambda = [[q if i == j else Fraction(0) for j in range(p)] for i in range(p)]
    H = _mat_add(Lambda, [[q * value for value in row] for row in V])
    _, _, detH = _ldl_positive(H)
    detLambda = q ** p
    det_ratio = detH / detLambda
    ell = 2 * ln_upper(1 / a) + ln_upper(det_ratio)
    return {
        "status": "ready",
        "reference_method": "known_variance_single_sequence",
        "feature_width": p, "observation_count": len(phi), "rank": p,
        "variance_upper_exact": str(q), "alpha_exact": str(a),
        "V_exact": _matrix_strings(V), "V_inverse_exact": _matrix_strings(V_inv),
        "c_exact": _vector_strings(c), "d_exact": _vector_strings(d),
        "theta_center_exact": _vector_strings(theta),
        "Lambda_exact": _matrix_strings(Lambda), "H_exact": _matrix_strings(H),
        "det_H_exact": str(detH), "det_Lambda_exact": str(detLambda),
        "det_ratio_exact": str(det_ratio), "ell_exact": str(ell),
        "float_display": {"det_ratio": _display(det_ratio), "ell": _display(ell)},
        "scope": ("represented finite inputs and caller-supplied unpenalized center; "
                   "conditional known-variance/time-uniform MGF assumptions; "
                   "no arbitrary-macro guarantee"),
    }


def project_known_variance_reference(payload, directions):
    """Project exact reference radius for each direction row ``B``."""
    if not isinstance(payload, dict) or payload.get("status") != "ready":
        raise ValueError("payload must be a ready known-variance reference")
    V_inv = [[_q(value) for value in row] for row in payload["V_inverse_exact"]]
    H = [[_q(value) for value in row] for row in payload["H_exact"]]
    d = _vector(payload["d_exact"], "payload.d_exact")
    ell = _q(payload["ell_exact"])
    p = len(V_inv)
    rows = _matrix(directions, "directions")
    if any(len(row) != p for row in rows):
        raise ValueError("directions width does not match the reference")
    projections = []
    for direction in rows:
        z = _mat_vec(V_inv, direction)
        center = abs(_dot(z, d))
        quadratic = _dot(z, _mat_vec(H, z))
        if quadratic < 0:
            raise ValueError("projection quadratic form is not nonnegative")
        radius = center + sqrt_upper(ell * quadratic)
        projections.append({
            "direction_exact": _vector_strings(direction),
            "center_correction_exact": str(center),
            "quadratic_exact": str(quadratic),
            "radius_exact": str(radius),
            "center_correction": _display(center),
            "radius": _display(radius),
        })
    return {"status": "ready", "reference_method": payload["reference_method"],
            "rows": projections, "scope": payload["scope"]}


__all__ = ["build_known_variance_reference", "project_known_variance_reference"]
