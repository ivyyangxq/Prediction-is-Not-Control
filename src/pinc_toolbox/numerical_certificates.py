"""Exact certificate re-evaluation for a represented finite-box source LP.

No solver calls and no model fitting. Python Fractions treat finite input
floats as exact binary rationals. This certifies the supplied region and
source matrix only; it does not bound earlier moment/region construction
roundoff, or establish the statistical assumptions behind that region.
"""
from __future__ import annotations
from fractions import Fraction
import math
from typing import Any, Sequence


def _q(value: Any) -> Fraction:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("certificate inputs must be finite")
    return Fraction.from_float(number)


def _outward(value: Fraction, *, upper: bool) -> float | None:
    try:
        rounded = float(value)
    except OverflowError:
        return None
    if not math.isfinite(rounded):
        return None
    represented = Fraction.from_float(rounded)
    if (upper and represented < value) or (not upper and represented > value):
        rounded = math.nextafter(rounded, math.inf if upper else -math.inf)
    return rounded if math.isfinite(rounded) else None


def exact_box_certificate(
    moments: Sequence[float], radii: Sequence[float],
    loading: Sequence[Sequence[float]], lower: Sequence[float], upper: Sequence[float],
    *, primal: Sequence[float] | None = None, dual: Sequence[float] | None = None,
    threshold: float = 1.0,
) -> dict[str, Any]:
    """Bound min_{lo<=x<=hi} max_j |v_j-(Ax)_j|/rho_j exactly.

    A feasible primal is obtained by coordinatewise box projection. Any dual
    direction is normalized by its exact l1 norm if necessary. The exact
    rational inequality, with no tolerance expansion, determines status.
    Returned float endpoints are rounded outwards; rational values are also
    saved for auditing. Missing evidence remains unresolved.
    """
    v, r = list(map(_q, moments)), list(map(_q, radii))
    lo, hi = list(map(_q, lower)), list(map(_q, upper))
    a = [list(map(_q, row)) for row in loading]
    boundary = _q(threshold)
    if not v or len(r) != len(v) or len(a) != len(v):
        raise ValueError("nonempty moment/radius/loading rows must agree")
    if len(lo) != len(hi) or any(len(row) != len(lo) for row in a):
        raise ValueError("loading columns and box sizes must agree")
    if any(x <= 0 for x in r) or any(x > y for x, y in zip(lo, hi)) or boundary < 0:
        raise ValueError("radii must be positive and bounds ordered")
    uq, lq, witness, direction = None, None, None, None
    if primal is not None:
        candidate = list(map(_q, primal))
        if len(candidate) != len(lo):
            raise ValueError("primal dimension mismatch")
        witness = [min(max(x, l), u) for x, l, u in zip(candidate, lo, hi)]
        uq = max(abs(y-sum(c*x for c, x in zip(row, witness)))/rad
                 for y, row, rad in zip(v, a, r))
    if dual is not None:
        direction = list(map(_q, dual))
        if len(direction) != len(v):
            raise ValueError("dual dimension mismatch")
        norm = max(Fraction(1), sum(abs(x) for x in direction))
        direction = [x/norm for x in direction]
        coefficient = [sum(direction[j]*a[j][k]/r[j] for j in range(len(v)))
                       for k in range(len(lo))]
        support = sum(max(c*l, c*u) for c, l, u in zip(coefficient, lo, hi))
        lq = max(Fraction(0), sum(d*y/rad for d, y, rad in zip(direction, v, r))-support)
    if lq is not None and uq is not None and lq > uq:
        raise ArithmeticError("exact lower certificate exceeds exact upper certificate")
    status = "compatible" if uq is not None and uq <= boundary else (
        "excluded" if lq is not None and lq > boundary else "unresolved")
    return dict(status=status, threshold=float(threshold),
                upper_bound=None if uq is None else _outward(uq, upper=True),
                lower_bound=None if lq is None else _outward(lq, upper=False),
                upper_exact=None if uq is None else str(uq),
                lower_exact=None if lq is None else str(lq),
                primal_exact=None if witness is None else [str(x) for x in witness],
                dual_exact=None if direction is None else [str(x) for x in direction],
                certification="exact rational for supplied finite binary-float inputs",
                statistical_construction_roundoff="not covered")
