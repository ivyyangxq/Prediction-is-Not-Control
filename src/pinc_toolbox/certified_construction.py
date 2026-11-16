"""Exact rational fallback for the finite public-panel construction.

This isolated II engineering copy represents only the supplied finite,
already-constructed inputs.  It has no kappa (``κ``) interface, is not yet
integrated with the public eta adapter, and was not imported or executed in
this handoff.

This module is a proposal for the engineering queue.  It deliberately has no
NumPy/SciPy dependency and is not imported by the installed toolbox.  Finite
Python floats are decoded with :meth:`Fraction.from_float`; the returned
fractions therefore certify the exact binary-float values supplied to this
module.  They do not certify an upstream data conversion, a statistical
premise, or an interval surrounding an unknown real-valued input.

The reference routines use the identity

    B(theta - t) = z' (c - V t) - z' score(theta),  V z = B',

and bound the second term with the score box.  The monitoring routines retain
the unnormalised moments ``m = W (y - Phi t)``, ``B = W Phi`` and source
loading ``A = W loading``.  ``raw_box_certificate`` evaluates the source box
against the original radii, including zero-radius coordinates.  This is the
same geometry as the normalised backend under the mapping
``lambda_normalised[j] = lambda_raw[j] * radius[j]`` when radii are positive;
the raw form is required when a radius is zero.
"""
from __future__ import annotations

from fractions import Fraction
from math import isfinite, isqrt, nextafter
from typing import Any, Sequence


DEFAULT_LN_TERMS = 24
DEFAULT_SQRT_BITS = 80


def _q(value: Any) -> Fraction:
    """Decode an input without needlessly rounding an already exact rational.

    Internal ``Fraction``/integer values are preserved.  A string is accepted
    only through ``Fraction``'s explicit rational syntax (including decimal
    strings, which are exact decimal rationals); all other numeric values are
    interpreted as finite Python floats and decoded as their exact binary
    rationals.
    """
    if isinstance(value, bool):
        raise ValueError("certificate inputs must be finite numbers")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, str):
        try:
            return Fraction(value.strip())
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("rational strings must be parseable by Fraction") from exc
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("certificate inputs must be finite numbers") from exc
    if not isfinite(number):
        raise ValueError("certificate inputs must be finite numbers")
    return Fraction.from_float(number)


def _q_vector(value: Sequence[Any], name: str) -> list[Fraction]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a finite vector")
    return [_q(item) for item in value]


def _q_matrix(value: Sequence[Sequence[Any]], name: str) -> list[list[Fraction]]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a finite matrix")
    rows = []
    for row in value:
        rows.append(_q_vector(row, name))
    return rows


def _dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    if len(left) != len(right):
        raise ValueError("vector dimensions do not agree")
    return sum((x * y for x, y in zip(left, right)), Fraction(0))


def _mat_vec(matrix: Sequence[Sequence[Fraction]], vector: Sequence[Fraction]) -> list[Fraction]:
    return [_dot(row, vector) for row in matrix]


def _matmul(left: Sequence[Sequence[Fraction]], right: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    if not left:
        return []
    if not right:
        return [[] for _ in left]
    n_inner = len(right)
    if any(len(row) != n_inner for row in left):
        raise ValueError("left matrix is ragged")
    n_cols = len(right[0])
    if any(len(row) != n_cols for row in right):
        raise ValueError("right matrix is ragged")
    return [
        [sum((row[k] * right[k][j] for k in range(n_inner)), Fraction(0))
         for j in range(n_cols)]
        for row in left
    ]


def _transpose(matrix: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    if not matrix:
        return []
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix is ragged")
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(width)]


def _solve_fraction(matrix: Sequence[Sequence[Fraction]], rhs: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    """Solve ``matrix * X = rhs`` by exact Gauss-Jordan elimination."""
    a = [list(row) for row in matrix]
    b = [list(row) for row in rhs]
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("coefficient matrix must be nonempty and square")
    if len(b) != n or any(len(row) != len(b[0]) for row in b):
        raise ValueError("right-hand side dimensions do not agree")
    aug = [a[i] + b[i] for i in range(n)]
    width = len(aug[0])
    for col in range(n):
        pivot = next((row for row in range(col, n) if aug[row][col] != 0), None)
        if pivot is None:
            raise ValueError("Gram matrix is singular")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        divisor = aug[col][col]
        aug[col] = [value / divisor for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor != 0:
                aug[row] = [x - factor * y for x, y in zip(aug[row], aug[col])]
    return [row[n:width] for row in aug]


def _pow2_floor(value: Fraction) -> int:
    """Return k with 2**k <= value < 2**(k+1), for value > 0."""
    if value <= 0:
        raise ValueError("power-of-two decomposition requires a positive value")
    k = value.numerator.bit_length() - value.denominator.bit_length()
    if k >= 0:
        lower = Fraction(1 << k, 1)
    else:
        lower = Fraction(1, 1 << (-k))
    if lower > value:
        k -= 1
    else:
        if k >= 0:
            upper = Fraction(1 << (k + 1), 1)
        else:
            upper = Fraction(1, 1 << (-(k + 1)))
        if upper <= value:
            k += 1
    return k


def _pow2(k: int) -> Fraction:
    return Fraction(1 << k, 1) if k >= 0 else Fraction(1, 1 << (-k))


def ln_upper(value: Any, *, terms: int = DEFAULT_LN_TERMS) -> Fraction:
    """Return a rational upper bound for ``ln(value)`` when ``value > 1``.

    The range reduction is ``value = 2**k * u``, ``1 <= u < 2``.  For
    ``t=(u-1)/(u+1)``, the atanh series and its positive tail bound give a
    one-sided rational result.  The same expansion at ``u=2`` supplies
    ``ln(2)``.  ``terms=24`` is the declared finite truncation.
    """
    x = _q(value)
    if x <= 1:
        raise ValueError("ln_upper requires value > 1")
    if not isinstance(terms, int) or isinstance(terms, bool) or terms < 1:
        raise ValueError("terms must be a positive integer")

    def ln_unit(u: Fraction) -> Fraction:
        t = (u - 1) / (u + 1)
        t2 = t * t
        series = sum(
            (t ** (2 * j + 1)) / (2 * j + 1) for j in range(terms)
        )
        tail = (t ** (2 * terms + 1)) / ((2 * terms + 1) * (1 - t2))
        return 2 * (series + tail)

    k = _pow2_floor(x)
    u = x / _pow2(k)
    ln_two = ln_unit(Fraction(2))
    return k * ln_two + ln_unit(u)


def sqrt_upper(value: Any, *, bits: int = DEFAULT_SQRT_BITS) -> Fraction:
    """Return a rational upper bound for a nonnegative rational square root."""
    q = _q(value) if not isinstance(value, Fraction) else value
    if q < 0:
        raise ValueError("sqrt_upper requires a nonnegative value")
    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 1:
        raise ValueError("bits must be a positive integer")
    if q == 0:
        return Fraction(0)
    scale = 1 << (2 * bits)
    scaled_numerator = q.numerator * scale
    floor_root = isqrt(scaled_numerator // q.denominator)
    if floor_root * floor_root * q.denominator < scaled_numerator:
        floor_root += 1
    return Fraction(floor_root, 1 << bits)


def _upper_float(value: Fraction) -> float:
    """Convert an exact upper bound outward to a finite Python float."""
    number = float(value)
    if not isfinite(number):
        raise OverflowError("exact certificate endpoint does not fit in a float")
    if Fraction.from_float(number) < value:
        number = nextafter(number, float("inf"))
        if not isfinite(number):
            raise OverflowError("outward certificate endpoint does not fit in a float")
    return number


def reference_score_bounds(
    feature_caps: Sequence[Any], *, sigma_common: Any, sigma_idiosyncratic: Any,
    loading_bound: Any, households: int, periods: int, total_prefixes: int,
    alpha: Any, ln_terms: int = DEFAULT_LN_TERMS,
    sqrt_bits: int = DEFAULT_SQRT_BITS,
) -> dict[str, Any]:
    """Construct Lemma C score bounds as exact rational upper bounds."""
    caps = _q_vector(feature_caps, "feature_caps")
    if not caps or any(x < 0 for x in caps):
        raise ValueError("feature_caps must be nonempty and nonnegative")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 1
           for x in (households, periods, total_prefixes)):
        raise ValueError("households, periods and total_prefixes must be positive integers")
    su, sv, loading, a = map(_q, (sigma_common, sigma_idiosyncratic, loading_bound, alpha))
    if any(x < 0 for x in (su, sv, loading)) or not 0 < a < 1:
        raise ValueError("noise/loading bounds must be nonnegative and alpha must lie in (0,1)")
    log_argument = Fraction(2 * len(caps) * total_prefixes, 1) / a
    log_bound = ln_upper(log_argument, terms=ln_terms)
    scale = su * su * households * households * loading * loading + sv * sv * households
    bounds = [sqrt_upper(2 * periods * cap * cap * scale * log_bound, bits=sqrt_bits)
              for cap in caps]
    return {
        "score_bounds_exact": [str(x) for x in bounds],
        "score_bounds": [_upper_float(x) for x in bounds],
        "log_argument_exact": str(log_argument),
        "log_upper_exact": str(log_bound),
        "ln_terms": ln_terms,
        "sqrt_bits": sqrt_bits,
        "construction": "exact Fraction.from_float inputs with rational ln/sqrt upper bounds",
    }


def reference_projection(
    features: Sequence[Sequence[Any]], outcomes: Sequence[Any],
    theta: Sequence[Any], score_bounds: Sequence[Any],
    directions: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    """Certify projected reference errors around a candidate float ``theta``."""
    phi = _q_matrix(features, "features")
    y = _q_vector(outcomes, "outcomes")
    t = _q_vector(theta, "theta")
    b = _q_vector(score_bounds, "score_bounds")
    B = _q_matrix(directions, "directions")
    if not phi or len(phi) != len(y) or any(len(row) != len(t) for row in phi):
        raise ValueError("features/outcomes/theta dimensions do not agree")
    p = len(t)
    if len(b) != p or any(x < 0 for x in b) or any(len(row) != p for row in B):
        raise ValueError("score bounds or directions have the wrong dimension")
    V = [[sum((row[j] * row[k] for row in phi), Fraction(0)) for k in range(p)] for j in range(p)]
    c = [sum((row[j] * yi for row, yi in zip(phi, y)), Fraction(0)) for j in range(p)]
    d = [ci - _dot(Vi, t) for ci, Vi in zip(c, V)]
    solved = _solve_fraction(V, _transpose(B)) if B else []
    errors = []
    for direction_index, direction in enumerate(B):
        # ``solved`` is p x m because V X = B'.  Each column is the z vector
        # belonging to one requested direction B[direction_index].
        z = [solved[row][direction_index] for row in range(p)]
        center = abs(_dot(z, d))
        radius = sum((abs(zj) * bj for zj, bj in zip(z, b)), Fraction(0))
        errors.append(center + radius)
    return {
        "V_exact": [[str(x) for x in row] for row in V],
        "c_exact": [str(x) for x in c],
        "candidate_theta_exact": [str(x) for x in t],
        "score_residual_at_candidate_exact": [str(x) for x in d],
        "directions_exact": [[str(x) for x in row] for row in B],
        "solved_Vz_equals_B_transpose_exact": [[str(x) for x in row] for row in solved],
        "projection_error_exact": [str(x) for x in errors],
        "projection_error": [_upper_float(x) for x in errors],
        "construction": "exact V=Phi'Phi, c=Phi'y, Fraction Gaussian elimination",
    }


def scaled_linf_support(
    points: Sequence[Any], anchors: Sequence[Sequence[Any]], floor: Sequence[Any], eta: Any,
) -> dict[str, Any]:
    """Check exact nearest-anchor scaled-L-infinity support.

    The scale is ``max(max(anchor_j)-min(anchor_j), floor_j)`` coordinatewise.
    The result is true exactly at the declared boundary, avoiding a floating
    ``<=`` decision for eta selection.  This is observable geometry only; it
    is not a population-overlap or causal-identification certificate.
    """
    point = _q_vector(points, "point")
    A = _q_matrix(anchors, "anchors")
    f = _q_vector(floor, "floor")
    e = _q(eta)
    if not A or not point or len(f) != len(point) or any(len(row) != len(point) for row in A):
        raise ValueError("point, anchors and floor dimensions do not agree")
    if any(x <= 0 for x in f) or e < 0:
        raise ValueError("floor must be positive and eta nonnegative")
    scale = [max(max(row[j] for row in A) - min(row[j] for row in A), f[j])
             for j in range(len(point))]
    distances = [max((abs(point[j] - row[j]) / scale[j] for j in range(len(point))), default=Fraction(0))
                 for row in A]
    minimum = min(distances)
    return {
        "scale_exact": [str(x) for x in scale],
        "anchor_distances_exact": [str(x) for x in distances],
        "minimum_distance_exact": str(minimum),
        "eta_exact": str(e),
        "supported": minimum <= e,
        "construction": "exact anchor range/floor and scaled Linf comparison",
    }


def monitor_region(
    features: Sequence[Sequence[Any]], outcomes: Sequence[Any], weights: Sequence[Sequence[Any]],
    theta: Sequence[Any], source_loadings: Sequence[Sequence[Any]],
    *, sigma_common: Any, sigma_idiosyncratic: Any, loading_bound: Any,
    households: int, window_dates: int, measurement_dimension: int,
    horizon: int, alpha_noise: Any, ln_terms: int = DEFAULT_LN_TERMS,
    sqrt_bits: int = DEFAULT_SQRT_BITS,
) -> dict[str, Any]:
    """Build exact monitor moments, reference map and strict noise radius.

    ``weights`` are raw public weights with ``abs(w[j,i]) <= 1``.  The
    function performs the exact division by ``N * window_dates`` internally,
    matching the theorem's normalized moment.  ``source_loadings`` is an
    independent rectangular ``n_rows x n_source`` matrix; its column count
    need not equal the reference feature count and may be zero.
    """
    phi = _q_matrix(features, "features")
    y = _q_vector(outcomes, "outcomes")
    W = _q_matrix(weights, "weights")
    t = _q_vector(theta, "theta")
    loading = _q_matrix(source_loadings, "source_loadings")
    rows = len(phi)
    if rows == 0 or len(y) != rows or len(W) != measurement_dimension:
        raise ValueError("monitor matrix dimensions do not agree")
    if any(len(row) != rows for row in W) or len(loading) != rows:
        raise ValueError("weights/loadings must cover all rows")
    if any(len(row) != len(t) for row in phi):
        raise ValueError("feature dimensions do not agree")
    n_source = len(loading[0]) if loading else 0
    if any(len(row) != n_source for row in loading):
        raise ValueError("source_loadings must be a rectangular matrix")
    if any(abs(value) > 1 for row in W for value in row):
        raise ValueError("raw weights must satisfy abs(weight) <= 1")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 1
           for x in (households, window_dates, measurement_dimension, horizon)):
        raise ValueError("monitor counts must be positive integers")
    if window_dates > horizon:
        raise ValueError("window_dates must not exceed the finite monitoring horizon")
    if rows != households * window_dates:
        raise ValueError("rows must equal households times window_dates")
    denominator = Fraction(households * window_dates, 1)
    W = [[value / denominator for value in row] for row in W]
    residual = [yi - _dot(row, t) for yi, row in zip(y, phi)]
    m = _mat_vec(W, residual)
    B = _matmul(W, phi)
    A = _matmul(W, loading)
    su, sv, load, alpha = map(_q, (sigma_common, sigma_idiosyncratic, loading_bound, alpha_noise))
    if any(x < 0 for x in (su, sv, load)) or not 0 < alpha < 1:
        raise ValueError("noise/loading bounds must be nonnegative and alpha must lie in (0,1)")
    log_argument = Fraction(2 * measurement_dimension * horizon, 1) / alpha
    log_bound = ln_upper(log_argument, terms=ln_terms)
    proxy = su * su * load * load + sv * sv / households
    radius = sqrt_upper(2 * proxy * log_bound / window_dates, bits=sqrt_bits)
    return {
        "moments_exact": [str(x) for x in m],
        "raw_weights_exact": [[str(x * denominator) for x in row] for row in W],
        "normalised_weights_exact": [[str(x) for x in row] for row in W],
        "reference_projection_matrix_exact": [[str(x) for x in row] for row in B],
        "source_loading_exact": [[str(x) for x in row] for row in A],
        "noise_radius_exact": str(radius),
        "moments": [_upper_float(x) for x in m],
        "reference_projection_matrix": [[_upper_float(x) for x in row] for row in B],
        "source_loading": [[_upper_float(x) for x in row] for row in A],
        "noise_radius": _upper_float(radius),
        "log_argument_exact": str(log_argument),
        "log_upper_exact": str(log_bound),
        "construction": "exact W(y-Phi t), W Phi, W loading with rational log/sqrt upper radius",
    }


def raw_box_certificate(
    moments: Sequence[Any], radii: Sequence[Any],
    loading: Sequence[Sequence[Any]], lower: Sequence[Any], upper: Sequence[Any],
    *, primal: Sequence[Any] | None = None, dual: Sequence[Any] | None = None,
    threshold: Any = 1,
) -> dict[str, Any]:
    """Evaluate exact primal/dual evidence against an unnormalised box.

    A primal is projected coordinatewise into the source box and retained only
    when every residual obeys ``abs(m - A x) <= threshold * radius``.  For a
    dual vector lambda, the exact separating gap is

        lambda'm - support(A'lambda) - threshold * sum_j abs(lambda_j) radius_j.

    A positive gap is a sound exclusion certificate.  Zero-radius rows are
    handled directly; no division by a radius occurs.  For strictly positive
    radii this is equivalent to the old normalised formulation after mapping
    ``lambda_normalised[j] = lambda_raw[j] * radius[j]``.  The returned
    ``lower_bound_exact`` is the ordinary distance lower bound; the separate
    ``separation_gap_exact`` retains the post-threshold gap used for exclusion.
    """
    v, r = _q_vector(moments, "moments"), _q_vector(radii, "radii")
    A = _q_matrix(loading, "loading")
    lo, hi = _q_vector(lower, "lower"), _q_vector(upper, "upper")
    bound = _q(threshold)
    if not v or len(r) != len(v) or len(A) != len(v):
        raise ValueError("moments/radii/loading rows must agree and be nonempty")
    if any(x < 0 for x in r) or len(lo) != len(hi) or any(x > y for x, y in zip(lo, hi)):
        raise ValueError("radii must be nonnegative and box bounds ordered")
    if any(len(row) != len(lo) for row in A) or bound < 0:
        raise ValueError("loading columns or threshold have the wrong dimension")
    uq = None
    witness = None
    primal_feasible = None
    if primal is not None:
        candidate = _q_vector(primal, "primal")
        if len(candidate) != len(lo):
            raise ValueError("primal dimension mismatch")
        witness = [min(max(x, l), u) for x, l, u in zip(candidate, lo, hi)]
        residual = [abs(yi - _dot(row, witness)) for yi, row in zip(v, A)]
        primal_feasible = all(err <= bound * rad for err, rad in zip(residual, r))
        ratios = [err / rad for err, rad in zip(residual, r) if rad > 0]
        if primal_feasible:
            uq = max(ratios, default=Fraction(0))
    lq = None
    dual_gap = None
    dual_denominator = None
    dual_exact = None
    if dual is not None:
        lam = _q_vector(dual, "dual")
        if len(lam) != len(v):
            raise ValueError("dual dimension mismatch")
        coeff = [sum((lam[j] * A[j][k] for j in range(len(v))), Fraction(0))
                 for k in range(len(lo))]
        support = sum((max(c * l, c * u) for c, l, u in zip(coeff, lo, hi)), Fraction(0))
        dual_gap = _dot(lam, v) - support - bound * sum((abs(x) * y for x, y in zip(lam, r)), Fraction(0))
        dual_denominator = sum((abs(x) * y for x, y in zip(lam, r)), Fraction(0))
        if dual_denominator > 0:
            # ``dual_gap`` is the separation margin after subtracting the
            # threshold-radius allowance.  Report the ordinary distance lower
            # bound by adding that allowance back before dividing.
            lq = max(Fraction(0), (dual_gap + bound * dual_denominator) / dual_denominator)
        elif dual_gap > 0:
            lq = None  # Represent +infinity below without a float sentinel.
        else:
            lq = Fraction(0)
        dual_exact = {
            "lambda": [str(x) for x in lam],
            "coefficient": [str(x) for x in coeff],
            "support": str(support),
            "gap": str(dual_gap),
            "radius_weight": str(dual_denominator),
        }
    excluded = dual_gap is not None and dual_gap > 0
    compatible = uq is not None and uq <= bound
    status = "compatible" if compatible else ("excluded" if excluded else "unresolved")
    return {
        "status": status,
        "threshold_exact": str(bound),
        "upper_bound_exact": None if uq is None else str(uq),
        "lower_bound_exact": "infinity" if dual_gap is not None and dual_denominator == 0 and dual_gap > 0 else (None if lq is None else str(lq)),
        "primal_exact": None if witness is None else [str(x) for x in witness],
        "primal_feasible": primal_feasible,
        "dual": dual_exact,
        "separation_gap_exact": None if dual_gap is None else str(dual_gap),
        "dual_gap_positive": excluded,
        "mapping_to_normalised_backend": "lambda_normalised[j] = lambda_raw[j] * radius[j] for positive radii",
        "certification": "exact rational for supplied finite binary-float inputs; upstream construction not covered",
    }


__all__ = [
    "ln_upper", "sqrt_upper", "reference_score_bounds", "reference_projection",
    "scaled_linf_support", "monitor_region", "raw_box_certificate",
]
