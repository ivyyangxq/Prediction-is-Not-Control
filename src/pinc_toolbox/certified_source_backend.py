"""Exact-input source-image diagnosis with an LP search front end.

The linear program in this module is deliberately only a candidate finder.
Its floating point primal and row marginals are sent back to
``raw_box_certificate`` together with the original rational inputs.  A class
is therefore affirmative only when the exact checker verifies a witness or a
separating direction.  This module does not construct a statistical region,
select an eta, or provide any continuation-value statement.
"""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from math import isfinite
import time
from typing import Any

from .certified_construction import raw_box_certificate


class CertifiedSourceBackendError(ValueError):
    """Raised when the exact source specification is malformed."""


def _fraction(value: Any, *, name: str) -> Fraction:
    """Decode one authoritative value without rounding exact rational input."""
    if isinstance(value, bool):
        raise CertifiedSourceBackendError(f"{name} must be a finite number")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, str):
        try:
            return Fraction(value.strip())
        except (ValueError, ZeroDivisionError) as exc:
            raise CertifiedSourceBackendError(
                f"{name} must be a parseable rational string"
            ) from exc
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CertifiedSourceBackendError(f"{name} must be a finite number") from exc
    if not isfinite(number):
        raise CertifiedSourceBackendError(f"{name} must be a finite number")
    return Fraction.from_float(number)


def _vector(value: Any, *, name: str, allow_empty: bool = True) -> list[Fraction]:
    if isinstance(value, (str, bytes, Mapping)):
        raise CertifiedSourceBackendError(f"{name} must be a finite vector")
    try:
        items = list(value)
    except TypeError as exc:
        raise CertifiedSourceBackendError(f"{name} must be a finite vector") from exc
    if not allow_empty and not items:
        raise CertifiedSourceBackendError(f"{name} must be nonempty")
    return [_fraction(item, name=name) for item in items]


def _matrix(value: Any, *, name: str, rows: int) -> list[list[Fraction]]:
    if isinstance(value, (str, bytes, Mapping)):
        raise CertifiedSourceBackendError(f"{name} must be a finite matrix")
    try:
        raw_rows = list(value)
    except TypeError as exc:
        raise CertifiedSourceBackendError(f"{name} must be a finite matrix") from exc
    if len(raw_rows) != rows:
        raise CertifiedSourceBackendError(
            f"{name} must have {rows} rows; got {len(raw_rows)}"
        )
    result: list[list[Fraction]] = []
    width: int | None = None
    for row_index, row in enumerate(raw_rows):
        if isinstance(row, (str, bytes, Mapping)):
            raise CertifiedSourceBackendError(f"{name}[{row_index}] must be a vector")
        try:
            items = list(row)
        except TypeError as exc:
            raise CertifiedSourceBackendError(
                f"{name}[{row_index}] must be a vector"
            ) from exc
        if width is None:
            width = len(items)
        elif len(items) != width:
            raise CertifiedSourceBackendError(f"{name} must be rectangular")
        result.append([_fraction(item, name=name) for item in items])
    return result


def _box(value: Any, *, name: str, dimension: int) -> tuple[list[Fraction], list[Fraction]]:
    if isinstance(value, (str, bytes, Mapping)):
        raise CertifiedSourceBackendError(f"{name} must be a (lower, upper) pair")
    try:
        pair = list(value)
    except TypeError as exc:
        raise CertifiedSourceBackendError(f"{name} must be a (lower, upper) pair") from exc
    if len(pair) != 2:
        raise CertifiedSourceBackendError(f"{name} must be a (lower, upper) pair")
    lower = _vector(pair[0], name=f"{name}.lower")
    upper = _vector(pair[1], name=f"{name}.upper")
    if len(lower) != dimension or len(upper) != dimension:
        raise CertifiedSourceBackendError(
            f"{name} bounds must have dimension {dimension}"
        )
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise CertifiedSourceBackendError(f"{name} lower bounds must not exceed upper bounds")
    if any(lo > 0 or hi < 0 for lo, hi in zip(lower, upper)):
        raise CertifiedSourceBackendError(f"{name} bounds must contain zero")
    return lower, upper


def _exact(value: Any) -> Any:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, list):
        return [_exact(item) for item in value]
    if isinstance(value, tuple):
        return [_exact(item) for item in value]
    return value


def _float(value: Fraction, *, name: str) -> float:
    try:
        converted = float(value)
    except OverflowError as exc:
        raise CertifiedSourceBackendError(
            f"{name} cannot be represented for LP candidate search"
        ) from exc
    if not isfinite(converted):
        raise CertifiedSourceBackendError(
            f"{name} cannot be represented for LP candidate search"
        )
    return converted


def _float_matrix(matrix: list[list[Fraction]], *, name: str) -> list[list[float]]:
    return [[_float(value, name=name) for value in row] for row in matrix]


def _finite_float_list(value: Any) -> tuple[bool, list[float]]:
    try:
        converted = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return False, []
    return bool(all(isfinite(item) for item in converted)), converted


def _search_one(
    moments: list[Fraction],
    radii: list[Fraction],
    loading: list[list[Fraction]],
    lower: list[Fraction],
    upper: list[Fraction],
    *,
    solver_options: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], int]:
    """Run one unnormalised Chebyshev LP and retain raw search output."""
    rows = len(moments)
    columns = len(lower)
    try:
        float_moments = [_float(value, name="moments") for value in moments]
        float_radii = [_float(value, name="radii") for value in radii]
        float_loading = _float_matrix(loading, name="loading")
        float_lower = [_float(value, name="lower") for value in lower]
        float_upper = [_float(value, name="upper") for value in upper]
    except CertifiedSourceBackendError as exc:
        return {
            "status": "unresolved",
            "reason": str(exc),
            "solver_success": False,
            "primal": None,
            "dual": None,
        }, 0

    # Importing SciPy lazily keeps static inspection and the exact checker
    # independent of NumPy.  The LP itself remains a floating search only.
    from scipy.optimize import linprog

    positive = [row[:] + [-float_radii[index]] for index, row in enumerate(float_loading)]
    negative = [[-item for item in row] + [-float_radii[index]]
                for index, row in enumerate(float_loading)]
    a_ub = positive + negative
    b_ub = float_moments + [-item for item in float_moments]
    objective = [0.0] * columns + [1.0]
    variable_bounds = list(zip(float_lower, float_upper)) + [(0.0, None)]
    try:
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=variable_bounds,
            method="highs",
            options=dict(solver_options or {}),
        )
    except MemoryError:
        return {
            "status": "unresolved",
            "reason": "solver_memory_error",
            "solver_success": False,
            "primal": None,
            "dual": None,
        }, 1
    except (FloatingPointError, RuntimeError) as exc:
        return {
            "status": "unresolved",
            "reason": f"solver_failure:{type(exc).__name__}",
            "solver_message": str(exc),
            "solver_success": False,
            "primal": None,
            "dual": None,
        }, 1

    raw_x = getattr(result, "x", None)
    finite_x, x = _finite_float_list(raw_x) if raw_x is not None else (False, [])
    solver_success = bool(getattr(result, "success", False))
    search: dict[str, Any] = {
        "solver_success": solver_success,
        "solver_status": getattr(result, "status", None),
        "solver_message": str(getattr(result, "message", "")),
        "raw_solution": x if finite_x else None,
        "raw_objective": x[-1] if finite_x and len(x) == columns + 1 else None,
        "primal": None,
        "dual": None,
    }
    if not solver_success:
        search["status"] = "unresolved"
        search["reason"] = "solver_failure"
        return search, 1
    if not finite_x or len(x) != columns + 1:
        search["status"] = "unresolved"
        search["reason"] = "nonfinite_or_malformed_solver_primal"
        return search, 1

    primal = x[:-1]
    search["primal"] = primal
    dual_value = None
    inequality = getattr(result, "ineqlin", None)
    marginals = getattr(inequality, "marginals", None) if inequality is not None else None
    finite_marginals, marginal_values = (
        _finite_float_list(marginals) if marginals is not None else (False, [])
    )
    if finite_marginals and len(marginal_values) == 2 * rows:
        # SciPy's <= marginals have the sign needed here: lambda is the
        # positive-row marginal minus the negative-row marginal.
        dual_value = [
            marginal_values[index] - marginal_values[rows + index]
            for index in range(rows)
        ]
        if all(isfinite(item) for item in dual_value):
            search["dual"] = dual_value
        else:
            search["dual_reason"] = "nonfinite_reconstructed_solver_marginals"
    elif marginals is not None:
        search["dual_reason"] = "nonfinite_or_malformed_solver_marginals"

    search["status"] = "search_complete"
    return search, 1


def _candidate(
    *,
    label: str,
    moments: list[Fraction],
    radii: list[Fraction],
    loading: list[list[Fraction]],
    lower: list[Fraction],
    upper: list[Fraction],
    solver_options: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    try:
        search, calls = _search_one(
            moments, radii, loading, lower, upper, solver_options=solver_options
        )
    except MemoryError:
        elapsed = time.monotonic() - started
        if not isfinite(elapsed):
            elapsed = 0.0
        return {
            "candidate": label,
            "status": "unresolved",
            "description": _candidate_description(label, "unresolved"),
            "reason": "solver_search_memory_error",
            "search": {"status": "unresolved", "reason": "solver_search_memory_error"},
            "certificate": {
                "status": "unresolved",
                "threshold_exact": "1",
                "certification": "solver search resource exhaustion",
            },
            "moments_exact": _exact(moments),
            "radii_exact": _exact(radii),
            "loading_exact": _exact(loading),
            "lower_exact": _exact(lower),
            "upper_exact": _exact(upper),
            "solver_calls": 0,
            "elapsed_seconds": elapsed,
        }, 0
    primal = search.get("primal")
    dual = search.get("dual")
    try:
        certificate = raw_box_certificate(
            moments,
            radii,
            loading,
            lower,
            upper,
            primal=primal,
            dual=dual,
            threshold=1,
        )
    except MemoryError:
        certificate = {
            "status": "unresolved",
            "threshold_exact": "1",
            "upper_bound_exact": None,
            "lower_bound_exact": None,
            "primal_exact": None,
            "primal_feasible": None,
            "dual": None,
            "separation_gap_exact": None,
            "dual_gap_positive": False,
            "certification": "exact checker resource exhaustion",
        }
        search = dict(search)
        search.setdefault("reason", "exact_checker_memory_error")
    status = certificate["status"]
    reason = None
    if status == "unresolved":
        reason = search.get("reason") or certificate.get("certification")
    elapsed = time.monotonic() - started
    if not isfinite(elapsed):
        elapsed = 0.0
    evidence = {
        "candidate": label,
        "status": status,
        "description": _candidate_description(label, status),
        "reason": reason,
        "search": search,
        "certificate": certificate,
        "moments_exact": _exact(moments),
        "radii_exact": _exact(radii),
        "loading_exact": _exact(loading),
        "lower_exact": _exact(lower),
        "upper_exact": _exact(upper),
        "solver_calls": calls,
        "elapsed_seconds": elapsed,
    }
    return evidence, calls


def _candidate_description(label: str, status: str) -> str:
    if status == "compatible":
        return f"{label} has an exact-verified source witness"
    if status == "excluded":
        return f"{label} has an exact-verified separating direction"
    return f"{label} remains unresolved by the exact source checker"


def _class_result(name: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [item["status"] for item in candidates]
    if any(status == "compatible" for status in statuses):
        status = "compatible"
        selected = next(item["candidate"] for item in candidates if item["status"] == status)
        reason = None
    elif candidates and all(status == "excluded" for status in statuses):
        status = "excluded"
        selected = None
        reason = None
    else:
        status = "unresolved"
        selected = None
        reason = "candidate_union_not_decided"
    return {
        "class_name": name,
        "status": status,
        "description": _class_description(name, status),
        "reason": reason,
        "selected_candidate": selected,
        "declared_candidate_count": len(candidates),
        "evaluated_candidate_count": sum(item.get("status") != "not_evaluated" for item in candidates),
        "not_evaluated_candidate_count": sum(item.get("status") == "not_evaluated" for item in candidates),
        "candidates": candidates,
    }


def _not_evaluated_candidate(label: str, moments, radii, loading, lower, upper,
                              reason: str) -> dict[str, Any]:
    return {
        "candidate": label, "status": "not_evaluated",
        "description": f"{label} was not evaluated by lazy execution",
        "reason": reason,
        "search": {"status": "not_evaluated", "reason": reason},
        "certificate": {"status": "not_evaluated", "threshold_exact": "1"},
        "moments_exact": _exact(moments), "radii_exact": _exact(radii),
        "loading_exact": _exact(loading), "lower_exact": _exact(lower),
        "upper_exact": _exact(upper), "solver_calls": 0,
        "elapsed_seconds": 0.0,
    }


def _class_description(name: str, status: str) -> str:
    if status == "compatible":
        return f"{name} has at least one exact-verified compatible candidate"
    if status == "excluded":
        return f"all finite {name} candidates are exactly excluded"
    return f"{name} remains unresolved because its finite candidate union is not decided"


def _verified_embedding(
    *,
    candidate: dict[str, Any],
    moments: list[Fraction],
    radii: list[Fraction],
    loading: list[list[Fraction]],
    lower: list[Fraction],
    upper: list[Fraction],
    witness: list[Fraction],
    origin: str,
) -> bool:
    """Replace a mixed search result with a separately reverified witness."""
    try:
        certificate = raw_box_certificate(
            moments, radii, loading, lower, upper, primal=witness, threshold=1
        )
    except MemoryError:
        candidate["compatibility_embedding"] = {
            "status": "unresolved",
            "reason": "embedding_checker_memory_error",
            "origin": origin,
        }
        raise
    if certificate.get("status") != "compatible":
        candidate["compatibility_embedding"] = {
            "status": certificate.get("status", "unresolved"),
            "reason": "embedding_reverification_failed",
            "origin": origin,
        }
        raise CertifiedSourceBackendError(
            f"exact {origin} witness failed mixed-source re-verification"
        )
    # Preserve the independent mixed LP attempt and its diagnostics.  The
    # canonical certificate is the only evidence used for the public status.
    candidate["search_certificate"] = candidate.get("certificate")
    candidate["canonical_certificate"] = certificate
    candidate["certificate"] = certificate
    candidate["status"] = "compatible"
    candidate["reason"] = None
    candidate["description"] = _candidate_description(candidate["candidate"], "compatible")
    candidate["compatibility_origin"] = origin
    return True


def diagnose_certified_sources(
    moments: Any,
    radii: Any,
    latent_loading: Any,
    latent_bounds: Any,
    mechanism_loadings: Mapping[str, Any],
    mechanism_bounds: Mapping[str, Any] | Any,
    *,
    solver_options: Mapping[str, Any] | None = None,
    execution_mode: str = "exhaustive",
) -> dict[str, Any]:
    """Diagnose exact supplied source boxes using LP candidate searches.

    All affirmative statuses are recomputed from the original exact inputs by
    :func:`raw_box_certificate`.  A candidate union is compatible when one
    member has a verified witness, excluded only when every member has a
    verified separator, and unresolved otherwise.

    ``execution_mode="exhaustive"`` retains every candidate search. ``"lazy"``
    stops a union after an exact witness and rechecks subset witnesses in the
    target source box. Unsearched candidates retain ``not_evaluated`` status;
    they cannot establish exclusion. Counts report actual searches separately
    from the full declared catalogue, and embedding failures remain visible.
    """
    if execution_mode not in ("exhaustive", "lazy"):
        raise CertifiedSourceBackendError("execution_mode must be exhaustive or lazy")
    started = time.monotonic()
    v = _vector(moments, name="moments", allow_empty=False)
    r = _vector(radii, name="radii")
    if len(r) != len(v):
        raise CertifiedSourceBackendError("moments and radii must have the same dimension")
    if any(value < 0 for value in r):
        raise CertifiedSourceBackendError("radii must be nonnegative")
    rows = len(v)
    latent = _matrix(latent_loading, name="latent_loading", rows=rows)
    latent_dimension = len(latent[0]) if latent else 0
    latent_lower, latent_upper = _box(
        latent_bounds, name="latent", dimension=latent_dimension
    )
    if not isinstance(mechanism_loadings, Mapping) or not mechanism_loadings:
        raise CertifiedSourceBackendError(
            "mechanism_loadings must be a nonempty candidate mapping"
        )
    if isinstance(mechanism_bounds, Mapping):
        bounds_for = mechanism_bounds
        missing = [name for name in mechanism_loadings if name not in bounds_for]
        if missing:
            raise CertifiedSourceBackendError(
                f"missing mechanism bounds for candidates {missing!r}"
            )
    else:
        bounds_for = None

    mechanisms: dict[str, tuple[list[list[Fraction]], list[Fraction], list[Fraction]]] = {}
    for candidate_name, loading_value in mechanism_loadings.items():
        if not isinstance(candidate_name, str) or not candidate_name:
            raise CertifiedSourceBackendError(
                "mechanism candidate names must be nonempty strings"
            )
        mechanism = _matrix(
            loading_value,
            name=f"mechanism_loadings[{candidate_name!r}]",
            rows=rows,
        )
        dimension = len(mechanism[0]) if mechanism else 0
        bound_value = bounds_for[candidate_name] if bounds_for is not None else mechanism_bounds
        lower, upper = _box(
            bound_value,
            name=f"mechanism {candidate_name!r}",
            dimension=dimension,
        )
        mechanisms[candidate_name] = (mechanism, lower, upper)

    solver_calls = 0
    embedding_failures: list[dict[str, Any]] = []
    mechanism_specs = list(mechanisms.values())
    zero_loading = [[] for _ in range(rows)]
    zero_lower: list[Fraction] = []
    zero_upper: list[Fraction] = []
    c0_primal = raw_box_certificate(
        v, r, zero_loading, zero_lower, zero_upper, primal=[], threshold=1
    )
    if c0_primal["status"] != "compatible":
        coordinate = next(
            index for index, (moment, radius) in enumerate(zip(v, r))
            if abs(moment) > radius
        )
        direction = [Fraction(0)] * rows
        direction[coordinate] = Fraction(1) if v[coordinate] > 0 else Fraction(-1)
        c0 = raw_box_certificate(
            v,
            r,
            zero_loading,
            zero_lower,
            zero_upper,
            dual=direction,
            threshold=1,
        )
        c0_reason = "coordinate_separating_direction"
    else:
        c0 = c0_primal
        c0_reason = "exact_empty_loading_zero_witness"
    c0_item = {
        "class_name": "C0",
        "status": c0["status"],
        "description": _class_description("C0", c0["status"]),
        "reason": c0_reason,
        "certificate": c0,
        "moments_exact": _exact(v),
        "radii_exact": _exact(r),
    }

    mechanism_candidates: list[dict[str, Any]] = []
    mixed_candidates: list[dict[str, Any]] = []
    mixed_specs: list[tuple[list[list[Fraction]], list[Fraction], list[Fraction]]] = []
    for name, (mechanism, mechanism_lower, mechanism_upper) in mechanisms.items():
        mixed_loading = [latent_row + mechanism_row for latent_row, mechanism_row in zip(latent, mechanism)]
        mixed_specs.append((mixed_loading, latent_lower + mechanism_lower, latent_upper + mechanism_upper))

    if execution_mode == "exhaustive":
        latent_candidate, calls = _candidate(label="latent", moments=v, radii=r, loading=latent,
                                             lower=latent_lower, upper=latent_upper,
                                             solver_options=solver_options)
        solver_calls += calls
        for name, (mechanism, mechanism_lower, mechanism_upper) in mechanisms.items():
            pure, calls = _candidate(label=name, moments=v, radii=r, loading=mechanism,
                                     lower=mechanism_lower, upper=mechanism_upper,
                                     solver_options=solver_options)
            solver_calls += calls
            mechanism_candidates.append(pure)
            mixed_loading, mixed_lower, mixed_upper = mixed_specs[len(mechanism_candidates) - 1]
            mixed, calls = _candidate(label=name, moments=v, radii=r, loading=mixed_loading,
                                      lower=mixed_lower, upper=mixed_upper,
                                      solver_options=solver_options)
            solver_calls += calls
            mixed_candidates.append(mixed)
    elif c0["status"] == "compatible":
        latent_candidate = _not_evaluated_candidate(
            "latent", v, r, latent, latent_lower, latent_upper,
            "zero witness inherited from C0")
        _verified_embedding(candidate=latent_candidate, moments=v, radii=r, loading=latent,
                            lower=latent_lower, upper=latent_upper,
                            witness=[Fraction(0)] * latent_dimension, origin="C0")
        for index, (name, (mechanism, mechanism_lower, mechanism_upper)) in enumerate(mechanisms.items()):
            pure = _not_evaluated_candidate(name, v, r, mechanism, mechanism_lower, mechanism_upper,
                                             "zero witness inherited from C0")
            if index == 0:
                _verified_embedding(candidate=pure, moments=v, radii=r, loading=mechanism,
                                    lower=mechanism_lower, upper=mechanism_upper,
                                    witness=[Fraction(0)] * len(mechanism_lower), origin="C0")
            else:
                pure["reason"] = "pure union already has a compatible C0 witness"
            mechanism_candidates.append(pure)
            mixed_loading, mixed_lower, mixed_upper = mixed_specs[len(mechanism_candidates) - 1]
            mixed = _not_evaluated_candidate(name, v, r, mixed_loading, mixed_lower, mixed_upper,
                                              "zero witness inherited from C0")
            if index == 0:
                _verified_embedding(candidate=mixed, moments=v, radii=r, loading=mixed_loading,
                                    lower=mixed_lower, upper=mixed_upper,
                                    witness=[Fraction(0)] * len(mixed_lower), origin="C0")
            else:
                mixed["reason"] = "mixed union already has a compatible C0 witness"
            mixed_candidates.append(mixed)
    else:
        # Lazy mode evaluates pure candidates until a compatible witness is
        # found; all candidates are required before declaring a pure union
        # excluded.  Mixed searches are skipped only after an exact subset
        # witness has been independently reverified in the mixed box.
        latent_candidate, calls = _candidate(label="latent", moments=v, radii=r, loading=latent,
                                             lower=latent_lower, upper=latent_upper,
                                             solver_options=solver_options)
        solver_calls += calls
        pure_witness = None
        pure_witness_index = None
        for index, (name, (mechanism, mechanism_lower, mechanism_upper)) in enumerate(mechanisms.items()):
            if pure_witness is not None:
                pure = _not_evaluated_candidate(name, v, r, mechanism, mechanism_lower, mechanism_upper,
                                                 "pure union already has a compatible witness")
            else:
                pure, calls = _candidate(label=name, moments=v, radii=r, loading=mechanism,
                                         lower=mechanism_lower, upper=mechanism_upper,
                                         solver_options=solver_options)
                solver_calls += calls
                if pure["status"] == "compatible":
                    pure_witness = pure
                    pure_witness_index = index
            mechanism_candidates.append(pure)
        if latent_candidate["status"] == "compatible":
            witness = [_fraction(value, name="latent witness")
                       for value in latent_candidate["certificate"]["primal_exact"]]
            origin = "CL"
        elif pure_witness is not None:
            witness = [_fraction(value, name="mechanism witness")
                       for value in pure_witness["certificate"]["primal_exact"]]
            origin = "CM"
        else:
            witness = None
            origin = None
        if witness is not None:
            target_index = 0 if origin == "CL" else pure_witness_index
            embedding_failed = False
            for index, (name, (mechanism, mechanism_lower, mechanism_upper)) in enumerate(mechanisms.items()):
                mixed_loading, mixed_lower, mixed_upper = mixed_specs[index]
                if index != target_index:
                    mixed = _not_evaluated_candidate(name, v, r, mixed_loading, mixed_lower, mixed_upper,
                                                      f"mixed union already has a compatible {origin} witness")
                else:
                    mixed = _not_evaluated_candidate(name, v, r, mixed_loading, mixed_lower, mixed_upper,
                                                      f"mixed union witness inherited from {origin}")
                    if origin == "CL":
                        mixed_witness = witness + [Fraction(0)] * len(mechanism_lower)
                    else:
                        mixed_witness = [Fraction(0)] * latent_dimension + witness
                    try:
                        _verified_embedding(candidate=mixed, moments=v, radii=r, loading=mixed_loading,
                                            lower=mixed_lower, upper=mixed_upper,
                                            witness=mixed_witness, origin=origin)
                    except (CertifiedSourceBackendError, MemoryError) as error:
                        embedding_failures.append({
                            "candidate": name, "origin": origin,
                            "reason": f"exact_embedding_reverification_failed:{type(error).__name__}",
                            "status": mixed.get("status"),
                            "compatibility_embedding": mixed.get("compatibility_embedding"),
                        })
                        embedding_failed = True
                        break
                mixed_candidates.append(mixed)
            if embedding_failed:
                mixed_candidates = []
        if witness is None or embedding_failed:
            for index, (name, (mechanism, mechanism_lower, mechanism_upper)) in enumerate(mechanisms.items()):
                mixed_loading, mixed_lower, mixed_upper = mixed_specs[index]
                mixed, calls = _candidate(label=name, moments=v, radii=r, loading=mixed_loading,
                                          lower=mixed_lower, upper=mixed_upper,
                                          solver_options=solver_options)
                solver_calls += calls
                mixed_candidates.append(mixed)
                if mixed["status"] == "compatible":
                    for rest_index, (rest_name, (rest_mechanism, rest_lower, rest_upper)) in enumerate(
                        list(mechanisms.items())[index + 1:], start=index + 1
                    ):
                        rest_loading, rest_mixed_lower, rest_mixed_upper = mixed_specs[rest_index]
                        mixed_candidates.append(_not_evaluated_candidate(
                            rest_name, v, r, rest_loading, rest_mixed_lower, rest_mixed_upper,
                            "mixed union already has a compatible witness"))
                    break

    # Canonicalize superset compatibility from exact subset witnesses.  A
    # failed mixed LP cannot invalidate a witness already verified in a
    # containing pure family; conversely, no pure lower bound is inherited.
    if execution_mode == "exhaustive" and c0["status"] == "compatible":
        zero_latent = [Fraction(0)] * latent_dimension
        for candidate in (latent_candidate,):
            _verified_embedding(
                candidate=candidate, moments=v, radii=r, loading=latent,
                lower=latent_lower, upper=latent_upper, witness=zero_latent,
                origin="C0",
            )
        for candidate, (mechanism, lower, upper) in zip(mechanism_candidates, mechanism_specs):
            _verified_embedding(
                candidate=candidate, moments=v, radii=r, loading=mechanism,
                lower=lower, upper=upper, witness=[Fraction(0)] * len(lower),
                origin="C0",
            )
        for candidate, (mixed_loading, mixed_lower, mixed_upper), (_, mechanism_lower, _) in zip(
            mixed_candidates, mixed_specs, mechanism_specs
        ):
            _verified_embedding(
                candidate=candidate, moments=v, radii=r, loading=mixed_loading,
                lower=mixed_lower,
                upper=mixed_upper,
                witness=zero_latent + [Fraction(0)] * len(mechanism_lower),
                origin="C0",
            )
    elif execution_mode == "exhaustive":
        if latent_candidate["status"] == "compatible":
            latent_witness = [
                _fraction(value, name="latent witness")
                for value in latent_candidate["certificate"]["primal_exact"]
            ]
            for candidate, (mixed_loading, mixed_lower, mixed_upper), (_, mechanism_lower, _) in zip(
                mixed_candidates, mixed_specs, mechanism_specs
            ):
                _verified_embedding(
                    candidate=candidate, moments=v, radii=r, loading=mixed_loading,
                    lower=mixed_lower,
                    upper=mixed_upper,
                    witness=latent_witness + [Fraction(0)] * len(mechanism_lower),
                    origin="CL",
                )
        for index, (candidate, (mixed_loading, mixed_lower, mixed_upper), (_, mechanism_lower, _)) in enumerate(zip(
            mixed_candidates, mixed_specs, mechanism_specs
        )):
            if candidate.get("compatibility_origin") == "CL":
                continue
            pure = mechanism_candidates[index]
            if pure["status"] != "compatible":
                continue
            mechanism_witness = [
                _fraction(value, name="mechanism witness")
                for value in pure["certificate"]["primal_exact"]
            ]
            _verified_embedding(
                candidate=candidate, moments=v, radii=r, loading=mixed_loading,
                lower=mixed_lower,
                upper=mixed_upper,
                witness=[Fraction(0)] * latent_dimension + mechanism_witness,
                origin="CM",
            )

    cl = _class_result("CL", [latent_candidate])
    cm = _class_result("CM", mechanism_candidates)
    cmix = _class_result("CMIX", mixed_candidates)
    elapsed = time.monotonic() - started
    if not isfinite(elapsed):
        elapsed = 0.0
    catalogue = []
    for class_name, class_result in (("CL", cl), ("CM", cm), ("CMIX", cmix)):
        for candidate in class_result["candidates"]:
            catalogue.append({
                "class_name": class_name,
                "candidate": candidate.get("candidate"),
                "status": candidate.get("status"),
                "reason": candidate.get("reason"),
            })
    return {
        "classes": {"C0": c0_item, "CL": cl, "CM": cm, "CMIX": cmix},
        "execution_mode": execution_mode,
        "candidate_catalogue": catalogue,
        "declared_candidate_count": len(catalogue),
        "evaluated_candidate_count": sum(item["status"] != "not_evaluated" for item in catalogue),
        "not_evaluated_candidate_count": sum(item["status"] == "not_evaluated" for item in catalogue),
        "embedding_failures": embedding_failures,
        "moments_exact": _exact(v),
        "radii_exact": _exact(r),
        "latent_loading_exact": _exact(latent),
        "latent_lower_exact": _exact(latent_lower),
        "latent_upper_exact": _exact(latent_upper),
        "mechanism_names": list(mechanisms),
        "solver_calls": solver_calls,
        "elapsed_seconds": elapsed,
        "threshold_exact": "1",
        "numerical_certification": (
            "exact rational certification of the supplied represented source region only; "
            "upstream statistical construction and continuation values are outside scope"
        ),
    }


__all__ = ["CertifiedSourceBackendError", "diagnose_certified_sources"]
