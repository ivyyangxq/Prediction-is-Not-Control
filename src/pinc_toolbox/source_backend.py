"""Bounded linear source-image inversion from public moments.

This module is deliberately independent of the research ``Tool/Detect``
directory.  It accepts a public moment vector, strictly positive coordinate
radii, a latent loading matrix, finitely many mechanism loading matrices, and
finite parameter boxes.  Every LP call returns both a feasible primal witness
and a dual lower certificate from that same solver result.

The reported distance for ``C0`` is exact.  LP distances are reported as a
feasible ``upper_bound`` and, when a finite dual certificate is available, a
``lower_bound``.  A pure or mixed class is compatible only when a feasible
witness is within the supplied threshold; it is excluded only when every
candidate image has a valid lower bound strictly above that threshold.  The
interval between those decisions is ``unresolved``.  This is numerical
source-image evidence, not a statistical coverage or an action-error claim.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linprog


class SourceBackendError(ValueError):
    """Raised when the public source specification is malformed."""


@dataclass(frozen=True)
class BoxBounds:
    """A finite coordinatewise box ``lower <= x <= upper``."""

    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def from_value(cls, value: Any, *, dimension: int, label: str) -> "BoxBounds":
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise SourceBackendError(f"{label} must be a (lower, upper) pair")
        lower = np.asarray(value[0], dtype=float)
        upper = np.asarray(value[1], dtype=float)
        expected = (dimension,)
        if lower.shape != expected or upper.shape != expected:
            raise SourceBackendError(
                f"{label} bounds must have shape {expected}; got {lower.shape} and {upper.shape}"
            )
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise SourceBackendError(f"{label} bounds must be finite")
        if np.any(lower > upper):
            raise SourceBackendError(f"{label} lower bounds must not exceed upper bounds")
        return cls(lower=lower.copy(), upper=upper.copy())

    def contains_zero(self, *, tolerance: float = 0.0) -> bool:
        return bool(np.all(self.lower <= tolerance) and np.all(self.upper >= -tolerance))


def _as_vector(value: Any, *, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.ndim != 1 or not np.isfinite(out).all():
        raise SourceBackendError(f"{name} must be a finite one-dimensional array")
    return out


def _as_loading(value: Any, *, rows: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.ndim != 2 or out.shape[0] != rows or not np.isfinite(out).all():
        raise SourceBackendError(
            f"{name} must be a finite matrix with {rows} rows; got shape {out.shape}"
        )
    return out


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _json_array(value: Any) -> list[float] | None:
    arr = np.asarray(value, dtype=float)
    return arr.tolist() if np.isfinite(arr).all() else None


def _box_support(coefficient: np.ndarray, bounds: BoxBounds) -> float:
    """Support function of a finite, possibly asymmetric box."""
    coefficient = np.asarray(coefficient, dtype=float)
    if coefficient.shape != bounds.lower.shape:
        raise SourceBackendError("box support coefficient has the wrong dimension")
    if coefficient.size == 0:
        return 0.0
    return float(
        np.maximum(coefficient * bounds.lower, coefficient * bounds.upper).sum()
    )


def _finite_max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def _lp_evidence(
    target: np.ndarray,
    latent_loading: np.ndarray,
    latent_bounds: BoxBounds,
    mechanism_loading: np.ndarray | None,
    mechanism_bounds: BoxBounds | None,
    *,
    tolerance: float,
    solver_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Solve one bounded Chebyshev LP and retain primal and dual evidence."""
    if mechanism_loading is None:
        mechanism_loading = np.empty((target.size, 0), dtype=float)
    if mechanism_bounds is None:
        mechanism_bounds = BoxBounds(np.empty(0), np.empty(0))
    if latent_loading.shape[0] != target.size or mechanism_loading.shape[0] != target.size:
        raise SourceBackendError("source loading rows must match the moment dimension")

    n_latent = latent_loading.shape[1]
    n_mechanism = mechanism_loading.shape[1]
    loading = np.column_stack((latent_loading, mechanism_loading))
    lower = np.r_[latent_bounds.lower, mechanism_bounds.lower]
    upper = np.r_[latent_bounds.upper, mechanism_bounds.upper]
    n_source = n_latent + n_mechanism

    # Put the positive residual block first and the negative residual block
    # second.  With SciPy's <= marginal convention, lambda = y_positive-y_negative
    # has the sign used by target'lambda - support(loading'lambda).
    a_positive = np.column_stack((loading, -np.ones(target.size)))
    a_negative = np.column_stack((-loading, -np.ones(target.size)))
    a_ub = np.vstack((a_positive, a_negative))
    b_ub = np.r_[target, -target]
    objective = np.zeros(n_source + 1)
    objective[-1] = 1.0
    variable_bounds = list(zip(lower.tolist(), upper.tolist())) + [(0.0, None)]

    try:
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=variable_bounds,
            method="highs",
            options=dict(solver_options or {}),
        )
    except (FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
        # Known numerical failures remain evidence, not exclusions.  Input or
        # programming errors from linprog intentionally propagate.
        return {
            "status": "unavailable",
            "reason": f"solver_exception:{type(exc).__name__}",
            "solver_message": str(exc),
            "success": False,
            "upper_bound": None,
            "lower_bound": None,
            "upper_kind": "upper_bound",
            "lower_kind": "lower_bound",
            "primal": None,
            "dual": None,
        }

    raw_x = None if result.x is None else np.asarray(result.x, dtype=float)
    if not bool(result.success) or raw_x is None or raw_x.shape != (n_source + 1,) or not np.isfinite(raw_x).all():
        return {
            "status": "unavailable",
            "reason": "solver_failure",
            "solver_message": str(result.message),
            "success": False,
            "upper_bound": None,
            "lower_bound": None,
            "upper_kind": "upper_bound",
            "lower_kind": "lower_bound",
            "primal": None,
            "dual": None,
        }

    # Recompute the witness after a conservative projection.  Thus a tiny
    # solver-side bound violation cannot turn an infeasible point into a claim.
    raw_source = raw_x[:-1]
    witness_source = np.clip(raw_source, lower, upper)
    predicted = loading @ witness_source
    upper_bound = _finite_max_abs(target - predicted)
    raw_t = float(raw_x[-1])
    primal_violation = max(
        0.0,
        float(np.max(lower - raw_source)) if raw_source.size else 0.0,
        float(np.max(raw_source - upper)) if raw_source.size else 0.0,
        -raw_t,
    )
    primal = {
        "source": _json_array(witness_source),
        "raw_source": _json_array(raw_source),
        "solver_objective": _json_float(raw_t),
        "recomputed_residual": _json_float(upper_bound),
        "bound_violation": _json_float(primal_violation),
        "projected": bool(not np.array_equal(raw_source, witness_source)),
    }
    if not np.isfinite(predicted).all() or not np.isfinite(target - predicted).all():
        return {
            "status": "unavailable",
            "reason": "nonfinite_primal_reconstruction",
            "solver_message": str(result.message),
            "success": True,
            "upper_bound": None,
            "lower_bound": None,
            "upper_kind": "upper_bound",
            "lower_kind": "lower_bound",
            "primal": primal,
            "dual": None,
        }

    dual = None
    lower_bound = None
    if hasattr(result, "ineqlin") and result.ineqlin is not None:
        marginals = np.asarray(result.ineqlin.marginals, dtype=float)
        if marginals.shape == (2 * target.size,) and np.isfinite(marginals).all():
            raw_lambda = marginals[: target.size] - marginals[target.size:]
            raw_l1 = float(np.sum(np.abs(raw_lambda)))
            scale = max(1.0, raw_l1) * (1.0 + 4.0 * np.finfo(float).eps)
            lam = raw_lambda / scale
            normalized_l1 = float(np.sum(np.abs(lam)))
            if np.isfinite(lam).all() and normalized_l1 <= 1.0:
                source_coefficient = loading.T @ lam
                if np.isfinite(source_coefficient).all():
                    support = _box_support(source_coefficient, BoxBounds(lower, upper))
                    raw_lower = float(lam @ target - support)
                    if np.isfinite(support) and np.isfinite(raw_lower):
                        lower_bound = max(0.0, raw_lower)
                        if lower_bound > upper_bound + tolerance:
                            lower_bound = None
                        dual = {
                            "lambda_raw": _json_array(raw_lambda),
                            "lambda_normalized": _json_array(lam),
                            "lambda_l1": _json_float(normalized_l1),
                            "raw_lower_bound": _json_float(raw_lower),
                            "support_value": _json_float(support),
                            "lower_bound": _json_float(lower_bound),
                            "normalization": "divide by max(1, raw lambda l1) and (1+4 machine eps)",
                        }

    return {
        "status": "complete" if lower_bound is not None else "unavailable",
        "reason": None if lower_bound is not None else "dual_unavailable",
        "solver_message": str(result.message),
        "success": True,
        "upper_bound": _json_float(upper_bound),
        "lower_bound": _json_float(lower_bound),
        "upper_kind": "upper_bound",
        "lower_kind": "lower_bound",
        "primal": primal,
        "dual": dual,
        "primal_feasibility_violation": _json_float(primal_violation),
    }


def _classify(
    evidence: dict[str, Any], *, threshold: float, tolerance: float
) -> str:
    upper = evidence.get("upper_bound")
    lower = evidence.get("lower_bound")
    if upper is not None and upper <= threshold + tolerance:
        return "compatible"
    if lower is not None and lower > threshold + tolerance:
        return "excluded"
    if evidence.get("status") == "unavailable":
        return "unavailable"
    return "unresolved"


def _description(name: str, status: str, candidate: str | None = None) -> str:
    suffix = "" if candidate is None else f" candidate {candidate!r}"
    if status == "compatible":
        return f"{name}{suffix} has a feasible public source witness"
    if status == "excluded":
        return f"{name}{suffix} is separated by a valid dual lower bound"
    if status == "unavailable":
        return f"{name}{suffix} has no usable numerical certificate"
    return f"{name}{suffix} remains between its saved upper and lower bounds"


def _candidate_class(
    name: str,
    candidates: Mapping[str, dict[str, Any]],
    *,
    threshold: float,
    tolerance: float,
) -> dict[str, Any]:
    records = []
    for candidate, evidence in candidates.items():
        status = _classify(evidence, threshold=threshold, tolerance=tolerance)
        item = dict(evidence)
        item.update(candidate=candidate, status=status, description=_description(name, status, candidate))
        records.append(item)

    compatible = next((item for item in records if item["status"] == "compatible"), None)
    all_excluded = bool(records) and all(item["status"] == "excluded" for item in records)
    if compatible is not None:
        status = "compatible"
        witness = compatible.get("primal")
        upper = compatible.get("upper_bound")
        lower = min(
            item["lower_bound"] for item in records if item.get("lower_bound") is not None
        ) if all(item.get("lower_bound") is not None for item in records) else None
        chosen = compatible["candidate"]
    elif all_excluded:
        status = "excluded"
        witness = None
        upper_values = [item["upper_bound"] for item in records if item.get("upper_bound") is not None]
        upper = min(upper_values) if upper_values else None
        lower = min(item["lower_bound"] for item in records)
        chosen = None
    elif any(item["status"] == "unavailable" for item in records):
        status, witness, upper, lower, chosen = "unavailable", None, None, None, None
    else:
        status, witness, chosen = "unresolved", None, None
        upper_values = [item["upper_bound"] for item in records if item.get("upper_bound") is not None]
        lower_values = [item["lower_bound"] for item in records if item.get("lower_bound") is not None]
        upper = min(upper_values) if upper_values else None
        lower = min(lower_values) if len(lower_values) == len(records) else None
    return {
        "status": status,
        "description": _description(name, status, chosen),
        "provenance": "pinc_toolbox.source_backend",
        "reason": None if status in {"compatible", "excluded"} else "candidate_union_not_decided",
        "distance_upper": upper,
        "distance_lower": lower,
        "distance_upper_kind": "upper_bound",
        "distance_lower_kind": "lower_bound",
        "candidate": chosen,
        "witness": witness,
        "candidates": records,
    }


def _inherit(parent: dict[str, Any], child: str, *, reason: str) -> dict[str, Any]:
    out = dict(parent)
    out["description"] = f"{child} inherits the feasible witness from {parent.get('description', 'a pure source')}"
    out["inherited_from"] = parent.get("class_name")
    out["inheritance_reason"] = reason
    out["upper_bound_from_subset"] = True
    out["reason"] = None
    upstream = {
        "source_class": parent.get("class_name"),
        "distance_lower": parent.get("distance_lower", parent.get("lower_bound")),
        "distance_lower_kind": parent.get("distance_lower_kind", parent.get("lower_kind")),
        "dual": parent.get("dual"),
        "candidates": parent.get("candidates"),
    }
    if any(value is not None for value in upstream.values()):
        out["upstream_evidence"] = upstream
    out.pop("dual", None)
    out.pop("candidates", None)
    # A feasible witness embeds into a superset, but the parent's lower bound
    # does not.  Keep only an upper-bound witness for the inherited class.
    parent_upper = parent.get("distance_upper", parent.get("distance"))
    out["distance"] = parent_upper
    out["distance_kind"] = "upper_bound"
    out["distance_upper"] = parent_upper
    out["distance_upper_kind"] = "upper_bound"
    out["distance_lower"] = None
    out["distance_lower_kind"] = "lower_bound"
    out.pop("lower_bound", None)
    out.pop("lower_kind", None)
    return out


def diagnose_linear_sources(
    moments: Any,
    radii: Any,
    latent_loading: Any,
    latent_bounds: Any,
    mechanism_loadings: Mapping[str, Any],
    mechanism_bounds: Mapping[str, Any] | Any,
    *,
    threshold: float = 1.0,
    tolerance: float = 1e-8,
    solver_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Diagnose ``C0``, latent, mechanism, and mixed bounded source images.

    ``mechanism_loadings`` is a non-empty finite mapping from candidate names
    to matrices.  ``mechanism_bounds`` is either a matching mapping of
    ``(lower, upper)`` pairs or one common pair.  Candidate mechanisms are a
    finite union; a witness for one candidate retains that union, whereas
    exclusion requires a valid lower bound for every candidate.

    The function consumes only supplied public arrays.  It does not select
    eta, infer a source onset, construct a statistical region, or inspect a
    DGP/truth object.
    """
    v = _as_vector(moments, name="moments")
    r = _as_vector(radii, name="radii")
    if v.size == 0 or r.shape != v.shape or np.any(r <= 0):
        raise SourceBackendError("moments and radii must have the same non-empty shape and positive radii")
    if not np.isfinite(threshold) or threshold < 0 or not np.isfinite(tolerance) or tolerance < 0:
        raise SourceBackendError("threshold must be nonnegative and tolerance must be finite and nonnegative")
    m = v.size
    latent = _as_loading(latent_loading, rows=m, name="latent_loading")
    latent_box = BoxBounds.from_value(latent_bounds, dimension=latent.shape[1], label="latent")
    if not isinstance(mechanism_loadings, Mapping) or not mechanism_loadings:
        raise SourceBackendError("mechanism_loadings must be a non-empty finite candidate mapping")

    common_mechanism_bounds = None
    if isinstance(mechanism_bounds, (tuple, list)) and len(mechanism_bounds) == 2:
        common_mechanism_bounds = mechanism_bounds
    elif not isinstance(mechanism_bounds, Mapping):
        raise SourceBackendError("mechanism_bounds must be a candidate mapping or a common (lower, upper) pair")

    mechanisms: dict[str, tuple[np.ndarray, BoxBounds]] = {}
    for candidate, loading_value in mechanism_loadings.items():
        if not isinstance(candidate, str) or not candidate:
            raise SourceBackendError("mechanism candidate names must be non-empty strings")
        loading = _as_loading(loading_value, rows=m, name=f"mechanism_loadings[{candidate!r}]")
        bounds_value = common_mechanism_bounds
        if bounds_value is None:
            if candidate not in mechanism_bounds:
                raise SourceBackendError(f"missing bounds for mechanism candidate {candidate!r}")
            bounds_value = mechanism_bounds[candidate]
        box = BoxBounds.from_value(bounds_value, dimension=loading.shape[1], label=f"mechanism {candidate!r}")
        if not box.contains_zero():
            raise SourceBackendError(
                f"mechanism {candidate!r} box must contain zero for nested source classes"
            )
        mechanisms[candidate] = (loading, box)
    if not latent_box.contains_zero():
        raise SourceBackendError("latent box must contain zero for nested source classes")

    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            normalized_target = v / r
            normalized_latent = latent / r[:, None]
    except FloatingPointError as exc:
        raise SourceBackendError("moments or loading normalization overflowed the finite range") from exc
    if not np.isfinite(normalized_target).all() or not np.isfinite(normalized_latent).all():
        raise SourceBackendError("moments or loading normalization produced non-finite values")
    normalized_mechanisms = {}
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            for candidate, (loading, box) in mechanisms.items():
                normalized_mechanisms[candidate] = loading / r[:, None]
    except FloatingPointError as exc:
        raise SourceBackendError("mechanism loading normalization overflowed the finite range") from exc
    if any(not np.isfinite(loading).all() for loading in normalized_mechanisms.values()):
        raise SourceBackendError("mechanism loading normalization produced non-finite values")
    common = dict(
        provenance="pinc_toolbox.source_backend",
        threshold=float(threshold),
        tolerance=float(tolerance),
        normalized=True,
        candidate_names=list(mechanisms),
        statistical_coverage="not supplied by this numerical backend",
        compatibility_rule="upper_bound <= threshold + tolerance; the tolerance is a numerical boundary expansion",
        exclusion_rule="lower_bound > threshold + tolerance; all finite candidates are required",
    )
    q0 = _finite_max_abs(normalized_target)
    c0_status = "compatible" if q0 <= threshold + tolerance else "excluded"
    c0 = {
        "class_name": "C0",
        "status": c0_status,
        "description": _description("C0", c0_status),
        "provenance": common["provenance"],
        "reason": None,
        "distance": q0,
        "distance_kind": "exact",
        "distance_upper": q0,
        "distance_lower": q0,
        "distance_upper_kind": "exact",
        "distance_lower_kind": "exact",
        "witness": {"latent": [0.0] * latent.shape[1], "mechanism": None},
    }

    # The source classes are nested by construction (all boxes contain zero),
    # so a C0 witness is a valid witness for all supersets and no LP is needed.
    if c0_status == "compatible":
        classes = {"C0": c0}
        for name in ("CL", "CM", "CMIX"):
            inherited = _inherit(c0, name, reason="C0 zero witness embeds in the declared source boxes")
            inherited["class_name"] = name
            classes[name] = inherited
        return {**common, "moments": v.tolist(), "radii": r.tolist(), "classes": classes}

    latent_lp = _lp_evidence(
        normalized_target, normalized_latent, latent_box, None, None,
        tolerance=tolerance, solver_options=solver_options,
    )
    latent_status = _classify(latent_lp, threshold=threshold, tolerance=tolerance)
    latent_result = dict(latent_lp)
    latent_result.update(
        class_name="CL", status=latent_status, description=_description("CL", latent_status),
        provenance=common["provenance"], candidate=None,
        distance_upper=latent_lp.get("upper_bound"), distance_lower=latent_lp.get("lower_bound"),
        distance_upper_kind="upper_bound", distance_lower_kind="lower_bound",
        witness=latent_lp.get("primal"),
        reason=None if latent_status in {"compatible", "excluded"} else latent_lp.get("reason"),
    )

    pure_mechanisms: dict[str, dict[str, Any]] = {}
    for candidate, (loading, box) in mechanisms.items():
        pure_mechanisms[candidate] = _lp_evidence(
            normalized_target, np.empty((m, 0), dtype=float), BoxBounds(np.empty(0), np.empty(0)),
            normalized_mechanisms[candidate], box, tolerance=tolerance, solver_options=solver_options,
        )
    mechanism_result = _candidate_class("CM", pure_mechanisms, threshold=threshold, tolerance=tolerance)
    mechanism_result["class_name"] = "CM"

    mixed_evidence: dict[str, dict[str, Any]] = {}
    latent_zero = latent_box.contains_zero()
    if c0_status == "compatible" and latent_zero and any(box.contains_zero() for _, box in mechanisms.values()):
        mixed_result = _inherit(c0, "CMIX", reason="C0 zero witness embeds in both source boxes")
        mixed_result["class_name"] = "CMIX"
    elif latent_status == "compatible" and any(box.contains_zero() for _, box in mechanisms.values()):
        mixed_result = _inherit(latent_result, "CMIX", reason="mechanism zero embeds the latent witness")
        mixed_result["class_name"] = "CMIX"
    elif mechanism_result["status"] == "compatible" and latent_zero:
        mixed_result = _inherit(mechanism_result, "CMIX", reason="latent zero embeds the mechanism witness")
        mixed_result["class_name"] = "CMIX"
    else:
        for candidate, (loading, box) in mechanisms.items():
            mixed_evidence[candidate] = _lp_evidence(
                normalized_target, normalized_latent, latent_box,
                normalized_mechanisms[candidate], box, tolerance=tolerance, solver_options=solver_options,
            )
        mixed_result = _candidate_class("CMIX", mixed_evidence, threshold=threshold, tolerance=tolerance)
        mixed_result["class_name"] = "CMIX"

    classes = {"C0": c0, "CL": latent_result, "CM": mechanism_result, "CMIX": mixed_result}

    return {**common, "moments": v.tolist(), "radii": r.tolist(), "classes": classes}


def router_evidence(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a safe status adapter accepted by the existing router.

    The legacy router copies callback mappings when it applies subset
    inheritance.  Numeric lower bounds, candidate records, and witnesses are
    therefore intentionally kept in the full backend result and omitted from
    this adapter; otherwise a pure-class lower certificate could be mistaken
    for a lower certificate on the mixed superset.
    """
    classes = result.get("classes")
    if not isinstance(classes, Mapping):
        raise SourceBackendError("result does not contain source classes")
    out = {}
    for name in ("C0", "CL", "CM", "CMIX"):
        item = classes.get(name)
        if not isinstance(item, Mapping):
            raise SourceBackendError(f"result is missing class {name}")
        out[name] = {
            "status": item.get("status"),
            "description": item.get("description"),
            "provenance": item.get("provenance"),
            "reason": item.get("reason"),
            "inherited_from": item.get("inherited_from"),
            "upper_bound_from_subset": item.get("upper_bound_from_subset", False),
        }
    return out


__all__ = [
    "BoxBounds",
    "SourceBackendError",
    "diagnose_linear_sources",
    "router_evidence",
]
