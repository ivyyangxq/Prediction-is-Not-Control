"""Exact eta-domain selection and native action masking.

The domain reuses the certified selector's nearest-joint-record, scaled
L-infinity rule. It is observable support bookkeeping, not a source or Q
confidence statement.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from fractions import Fraction
from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np

from .certified_construction import _q
from .certified_reference import select_eta_certified
from .eta_monitor import _integer


_COORDINATE_NAMES = ("physical_state", "previous_action", "action")


def _json_copy(value: Any, name: str):
    import json
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be strict JSON data") from exc


def _record_ids(values: Sequence[Any]) -> list[Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("selection_record_ids must be a nonempty sequence")
    result = [_json_copy(value, "selection_record_id") for value in values]
    if not result or any(value is None or isinstance(value, (dict, list)) for value in result):
        raise ValueError("selection_record_ids must contain nonempty scalar IDs")
    if len({(type(value).__name__, repr(value)) for value in result}) != len(result):
        raise ValueError("selection_record_ids must be distinct")
    return result


def _unavailable(reason: str, *, initialization=None, selection_record_ids=None,
                 elapsed_seconds=0.0, as_of=None) -> dict[str, Any]:
    return {
        "status": "not_ready", "reason": reason, "reasons": [reason],
        "eta_exact": None, "anchors_exact": None, "scale_exact": None,
        "as_of": as_of, "eta_fit_cutoff": None, "monitoring_start": None,
        "coordinate_names": list(_COORDINATE_NAMES),
        "selection_record_ids": selection_record_ids or [],
        "selection_independence": "not_asserted_by_record_ids",
        "timing": {"elapsed_seconds": elapsed_seconds,
                   "selection_elapsed_seconds": elapsed_seconds},
        "initialization": initialization,
    }


def select_eta_domain(public_rows, reference_spec, *, as_of,
                      selection_record_ids) -> dict[str, Any]:
    """Select the certified eta domain and retain its complete provenance.

    ``selection_record_ids`` labels the explicitly supplied selection panel;
    IDs preserve provenance and do not establish independence.
    """
    ids = _record_ids(selection_record_ids)
    decision_clock = _integer(as_of, "as_of")
    started = perf_counter()
    try:
        initialization = select_eta_certified(public_rows, reference_spec, as_of=decision_clock)
    except (MemoryError, OverflowError) as exc:
        elapsed = perf_counter() - started
        return _unavailable(type(exc).__name__, selection_record_ids=ids,
                            elapsed_seconds=elapsed, as_of=decision_clock)
    elapsed = perf_counter() - started
    if initialization.get("status") != "ready":
        reasons = initialization.get("reasons") or ["certified_eta_not_ready"]
        result = _unavailable("certified_eta_not_ready", initialization=_json_copy(
            initialization, "initialization"), selection_record_ids=ids,
            elapsed_seconds=elapsed, as_of=decision_clock)
        result["reasons"] = _json_copy(reasons, "initialization.reasons")
        return result
    eta = initialization.get("selected_eta_exact", initialization.get("eta_exact"))
    anchors = initialization.get("anchors_exact")
    scale = initialization.get("scale_exact")
    try:
        if eta is None or not isinstance(anchors, list) or not anchors:
            raise ValueError("certified selector omitted eta or anchors")
        if not isinstance(scale, list) or not scale:
            raise ValueError("certified selector omitted scale")
        exact_anchors = [[str(_q(value)) for value in row] for row in anchors]
        exact_scale = [str(_q(value)) for value in scale]
        exact_eta = str(_q(eta))
        if len(exact_scale) != 3 or any(len(row) != 3 for row in exact_anchors):
            raise ValueError("certified support coordinates must be [physical_state, previous_action, action]")
        if any(_q(value) <= 0 for value in exact_scale):
            raise ValueError("certified scales must be positive")
    except (TypeError, ValueError, KeyError) as exc:
        return _unavailable("invalid_certified_domain:" + str(exc),
                            initialization=_json_copy(initialization, "initialization"),
                            selection_record_ids=ids, elapsed_seconds=elapsed,
                            as_of=decision_clock)
    return {
        "status": "ready", "reason": None, "reasons": [],
        "eta_exact": exact_eta, "anchors_exact": exact_anchors,
        "scale_exact": exact_scale,
        "as_of": decision_clock,
        "eta_fit_cutoff": initialization.get("eta_fit_cutoff"),
        "monitoring_start": initialization.get("monitoring_start"),
        "coordinate_names": list(_COORDINATE_NAMES),
        "distance_rule": "nearest_joint_record_scaled_Linf",
        "selection_record_ids": ids,
        "selection_independence": "not_asserted_by_record_ids",
        "timing": {"elapsed_seconds": elapsed,
                   "selection_elapsed_seconds": elapsed},
        "initialization": _json_copy(initialization, "initialization"),
    }


def _domain_geometry(domain):
    if not isinstance(domain, Mapping) or domain.get("status") != "ready":
        raise ValueError("eta_domain_not_ready")
    anchors = domain.get("anchors_exact")
    scale = domain.get("scale_exact")
    eta = domain.get("eta_exact")
    if not isinstance(anchors, list) or not anchors or not isinstance(scale, list) or len(scale) != 3:
        raise ValueError("eta_domain_geometry_missing")
    exact_anchors = []
    for row in anchors:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError("eta_domain_anchor_dimension_mismatch")
        exact_anchors.append(tuple(_q(value) for value in row))
    exact_scale = tuple(_q(value) for value in scale)
    if any(value <= 0 for value in exact_scale):
        raise ValueError("eta_domain_scale_not_positive")
    exact_eta = _q(eta)
    if exact_eta < 0:
        raise ValueError("eta_domain_eta_negative")
    return exact_anchors, exact_scale, exact_eta


def _point_distance(point, anchors, scale):
    distances = [max(abs(value - anchor[j]) / scale[j] for j, value in enumerate(point))
                 for anchor in anchors]
    index = min(range(len(distances)), key=distances.__getitem__)
    return distances[index], index


def domain_mask(domain, points) -> dict[str, Any]:
    """Evaluate exact eta support for points ``[x, previous_action, action]``."""
    if isinstance(points, (str, bytes)):
        raise ValueError("points must be a sequence of three-coordinate points")
    try:
        point_rows = list(points)
    except TypeError as exc:
        raise ValueError("points must be a sequence of three-coordinate points") from exc
    try:
        anchors, scale, eta = _domain_geometry(domain)
    except (TypeError, ValueError, KeyError) as exc:
        reason = str(exc)
        return {"status": "unavailable", "mask": [None] * len(point_rows),
                "rows": [{"status": "unavailable", "supported": None, "reason": reason}
                         for _ in point_rows], "reason": reason,
                "distance_rule": "nearest_joint_record_scaled_Linf"}
    mask, rows = [], []
    for index, raw in enumerate(point_rows):
        try:
            if not isinstance(raw, (list, tuple)) or len(raw) != 3:
                raise ValueError("point_must_have_three_coordinates")
            point = tuple(_q(value) for value in raw)
            distance, anchor_index = _point_distance(point, anchors, scale)
            supported = distance <= eta
            mask.append(bool(supported))
            rows.append({"status": "supported" if supported else "outside_eta",
                         "supported": bool(supported), "minimum_distance_exact": str(distance),
                         "nearest_anchor_index": anchor_index, "eta_exact": str(eta),
                         "reason": None if supported else "outside_eta_support"})
        except (TypeError, ValueError, OverflowError) as exc:
            mask.append(None)
            rows.append({"status": "unavailable", "supported": None,
                         "reason": str(exc)})
    status = "ready" if all(item is not None for item in mask) else "partial"
    return {"status": status, "mask": mask, "rows": rows, "reason": None,
            "distance_rule": "nearest_joint_record_scaled_Linf"}


def eta_qualified_native_action(policy, states, domain, *, feasible=None,
                                remaining_horizon=None, auxiliary=None) -> dict[str, Any]:
    """Query native Q values only for eta-qualified and feasible actions.

    ``states`` columns are ``[physical_state, w, previous_action]``. The
    support query uses ``[physical_state, previous_action, candidate_action]``.
    """
    try:
        state_array = np.asarray(states, dtype=float)
        if state_array.ndim != 2 or state_array.shape[1] != 3 or not np.isfinite(state_array).all():
            raise ValueError("states must be finite with shape (n, 3)")
        if state_array.shape[0] == 0:
            raise ValueError("states must contain at least one row")
        action_grid = np.asarray(policy.action_grid_, dtype=float).reshape(-1)
        if action_grid.size == 0 or not np.isfinite(action_grid).all():
            raise ValueError("policy.action_grid_ must be finite and nonempty")
    except Exception as exc:
        n = int(np.asarray(states).shape[0]) if np.asarray(states).ndim else 0
        return {"status": "unavailable", "reason": str(exc), "actions": [None] * n,
                "statuses": ["unavailable"] * n, "eta_mask": None,
                "feasible_mask": None, "native_mask": None,
                "trace": {"native_call_made": False}}
    n, m = len(state_array), len(action_grid)
    supplied = np.ones((n, m), dtype=bool)
    if feasible is not None:
        supplied = np.asarray(feasible, dtype=bool)
        if supplied.shape == (m,):
            supplied = np.broadcast_to(supplied, (n, m)).copy()
        if supplied.shape != (n, m):
            return {"status": "unavailable", "reason": "feasible_shape_mismatch",
                    "actions": [None] * n, "statuses": ["unavailable"] * n,
                    "eta_mask": None, "feasible_mask": supplied.tolist(),
                    "native_mask": None, "trace": {"native_call_made": False}}
    try:
        _domain_geometry(domain)
        query_points = [[state_array[i, 0], state_array[i, 2], action]
                        for i in range(n) for action in action_grid]
        checked = domain_mask(domain, query_points)
        if checked["status"] not in {"ready", "partial"} or any(v is None for v in checked["mask"]):
            raise ValueError(checked.get("reason") or "eta_domain_support_unavailable")
        eta = np.asarray(checked["mask"], dtype=bool).reshape(n, m)
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc), "actions": [None] * n,
                "statuses": ["domain_unavailable"] * n, "eta_mask": None,
                "feasible_mask": supplied.tolist(), "native_mask": None,
                "trace": {"native_call_made": False}}
    qualified = eta & supplied
    if getattr(policy, "auxiliary_columns_", ()) and auxiliary is None:
        return {"status": "unavailable", "reason": "auxiliary_policy_inputs_not_supplied",
                "actions": [None] * n, "statuses": ["unavailable"] * n,
                "eta_mask": eta.tolist(), "feasible_mask": supplied.tolist(),
                "qualified_mask": qualified.tolist(), "native_mask": None,
                "trace": {"native_call_made": False}}
    q_method = getattr(policy, "q_values", None)
    if not callable(q_method):
        return {"status": "unavailable", "reason": "policy_q_values_unavailable",
                "actions": [None] * n, "statuses": ["unavailable"] * n,
                "eta_mask": eta.tolist(), "feasible_mask": supplied.tolist(),
                "qualified_mask": qualified.tolist(), "native_mask": None,
                "trace": {"native_call_made": False}}
    try:
        q_values = np.asarray(q_method(state_array, auxiliary, feasible=qualified,
                                       remaining_horizon=remaining_horizon), dtype=float)
        if q_values.shape != (n, m):
            raise ValueError("policy q_values returned an unexpected shape")
        native_raw_mask = np.isfinite(q_values)
        native_mask = native_raw_mask & qualified
        permitted = native_mask
        actions, statuses = [], []
        for row in range(n):
            if not permitted[row].any():
                actions.append(None)
                statuses.append("unsupported")
                continue
            safe_q = np.where(permitted[row], q_values[row], -np.inf)
            # Match native .act: np.argmax selects the first maximizing grid cell.
            chosen = int(np.argmax(safe_q))
            actions.append(float(action_grid[chosen]))
            statuses.append("available")
    except Exception as exc:
        return {"status": "unavailable", "reason": "native_query_failed:" + str(exc),
                "actions": [None] * n, "statuses": ["unavailable"] * n,
                "eta_mask": eta.tolist(), "feasible_mask": supplied.tolist(),
                "qualified_mask": qualified.tolist(), "native_mask": None,
                "trace": {"native_call_made": True,
                          "native_method": "q_values",
                          "feasible_applied_before_call": True}}
    return {"status": "ready" if all(s == "available" for s in statuses) else "partial",
            "reason": None, "actions": actions, "statuses": statuses,
            "eta_mask": eta.tolist(), "feasible_mask": supplied.tolist(),
            "qualified_mask": qualified.tolist(), "native_mask": native_mask.tolist(),
            "trace": {"native_call_made": True, "native_method": "q_values",
                      "feasible_applied_before_call": True,
                      "eta_filter_applied_before_call": True,
                      "remaining_horizon": remaining_horizon,
                      "native_raw_mask_intersected_with_qualified": True,
                      "native_raw_mask": native_raw_mask.tolist()}}


__all__ = ["select_eta_domain", "domain_mask", "eta_qualified_native_action"]
