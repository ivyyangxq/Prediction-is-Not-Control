"""Stateless raw-history region preparation for the certified source workflow.

This module selects the already declared exact reference and asks the
certified monitor to construct declared regions without source evaluation,
then selects one requested window.  It never calls the eager source backend,
the legacy router, or a source decision callback.  A driver owns request
identity and persistence when restoring a prepared result.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from time import perf_counter
from typing import Any


def _validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ValueError("request must be a mapping")
    if request.get("operation") != "eta_monitor_certified":
        raise ValueError("operation='eta_monitor_certified' is required")
    if "release_events" in request:
        raise ValueError("raw history preparation accepts one batch request; release_events are not supported")
    for key in ("public_rows", "reference", "monitor", "as_of"):
        if key not in request:
            raise ValueError(f"request is missing {key}")
    return deepcopy(dict(request))


def _validate_window_index(window_index: Any) -> int:
    if isinstance(window_index, bool) or not isinstance(window_index, int) or window_index < 0:
        raise ValueError("window_index must be a nonnegative integer")
    return window_index


def _timing(total_started, eta_elapsed, region_elapsed, *, status, reason=None):
    return {
        "status": status,
        "reason": reason,
        "elapsed_seconds": perf_counter() - total_started,
        "eta_prepare_elapsed_seconds": eta_elapsed,
        "region_prepare_elapsed_seconds": region_elapsed,
        "source_solver_calls": 0,
        "source_evaluation_elapsed_seconds": 0.0,
    }


def _readiness(initialization: Mapping[str, Any]) -> dict[str, Any]:
    if initialization.get("status") == "ready":
        return {"status": "ready", "reason": None}
    reasons = initialization.get("reasons")
    if not reasons:
        reasons = ["no_completed_qualified_prefix"]
    return {"status": "not_ready", "reason": reasons}


def prepare_raw_history(request: Mapping[str, Any], *, window_index: int) -> dict[str, Any]:
    """Prepare one exact represented-input source region from a batch request."""
    total_started = perf_counter()
    request_copy = _validate_request(request)
    index = _validate_window_index(window_index)
    monitor_spec = request_copy["monitor"]
    if not isinstance(monitor_spec, Mapping) or "windows" not in monitor_spec:
        raise ValueError("request.monitor.windows is required")
    windows = monitor_spec["windows"]
    if not isinstance(windows, (list, tuple)) or index >= len(windows):
        raise ValueError("window_index is outside the declared monitor windows")

    # These imports are deliberately inside the preparation call.  There is
    # no package or numerical import merely from importing this workflow module.
    from .certified_reference import select_eta_certified
    from .certified_monitor import monitor_history_certified

    eta_started = perf_counter()
    try:
        initialization = select_eta_certified(
            request_copy["public_rows"],
            request_copy["reference"],
            as_of=request_copy["as_of"],
        )
    except (MemoryError, OverflowError) as error:
        return {
            "readiness": {"status": "unavailable", "reason": type(error).__name__},
            "source": None,
            "initialization": None,
            "region": None,
            "monitoring": {"status": "unavailable", "records": []},
            "window_index": index,
            "timing": _timing(total_started, perf_counter() - eta_started, 0.0,
                               status="unavailable", reason=type(error).__name__),
        }
    eta_elapsed = perf_counter() - eta_started

    readiness = _readiness(initialization)
    if initialization.get("status") != "ready":
        return {
            "readiness": readiness,
            "source": None,
            "initialization": initialization,
            "region": None,
            "monitoring": {"status": "not_ready", "records": []},
            "window_index": index,
            "timing": _timing(total_started, eta_elapsed, 0.0,
                               status="not_ready", reason=readiness["reason"]),
        }

    region_started = perf_counter()
    try:
        monitoring = monitor_history_certified(
            request_copy["public_rows"],
            initialization,
            request_copy["monitor"],
            request_copy["reference"],
            as_of=request_copy["as_of"],
            region_only=True,
        )
    except (MemoryError, OverflowError) as error:
        return {
            "readiness": {"status": "unavailable", "reason": type(error).__name__},
            "source": None,
            "initialization": initialization,
            "region": None,
            "monitoring": {"status": "unavailable", "records": []},
            "window_index": index,
            "timing": _timing(total_started, eta_elapsed, perf_counter() - region_started,
                               status="unavailable", reason=type(error).__name__),
        }
    region_elapsed = perf_counter() - region_started

    records = monitoring.get("records", [])
    selected = next((item for item in records if item.get("window_index") == index), None)
    source = None
    if selected is not None and selected.get("availability") == "available":
        candidate_source = selected.get("source_spec")
        if isinstance(candidate_source, Mapping):
            source = deepcopy(candidate_source)
            readiness = {"status": "ready", "reason": None}
        else:
            readiness = {"status": "unavailable", "reason": "missing_source_spec"}
    elif selected is not None and selected.get("availability") == "unavailable":
        if selected.get("reason") == "resource_exhaustion":
            readiness = {"status": "unavailable", "reason": "resource_exhaustion"}
        else:
            readiness = {
                "status": "unavailable",
                "reason": (selected.get("stop") or {}).get("reason") or "window_unavailable",
            }
    elif selected is not None and selected.get("availability") == "not_yet_released":
        readiness = {"status": "not_ready", "reason": "not_yet_released"}
    else:
        readiness = {"status": "unavailable", "reason": "window_not_found"}
    return {
        "readiness": readiness,
        "source": source,
        "initialization": deepcopy(initialization),
        "region": deepcopy(selected),
        "monitoring": deepcopy(monitoring),
        "window_index": index,
        "timing": _timing(
            total_started, eta_elapsed, region_elapsed,
            status=readiness["status"], reason=readiness.get("reason"),
        ),
    }


__all__ = ["prepare_raw_history"]
