"""Explicit, resumable controller for one certified public-history request.

The controller separates reference fitting, declared-region construction, and
one-at-a-time source decisions.  Importing or inspecting it performs no
numerical work; each potentially expensive stage is selected explicitly.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import json
from time import perf_counter
from typing import Any


_ACTIONS = {"fit_reference", "build_region", "continue", "test", "skip",
            "stop", "resume", "more_data"}
_SOURCE_ACTIONS = {"continue", "test", "skip", "stop", "resume", "more_data"}


def _json_copy(value: Any, label: str) -> Any:
    def validate(item: Any) -> None:
        if item is None or isinstance(item, (bool, int, float, str)):
            return
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError(f"{label} must use string JSON object keys")
            for key, child in item.items():
                validate(key)
                validate(child)
            return
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        raise ValueError(f"{label} must use JSON-native lists and mappings")

    validate(value)
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be strict JSON-compatible") from error
    return deepcopy(value)


def _request_copy(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    if request.get("operation") != "eta_monitor_certified":
        raise ValueError("operation='eta_monitor_certified' is required")
    if "release_events" in request:
        raise ValueError("PublicHistoryWorkflow accepts one complete batch request")
    required = ("public_rows", "reference", "monitor", "as_of", "window_index")
    missing = [key for key in required if key not in request]
    if missing:
        raise ValueError(f"request is missing {missing}")
    return _json_copy(dict(request), "request")


def _window_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("window_index must be a nonnegative integer")
    return value


def _region_result(initialization, monitoring, index):
    records = monitoring.get("records", [])
    selected = next((item for item in records if item.get("window_index") == index), None)
    if selected is None:
        return {"status": "unavailable", "reason": "window_not_found"}, None, None
    availability = selected.get("availability")
    if availability == "available" and isinstance(selected.get("source_spec"), Mapping):
        return {"status": "ready", "reason": None}, deepcopy(selected["source_spec"]), deepcopy(selected)
    if availability == "not_yet_released":
        return {"status": "not_ready", "reason": "not_yet_released"}, None, deepcopy(selected)
    if availability == "unavailable":
        reason = (selected.get("stop") or {}).get("reason") or "window_unavailable"
        return {"status": "unavailable", "reason": reason}, None, deepcopy(selected)
    return {"status": "unavailable", "reason": "missing_source_spec"}, None, deepcopy(selected)


class PublicHistoryWorkflow:
    """Transactional controller for fitting, region preparation, and source review."""

    def __init__(self, request: Mapping[str, Any]):
        self._request = _request_copy(request)
        self._window_index = _window_index(self._request["window_index"])
        if not isinstance(self._request["monitor"], Mapping):
            raise ValueError("request.monitor must be a mapping")
        windows = self._request["monitor"].get("windows")
        if not isinstance(windows, list) or self._window_index >= len(windows):
            raise ValueError("window_index is outside the declared monitor windows")
        self._phase = "input"
        self._initialization = None
        self._prepared = None
        self._source_spec = None
        self._source_evaluator = None
        self._source_workflow = None
        self._stopped = False
        self._more_data = False
        self._events = []
        self._timing = {"fit_reference_elapsed_seconds": 0.0,
                        "region_build_elapsed_seconds": 0.0,
                        "region_source_solver_calls": 0}

    @classmethod
    def from_state(cls, state: Mapping[str, Any], request: Mapping[str, Any]):
        if not isinstance(state, Mapping):
            raise ValueError("workflow state must be a mapping")
        current = _request_copy(request)
        saved_request = _json_copy(state.get("request"), "saved request")
        if saved_request != current:
            raise ValueError("workflow request identity does not match saved state")
        phase = state.get("phase")
        if phase not in {"input", "reference_review", "region_review", "source"}:
            raise ValueError("saved workflow phase is invalid")
        obj = cls.__new__(cls)
        obj._request = current
        obj._window_index = _window_index(current["window_index"])
        obj._phase = phase
        obj._initialization = _json_copy(state.get("initialization"), "saved initialization")
        obj._prepared = _json_copy(state.get("prepared"), "saved prepared region")
        obj._source_spec = _json_copy(state.get("source_spec"), "saved source spec")
        obj._stopped = state.get("status") == "stopped"
        obj._more_data = state.get("status") == "more_data"
        obj._events = _json_copy(state.get("path", []), "saved path")
        if not isinstance(obj._events, list):
            raise ValueError("saved path must be a list")
        obj._timing = _json_copy(state.get("timing", {
            "fit_reference_elapsed_seconds": 0.0,
            "region_build_elapsed_seconds": 0.0,
            "region_source_solver_calls": 0,
        }), "saved timing")
        if phase == "input" and any(state.get(key) is not None
                                     for key in ("initialization", "prepared", "source_spec", "source_workflow")):
            raise ValueError("input phase cannot contain prepared state")
        if phase == "reference_review" and not isinstance(obj._initialization, Mapping):
            raise ValueError("reference_review requires initialization")
        if phase == "region_review" and (not isinstance(obj._initialization, Mapping)
                                          or not isinstance(obj._prepared, Mapping)):
            raise ValueError("region_review requires initialization and prepared region")
        if phase == "source" and (not isinstance(obj._initialization, Mapping)
                                   or not isinstance(obj._prepared, Mapping)
                                   or not isinstance(obj._source_spec, Mapping)):
            raise ValueError("source phase requires complete prepared state")
        if phase == "source":
            if not isinstance(obj._source_spec, Mapping):
                raise ValueError("source phase requires a saved source spec")
            from .certified_source_workflow_backend import CertifiedSourceEvaluator
            from .source_workflow import SourceWorkflow
            obj._source_evaluator = CertifiedSourceEvaluator(obj._source_spec)
            source_state = state.get("source_workflow")
            if not isinstance(source_state, Mapping):
                raise ValueError("source phase requires source workflow state")
            obj._source_workflow = SourceWorkflow.from_state(
                source_state, obj._source_evaluator, context=obj._request)
        else:
            obj._source_evaluator = None
            obj._source_workflow = None
        return obj

    def _status(self):
        if self._stopped:
            return "stopped"
        if self._more_data:
            return "more_data"
        if self._phase == "input":
            return "pending"
        if self._phase == "reference_review":
            return "reference_ready" if self._initialization.get("status") == "ready" else "not_ready"
        if self._phase == "region_review":
            return self._prepared.get("readiness", {}).get("status", "unavailable")
        return self._source_workflow.snapshot().get("status", "pending")

    def _choices(self):
        if self._stopped or self._more_data:
            return [{"action": "resume", "node": None, "label": "resume"},
                    {"action": "more_data", "node": None, "label": "request more data"}]
        if self._phase == "input":
            return [{"action": "fit_reference", "node": None, "label": "fit certified reference"},
                    {"action": "stop", "node": None, "label": "stop"},
                    {"action": "more_data", "node": None, "label": "request more data"}]
        if self._phase == "reference_review":
            if self._initialization.get("status") != "ready":
                return [{"action": "stop", "node": None, "label": "stop"},
                        {"action": "more_data", "node": None, "label": "request more data"}]
            return [{"action": "build_region", "node": None, "label": "build declared region"},
                    {"action": "stop", "node": None, "label": "stop"},
                    {"action": "more_data", "node": None, "label": "request more data"}]
        if self._phase == "region_review":
            return [{"action": "stop", "node": None, "label": "stop"},
                    {"action": "more_data", "node": None, "label": "request more data"}]
        return self._source_workflow.snapshot().get("choices", [])

    def snapshot(self):
        source_state = self._source_workflow.snapshot() if self._source_workflow is not None else None
        result = {
            "status": self._status(), "phase": self._phase,
            "request": deepcopy(self._request), "window_index": self._window_index,
            "initialization": deepcopy(self._initialization),
            "prepared": deepcopy(self._prepared), "source_spec": deepcopy(self._source_spec),
            "source_workflow": source_state, "path": deepcopy(self._events),
            "timing": deepcopy(self._timing),
            "construction_scope": "all_declared_regions_then_selected_window",
            "choices": deepcopy(self._choices()),
        }
        return _json_copy(result, "workflow snapshot")

    def step(self, action: str, node: str | None = None):
        if action not in _ACTIONS:
            raise ValueError(f"unknown workflow action {action!r}")
        if action in {"stop", "resume", "more_data"}:
            if action == "resume" and not (self._stopped or self._more_data):
                raise ValueError("resume is only available after stop or more_data")
            if action == "stop":
                self._stopped, self._more_data = True, False
            elif action == "more_data":
                self._stopped, self._more_data = False, True
            else:
                self._stopped, self._more_data = False, False
            self._events.append({"action": action, "node": None})
            return self.snapshot()
        if self._stopped or self._more_data:
            raise ValueError("resume before selecting a workflow action")
        if action == "fit_reference":
            if self._phase != "input" or node is not None:
                raise ValueError("fit_reference is only available at initial input")
            from .certified_reference import select_eta_certified
            started = perf_counter()
            initialization = select_eta_certified(
                self._request["public_rows"], self._request["reference"],
                as_of=self._request["as_of"],
            )
            old = (self._initialization, self._phase, self._events, self._timing)
            self._initialization = deepcopy(initialization)
            self._phase = "reference_review"
            self._events = self._events + [{"action": action, "node": None}]
            self._timing = dict(self._timing,
                                fit_reference_elapsed_seconds=perf_counter() - started)
            try:
                self.snapshot()
            except Exception:
                self._initialization, self._phase, self._events, self._timing = old
                raise
            return self.snapshot()
        if action == "build_region":
            if node is not None:
                raise ValueError("build_region does not accept a source node")
            if self._phase != "reference_review" or not self._initialization or self._initialization.get("status") != "ready":
                raise ValueError("build_region requires a ready fitted reference")
            from .certified_monitor import monitor_history_certified
            started = perf_counter()
            monitoring = monitor_history_certified(
                self._request["public_rows"], self._initialization,
                self._request["monitor"], self._request["reference"],
                as_of=self._request["as_of"], region_only=True,
            )
            readiness, source_spec, selected = _region_result(
                self._initialization, monitoring, self._window_index)
            prepared = {
                "readiness": readiness, "source": deepcopy(source_spec),
                "region": deepcopy(selected),
                "initialization": deepcopy(self._initialization),
                "monitoring": deepcopy(monitoring), "window_index": self._window_index,
                "construction_scope": "all_declared_regions_then_selected_window",
                "timing": {"region_build_elapsed_seconds": perf_counter() - started,
                           "region_source_solver_calls": 0},
            }
            old = (self._prepared, self._source_spec, self._source_evaluator,
                   self._source_workflow, self._phase, self._events, self._timing)
            try:
                if readiness["status"] == "ready":
                    from .certified_source_workflow_backend import CertifiedSourceEvaluator
                    from .source_workflow import SourceWorkflow
                    evaluator = CertifiedSourceEvaluator(source_spec)
                    source_workflow = SourceWorkflow(
                        evaluator, self._request, goal="full",
                        available=("C0", "CL", "CM", "CMIX"), readiness=readiness,
                    )
                    self._source_evaluator = evaluator
                    self._source_workflow = source_workflow
                    self._source_spec = deepcopy(source_spec)
                    self._phase = "source"
                else:
                    self._phase = "region_review"
                self._prepared = prepared
                self._events = self._events + [{"action": action, "node": None}]
                self._timing = dict(self._timing,
                                    region_build_elapsed_seconds=prepared["timing"]["region_build_elapsed_seconds"],
                                    region_source_solver_calls=0)
                self.snapshot()
            except Exception:
                (self._prepared, self._source_spec, self._source_evaluator,
                 self._source_workflow, self._phase, self._events, self._timing) = old
                raise
            return self.snapshot()
        if self._phase != "source" or action not in _SOURCE_ACTIONS:
            raise ValueError("source action requires a ready built region")
        candidate = self._source_workflow.snapshot()
        from .source_workflow import SourceWorkflow
        trial = SourceWorkflow.from_state(candidate, self._source_evaluator, context=self._request)
        result = trial.step(action, node=node, context=self._request)
        self._source_workflow = trial
        self._events.append({"action": action, "node": node})
        return self.snapshot()


__all__ = ["PublicHistoryWorkflow"]
