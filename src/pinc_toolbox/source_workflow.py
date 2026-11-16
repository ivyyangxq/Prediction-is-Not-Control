"""Resumable, callback-driven source decision workflow.

The controller owns only thin source statuses and the interactive path.  It
never prompts, calls a backend on its own, or copies raw certificates into an
inherited class.  A caller supplies ``evaluate(class_name)`` and explicitly
chooses each node.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import json
from typing import Any

from .diagnosis import diagnose_sources


_NODES = ("C0", "CL", "CM", "CMIX")
_STATUSES = {"compatible", "excluded", "unresolved", "unavailable"}
_NODE_QUESTIONS = {
    "C0": "no-change family",
    "CL": "latent-only explanation",
    "CM": "model-only explanation",
    "CMIX": "joint explanation",
}


class _PendingNode(Exception):
    def __init__(self, node: str):
        super().__init__(node)
        self.node = node


def _copy(value: Any) -> Any:
    return deepcopy(value)


def _json_copy(value: Any, label: str) -> Any:
    def validate(item: Any) -> None:
        if item is None or isinstance(item, (bool, int, float, str)):
            return
        if isinstance(item, Mapping):
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
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be strict JSON-compatible") from exc
    return deepcopy(value)


def _thin(evidence: Mapping[str, Any]) -> dict[str, Any]:
    status = evidence.get("status")
    if status not in _STATUSES:
        raise ValueError(f"source evidence has invalid status {status!r}")
    result = {"status": status}
    if "description" in evidence:
        result["description"] = evidence["description"]
    if (status in {"compatible", "excluded"}
            and (not isinstance(result.get("description"), str)
                 or not result["description"].strip())):
        raise ValueError("compatible/excluded evidence requires a description")
    for key in ("reason", "provenance"):
        if key in evidence:
            result[key] = evidence[key]
    return result


def _choices(*, pending: str | None, tested: Mapping[str, Any], skipped: set[str],
             unavailable: set[str], resolved: bool, stopped: bool, more_data: bool,
             blocked: bool, inherited: set[str]) -> list[dict[str, Any]]:
    if stopped or more_data:
        return [
            {"action": "resume", "node": None, "label": "resume"},
            {"action": "more_data", "node": None, "label": "request more data"},
        ]
    if resolved:
        return [{"action": "stop", "node": None, "label": "stop"}]
    if blocked:
        return [
            {"action": "stop", "node": None, "label": "stop"},
            {"action": "more_data", "node": None, "label": "request more data"},
        ]
    result: list[dict[str, Any]] = []
    if pending is not None:
        result.append({"action": "continue", "node": pending,
                       "label": f"continue with {pending}: {_NODE_QUESTIONS[pending]}"})
    for node in _NODES:
        if node not in tested and node not in skipped and node not in unavailable and node not in inherited:
            question = _NODE_QUESTIONS[node]
            result.append({"action": "test", "node": node, "label": f"test {node}: {question}"})
            result.append({"action": "skip", "node": node, "label": f"skip {node}: {question}"})
    result.extend([
        {"action": "stop", "node": None, "label": "stop"},
        {"action": "more_data", "node": None, "label": "request more data"},
    ])
    return result


class SourceWorkflow:
    """Interactive finite source decision tree over a caller-supplied callback."""

    def __init__(self, evaluate, context, *, goal="full", available=None, readiness=None):
        if not callable(evaluate):
            raise TypeError("evaluate must be callable")
        if not isinstance(context, Mapping):
            raise ValueError("context must be a JSON mapping")
        if goal not in {"full", "model_required"}:
            raise ValueError("goal must be 'full' or 'model_required'")
        if available is not None:
            available = tuple(available)
            if any(not isinstance(node, str) for node in available):
                raise ValueError("available source classes must be strings")
            unknown = set(available).difference(_NODES)
            if unknown:
                raise ValueError(f"unknown source classes: {sorted(unknown)}")
        self._evaluate = evaluate
        self._context = _json_copy(dict(context), "context")
        self._goal = goal
        self._available = None if available is None else tuple(dict.fromkeys(available))
        self._readiness = _json_copy(readiness if readiness is not None else {"status": "ready"}, "readiness")
        self._evidence: dict[str, dict[str, Any]] = {}
        self._skipped: set[str] = set()
        self._events: list[dict[str, Any]] = []
        self._stopped = False
        self._more_data = False

    @classmethod
    def from_state(cls, state, evaluate, *, context):
        if not isinstance(state, Mapping):
            raise ValueError("workflow state must be a mapping")
        if not callable(evaluate):
            raise TypeError("evaluate must be callable")
        _json_copy(context, "context")
        if state.get("context") != context:
            raise ValueError("workflow context does not match saved state")
        if state.get("goal") not in {"full", "model_required"}:
            raise ValueError("saved workflow goal is invalid")
        if state.get("status") not in {"pending", "resolved", "unresolved", "not_ready", "stopped", "more_data"}:
            raise ValueError("saved workflow status is invalid")
        obj = cls.__new__(cls)
        obj._evaluate = evaluate
        obj._context = _copy(state["context"])
        obj._goal = state["goal"]
        available = state.get("available")
        if available is not None and not isinstance(available, (list, tuple, set)):
            raise ValueError("saved workflow available classes must be a list")
        if available is not None and any(not isinstance(node, str) for node in available):
            raise ValueError("saved workflow available classes must be strings")
        obj._available = None if available is None else tuple(available)
        if obj._available is not None and set(obj._available).difference(_NODES):
            raise ValueError("saved workflow available classes are invalid")
        obj._readiness = _json_copy(state.get("readiness", {"status": "ready"}), "saved readiness")
        raw = state.get("evidence", {})
        if not isinstance(raw, Mapping):
            raise ValueError("saved evidence must be a mapping")
        if set(raw).difference(_NODES):
            raise ValueError("saved evidence contains an unknown source class")
        if any(not isinstance(item, Mapping) for item in raw.values()):
            raise ValueError("saved evidence items must be mappings")
        obj._evidence = {node: _json_copy(item, "saved evidence") for node, item in raw.items()}
        for item in obj._evidence.values():
            _thin(item)
        skipped = state.get("statuses", {}).get("user_skipped", [])
        if not isinstance(skipped, (list, tuple, set)):
            raise ValueError("saved skipped classes must be a list")
        if any(not isinstance(node, str) for node in skipped):
            raise ValueError("saved skipped classes must be strings")
        obj._skipped = set(skipped)
        if obj._skipped.difference(_NODES) or obj._skipped.intersection(obj._evidence):
            raise ValueError("saved workflow skipped classes are invalid")
        obj._events = _json_copy(state.get("path", []), "saved path")
        if not isinstance(obj._events, list):
            raise ValueError("saved workflow path must be a list")
        for event in obj._events:
            if not isinstance(event, Mapping) or event.get("action") not in {"continue", "test", "skip", "stop", "resume", "more_data"}:
                raise ValueError("saved workflow path contains an invalid action")
            node = event.get("node")
            if node is not None and node not in _NODES:
                raise ValueError("saved workflow path contains an invalid node")
        obj._check_contradictions(obj._evidence, None)
        obj._stopped = state.get("status") == "stopped"
        obj._more_data = state.get("status") == "more_data"
        return obj

    def _blocked(self) -> bool:
        readiness = self._readiness
        if isinstance(readiness, Mapping):
            status = readiness.get("status")
        else:
            status = readiness
        return status not in {"ready", "supported", "available"}

    def _route(self, evidence: Mapping[str, Mapping[str, Any]], skipped: set[str]):
        def evaluate(node: str):
            if node in evidence:
                return _thin(evidence[node])
            if node in skipped:
                return {"status": "unavailable", "reason": "user_skipped"}
            if self._available is not None and node not in self._available:
                return {"status": "unavailable", "reason": "class_not_available"}
            raise _PendingNode(node)

        try:
            return diagnose_sources(evaluate, available=self._available, goal=self._goal), None
        except _PendingNode as pending:
            return None, pending.node

    def _effective(self, router, evidence=None):
        inherited: dict[str, str] = {}
        provenance: dict[str, dict[str, str]] = {}
        evidence = getattr(self, "_evidence", {}) if evidence is None else evidence
        if evidence.get("C0", {}).get("status") == "compatible":
            for node in ("CL", "CM", "CMIX"):
                inherited[node] = "compatible"
                provenance[node] = {"from": "C0", "relation": "subset_containment"}
        elif evidence.get("CL", {}).get("status") == "compatible" or evidence.get("CM", {}).get("status") == "compatible":
            inherited["CMIX"] = "compatible"
            source = "CL" if evidence.get("CL", {}).get("status") == "compatible" else "CM"
            provenance["CMIX"] = {"from": source, "relation": "subset_containment"}
        if router is not None:
            for node, item in router.get("evidence", {}).items():
                if item.get("inherited_from") is not None:
                    inherited[node] = item.get("status")
                    provenance[node] = {"from": item["inherited_from"], "relation": "router_inheritance"}
        return inherited, provenance

    def _check_contradictions(self, evidence, router):
        affirmative = {node: item.get("status") for node, item in evidence.items()}
        containment_pairs = (("C0", "CL"), ("C0", "CM"), ("C0", "CMIX"),
                             ("CL", "CMIX"), ("CM", "CMIX"))
        for parent, child in containment_pairs:
            if affirmative.get(parent) == "compatible" and affirmative.get(child) == "excluded":
                raise ValueError(f"independent {child} evidence contradicts {parent} containment")
        if router is None:
            return
        for node, raw in evidence.items():
            effective = router.get("evidence", {}).get(node, {})
            raw_status = raw.get("status")
            effective_status = effective.get("status")
            if (effective.get("inherited_from") is not None
                    and raw_status in {"compatible", "excluded"}
                    and effective_status in {"compatible", "excluded"}
                    and raw_status != effective_status):
                raise ValueError(f"independent {node} evidence contradicts containment")

    def _view(self, *, evidence=None, skipped=None, status_override=None, event=None):
        evidence = self._evidence if evidence is None else evidence
        skipped = self._skipped if skipped is None else skipped
        blocked = self._blocked()
        if blocked:
            router, pending = None, None
            if status_override is not None:
                status = status_override
            elif self._stopped:
                status = "stopped"
            elif self._more_data:
                status = "more_data"
            else:
                status = "not_ready"
        else:
            router, pending = self._route(evidence, skipped)
            if status_override is not None:
                status = status_override
            elif self._stopped:
                status = "stopped"
            elif self._more_data:
                status = "more_data"
            elif router is None:
                status = "pending"
            elif router.get("outcome") in {"unresolved", "ambiguous"}:
                status = "unresolved"
            else:
                status = "resolved"
        if event is not None:
            events = self._events + [_copy(event)]
        else:
            events = self._events
        effective_inherited, inherited_provenance = self._effective(router, evidence)
        unavailable = set(skipped) | {
            node for node, item in evidence.items() if item.get("status") == "unavailable"
        } | {
            node for node, item in (router or {}).get("evidence", {}).items()
            if item.get("status") == "unavailable"
        }
        if self._available is not None:
            unavailable.update(set(_NODES).difference(self._available))
        unresolved = {
            node for node, item in evidence.items() if item.get("status") == "unresolved"
        }
        tested = {node: item.get("status") for node, item in evidence.items()}
        used = set(tested) | set(effective_inherited) | unavailable
        not_run = [node for node in _NODES if node not in used]
        choices = _choices(
            pending=pending, tested=tested, skipped=skipped, unavailable=unavailable,
            resolved=status == "resolved", stopped=status == "stopped", more_data=status == "more_data",
            blocked=blocked, inherited=set(effective_inherited),
        )
        result = {
            "status": status,
            "context": _copy(self._context),
            "goal": self._goal,
            "available": None if self._available is None else list(self._available),
            "readiness": _copy(self._readiness),
            "evidence": _copy(dict(evidence)),
            "statuses": {
                "tested": tested,
                "inherited": effective_inherited,
                "inheritance_provenance": inherited_provenance,
                "not_run": not_run,
                "user_skipped": sorted(skipped),
                "unavailable": sorted(unavailable),
                "unresolved": sorted(unresolved),
            },
            "pending_node": pending,
            "choices": choices,
            "path": _copy(events),
            "router": _copy(router),
        }
        return _json_copy(result, "workflow snapshot")

    def snapshot(self):
        return self._view()

    def step(self, action, node=None, context=None):
        if action not in {"continue", "test", "skip", "stop", "resume", "more_data"}:
            raise ValueError(f"unknown workflow action {action!r}")
        if context is not None and context != self._context:
            raise ValueError("workflow context does not match current state")
        if action in {"stop", "resume", "more_data"}:
            if action == "stop":
                event = {"action": action, "node": None}
                self._stopped, self._more_data = True, False
                self._events.append(event)
                return self.snapshot()
            if action == "more_data":
                event = {"action": action, "node": None}
                self._more_data, self._stopped = True, False
                self._events.append(event)
                return self._view(status_override="more_data", event=None)
            self._stopped, self._more_data = False, False
            self._events.append({"action": action, "node": None})
            return self.snapshot()
        if self._stopped or self._more_data:
            raise ValueError("resume before selecting or skipping a source node")
        if self._blocked():
            raise ValueError("readiness is not sufficient; choose stop or more_data")
        preview, pending = self._route(self._evidence, self._skipped)
        if action == "continue" and node is not None and node != pending:
            raise ValueError("explicit continue node does not match the pending node")
        if (action in {"test", "skip"} and preview is not None
                and preview.get("outcome") not in {"unresolved", "ambiguous"}):
            raise ValueError("workflow is resolved; node actions are no longer permitted")
        chosen = pending if action == "continue" else node
        if chosen not in _NODES:
            raise ValueError("a valid source node is required")
        if chosen in self._evidence or chosen in self._skipped:
            raise ValueError("source node has already been tested or skipped")
        if self._available is not None and chosen not in self._available:
            raise ValueError("source node is unavailable")
        if action == "continue" and pending != chosen:
            raise ValueError("continue requires the router's pending node")
        inferred_before, _ = self._effective(None, self._evidence)
        if chosen in inferred_before:
            raise ValueError("source node is already inferable by containment")
        candidate_evidence = {key: _copy(value) for key, value in self._evidence.items()}
        candidate_skipped = set(self._skipped)
        if action == "skip":
            candidate_skipped.add(chosen)
        else:
            result = self._evaluate(chosen)
            if not isinstance(result, Mapping):
                raise ValueError("evaluate must return a source evidence mapping")
            candidate_evidence[chosen] = _json_copy(dict(result), "source evidence")
            _thin(candidate_evidence[chosen])
        router, next_pending = self._route(candidate_evidence, candidate_skipped)
        self._check_contradictions(candidate_evidence, router)
        self._evidence = candidate_evidence
        self._skipped = candidate_skipped
        self._events.append({"action": action, "node": chosen})
        return self._view()


__all__ = ["SourceWorkflow"]
