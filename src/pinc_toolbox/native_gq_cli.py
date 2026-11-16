"""Explicit JSON history interface for :class:`pinc_toolbox.native_gq.RND2GQ`.

The request contains one caller-supplied ``history`` list, one explicit
``settings`` object, an ``initial`` fit event, optional ``update_events``,
optional public action masks, and query records with ``x`` and ``aux``.  No
repository paths, generators, hidden labels, automatic RA decisions, or
auxiliary-history builders are used.  Updates are the historical events the
caller supplies; an update failure retains the last committed snapshot and
records the actual exception.

Run with ``python -m pinc_toolbox.native_gq_cli --input REQUEST --output
OUTPUT``.  OUTPUT must be new.  The JSON response keeps every event snapshot
immutable.  Unsupported Q cells are serialized as JSON ``null`` and their
support mask is serialized separately.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .native_gq import AUX, RND2GQ
from .g_estimators import EstimatorConfig


_SCHEMA = "native_gq.v1"


def _error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _plain(value: Any) -> Any:
    """Convert ordinary finite values to strict-JSON-compatible objects."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite value outside an explicitly unsupported Q cell")
        return number
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _q_json(value: Any, support: Any) -> list[list[float | None]]:
    q = np.asarray(value, dtype=float)
    mask = np.asarray(support, dtype=bool)
    if q.shape != mask.shape or q.ndim != 2:
        raise ValueError("Q table and support mask must have the same two-dimensional shape")
    if not np.isfinite(q[mask]).all():
        raise ValueError("supported Q cells must be finite")
    return [
        [None if not mask[row, action] else float(q[row, action])
         for action in range(q.shape[1])]
        for row in range(q.shape[0])
    ]


def _query_q_json(value: Any) -> tuple[list[float | None], list[bool]]:
    q = np.asarray(value, dtype=float)
    if q.shape != (1, 11):
        raise ValueError("query Q must have shape (1, 11)")
    support = np.isfinite(q[0])
    return [None if not support[action] else float(q[0, action]) for action in range(11)], support.tolist()


def _action_json(value: Any) -> float | None:
    number = float(np.asarray(value, dtype=float).reshape(-1)[0])
    return None if not math.isfinite(number) else number


def _mask(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 11 or any(type(item) is not bool for item in value):
        raise ValueError(f"{name} must contain exactly 11 JSON booleans")
    result = np.asarray(value, dtype=bool)
    if not result.any():
        raise ValueError(f"{name} must leave at least one action publicly feasible")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _settings(request: Mapping[str, Any]) -> dict[str, Any]:
    settings = request.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("request.settings must be an object")
    branch = settings.get("branch")
    if branch not in AUX:
        raise ValueError("settings.branch must be base, gamma, epsilon, or full")
    if "beta" not in settings:
        raise ValueError("settings.beta is required")
    beta = float(settings["beta"])
    if not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("settings.beta must be finite and satisfy 0 <= beta < 1")
    horizon = settings.get("horizon")
    if horizon == "infinite" or horizon is None:
        horizon = None
    else:
        horizon = _integer(horizon, "settings.horizon")
        if horizon < 1:
            raise ValueError("settings.horizon must be positive or 'infinite'")
    window = _integer(settings.get("window"), "settings.window")
    if window < 1:
        raise ValueError("settings.window must be positive")
    nodes_per_memory = _integer(settings.get("nodes_per_memory", 5), "settings.nodes_per_memory")
    noise_nodes = _integer(settings.get("noise_nodes", 8), "settings.noise_nodes")
    if nodes_per_memory < 1 or noise_nodes < 1:
        raise ValueError("settings node and residual counts must be positive")
    estimator = settings.get("estimator", "linear")
    pipeline = settings.get("pipeline", "planning_only")
    estimator_params = settings.get("estimator_params", {})
    config = EstimatorConfig(estimator=estimator, pipeline=pipeline,
                             estimator_params=estimator_params)
    result = {
        "branch": branch,
        "beta": beta,
        "horizon": horizon,
        "window": window,
        "nodes_per_memory": nodes_per_memory,
        "noise_nodes": noise_nodes,
        "estimator": config.estimator,
        "pipeline": config.pipeline,
        "estimator_params": dict(config.estimator_params),
    }
    if "reward_coefficients" in settings:
        result["reward_coefficients"] = settings["reward_coefficients"]
    if settings.get('g_spec','legacy')!='legacy' or any(k in settings for k in
            ('outer_estimator','outer_estimator_params','policy_columns')):
        # Constructor validates module/outer separation and the macro schema.
        result.update(g_spec=settings.get('g_spec','legacy'),
            outer_estimator=settings.get('outer_estimator','linear'),
            outer_estimator_params=settings.get('outer_estimator_params',{}),
            policy_columns=settings.get('policy_columns',[]))
    return result


def _queries(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("queries must be a list")
    result = []
    for index, record in enumerate(value):
        if not isinstance(record, Mapping) or "x" not in record:
            raise ValueError(f"queries[{index}] must contain x")
        x = record["x"]
        if not isinstance(x, list) or len(x) != 3:
            raise ValueError(f"queries[{index}].x must have three values")
        aux = record.get("aux")
        if aux is not None and not isinstance(aux, list):
            raise ValueError(f"queries[{index}].aux must be a list or null")
        detail = record.get("detail", False)
        if type(detail) is not bool:
            raise ValueError(f"queries[{index}].detail must be a JSON boolean")
        evaluation = record.get("evaluation", "nearest")
        if evaluation not in ("nearest", "bellman"):
            raise ValueError(f"queries[{index}].evaluation must be nearest or bellman")
        remaining = record.get("remaining_horizon")
        if remaining is not None:
            remaining = _integer(remaining, f"queries[{index}].remaining_horizon")
            if remaining < 1:
                raise ValueError("remaining_horizon must be positive or null")
        if detail and evaluation == "bellman":
            raise ValueError("detail describes nearest-node lookup only; use detail=false for Bellman queries")
        result.append({"x": deepcopy(x), "aux": deepcopy(aux), "detail": detail,
                       "evaluation": evaluation, "remaining_horizon": remaining})
    return result


def _event_mask(request: Mapping[str, Any], event: Mapping[str, Any], as_of: int,
                *, initial: bool, index: int) -> np.ndarray | None:
    if "feasible" in event:
        return _mask(event["feasible"], "event.feasible")
    public_masks = request.get("public_masks")
    if public_masks is None:
        return None
    if not isinstance(public_masks, Mapping):
        raise ValueError("public_masks must be an object")
    candidates = (["initial", str(as_of)] if initial else
                  [f"update_{index}", str(as_of)])
    for key in candidates:
        if key in public_masks:
            return _mask(public_masks[key], f"public_masks[{key!r}]")
    return None


def _validate_request(request: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(request, Mapping) or request.get("operation") != "native_gq":
        raise ValueError("operation='native_gq' is required")
    history = request.get("history")
    if not isinstance(history, list) or not history or any(not isinstance(row, Mapping) for row in history):
        raise ValueError("history must be a nonempty list of records")
    settings = _settings(request)
    initial = request.get("initial")
    if not isinstance(initial, Mapping) or "as_of" not in initial:
        raise ValueError("initial must contain as_of")
    initial = {"as_of": _integer(initial["as_of"], "initial.as_of")}
    if "feasible" in request.get("initial", {}):
        initial["feasible"] = deepcopy(request["initial"]["feasible"])
    updates = request.get("update_events", [])
    if not isinstance(updates, list):
        raise ValueError("update_events must be a list")
    normalized_updates = []
    for index, event in enumerate(updates):
        if not isinstance(event, Mapping) or "as_of" not in event:
            raise ValueError(f"update_events[{index}] must contain as_of")
        normalized = {"as_of": _integer(event["as_of"], f"update_events[{index}].as_of")}
        if "feasible" in event:
            normalized["feasible"] = deepcopy(event["feasible"])
        normalized_updates.append(normalized)
    queries = _queries(request.get("queries"))
    return pd.DataFrame([dict(row) for row in history]), settings, initial, normalized_updates, queries


def _snapshot(model: RND2GQ, queries: list[dict[str, Any]]) -> dict[str, Any]:
    coefficients = {}
    for name, fitted in (("G", model.model_), ("base_G", model.base_model_)):
        if model.g_spec=='v012':
            coefficients[name]={"status":"available_modules", "C":None,"Psi":None,
                "module_alpha":_plain(fitted.core.modules._C),
                "module_psi":_plain(fitted.core.modules._Psi),
                "outer_coefficients":_plain(getattr(fitted.core,'outer_coefficients_',None)),
                "outer_config":_plain(model.outer_config_.as_dict()),
                "diagnostics":_plain(fitted.diagnostics_)}
        elif hasattr(fitted.core, "_C") and hasattr(fitted.core, "_Psi"):
            coefficients[name] = {
                "status": "available",
                "C": _plain(fitted.core._C),
                "Psi": _plain(fitted.core._Psi),
                "diagnostics": _plain(fitted.diagnostics_),
            }
        else:
            coefficients[name] = {
                "status": "unavailable",
                "C": None,
                "Psi": None,
                "reason": "selected estimator does not expose SimpleGQ coefficients",
                "diagnostics": _plain(fitted.diagnostics_),
            }
    nodes = model.nodes_.to_dict(orient="records")
    finite_q = None
    if model.q_by_horizon_ is not None:
        finite_q = {str(horizon): _q_json(value, model.mask_)
                    for horizon, value in model.q_by_horizon_.items()}
    infinite_q = None if model.q_by_horizon_ is not None else _q_json(model.q_, model.mask_)
    query_outputs = []
    actions = []
    for query in queries:
        x = np.asarray([query["x"]], dtype=float)
        auxiliary = None if query["aux"] is None else np.asarray([query["aux"]], dtype=float)
        evaluation = query.get("evaluation", "nearest")
        remaining = query.get("remaining_horizon")
        detail = model.explain_q_values(x, auxiliary, remaining_horizon=remaining) if query.get("detail", False) else None
        q = model.q_values(x, auxiliary, evaluation=evaluation, remaining_horizon=remaining) if detail is None else detail["q_values"]
        action = model.act(x, auxiliary, evaluation=evaluation, remaining_horizon=remaining)
        q_values, supported = _query_q_json(q)
        action_value = _action_json(action)
        query_outputs.append({
            "x": deepcopy(query["x"]), "aux": deepcopy(query["aux"]),
            "q": q_values, "supported": supported, "action": action_value,
            "evaluation": evaluation,
            "remaining_horizon": model.horizon if remaining is None else remaining,
        })
        if detail is not None:
            query_outputs[-1]["detail"] = _plain(detail["rows"][0])
        actions.append(action_value)
    return {
        "as_of": int(model.as_of_),
        "window": int(model.window_),
        "branch": model.auxiliary,
        "beta": float(model.beta),
        "horizon": model.horizon,
        "estimator_config": _plain(model.estimator_config_.as_dict()),
        "action_grid": _plain(model.action_grid_),
        "coefficients": coefficients,
        "nodes": _plain(nodes),
        "mask": _plain(model.mask_),
        "planning_feasible": _plain(model.planning_feasible_),
        "reward_coefficients": _plain(model.reward_coefficients_),
        "rewards": _plain(model.rewards_),
        "kernel": _plain(model.transitions_),
        "q": _q_json(model.q_, model.mask_),
        "finite_q": finite_q,
        "infinite_q": infinite_q,
        "queries": query_outputs,
        "actions": actions,
        "diagnostics": _plain(model.diagnostics_),
    }


def run_request(request: Mapping[str, Any]) -> dict[str, Any]:
    history, settings, initial, updates, queries = _validate_request(request)
    model = RND2GQ(
        auxiliary=settings["branch"], beta=settings["beta"], horizon=settings["horizon"],
        nodes_per_memory=settings["nodes_per_memory"], noise_nodes=settings["noise_nodes"],
        estimator=settings["estimator"], pipeline=settings["pipeline"],
        estimator_params=settings["estimator_params"],
        g_spec=settings.get('g_spec','legacy'),
        outer_estimator=settings.get('outer_estimator','linear'),
        outer_estimator_params=settings.get('outer_estimator_params'),
        policy_columns=settings.get('policy_columns',()),
        reward_coefficients=settings.get("reward_coefficients"),
    )
    events = []
    previous = None
    failed = False
    try:
        mask = _event_mask(request, initial, initial["as_of"], initial=True, index=0)
        model.fit(history, as_of=initial["as_of"], window=settings["window"], feasible=mask)
        previous = _snapshot(model, queries)
        events.append({"event": "initial", "as_of": initial["as_of"],
                       "status": "completed", "snapshot": deepcopy(previous)})
    except Exception as error:
        failed = True
        events.append({"event": "initial", "as_of": initial["as_of"],
                       "status": "failed", "error": _error(error)})

    for index, update in enumerate(updates):
        if failed or previous is None:
            events.append({"event": f"update_{index}", "as_of": update["as_of"],
                           "status": "not_run", "snapshot": deepcopy(previous),
                           "error": {"type": "PreviousEventFailed", "message": "previous event failed"}})
            continue
        try:
            mask = _event_mask(request, update, update["as_of"], initial=False, index=index)
            model.update(history, as_of=update["as_of"], window=settings["window"], feasible=mask)
            previous = _snapshot(model, queries)
            events.append({"event": f"update_{index}", "as_of": update["as_of"],
                           "status": "completed", "snapshot": deepcopy(previous)})
        except Exception as error:
            failed = True
            events.append({"event": f"update_{index}", "as_of": update["as_of"],
                           "status": "failed", "error": _error(error),
                           "snapshot": deepcopy(previous), "retained_previous_snapshot": True})
    return {
        "schema": _SCHEMA,
        "operation": "native_gq",
        "status": "failed" if failed else "completed",
        "settings": _plain(settings),
        "history_records": len(history),
        "query_records": len(queries),
        "events": events,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; supply a fresh output path")
    try:
        request = json.loads(args.input.read_text(encoding="utf-8"))
        report = run_request(request)
    except Exception as error:
        report = {
            "schema": _SCHEMA, "operation": "native_gq", "status": "failed",
            "events": [], "error": _error(error),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_request", "main"]
