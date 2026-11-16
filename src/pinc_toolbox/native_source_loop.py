"""Portable host-in-the-loop runner for a fixed-calendar native source study.

The host is an explicit trust boundary. It receives public configuration and
proposed decisions only; reference panels and detector inputs remain local.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np
import pandas as pd

from .eta_domain import select_eta_domain, eta_qualified_native_action
from .native_gq import RND2GQ
from .source_acquisition import recommend_source_action
from .eta_online_gq_two_stage import _qualified_block
from .predictable_gq_two_stage import prepare_predictable_gq_region, run_predictable_gq_two_stage
from .certified_source_workflow_backend import CertifiedSourceEvaluator
from .two_stage_report import run_two_stage_report


_FORBIDDEN_RECEIPT = {"next_state", "next_physical", "next_w", "reward", "outcome",
                      "outcomes", "innovation", "shock", "observation"}


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _id(value, name):
    if not isinstance(value, (str, int)) or isinstance(value, bool) or value == "":
        raise ValueError(f"{name} must be a nonempty string or integer")
    return value


def _finite(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_reward(value):
    """Keep evaluator reward metadata when finite; never make it learner input."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _bin_weights(action):
    # A single pre-innovation action selects exactly one of the four declared bins.
    return [float(action < .025), float(.025 <= action < .05),
            float(.05 <= action < .075), float(action >= .075)]


def _reference(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    result = deepcopy(dict(value))
    for key in ("states", "actions", "next_physical", "dates", "record_ids"):
        if key not in result:
            raise ValueError(f"{name}.{key} is required")
    n = len(result["record_ids"])
    if not n or any(len(result[key]) != n for key in ("states", "actions", "next_physical", "dates")):
        raise ValueError(f"{name} arrays must have the same positive length")
    if len(set(map(repr, result["record_ids"]))) != n:
        raise ValueError(f"{name}.record_ids must be distinct")
    result["dates"] = [_integer(x, f"{name}.dates") for x in result["dates"]]
    if any(a >= b for a, b in zip(result["dates"], result["dates"][1:])):
        raise ValueError(f"{name}.dates must be strictly increasing")
    result["states"] = np.asarray(result["states"], dtype=float).tolist()
    result["actions"] = np.asarray(result["actions"], dtype=float).reshape(-1).tolist()
    result["next_physical"] = np.asarray(result["next_physical"], dtype=float).reshape(n, -1).tolist()
    if not all(np.isfinite(np.asarray(result[k], dtype=float)).all()
               for k in ("states", "actions", "next_physical")):
        raise ValueError(f"{name} contains nonfinite values")
    if result["next_physical"] and len(result["next_physical"][0]) != 1:
        raise ValueError(f"{name} must have scalar physical outcomes")
    return result


def _unresolved(stage1, context, reason, trace):
    readiness = {"status": "unavailable", "reason": str(reason)}
    report = run_two_stage_report(stage1, context=context, readiness=readiness)
    report["completion"]["failures"].append({"reason": str(reason)})
    report["host_loop"] = trace
    return report


def run_native_source_loop(request: Mapping, host) -> dict:
    """Run a predeclared source loop using ``host.release/state/step``.

    ``host`` must have been created by a caller-supplied trusted factory. The
    only input passed to that factory by the CLI is ``public_host_configuration``.
    """
    if not isinstance(request, Mapping) or not isinstance(host, object):
        raise ValueError("request and host are required")
    calendar = deepcopy(dict(request["calendar"]))
    policy_spec = deepcopy(dict(request["policy"]))
    detector_spec = deepcopy(dict(request["detector"]))
    source_spec = deepcopy(dict(request["source"]))
    if policy_spec.get("auxiliary", "base") != "base":
        raise ValueError("native host loop currently accepts the base policy only")
    epsilon = _finite(policy_spec.get("epsilon", 0.0), "policy.epsilon")
    if not 0 <= epsilon <= 1:
        raise ValueError("policy.epsilon must lie in [0,1]")
    for key in ("alpha_reference", "alpha_noise"):
        alpha = _finite(source_spec[key], "source." + key)
        if not 0 < alpha < 1:
            raise ValueError("source error budgets must lie in (0,1)")
    for key in ("alpha_reference", "alpha_noise"):
        alpha = _finite(detector_spec[key], "detector." + key)
        if not 0 < alpha < 1:
            raise ValueError("detector error budgets must lie in (0,1)")
    start = _integer(calendar["start_time"], "start_time")
    deadline = _integer(calendar["final_deadline"], "final_deadline")
    source_start = _integer(calendar["source_start"], "source_start")
    source_deadline = _integer(calendar["source_deadline"], "source_deadline")
    if deadline < start or source_start < start or source_deadline != deadline:
        raise ValueError("invalid predeclared calendar")
    source_n = _integer(calendar["source_n"], "source_n")
    if source_n < 1 or source_n != source_deadline - source_start:
        raise ValueError("fixed source_n must equal source_deadline - source_start")
    looks = deepcopy(list(calendar["detector_looks"]))
    if not looks or any(_integer(x["look_time"], "look_time") > source_start for x in looks):
        raise ValueError("all fixed detector looks must be completed by source_start")
    look_times = [_integer(x["look_time"], "look_time") for x in looks]
    if look_times != sorted(set(look_times)):
        raise ValueError("detector look times must be strictly increasing")
    for look in looks:
        look.update(look_time=_integer(look["look_time"], "look_time"),
                    start_time=_integer(look["start_time"], "look.start_time"),
                    intended_n=_integer(look["intended_n"], "look.intended_n"))
        if look["intended_n"] < 1 or look["start_time"] + look["intended_n"] != look["look_time"]:
            raise ValueError("detector window must end at its fixed look_time")
    selection = deepcopy(dict(request["eta_selection"]))
    selection_ids = [_id(x, "eta selection record id") for x in selection["record_ids"]]
    domain = select_eta_domain(selection["public_rows"], selection["spec"],
                               as_of=selection["as_of"], selection_record_ids=selection_ids)
    detector_ref = _reference(request["detector_reference"], "detector_reference")
    source_ref = _reference(request["source_reference"], "source_reference")
    if max(detector_ref["dates"]) > start or max(source_ref["dates"]) > start:
        raise ValueError("reference dates must be fixed before online start")
    init = deepcopy(list(request["initial_history"]))
    init_ids = [_id(x["record_id"], "initial record id") for x in init]
    if len(set(init_ids)) != len(init_ids):
        raise ValueError("initial record IDs must be distinct")
    all_reserved = set(map(repr, selection_ids + init_ids + detector_ref["record_ids"] + source_ref["record_ids"]))
    if len(all_reserved) != len(selection_ids) + len(init_ids) + len(detector_ref["record_ids"]) + len(source_ref["record_ids"]):
        raise ValueError("selection, behavior, detector reference and source reference IDs must be disjoint")
    if source_ref.get("independent_of_alarm_history") is not True or detector_ref.get("fixed_design") is not True:
        raise ValueError("reference design declarations are required")

    trace = {"action_trace": [], "release_trace": [], "fit_events": [],
             "detector_records": [], "domain": deepcopy(domain),
             "source": {"status": "not_entered", "planned_n": source_n,
                        "observed_n": 0, "deadline": source_deadline}}
    context = {"route": "native_host_source_loop", "calendar": calendar,
               "eta_domain": domain, "source_deadline": source_deadline,
               "statistical_error_bounds": None,
               "statistical_error_bounds_status": "null_assumptions_unverified"}
    stage1 = {"status": "unavailable", "alarm_time": None, "first_alarm": None,
              "calendar": [x["look_time"] for x in looks], "records": [],
              "not_visited_times": [], "event": "first C0 exclusion on fixed released windows"}
    if domain.get("status") != "ready":
        stage1["status"] = "unresolved"
        stage1["records"].append({"status": "unavailable", "reason": "eta_initialization_unavailable"})
        return _unresolved(stage1, context, "eta_initialization_unavailable", trace)

    # Host factory receives only this explicitly public configuration map.
    # This runner never passes either reference panel to host methods.
    releases_by_id = {}
    ack_by_id = {}
    released_ids = set()
    released_decisions = set()
    used_execution_ids = set()
    behavior_rows = []
    for row in init:
        behavior_rows.append({"state": row["state"][0], "w": row["state"][1],
                              "previous_action": row["state"][2], "action": row["executed_action"],
                              "next_state": row["next_physical"], "next_w": row["next_w"],
                              "period": row["decision_time"],
                              "release_period": row["outcome_release_time"], "path_id": row["path_id"]})
    model = RND2GQ(policy_spec.get("auxiliary", "base"), beta=policy_spec.get("beta", .95),
                   horizon=policy_spec.get("horizon"),
                   estimator=policy_spec.get("estimator", "linear"),
                   estimator_params=policy_spec.get("estimator_params"),
                   pipeline=policy_spec.get("pipeline", "planning_only"),
                   reward_coefficients=policy_spec.get("reward_coefficients"))
    raw_feasible = policy_spec["feasible_mask"]
    if (not isinstance(raw_feasible, (list, tuple, np.ndarray))
            or any(type(v) not in (bool, np.bool_) for v in raw_feasible)):
        return _unresolved(stage1, context, "policy feasible_mask entries must be booleans", trace)
    feasible = np.asarray(raw_feasible, dtype=bool)
    if feasible.shape != (11,) or not np.array_equal(np.asarray(policy_spec["action_grid"], float), model.action_grid_):
        return _unresolved(stage1, context, "policy action grid/mask differs from installed native policy", trace)
    rng = np.random.default_rng(policy_spec["seed"])
    online_path = _id(policy_spec["online_path_id"], "online_path_id")
    if str(online_path) in {str(row["path_id"]) for row in init}:
        raise ValueError("online_path_id must be distinct from initialization paths")
    if _integer(selection["as_of"], "eta_selection.as_of") > start:
        return _unresolved(stage1, context, "eta_selection_after_online_start", trace)
    for row in init:
        dt = _integer(row["decision_time"], "initial decision_time")
        rt = _integer(row["outcome_release_time"], "initial outcome_release_time")
        if dt >= start or rt <= dt or rt > start:
            return _unresolved(stage1, context, "initial_history_not_released_by_start", trace)
    fitted = False
    alarm_time = None
    detector_by_time = {int(x["look_time"]): x for x in looks}
    release_failure = None
    terminal_report = None

    def release_at(t):
        nonlocal release_failure
        try:
            rows = host.release(as_of=t)
            if not isinstance(rows, (list, tuple)):
                raise ValueError("release must return a list of released transitions")
            for raw in rows:
                if not isinstance(raw, Mapping):
                    raise ValueError("release row must be an object")
                row = dict(raw)
                did = _id(row.get("decision_id"), "released decision_id")
                # The host schema keys a released outcome by its acknowledged
                # execution when it has no separate observation identifier.
                rid = _id(row.get("record_id", "outcome/" + str(did)), "released record_id")
                available = _integer(row.get("available_at"), "available_at")
                outcome_time = _integer(row.get("outcome_release_time"), "outcome_release_time")
                if available > t or outcome_time > t or available < outcome_time:
                    raise ValueError("future_release")
                if (rid in released_ids or rid in init_ids or rid in selection_ids
                        or rid in detector_ref["record_ids"] or rid in source_ref["record_ids"]):
                    raise ValueError("duplicate_record_id")
                if did in released_decisions:
                    raise ValueError("duplicate_release_for_decision")
                if did not in ack_by_id:
                    raise ValueError("release_without_acknowledged_execution")
                ack = ack_by_id[did]
                if row.get("execution_id", ack["execution_id"]) != ack["execution_id"]:
                    raise ValueError("released_execution_id_does_not_match_receipt")
                if (_integer(row.get("decision_time"), "decision_time") != ack["decision_time"]
                        or float(row.get("executed_action")) != ack["action"]):
                    raise ValueError("released_transition_does_not_match_receipt")
                if rid == ack["record_id"] if ack.get("record_id") is not None else False:
                    raise ValueError("record_id must be new at release")
                px = _finite(row["next_physical"], "next_physical")
                nw = _finite(row["next_w"], "next_w")
                # A missing or invalid evaluator label must not block learning
                # from an otherwise valid released physical transition.
                reward = _optional_reward(row.get("reward"))
                if outcome_time <= ack["decision_time"]:
                    raise ValueError("outcome_release_time must follow decision_time")
                released_ids.add(rid)
                released_decisions.add(did)
                item = {**ack, "record_id": rid, "next_physical": px, "next_w": nw,
                        "reward": reward, "release_time": t, "outcome_release_time": outcome_time}
                releases_by_id[did] = item
                behavior_rows.append({"state": item["state"][0], "w": item["state"][1],
                                      "previous_action": item["state"][2], "action": item["action"],
                                      "next_state": px, "next_w": nw,
                                      "period": item["decision_time"], "release_period": outcome_time,
                                      "path_id": online_path})
                trace["release_trace"].append({"record_id": rid, "decision_id": did,
                    "available_at": available, "outcome_release_time": outcome_time,
                    "released_at": t, "decision_time": item["decision_time"],
                    "state": item["state"], "executed_action": item["action"],
                    "next_physical": px, "next_w": nw, "reward": reward,
                    "predictable_bin_weights": item["weights"], "status": "released"})
                if alarm_time is not None and source_start <= item["decision_time"] < source_deadline:
                    observed = sum(source_start <= z["decision_time"] < source_deadline
                                   for z in releases_by_id.values())
                    trace["source"].update(status="pending", observed_n=observed)
        except Exception as exc:
            release_failure = str(exc)
            trace["release_trace"].append({"released_at": t, "status": "failed", "reason": release_failure})
        return release_failure is None

    def make_block(look):
        ts = range(int(look["start_time"]), int(look["start_time"]) + int(look["intended_n"]))
        rows = [next((v for v in releases_by_id.values() if v["decision_time"] == u), None) for u in ts]
        if any(v is None for v in rows):
            raise ValueError("incomplete_fixed_window: missing legally released detector transitions")
        # Eta observation features are (physical state, previous action, action).
        block = {"states": [[x["state"][0], x["state"][2]] for x in rows],
                 "actions": [x["action"] for x in rows],
                 "weights": np.asarray([x["weights"] for x in rows], dtype=float).T,
                 "intended_n": int(look["intended_n"]),
                 "record_ids": [x["record_id"] for x in rows],
                 "dates": [x["outcome_release_time"] for x in rows],
                 "next_physical": [[x["next_physical"]] for x in rows]}
        qualified, support = _qualified_block(block, domain)
        if support["informative_weight_rows"] == 0:
            raise ValueError("no_eta_qualified_weighted_detection_rows")
        future = {"states": block["states"],
                  "actions": block["actions"], "next_physical": block["next_physical"],
                  "weights": qualified["weights"], "dates": block["dates"],
                  "record_ids": block["record_ids"]}
        measured = prepare_predictable_gq_region(detector_ref, future, sigma=detector_spec["sigma"] if "sigma" in detector_spec else source_spec["sigma"],
            alpha_reference=detector_spec["alpha_reference"], alpha_noise=detector_spec["alpha_noise"],
            calendar_size=len(looks), latent_bound=source_spec.get("latent_bound", .6),
            model_bound=source_spec.get("model_bound", 2.))
        evidence = CertifiedSourceEvaluator(measured["source_spec"])("C0")
        status = {"excluded": "alarm", "compatible": "no_alarm"}.get(evidence.get("status"), "unresolved")
        return {"time": int(look["look_time"]), "status": status, "C0": evidence,
                "measurement": measured, "eta_support": support,
                "diagnostic_block": {**block, "weights": qualified["weights"]},
                "provenance": {"record_ids": block["record_ids"],
                               "reference_ids": detector_ref["record_ids"]}}

    try:
        for t in range(start, deadline + 1):
            if not release_at(t):
                raise ValueError(release_failure)
            # Evaluate only predeclared detector looks, with complete released blocks.
            if t in detector_by_time and alarm_time is None:
                look = detector_by_time[t]
                rec = make_block(look)
                stage1["records"].append(rec); trace["detector_records"].append(deepcopy(rec))
                if rec["status"] == "alarm":
                    alarm_time = t; stage1.update(status="alarm", alarm_time=t, first_alarm=t,
                        provenance={"record_ids": selection_ids + init_ids +
                            [rid for prior in stage1["records"] for rid in prior.get("provenance", {}).get("record_ids", [])],
                            "detector_reference_ids": detector_ref["record_ids"]})
                    stage1["not_visited_times"] = [u for u in stage1["calendar"] if u > t]
                    trace["source"].update(status="pending", observed_n=0)
                    # Detector ends at its first fixed alarm; source never shifts.
                elif rec["status"] == "unresolved":
                    stage1.update(status="unresolved")
                    raise ValueError("detector_unresolved")
                elif t == max(detector_by_time):
                    stage1.update(status="no_alarm")
                    terminal_report = run_two_stage_report(stage1, context=context)
                    break
            if terminal_report is not None:
                break
            # Do not execute any source decision unless an alarm is frozen.
            if alarm_time is not None and alarm_time > source_start:
                raise ValueError("alarm_after_fixed_source_start")
            if t == deadline:
                break
            if t in set(calendar.get("refresh_times", [])):
                frame = pd.DataFrame(behavior_rows)
                try:
                    if not fitted:
                        model.fit(frame, as_of=t, window=policy_spec["fit_window"], feasible=feasible)
                        fitted = True
                    else:
                        model.update(frame, as_of=t, window=policy_spec["fit_window"], feasible=feasible)
                    trace["fit_events"].append({"as_of": t, "status": "ready", "rows": len(frame),
                        "estimator_config": model.estimator_config_.as_dict(),
                        "source_measurement": "existing linear predictable-G source diagnostic"})
                except Exception as exc:
                    trace["fit_events"].append({"as_of": t, "status": "failed", "reason": str(exc)})
                    raise ValueError("native_fit_failed:" + str(exc)) from exc
            if t < start:
                continue
            if len(ack_by_id) != len(releases_by_id):
                raise ValueError("previous_execution_outcome_not_released")
            state = host.state(as_of=t)
            if not isinstance(state, Mapping) or state.get("status") != "ready":
                raise ValueError("host_state_unavailable")
            leaked_state = sorted(_FORBIDDEN_RECEIPT.intersection(state))
            if leaked_state:
                raise ValueError("host_state_contains_outcomes:" + ",".join(leaked_state))
            if _integer(state.get("available_at"), "state.available_at") > t:
                raise ValueError("future_host_state")
            x = np.asarray(state.get("state"), dtype=float)
            if x.shape != (3,) or not np.isfinite(x).all():
                raise ValueError("host_state_must_be_finite_physical_w_previous_action")
            last = releases_by_id.get(next(reversed(releases_by_id), None)) if releases_by_id else None
            if last is not None:
                expected = [last["next_physical"], last["next_w"], last["action"]]
                if not np.allclose(x, expected, rtol=0, atol=1e-12):
                    raise ValueError("host_state_disagrees_with_latest_released_transition")
            if not fitted:
                raise ValueError("native_model_not_fitted_before_action")
            raw_host_feasible = state.get("feasible_mask", feasible.tolist())
            if (not isinstance(raw_host_feasible, (list, tuple, np.ndarray))
                    or any(type(v) not in (bool, np.bool_) for v in raw_host_feasible)):
                raise ValueError("host feasible_mask entries must be booleans")
            host_feasible = np.asarray(raw_host_feasible, dtype=bool)
            if host_feasible.shape != (11,):
                raise ValueError("host feasible_mask must have 11 action entries")
            current_feasible = feasible & host_feasible
            query = eta_qualified_native_action(model, [x], domain, feasible=current_feasible,
                remaining_horizon=policy_spec.get("horizon"))
            if query.get("status") not in {"ready", "partial"}:
                raise ValueError("eta_qualified_native_query_unavailable:" + str(query.get("reason")))
            allowed = [a for a, ok in zip(model.action_grid_, query["qualified_mask"][0]) if ok]
            if not allowed:
                raise ValueError("no_eta_feasible_native_action")
            q_action = query["actions"][0]
            # The epsilon draw starts from the eta-qualified mask, then is
            # restricted to cells with an actual native Q value before it can
            # be supplied as the recommendation helper's behavior snapshot.
            native_supported = [a for a, ok in zip(model.action_grid_, query["native_mask"][0]) if ok]
            if not native_supported:
                raise ValueError("no_eta_feasible_native_Q_cell")
            behavior_action = float(rng.choice(native_supported)) if rng.random() < float(policy_spec["epsilon"]) else float(q_action)
            acquisition = recommend_source_action(action_grid=model.action_grid_.tolist(), qualified=query,
                mode=policy_spec.get("acquisition_mode", "passive"),
                stage1_status="alarm" if alarm_time is not None else "no_alarm",
                alarm_release=alarm_time, decision_time=t, source_start_decision=source_start,
                source_end_decision=source_deadline-1,
                completed_action_count=sum(1 for a in trace["action_trace"] if a.get("status") == "acknowledged"),
                behavior_action=behavior_action)
            if acquisition.get("status") != "ready":
                raise ValueError("source_action_recommendation_unavailable:" + str(acquisition.get("reason")))
            action = float(acquisition["action"])
            predictable_weights = _bin_weights(action)
            did = f"online/{online_path}/{t}"
            proposal = {"decision_id": did, "decision_time": t, "state": x.tolist(),
                        "action": action, "behavior_action": behavior_action,
                        "source_probe": bool(acquisition.get("probe")),
                        "eta_support_status": query.get("status"),
                        "q_suggested_action": q_action,
                        "qualified_mask": query.get("qualified_mask"),
                        "native_mask": query.get("native_mask"),
                        "action_recommendation": deepcopy(acquisition),
                        "predictable_bin_weights": predictable_weights}
            trace["action_trace"].append({**proposal, "status": "proposed"})
            try:
                receipt = host.step(deepcopy(proposal))
            except Exception as exc:
                trace["action_trace"][-1].update(status="execution_unconfirmed", reason=str(exc))
                raise ValueError("host_step_execution_unconfirmed:" + str(exc)) from exc
            if not isinstance(receipt, Mapping):
                trace["action_trace"][-1].update(status="execution_unconfirmed", reason="invalid_receipt")
                raise ValueError("host_step_execution_unconfirmed:invalid_receipt")
            receipt = dict(receipt)
            bad = sorted(_FORBIDDEN_RECEIPT.intersection(receipt))
            if bad:
                trace["action_trace"][-1].update(status="execution_unconfirmed", reason="receipt_contains_outcomes")
                raise ValueError("receipt_contains_outcomes:" + ",".join(bad))
            rid = receipt.get("execution_id")
            try:
                receipt_valid = (receipt.get("status") == "executed" and isinstance(rid, str) and bool(rid)
                    and receipt.get("decision_id") == did
                    and _integer(receipt.get("decision_time"), "receipt.decision_time") == t
                    and _integer(receipt.get("executed_at"), "receipt.executed_at") == t
                    and float(receipt.get("executed_action")) == action)
            except Exception:
                receipt_valid = False
            if not receipt_valid:
                trace["action_trace"][-1].update(status="execution_unconfirmed", reason="receipt_mismatch")
                raise ValueError("host_step_execution_unconfirmed:receipt_mismatch")
            if rid in used_execution_ids:
                trace["action_trace"][-1].update(status="execution_unconfirmed", reason="duplicate_execution_id")
                raise ValueError("duplicate_execution_id")
            used_execution_ids.add(rid)
            prev = float(x[2]); weights = predictable_weights
            ack_by_id[did] = {"decision_id": did, "decision_time": t, "state": x.tolist(),
                "action": action, "previous_action": prev, "weights": weights,
                "execution_id": rid, "record_id": None}
            trace["action_trace"][-1].update(status="acknowledged", execution_id=rid,
                                             executed_at=t, executed_action=action)
            # Acknowledgement alone is not an observation; release_at stores outcomes later.
        if terminal_report is not None:
            terminal_report["host_loop"] = trace
            return terminal_report
        if alarm_time is None:
            stage1["status"] = "unresolved" if stage1["status"] != "no_alarm" else "no_alarm"
            if stage1["status"] == "no_alarm":
                terminal_report = run_two_stage_report(stage1, context=context)
                terminal_report["host_loop"] = trace
                return terminal_report
            return _unresolved(stage1, context, "detector_calendar_incomplete", trace)
        source_rows = [releases_by_id.get(f"online/{online_path}/{t}") for t in range(source_start, source_deadline)]
        if any(x is None for x in source_rows):
            trace["source"].update(status="incomplete", observed_n=sum(x is not None for x in source_rows))
            return _unresolved(stage1, context, "incomplete_fixed_source_window", trace)
        trace["source"].update(status="complete", observed_n=len(source_rows))
        block = {"states": [[x["state"][0], x["state"][2]] for x in source_rows],
                 "actions": [x["action"] for x in source_rows],
                 "weights": np.asarray([x["weights"] for x in source_rows], dtype=float).T.tolist(), "intended_n": source_n,
                 "record_ids": [x["record_id"] for x in source_rows],
                 "dates": [x["outcome_release_time"] for x in source_rows],
                 "next_physical": [[x["next_physical"]] for x in source_rows]}
        qualified, support = _qualified_block(block, domain)
        trace["source"].update(diagnostic_block={**block, "weights": qualified["weights"]},
                               eta_support=support)
        future = {"states": block["states"],
                  "actions": block["actions"],
                  "next_physical": block["next_physical"],
                  "weights": qualified["weights"],
                  "dates": [_integer(x["outcome_release_time"], "outcome_release_time") for x in source_rows],
                  "record_ids": block["record_ids"], "weights_predictable": True,
                  "window_fixed_at_alarm": True, "intended_n": source_n}
        out = run_predictable_gq_two_stage(stage1=stage1, reference=source_ref, future=future,
            sigma=source_spec["sigma"], alpha_reference=source_spec["alpha_reference"],
            alpha_noise=source_spec["alpha_noise"], calendar_size=1,
            source_deadline=source_deadline, latent_bound=source_spec.get("latent_bound", .6),
            model_bound=source_spec.get("model_bound", 2.), assumptions=source_spec.get("assumptions"))
        out["context"].update(context)
        out["host_loop"] = trace
        out["host_loop"]["source"]["eta_support"] = support
        return out
    except Exception as exc:
        if alarm_time is not None:
            trace["source"].update(status="incomplete")
            stage1.update(status="alarm", alarm_time=alarm_time, first_alarm=alarm_time)
        elif stage1["status"] not in {"no_alarm"}:
            stage1["status"] = "unresolved"
        return _unresolved(stage1, context, str(exc), trace)


__all__ = ["run_native_source_loop"]
