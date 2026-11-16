"""Finite-candidate source evidence using the public residual-energy statistic.

This module deliberately contains no likelihood, optimizer, or data generator.
It consumes complete public paths and a fixed candidate schedule, then routes
the resulting evidence through the common source diagnosis state machine.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .diagnosis import diagnose_sources
from .residual_energy import (envelope_eligibility, rank_cutoff, sequential_slope_energy,
                              sequential_slope_energy_compact, sequential_slope_energy_blocked,
                              prepare_slope_energy_prefixes, sequential_slope_energy_prepared)


def _array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _path_residuals(paths: np.ndarray, reference: Mapping[str, Any], candidate_f: np.ndarray,
                    candidate_d: np.ndarray, start: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if paths.ndim < 3 or paths.shape[-1] != candidate_f.shape[-1]:
        raise ValueError("paths must have leading dimensions, time, and state coordinates")
    if paths.shape[-2] < start + horizon + 1:
        raise ValueError("paths do not cover monitor_start plus horizon transitions")
    f_ref = _array(reference["F"], "reference_coefficients['F']", 2)
    d_ref = _array(reference["d"], "reference_coefficients['d']", 1)
    q, p = candidate_f.shape[-2:]
    if f_ref.shape != (q, p) or d_ref.shape != (q,):
        raise ValueError("reference coefficients have incompatible shapes")
    if candidate_d.shape != (horizon, q) or candidate_f.shape != (horizon, q, p):
        raise ValueError("candidate schedule must cover exactly the monitored horizon")
    x = paths[..., start:start + horizon, :]
    xn = paths[..., start + 1:start + horizon + 1, :]
    if x.shape[-1] != p or xn.shape[-1] != q:
        raise ValueError("path state width must match candidate predictor and response widths")
    ref_e = xn - np.einsum("...tp,pq->...tq", x, f_ref.T) - d_ref
    delta_f = candidate_f - f_ref[None, :, :]
    delta_d = candidate_d - d_ref[None, :]
    mu = np.einsum("...tp,tqp->...tq", x, delta_f) + delta_d
    adjusted = ref_e - mu
    # The explicit expression above keeps the public definition visible:
    # adjusted residual = e_ref - [(F_candidate-F_ref)x + (d_candidate-d_ref)].
    return adjusted, x, xn


def score_residual_candidate(
    evaluation_full_paths: Any,
    calibration_full_paths: Any,
    *,
    reference_coefficients: Mapping[str, Any],
    candidate_F: Any,
    candidate_d: Any,
    monitor_start: int,
    horizon: int,
    residual_left_inverse: Any,
    support: Any,
    etas: Any,
    weight_index: int,
    alpha: float = 0.05,
    return_trace: bool = True,
) -> dict[str, Any]:
    """Score one complete candidate law for all fixed eta values."""
    if isinstance(monitor_start, bool) or not isinstance(monitor_start, (int, np.integer)) or monitor_start < 0:
        raise ValueError("monitor_start must be a nonnegative integer")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if isinstance(weight_index, bool) or not isinstance(weight_index, (int, np.integer)):
        raise ValueError("weight_index must be an integer")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    ev = _array(evaluation_full_paths, "evaluation_full_paths")
    cal = _array(calibration_full_paths, "calibration_full_paths")
    if ev.ndim < 3 or cal.ndim != 3:
        raise ValueError("evaluation and calibration paths must have time and state dimensions")
    f = _array(candidate_F, "candidate_F", 3)
    d = _array(candidate_d, "candidate_d", 2)
    inv = _array(residual_left_inverse, "residual_left_inverse", 2)
    adjusted_eval, predictors_eval, _ = _path_residuals(ev, reference_coefficients, f, d, int(monitor_start), int(horizon))
    adjusted_cal, predictors_cal, _ = _path_residuals(cal, reference_coefficients, f, d, int(monitor_start), int(horizon))
    q = adjusted_eval.shape[-1]
    if inv.shape != (q, q):
        raise ValueError("residual_left_inverse has the wrong shape")
    if weight_index < 0 or weight_index >= predictors_eval.shape[-1]:
        raise ValueError("weight_index is outside the predictor width")
    eval_flat = adjusted_eval.reshape((-1, int(horizon), q))
    pred_eval = predictors_eval.reshape((-1, int(horizon), predictors_eval.shape[-1]))
    cal_flat = adjusted_cal.reshape((-1, int(horizon), q))
    pred_cal = predictors_cal.reshape((-1, int(horizon), predictors_cal.shape[-1]))
    if cal_flat.shape[0] < 1:
        raise ValueError("calibration_full_paths must contain at least one path")
    eta_array = np.asarray(etas, dtype=float)
    if eta_array.ndim != 1 or not len(eta_array) or not np.isfinite(eta_array).all() \
            or np.any(eta_array < 0) or len(np.unique(eta_array)) != len(eta_array):
        raise ValueError("etas must be a distinct one-dimensional finite nonnegative array")
    eval_eligible = envelope_eligibility(pred_eval, support, eta_array)
    cal_eligible = envelope_eligibility(pred_cal, support, eta_array)
    eval_result = sequential_slope_energy(
        eval_flat, pred_eval[..., weight_index], residual_left_inverse=inv,
        eligible=eval_eligible, return_trace=bool(return_trace),
    )
    cal_result = sequential_slope_energy(
        cal_flat, pred_cal[..., weight_index], residual_left_inverse=inv,
        eligible=cal_eligible, return_trace=False,
    )
    cut = rank_cutoff(cal_result["scores"], alpha=alpha)
    eval_leading = adjusted_eval.shape[:-2]
    score = eval_result["scores"].reshape(eval_leading + (len(eta_array),))
    strict = score > cut.reshape((1,) * len(eval_leading) + (len(eta_array),))
    trace = eval_result.get("score_trace")
    first_flat = np.full(eval_result["scores"].shape, int(horizon), dtype=np.int64)
    if trace is not None:
        crosses = trace > cut[None, None, :]
        for i in range(crosses.shape[0]):
            for r in range(crosses.shape[2]):
                hit = np.flatnonzero(crosses[i, :, r])
                if hit.size:
                    first_flat[i, r] = int(hit[0])
    first = first_flat.reshape(eval_leading + (len(eta_array),))
    first_release = np.where(first < int(horizon), int(monitor_start) + 1 + first, -1)
    score_trace = None if trace is None else trace.reshape(eval_leading + (int(horizon), len(eta_array)))
    eligible_trace = None if eval_result.get("eligible_trace") is None else eval_result["eligible_trace"].reshape(
        eval_leading + (int(horizon), len(eta_array))
    )
    excluded_trace = None if score_trace is None else score_trace > cut.reshape(
        (1,) * len(eval_leading) + (1, len(eta_array))
    )
    out = {
        "status": "ready", "etas": eta_array,
        "cutoffs": cut, "evaluation_scores": score,
        "strict_excluded": strict, "first_exclusion": first,
        "calibration_scores": cal_result["scores"],
        "calibration_count": int(cal_flat.shape[0]),
        "evaluation_valid_count": eval_result["valid_count"].reshape(eval_leading + (len(eta_array),)),
        "evaluation_first_exit": eval_result["first_exit"].reshape(eval_leading + (len(eta_array),)),
        "statistic": "sequential_slope_energy",
        "calibration_scope": "candidate_specific_complete_paths", "Q": "not_applicable",
    }
    if return_trace:
        out["score_trace"] = score_trace
        out["eligible_trace"] = eligible_trace
        out["evaluation_excluded_trace"] = excluded_trace
        out["evaluation_first_exclusion_release"] = first_release
    out["arrays"] = {
        "calibration_scores": out["calibration_scores"],
        "calibration_first_exit": cal_result["first_exit"],
        "calibration_valid_count": cal_result["valid_count"],
        "evaluation_scores": out["evaluation_scores"],
        "thresholds": out["cutoffs"],
        "etas": out["etas"],
        "evaluation_valid_count": out["evaluation_valid_count"],
        "evaluation_first_exit": out["evaluation_first_exit"],
        "strict_excluded": out["strict_excluded"],
    }
    if return_trace:
        out["arrays"].update({
            "evaluation_score_trace": out["score_trace"],
            "evaluation_eligible_trace": out["eligible_trace"],
            "evaluation_excluded_trace": out["evaluation_excluded_trace"],
            "evaluation_first_exclusion_release": out["evaluation_first_exclusion_release"],
        })
    return out


def _compact_candidate_energy(paths, *, reference_coefficients, candidate_F, candidate_d,
                              monitor_start, horizon, residual_left_inverse, support,
                              etas, weight_index, thresholds=None,
                              implementation='shared_prefix_compact', block_size=128):
    if isinstance(monitor_start, bool) or not isinstance(monitor_start, (int, np.integer)) or monitor_start < 0:
        raise ValueError('monitor_start must be a nonnegative integer')
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    if isinstance(weight_index, bool) or not isinstance(weight_index, (int, np.integer)):
        raise ValueError('weight_index must be an integer')
    path_array=_array(paths,'paths')
    f=_array(candidate_F,'candidate_F',3); d=_array(candidate_d,'candidate_d',2)
    adjusted,predictors,_=_path_residuals(path_array,reference_coefficients,f,d,int(monitor_start),int(horizon))
    if not 0 <= weight_index < predictors.shape[-1]:
        raise ValueError('weight_index is outside the predictor width')
    eta=np.asarray(etas,dtype=float)
    if eta.ndim!=1 or not len(eta) or not np.isfinite(eta).all() or np.any(eta<0) or len(np.unique(eta))!=len(eta):
        raise ValueError('etas must be a distinct one-dimensional finite nonnegative array')
    leading=adjusted.shape[:-2]
    errors=adjusted.reshape((-1,int(horizon),adjusted.shape[-1]))
    pred=predictors.reshape((-1,int(horizon),predictors.shape[-1]))
    if not len(errors): raise ValueError('paths must contain at least one path')
    eligible=envelope_eligibility(pred,support,eta)
    if implementation=='shared_prefix_compact':
        result=sequential_slope_energy_compact(errors,pred[...,weight_index],
            residual_left_inverse=residual_left_inverse,eligible=eligible,cutoffs=thresholds)
    elif implementation=='numpy_prefix_blocks':
        result=sequential_slope_energy_blocked(errors,pred[...,weight_index],
            residual_left_inverse=residual_left_inverse,eligible=eligible,cutoffs=thresholds,block_size=block_size)
    else: raise ValueError('unknown residual energy implementation')
    return result,leading,eta


def calibrate_residual_candidate(calibration_full_paths, *, reference_coefficients,
                                 candidate_F, candidate_d, monitor_start, horizon,
                                 residual_left_inverse, support, etas, weight_index,
                                 alpha=0.05, implementation='shared_prefix_compact', block_size=128):
    """Compute one candidate's fixed calibration once, without time traces.

    Reuse requires unchanged candidate law, calibration paths, reference,
    whitening, support, eta rules, monitor interval and alpha. The caller owns
    that experiment specification; this function does not infer or cache it.
    """
    cal=_array(calibration_full_paths,'calibration_full_paths',3)
    result,_,eta=_compact_candidate_energy(cal,reference_coefficients=reference_coefficients,
        candidate_F=candidate_F,candidate_d=candidate_d,monitor_start=monitor_start,horizon=horizon,
        residual_left_inverse=residual_left_inverse,support=support,etas=etas,weight_index=weight_index,
        implementation=implementation,block_size=block_size)
    cuts=rank_cutoff(result['scores'],alpha=alpha)
    arrays={'calibration_scores':result['scores'],'calibration_terminal_energy':result['terminal_energy'],
            'calibration_first_exit':result['first_exit'],'calibration_valid_count':result['valid_count'],
            'thresholds':cuts,'etas':eta}
    return {'status':'ready','arrays':arrays,'calibration_count':len(cal),
            'calibration_scope':'candidate_specific_complete_paths','statistic':'sequential_slope_energy',
            'implementation':implementation,'Q':'not_applicable'}


def evaluate_residual_candidate(evaluation_full_paths, *, thresholds, reference_coefficients,
                                candidate_F, candidate_d, monitor_start, horizon,
                                residual_left_inverse, support, etas, weight_index,
                                implementation='shared_prefix_compact', block_size=128):
    """Evaluate a path block with fixed candidate cutoffs; always retain dates.

    First-exclusion dates use absolute releases, with -1 for no crossing in a
    successfully computed path. Missing/failed candidates must be represented
    separately as unavailable, never synthesized as an all--1 result. The
    cumulative exclusion stays true after exit; publication still needs the
    ordinary alarm and active-support gates in the shared source router.
    """
    result,leading,eta=_compact_candidate_energy(evaluation_full_paths,
        reference_coefficients=reference_coefficients,candidate_F=candidate_F,candidate_d=candidate_d,
        monitor_start=monitor_start,horizon=horizon,residual_left_inverse=residual_left_inverse,
        support=support,etas=etas,weight_index=weight_index,thresholds=thresholds,
        implementation=implementation,block_size=block_size)
    shape=leading+(len(eta),)
    first=result['first_crossing'].reshape(shape)
    score=result['scores'].reshape(shape); cuts=np.asarray(thresholds,dtype=float)
    arrays={'evaluation_scores':score,'evaluation_terminal_energy':result['terminal_energy'].reshape(shape),
            'evaluation_valid_count':result['valid_count'].reshape(shape),
            'evaluation_first_exit':result['first_exit'].reshape(shape),'strict_excluded':score>cuts,
            'evaluation_first_exclusion_release':np.where(first<int(horizon),first+int(monitor_start)+1,-1),
            'evaluation_first_exclusion_score':result['first_crossing_score'].reshape(shape),
            'thresholds':cuts,'etas':eta}
    return {'status':'ready','arrays':arrays,'statistic':'sequential_slope_energy',
            'implementation':implementation,'trace_storage':'first_exclusion_and_first_exit',
            'Q':'not_applicable'}


def prepare_residual_evaluation(evaluation_full_paths, *, reference_coefficients,
                                monitor_start, horizon, residual_left_inverse,
                                support, etas, weight_index):
    """Cache one public EVAL block's candidate-independent arrays.

    The cache belongs to these exact input paths and this reference/support/
    whitening/monitor specification. It is not shared across candidate-specific
    CAL paths or changed experiment settings. No candidates or truth enter it.
    """
    if isinstance(monitor_start,bool) or not isinstance(monitor_start,(int,np.integer)) or monitor_start<0:
        raise ValueError('monitor_start must be a nonnegative integer')
    if isinstance(horizon,bool) or not isinstance(horizon,(int,np.integer)) or horizon<1:
        raise ValueError('horizon must be a positive integer')
    paths=_array(evaluation_full_paths,'evaluation_full_paths')
    f=_array(reference_coefficients['F'],'reference F',2);d=_array(reference_coefficients['d'],'reference d',1)
    # The reference candidate makes mu exactly zero and preserves the existing
    # explicit e_ref subtraction order used by every candidate scorer.
    schedule=np.broadcast_to(f,(horizon,)+f.shape);intercepts=np.broadcast_to(d,(horizon,)+d.shape)
    residuals,predictors,_=_path_residuals(paths,reference_coefficients,schedule,intercepts,monitor_start,horizon)
    if isinstance(weight_index,bool) or not isinstance(weight_index,(int,np.integer)) or not 0<=weight_index<predictors.shape[-1]:
        raise ValueError('weight_index is outside the predictor width')
    eta=np.asarray(etas,dtype=float)
    if eta.ndim!=1 or not len(eta) or not np.isfinite(eta).all() or np.any(eta<0) or len(np.unique(eta))!=len(eta):
        raise ValueError('etas must be distinct finite nonnegative values')
    pred=predictors.reshape((-1,horizon,predictors.shape[-1]))
    if not len(pred):raise ValueError('paths must contain at least one path')
    inverse=_array(residual_left_inverse,'residual_left_inverse',2)
    if inverse.shape!=(residuals.shape[-1],residuals.shape[-1]):raise ValueError('whitening dimensions disagree')
    prefixes=prepare_slope_energy_prefixes(pred[...,weight_index],envelope_eligibility(pred,support,eta))
    return {'predictors':predictors,'reference_residuals':residuals,
            'reference_F':f.copy(),'reference_d':d.copy(),'residual_left_inverse':inverse.copy(),
            'prefixes':prefixes,'etas':eta.copy(),'monitor_start':int(monitor_start),'horizon':int(horizon)}


def evaluate_prepared_residual_candidate(prepared, *, candidate_F, candidate_d,
                                         thresholds, block_size=128):
    """Evaluate one candidate against a successfully prepared public block."""
    x=prepared['predictors'];ref_e=prepared['reference_residuals'];horizon=prepared['horizon']
    f=_array(candidate_F,'candidate_F',3);d=_array(candidate_d,'candidate_d',2)
    if f.shape!=(horizon,ref_e.shape[-1],x.shape[-1]) or d.shape!=(horizon,ref_e.shape[-1]):
        raise ValueError('candidate schedule does not match the prepared block')
    delta_f=f-prepared['reference_F'][None,:,:];delta_d=d-prepared['reference_d'][None,:]
    mu=np.einsum('...tp,tqp->...tq',x,delta_f)+delta_d
    adjusted=ref_e-mu
    result=sequential_slope_energy_prepared(adjusted.reshape((-1,horizon,adjusted.shape[-1])),
        prepared=prepared['prefixes'],residual_left_inverse=prepared['residual_left_inverse'],
        cutoffs=thresholds,block_size=block_size)
    eta=prepared['etas'];shape=adjusted.shape[:-2]+(len(eta),)
    first=result['first_crossing'].reshape(shape);scores=result['scores'].reshape(shape)
    arrays={'evaluation_scores':scores,'evaluation_terminal_energy':result['terminal_energy'].reshape(shape),
            'evaluation_first_exit':result['first_exit'].reshape(shape),'evaluation_valid_count':result['valid_count'].reshape(shape),
            'strict_excluded':scores>thresholds,
            'evaluation_first_exclusion_release':np.where(first<horizon,first+prepared['monitor_start']+1,-1),
            'evaluation_first_exclusion_score':result['first_crossing_score'].reshape(shape),
            'etas':eta,'thresholds':np.asarray(thresholds,dtype=float)}
    return {'status':'ready','arrays':arrays,'statistic':'sequential_slope_energy',
            'implementation':'prepared_numpy_prefix_blocks','Q':'not_applicable'}


def route_residual_candidates(candidate_evidence: Mapping[str, Any], family_members: Mapping[str, Any],
                             ordinary_alarm: Any, active: bool, release: Any) -> dict[str, Any]:
    """Route one path/time/eta candidate table through ``diagnose_sources``."""
    if not isinstance(candidate_evidence, Mapping) or not isinstance(family_members, Mapping):
        raise TypeError("candidate_evidence and family_members must be mappings")
    if not isinstance(active, (bool, np.bool_)):
        raise ValueError("active must be boolean")
    statuses = {}
    for name, item in candidate_evidence.items():
        if not isinstance(item, Mapping) or item.get("status") not in {"compatible", "excluded", "unresolved", "unavailable"}:
            statuses[name] = {"status": "unresolved", "reason": "missing_or_invalid_candidate_evidence"}
        else:
            statuses[name] = dict(item)

    def family_item(class_name: str) -> dict[str, Any]:
        names = family_members.get(class_name, [])
        if isinstance(names, str):
            names = [names]
        names = list(names)
        if not names:
            return {"status": "unresolved", "description": f"{class_name} has no declared candidates", "reason": "empty_family"}
        items = [statuses.get(n, {"status": "unresolved", "reason": "candidate_not_returned"}) for n in names]
        if any(x["status"] == "compatible" for x in items):
            status = "compatible"
        elif all(x["status"] == "excluded" for x in items):
            status = "excluded"
        elif any(x["status"] in {"unresolved", "unavailable"} for x in items):
            status = "unresolved"
        else:
            status = "unresolved"
        return {"status": status, "description": f"{class_name} candidate evidence is {status}",
                "candidates": items, "release": release}

    evidence = {name: family_item(name) for name in ("C0", "CL", "CM", "CMIX")}
    alarm_is_true = isinstance(ordinary_alarm, (bool, np.bool_)) and bool(ordinary_alarm)
    gate_open = bool(active) and alarm_is_true
    router = diagnose_sources(
        lambda name: evidence[name], alarm={"release": release, "detected": ordinary_alarm},
        available=None if gate_open else [], goal="full",
    )
    cmix_witness = any(
        statuses.get(name, {}).get("status") == "compatible"
        for name in (list(family_members.get("CMIX", [])) if not isinstance(family_members.get("CMIX", []), str)
                     else [family_members["CMIX"]])
    )
    outcome = router["outcome"]
    opposite_pure_excluded = (
        outcome == "both_required" and evidence["CL"]["status"] == "excluded" and evidence["CM"]["status"] == "excluded"
    ) or (outcome == "model_required" and evidence["CL"]["status"] == "excluded") \
        or (outcome == "latent_required" and evidence["CM"]["status"] == "excluded")
    report_allowed = gate_open and cmix_witness and opposite_pure_excluded
    return {"status": "ready" if active else "unavailable", "ordinary_alarm": ordinary_alarm,
            "ordinary_gate_open": gate_open, "report_allowed": report_allowed,
            "release": release, "family_evidence": evidence,
            "candidate_evidence": statuses, "router": router}


def batch_route_residual_candidates(
    excluded: Any,
    candidate_ids: Any,
    family_members: Mapping[str, Any],
    ordinary_alarm: Any,
    active: Any,
    available: Any = None,
    release: Any = None,
) -> dict[str, Any]:
    """Vectorized family routing for candidate status panels.

    ``excluded`` is boolean with candidate axis first and arbitrary trailing
    path/time/eta axes.  A false available candidate is unresolved, never
    compatible.  Family codes are 0=compatible, 1=excluded, 2=unresolved.
    The finite router calls are made only for distinct family/gate states.
    """
    panel = np.asarray(excluded)
    if panel.dtype != np.bool_:
        raise ValueError("excluded must contain boolean evidence")
    if panel.ndim < 2:
        raise ValueError("excluded must have candidate and panel dimensions")
    ids = list(candidate_ids)
    if len(ids) != panel.shape[0] or len(set(ids)) != len(ids):
        raise ValueError("candidate_ids must uniquely name the candidate axis")
    index = {name: i for i, name in enumerate(ids)}
    if not isinstance(family_members, Mapping):
        raise TypeError("family_members must be a mapping")
    families = ("C0", "CL", "CM", "CMIX")
    member_indices = {}
    missing_members = {}
    for family in families:
        raw = family_members.get(family, [])
        if isinstance(raw, str):
            raw = [raw]
        if not raw or len(set(raw)) != len(raw):
            raise ValueError(f"{family} must declare distinct nonempty candidates")
        member_indices[family] = [index[name] for name in raw if name in index]
        missing_members[family] = any(name not in index for name in raw)
    tail = panel.shape[1:]
    if np.asarray(ordinary_alarm).shape not in {(), tail} or np.asarray(active).shape not in {(), tail}:
        raise ValueError("ordinary_alarm and active must be scalars or match the panel shape")
    if np.asarray(ordinary_alarm).dtype != np.bool_ or np.asarray(active).dtype != np.bool_:
        raise ValueError("ordinary_alarm and active must contain boolean evidence")
    alarm = np.broadcast_to(np.asarray(ordinary_alarm), tail)
    live = np.broadcast_to(np.asarray(active), tail)
    if available is None:
        avail = np.ones(panel.shape[0], dtype=bool)
    else:
        avail = np.asarray(available)
        if avail.dtype != np.bool_:
            raise ValueError("available must contain boolean evidence")
        if avail.shape != (panel.shape[0],):
            raise ValueError("available must have one boolean per candidate")
    codes = {}
    witness = np.full(tail, -1, dtype=np.int64)
    for family in families:
        members = np.asarray(member_indices[family], dtype=np.int64)
        member_available = avail[members]
        member_excluded = panel[members]
        compatible = np.any((~member_excluded) & member_available[:, None].reshape((len(members),) + (1,) * len(tail)), axis=0)
        excluded_all = np.all(member_excluded & member_available[:, None].reshape((len(members),) + (1,) * len(tail)), axis=0)
        code = np.full(tail, 2, dtype=np.int8)
        code[compatible] = 0
        code[excluded_all & ~np.bool_(missing_members[family])] = 1
        codes[family] = code
        if family == "CMIX":
            for candidate in members:
                present = (~panel[candidate]) & avail[candidate] & (witness < 0)
                witness[present] = int(candidate)

    # Diagnose only unique family/gate states; raw arrays remain compact.
    state_fields = np.stack([codes[name].reshape(-1) for name in families] +
                            [alarm.reshape(-1).astype(np.int8), live.reshape(-1).astype(np.int8)], axis=1)
    unique, inverse = np.unique(state_fields, axis=0, return_inverse=True)
    outcomes = np.empty(len(unique), dtype="U32")
    for i, state in enumerate(unique):
        family_evidence = {
            name: {"status": ("compatible" if state[j] == 0 else "excluded" if state[j] == 1 else "unresolved"),
                   "description": f"{name} batch evidence"}
            for j, name in enumerate(families)
        }
        gate_open = bool(state[4]) and bool(state[5])
        routed = diagnose_sources(lambda name: family_evidence[name],
                                  alarm={"release": release, "detected": bool(state[4])},
                                  available=None if gate_open else [], goal="full")
        outcomes[i] = routed["outcome"]
    outcome_array = outcomes[inverse].reshape(tail)
    cmix = codes["CMIX"] == 0
    model_necessary = cmix & (codes["CL"] == 1) & alarm & live
    latent_necessary = cmix & (codes["CM"] == 1) & alarm & live
    both_necessary = cmix & (codes["CL"] == 1) & (codes["CM"] == 1) & alarm & live
    return {
        "family_codes": codes,
        "joint_witness_index": witness,
        "router_outcome": outcome_array,
        "model_necessary": model_necessary,
        "latent_necessary": latent_necessary,
        "both_necessary": both_necessary,
        "report_allowed": model_necessary | latent_necessary | both_necessary,
        "status_code_legend": {"compatible": 0, "excluded": 1, "unresolved": 2},
    }
