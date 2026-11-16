"""Pure branch-specific receipt compaction and first-39 bank state.

This module consumes already fabricated or externally produced receipts. It
does not launch episodes, assign seeds, load a DGP, or calibrate H.
"""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MEASUREMENTS_PATH = HERE.parent / "fixed_action_probe_dev_attempt001" / "measurements.py"
RUNTIME_SOURCE = HERE.parent / "persistent_panel_frontier_attempt001" / "runtime_source"
sys.path.insert(0, str(RUNTIME_SOURCE))
_spec = importlib.util.spec_from_file_location("branch_existing_measurements", MEASUREMENTS_PATH)
measurements = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measurements)

CELLS = ("N", "CL", "CM", "MIX")
TAILS = {"N": ("upper", "upper"), "CL": ("upper", "lower"),
         "CM": ("lower", "upper"), "MIX": ("lower", "lower")}
LOW_ACTION = 0.02
HIGH_ACTION_MAX = 0.08
DATES = tuple(range(40, 48))
TARGET = 39
CAP = 128
ENDPOINT_STATES = {"complete", "support_refusal", "main_unavailable", "not_run_no_alarm",
                   "not_run_monitor_unavailable"}


class IntegrityFailure(RuntimeError):
    """Malformed or numerically invalid completed receipt; never a U row."""


def _endpoint_identity(spec, endpoints):
    if len(endpoints) != 16:
        raise ValueError("endpoint allocation must contain two fixed slots on eight dates")
    for t in DATES:
        rows=[e for e in endpoints if e.get('time')==t]
        low=[e for e in rows if e.get('action')==LOW_ACTION]
        high=[e for e in rows if e.get('action') is None or LOW_ACTION < e.get('action',-1) <= HIGH_ACTION_MAX]
        if len(rows)!=2 or len(low)!=1 or len(high)!=1 or low[0] is high[0]:
            raise ValueError("each date requires the fixed .02 slot and one selected-high slot")
        selected=high[0]
        if selected.get('action') is not None and selected.get('selected_high_action')!=selected.get('action'):
            raise ValueError("selected high endpoint label differs from its pre-outcome selection")
    if any(e.get('seed') != spec['seed'] or e.get('cell') != spec['cell'] for e in endpoints):
        raise ValueError("endpoint seed/cell identity mismatch")


def _endpoint_qualified(e):
    if e.get("status") != "complete":
        return False
    if e.get("physical_action_supported") is not True or e.get("public_action_supported") is not True or e.get("finite_executed_Q") is not True:
        raise IntegrityFailure("completed endpoint lacks support or finite executed Q")
    try:
        q_value = float(e["executed_Q"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityFailure("completed endpoint executed Q is missing or nonnumeric") from exc
    if not math.isfinite(q_value):
        raise IntegrityFailure("completed endpoint executed Q is nonfinite")
    if e.get("release_time") != e["time"] + 1:
        raise IntegrityFailure("completed endpoint has wrong release clock")
    if not all(math.isfinite(float(e.get(k, float("nan")))) for k in ("full_x_error", "full_w_error")):
        raise IntegrityFailure("completed endpoint has nonfinite residual")
    issued = e.get("issued")
    if not isinstance(issued, dict):
        raise IntegrityFailure("completed endpoint is missing issued receipt")
    main = issued.get("main")
    if not isinstance(main, dict):
        raise IntegrityFailure("completed endpoint is missing main issue")
    if issued.get("time") != e["time"] or issued.get("fit_as_of") != e["time"]:
        raise IntegrityFailure("completed endpoint issue date/vintage mismatch")
    if main.get("fit_as_of") != e["time"] or main.get("action") != e["action"]:
        raise IntegrityFailure("completed endpoint main issue vintage/action mismatch")
    if e.get("action") != main.get("action"):
        raise IntegrityFailure("completed endpoint action mismatch")
    prediction = main.get("full_physical_prediction")
    outcome = e.get("next_physical")
    if not isinstance(prediction, (list, tuple)) or not isinstance(outcome, (list, tuple)) or len(prediction) < 2 or len(outcome) < 2:
        raise IntegrityFailure("completed endpoint prediction/outcome missing")
    try:
        pred_x, pred_w = float(prediction[0]), float(prediction[1])
        actual_x, actual_w = float(outcome[0]), float(outcome[1])
        if not all(math.isfinite(x) for x in (pred_x, pred_w, actual_x, actual_w)):
            raise IntegrityFailure("completed endpoint prediction/outcome nonfinite")
        if abs(float(e["full_x_error"]) - (actual_x - pred_x)) > 1e-12:
            raise IntegrityFailure("completed endpoint x residual mismatch")
        if abs(float(e["full_w_error"]) - (actual_w - pred_w)) > 1e-12:
            raise IntegrityFailure("completed endpoint w residual mismatch")
        query_x = issued.get("main_query_physical")
        query_aux = main.get("auxiliary_at_issue")
        parent_x, parent_aux = e.get("query_physical"), e.get("query_auxiliary")
        if not all(isinstance(v, (list, tuple)) for v in (query_x, query_aux, parent_x, parent_aux)):
            raise IntegrityFailure("completed endpoint issue context missing")
        if (len(query_x) != 3 or len(query_aux) != 3 or len(parent_x) != 128 or len(parent_aux) != 97):
            raise IntegrityFailure("completed endpoint context has wrong packed or issue dimensions")
        if not all(math.isfinite(float(v)) for v in list(query_x) + list(query_aux) + list(parent_x) + list(parent_aux)):
            raise IntegrityFailure("completed endpoint context is nonfinite")
        if list(query_x) != list(parent_x[:3]) or list(query_aux) != list(parent_aux[:3]):
            raise IntegrityFailure("completed endpoint issue context mismatch")
    except IntegrityFailure:
        raise
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise IntegrityFailure("completed endpoint issue context malformed") from exc
    return True


def _pair_actual(low,high):
    out=dict(time=low['time'],low_action=low['action'],high_action=high['action'],status='unavailable',
        actual_x_difference=None,predicted_x_difference=None,x_residual_difference=None,x_identity_error=None,
        actual_w_difference=None,predicted_w_difference=None,w_residual_difference=None,w_identity_error=None)
    if high.get('action') is None or low.get('status')!='complete' or high.get('status')!='complete':return out
    if low['time']!=high['time'] or not (low['action']<high['action']<=HIGH_ACTION_MAX):
        raise IntegrityFailure('paired date/action labels invalid')
    for j,key in ((0,'x'),(1,'w')):
        actual=float(high['next_physical'][j])-float(low['next_physical'][j])
        pred=float(high['issued']['main']['full_physical_prediction'][j])-float(low['issued']['main']['full_physical_prediction'][j])
        residual=float(high['full_'+key+'_error'])-float(low['full_'+key+'_error'])
        identity=residual-(actual-pred)
        if not all(math.isfinite(v) for v in (actual,pred,residual,identity)) or abs(identity)>1e-12:
            raise IntegrityFailure('paired residual identity failure')
        out.update({f'actual_{key}_difference':actual,f'predicted_{key}_difference':pred,
                    f'{key}_residual_difference':residual,f'{key}_identity_error':identity})
    if out['actual_w_difference']!=0:raise IntegrityFailure('same-date probes changed actual proxy')
    out['status']='available';return out


def compact(record, score_module=None):
    """Validate a single H-routed history and release A/B scores separately."""
    score_module = measurements if score_module is None else score_module
    spec = record["spec"]
    if spec.get("cell") not in CELLS:
        raise ValueError("unknown candidate cell")
    if "H" not in record:
        raise IntegrityFailure("explicit H input is required")
    h = record.get("H")
    endpoints = record.get("endpoints", [])
    fatal = (record.get("status") == "fatal" or
             bool(record.get("failure") and record["failure"].get("classification", {}).get("status") == "fatal") or
             any(e.get("status") in ("software_failure", "fatal") for e in endpoints))
    if fatal:
        return dict(spec=deepcopy(spec), counts=deepcopy(record.get("counts", {})), status="fatal", H=h if h is True or h is False else None,
                    H_reason=None if h is True or h is False else record.get("status"),
                    C_A=None, C_B=None, A=None, B=None,
                    main_source_completed=record.get("main_source_completed"),
                    monitor_completed=record.get("monitor_completed"), endpoints=deepcopy(endpoints),
                    accept_A=False, accept_B=False, reasons={"A": ["fatal"], "B": ["fatal"]},
                    failure=deepcopy(record.get("failure")))
    if h is not True and h is not False and h is not None:
        raise ValueError("H must be explicitly true, false, or null")
    if h is False:
        if record.get("status") != "no_alarm" or record.get("source_callback_calls") != 0 or record.get("monitor_completed") != 8:
            raise ValueError("H=false must skip source acquisition")
        _endpoint_identity(spec, endpoints)
        if any(e.get("status") != "not_run_no_alarm" for e in endpoints):
            raise ValueError("no-alarm endpoints must be marked not_run_no_alarm")
        return dict(spec=deepcopy(spec), counts=deepcopy(record.get("counts", {})), status="no_alarm", H=False, H_reason=None,
                    C_A=None, C_B=None, A=None, B=None, main_source_completed=None,
                    monitor_completed=record.get("monitor_completed"), endpoints=deepcopy(endpoints),
                    accept_A=False, accept_B=False, reasons={"A": ["no_alarm"], "B": ["no_alarm"]})
    if h is None:
        if record.get("status") not in ("monitor_unavailable", "detector_not_calibrated"):
            raise ValueError("H=null requires an explicit unavailable reason")
        if record.get("source_callback_calls") != 0:
            raise ValueError("H=null must skip source acquisition")
        _endpoint_identity(spec, endpoints)
        if any(e.get("status") != "not_run_monitor_unavailable" for e in endpoints):
            raise ValueError("H-null endpoints must retain monitor-unavailable status")
        status = record["status"]
        return dict(spec=deepcopy(spec), counts=deepcopy(record.get("counts", {})), status=status, H=None, H_reason=status,
                    C_A=None, C_B=None, A=None, B=None, main_source_completed=None,
                    monitor_completed=record.get("monitor_completed"), endpoints=deepcopy(endpoints),
                    accept_A=False, accept_B=False, reasons={"A": [status], "B": [status]})

    _endpoint_identity(spec, endpoints)
    if any(e.get("status") not in ENDPOINT_STATES for e in endpoints):
        return dict(spec=deepcopy(spec), counts=deepcopy(record.get("counts", {})), status="fatal", H=True, H_reason=None,
                    C_A=None, C_B=None, A=None, B=None,
                    main_source_completed=record.get("main_source_completed"),
                    monitor_completed=record.get("monitor_completed"), endpoints=deepcopy(endpoints),
                    accept_A=False, accept_B=False,
                    reasons={"A": ["unknown_endpoint_state"], "B": ["unknown_endpoint_state"]})
    if any(e.get("status") in ("not_run_no_alarm", "not_run_monitor_unavailable") for e in endpoints):
        raise IntegrityFailure("source endpoints cannot carry monitor-only not-run reasons")
    if record.get("monitor_completed") != 8:
        raise IntegrityFailure("explicit H requires a complete eight-date monitor")
    if record.get("status") not in ("complete", "MAIN_unavailable"):
        raise IntegrityFailure("unknown source history state cannot be a scientific shortage")
    if record.get("status") == "complete" and record.get("main_source_completed") != 8:
        raise IntegrityFailure("complete history must contain all eight MAIN source transitions")
    if record.get("status") == "MAIN_unavailable" and not (0 <= record.get("main_source_completed", -1) < 8):
        raise IntegrityFailure("MAIN_unavailable history has inconsistent MAIN completion count")
    for e in endpoints:
        state = e.get("status")
        if state in ("complete", "support_refusal"):
            px, pe = e.get("query_physical"), e.get("query_auxiliary")
            if (not isinstance(px, (list, tuple)) or len(px) != 128 or
                    not isinstance(pe, (list, tuple)) or len(pe) != 97):
                raise IntegrityFailure("attempted endpoint parent context has wrong packed dimensions")
            try:
                if not all(math.isfinite(float(v)) for v in list(px) + list(pe)):
                    raise IntegrityFailure("attempted endpoint parent context is nonfinite")
            except (TypeError, ValueError) as exc:
                raise IntegrityFailure("attempted endpoint parent context is malformed") from exc
            if type(e.get("physical_action_supported")) is not bool or type(e.get("finite_executed_Q")) is not bool:
                raise IntegrityFailure("attempted source endpoint is missing Boolean qualification flags")
            q = e.get("executed_Q")
            if e["finite_executed_Q"]:
                try:
                    q = float(q)
                except (TypeError, ValueError) as exc:
                    raise IntegrityFailure("finite-Q flag has no numeric Q receipt") from exc
                if not math.isfinite(q):
                    raise IntegrityFailure("finite-Q flag disagrees with nonfinite executed Q")
            elif q is not None:
                raise IntegrityFailure("nonfinite-Q flag must retain null executed_Q")
        elif state == "main_unavailable":
            if e.get("physical_action_supported") is not None or e.get("finite_executed_Q") is not None:
                raise IntegrityFailure("main-unavailable placeholder cannot claim endpoint qualification")
        if e.get("status") == "complete":
            for field in ("full_x_error", "full_w_error"):
                try:
                    value = float(e[field])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("malformed completed endpoint residual") from exc
                if not math.isfinite(value):
                    raise FloatingPointError("nonfinite completed endpoint residual is fatal")
    by_date={t:[e for e in endpoints if e['time']==t] for t in DATES}
    lows={t:next(e for e in by_date[t] if e.get('action')==LOW_ACTION) for t in DATES}
    highs={t:next(e for e in by_date[t] if e.get('action')!=LOW_ACTION) for t in DATES}
    main_ok = (record.get("monitor_completed") == 8 and
               record.get("main_source_completed") == 8 and
               record.get("status") == "complete")
    q02 = {t: _endpoint_qualified(lows[t]) for t in DATES}
    qhigh = {t: _endpoint_qualified(highs[t]) for t in DATES}
    for t in DATES:
        low, high = lows[t], highs[t]
        if low["status"] in ("complete", "support_refusal") and high["status"] in ("complete", "support_refusal"):
            if (low.get("query_physical") != high.get("query_physical") or
                    low.get("query_auxiliary") != high.get("query_auxiliary")):
                raise IntegrityFailure("same-date action probes do not share their actual MAIN parent context")
    C_B = bool(main_ok and all(q02.values()))
    C_A = bool(main_ok and all(q02.values()) and all(qhigh.values()))
    if C_A and not C_B:
        raise AssertionError("C_A must imply C_B")
    result = dict(spec=deepcopy(spec), counts=deepcopy(record.get("counts", {})), status=record["status"], H=True, H_reason=None,
                  C_A=C_A, C_B=C_B, A=None, B=None,
                  main_source_completed=record.get("main_source_completed"),
                  monitor_completed=record.get("monitor_completed"), endpoints=deepcopy(endpoints),
                  accept_A=C_A, accept_B=C_B,
                  reasons={"A": [], "B": []})
    if not main_ok:
        result["reasons"]["A"].append("main_unavailable_or_incomplete")
        result["reasons"]["B"].append("main_unavailable_or_incomplete")
    if not all(q02.values()):
        result["reasons"]["A"].append("action_002_unavailable")
        result["reasons"]["B"].append("action_002_unavailable")
    if not all(qhigh.values()):
        result["reasons"]["A"].append("selected_high_action_unavailable")
    result["pairs"]=[_pair_actual(lows[t],highs[t]) for t in DATES]
    if C_A:
        rows = [dict(time=t, status="complete",
                     full_x_error=highs[t]["full_x_error"] - lows[t]["full_x_error"],
                     full_w_error=lows[t]["full_w_error"]) for t in DATES]
        score = score_module.score(rows)
        if not score["complete"]:
            raise ValueError("true C_A produced incomplete original energy score")
        if not all(math.isfinite(float(score[k])) for k in ("scalar", "serial", "signed_serial_numerator", "serial_denominator")):
            raise IntegrityFailure("A score output is nonfinite")
        result["A"] = score["scalar"]
        result["A_measurement"] = score
    if C_B:
        rows = [dict(time=t, status="complete", full_x_error=lows[t]["full_x_error"],
                     full_w_error=lows[t]["full_w_error"]) for t in DATES]
        score = score_module.score(rows)
        if not score["complete"]:
            raise ValueError("true C_B produced incomplete original serial score")
        if not all(math.isfinite(float(score[k])) for k in ("scalar", "serial", "signed_serial_numerator", "serial_denominator")):
            raise IntegrityFailure("B score output is nonfinite")
        result["B"] = score["serial"]
        result["B_measurement"] = score
    return result


class CandidateBanks:
    """One shared proposal counter and independently frozen first39 A/B banks."""
    def __init__(self, cell):
        if cell not in CELLS:
            raise ValueError("unknown candidate")
        self.cell = cell
        self.proposals = 0
        self.banks = {"A": [], "B": []}
        self.rows = []
        self.fatal = False
        self.seen_seeds = set()
        self.aggregate_counts = {}

    def next_slot_status(self):
        if self.fatal:
            return "not_run_after_fatal"
        if all(len(v) >= TARGET for v in self.banks.values()):
            return "not_run_both_targets_reached"
        if self.proposals >= CAP:
            return "cap_incomplete"
        return "attempt"

    def record(self, proposal):
        if self.next_slot_status() != "attempt":
            raise ValueError("proposal submitted after stopping condition")
        if proposal["spec"]["cell"] != self.cell:
            raise ValueError("candidate proposal routed to wrong bank")
        seed = proposal["spec"].get("seed")
        if seed in self.seen_seeds:
            raise ValueError("duplicate candidate proposal identity")
        index = self.proposals + 1
        self.proposals = index
        self.seen_seeds.add(seed)
        for name, amount in proposal.get("counts", {}).items():
            self.aggregate_counts[name] = self.aggregate_counts.get(name, 0) + amount
        row = deepcopy(proposal)
        row["proposal_position"] = index
        row["branch"] = {}
        if proposal.get("status") == "fatal" or proposal.get("C_A") is None and proposal.get("H") is True:
            self.fatal = True
            for branch in ("A", "B"):
                row["branch"][branch] = dict(eligible=False, insertion="unknown_fatal", bank_count=len(self.banks[branch]))
        else:
            for branch in ("A", "B"):
                eligible = proposal.get("H") is True and proposal.get("C_" + branch) is True
                if eligible and len(self.banks[branch]) < TARGET:
                    self.banks[branch].append(dict(seed=proposal["spec"].get("seed"), score=proposal[branch],
                                                   proposal_position=index))
                    insertion = "inserted"
                elif eligible:
                    insertion = "eligible_not_inserted_target_reached"
                else:
                    insertion = "not_eligible"
                row["branch"][branch] = dict(eligible=eligible, insertion=insertion,
                                              bank_count=len(self.banks[branch]))
        self.rows.append(row)
        return deepcopy(row)

    def ledger(self):
        rows = deepcopy(self.rows)
        while len(rows) < CAP:
            if self.fatal:
                reason = "not_run_after_fatal"
            elif all(len(v) >= TARGET for v in self.banks.values()):
                reason = "not_run_both_targets_reached"
            else:
                reason = "not_run_cap_exhausted" if self.proposals >= CAP else "unassigned_potential"
            rows.append(dict(proposal_position=len(rows) + 1, status=reason, seed=None))
        return rows

    def status(self):
        return dict(cell=self.cell, proposals=self.proposals, cap=CAP, target=TARGET,
                    accepted={k: len(v) for k, v in self.banks.items()},
                    available={k: len(v) == TARGET and not self.fatal for k, v in self.banks.items()},
                    accepted_rows=deepcopy(self.banks), fatal=self.fatal,
                    H_counts={"true": sum(r.get("H") is True for r in self.rows),
                              "false": sum(r.get("H") is False for r in self.rows),
                              "null": sum(r.get("H") is None for r in self.rows)},
                    counts=deepcopy(self.aggregate_counts), potential_slots=self.ledger(),
                    cap_incomplete=self.proposals == CAP and any(len(v) < TARGET for v in self.banks.values()),
                    both_targets_reached=all(len(v) == TARGET for v in self.banks.values()))


def proposal_slot(h, *, monitor_reason=None, monitor_callback=None, source_callback=None):
    """Ordinary callback router; H=None/false never invokes source callback."""
    if h is not True and h is not False and h is not None:
        raise ValueError("H must be bool or null")
    monitor = monitor_callback() if monitor_callback is not None else None
    if h is None:
        return dict(H=None, status=monitor_reason or "monitor_unavailable", source_callback_calls=0,
                    monitor=monitor, endpoints=None)
    if h is False:
        return dict(H=False, status="no_alarm", source_callback_calls=0, monitor=monitor,
                    endpoints="not_run_no_alarm")
    if source_callback is None:
        raise ValueError("H=true requires an injected ordinary source stub")
    out = source_callback()
    if not isinstance(out, dict):
        raise TypeError("source stub must return a fabricated receipt mapping")
    result = dict(out)
    result.update(H=True, source_callback_calls=1, monitor=monitor)
    return result


def operational_candidate_p(bank, observed, cell, branch, *, campaign_valid=True):
    """Inclusive rank or operational p=1 with explicit abstention reason."""
    if branch not in ("A", "B") or cell not in TAILS:
        raise ValueError("invalid branch/candidate")
    if not campaign_valid:
        return dict(p=None, p_exact=None, test_performed=False, reason="campaign_invalid", reasons=["campaign_invalid"])
    if observed.get("status") == "fatal":
        return dict(p=None, p_exact=None, test_performed=False, reason="fatal", reasons=["fatal"])
    if observed.get("H") is False:
        reasons = ["no_alarm"] + (["bank_incomplete"] if len(bank) != TARGET else [])
        return dict(p=1.0, p_exact="1", test_performed=False, reason="no_alarm", reasons=reasons)
    if observed.get("H") is None:
        reason = observed.get("H_reason") or observed.get("status") or "monitor_unavailable"
        reasons = [reason] + (["bank_incomplete"] if len(bank) != TARGET else [])
        return dict(p=1.0, p_exact="1", test_performed=False, reason=reason, reasons=reasons)
    reasons = []
    if observed.get("C_" + branch) is not True:
        reasons.append("observed_branch_unavailable")
    if len(bank) != TARGET:
        reasons.append("bank_incomplete")
    if reasons:
        return dict(p=1.0, p_exact="1", test_performed=False, reason=reasons[0], reasons=reasons)
    values = [float(row["score"]) for row in bank]
    value = float(observed[branch])
    if not all(math.isfinite(x) for x in [*values, value]):
        raise ValueError("nonfinite rank score is an integrity fault")
    tail = TAILS[cell][0 if branch == "A" else 1]
    n = sum((x >= value if tail == "upper" else x <= value) for x in values)
    numerator = 1 + n
    return dict(p=numerator / 40, p_exact=f"{numerator}/40", test_performed=True,
                reason=None, reasons=[], tail=tail)


def infer_eval(observed, banks, *, campaign_valid=True):
    if not campaign_valid or observed.get("status") == "fatal" or any(bank.fatal for bank in banks.values()):
        return dict(status="campaign_invalid_no_inference", candidates=None, retained=None,
                    model="unresolved", latent="unresolved")
    decisions = []
    for cell in CELLS:
        cellbanks = banks[cell]
        ps = {b: operational_candidate_p(cellbanks.banks[b], observed, cell, b,
                                         campaign_valid=campaign_valid) for b in ("A", "B")}
        excluded = any(p["test_performed"] and p["p"] <= 1/40 for p in ps.values())
        decisions.append(dict(candidate=cell, branches=ps, excluded=excluded))
    retained = [d["candidate"] for d in decisions if not d["excluded"]]
    if not retained:
        model = latent = "unresolved"
    else:
        model = "present" if set(retained) <= {"CM", "MIX"} else "absent" if set(retained) <= {"N", "CL"} else "unresolved"
        latent = "present" if set(retained) <= {"CL", "MIX"} else "absent" if set(retained) <= {"N", "CM"} else "unresolved"
    if observed.get("H") is False:
        status = "no_alarm"
    elif observed.get("H") is None:
        status = observed.get("status", "monitor_unavailable")
    elif any(not any(p["test_performed"] for p in d["branches"].values()) for d in decisions):
        status = "branch_unavailable"
    else:
        status = "complete"
    return dict(status=status, H=observed.get("H"), candidates=decisions,
                retained=retained, model=model, latent=latent)
