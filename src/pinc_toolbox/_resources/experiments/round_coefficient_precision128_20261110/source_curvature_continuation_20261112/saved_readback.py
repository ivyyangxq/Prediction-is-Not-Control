#!/usr/bin/env python3
"""Saved-only common-innovation action-curvature continuation."""
from __future__ import annotations

import argparse
import csv
import json
import lzma
import math
import pickle
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "experiments/round_coefficient_precision128_20261110/campaign001"
OUT = Path(__file__).resolve().parent / "attempt002"
TOL = Decimal("5e-10")
getcontext().prec = 60


def D(x):
    return Decimal.from_float(float(x))


def discrepancy(points):
    """points: exactly three (action, next_x) pairs, sorted by action."""
    (al, yl), (am, ym), (ah, yh) = points
    al, am, ah, yl, ym, yh = map(D, (al, am, ah, yl, ym, yh))
    if not al < am < ah:
        raise ValueError("actions must be distinct and increasing")
    w = (am - al) / (ah - al)
    return ym - ((Decimal(1) - w) * yl + w * yh) - Decimal(2) * (am - al) * (ah - am)


def ordinary():
    # The first two are arbitrary intercept/slope and therefore cover latent shifts.
    for u, v in ((0.2, 0.7), (-1.0, 3.2)):
        pts = [(a, u + v*a - 2*a*a) for a in (0.0, 0.04, 0.10)]
        assert abs(discrepancy(pts)) < Decimal("1e-15")
    pts = [(a, 0.3 + 0.8*a - a*a) for a in (0.0, 0.04, 0.10)]
    theoretical = -(D(.04)-D(0))*(D(.10)-D(.04))
    assert abs(discrepancy(pts) - theoretical) < Decimal("1e-16")
    try:
        discrepancy([(0.0, 1.0), (0.0, 2.0), (0.1, 0.0)])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate actions must be unavailable")
    assert abs(discrepancy([(0.0, 1.0), (0.05, 1.0 + 0.2*.05 - 2*.05**2), (0.1, 1.0 + 0.2*.1 - 2*.1**2)])) < Decimal("1e-15")
    return {"status":"PASS", "tests":["null affine-plus-quadratic", "new curvature separation", "duplicate actions rejected", "isolated qualified dates remain independently usable"], "tolerance":str(TOL)}


def validate_dates(dates):
    if any(b != a+1 for a,b in zip(dates, dates[1:])):
        raise ValueError("nonconsecutive dates")


def archive_raw(path):
    out = {}
    with lzma.open(path, "rb") as stream:
        while True:
            try:
                row = pickle.load(stream)
            except EOFError:
                break
            if not isinstance(row, tuple) or not row or row[0] != "raw":
                continue
            _, name, data = row
            if name in {"MAIN_OBSERVATIONS.json", "PAIRED_PROBE_ENDPOINTS.json"} or name.startswith(("ISSUED_", "PAIR_ENDPOINT_", "RELEASED_")):
                if isinstance(data, bytes) and name.endswith(".json"):
                    out[name] = json.loads(data)
    return out


def endpoint_by_date(raw, date, side):
    key = f"PAIR_ENDPOINT_{date}_{side}.json"
    return raw.get(key)


def qualify(raw, date):
    mains = [r for r in raw.get("MAIN_OBSERVATIONS.json", []) if r.get("path") == 0 and r.get("time") == date]
    if len(mains) != 1:
        return None, "main_missing_or_duplicate"
    main = mains[0]
    if not all(k in main for k in ("action", "next_x", "state_x")):
        return None, "main_endpoint_missing"
    lo, hi = endpoint_by_date(raw, date, "LOW"), endpoint_by_date(raw, date, "HIGH")
    if lo is None or hi is None:
        return None, "probe_missing"
    for p in (lo, hi):
        if p.get("status") != "complete" or not p.get("physical_action_supported") or not p.get("finite_executed_Q") or not p.get("public_action_supported"):
            return None, "probe_unsupported_or_unexecuted"
        if "action" not in p:
            return None, "probe_endpoint_missing"
        issued = p.get("issued", {})
        if issued.get("time") != date or issued.get("fit_as_of") != date:
            raise ValueError(f"probe issue chronology error at {date}")
    issued = raw.get(f"ISSUED_{date}.json")
    mains_issued = [r for r in (issued or []) if r.get("path_id") == 0 and r.get("role") == "main_greedy"]
    if len(mains_issued) != 1:
        return None, "main_issue_missing_or_duplicate"
    mi = mains_issued[0]
    if mi.get("time") != date or mi.get("issued", {}).get("fit_as_of") != date:
        raise ValueError(f"main issue chronology error at {date}")
    # Confirm MAIN is an issued, locally qualified action.
    grid = [i/100 for i in range(11)]
    idx = min(range(len(grid)), key=lambda i: abs(grid[i] - float(main["action"])))
    if abs(grid[idx] - float(main["action"])) > 1e-12 or not mi.get("issued", {}).get("qualified_action_mask", [])[idx]:
        return None, "main_action_not_qualified"
    expanded = raw.get("PAIRED_PROBE_ENDPOINTS.json", [])
    def expanded_probe(envelope):
        matches=[e for e in expanded if e.get("path")==0 and e.get("time")==date and abs(float(e.get("action",-99))-float(envelope["action"]))<=1e-12]
        return matches[0] if len(matches)==1 else None
    elo,ehi=expanded_probe(lo),expanded_probe(hi)
    if elo is None or ehi is None:
        return None, "expanded_probe_endpoint_missing_or_ambiguous"
    if any(e.get("release_time")!=date+1 or e.get("status")!="complete" for e in (elo,ehi)):
        raise ValueError(f"probe release chronology error at {date}")
    for envelope, expanded_row in ((lo, elo), (hi, ehi)):
        physical = envelope.get("next_physical")
        if not isinstance(physical, list) or len(physical) < 2:
            raise ValueError("completed probe missing raw outcome")
        if physical[0] != expanded_row["next_x"] or physical[1] != expanded_row["next_w"]:
            raise ValueError("expanded and raw endpoint outcomes disagree")
        if envelope["state_x"] != expanded_row["state_x"] or envelope["release_time"] != expanded_row["release_time"]:
            raise ValueError("expanded and raw endpoint clocks disagree")
    points = [(main["action"], main["next_x"]), (lo["action"], elo["next_x"]), (hi["action"], ehi["next_x"])]
    if len({D(a) for a,_ in points}) != 3:
        return None, "fewer_than_three_distinct_actions"
    if any(r.get("path") != 0 or r.get("time") != date or r.get("release_time") != date+1 for r in (main,elo,ehi)):
        raise ValueError(f"path/date/release mismatch at {date}")
    if any(not math.isfinite(float(v)) for pair in points for v in pair):
        return None, "nonfinite_endpoint"
    xs = [D(r["state_x"]) for r in (main,elo,ehi)]
    if max(xs)-min(xs) > D("1e-14"):
        return None, "common_state_mismatch"
    released = raw.get(f"RELEASED_{date+1}.json", {})
    physical = released.get("physical", [])
    if not physical:
        return None, "release_public_state_missing"
    main_release=physical[0]
    if len(main_release)<3 or abs(float(main_release[0])-float(main["next_x"]))>1e-12 or abs(float(main_release[2])-float(main["action"]))>1e-12:
        raise ValueError(f"MAIN release does not match path 0 at {date}")
    ws = [D(main_release[1]), D(elo["next_w"]), D(ehi["next_w"])]
    if max(ws)-min(ws) > D("1e-14"):
        return None, "common_innovation_proxy_mismatch"
    points.sort(key=lambda p: D(p[0]))
    return {"date":date,"release":date+1,"state_x":main["state_x"],"points":points,"discrepancy":str(discrepancy(points)),"proxy_next_w_range":str(max(ws)-min(ws))}, "qualified"


def result_for(folder):
    return json.loads((folder/"RESULT.json").read_text())


def run_readback():
    OUT.mkdir(parents=True, exist_ok=False)
    histories = sorted(CAMPAIGN.glob("history_*/RESULT.json"))
    if len(histories) != 512:
        raise RuntimeError(f"expected 512 history results, found {len(histories)}")
    rows=[]
    now=datetime.now(timezone.utc).isoformat()
    for rp in histories:
        folder=rp.parent
        res=json.loads(rp.read_text())
        ordinal=int(folder.name.split("_")[1])
        alarm=bool(res.get("monitor_alarm"))
        inference=res.get("inference") or {}
        receipt={"ordinal":ordinal,"seed":res.get("seed"),"candidate":res.get("candidate"),"original_alarm":alarm,"original_decision":res.get("decision"),"original_monitor_status":res.get("monitor_status"),"original_model_report":res.get("decision")=="model_involvement","latent_status":"not_executed_by_this_branch","method":"common_innovation_action_curve_exclusion","source_branch":"maintained_no_model_action_curvature_minus_2","scope":"controlled_DGP_common_innovation_probe","publication_kind":"retrospective_saved_history_replay","replay_model_clock_release":40,"actual_utc_publication_timestamp":now,"qualified_triples":[],"unavailable_dates":[],"last_input_release":None}
        if not alarm:
            receipt["outcome"]=("not_entered_monitor_support_unavailable" if res.get("monitor_status")=="support_unavailable" else "not_entered_no_first_layer_alarm")
            rows.append(receipt); continue
        archive=folder/"monitors"/f"monitor_{ordinal:03d}"/"records.xz"
        receipt["public_record_path"]=str(archive.relative_to(ROOT))
        if not archive.exists():
            receipt["outcome"]="source_unavailable_monitor_archive_missing"
            rows.append(receipt); continue
        raw=archive_raw(archive)
        qualified=[]
        # Failures of an individual support check are unavailable dates; corrupted chronology is a technical error.
        for date in range(32,40):
            q,why=qualify(raw,date)
            if q: qualified.append(q)
            else: receipt["unavailable_dates"].append({"date":date,"reason":why})
        receipt["qualified_triples"]=qualified
        receipt["last_input_release"]=40
        viol=[q for q in qualified if abs(Decimal(q["discrepancy"]))>TOL]
        receipt["outcome"]="model_involvement" if viol else ("source_unavailable_no_qualified_triples" if not qualified else "no_rejection_unresolved")
        receipt["violating_dates"]=[q["date"] for q in viol]
        receipt["arithmetic_allowance"]=str(TOL)
        receipt["identifying_assumption"]="same path/date/start state and common innovation; every admitted no-model law has q(a)=u+v*a-2*a^2"
        rows.append(receipt)
    published=datetime.now(timezone.utc).isoformat()
    for row in rows:
        row["actual_utc_publication_timestamp"]=published
        row["arithmetic_error_condition"]="Validity assumes each common-innovation endpoint arithmetic error <=2.5e-10; this is not a universal machine-certified bound"
    # summarize by candidate label, only after the classification has been computed.
    summary={"scope":"saved-only continuation; no simulation or calibration", "n_histories":len(rows),"method":"common_innovation_action_curve_exclusion","outcomes":dict(Counter(r["outcome"] for r in rows)),"by_candidate":{},"original_by_candidate":{},"combined_union_by_candidate":{},"latent":"not executed; no latent rate inferred", "arithmetic_tolerance":str(TOL),"interpretation":"post-hoc same-sample branch; descriptive finite-sample counts, not calibrated type-I error or prospective power"}
    for cand in sorted({str(r["candidate"]) for r in rows}):
        group=[r for r in rows if str(r["candidate"])==cand]
        alarms=[r for r in group if r["original_alarm"]]
        models=[r for r in alarms if r["outcome"]=="model_involvement"]
        old=[r for r in group if r.get("original_model_report") is True]
        union={r["ordinal"] for r in old+models}
        summary["by_candidate"][cand]={"n":len(group),"first_layer_alarm":len(alarms),"new_branch_model_involvement_full_n":len(models),"new_branch_model_involvement_conditional_alarm_n":len(models),"new_branch_source_unavailable":sum(r["outcome"].startswith("source_unavailable") for r in alarms),"new_branch_no_rejection_unresolved":sum(r["outcome"]=="no_rejection_unresolved" for r in alarms),"new_branch_alarm_conditional_denominator":len(alarms)}
        summary["original_by_candidate"][cand]={"original_model_report_n":len(old),"first_layer_alarm_n":sum(r["original_alarm"] for r in group)}
        summary["combined_union_by_candidate"][cand]={"original_or_new_branch_model_report_n":len(union),"full_n":len(group)}
    (OUT/"ORDINARY.json").write_text(json.dumps(ordinary(),indent=2)+"\n")
    (OUT/"SAVED_MODEL_RECEIPTS.json").write_text(json.dumps(rows,indent=2)+"\n")
    (OUT/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (OUT/"SAVED_MODEL_RECEIPTS.csv").open("w",newline="") as f:
        fields=sorted({k for r in rows for k in r if k not in ("qualified_triples","unavailable_dates")})
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:(json.dumps(r.get(k),separators=(",",":")) if isinstance(r.get(k),(dict,list)) else r.get(k)) for k in fields})
    return summary


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=["ordinary","readback"]); args=ap.parse_args()
    if args.mode=="ordinary": print(json.dumps(ordinary(),indent=2))
    else: print(json.dumps(run_readback(),indent=2))
