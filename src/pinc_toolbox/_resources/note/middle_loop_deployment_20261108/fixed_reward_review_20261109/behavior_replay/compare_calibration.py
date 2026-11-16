"""Compare both assigned October calibration runs with the fixed-reward replay."""
from collections import Counter
import json

from compare import compare_value, scientific_records
from run_matched import HERE, RECOVERY

OLD = RECOVERY / "recovery_v003"
RUNS = {"calibration_scalar_serial": "cal_scalar_serial_001",
        "calibration_paired": "cal_paired_001"}


def comparable_failure(failure):
    if not failure:
        return None
    return {k: failure.get(k) for k in ("phase", "time", "type", "reason", "classification")}


def scalar_assignment(result):
    rows = []
    for r in result["assigned"]:
        episode = r["result"] or {}
        rows.append({**{k: r[k] for k in ("slot", "attempt", "seed", "status", "executed")},
                     "scores": episode.get("scores"), "dates": episode.get("completed_dates"),
                     "failure": comparable_failure(episode.get("failure"))})
    return rows


def paired_assignment(result):
    return [dict(r, failure=comparable_failure(r.get("failure"))) for r in result["attempts"]]


def compare_run(group, new_folder):
    previous = OLD / group / "unit_0000/artifacts"
    current = HERE / new_folder
    old, new = [json.loads((p / "RESULT.json").read_text()) for p in (previous, current)]
    discrepancies, deltas = [], []
    scalar = group == "calibration_scalar_serial"
    if scalar:
        for key in ("slots", "status", "threshold", "R_execution_complete", "A_all_statistical_slots_complete"):
            compare_value(old[key], new[key], key, discrepancies, deltas)
        old_assign, new_assign = map(scalar_assignment, (old, new))
        thresholds = {k: max(r["scores"][k] for r in new_assign if r["status"] == "available")
                      for k in ("scalar", "serial")}
        status_counts = Counter(r["status"] for r in new_assign)
    else:
        for key in ("status", "qualified", "bank", "rank_minimum", "assigned"):
            compare_value(old[key], new[key], key, discrepancies, deltas)
        old_assign, new_assign = map(paired_assignment, (old, new))
        thresholds = {"paired": new["bank"]}
        status_counts = Counter(r["qualification"] for r in new_assign)
    compare_value(old_assign, new_assign, "assignment", discrepancies, deltas)
    paths = []
    for base in (previous, current):
        paths.append({p.relative_to(base) for p in base.rglob("records.xz")})
    for p in sorted(paths[0] ^ paths[1]):
        discrepancies.append(f"episode present in one run only: {p}")
    for p in sorted(paths[0] & paths[1]):
        compare_value(scientific_records(previous / p), scientific_records(current / p),
                      str(p.parent), discrepancies, deltas)
    return dict(group=group, status=new["status"], assignments=len(new_assign),
                status_counts=dict(status_counts), archives_compared=len(paths[0] & paths[1]),
                discrepancies=len(discrepancies), first_discrepancies=discrepancies[:20],
                maximum_q_roundoff=max(deltas, default=None), thresholds=thresholds)


if __name__ == "__main__":
    reports = [compare_run(group, folder) for group, folder in RUNS.items()]
    (HERE / "CALIBRATION_PARITY.json").write_text(json.dumps(reports, indent=2) + "\n")
    for row in reports:
        print({k: v for k, v in row.items() if k != "thresholds"})
