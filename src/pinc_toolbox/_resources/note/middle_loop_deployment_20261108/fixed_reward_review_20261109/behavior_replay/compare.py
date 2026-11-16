"""Compare released behavior and conditional proposal paths, not file bytes."""
from collections import Counter
import json
import lzma
import pickle
import re
import sys

import numpy as np

from run_matched import HERE, historic

LABELS = ("N", "CL", "CM", "MIX")
RAW = re.compile(r"^(states|ISSUED|RELEASED|AUXILIARY|PAIR(?:_ENDPOINT)?|EXPANDED_ENDPOINT|probe|MAIN)_\d+.*\.json$")
FIXED_NAMES = {"MAIN_OBSERVATIONS.json", "PAIRED_PROBE_ENDPOINTS.json", "PAIRED_MONITOR_SCORE.json", "BOUNDARY.json"}
Q_KEYS = {"q_values", "executed_action_q", "executed_Q", "q_value", "direct_Q", "nearest_Q"}


def scientific_records(path):
    records = {}
    with lzma.open(path, "rb") as stream:
        while True:
            event = pickle.load(stream)
            if event[0] == "end":
                return records
            if event[0] == "raw" and (RAW.fullmatch(event[1]) or event[1] in FIXED_NAMES):
                assert event[1] not in records, (path, event[1])
                records[event[1]] = json.loads(event[2])


def compare_value(old, new, where, mismatches, q_deltas):
    if type(old) is not type(new):
        mismatches.append(f"{where}: type {type(old).__name__} vs {type(new).__name__}")
        return
    if isinstance(old, dict):
        if old.keys() != new.keys():
            mismatches.append(f"{where}: keys differ {old.keys() ^ new.keys()}")
        for key in old.keys() & new.keys():
            if key in Q_KEYS:
                a, b = np.asarray(old[key], dtype=float), np.asarray(new[key], dtype=float)
                if a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)):
                    mismatches.append(f"{where}/{key}: finite Q mask differs")
                elif np.isfinite(a).any():
                    q_deltas.append(float(np.max(np.abs(a[np.isfinite(a)] - b[np.isfinite(b)]))))
            else:
                compare_value(old[key], new[key], f"{where}/{key}", mismatches, q_deltas)
    elif isinstance(old, list):
        if len(old) != len(new):
            mismatches.append(f"{where}: length {len(old)} vs {len(new)}")
        else:
            for i, (a, b) in enumerate(zip(old, new)):
                compare_value(a, b, f"{where}/{i}", mismatches, q_deltas)
    elif old != new:
        mismatches.append(f"{where}: {old!r} != {new!r}")


def compare(ordinal):
    previous, old = historic(ordinal)
    current = HERE / f"attempt002_{LABELS[ordinal]}"
    new = json.loads((current / "RESULT.json").read_text())
    if new["execution_status"] != "complete":
        return dict(ordinal=ordinal, candidate=old["candidate"], status="invalid",
                    failure=new.get("failure"))
    mismatches, q_deltas = [], []
    for key in ("monitor_alarm", "monitor_status", "decision", "latent", "thresholds",
                "banks", "observed", "inference"):
        compare_value(old.get(key), new.get(key), key, mismatches, q_deltas)
    for key in ("status", "scores", "completed_dates", "failure"):
        compare_value(old["monitor"].get(key), new["monitor"].get(key),
                      f"monitor/{key}", mismatches, q_deltas)
    paths = {}
    for directory in (previous, current):
        paths[directory] = {p.relative_to(directory) for p in directory.rglob("records.xz")}
    old_paths, new_paths = paths[previous], paths[current]
    for p in sorted(old_paths ^ new_paths):
        mismatches.append(f"episode only in one execution: {p}")
    totals = Counter()
    for path in sorted(old_paths & new_paths):
        original = scientific_records(previous / path)
        rerun = scientific_records(current / path)
        totals[path.parts[0]] += 1
        compare_value(original, rerun, str(path.parent), mismatches, q_deltas)
    return dict(ordinal=ordinal, candidate=old["candidate"], status="complete",
                old_alarm=old["monitor_status"], new_alarm=new["monitor_status"],
                old_source=old["decision"], new_source=new["decision"],
                scores=new["monitor"]["scores"],
                source_accepted={k: len(v["values"]) for k, v in new["banks"].items()},
                ranks=(new.get("inference") or {}).get("rank_p"),
                archives_compared=dict(totals), discrepancies=len(mismatches),
                first_discrepancies=mismatches[:16], max_q_roundoff=max(q_deltas, default=None))


if __name__ == "__main__":
    indices = [int(i) for i in sys.argv[1:]] or list(range(4))
    result = [compare(i) for i in indices]
    print(json.dumps(result, indent=2))
    if indices == list(range(4)):
        (HERE / "BEHAVIOR_AND_SOURCE_PARITY.json").write_text(json.dumps(result, indent=2) + "\n")
