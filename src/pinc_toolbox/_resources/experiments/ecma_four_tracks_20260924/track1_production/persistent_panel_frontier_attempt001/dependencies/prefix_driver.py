"""Native Stage-1 multibank runner; importing this module performs no fits or DGP steps.

The only execution entry point is ``run`` / ``--run``.  CAL and EVAL seeds are
read from the sibling, immutable planning ledger.  Every attempted episode
constructs the requested v004 source at its requested strength from period 0;
the step profile is zero through the common 24-date STABLE-law prefix.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from pinc_toolbox import native_gq, rnd2_counterfactual
from pinc_toolbox.counterfactual import run_four_paths
from pinc_toolbox.g import SupportError
from pinc_toolbox.rnd2_counterfactual import prepare_rnd2_learner

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'DGP').is_dir() and (p/'experiments').is_dir())
DESIGN_DIR = HERE.parent / "native_power_frontier_design_attempt001"
PLAN_PATH = DESIGN_DIR / "MULTIBANK_PLAN.json"
STREAM_PATH = DESIGN_DIR / "MULTIBANK_STREAMS.csv"
ENV_FILE = ROOT / "DGP/RND/02_state_dependent_local/model_source_v004/environment.py"
MECHANICS_FILE = ROOT / "experiments/ecma_four_tracks_20260924/track1_production/stage1_native_rank_pilot_attempt001/runner.py"
EXPECTED_PACKAGE = Path("/tmp/pinc-counterfactual-root-2412/lib/python3.9/site-packages/pinc_toolbox").resolve()

FIT_COUNTS = {"prefix_base_G_fit_attempted": 0, "prefix_base_G_fit_succeeded": 0,
              "native_G_fit_attempted": 0, "native_G_fit_succeeded": 0,
              "finite_Q_fit_attempted": 0, "finite_Q_fit_succeeded": 0}
DGP_COUNTS = {"vector_calls_attempted": 0, "vector_calls_succeeded": 0,
              "path_transitions_attempted": 0, "path_transitions_succeeded": 0}


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_dgp():
    return _load_file("native_power_frontier_model_source_v004", ENV_FILE)


def _load_mechanics():
    return _load_file("native_power_frontier_stage1_mechanics", MECHANICS_FILE)


def _load_ranker():
    rank_path = ROOT / "experiments/ecma_four_tracks_20260924/track1_production/stage1_support_conditional_attempt001/conditional_rank.py"
    return _load_file("native_power_frontier_conditional_rank", rank_path)


def load_plan_and_streams():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    with STREAM_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["bank"] = int(row["bank"])
        row["seed"] = int(row["seed"])
        for field in ("slot", "attempt", "index"):
            row[field] = None if row[field] == "" else int(row[field])
        row["strength"] = float(row["strength"])
    return plan, rows


def cell_key(cell: str, strength: float) -> str:
    """Stable seven-cell label; source and strength remain in the source ledger."""
    return "%s_%s" % (cell, ("%g" % float(strength)).replace(".", "p"))


def _install_counters(dgp):
    """Mirror the reviewed T1 counters, scoped to this process only."""
    old_prefix_fit, old_native_fit, old_q_fit = (rnd2_counterfactual._fit_model,
        native_gq._fit, native_gq._finite_q)

    def prefix_fit(*args, **kwargs):
        FIT_COUNTS["prefix_base_G_fit_attempted"] += 1
        result = old_prefix_fit(*args, **kwargs)
        FIT_COUNTS["prefix_base_G_fit_succeeded"] += 1
        return result

    def native_fit(*args, **kwargs):
        FIT_COUNTS["native_G_fit_attempted"] += 1
        result = old_native_fit(*args, **kwargs)
        FIT_COUNTS["native_G_fit_succeeded"] += 1
        return result

    def q_fit(*args, **kwargs):
        FIT_COUNTS["finite_Q_fit_attempted"] += 1
        result = old_q_fit(*args, **kwargs)
        FIT_COUNTS["finite_Q_fit_succeeded"] += 1
        return result

    rnd2_counterfactual._fit_model = prefix_fit
    native_gq._fit = native_fit
    native_gq._finite_q = q_fit
    original_step = dgp.Environment.step

    def counted_step(self, t, actions):
        DGP_COUNTS["vector_calls_attempted"] += 1
        DGP_COUNTS["path_transitions_attempted"] += int(self.paths)
        result = original_step(self, t, actions)
        DGP_COUNTS["vector_calls_succeeded"] += 1
        DGP_COUNTS["path_transitions_succeeded"] += int(self.paths)
        return result
    dgp.Environment.step = counted_step


def _prefix(dgp, seed: int, source: str, strength: float):
    """Build the fresh 32x24 prefix using the requested v004 constructor.

    All allowed source mechanisms have g_t=0 for t<24, so source-specific
    dispatch is already correctly installed before the first transition while
    the prefix law remains STABLE.  The T1 seeded prefix action policy is kept.
    """
    env = dgp.Environment(int(seed), str(source), paths=32, horizon=64,
                          switch_date=32, strength=float(strength), profile="step")
    policy_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 20260925]))
    previous = np.full(32, dgp.A0, dtype=float)
    records = []
    for t in range(32):
        state, w = env.public(t)
        actions = policy_rng.choice(dgp.ACTION_GRID, 32)
        env.step(t, actions)
        next_state, next_w = env.public(t + 1)
        reward = env.released_reward(t + 1)
        for path in range(32):
            records.append(dict(path_id=path, period=t, release_period=t + 1,
                state=float(state[path]), w=float(w[path]), previous_action=float(previous[path]),
                action=float(actions[path]), next_state=float(next_state[path]),
                next_w=float(next_w[path]), reward=float(reward[path])))
        previous = actions
    return pd.DataFrame.from_records(records), env


def _single_path_copy(env, mechanics):
    return mechanics._single_path_copy(env)


def run_episode(row, out: Path, *, dgp, mechanics):
    seed, role = int(row["seed"]), row["role"]
    source, strength = row["source"], float(row["strength"])
    base = "episode_%010d" % seed
    before_fit, before_dgp = dict(FIT_COUNTS), dict(DGP_COUNTS)
    history, prefix_env = _prefix(dgp, seed, source, strength)
    history.to_csv(out / (base + "_prefix.csv"), index=False, float_format="%.17g")
    live = _single_path_copy(prefix_env, mechanics)
    _save_json(out / (base + "_oracle_truth.json"), {
        "seed": seed, "role": role, "bank": row["bank"], "cell": row["cell"],
        "source": source, "strength": strength,
        "initial_Z": live.latent[0, 24].tolist(),
        "latent_innovations": live.latent_noise[0, 24:32].tolist(),
        "state_innovations": live.state_noise[0, 24:32].tolist(),
        "measurement_innovations": live.measurement_noise[0, 24:33].tolist(),
        "F0": dgp.F0.tolist(), "FSTAR": dgp.FSTAR.tolist(), "oracle_only": True})
    try:
        learner, initial_state = prepare_rnd2_learner(history, as_of=24, warmup_as_of=8,
            window=16, path_id=0, refit_every=1, beta=.95, horizon=10,
            nodes_per_memory=5, noise_nodes=4)
    except SupportError as exc:
        episode = {**row, "status": "initialization_support_refusal", "available": False,
            "score": None, "score_kind": "unavailable", "reason": str(exc),
            "branch_status": {}, "branch_completed_steps": {},
            "all_four_comparable": False,
            "pair_comparable": {"main_fixed_action": False,
                "fixed_public_environment_both_fixed": False,
                "main_fixed_public_environment": False,
                "fixed_action_both_fixed": False},
            "full_paths_file": None,
            "main_refusal": {"time": None, "reason": str(exc), "type": type(exc).__name__}}
        return _with_deltas(episode, before_fit, before_dgp)
    report = run_four_paths(learner=learner, environment=mechanics.TrueRND2Adapter(live),
        initial_state=initial_state, start_time=24, steps=8, beta=.95)
    _save_json(out / (base + "_four_paths.json"), report)
    score = mechanics._score_main_window(report)
    if score["available"]:
        status, kind, value = "available", "finite", score["score"]
    else:
        status, kind, value = "main_support_refusal", "unavailable", None
    episode = {**row, "status": status, "available": bool(score["available"]),
        "score": value, "score_kind": kind, "reason": score["reason"],
        "full_paths_file": base + "_four_paths.json",
        "branch_status": {k: v["status"] for k, v in report["branches"].items()},
        "branch_completed_steps": {k: v["completed_steps"] for k, v in report["branches"].items()},
        "all_four_comparable": all(v["status"] == "complete" for v in report["branches"].values()),
        "pair_comparable": {"main_fixed_action": all(report["branches"][k]["status"] == "complete"
            for k in ("main", "fixed_action")),
            "fixed_public_environment_both_fixed": all(report["branches"][k]["status"] == "complete"
            for k in ("fixed_public_environment", "both_fixed")),
            "main_fixed_public_environment": all(report["branches"][k]["status"] == "complete"
            for k in ("main", "fixed_public_environment")),
            "fixed_action_both_fixed": all(report["branches"][k]["status"] == "complete"
            for k in ("fixed_action", "both_fixed"))},
        "main_refusal": (report["branches"]["main"].get("failures", [None])[0]
            if not score["available"] else None)}
    return _with_deltas(episode, before_fit, before_dgp)


def _with_deltas(episode, before_fit, before_dgp):
    episode["fit_counts_delta"] = {k: FIT_COUNTS[k] - before_fit[k] for k in FIT_COUNTS}
    episode["dgp_counts_delta"] = {k: DGP_COUNTS[k] - before_dgp[k] for k in DGP_COUNTS}
    return episode


def _save_json(path: Path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def _write_json(path: Path, value):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def _log_event(out, event):
    line = json.dumps(event, allow_nan=False)
    for name in ("PROGRESS.jsonl", "run.log"):
        with (out / name).open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def _rank(cal_slots, evaluation, ranker=None):
    if ranker is None:
        ranker = _load_ranker()
    return ranker.rank_conditional(cal_slots, evaluation, alpha=.05)


def execute(output_dir: Path, *, episode_runner=None, bank_callback=None):
    """Execute the declared multibank experiment (never called by checks)."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    mock_mode = episode_runner is not None
    if not mock_mode and EXPECTED_PACKAGE not in Path(rnd2_counterfactual.__file__).resolve().parents:
        raise RuntimeError("installed .2412 package required; got %s" % rnd2_counterfactual.__file__)
    plan, rows = load_plan_and_streams()
    if plan.get("banks") != 16 or len(rows) != 1360:
        raise ValueError("planning ledger does not match the reviewed 16-bank design")
    if mock_mode:
        dgp = mechanics = None
    else:
        dgp, mechanics = _load_dgp(), _load_mechanics()
        _install_counters(dgp)
    ranker = _load_ranker()
    config = {"design": "native_power_frontier_16bank", "plan": str(PLAN_PATH),
        "streams": str(STREAM_PATH), "environment_file": str(ENV_FILE),
        "mechanics_file": str(MECHANICS_FILE), "installed_package_file": str(rnd2_counterfactual.__file__),
        "source_status": "unavailable_not_run", "eta": "not_supplied",
        "learner": {"package": "installed PINC .2412", "beta": .95, "horizon": 10,
            "nodes_per_memory": 5, "noise_nodes": 4, "refit_every": 1,
            "as_of": 24, "warmup_as_of": 8, "window": 16},
        "prefix": {"paths": 32, "dates": 24, "source_specific_v004_constructor": True,
            "profile": "step", "switch_date": 24},
        "calibration": {"slots": 19, "attempt_cap": 3, "selection": "first_main_available"},
        "evaluation": {"per_bank_cell": 4, "first_outcome_retained": True},
        "execution_kind": "mock_protocol_check" if mock_mode else "science_run"}
    _save_json(output_dir / "CONFIG.json", {"design_only_metadata": config, "plan": plan})
    eval_by_key = {}
    cal_by_bank = {}
    for row in rows:
        row["cell_key"] = cell_key(row["cell"], row["strength"])
        if row["role"] == "CAL":
            cal_by_bank.setdefault(row["bank"], []).append(row)
        else:
            eval_by_key.setdefault((row["bank"], row["cell_key"]), []).append(row)
    banks = []
    stopped_failure = None
    start = time.monotonic()
    for bank in range(16):
        bank_record = {"bank": bank, "cells": {}, "status": "running", "calibration_slots": []}
        cell_ids = [cell_key(x["case"], x["strength"]) for x in plan["cells"]]
        bank_cal = sorted(cal_by_bank[bank], key=lambda r: (r["slot"], r["attempt"]))
        for slot in range(19):
            slot_rows = [r for r in bank_cal if r["slot"] == slot]
            attempts, accepted = [], None
            for row in slot_rows:
                if accepted is not None:
                    attempts.append({"seed": row["seed"], "status": "not_run_after_acceptance"})
                    continue
                try:
                    episode = (episode_runner(row, output_dir) if mock_mode else
                               run_episode(row, output_dir, dgp=dgp, mechanics=mechanics))
                except Exception as exc:
                    stopped_failure = {"seed": row["seed"], "role": row["role"], "bank": bank,
                        "slot": slot, "status": "software_failure", "type": type(exc).__name__,
                        "reason": str(exc), "traceback": traceback.format_exc()}
                    _save_json(output_dir / ("episode_%010d_software_failure.json" % row["seed"]), stopped_failure)
                    break
                _save_json(output_dir / ("episode_%010d_attempt.json" % row["seed"]), episode)
                attempts.append(episode)
                _log_event(output_dir, {"event": "cal_attempt", "episode": episode,
                    "fit_counts": FIT_COUNTS, "dgp_counts": DGP_COUNTS})
                if episode["available"]:
                    accepted = {"status": "accepted", "accepted_attempt": row["attempt"],
                        "accepted_seed": row["seed"], "score": episode["score"], "score_kind": "finite"}
            if stopped_failure:
                break
            if accepted is None:
                accepted = {"status": "cap_exhausted", "accepted_attempt": None,
                    "accepted_seed": None, "score": None, "score_kind": "positive_infinity"}
            bank_record["calibration_slots"].append({"slot": slot, **accepted,
                "attempts": attempts, "attempts_executed": sum(x.get("status") != "not_run_after_acceptance" for x in attempts)})
        if stopped_failure:
            bank_record["status"] = "software_failure_partial"
            banks.append(bank_record)
            _write_json(output_dir / "BANKS_PARTIAL.json", banks)
            break
        # All 28 preassigned EVALs run even if one or more CAL slots exhausted.
        for cell in cell_ids:
            bank_record["cells"][cell] = {"evals": []}
        # All 28 preassigned EVALs run even for an exhausted calibration bank.
        cal_for_rank = bank_record["calibration_slots"]
        for cell in cell_ids:
            for row in eval_by_key[(bank, cell)]:
                try:
                    episode = (episode_runner(row, output_dir) if mock_mode else
                               run_episode(row, output_dir, dgp=dgp, mechanics=mechanics))
                    episode["rank"] = _rank(cal_for_rank, episode, ranker)
                    episode["alarm"] = bool(episode["rank"]["reject"])
                except Exception as exc:
                    stopped_failure = {"seed": row["seed"], "role": row["role"], "bank": bank,
                        "cell": cell, "status": "software_failure", "type": type(exc).__name__,
                        "reason": str(exc), "traceback": traceback.format_exc()}
                    _save_json(output_dir / ("episode_%010d_software_failure.json" % row["seed"]), stopped_failure)
                    break
                _save_json(output_dir / ("episode_%010d_eval.json" % row["seed"]), episode)
                bank_record["cells"][cell]["evals"].append(episode)
                _log_event(output_dir, {"event": "eval", "episode": episode,
                    "fit_counts": FIT_COUNTS, "dgp_counts": DGP_COUNTS})
            if stopped_failure:
                break
        bank_record["status"] = "software_failure_partial" if stopped_failure else "complete"
        banks.append(bank_record)
        _write_json(output_dir / "BANKS_PARTIAL.json", banks)
        if stopped_failure:
            break
        if bank_callback is not None:
            try:
                bank_callback(output_dir, bank_record)
            except Exception as exc:
                stopped_failure = {"bank": bank, "status": "software_failure",
                    "stage": "bank_callback", "type": type(exc).__name__,
                    "reason": str(exc), "traceback": traceback.format_exc()}
                _write_json(output_dir / "CALLBACK_FAILURE.json", stopped_failure)
                break
    result = {"status": "software_failure_stopped" if stopped_failure else "complete",
        "source_status": "unavailable_not_run", "eta": "not_supplied", "banks": banks,
        "software_failure": stopped_failure, "fit_counts": FIT_COUNTS, "dgp_counts": DGP_COUNTS,
        "elapsed_seconds": time.monotonic() - start}
    _save_json(output_dir / "RESULT.json", result)
    if stopped_failure:
        _save_json(output_dir / "STOPPED.json", {"reason": "software_failure_no_replacement",
            "failure": stopped_failure, "fit_counts": FIT_COUNTS, "dgp_counts": DGP_COUNTS})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="explicitly execute learner/DGP episodes")
    parser.add_argument("--output-dir", help="new output directory; required with --run")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("no execution: pass --run only after explicit owner authorization")
    if not args.output_dir:
        parser.error("--output-dir is required with --run")
    execute(Path(args.output_dir))


if __name__ == "__main__":
    main()
