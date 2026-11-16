"""Original paired detection experiment with a new CM/MIX transition term."""
from __future__ import annotations

import argparse
import importlib.util
import json
import lzma
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import ModuleType

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDIT_DIR = ROOT / "note/middle_loop_deployment_20261108/fixed_reward_review_20261109/behavior_replay"
PRECISION = ROOT / "experiments/model_switch_scope_20261003/rnd2_model_involvement_precision_attempt001/run.py"
SEED_COUNT = 128
OUTPUT_CAP = 5_900_000_000
RESERVE = 1024**3


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def round_environment(original):
    """Keep all latent/noise/proxy machinery and replace only model effects."""
    module = ModuleType("round_coefficients_" + original.__name__)
    module.__dict__.update(original.__dict__)
    parent = original.Environment

    class Environment(parent):
        def step(self, date, actions):
            chosen = self._chosen_source
            x = self.state[:, date].copy()
            a = np.broadcast_to(np.asarray(actions, dtype=float), x.shape)
            g = float(self.g_profile[date])
            super().step(date, actions)
            if chosen in {"MODEL_A", "MODEL_LATENT"}:
                old_term = -.465 * g * (a - .05)
                new_term = g * (.02 + .8*a + a*a + .5*x*a)
                self.state[:, date + 1] += new_term - old_term

    module.Environment = Environment
    return module



class SourceConditioningUnavailable(RuntimeError):
    """Expected public-data support refusal after a valid first-stage alarm."""


def structural_compatibility(archive):
    """Check predeclared old and round action curves using public endpoints only."""
    raw = {}
    with lzma.open(archive, "rb") as stream:
        while True:
            event = pickle.load(stream)
            if event[0] == "end":
                break
            if event[0] == "raw" and event[1] in {
                "MAIN_OBSERVATIONS.json", "PAIRED_PROBE_ENDPOINTS.json"
            }:
                raw[event[1]] = json.loads(event[2])
    if set(raw) != {"MAIN_OBSERVATIONS.json", "PAIRED_PROBE_ENDPOINTS.json"}:
        return {"status": "diagnostic_unavailable", "reason": "public main/probe records absent"}

    groups = {}
    for row in raw["MAIN_OBSERVATIONS.json"]:
        groups[(row["path"], row["time"])] = [row]
    for row in raw["PAIRED_PROBE_ENDPOINTS.json"]:
        if row["status"] == "complete":
            key = (row["path"], row["time"])
            if key in groups:
                groups[key].append(row)

    rows = []
    for (path, date), observations in sorted(groups.items()):
        if len(observations) < 3:
            rows.append({"path": path, "date": date, "status": "fewer_than_three_observed_actions",
                         "action_count": len({r["action"] for r in observations})})
            continue
        xvalues = [float(row["state_x"]) for row in observations]
        if max(xvalues) - min(xvalues) > 1e-12:
            rows.append({"path": path, "date": date, "status": "initial_state_mismatch"})
            continue
        x = xvalues[0]
        actions = np.asarray([float(row["action"]) for row in observations])
        outcomes = np.asarray([float(row["next_x"]) for row in observations])
        if len(np.unique(actions)) < 3:
            rows.append({"path": path, "date": date, "status": "fewer_than_three_distinct_actions",
                         "action_count": int(len(np.unique(actions)))})
            continue
        old_curve = .8*actions - 2*actions**2 + .5*x*actions
        g = 1.0 if date >= 32 else 0.0
        round_curve = old_curve + g*(.02 + .8*actions + actions**2 + .5*x*actions)
        def line_error(curve):
            corrected = outcomes - curve
            design = np.column_stack((np.ones(len(actions)), actions))
            fitted, *_ = np.linalg.lstsq(design, corrected, rcond=None)
            return float(np.max(np.abs(corrected - design @ fitted)))
        old_error = line_error(old_curve)
        round_error = line_error(round_curve)
        rows.append({"path": path, "date": date, "status": "checked",
                     "action_count": int(len(np.unique(actions))),
                     "old_no_model_max_affine_residual": old_error,
                     "round_model_max_affine_residual": round_error})
    checked = [row for row in rows if row["status"] == "checked"]
    tol = 5e-10
    return {"status": "computed" if checked else "diagnostic_unavailable",
            "tolerance": tol, "groups": rows, "checked_groups": len(checked),
            "old_no_model_compatible_groups": sum(r["old_no_model_max_affine_residual"] <= tol for r in checked),
            "round_model_compatible_groups": sum(r["round_model_max_affine_residual"] <= tol for r in checked),
            "interpretation": "public structural compatibility diagnostic only; not a calibrated rank, score, latent observation, or source label"}


def campaign_summary(plan, results):
    families = ("N", "CL_Q100", "CM", "MIX")
    summary = {}
    for family in families:
        assigned = [row for row in plan if row["candidate"] == family]
        cell = {"assigned": len(assigned), "first_layer_alarm": 0, "first_layer_no_alarm": 0,
                "monitor_support_unavailable": 0, "second_stage_model_report": 0,
                "source_unavailable": 0, "source_unresolved_after_alarm": 0,
                "technical_failure": 0, "not_run": 0,
                "false_model_report": 0, "missed_model_report": 0}
        for row in assigned:
            result = results.get(row["ordinal"])
            if result is None or result.get("execution_status") == "not_run":
                cell["not_run"] += 1
                continue
            if result.get("execution_status") == "invalid":
                cell["technical_failure"] += 1
            monitor_status = result.get("monitor_status")
            if monitor_status == "alarm":
                cell["first_layer_alarm"] += 1
            elif monitor_status == "no_alarm":
                cell["first_layer_no_alarm"] += 1
            elif monitor_status == "support_unavailable":
                cell["monitor_support_unavailable"] += 1
            model_report = result.get("decision") == "model_involvement"
            if model_report:
                cell["second_stage_model_report"] += 1
                if family in ("N", "CL_Q100"):
                    cell["false_model_report"] += 1
            if family in ("CM", "MIX") and not model_report and result.get("execution_status") != "not_run":
                cell["missed_model_report"] += 1
            if result.get("source_stage_status") == "source_unavailable":
                cell["source_unavailable"] += 1
            if monitor_status == "alarm" and not model_report:
                cell["source_unresolved_after_alarm"] += 1
        if cell["assigned"] != SEED_COUNT:
            raise ValueError("family denominator changed")
        cell["conditional_model_report_given_alarm"] = (
            cell["second_stage_model_report"] / cell["first_layer_alarm"]
            if cell["first_layer_alarm"] else None)
        summary[family] = cell
    return summary

def build_worker():
    sys.path.insert(0, str(AUDIT_DIR))
    import run_matched
    precision = load("round_precision_design", PRECISION)
    driver = precision.build_worker()
    run_matched.install_evaluator_compatibility(driver)
    original_load = driver._load

    def local_load(name, path):
        module = original_load(name, path)
        if name == "paired16_expanded":
            factory = module.science_class

            def science_class(base):
                parent = factory(base)

                class RoundScience(parent):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.dgp = round_environment(self.dgp)

                    def condition(self, selected_index):
                        try:
                            return super().condition(selected_index)
                        except ValueError as exc:
                            if str(exc) != "observed endpoints violate common-noise affine action relation":
                                raise
                            diagnostic = structural_compatibility(self.selected_archive)
                            self.write_json(self.out / "STRUCTURAL_DIAGNOSTIC.json", diagnostic)
                            if diagnostic.get("status") == "computed":
                                raise SourceConditioningUnavailable(
                                    "public acquired endpoints violate the legacy paired-conditioning relation; "
                                    "the predeclared old/round structural diagnostics are recorded separately"
                                ) from exc
                            raise

                return RoundScience

            module.science_class = science_class
        return module

    driver._load = local_load
    driver.load_corrected_stage1 = run_matched.replayed_thresholds
    return precision, driver


def ordinary():
    source = ROOT / "DGP/RND/02_state_dependent_local/model_source_v004/environment.py"
    original = load("round_ordinary_dgp", source)
    new = round_environment(original)
    checks = []
    for family in ("STABLE", "LATENT_F", "MODEL_A", "MODEL_LATENT"):
        env = new.Environment(3260040000, family, paths=32, horizon=64, switch_date=32)
        control = original.Environment(3260040000, family, paths=32, horizon=64, switch_date=32)
        for t in range(34):
            actions = np.linspace(0, .1, 32)
            x = env.state[:, t].copy()
            z = env.latent[:, t].copy()
            env.step(t, actions)
            control.step(t, actions)
            if t < 32 or family in {"STABLE", "LATENT_F"}:
                np.testing.assert_array_equal(env.state[:, t+1], control.state[:, t+1])
            if t >= 32 and family in {"MODEL_A", "MODEL_LATENT"}:
                expected = ((.8+actions)*x + 1.6*actions - actions**2 + .02
                            + z @ original.ETA + actions*(z @ original.LAMBDA)
                            + env.state_noise[:, t])
                np.testing.assert_allclose(env.state[:, t+1], expected, atol=1e-13, rtol=0)
            np.testing.assert_array_equal(env.latent[:, t+1], control.latent[:, t+1])
            np.testing.assert_array_equal(env.w[:, t+1], control.w[:, t+1])
            np.testing.assert_allclose(env.released_reward(t+1),
                                       x-.25*x*x-2.025*actions-.5*actions*actions,
                                       atol=1e-14, rtol=0)
        checks.append(family)
    receipt = {"status": "passed", "families": checks, "transitions": 34*32*4,
               "same_prefix_and_nulls": True, "latent_and_proxy_unchanged": True,
               "new_model_equation": True, "prescribed_reward": True}
    (HERE / "ORDINARY.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


def worker(ordinal, output):
    precision, driver = build_worker()
    row = precision.plan_rows()[ordinal]
    result = driver.worker(row, output, 256*1024**2, 360)
    failure = result.get("failure") or {}
    if (result.get("execution_status") == "invalid"
            and failure.get("type") == "SourceConditioningUnavailable"
            and "public acquired endpoints violate the legacy paired-conditioning relation" in failure.get("reason", "")):
        # Preserve the failure trace as an expected second-stage support refusal.
        result["execution_status"] = "complete"
        result["source_stage_status"] = "source_unavailable"
        result["source_conditioning_refusal"] = failure
        result["failure"] = None
        result["decision"] = "unresolved"
        result["latent"] = "unresolved"
        result["inference"] = {"decision": "unresolved", "status": "source_unavailable",
                                "reason": "legacy acquired-pair physics did not support the observed endpoints"}
        result["conditional_banks"] = "not_run_after_source_unavailable"
        (Path(output) / "RESULT.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result.get(key) for key in
                      ("ordinal", "candidate", "execution_status", "monitor_status", "source_stage_status", "decision")}), flush=True)
    return result


def queue(directory):
    directory.mkdir(parents=True, exist_ok=False)
    precision = load("round_queue_design", PRECISION)
    plan = precision.plan_rows()
    (directory / "PLAN.json").write_text(json.dumps({"status": "predeclared", "assigned": plan,
        "per_family_denominator": SEED_COUNT, "total_output_cap_bytes": OUTPUT_CAP,
        "reserve_bytes": RESERVE, "per_history_output_cap_bytes": 256*1024**2,
        "first_layer": "original scalar/serial/paired union through dates 32-39",
        "second_layer": "original conditional future rank where compatible; else retained source_unavailable",
        "no_replacement": True}, indent=2)+"\n")
    results = {}
    started = time.monotonic()
    (directory / "RUN.json").write_text(json.dumps({"status":"running","pid":os.getpid(),
        "started_unix":time.time(),"assigned":512},indent=2)+"\n")
    status = "running"
    for row in plan:
        used = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        if used >= OUTPUT_CAP or shutil.disk_usage(directory).free < RESERVE + 256*1024**2:
            status = "resource_stopped"
            break
        if time.monotonic() - started > 5*3600:
            status = "time_budget_stopped"
            break
        output = directory / f"history_{row['ordinal']:04d}"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(row["ordinal"]),
                   "--output", str(output)]
        with (directory / "WORKERS.log").open("a") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        path = output / "RESULT.json"
        if path.is_file():
            results[row["ordinal"]] = json.loads(path.read_text())
        summary = campaign_summary(plan, results)
        progress = {"status": status, "completed_receipts": len(results), "assigned": 512,
                    "elapsed_seconds": time.monotonic()-started, "summary": summary,
                    "last_ordinal": row["ordinal"]}
        (directory / "PROGRESS.json").write_text(json.dumps(progress, indent=2)+"\n")
        if row["ordinal"] not in results:
            status = "technical_stopped"
            break
        # Terminal invalid receipts remain in the fixed denominator; the next
        # preassigned slot still runs unless a time or resource boundary stops us.
    else:
        status = "complete"
    final = {"status": status, "completed_receipts": len(results), "assigned": 512,
             "summary": campaign_summary(plan, results), "elapsed_seconds": time.monotonic()-started}
    (directory / "FINAL.json").write_text(json.dumps(final, indent=2)+"\n")
    run_receipt = {"status":status,"pid":os.getpid(),"finished_unix":time.time(),
                   "completed_receipts":len(results),"assigned":512}
    (directory / "RUN.json").write_text(json.dumps(run_receipt,indent=2)+"\n")
    print(json.dumps(final), flush=True)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ordinary", action="store_true")
    mode.add_argument("--worker", type=int)
    mode.add_argument("--queue", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ordinary:
        ordinary()
    elif args.worker is not None:
        result = worker(args.worker, args.output)
        if result["execution_status"] != "complete":
            raise SystemExit(1)
    else:
        queue(args.output)


if __name__ == "__main__":
    main()
