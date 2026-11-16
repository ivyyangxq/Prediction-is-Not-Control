"""Same-seed deployment parity replay, with the old thresholds as audit inputs."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRECISION = ROOT / "experiments/model_switch_scope_20261003/rnd2_model_involvement_precision_attempt001/run.py"
RECOVERY = ROOT / "codex/physical_g_repair_20261017/runs"


def historic(ordinal):
    for version in ("recovery_v004", "recovery_v003"):
        directory = RECOVERY / version / f"precision512/unit_{ordinal:04d}/artifacts"
        if (directory / "RESULT.json").is_file():
            saved = json.loads((directory / "RESULT.json").read_text())
            if saved["execution_status"] == "complete":
                return directory, saved
    raise ValueError(f"No completed October physical-G comparator for {ordinal}")


def prepare_reward_free_history(front):
    """Keep archive clocks and physical checks on the reward-free learner."""
    import numpy as np

    def validate_history(frame):
        columns = ["path_id", "period", "release_period", "state", "w",
                   "previous_action", "action", "next_state", "next_w"]
        columns += [name for name in ("delta_gamma", "delta_m", "residual",
                    "residual_source_period", "auxiliary_as_of") if name in frame]
        if not np.isfinite(frame[columns].to_numpy(float)).all():
            raise ValueError("nonfinite released training history")
        if frame.duplicated(["path_id", "period"]).any() or not np.array_equal(frame.release_period, frame.period + 1):
            raise ValueError("duplicate or mistimed training history")
        if not np.equal(frame.period, np.floor(frame.period)).all():
            raise ValueError("invalid period")
        if "residual_source_period" in frame and not np.array_equal(frame.residual_source_period, frame.period - 1):
            raise ValueError("invalid residual clock")
        return True

    front.validate_history = validate_history
    observe = front.Panel.observe

    def observe_without_label(panel, *args, **kwargs):
        result = observe(panel, *args, **kwargs)
        panel.learner.history = panel.learner.history.drop(columns=["reward"], errors="ignore")
        return result

    front.Panel.observe = observe_without_label


def install_evaluator_compatibility(driver):
    load = driver._load

    def load_with_compatibility(name, path):
        module = load(name, path)
        if name != "paired16_expanded":
            return module
        factory = module.science_class

        def science_class(base):
            parent = factory(base)

            class AuditScience(parent):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    prepare_reward_free_history(self.b.front)
                    # The physical learner-history snapshot loses only reward.
                    # Keep the separate raw release and its deterministic-payoff check.
                    path = Path(self.extractor.__file__)
                    code = path.read_text()
                    line = " expected=HISTORY_COLUMNS if RELEASE_METADATA in columns else LEGACY_HISTORY_COLUMNS"
                    crosscheck = " close(actual[BASE_COLUMNS].to_numpy(),expected[BASE_COLUMNS].to_numpy(),'learner/public physical history')"
                    assert code.count(line) == code.count(crosscheck) == 1
                    code = code.replace(line, line + "\n expected=[c for c in expected if c!='reward']")
                    code = code.replace(crosscheck, " physical=[c for c in BASE_COLUMNS if c!='reward']\n close(actual[physical].to_numpy(),expected[physical].to_numpy(),'learner/public physical history')")
                    exec(compile(code, str(path) + "::fixed_reward_audit", "exec"), self.extractor.__dict__)

            return AuditScience

        module.science_class = science_class
        return module

    driver._load = load_with_compatibility


def replayed_thresholds():
    scalar = json.loads((HERE / "cal_scalar_serial_001/RESULT.json").read_text())
    paired = json.loads((HERE / "cal_paired_001/RESULT.json").read_text())
    if scalar["status"] != "complete" or paired["status"] != "bank_complete":
        raise ValueError("both fixed-reward calibration runs must complete")
    scores = [item["result"]["scores"] for item in scalar["assigned"]
              if item["status"] == "available"]
    if len(scores) != 59 or len(paired["bank"]) != 119:
        raise ValueError("expected 59 accepted scalar/serial slots and 119 qualified paired scores")
    return dict(scalar=max(item["scalar"] for item in scores),
                serial=max(item["serial"] for item in scores),
                paired_bank=paired["bank"])


def run(ordinal, output, use_replayed_calibration=False):
    spec = importlib.util.spec_from_file_location("fixed_reward_precision_replay", PRECISION)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    row = module.plan_rows()[ordinal]
    if use_replayed_calibration:
        thresholds = replayed_thresholds()
        receipt = {"threshold_use": "fixed-reward replayed stable calibration",
                   "scalar_serial": "cal_scalar_serial_001/RESULT.json",
                   "paired": "cal_paired_001/RESULT.json"}
    else:
        directory, saved = historic(ordinal)
        for key in ("ordinal", "candidate", "seed"):
            assert row[key] == saved[key]
        assert row["conditional_future_campaign_entropy"] == saved["entropy"]
        thresholds = saved["thresholds"]
        receipt = {"source": str(directory.relative_to(ROOT)),
                   "threshold_use": "held fixed for historical parity; no new error-rate claim"}
    d = module.build_worker()
    install_evaluator_compatibility(d)
    # The old thresholds are held constant solely to reproduce the historical
    # alarm selection. The normal calibration validation is left untouched.
    # Do not publish these ranks as a calibration for the current deployment.
    d.load_corrected_stage1 = lambda: thresholds
    result = d.worker(row, output, 256 * 1024**2, 360)
    (output / "AUDIT_COMPARATOR.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(ordinal, result["execution_status"], result["monitor_status"], result["decision"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ordinal", type=int)
    parser.add_argument("out", type=Path)
    parser.add_argument("--replayed-calibration", action="store_true")
    args = parser.parse_args()
    if run(args.ordinal, args.out, args.replayed_calibration)["execution_status"] != "complete":
        sys.exit(1)
