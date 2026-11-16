"""Repeat the October stable calibration under the prescribed reward."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from run_matched import ROOT, RECOVERY, prepare_reward_free_history

TRACK = ROOT / "experiments/ecma_four_tracks_20260924/track1_production"
OLD = RECOVERY / "recovery_v003"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scalar(output):
    directory = TRACK / "direct_bellman_h8_cal_attempt001"
    sys.path.insert(0, str(directory))
    worker = load("fixed_reward_scalar_serial", directory / "driver.py")

    def setup():
        backend = worker.setup()
        prepare_reward_free_history(backend.front)
        return backend

    result = worker.run(output, setup_factory=setup)
    return result["status"] == "complete"


def paired(output):
    worker = load("fixed_reward_paired", TRACK / "paired_monitor_calibration_attempt001/runner.py")
    original_load = worker.load

    def load_compat(name, path):
        module = original_load(name, path)
        if name == "paired_cal_base":
            original = module.Science

            class AuditScience(original):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    prepare_reward_free_history(self.b.front)

            module.Science = AuditScience
        return module

    worker.load = load_compat
    # The paired CAL runner ignores the alarm and uses all eligible episodes.
    # These historical scalar/serial cutoffs only reproduce ancillary logging.
    old = json.loads((OLD / "calibration_paired/unit_0000/artifacts/RESULT.json").read_text())
    result = worker.run(output, thresholds=old["thresholds"])
    return result["status"] == "bank_complete"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("scalar", "paired"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    ok = (scalar if args.mode == "scalar" else paired)(args.out)
    print(args.mode, "complete" if ok else "invalid")
    if not ok:
        sys.exit(1)
