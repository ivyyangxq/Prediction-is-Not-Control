"""JSON CLI for the finite-candidate public residual source statistic.

``residual_candidate_source`` reads an absolute caller-supplied NPZ containing
complete evaluation/calibration paths and fixed candidate/reference arrays.
It calls the existing ``score_residual_candidate`` statistic, writes its
array-valued evidence to a fresh NPZ artifact, and returns only compact JSON
summary metadata.  ``residual_candidate_route`` accepts already serialized
candidate evidence and calls the existing common source router.  Neither
operation generates data, fits a model, evaluates truth, uses a likelihood,
or calls a source LP.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any


_OPERATION = "residual_candidate_source"
_ROUTE_OPERATION = "residual_candidate_route"
_SCHEMA = "residual_candidate_source.v1"
_VERSION = "1"


def _error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _package_version() -> str:
    try:
        return importlib_metadata.version("pinc-toolbox")
    except importlib_metadata.PackageNotFoundError:
        return "uninstalled"


def _absolute_file(value: Any, name: str, *, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{name} must be an absolute POSIX path")
    path = Path(value)
    if must_exist and (not path.exists() or not path.is_file()):
        raise FileNotFoundError(f"{name} does not identify a file: {path}")
    return path


def _settings(request: Mapping[str, Any]) -> dict[str, Any]:
    settings = request.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("request.settings must be an object")
    monitor_start = settings.get("monitor_start")
    horizon = settings.get("horizon")
    weight_index = settings.get("weight_index")
    if isinstance(monitor_start, bool) or not isinstance(monitor_start, int) or monitor_start < 0:
        raise ValueError("settings.monitor_start must be a nonnegative integer")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("settings.horizon must be a positive integer")
    if isinstance(weight_index, bool) or not isinstance(weight_index, int):
        raise ValueError("settings.weight_index must be an integer")
    alpha = float(settings.get("alpha", 0.05))
    if not 0 < alpha < 1:
        raise ValueError("settings.alpha must lie in (0,1)")
    etas = settings.get("etas")
    if not isinstance(etas, list) or not etas:
        raise ValueError("settings.etas must be a nonempty list")
    if not isinstance(settings.get("return_trace", True), bool):
        raise ValueError("settings.return_trace must be boolean")
    return {
        "monitor_start": monitor_start, "horizon": horizon,
        "etas": list(etas), "weight_index": weight_index,
        "alpha": alpha, "return_trace": settings.get("return_trace", True),
    }


def _npz_arrays(data_path: Path) -> dict[str, Any]:
    import numpy as np

    required = (
        "evaluation_full_paths", "calibration_full_paths", "candidate_F", "candidate_d",
        "reference_F", "reference_d", "residual_left_inverse",
        "support_low", "support_high", "support_width",
    )
    with np.load(data_path, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError(f"data_path NPZ is missing required arrays: {missing}")
        arrays = {key: archive[key] for key in required}
    return arrays


def _write_artifact(path: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    if path.exists():
        raise FileExistsError(f"artifact_output already exists: {path}")
    raw_arrays = result.get("arrays")
    if not isinstance(raw_arrays, Mapping) or not raw_arrays:
        raise ValueError("statistic returned no array evidence")
    arrays = {str(name): np.asarray(value) for name, value in raw_arrays.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return {
        "path": str(path), "array_count": len(arrays),
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
        "bytes": path.stat().st_size,
    }


def _module_paths() -> dict[str, str]:
    candidate = importlib.import_module("pinc_toolbox.residual_candidate_source")
    energy = importlib.import_module("pinc_toolbox.residual_energy")
    diagnosis = importlib.import_module("pinc_toolbox.diagnosis")
    return {name: str(Path(module.__file__).resolve()) for name, module in (
        ("residual_candidate_source", candidate), ("residual_energy", energy),
        ("diagnosis", diagnosis),
    )}


def _source_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if request.get("operation") != _OPERATION:
        raise ValueError(f"operation={_OPERATION!r} is required")
    data_path = _absolute_file(request.get("data_path"), "data_path")
    artifact = _absolute_file(request.get("artifact_output"), "artifact_output", must_exist=False)
    settings = _settings(request)
    arrays = _npz_arrays(data_path)
    from .residual_candidate_source import score_residual_candidate

    result = score_residual_candidate(
        arrays["evaluation_full_paths"], arrays["calibration_full_paths"],
        reference_coefficients={"F": arrays["reference_F"], "d": arrays["reference_d"]},
        candidate_F=arrays["candidate_F"], candidate_d=arrays["candidate_d"],
        monitor_start=settings["monitor_start"], horizon=settings["horizon"],
        residual_left_inverse=arrays["residual_left_inverse"],
        support=(arrays["support_low"], arrays["support_high"], arrays["support_width"]),
        etas=settings["etas"], weight_index=settings["weight_index"],
        alpha=settings["alpha"], return_trace=settings["return_trace"],
    )
    artifact_metadata = _write_artifact(artifact, result)
    import numpy as np
    eta_values = np.asarray(result.get("etas", ())).reshape(-1)
    eta_summary = [{"eta": float(eta)} for eta in eta_values]
    array_summary = {
        str(name): {"shape": list(np.asarray(value).shape),
                    "dtype": str(np.asarray(value).dtype)}
        for name, value in result["arrays"].items()
    }
    return {
        "schema": _SCHEMA, "version": _VERSION, "package_version": _package_version(),
        "status": "ready", "operation": _OPERATION,
        "artifact": artifact_metadata,
        "summary": {"etas": eta_summary, "arrays": array_summary,
                     "statistic": result["statistic"],
                     "calibration_scope": result["calibration_scope"]},
        "settings": settings, "data_path": str(data_path),
        "module_paths": _module_paths(), "python": sys.executable,
        "python_version": platform.python_version(),
        "source_scope": "fixed candidate residual statistic; no likelihood, LP, DGP, truth, or Q",
    }


def _route_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if request.get("operation") != _ROUTE_OPERATION:
        raise ValueError(f"operation={_ROUTE_OPERATION!r} is required")
    from .residual_candidate_source import route_residual_candidates

    required = ("candidate_evidence", "family_members", "ordinary_alarm", "active", "release")
    missing = [key for key in required if key not in request]
    if missing:
        raise ValueError(f"route request is missing fields: {missing}")
    result = route_residual_candidates(
        request["candidate_evidence"], request["family_members"],
        request["ordinary_alarm"], request["active"], request["release"],
    )
    return {
        "schema": _SCHEMA, "version": _VERSION, "package_version": _package_version(),
        "status": "ready", "operation": _ROUTE_OPERATION,
        "result": result, "module_paths": _module_paths(),
        "python": sys.executable, "python_version": platform.python_version(),
        "source_scope": "existing residual candidate evidence routed through common diagnosis; no new statistic",
    }


def run_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ValueError("request must be a JSON object")
    if request.get("operation") == _OPERATION:
        return _source_request(request)
    if request.get("operation") == _ROUTE_OPERATION:
        return _route_request(request)
    raise ValueError("operation must be residual_candidate_source or residual_candidate_route")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; supply a fresh output path")
    request: Any = None
    try:
        request = json.loads(args.input.read_text(encoding="utf-8"))
        result = run_request(request)
    except Exception as error:
        result = {
            "schema": _SCHEMA, "version": _VERSION, "package_version": _package_version(),
            "status": "error",
            "operation": request.get("operation") if isinstance(request, Mapping) else None,
            "error": _error(error),
            "source_scope": "request failed before a scientific result; no truth or fallback",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_request", "main"]
