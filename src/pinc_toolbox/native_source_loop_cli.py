"""CLI for :mod:`pinc_toolbox.native_source_loop` with an explicit host file."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from .native_source_loop import run_native_source_loop
from .two_stage_report_html import render_html


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    return value


def _load_host(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("host file must be a regular Python file")
    spec = importlib.util.spec_from_file_location("pinc_explicit_host", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load explicit host file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, "create_host", None)
    if not callable(factory):
        raise ValueError("host file must define create_host(public_configuration)")
    return factory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="predeclared request JSON")
    parser.add_argument("--host-file", required=True, help="trusted Python host adapter")
    parser.add_argument("--output", required=True, help="strict JSON report path")
    parser.add_argument("--html-output", required=True, help="HTML report path")
    args = parser.parse_args(argv)
    request = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(request, dict) or not isinstance(request.get("public_host_configuration"), dict):
        parser.error("input requires a public_host_configuration object")
    factory = _load_host(args.host_file)
    try:
        host = factory(dict(request["public_host_configuration"]))
    except Exception as exc:
        failure = str(exc)
        class FailedHost:
            def release(self, *, as_of):
                raise RuntimeError("host_initialization_failed:" + failure)
            def state(self, *, as_of):
                return {"status": "unavailable", "reason": "host_initialization_failed"}
            def step(self, decision):
                raise RuntimeError("host_initialization_failed:" + failure)
        host = FailedHost()
    report = run_native_source_loop(request, host)
    payload = _jsonable(report)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    Path(args.output).write_text(encoded, encoding="utf-8")
    Path(args.html_output).write_text(render_html(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
