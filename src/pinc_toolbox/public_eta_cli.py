"""Optional JSON interface for public-history eta/source and released G.

Candidate source module: not part of the previously verified 0.1.1 wheel.
No repository imports, .mod parsing, simulations, or hidden evaluation labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _g_request(request: Mapping[str, Any]) -> dict[str, Any]:
    from .released_g import ReleasedG, fit_g_offline
    config = request["g"]
    settings = dict(memory_index=config["memory_index"],
                    physical_indices=config["physical_indices"], window=config["window"])
    if request["operation"] == "g_offline":
        stream = fit_g_offline(request["public_rows"], as_of=request["as_of"], **settings)
    else:
        stream = ReleasedG(**settings)
        for event in request["events"]:
            kind = event["kind"]
            if kind == "observe":
                stream.ingest(event["public_rows"], as_of=event["as_of"])
            elif kind == "fit":
                stream.fit(as_of=event["as_of"])
            elif kind == "predict":
                stream.issue(**{key: event[key] for key in (
                    "record_id", "states", "action", "decision_time", "outcome_release", "as_of")})
            else:
                raise ValueError(f"unknown released G event {kind!r}")
    return dict(operation=request["operation"], result=stream.snapshot(),
                source_reference="separate fixed reference; not replaced by rolling G",
                action_Q_certification="not supplied")


def run_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ValueError("request must be a JSON object")
    operation = request["operation"]
    if operation in {"g_offline", "g_stream"}:
        return _g_request(request)
    if operation == "eta_monitor":
        return _eta_request(request)
    if operation == "eta_monitor_certified":
        from .certified_eta_monitor import run_public_request
        return run_public_request(request)
    if operation == "hs_observed_likelihood":
        from .hs_observed_likelihood import run_public_request as hs_request
        return hs_request(request)
    raise ValueError(f"unsupported operation {operation!r}")


def _eta_request(request: Mapping[str, Any]) -> dict[str, Any]:
    from .eta_monitor import run_public_request
    return run_public_request(request)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.input.read_text(encoding="utf-8"))
    result = run_request(request)
    monitor_config = request.get("monitor")
    if (request.get("operation") == "eta_monitor_certified"
            and isinstance(monitor_config, Mapping)
            and monitor_config.get("source_execution_mode", "exhaustive") == "lazy"):
        from .certified_source_codec import compact_source_payload
        result = compact_source_payload(result)
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    # Write only after the whole calculation and serialization succeeded.
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
