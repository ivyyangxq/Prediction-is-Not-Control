"""JSON command-line bridge to the shared two-stage report API.

This adapter accepts an already computed Stage-1 result. It does not estimate
or reinterpret the alarm, and it does not claim conditional source guarantees.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_CONTROLLER_FIELDS = frozenset({"stage1", "context", "source", "readiness", "available"})
_RESET_FIELDS = frozenset({
    "actions", "detection_reference", "detection_windows", "source_reference",
    "source_window", "sigma", "alpha_detection", "alpha_source", "latent_bound",
    "model_bound", "source_deadline", "assumptions",
})
_ONLINE_FIELDS = frozenset({"detection_reference", "detection_windows", "source_reference",
    "source_window", "sigma", "source_deadline", "alpha_detection_reference",
    "alpha_detection_noise", "alpha_source_reference", "alpha_source_noise",
    "latent_bound", "model_bound", "assumptions"})
_HS_FIELDS = frozenset({"stage1", "candidate_evidence", "family_members", "active", "release", "context"})
_OPERATIONS = {"hs_candidate_two_stage", "controller", "reset_bin_two_stage", "online_gq_two_stage", "raw_history_two_stage", "eta_online_gq_two_stage"}
_STAGE1 = frozenset({"alarm", "no_alarm", "unavailable", "unresolved"})


def _read_from_mapping(value):
    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    operation = value.get("operation", "controller")
    if operation not in _OPERATIONS:
        raise ValueError(f"operation must be one of {sorted(_OPERATIONS)}")
    fields = {"hs_candidate_two_stage": _HS_FIELDS, "controller": _CONTROLLER_FIELDS, "reset_bin_two_stage": _RESET_FIELDS,
              "online_gq_two_stage": _ONLINE_FIELDS,
              "eta_online_gq_two_stage": _ONLINE_FIELDS | {"domain", "eta_selection"},
              "raw_history_two_stage": frozenset({"request", "window_index"})}
    allowed = fields[operation] | {"operation"}
    unknown = set(value).difference(allowed)
    if unknown:
        raise ValueError(f"unknown request fields: {sorted(unknown)}")
    if operation == "hs_candidate_two_stage":
        return value
    if operation in {"online_gq_two_stage", "raw_history_two_stage", "eta_online_gq_two_stage"}:
        return value
    if operation == "reset_bin_two_stage":
        required = {"actions", "detection_reference", "detection_windows", "source_reference",
                    "source_window", "sigma", "source_deadline"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"reset_bin_two_stage request missing fields: {sorted(missing)}")
        return value
    if "stage1" not in value or "context" not in value:
        raise ValueError("request requires stage1 and context")
    stage1, context = value["stage1"], value["context"]
    if not isinstance(stage1, dict) or stage1.get("status") not in _STAGE1:
        raise ValueError(f"stage1.status must be one of {sorted(_STAGE1)}")
    if not isinstance(context, dict):
        raise ValueError("context must be a JSON object")
    if "source" in value and value["source"] is not None and not isinstance(value["source"], dict):
        raise ValueError("source must be a JSON object when supplied")
    if "readiness" in value and not isinstance(value["readiness"], dict):
        raise ValueError("readiness must be a JSON object")
    if "available" in value and value["available"] is not None and not isinstance(value["available"], list):
        raise ValueError("available must be a JSON list or null")
    return value


def _read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return _read_from_mapping(json.load(handle))


def execute(request):
    """Run the shared report API with a validated request mapping."""
    request = _read_from_mapping(request)
    if request.get("operation") == "hs_candidate_two_stage":
        from .hs_candidate_two_stage import run_hs_candidate_two_stage
        from .two_stage_report import render_report
        report = run_hs_candidate_two_stage(**{k:v for k,v in request.items() if k != "operation"})
        return {"report": report, "text": render_report(report)}
    if request.get("operation") in {"online_gq_two_stage", "raw_history_two_stage", "eta_online_gq_two_stage"}:
        from .two_stage_report import render_report
        if request["operation"] == "eta_online_gq_two_stage":
            from .eta_online_gq_two_stage import run_eta_online_gq_two_stage as runner
            if "eta_selection" in request:
                if "domain" in request:
                    raise ValueError("supply domain or eta_selection, not both")
                from .eta_domain import select_eta_domain
                request = dict(request)
                request["domain"] = select_eta_domain(**request.pop("eta_selection"))
        elif request["operation"] == "online_gq_two_stage":
            from .online_gq_two_stage import run_online_gq_two_stage as runner
        else:
            from .raw_history_two_stage import run_raw_history_two_stage as runner
        report = runner(**{k:v for k,v in request.items() if k != "operation"})
        return {"report": report, "text": render_report(report)}
    if request.get("operation") == "reset_bin_two_stage":
        from .reset_bin_two_stage import run_reset_bin_two_stage
        from .two_stage_report import render_report

        kwargs = {key: value for key, value in request.items() if key != "operation"}
        report = run_reset_bin_two_stage(**kwargs)
        return {"report": report, "text": render_report(report)}

    from .two_stage_report import run_two_stage_report, render_report

    report = run_two_stage_report(
        request["stage1"],
        context=request["context"],
        source=request.get("source"),
        readiness=request.get("readiness"),
        available=request.get("available"),
    )
    return {"report": report, "text": render_report(report)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-output", type=Path)
    parser.add_argument("--html-output", type=Path)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must be distinct")
    if args.text_output is not None and args.text_output.resolve() in {
        args.input.resolve(), args.output.resolve()
    }:
        parser.error("--text-output must be distinct from --input and --output")
    if args.html_output is not None:
        occupied = {args.input.resolve(), args.output.resolve()}
        if args.text_output is not None:
            occupied.add(args.text_output.resolve())
        if args.html_output.resolve() in occupied:
            parser.error("--html-output must be distinct from all other paths")
    try:
        result = execute(_read(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result["report"], indent=2, sort_keys=True,
                                          allow_nan=False) + "\n", encoding="utf-8")
        if args.text_output is not None:
            args.text_output.parent.mkdir(parents=True, exist_ok=True)
            args.text_output.write_text(result["text"] + "\n", encoding="utf-8")
        if args.html_output is not None:
            from .two_stage_report_html import render_html
            args.html_output.parent.mkdir(parents=True, exist_ok=True)
            args.html_output.write_text(render_html(result["report"]), encoding="utf-8")
        print(result["text"])
        return 0
    except Exception as error:
        print(f"two-stage report failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
