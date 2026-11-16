"""Existing certified public-history measurements with strict two-stage publication."""
from .raw_history_source_workflow import prepare_raw_history
from .certified_source_workflow_backend import CertifiedSourceEvaluator
from .two_stage_report import run_two_stage_report


def run_raw_history_two_stage(request, *, window_index):
    """Run the existing eta/reference/region path and its exact C0 detector.

    This same-data route intentionally leaves conditional source guarantees
    unavailable; no independent-reference theorem is attached to it.
    """
    prepared = prepare_raw_history(request, window_index=window_index)
    context = {"route": "certified_public_history_C0", "window_index": window_index,
               "as_of": request["as_of"], "monitor": request["monitor"],
               "reference": request["reference"],
               "source_deadline": request["as_of"],
               "reference_mode": "fixed_prechange_not_rolling_behavior_G"}
    evaluator = None
    c0 = None
    if prepared["readiness"]["status"] == "ready":
        evaluator = CertifiedSourceEvaluator(prepared["source"])
        c0 = evaluator("C0")
    status = ({"excluded": "alarm", "compatible": "no_alarm", "unresolved": "unresolved"}
              .get((c0 or {}).get("status"), "unavailable"))
    stage1 = {"status": status, "event": "C0 excluded by existing certified public-history region",
              "evidence": c0, "window_index": window_index,
              "date": request["as_of"],
              "first_alarm": None, "first_alarm_status": "single_requested_window_not_calendar_scan"}
    result = run_two_stage_report(stage1, context=context, evaluate=evaluator,
                                  readiness=prepared["readiness"])
    result["preparation"] = prepared
    result["reliability"]["conditional_source_reason"] = "same_data_selection_and_reference_not_independent"
    return result
