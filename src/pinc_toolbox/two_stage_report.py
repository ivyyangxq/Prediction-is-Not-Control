"""Two-stage publication controller; statistical validity belongs to the route.

Source necessity is relative to the supplied CL/CM/CMIX families, never a
posterior probability. The legacy exploratory router remains unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import json
from .diagnosis import diagnose_sources, _evidence
from .calibration_rejection_capacity import calibration_rejection_capacity


def _strict(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _errors(condition):
    return {kind: {"conditioning": condition, "target_budget": None,
                   "proved_bound": None, "empirical": None,
                   "status": "not_established" if kind == "type_i" else "not_estimated"}
            for kind in ("type_i", "type_ii")}


def run_two_stage_report(stage1, *, context, source=None, evaluate=None,
                         readiness=None, available=None):
    """Publish only after an affirmative detector record.

    With ``context.route="conditional_history_saved_report"``, ``source`` is a
    saved protocol terminal; this opt-in route only validates and reports it.
    Optional maintained assumptions belong to context and are not verified.
    Otherwise ``source`` is an exact CertifiedSourceEvaluator specification. Alternatively
    supply an evidence callback. Detector and source context alignment is the
    calling measurement route's responsibility. No statistical guarantee can be
    inferred from caller evidence, solver certification, or an alarm alone.
    """
    if not isinstance(stage1, dict) or stage1.get("status") not in {
            "alarm", "no_alarm", "unavailable", "unresolved"}:
        raise ValueError("stage1 requires an explicit valid status")
    if not isinstance(context, dict):
        raise ValueError("context must be a mapping")
    if context.get("route") == "conditional_history_saved_report":
        from .conditional_history_saved_report import adapt
        if any(value is not None for value in (evaluate, readiness, available)):
            raise ValueError("saved-terminal route does not accept evaluate, readiness or available")
        if set(context).difference({"route", "history_reference", "detector_reference", "maintained_assumptions"}):
            raise ValueError("unknown saved-terminal context fields")
        if set(stage1).difference({"status", "H"}):
            raise ValueError("saved-terminal stage1 accepts only status and optional H")
        report = adapt(source,
                       context={k: v for k, v in context.items()
                                if k in {"history_reference", "detector_reference"}},
                       maintained_assumptions=context.get("maintained_assumptions"))
        if stage1["status"] != report["stage1"]["status"]:
            raise ValueError("stage1.status disagrees with saved terminal")
        if "H" in stage1 and (type(stage1["H"]) is not type(report["stage1"].get("H"))
                              or stage1["H"] != report["stage1"].get("H")):
            raise ValueError("stage1.H disagrees with saved terminal")
        return report
    if source is not None and evaluate is not None:
        raise ValueError("supply source or evaluate, not both")
    if readiness is not None and not isinstance(readiness, dict):
        raise ValueError("readiness must be a mapping")
    result = {
        "context": _strict(context), "stage1": _strict(stage1),
        "stage2": {"status": "not_entered", "latent": "not_entered",
                   "model": "not_entered", "mixed": False,
                   "outcome": "not_entered", "evidence": {},
                   "trace": {"visited": [], "inherited": [],
                             "not_run": ["CL", "CM", "CMIX"]}},
        "errors": {"detection": _errors("unconditional"),
                   "latent": _errors("conditional_on_alarm"),
                   "model": _errors("conditional_on_alarm"),
                   "mixed": _errors("conditional_on_alarm"),
                   "joint": _errors("unconditional")},
        "reliability": {"statistical_coverage": "not_established",
                        "numerical_certification": "only_given_region_and_candidate_geometry",
                        "source_probability": {"status": "unavailable", "value": None},
                        "assumptions": []},
        "completion": {"status": "complete", "failures": []},
        "action_comparison": {"status": "unavailable", "reason": "no_matched_eta_Q_input"},
    }
    # Counts supplied by the measurement route describe this rank's failure floor.
    # Do not infer counts from generic incompletion, or alter the detector decision.
    if all(key in stage1 for key in ("cal_total_B", "cal_failed_count", "alpha")):
        result["stage1"].update(calibration_rejection_capacity(
            cal_total_B=stage1["cal_total_B"],
            cal_failed_count=stage1["cal_failed_count"], alpha=stage1["alpha"]))
    if stage1["status"] != "alarm":
        if stage1["status"] != "no_alarm":
            result["completion"]["status"] = "incomplete"
        return result
    stage2 = result["stage2"]
    stage2.update(status="unavailable", latent="unresolved", model="unresolved", outcome="unresolved")
    if readiness is not None and readiness.get("status") not in {"ready", "available", "supported"}:
        result["completion"].update(status="incomplete", failures=[_strict(readiness)])
        return result
    if source is None and evaluate is None:
        result["completion"].update(status="incomplete", failures=[{"reason": "source_not_supplied"}])
        return result
    failures = result["completion"]["failures"]
    try:
        if evaluate is None:
            from .certified_source_workflow_backend import CertifiedSourceEvaluator
            evaluate = CertifiedSourceEvaluator(source)
    except Exception as exc:
        failures.append({"reason": "source_initialization_failed", "error": str(exc)})
        result["completion"]["status"] = "incomplete"
        return result

    def retained(node):
        try:
            return _evidence(node, evaluate(node))
        except Exception as exc:
            failures.append({"node": node, "reason": "evaluation_failed", "error": str(exc)})
            return {"status": "unresolved", "reason": "evaluation_failed"}

    # Source-stage C0 is not the external detector and must not suppress CL/CM.
    allowed = ["CL", "CM", "CMIX"] if available is None else list(available)
    if set(allowed).difference({"C0", "CL", "CM", "CMIX"}):
        raise ValueError("unknown source class")
    allowed = [name for name in allowed if name != "C0"]
    routed = diagnose_sources(retained, alarm=True, available=allowed, goal="full")
    evidence = routed["evidence"]
    mix = evidence["CMIX"]["status"]
    for component, pure in (("latent", "CM"), ("model", "CL")):
        status = evidence[pure]["status"]
        stage2[component] = ("required" if status == "excluded" and mix == "compatible"
                             else "not_established" if status == "compatible" else "unresolved")
    stage2.update(status="reported", outcome=routed["outcome"],
                  mixed=stage2["latent"] == stage2["model"] == "required",
                  evidence=evidence, trace=routed["trace"], callbacks=routed["callbacks"])
    stage2["trace"]["source_C0"] = "not_used_external_detector"
    if mix == "excluded":
        stage2.update(latent="unresolved", model="unresolved", mixed=False)
    if (failures or any(evidence[n]["status"] in {"unresolved", "unavailable", "not_run"}
                        for n in ("CL", "CM", "CMIX"))):
        result["completion"]["status"] = "incomplete"
    return _strict(result)


def render_report(report):
    """Details are a pure rendering of the saved result, without new tests."""
    if report.get("context", {}).get("route") == "conditional_history_saved_report":
        from .conditional_history_saved_report import render_report as render_saved
        return (render_saved(report) + "Completion: " + report["completion"]["status"] +
                "\nDetails:\n" + json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    s = report["stage2"]
    labels = report.get("source_labels", {})
    lines = ["Stage 1: " + report["stage1"]["status"],
             "Stage 2: " + s["outcome"],
             labels.get("latent", "Latent") + ": " + s["latent"] + "; " + labels.get("model", "model") + ": " + s["model"] +
             "; mixed: " + str(s["mixed"]),
             "Source probabilities: unavailable (necessity is relative to declared families).",
             "Completion: " + report["completion"]["status"], "Details:",
             json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)]
    capacity = report["stage1"]
    if "calibration_capacity_explanation" in capacity:
        lines.insert(1, "Calibration failure-count p floor: " +
                     str(capacity["calibration_failure_p_floor"]) +
                     "; failure floor alone permits rejection: " +
                     str(capacity["calibration_allows_rejection"]) + ". " +
                     capacity["calibration_capacity_explanation"])
    if report.get("semantic_disclosure"):
        lines.insert(3, report["semantic_disclosure"])
    return "\n".join(lines) + "\n"
