"""Fixed-action reset-bin measurements and two-stage GQ source reporting.

This restricted route does not validate endogenous online RA-GQ histories.
The caller supplies observed blocks, a predeclared finite detector calendar,
and an independent old source reference held out from the entire alarm rule.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import norm
from .two_stage_report import run_two_stage_report

_REQUIRED_ASSUMPTIONS = (
    "fixed_actions_and_reset_baseline", "independent_gaussian_observation_noise",
    "known_sigma", "source_reference_independent_of_alarm",
    "fresh_source_noise_independent_of_alarm_and_source_reference",
    "calendar_and_source_deadline_predeclared", "candidate_families_contain_truth",
)


def run_reset_bin_two_stage(*, actions, detection_reference, detection_windows,
                           source_reference, source_window, sigma, source_deadline,
                           alpha_detection=.01, alpha_source=.05,
                           latent_bound=.6, model_bound=2., assumptions=None):
    """Run the actual detector/measurement/backend route on supplied observations.

    Blocks have ``id`` and ``values`` (one list of observations per fixed action).
    Current blocks also have numeric ``time``. All detector looks must be supplied
    in their predeclared order; missing/empty bins are retained as noncompletion.
    Source samples are used only after the first alarm. Assumption declarations
    support theorem-scoped bounds, never certify data independence automatically.
    """
    a = np.asarray(actions, dtype=float)
    if a.ndim != 1 or len(a) == 0 or not np.all(np.isfinite(a)):
        raise ValueError("actions must be a nonempty finite vector")
    m = len(a)
    for name, value in (("sigma", sigma), ("alpha_detection", alpha_detection),
                        ("alpha_source", alpha_source), ("latent_bound", latent_bound),
                        ("model_bound", model_bound), ("source_deadline", source_deadline)):
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(name + " must be finite numeric")
    if sigma <= 0 or latent_bound < 0 or model_bound < 0:
        raise ValueError("sigma must be positive; family bounds nonnegative")
    if not 0 < alpha_detection < 1 or not 0 < alpha_source < 1:
        raise ValueError("error budgets must be between zero and one")
    if not isinstance(detection_windows, list) or not detection_windows:
        raise ValueError("a finite nonempty detection calendar is required")
    times = [b["time"] for b in detection_windows]
    if any(isinstance(t, bool) or not math.isfinite(t) for t in times):
        raise ValueError("calendar times must be finite numeric")
    if any(t >= u for t, u in zip(times, times[1:])):
        raise ValueError("calendar must be strictly increasing")
    if source_deadline <= times[0]:
        raise ValueError("source deadline must follow first detection opportunity")
    declared = {} if assumptions is None else dict(assumptions)
    context = {
        "route": "fixed_action_reset_bins", "actions": a.tolist(),
        "detection_calendar": times, "source_deadline": source_deadline,
        "reference": {"detection": detection_reference.get("id"),
                      "source": source_reference.get("id")},
        "source_families": {"CL": "independent bounded latent coordinate per action bin",
                            "CM": "one constant action-response coefficient",
                            "CMIX": "CL plus CM", "latent_bound": latent_bound,
                            "model_bound": model_bound},
        "scope": "matched fixed reset state/baseline; no general online guarantee",
        "eta": "not_computed_by_reset_component_route",
        "source_sampling": "all source observations strictly post-alarm; time is endpoint only",
        "count_design": "counts fixed before observing block outcomes",
    }
    failures = []

    def means(block):
        values = block.get("values")
        if not isinstance(values, list) or len(values) != m:
            raise ValueError("every block requires one observation list per action")
        arrays = [np.asarray(v, dtype=float) for v in values]
        if any(v.ndim != 1 or not np.all(np.isfinite(v)) for v in arrays):
            raise ValueError("observations must be finite one-dimensional arrays")
        if any(len(v) == 0 for v in arrays):
            return None
        return np.array([v.mean() for v in arrays]), np.array([len(v) for v in arrays])

    def contrast(old, new, alpha):
        left, right = means(old), means(new)
        if left is None or right is None:
            return None
        mu0, n0 = left
        mu1, n1 = right
        radii = float(norm.ppf(1 - alpha / (2 * m))) * sigma * np.sqrt(1 / n0 + 1 / n1)
        return {"moments": (mu1 - mu0).tolist(), "radii": radii.tolist(),
                "reference_counts": n0.tolist(), "current_counts": n1.tolist(),
                "normal_quantile": float(norm.ppf(1 - alpha / (2 * m)))}

    # IDs detect obvious accidental reuse, but are not evidence of independence.
    ids = [detection_reference.get("id")] + [b.get("id") for b in detection_windows]
    source_ids = [source_reference.get("id"), source_window.get("id")]
    if any(not isinstance(i, str) or not i for i in ids + source_ids):
        raise ValueError("all observed blocks require nonempty provenance IDs")
    if len(set(ids + source_ids)) != len(ids + source_ids):
        raise ValueError("blocks require distinct IDs; source reference cannot reuse detection data")
    records = []
    first_alarm = None
    for block in detection_windows:
        measurement = contrast(detection_reference, block, alpha_detection / len(times))
        if measurement is None:
            records.append({"time": block["time"], "status": "unavailable", "reason": "empty_action_bin"})
            failures.append({"stage": "detection", "time": block["time"], "reason": "empty_action_bin"})
            continue
        alarm = any(abs(x) > r for x, r in zip(measurement["moments"], measurement["radii"]))
        records.append(dict(measurement, time=block["time"], status="alarm" if alarm else "no_alarm"))
        if alarm:
            first_alarm = block["time"]
            break
    stage1 = {"status": "alarm" if first_alarm is not None else "unavailable" if failures else "no_alarm",
              "event": "first declared look with any abs(bin_mean_difference) > radius",
              "first_alarm": first_alarm, "records": records,
              "not_visited_times": times[len(records):],
              "threshold_rule": "two-sided Gaussian Bonferroni over bins and all declared looks"}
    source = None
    readiness = {"status": "ready"}
    measurement = None
    if first_alarm is not None:
        t = source_window.get("time")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
            raise ValueError("source time must be finite numeric")
        if not first_alarm < t <= source_deadline:
            readiness = {"status": "unavailable", "reason": "source_not_after_alarm_or_deadline_missed"}
        else:
            measurement = contrast(source_reference, source_window, alpha_source)
            if measurement is None:
                readiness = {"status": "unavailable", "reason": "empty_action_bin"}
            else:
                source = {
                    "moments": measurement["moments"], "radii": measurement["radii"],
                    "latent_loading": np.diag(.1 + .2 * a).tolist(),
                    "latent_bounds": [[-latent_bound] * m, [latent_bound] * m],
                    "mechanism_loadings": {"constant_response": a.reshape(-1, 1).tolist()},
                    "mechanism_bounds": [[-model_bound], [model_bound]],
                }
    report = run_two_stage_report(stage1, context=context, source=source, readiness=readiness)
    report["measurements"] = {"source": measurement, "source_spec": source,
                              "source_window": {"id": source_window["id"], "time": source_window.get("time")},
                              "source_reference": source_reference["id"]}
    report["completion"]["failures"] = failures + report["completion"]["failures"]
    report["reliability"]["assumptions"] = [
        {"name": key, "status": "declared" if declared.get(key) is True else "unverified"}
        for key in _REQUIRED_ASSUMPTIONS]
    report["reliability"]["theory_reference"] = "track2_errors: fixed matched-bin reset theorem"
    report["reliability"]["numerical_radius"] = "SciPy binary64 normal quantile and float arithmetic, not outward-certified; theorem is ideal-real arithmetic; exact backend certifies supplied region only"
    for name, budget in (("detection", alpha_detection), ("latent", alpha_source),
                         ("model", alpha_source), ("mixed", alpha_source),
                         ("joint", min(1., alpha_detection + alpha_source))):
        item = report["errors"][name]["type_i"]
        item["target_budget"] = budget
        if all(declared.get(key) is True for key in _REQUIRED_ASSUMPTIONS):
            item.update(status="theorem_under_declared_assumptions", proved_bound=budget)
    report["errors"]["source_joint"] = {
        "type_i": {"conditioning": "conditional_on_alarm", "target_budget": alpha_source,
                   "proved_bound": alpha_source if all(declared.get(k) is True for k in _REQUIRED_ASSUMPTIONS) else None,
                   "status": "theorem_under_declared_assumptions" if all(declared.get(k) is True for k in _REQUIRED_ASSUMPTIONS) else "not_established",
                   "empirical": None}}
    return report
