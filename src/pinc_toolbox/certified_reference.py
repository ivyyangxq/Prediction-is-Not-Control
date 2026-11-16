"""Isolated exact reference/eta selector for the finite public panel.

This module is an unintegrated engineering candidate.  Floating
``fit_prefix`` is used only to propose a candidate reference and to report
conservative not-ready states.  Every readiness, support, and precision
decision is recomputed from the original represented public rows with the
exact rational construction.  The module has no kappa (``κ``) interface and
does not certify ``e_eta`` or an action-Q quantity.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import isfinite
from typing import Any

from .certified_construction import (
    _q,
    _q_vector,
    reference_projection,
    reference_score_bounds,
    scaled_linf_support,
)
from .eta_monitor import _integer, _public_row
from .panel_reference import fit_prefix
from .domain_coverage import cell_certificate, whole_domain_certificate
from .joint_score_reference import build_joint_reference, project_joint_reference


class _ProposalUnavailable(ValueError):
    """A finite exact input cannot be represented by the float proposal."""


def _display_or_none(value: Any) -> float | None:
    try:
        result = float(_q(value))
    except (OverflowError, TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _proposal_float(value: Any) -> float:
    try:
        exact = _q(value)
    except (TypeError, ValueError):
        raise
    try:
        result = float(exact)
    except OverflowError as exc:
        raise _ProposalUnavailable("numeric_unavailable") from exc
    if not isfinite(result):
        raise _ProposalUnavailable("numeric_unavailable")
    return result


def _prefixes(spec: Mapping[str, Any]) -> list[tuple[int, int]]:
    values = [(_integer(item[0], "N", 1), _integer(item[1], "T", 1))
              for item in spec["prefixes"]]
    if not values or len(set(values)) != len(values):
        raise ValueError("prefixes must be nonempty and distinct")
    return values


def _eta_values(spec: Mapping[str, Any]) -> list[tuple[Any, float | None]]:
    values: dict[Any, float | None] = {}
    for raw in spec["eta_grid"]:
        exact = _q(raw)
        display = _display_or_none(exact)
        if display is not None and not isfinite(display) or exact < 0:
            raise ValueError("eta_grid must be finite and nonnegative")
        values.setdefault(exact, display)
    if not values:
        raise ValueError("eta_grid must be nonempty")
    return sorted(values.items(), key=lambda item: item[0])


def _known_rows(public_rows: Sequence[Mapping[str, Any]], clock: int) -> list[dict[str, Any]]:
    """Filter by release clock before stripping unknown fields."""
    result: list[dict[str, Any]] = []
    for raw in public_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("public row must be a mapping")
        if "outcome_release" not in raw:
            raise ValueError("public row must contain outcome_release")
        release = _integer(raw["outcome_release"], "outcome_release")
        if release > clock:
            continue
        result.append(_public_row(raw))
    return result


def _original_selected(candidate_rows: Sequence[Mapping[str, Any]],
                       known_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in known_rows:
        if "household" in row and "period" in row:
            key = (_integer(row["period"], "period"), _integer(row["household"], "household"))
            by_key[key] = row
    selected: list[dict[str, Any]] = []
    for row in candidate_rows:
        key = (_integer(row["period"], "period"), _integer(row["household"], "household"))
        if key not in by_key:
            raise ValueError("candidate selected row is absent from original public rows")
        selected.append(deepcopy(dict(by_key[key])))
    return selected


def _prefix_rows(known_rows: Sequence[Mapping[str, Any]], n: int, t: int) -> list[dict[str, Any]]:
    """Return original rows in one prefix, preserving their supplied order."""
    selected = []
    for row in known_rows:
        household = _integer(row["household"], "household")
        period = _integer(row["period"], "period")
        if 0 <= household < n and 0 <= period < t:
            selected.append(deepcopy(dict(row)))
    return selected


def _proposal_rows(known_rows: Sequence[Mapping[str, Any]], n: int, t: int) -> list[dict[str, Any]]:
    """Decode only one completed prefix for the floating proposal fit."""
    rows = []
    for raw in known_rows:
        household = _integer(raw["household"], "household")
        period = _integer(raw["period"], "period")
        if not (0 <= household < n and 0 <= period < t):
            continue
        row = deepcopy(dict(raw))
        row["features"] = [_proposal_float(value) for value in row["features"]]
        row["outcome"] = _proposal_float(row["outcome"])
        row["noise_loading"] = _proposal_float(row["noise_loading"])
        rows.append(row)
    return rows


def _proposal_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Decode only declarations consumed by ``fit_prefix``."""
    return {
        "feature_bounds": [_proposal_float(value) for value in spec["feature_bounds"]],
        "sigma_common": _proposal_float(spec["sigma_common"]),
        "sigma_idiosyncratic": _proposal_float(spec["sigma_idiosyncratic"]),
        "loading_bound": _proposal_float(spec["loading_bound"]),
        "alpha_reference": _proposal_float(spec["alpha_reference"]),
    }


def _reference_declaration(spec: Mapping[str, Any], n: int, t: int,
                           total_prefixes: int) -> dict[str, Any]:
    """Capture only declarations that affect the exact reference calculation."""
    return deepcopy({
        "feature_bounds": spec["feature_bounds"],
        "sigma_common": spec["sigma_common"],
        "sigma_idiosyncratic": spec["sigma_idiosyncratic"],
        "loading_bound": spec["loading_bound"],
        "alpha_reference": spec["alpha_reference"],
        "directions": [task["direction"] for task in spec["tasks"]],
        "region": spec.get("region"),
        "monitoring_domain": spec.get("monitoring_domain"),
        "scale_floor": spec.get("scale_floor"),
        "N": n, "T": t, "total_prefixes": total_prefixes,
    })


def _cache_reference(prefix_cache: Any, key: tuple[int, int], selected_rows: list[dict[str, Any]],
                     declaration: Mapping[str, Any], reference: dict[str, Any] | None) -> dict[str, Any]:
    """Read/write a completed-prefix cache while refusing changed inputs."""
    if prefix_cache is None:
        return reference
    try:
        present = key in prefix_cache
    except Exception as exc:
        raise ValueError("prefix_cache must support key lookup") from exc
    if present:
        entry = prefix_cache[key]
        if not isinstance(entry, Mapping):
            raise ValueError("prefix_cache entry is invalid")
        try:
            same_rows = entry.get("selected_rows") == selected_rows
            same_spec = entry.get("declaration") == declaration
        except Exception as exc:
            raise ValueError("prefix_cache identity comparison failed") from exc
        if not same_rows or not same_spec:
            raise ValueError("prefix_cache contains changed public rows or specification")
        return deepcopy(entry["reference"])
    if reference is None:
        raise ValueError("missing reference for new prefix_cache entry")
    entry = {"selected_rows": deepcopy(selected_rows), "declaration": deepcopy(dict(declaration)),
             "reference": deepcopy(reference)}
    try:
        prefix_cache[key] = entry
    except Exception as exc:
        raise ValueError("prefix_cache must be an append-only mutable mapping") from exc
    return reference


def _exact_reference(candidate: Mapping[str, Any], selected_rows: list[dict[str, Any]],
                     spec: Mapping[str, Any], n: int, t: int,
                     total_prefixes: int) -> dict[str, Any]:
    caps = spec["feature_bounds"]
    theta = candidate["theta_hat"]
    score = reference_score_bounds(
        caps, sigma_common=spec["sigma_common"],
        sigma_idiosyncratic=spec["sigma_idiosyncratic"],
        loading_bound=spec["loading_bound"], households=n, periods=t,
        total_prefixes=total_prefixes, alpha=spec["alpha_reference"],
    )
    features = [row["features"] for row in selected_rows]
    outcomes = [row["outcome"] for row in selected_rows]
    directions = [task["direction"] for task in spec["tasks"]]
    projection = reference_projection(features, outcomes, theta,
                                      score["score_bounds_exact"], directions)
    reference = deepcopy(dict(candidate))
    reference.update({
        "selected_rows": deepcopy(selected_rows),
        "theta_hat": list(candidate["theta_hat"]),
        "score_bounds": score["score_bounds"],
        "score_bounds_exact": score["score_bounds_exact"],
        "log_argument_exact": score["log_argument_exact"],
        "log_upper_exact": score["log_upper_exact"],
        "V_exact": projection["V_exact"],
        "c_exact": projection["c_exact"],
        "candidate_theta_exact": projection["candidate_theta_exact"],
        "score_residual_at_candidate_exact": projection["score_residual_at_candidate_exact"],
        "directions_exact": projection["directions_exact"],
        "solved_Vz_equals_B_transpose_exact": projection["solved_Vz_equals_B_transpose_exact"],
        "projection_error_exact": projection["projection_error_exact"],
        "projection_error": projection["projection_error"],
        "numerical_certification": (
            "exact represented public rows/specification only; application assumptions, "
            "e_eta, and action-Q are not certified"
        ),
        "e_eta": None,
        "D_L_minus_M": None,
        "action_Q_certification": "unavailable",
    })
    return reference


def _check_exact_domain(selected_rows: Sequence[Mapping[str, Any]], caps: Sequence[Any],
                        loading_bound: Any) -> str | None:
    """Return an exact domain failure reason, if any."""
    bound = _q(loading_bound)
    for row in selected_rows:
        features = _q_vector(row["features"], "features")
        if len(features) != len(caps) or any(abs(x) > cap for x, cap in zip(features, caps)):
            return "feature_domain_exit"
        if abs(_q(row["noise_loading"])) > bound:
            return "noise_loading_domain_exit"
    return None


def _not_ready(diagnostics: list[dict[str, Any]], region_method: str = "legacy") -> dict[str, Any]:
    domain_evidence = [item.get("domain_coverage") for item in diagnostics
                       if item.get("domain_coverage") is not None]
    return {
        "status": "not_ready", "reasons": ["no_completed_qualified_prefix"],
        "candidates": diagnostics, "eta": None, "monitoring_start": None,
        "eta_fit_cutoff": None,
        "reference_region_method": region_method,
        "domain_coverage": domain_evidence[-1] if domain_evidence else None,
        "task_scope": "linear_reference_projection_and_observable_geometry_only",
        "e_eta": None, "D_L_minus_M": None,
        "action_Q_certification": "unavailable",
    }


def select_eta_certified(public_rows, spec, *, as_of, prefix_cache=None):
    """Select the first completed prefix and smallest exactly supported eta.

    The result follows :func:`pinc_toolbox.eta_monitor.select_eta`'s public
    schema.  The float fit is a proposal only; exact source rows, score
    bounds, projection errors, anchor geometry, and task tolerances determine
    readiness.
    """
    clock = _integer(as_of, "as_of")
    known_rows = _known_rows(public_rows, clock)
    prefixes = _prefixes(spec)
    eta_values = _eta_values(spec)
    tasks = spec["tasks"]
    if not tasks:
        raise ValueError("a predeclared nonempty task list is required")
    required = _integer(spec["required_task_count"], "required_task_count", 1)
    if required > len(tasks):
        raise ValueError("required_task_count exceeds the fixed task denominator")
    # Force exact dimension/bound validation on the original declarations.
    caps = _q_vector(spec["feature_bounds"], "feature_bounds")
    floor = _q_vector(spec["scale_floor"], "scale_floor")
    if not caps or any(x < 0 for x in caps):
        raise ValueError("feature_bounds must be nonnegative and nonempty")
    if not floor or any(x <= 0 for x in floor):
        raise ValueError("scale_floor must be strictly positive")
    _q(spec["alpha_reference"])
    for task in tasks:
        if _q(task["max_error"]) < 0:
            raise ValueError("task max_error must be nonnegative")
    region = spec.get("region")
    joint = bool(region)
    if joint:
        if not isinstance(region, Mapping) or region.get("kind") != "joint_score":
            raise ValueError("unknown region.kind; expected 'joint_score'")
        monitoring_domain = spec.get("monitoring_domain")
        if not isinstance(monitoring_domain, Mapping):
            raise ValueError("joint_score requires a monitoring_domain mapping")
        lower = monitoring_domain.get("lower")
        upper = monitoring_domain.get("upper")
        if lower is None or upper is None or len(lower) != len(upper):
            raise ValueError("monitoring_domain.lower and upper are required for joint_score")
        domain_bounds = [[lo, hi] for lo, hi in zip(lower, upper)]
        domain_subdivisions = monitoring_domain.get("subdivisions", 4)
        if domain_subdivisions != 4:
            raise ValueError("joint_score requires the declared four-way cell refinement")
        lambda_scales = region.get("lambda_scales")
        if lambda_scales != [1, 128] and lambda_scales != ["1", "128"]:
            raise ValueError("joint_score.lambda_scales must be [1, 128]")
        branch_count = len({item[0] for item in prefixes})
        expected_delta = _q(spec["alpha_reference"]) / (branch_count * 2)
        if not 0 < _q(region.get("delta_per_scale")) <= expected_delta:
            raise ValueError("joint_score.delta_per_scale exceeds the declared N-branch and Lambda-scale allocation")
        region_method = "joint_score"
    else:
        domain_bounds = domain_subdivisions = None
        region_method = "legacy"
    diagnostics: list[dict[str, Any]] = []
    for n, t in sorted(prefixes, key=lambda item: item[1]):
        if t > clock:
            continue
        key = (n, t)
        selected_rows = _prefix_rows(known_rows, n, t)
        declaration = _reference_declaration(spec, n, t, len(prefixes))
        try:
            cached = prefix_cache is not None and key in prefix_cache
            if cached:
                candidate = None
                reference = _cache_reference(prefix_cache, key, selected_rows, declaration, None)
            else:
                proposal_rows = _proposal_rows(known_rows, n, t)
                proposal = _proposal_spec(spec)
                candidate = fit_prefix(
                    proposal_rows, households=n, periods=t, total_prefixes=len(prefixes),
                    feature_bounds=proposal["feature_bounds"],
                    sigma_common=proposal["sigma_common"],
                    sigma_idiosyncratic=proposal["sigma_idiosyncratic"],
                    loading_bound=proposal["loading_bound"],
                    alpha=proposal["alpha_reference"], as_of=clock,
                )
                if candidate["status"] != "ready":
                    diagnostics.append(dict(N=n, T=t, status="not_ready",
                                            reasons=candidate["reasons"], eta=None,
                                            reference_region_method=region_method))
                    continue
                selected_rows = _original_selected(candidate["selected_rows"], known_rows)
                reason = _check_exact_domain(selected_rows, caps, spec["loading_bound"])
                if reason is not None:
                    diagnostics.append(dict(N=n, T=t, status="not_ready",
                                            reasons=[reason], eta=None,
                                            reference_region_method=region_method))
                    continue
                if joint:
                    budget = region.get("family_budget")
                    if budget is None:
                        budget = {"alpha_reference": spec["alpha_reference"],
                                  "branch_count": len({item[0] for item in prefixes}),
                                  "scale_count": 2, "delta_per_scale": region["delta_per_scale"]}
                    elif not isinstance(budget, Mapping):
                        raise ValueError("region.family_budget must be a mapping")
                    if budget.get("branch_count") != branch_count or budget.get("scale_count", 2) != 2 or _q(budget.get("delta_per_scale", region["delta_per_scale"])) != _q(region["delta_per_scale"]):
                        raise ValueError("family_budget must match region allocation")
                    budget = dict(budget, delta_per_scale=region["delta_per_scale"])
                    joint_payload = build_joint_reference(
                        selected_rows, theta_float=candidate["theta_hat"],
                        feature_caps=caps, sigma_common=spec["sigma_common"],
                        sigma_idiosyncratic=spec["sigma_idiosyncratic"],
                        loading_bound=spec["loading_bound"], households=n, periods=t,
                        total_prefixes=len(prefixes), alpha_reference=spec["alpha_reference"],
                        family_budget=budget,
                    )
                    projection = project_joint_reference(
                        joint_payload, [task["direction"] for task in tasks])
                    exact = deepcopy(dict(candidate))
                    exact.update({
                        "selected_rows": deepcopy(selected_rows),
                        "theta_hat": list(candidate["theta_hat"]),
                        "V_exact": joint_payload["V_exact"], "c_exact": joint_payload["c_exact"],
                        "candidate_theta_exact": joint_payload["theta_exact_from_float"],
                        "projection_error_exact": [row["minimum"]["radius_exact"]
                                                    for row in projection["rows"]],
                        "projection_error": [_display_or_none(row["minimum"]["radius_exact"])
                                             for row in projection["rows"]],
                        "joint_score_region": joint_payload,
                        "joint_score_projection": projection,
                        "region_method": region_method,
                        "reference_region_method": region_method,
                        "numerical_certification": (
                            "exact represented public rows/specification and joint-score region only; "
                            "application assumptions, e_eta, and action-Q are not certified"),
                        "e_eta": None, "D_L_minus_M": None,
                        "action_Q_certification": "unavailable",
                    })
                else:
                    exact = _exact_reference(candidate, selected_rows, spec, n, t, len(prefixes))
                    exact["region_method"] = region_method
                    exact["reference_region_method"] = region_method
                reference = _cache_reference(prefix_cache, key, selected_rows, declaration, exact)
        except _ProposalUnavailable:
            diagnostics.append(dict(N=n, T=t, status="not_ready",
                                    reasons=["numeric_unavailable"], eta=None,
                                    reference_region_method=region_method))
            continue
        except ValueError as exc:
            message = str(exc).lower()
            if ("reference fit returned nonfinite" in message
                    or "score_bounds are nonfinite" in message):
                diagnostics.append(dict(N=n, T=t, status="not_ready",
                                        reasons=["numeric_unavailable"], eta=None,
                                        reference_region_method=region_method))
                continue
            if "singular" in str(exc).lower():
                diagnostics.append(dict(N=n, T=t, status="not_ready",
                                        reasons=["exact_singular"], eta=None,
                                        reference_region_method=region_method))
                continue
            raise
        except (MemoryError, OverflowError):
            diagnostics.append(dict(N=n, T=t, status="not_ready",
                                    reasons=["resource_exhaustion"], eta=None,
                                    reference_region_method=region_method))
            continue
        anchors_raw = [row["support_point"] for row in selected_rows]
        support_floor = region.get("scale_floor", spec["scale_floor"]) if joint else spec["scale_floor"]
        support_floor = _q_vector(support_floor, "scale_floor")
        scale_probe = scaled_linf_support(tasks[0]["support_point"], anchors_raw, support_floor, eta_values[0][0])
        scale_exact = scale_probe["scale_exact"]
        anchors_display = [[_display_or_none(x) for x in row] for row in anchors_raw]
        scale_display = [_display_or_none(x) for x in scale_exact]
        error_exact = [_q(value) for value in reference["projection_error_exact"]]
        selected_eta = None
        selected_eta_exact = None
        selected_domain_details = None
        domain_cell = None
        domain_whole = None
        for eta_exact, eta_display in eta_values:
            details = []
            qualified_count = 0
            for index, task in enumerate(tasks):
                support = scaled_linf_support(task["support_point"], anchors_raw, support_floor, eta_exact)
                tolerance = _q(task["max_error"])
                precise = error_exact[index] <= tolerance
                qualified = bool(support["supported"] and precise)
                qualified_count += int(qualified)
                details.append(dict(
                    task=index,
                    distance=_display_or_none(support["minimum_distance_exact"]),
                    distance_exact=support["minimum_distance_exact"],
                    supported=bool(support["supported"]),
                    reference_error=_display_or_none(error_exact[index]),
                    reference_error_exact=str(error_exact[index]),
                    max_error=_display_or_none(tolerance),
                    max_error_exact=str(tolerance),
                    precise=bool(precise), qualified=qualified,
                    reasons=([] if support["supported"] else ["outside_eta_support"]) +
                            ([] if precise else ["reference_precision_insufficient"]),
                ))
            domain_details = None
            domain_enough = True
            if joint:
                if qualified_count < required:
                    domain_details = None
                    domain_enough = False
                else:
                    if domain_whole is None:
                        domain_whole = whole_domain_certificate(domain_bounds, scale_exact, anchors_raw)
                    whole_bound = _q(domain_whole["upper_bound_exact"])
                    if whole_bound <= eta_exact:
                        domain_details = dict(method=domain_whole["method"],
                                              upper_bound_exact=domain_whole["upper_bound_exact"],
                                              cells_checked=domain_whole["cells_checked"],
                                              anchor_index=domain_whole["anchor_index"])
                    else:
                        if domain_cell is None:
                            domain_cell = cell_certificate(domain_bounds, scale_exact, anchors_raw,
                                                           subdivisions=domain_subdivisions)
                        domain_enough = _q(domain_cell["upper_bound_exact"]) <= eta_exact
                        domain_details = dict(domain_cell)
                    domain_details["enough"] = domain_enough if domain_details["method"] != domain_whole["method"] else True
            ready = qualified_count >= required and domain_enough
            diagnostics.append(dict(
                N=n, T=t, eta=eta_display, eta_exact=str(eta_exact), status="ready" if ready else "not_ready",
                task_denominator=len(tasks), required_task_count=required,
                qualified_tasks=qualified_count, tasks=details,
                scale_exact=scale_exact,
                reference_region_method=region_method,
                region_method=region_method,
                domain_coverage=domain_details,
                reasons=([] if ready else (["required_tasks_not_qualified"] if qualified_count < required else []) +
                         ([] if domain_enough else ["domain_coverage_insufficient"])),
            ))
            if ready and selected_eta_exact is None:
                selected_eta = eta_display
                selected_eta_exact = eta_exact
                selected_domain_details = deepcopy(domain_details)
        if selected_eta_exact is not None:
            reference["scale_exact"] = scale_exact
            anchors_exact = [[str(_q(x)) for x in row] for row in anchors_raw]
            reference["anchors_exact"] = anchors_exact
            return dict(
                status="ready", N=n, T=t, eta=selected_eta,
                eta_exact=str(selected_eta_exact), selected_eta_exact=str(selected_eta_exact),
                eta_fit_cutoff=t,
                monitoring_start=t + 1, reference=reference,
                scale=scale_display, scale_exact=scale_exact,
                anchors=anchors_display, anchors_exact=anchors_exact,
                reference_region_method=region_method,
                region_method=region_method,
                domain_coverage=selected_domain_details,
                joint_score_region=reference.get("joint_score_region") if joint else None,
                candidates=diagnostics, task_denominator=len(tasks),
                distance_rule="nearest_joint_record_scaled_Linf",
                scale_rule="reference_range_with_predeclared_positive_floor",
                fixed_after_start=["eta", "scale", "anchors", "reference", "tasks"],
                task_scope="linear_reference_projection_and_observable_geometry_only",
                e_eta=None, D_L_minus_M=None,
                action_Q_certification="unavailable",
            )
    return _not_ready(diagnostics, region_method=region_method)


__all__ = ["select_eta_certified"]
