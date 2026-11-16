"""Bounded predictable post-alarm source route using the actual SimpleGQ fit.

This runner constructs the Section 6 confidence rectangle and sends it to the
existing exact CL/CM/CMIX source controller. Its statistical assumptions are
caller supplied and cannot be verified from arrays; consequently the output
keeps bounds null by default and records a conditional budget only when every
condition is declared.
It is a restricted scalar-physical, linear-dictionary GQ route, not generic
GQ/RA-GQ or an economic-Q certificate.
"""
from __future__ import annotations

from math import isfinite, log, sqrt
from fractions import Fraction
from typing import Any, Mapping

import numpy as np
from scipy.stats import chi2

from .g import SimpleGQ, linear_physical_memory_features, scalar_action_features
from .two_stage_report import run_two_stage_report, _errors


_REQUIRED_ASSUMPTIONS = (
    "reference_independent_of_entire_alarm_filtration",
    "conditional_subgaussian_future_innovations",
    "weights_predictable",
    "fixed_calendar",
    "known_common_sigma",
    "reference_and_future_share_baseline",
    "full_rank_fixed_old_design",
    "independent_gaussian_reference_errors",
    "candidate_witnesses_are_in_declared_families",
    "correctly_specified_full_linear_dictionary",
)


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.ndim != ndim or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return out


def _clock(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer calendar time")
    return int(value)


def _design(model: SimpleGQ, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Return the raw joint [B,D] design that parameterizes SimpleGQ exactly."""
    memory = model.memory_index
    j = memory if memory >= 0 else states.shape[1] + memory
    b = np.asarray(model.state_features(states), dtype=float)
    d_action = np.asarray(model.action_features(states, actions), dtype=float)
    d_anchor = np.asarray(model.action_features(states, states[:, j]), dtype=float)
    d = d_action - d_anchor
    if (b.ndim != 2 or d.ndim != 2 or b.shape[0] != len(states)
            or d.shape[0] != len(states) or not np.isfinite(b).all()
            or not np.isfinite(d).all()):
        raise ValueError("SimpleGQ features returned invalid joint design")
    return np.column_stack((b, d))


def _fwl_covariance(b: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Unit-variance coefficient covariance from the FWL block identities."""
    btb_inv = np.linalg.solve(b.T @ b, np.eye(b.shape[1]))
    projected = b.T @ d
    residual_d = d - b @ (btb_inv @ projected)
    psi_cov = np.linalg.solve(residual_d.T @ residual_d,
                              np.eye(d.shape[1]))
    cross = -btb_inv @ projected @ psi_cov
    c_cov = btb_inv + btb_inv @ projected @ psi_cov @ projected.T @ btb_inv
    return np.block([[c_cov, cross], [cross.T, psi_cov]])


def _source_spec(moments: np.ndarray, radii: np.ndarray, weights: np.ndarray,
                 actions: np.ndarray, latent_profile: np.ndarray,
                 n: int, latent_bound: float, model_bound: float) -> dict[str, Any]:
    # One target coordinate per row of W. Latent b_i is allowed to vary at
    # every post-alarm row; the bounded profile is h_i = .1 + .2 a_i.
    latent_loading = (weights * latent_profile[None, :]) / n
    action_loading = (weights @ actions)[:, None] / n
    # Disjoint bins have an exact box image. Preserve the backend's binary-float
    # rational interpretation when adding generators; overlapping columns keep
    # their full shared latent variable (a zonotope, generally not a box).
    columns = n
    latent_values = latent_loading.tolist()
    if np.all(np.count_nonzero(latent_loading, axis=0) <= 1):
        columns = latent_loading.shape[0]
        widths = [sum((abs(Fraction.from_float(float(v))) for v in row), Fraction(0))
                  for row in latent_loading]
        latent_values = [[str(widths[j]) if j == k else "0" for k in range(columns)]
                         for j in range(columns)]
    return {
        "moments": moments.tolist(),
        "radii": radii.tolist(),
        "latent_loading": latent_values,
        "latent_bounds": [[-latent_bound] * columns, [latent_bound] * columns],
        "mechanism_loadings": {"constant_action_response": action_loading.tolist()},
        "mechanism_bounds": {"constant_action_response": [[-model_bound], [model_bound]]},
    }


def prepare_predictable_gq_region(
    reference: dict[str, Any], future: dict[str, Any], *, sigma: float,
    alpha_reference: float = .025, alpha_noise: float = .025,
    calendar_size: int = 1, latent_bound: float = .6, model_bound: float = 2.,
) -> dict[str, Any]:
    """Prepare a Section 6 region from a held-out reference and fixed window.

    This helper does not inspect or invent a detector/alarm. It is suitable for
    composing a detector and a source route when provenance and timing are
    supplied by the caller. Its readiness reports computational checks only;
    the statistical assumptions remain unverified.
    """
    x0 = _array(reference.get("states"), "reference.states", 2)
    a0 = _array(reference.get("actions"), "reference.actions", 1)
    y0 = _array(reference.get("next_physical"), "reference.next_physical", 2)
    x = _array(future.get("states"), "future.states", 2)
    a = _array(future.get("actions"), "future.actions", 1)
    y = _array(future.get("next_physical"), "future.next_physical", 2)
    w = _array(future.get("weights"), "future.weights", 2)
    if (x0.shape[0] != len(a0) or x0.shape[0] != y0.shape[0]
            or x.shape[0] != len(a) or x.shape[0] != y.shape[0]
            or x0.shape[1] != x.shape[1] or y0.shape[1] != 1 or y.shape[1] != 1):
        raise ValueError("row dimensions must agree; this route supports scalar physical output")
    n = len(x)
    if n < 1 or w.shape[1] != n or w.shape[0] < 1 or np.any(w < 0) or np.any(w > 1):
        raise ValueError("weights must have shape (m,n), n positive, and lie in [0,1]")
    sigma = float(sigma)
    alpha_reference, alpha_noise = float(alpha_reference), float(alpha_noise)
    if (not isfinite(sigma) or sigma <= 0 or not 0 < alpha_reference < 1
            or not 0 < alpha_noise < 1 or alpha_reference + alpha_noise >= 1
            or isinstance(calendar_size, bool) or not isinstance(calendar_size, int)
            or calendar_size < 1):
        raise ValueError("invalid variance, error budget, or calendar")
    memory_index = reference.get("memory_index", -1)
    physical_indices = tuple(reference.get("physical_indices", (0,)))
    if len(physical_indices) != 1:
        raise ValueError("the bounded runner currently supports one physical GQ output")
    model = SimpleGQ(
        lambda z: linear_physical_memory_features(z, physical_indices, memory_index),
        scalar_action_features, memory_index=memory_index,
        physical_indices=physical_indices,
    )
    model.fit(x0, a0, y0)
    x0_joint = _design(model, x0, a0)
    x_joint = _design(model, x, a)
    p = x0_joint.shape[1]
    if x_joint.shape[1] != p or len(x0) < p:
        raise ValueError("reference joint design must have at least p rows")
    gram = x0_joint.T @ x0_joint
    if np.linalg.matrix_rank(x0_joint) != p:
        raise ValueError("reference joint GQ design is not full column rank")
    a0_map = np.linalg.solve(gram, x0_joint.T)
    beta_gq = np.vstack((model._C, model._Psi))[:, 0]
    beta_joint, _, rank, _ = np.linalg.lstsq(x0_joint, y0[:, 0], rcond=None)
    if rank != p or not np.allclose(x0_joint @ beta_gq, x0_joint @ beta_joint,
                                    rtol=2e-9, atol=2e-10):
        raise RuntimeError("SimpleGQ prediction is not the full-rank joint-design OLS fit")
    if not np.allclose(beta_gq, beta_joint, rtol=2e-8, atol=2e-9):
        raise RuntimeError("joint GQ coefficients differ from direct joint OLS")
    b0 = np.asarray(model.state_features(x0), dtype=float)
    da0 = (np.asarray(model.action_features(x0, a0), dtype=float)
           - np.asarray(model.action_features(x0, x0[:, model._memory_index]), dtype=float))
    fwl_covariance = _fwl_covariance(b0, da0)
    joint_covariance = np.linalg.solve(gram, np.eye(p))
    if not np.allclose(fwl_covariance, joint_covariance, rtol=3e-8, atol=3e-10):
        raise RuntimeError("FWL coefficient covariance differs from joint-design covariance")
    g_hat = model.predict_physical(x, a)[:, 0]
    m = w.shape[0]
    radius_reference_scalar = sigma * sqrt(float(chi2.ppf(1-alpha_reference, p)))
    ref_map = (w @ x_joint / n) @ a0_map
    radius_reference = radius_reference_scalar * np.linalg.norm(ref_map, axis=1)
    radius_noise = sigma * sqrt(2 * log(2*m*calendar_size/alpha_noise) / n)
    radii = radius_reference + radius_noise
    moments = w @ (y[:, 0] - g_hat) / n
    latent_bound, model_bound = float(latent_bound), float(model_bound)
    if (not isfinite(latent_bound) or latent_bound < 0
            or not isfinite(model_bound) or model_bound < 0):
        raise ValueError("source bounds must be finite and nonnegative")
    source_spec = _source_spec(moments, radii, w, a, .1+.2*a, n,
                               latent_bound, model_bound)
    return {
        "moments": moments.tolist(), "radii": radii.tolist(),
        "reference_radius": radius_reference.tolist(),
        "future_noise_radius": float(radius_noise),
        "source_spec": source_spec,
        "readiness": {"status": "ready", "computational_checks": [
            "SimpleGQ fit completed", "joint design full rank",
            "SimpleGQ coefficients and predictions match joint OLS"]},
        "fit_diagnostics": {
            "estimator": "SimpleGQ_unpenalized_FWL",
            "joint_design": "[state_features, action_features-anchor_features]",
            "joint_rank": int(rank), "joint_columns": int(p),
            "max_prediction_difference_from_joint_ols": float(np.max(np.abs(
                x0_joint @ beta_gq - x0_joint @ beta_joint))),
            "parameter_difference_from_joint_ols": float(np.max(np.abs(beta_gq-beta_joint))),
            "max_unit_variance_covariance_difference_from_joint_ols": float(
                np.max(np.abs(fwl_covariance-joint_covariance))),
        },
        "nominal_conditional_error_budget": alpha_reference + alpha_noise,
        "statistical_error_bounds": None,
        "statistical_error_bounds_status": "null_assumptions_unverified",
        "assumptions_status": "unverified",
        "statistical_radius_arithmetic": "binary64 chi-square and matrix arithmetic; ideal-real theorem, not outward-certified",

    }


def run_predictable_gq_two_stage(
    *, stage1: dict[str, Any], reference: dict[str, Any], future: dict[str, Any],
    sigma: float, alpha_reference: float = .025, alpha_noise: float = .025,
    calendar_size: int = 1, source_deadline: int, latent_bound: float = .6,
    model_bound: float = 2., assumptions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit held-out SimpleGQ and diagnose one predictable post-alarm window.

    ``reference`` keys: ``states``, ``actions``, ``next_physical``, ``dates``,
    ``record_ids``, ``memory_index``, ``physical_indices``, and the external
    provenance declaration ``independent_of_alarm_history``.

    ``future`` keys: ``states``, ``actions``, ``next_physical``, ``dates``,
    ``record_ids``, ``weights_predictable=True``, and ``weights`` with shape ``(m,n)``. The fixed denominator
    is the number n of future rows. ``stage1`` is an external alarm record with
    ``status='alarm'``, ``alarm_time`` and provenance ``record_ids``. The alarm
    record must also carry ``detector_reference_ids`` so ID overlap can be
    rejected. The calendar correction covers ``calendar_size`` predeclared
    looks; this call evaluates its supplied window only.

    The confidence radius is the Section 6 union/ellipsoid radius. Statistical
    size is conditional on the alarm and on every listed law/design assumption;
    this code cannot establish those assumptions from supplied data.
    """
    if not isinstance(stage1, dict) or stage1.get("status") != "alarm":
        raise ValueError("stage1 must be the external affirmative alarm record")
    alarm_time = _clock(stage1.get("alarm_time"), "stage1.alarm_time")
    deadline = _clock(source_deadline, "source_deadline")
    if deadline <= alarm_time:
        raise ValueError("source_deadline must follow the alarm")
    if not isinstance(reference, dict) or not isinstance(future, dict):
        raise ValueError("reference and future must be mappings")

    x0 = _array(reference.get("states"), "reference.states", 2)
    a0 = _array(reference.get("actions"), "reference.actions", 1)
    y0 = _array(reference.get("next_physical"), "reference.next_physical", 2)
    x = _array(future.get("states"), "future.states", 2)
    a = _array(future.get("actions"), "future.actions", 1)
    y = _array(future.get("next_physical"), "future.next_physical", 2)
    dates0 = np.asarray(reference.get("dates"))
    dates = np.asarray(future.get("dates"))
    ids0 = reference.get("record_ids")
    ids = future.get("record_ids")
    if (x0.shape[0] != len(a0) or x0.shape[0] != y0.shape[0]
            or x.shape[0] != len(a) or x.shape[0] != y.shape[0]
            or x0.shape[1] != x.shape[1] or y0.shape[1] != 1 or y.shape[1] != 1):
        raise ValueError("row dimensions must agree; this route supports scalar physical output")
    n = len(x)
    if n < 1:
        raise ValueError("future window must contain at least one row")
    w = _array(future.get("weights"), "future.weights", 2)
    m = w.shape[0]
    if w.shape[1] != n or m < 1 or np.any(w < 0) or np.any(w > 1):
        raise ValueError("weights must have shape (m,n) and lie in [0,1]")
    if (dates0.shape != (len(x0),) or dates.shape != (n,)
            or not np.issubdtype(dates0.dtype, np.integer)
            or not np.issubdtype(dates.dtype, np.integer)):
        raise ValueError("dates must be integer vectors matching their rows")
    if np.any(dates0 > alarm_time) or np.any(dates <= alarm_time) or np.any(dates > deadline):
        raise ValueError("reference must be old and every future row post-alarm by deadline")
    if (not isinstance(ids0, list) or not isinstance(ids, list)
            or len(ids0) != len(x0) or len(ids) != n
            or len(set(ids0)) != len(ids0) or len(set(ids)) != n):
        raise ValueError("reference and future record_ids must be unique lists aligned to rows")
    provenance = stage1.get("provenance")
    if not isinstance(provenance, dict) or not {
            "record_ids", "detector_reference_ids"}.issubset(provenance):
        raise ValueError("stage1 provenance must explicitly declare both detector ID lists")
    detector_ids = provenance["record_ids"]
    detector_reference_ids = provenance["detector_reference_ids"]
    if not isinstance(detector_ids, list) or not isinstance(detector_reference_ids, list):
        raise ValueError("stage1 provenance must declare record_ids and detector_reference_ids")
    if set(ids0) & (set(detector_ids) | set(detector_reference_ids)):
        raise ValueError("held-out source reference overlaps detector provenance")
    if set(ids) & (set(ids0) | set(detector_ids) | set(detector_reference_ids)):
        raise ValueError("future source rows overlap reference or detector provenance")
    if reference.get("independent_of_alarm_history") is not True:
        raise ValueError("held-out reference independence must be explicitly declared")
    if future.get("weights_predictable") is not True:
        raise ValueError("future weights must be explicitly declared predictable")

    sigma = float(sigma)
    alpha_reference, alpha_noise = float(alpha_reference), float(alpha_noise)
    latent_bound, model_bound = float(latent_bound), float(model_bound)
    if (not isfinite(sigma) or sigma <= 0 or not 0 < alpha_reference < 1
            or not 0 < alpha_noise < 1 or alpha_reference + alpha_noise >= 1
            or isinstance(calendar_size, bool) or not isinstance(calendar_size, int)
            or calendar_size < 1 or latent_bound < 0 or model_bound < 0):
        raise ValueError("invalid variance, error budget, calendar, or source bounds")

    prepared = prepare_predictable_gq_region(
        reference, future, sigma=sigma, alpha_reference=alpha_reference,
        alpha_noise=alpha_noise, calendar_size=calendar_size,
        latent_bound=latent_bound, model_bound=model_bound,
    )
    moments = np.asarray(prepared["moments"])
    radii = np.asarray(prepared["radii"])
    source = prepared["source_spec"]
    context = {
        "route": "predictable_post_alarm_simple_gq",
        "alarm_time": alarm_time, "source_deadline": deadline,
        "reference_record_ids": list(ids0), "future_record_ids": list(ids),
        "calendar_size": calendar_size, "window_rows": n, "moment_coordinates": m,
    }
    report = run_two_stage_report(stage1, context=context, source=source,
                                  readiness=prepared["readiness"])
    report["reliability"]["statistical_radius_arithmetic"] = (
        "binary64 chi-square quantile and matrix arithmetic; not outward-certified; "
        "theorem bounds describe ideal-real-arithmetic construction, not a machine-exact nominal guarantee")
    report["errors"]["source_joint"] = _errors("conditional_on_alarm")
    declared = dict(assumptions or {})
    assumptions_out = {
        name: {"status": "unverified_by_runner", "caller_declaration": declared.get(name)}
        for name in _REQUIRED_ASSUMPTIONS
    }
    all_asserted = (all(declared.get(name) is True for name in _REQUIRED_ASSUMPTIONS)
                    and reference.get("independent_of_alarm_history") is True
                    and future.get("weights_predictable") is True)
    conditional_bound = alpha_reference + alpha_noise if all_asserted else None
    report["predictable_gq"] = {
        "status": "computed",
        **prepared["fit_diagnostics"],
        "moments": moments.tolist(), "radii": radii.tolist(),
        "reference_radius": prepared["reference_radius"],
        "future_noise_radius": prepared["future_noise_radius"],
        "nominal_conditional_error_budget": alpha_reference + alpha_noise,
        "statistical_error_bounds": conditional_bound,
        "statistical_error_bounds_status": (
            "conditional_on_caller_declarations_not_independently_verified"
            if all_asserted else "null_assumptions_unverified"),
        "assumptions": assumptions_out,
        "geometry": {
            "CL": "dynamic rowwise latent b in [-B,B], loading W*diag(.1+.2*action)/N",
            "CM": "constant bounded action-response coefficient loading W*action/N",
            "CMIX": "sum of the declared CL and CM images",
        },
        "scope": "scalar physical outcome and linear state/action dictionaries; not generic GQ/RA-GQ or economic-Q",
    }
    # Assertions are retained as caller declarations. The API cannot independently
    # verify provenance or a stochastic law; only expose the theorem's conditional
    # budget when every required condition was affirmatively declared.
    # This module has only the conditional source theorem. Unconditional joint
    # source+detector error remains unavailable until a detector proof is joined.
    for key in ("latent", "model", "mixed", "source_joint"):
        report["errors"][key]["type_i"]["target_budget"] = (
            alpha_reference + alpha_noise)
        report["errors"][key]["type_i"]["proved_bound"] = (
            alpha_reference + alpha_noise if all_asserted else None)
        report["errors"][key]["type_i"]["status"] = (
            "conditional_on_declared_assumptions_not_independently_verified"
            if all_asserted else "not_established_assumptions_unverified")
        report["errors"][key]["type_i"]["assumptions"] = assumptions_out
        report["errors"][key]["type_ii"]["target_budget"] = None
        report["errors"][key]["type_ii"]["proved_bound"] = None
    return report
