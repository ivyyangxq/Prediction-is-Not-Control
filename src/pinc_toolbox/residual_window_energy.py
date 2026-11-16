"""Fixed-window residual energy for a permanently supported prefix.

This is a detector statistic only.  It consumes released prediction errors
and predictable support/weights supplied by the caller; it fits no model and
provides no source, Q, or certification claim.
"""
from __future__ import annotations

import math
import numpy as np


def _windows(value):
    if isinstance(value, (str, bytes)):
        raise ValueError("windows must be positive distinct integers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError("windows must be positive distinct integers") from exc
    if not values or any(isinstance(item, (bool, np.bool_)) or
                         not isinstance(item, (int, np.integer)) or item < 1
                         for item in values) or len(set(values)) != len(values):
        raise ValueError("windows must be positive distinct integers")
    return tuple(int(item) for item in values)


def window_slope_energy(residuals, weights, *, residual_left_inverse,
                        windows=(40, 80, 160), eligible=None,
                        denominator_floor=1e-15, cutoffs=None,
                        return_trace=False):
    """Return fixed-window slope energy with permanent first support exit.

    ``residuals`` has shape ``(paths, transitions, coordinates)`` and
    ``weights`` has shape ``(paths, transitions)``.  ``eligible`` is a boolean
    ``(paths, transitions, rules)`` array; a false entry at transition ``t``
    records ``first_exit=t`` and excludes that transition and every later one.
    A window ending at exclusive index ``end`` is available only when all
    transitions ``end-width:end`` were consumed.  ``valid_count`` and
    ``consumed_count`` count consumed transitions, while
    ``available_window_count`` counts available path/rule/width windows and
    ``scores`` is their maximum.
    ``first_crossing`` uses the zero-based final transition index of the
    crossing window and is ``steps``
    when no strict crossing occurs.  No full window yields ``-inf``; a full
    window with denominator at or below ``denominator_floor`` contributes zero.
    """
    errors = np.asarray(residuals, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    inverse = np.asarray(residual_left_inverse, dtype=float)
    if errors.ndim != 3 or weight_array.shape != errors.shape[:2]:
        raise ValueError("residuals and weights dimensions disagree")
    if inverse.shape != (errors.shape[2], errors.shape[2]):
        raise ValueError("residual_left_inverse dimensions disagree")
    if (not np.isfinite(errors).all() or not np.isfinite(weight_array).all()
            or not np.isfinite(inverse).all()):
        raise ValueError("residuals, weights, and whitening must be finite")
    floor = float(denominator_floor)
    if not math.isfinite(floor) or floor < 0:
        raise ValueError("denominator_floor must be finite and nonnegative")
    if not isinstance(return_trace, (bool, np.bool_)):
        raise ValueError("return_trace must be boolean")
    widths = _windows(windows)
    paths, steps, dimension = errors.shape
    if eligible is None:
        support = np.ones((paths, steps, 1), dtype=bool)
    else:
        support = np.asarray(eligible)
        if support.dtype != np.bool_ or support.ndim != 3 or support.shape[:2] != (paths, steps) or support.shape[2] < 1:
            raise ValueError("eligible must be a boolean paths x transitions x rules array")
    rules = support.shape[2]
    thresholds = None if cutoffs is None else np.asarray(cutoffs, dtype=float)
    if thresholds is not None and (thresholds.shape != (rules,) or np.isnan(thresholds).any()):
        raise ValueError("cutoffs must have one non-NaN value per rule")

    scores = np.full((paths, rules), -np.inf)
    first_exit = np.full((paths, rules), steps, dtype=np.int64)
    valid_count = np.zeros((paths, rules), dtype=np.int64)
    available_window_count = np.zeros((paths, rules), dtype=np.int64)
    first_crossing = np.full((paths, rules), steps, dtype=np.int64)
    first_crossing_score = np.full((paths, rules), -np.inf)
    active = np.ones((paths, rules), dtype=bool)
    prefix_numerator = np.zeros((paths, steps + 1, rules, dimension), dtype=float)
    prefix_denominator = np.zeros((paths, steps + 1, rules), dtype=float)
    if return_trace:
        score_trace = np.full((paths, steps, rules), -np.inf)
        window_energy_trace = np.full((paths, steps, rules), -np.inf)
        window_available_trace = np.zeros((paths, steps, rules), dtype=bool)
        active_trace = np.zeros((paths, steps, rules), dtype=bool)

    for transition in range(steps):
        active &= support[:, transition, :]
        first_exit[(first_exit == steps) & ~active] = transition
        valid_count += active
        live = active.any(axis=1)
        whitened = np.zeros((paths, dimension), dtype=float)
        if live.any():
            whitened[live] = errors[live, transition, :] @ inverse.T
        masked_weights = np.where(live, weight_array[:, transition], 0.0)
        contribution = masked_weights[:, None, None] * whitened[:, None, :]
        prefix_numerator[:, transition + 1] = prefix_numerator[:, transition]
        prefix_denominator[:, transition + 1] = prefix_denominator[:, transition]
        prefix_numerator[:, transition + 1] += np.where(active[:, :, None], contribution, 0.0)
        prefix_denominator[:, transition + 1] += np.where(
            active, masked_weights[:, None] ** 2, 0.0)
        if (not np.isfinite(prefix_numerator[:, transition + 1]).all()
                or not np.isfinite(prefix_denominator[:, transition + 1]).all()):
            raise ValueError("nonfinite consumed residual accumulator")
        if return_trace:
            active_trace[:, transition] = active
        for width in widths:
            end = transition + 1
            if end < width:
                continue
            begin = end - width
            available = active & (first_exit >= end)
            numer = prefix_numerator[:, end] - prefix_numerator[:, begin]
            denom = prefix_denominator[:, end] - prefix_denominator[:, begin]
            energy = np.zeros((paths, rules), dtype=float)
            if available.any():
                selected_numer = numer[available]
                selected_denom = denom[available]
                selected_energy = np.zeros(selected_denom.shape)
                nondegenerate = selected_denom > floor
                selected_energy[nondegenerate] = (
                    (selected_numer[nondegenerate] ** 2).sum(axis=1)
                    / selected_denom[nondegenerate])
                energy[available] = selected_energy
                if not np.isfinite(energy[available]).all():
                    raise ValueError("nonfinite accumulated residual energy")
            available_window_count += available
            current = np.where(available, energy, -np.inf)
            scores = np.maximum(scores, current)
            if return_trace:
                window_available_trace[:, transition] |= available
                window_energy_trace[:, transition] = np.maximum(window_energy_trace[:, transition], current)
        if thresholds is not None:
            available = active & (transition + 1 >= min(widths))
            hit = available & (first_crossing == steps) & (scores > thresholds[None, :])
            first_crossing[hit] = transition
            first_crossing_score[hit] = scores[hit]
        if return_trace:
            score_trace[:, transition] = scores

    result = {
        "scores": scores,
        "first_exit": first_exit,
        "valid_count": valid_count,
        "consumed_count": valid_count,
        "available_window_count": available_window_count,
        "windows": widths,
        "statistic": "window_slope_energy",
        "method_module": __name__,
        "source_certification": "not_supplied",
        "Q": "not_applicable",
    }
    if thresholds is not None:
        result.update(first_crossing=first_crossing,
                      first_crossing_score=first_crossing_score)
    if return_trace:
        result.update(score_trace=score_trace,
                      window_energy_trace=window_energy_trace,
                      window_available_trace=window_available_trace,
                      active_trace=active_trace)
    return result


__all__ = ["window_slope_energy"]
