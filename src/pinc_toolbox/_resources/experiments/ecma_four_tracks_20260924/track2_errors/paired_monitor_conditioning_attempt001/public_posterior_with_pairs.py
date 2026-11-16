# Experimental owned extension; original MAIN-only filter is unchanged.
# Same-state/common-innovation acquisition must be established externally.
"""Public-history Gaussian filtering for the four known RND2 candidates.

This module consumes only sanitized public arrays. It has no environment,
seed, latent-state, innovation, or future-tape interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from exact_update import condition_exact

# Copied from the actual source chain (mother -> ladder -> source separation
# -> model_source_v004); column-vector convention is used in this module.
F0 = np.array([[0.72, 0.08, 0.00], [0.03, 0.65, 0.06], [0.00, 0.05, 0.58]])
FSTAR = np.array([[-0.55, 0.08, 0.00], [0.03, 0.85, 0.06], [0.00, 0.05, 0.30]])
ETA = np.array([0.08, -0.06, 0.05])
LAMBDA = np.array([0.35, -0.25, 0.20])
H = np.array([1.0, 0.2, -0.1])
Q = np.diag(np.array([0.040, 0.035, 0.030]) ** 2)
OBSERVATION_VARIANCE = 0.020 ** 2
SWITCH_DATE = 32
ACTION_GRID = np.linspace(0.0, 0.10, 11)
REFERENCE_ACTION = 0.05
# Exact operation order in auxiliary_switch_attribution_v001.
MODEL_OFFSET_COEFFICIENT = -1.2 * 0.3875
# Exact reward calibration expression from that same source.
P_PRE = 0.80
DISCOUNT = 0.95
TAU1 = REFERENCE_ACTION - DISCOUNT / (1.0 - DISCOUNT * P_PRE) * (
    0.8 - 4.0 * REFERENCE_ACTION
) + 0.3


@dataclass(frozen=True)
class GaussianPosterior:
    """Analytic Gaussian posterior evaluated with floating-point arithmetic."""

    candidate: str
    means: np.ndarray  # path x 3, posterior z_T mean
    covariances: np.ndarray  # path x 3 x 3, posterior z_T covariance
    final_date: int


def finite_burn_covariance() -> np.ndarray:
    """Covariance after exactly 250 F0 updates from a zero state."""
    covariance = np.zeros((3, 3), dtype=float)
    for _ in range(250):
        covariance = F0 @ covariance @ F0.T + Q
        covariance = 0.5 * (covariance + covariance.T)
    return covariance


def _as_numeric_array(name: str, value: object) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _source_law(candidate: str, transition_date: int) -> tuple[np.ndarray, float]:
    post = transition_date >= SWITCH_DATE
    if candidate == "N":
        return F0, 0.0
    if candidate == "CL":
        f = F0 + 0.69375 * (FSTAR - F0) if post else F0
        return f, 0.0
    if candidate == "CM":
        return F0, MODEL_OFFSET_COEFFICIENT if post else 0.0
    if candidate == "MIX":
        f = F0 + 0.69375 * (FSTAR - F0) if post else F0
        return f, MODEL_OFFSET_COEFFICIENT if post else 0.0
    raise ValueError("candidate must be one of 'N', 'CL', 'CM', or 'MIX'")


def _scalar_update(mean: np.ndarray, covariance: np.ndarray,
                   loading: np.ndarray, observation: float) -> tuple[np.ndarray, np.ndarray]:
    """Condition on one scalar Gaussian measurement using Joseph covariance form."""
    innovation_variance = float(loading @ covariance @ loading + OBSERVATION_VARIANCE)
    if not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
        raise FloatingPointError("nonpositive or non-finite Gaussian innovation variance")
    cross = covariance @ loading
    gain = cross / innovation_variance
    innovation = float(observation - loading @ mean)
    updated_mean = mean + gain * innovation
    residual_map = np.eye(3) - np.outer(gain, loading)
    updated_covariance = (residual_map @ covariance @ residual_map.T
                          + OBSERVATION_VARIANCE * np.outer(gain, gain))
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    if not np.isfinite(updated_mean).all() or not np.isfinite(updated_covariance).all():
        raise FloatingPointError("non-finite Gaussian update")
    if np.linalg.eigvalsh(updated_covariance)[0] < -1e-12:
        raise FloatingPointError("Gaussian update covariance lost positive definiteness")
    return updated_mean, updated_covariance


def filter_public_history(*, candidate: str, x: object, w: object,
                          actions: object, rewards: object,
                          dates: object, paired_observations=None) -> GaussianPosterior:
    """Filter a complete public transcript dated 0..T under one candidate.

    ``x`` and ``w`` are path-by-date arrays for dates 0..T. ``actions`` and
    ``rewards`` are path-by-transition arrays for transitions 0..T-1; the
    reward at t is checked against the source's deterministic reward equation.
    ``dates`` must explicitly equal [0, ..., T], which rules out missing date zero/gaps.

    The result is path-factorized under the maintained independent-path,
    public-nonanticipating policy model. Rewards and actions are checked as
    transcript fields; the function does not certify policy execution. No
    history likelihood is calculated and no candidate is scored or excluded.
    """
    candidate = str(candidate).upper()
    if candidate not in {"N", "CL", "CM", "MIX"}:
        raise ValueError("candidate must be one of 'N', 'CL', 'CM', or 'MIX'")

    x_arr = _as_numeric_array("x", x)
    w_arr = _as_numeric_array("w", w)
    a_arr = _as_numeric_array("actions", actions)
    r_arr = _as_numeric_array("rewards", rewards)
    if x_arr.ndim != 2 or w_arr.ndim != 2:
        raise ValueError("x and w must have shape (paths, T+1)")
    if x_arr.shape != w_arr.shape or x_arr.shape[0] < 1 or x_arr.shape[1] < 1:
        raise ValueError("x and w must have matching nonempty path-by-date shapes")
    paths, date_count = x_arr.shape
    transitions = date_count - 1
    if a_arr.shape != (paths, transitions) or r_arr.shape != (paths, transitions):
        raise ValueError("actions and rewards must have shape (paths, T)")
    date_arr = _as_numeric_array("dates", dates)
    expected = np.arange(date_count, dtype=float)
    if date_arr.ndim != 1 or not np.array_equal(date_arr, expected):
        raise ValueError("dates must be the complete integer date sequence 0..T")

    # The actual public action grid is [0,.01,...,.10].
    if np.any(a_arr < ACTION_GRID[0]) or np.any(a_arr > ACTION_GRID[-1]):
        raise ValueError("actions are outside the public action bounds [0, .1]")
    nearest_grid = ACTION_GRID[np.argmin(np.abs(a_arr[..., None] - ACTION_GRID), axis=-1)]
    if not np.allclose(a_arr, nearest_grid, rtol=0.0, atol=1e-12):
        raise ValueError("actions must lie on the declared 0.01 action grid")

    expected_rewards = x_arr[:, :-1] - 0.25 * x_arr[:, :-1] ** 2 - 0.50 * a_arr ** 2 + TAU1 * a_arr
    if not np.allclose(r_arr, expected_rewards, rtol=1e-12, atol=1e-12):
        raise ValueError("rewards do not match the deterministic public reward equation")

    # Optional acquired endpoints are additional observed information, not labels.
    pairs = {}
    for row in (() if paired_observations is None else paired_observations):
        path, t = row['path'], row['time']
        if type(path) is not int or type(t) is not int or not (0 <= path < paths and 0 <= t < transitions):
            raise ValueError('invalid paired observation path/time')
        if (path,t) in pairs:
            raise ValueError('duplicate paired observation')
        if row['release_time'] != t+1:
            raise ValueError('paired observation release mismatch')
        low,high = float(row['low_action']),float(row['high_action'])
        values = [low, high, row['low_next_x'], row['high_next_x'], row['state_x']]
        if not np.isfinite(values).all() or not 0 <= low < high <= .1:
            raise ValueError('invalid paired endpoints')
        if any(abs(a-ACTION_GRID).min()>1e-12 for a in (low,high)):
            raise ValueError('paired actions outside declared grid')
        if not np.isclose(row['state_x'],x_arr[path,t],rtol=1e-12,atol=1e-12):
            raise ValueError('probe and main initial state differ')
        contrast = ((row['high_next_x']-row['low_next_x'])/(high-low)
                    - (.8-2*(high+low)+.5*x_arr[path,t]))
        pairs[path,t] = contrast - _source_law(candidate,t)[1]

    # x_0 is checked above as finite public data, but is independent of z_0.
    initial_covariance = finite_burn_covariance()
    means = np.empty((paths, 3), dtype=float)
    covariances = np.empty((paths, 3, 3), dtype=float)
    for path in range(paths):
        mean = np.zeros(3, dtype=float)
        covariance = initial_covariance.copy()
        # w_0 is an observation of z_0 and must be used before transition 0.
        mean, covariance = _scalar_update(mean, covariance, H, float(w_arr[path, 0]))
        for t in range(transitions):
            if (path,t) in pairs:
                mean,covariance=condition_exact(mean,covariance,LAMBDA,pairs[path,t])
            loading = ETA + a_arr[path, t] * LAMBDA
            offset = (0.8 * x_arr[path, t] + 0.8 * a_arr[path, t]
                      - 2.0 * a_arr[path, t] ** 2
                      + 0.5 * x_arr[path, t] * a_arr[path, t]
                      + _source_law(candidate, t)[1] * (a_arr[path, t] - REFERENCE_ACTION))
            physical_residual = float(x_arr[path, t + 1] - offset)
            # x_(t+1) measures z_t. Only after conditioning on it do we
            # propagate the latent state through F_t and add Q.
            mean, covariance = _scalar_update(mean, covariance, loading, physical_residual)
            transition, _ = _source_law(candidate, t)
            mean = transition @ mean
            covariance = transition @ covariance @ transition.T + Q
            covariance = 0.5 * (covariance + covariance.T)
            if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
                raise FloatingPointError("non-finite Gaussian prediction")
            if np.linalg.eigvalsh(covariance)[0] <= 0.0:
                raise FloatingPointError("Gaussian prediction covariance lost positive definiteness")
            mean, covariance = _scalar_update(mean, covariance, H, float(w_arr[path, t + 1]))
        means[path] = mean
        covariances[path] = covariance
    return GaussianPosterior(candidate, means, covariances, transitions)
