"""Deterministic public-state initializer and host for transitions 40..47.

Inputs are sanitized release-40 public values, a caller-supplied z_40 posterior
draw, and already-scaled independent future additive innovations. This module
never constructs or samples a DGP environment.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

# Constants copied from the actual RND2 source chain. The equations below use
# row vectors to match model_source_v004's Environment.step implementation.
P0 = 0.80
ACTION_SLOPE = 0.80
ACTION_CURVATURE = 2.0
STATE_ACTION_LOADING = 0.50
ETA = np.array([0.08, -0.06, 0.05])
LAMBDA = np.array([0.35, -0.25, 0.20])
F0 = np.array([[0.72, 0.08, 0.00], [0.03, 0.65, 0.06], [0.00, 0.05, 0.58]])
FSTAR = np.array([[-0.55, 0.08, 0.00], [0.03, 0.85, 0.06], [0.00, 0.05, 0.30]])
PROXY_LOADING = np.array([[1.0, 0.2, -0.1], [0.7, -0.3, 0.4], [-0.4, 1.0, 0.2]])
ACTION_GRID = np.linspace(0.0, 0.10, 11)
A0 = 0.05
TAU1 = A0 - 0.95 / (1.0 - 0.95 * P0) * (0.8 - 4.0 * A0) + 0.3
MODEL_OFFSET_COEFFICIENT = -1.2 * 0.3875
INITIAL_DATE = 40
FINAL_DATE = 48
HORIZON = FINAL_DATE - INITIAL_DATE


class PublicForwardHost:
    """Minimal compatible host initialized at observed public date 40.

    The host accepts no seed, environment, latent truth, oracle metadata, or
    prefix inputs. Earlier public dates are inaccessible. Supplied innovation
    arrays contain additive shocks in the source's actual units, including
    their declared standard-deviation scaling.
    """

    def __init__(self, *, candidate: str, x40: object, w40: object,
                 previous_action39: object, posterior_draw: object,
                 state_noise: object, latent_noise: object,
                 measurement_noise: object) -> None:
        candidate = str(candidate).upper()
        if candidate not in {"N", "CL", "CM", "MIX"}:
            raise ValueError("candidate must be one of 'N', 'CL', 'CM', or 'MIX'")
        x0 = self._array("x40", x40)
        w0 = self._array("w40", w40)
        previous = self._array("previous_action39", previous_action39)
        z0 = self._array("posterior_draw", posterior_draw)
        if x0.ndim != 1 or x0.size < 1:
            raise ValueError("x40 must be a nonempty vector with shape (paths,)")
        paths = x0.size
        if w0.shape != (paths,) or previous.shape != (paths,) or z0.shape != (paths, 3):
            raise ValueError("w40, previous_action39, or posterior_draw has the wrong shape")
        self._validate_actions(previous, "previous_action39")

        future_state = self._array("state_noise", state_noise)
        future_latent = self._array("latent_noise", latent_noise)
        future_measurement = self._array("measurement_noise", measurement_noise)
        if future_state.shape != (paths, HORIZON):
            raise ValueError("state_noise must have shape (paths, 8) for transitions 40..47")
        if future_latent.shape != (paths, HORIZON, 3):
            raise ValueError("latent_noise must have shape (paths, 8, 3) for transitions 40..47")
        if future_measurement.shape != (paths, HORIZON, 3):
            raise ValueError("measurement_noise must have shape (paths, 8, 3) for releases 41..48")

        self.candidate = candidate
        self.paths = paths
        self.horizon = FINAL_DATE
        self.switch_date = 32
        self.strength = 1.0
        self.profile = "step"
        self._next_t = INITIAL_DATE
        self._initial_previous_action39 = previous.copy()
        self.previous_action = previous.copy()

        # Allocate absolute-date arrays, leaving dates before 40 unavailable.
        # NaN marks absent historical content; the public API refuses it.
        self.state = np.full((paths, FINAL_DATE + 1), np.nan, dtype=float)
        self.latent = np.full((paths, FINAL_DATE + 1, 3), np.nan, dtype=float)
        self.proxy_all = np.full((paths, FINAL_DATE + 1, 3), np.nan, dtype=float)
        self.action = np.full((paths, FINAL_DATE), np.nan, dtype=float)
        self.reward = np.full((paths, FINAL_DATE), np.nan, dtype=float)
        self.original_reward = np.full((paths, FINAL_DATE), np.nan, dtype=float)
        self.reward_observation = np.full((paths, FINAL_DATE), np.nan, dtype=float)

        self.state[:, INITIAL_DATE] = x0.copy()
        self.latent[:, INITIAL_DATE] = z0.copy()
        self.proxy_all[:, INITIAL_DATE, 0] = w0.copy()
        self.action[:, INITIAL_DATE - 1] = previous.copy()
        self.w = self.proxy_all[:, :, 0]

        # Preserve source-class bookkeeping. For N/CM baseline equals the
        # actual latent. For CL/MIX it is an auxiliary F0 trajectory: the
        # candidate's released x/w equations use `latent`, never this field.
        self._baseline_latent = np.full_like(self.latent, np.nan)
        self._baseline_latent[:, INITIAL_DATE] = z0.copy()

        self.state_noise = np.full((paths, FINAL_DATE), np.nan, dtype=float)
        self.latent_noise = np.full((paths, FINAL_DATE, 3), np.nan, dtype=float)
        self.measurement_noise = np.full((paths, FINAL_DATE + 1, 3), np.nan, dtype=float)
        self.state_noise[:, INITIAL_DATE:FINAL_DATE] = future_state.copy()
        self.latent_noise[:, INITIAL_DATE:FINAL_DATE] = future_latent.copy()
        self.measurement_noise[:, INITIAL_DATE + 1:FINAL_DATE + 1] = future_measurement.copy()

        self.g_profile = np.zeros(FINAL_DATE + 1, dtype=float)
        self.g_profile[self.switch_date:] = 1.0
        self.g_t = self.g_profile
        parent_source = {"N": "STABLE", "CL": "LATENT_F", "CM": "MODEL_A", "MIX": "MODEL_LATENT"}[candidate]
        self.source = parent_source
        self._chosen_source = parent_source
        self._parent_source = "MODEL_A" if candidate == "MIX" else parent_source
        self.metadata: dict[str, Any] = {
            "dgp": "RND2-MODEL-SOURCE-004-forward-initialized",
            "source": parent_source,
            "candidate": candidate,
            "initial_public_date": INITIAL_DATE,
            "next_transition": INITIAL_DATE,
            "final_public_date": FINAL_DATE,
            "learner_public": ("state", "w", "released_reward"),
            "initializer_inputs": ("x40", "w40", "previous_action39", "posterior_draw",
                                   "state_noise", "latent_noise", "measurement_noise"),
        }

    @staticmethod
    def _array(name: str, value: object) -> np.ndarray:
        try:
            result = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric array") from exc
        if not np.isfinite(result).all():
            raise ValueError(f"{name} must contain only finite values")
        return result.copy()

    @staticmethod
    def _validate_actions(actions: np.ndarray, name: str = "actions") -> None:
        if np.any(actions < ACTION_GRID[0]) or np.any(actions > ACTION_GRID[-1]):
            raise ValueError(f"{name} are outside [0, .1]")

    def public(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        t = int(t)
        if not INITIAL_DATE <= t <= self._next_t:
            raise IndexError("requested date is outside the initialized released public history")
        return self.state[:, t].copy(), self.w[:, t].copy()

    def step(self, t: int, actions: object) -> None:
        t = int(t)
        if t != self._next_t or t >= FINAL_DATE:
            raise ValueError("nonsequential or out-of-horizon transition")
        action = self._array("actions", actions).reshape(-1)
        if action.size == 1 and self.paths != 1:
            action = np.repeat(action, self.paths)
        if action.shape != (self.paths,):
            raise ValueError("actions must be one finite value per path")
        self._validate_actions(action)

        g_t = float(self.g_profile[t])
        x = self.state[:, t]
        z = self.latent[:, t]
        self.action[:, t] = action
        self.previous_action = action.copy()

        base_state_next = (
            P0 * x + z @ ETA + ACTION_SLOPE * action - ACTION_CURVATURE * action**2
            + STATE_ACTION_LOADING * x * action + action * (z @ LAMBDA)
            + self.state_noise[:, t]
        )
        c_t = MODEL_OFFSET_COEFFICIENT * g_t if self.candidate in {"CM", "MIX"} else 0.0
        state_next = base_state_next + c_t * (action - A0)

        baseline_next = self._baseline_latent[:, t] @ F0.T + self.latent_noise[:, t]
        self._baseline_latent[:, t + 1] = baseline_next
        if self.candidate in {"CL", "MIX"}:
            f_t = F0 + 0.69375 * g_t * (FSTAR - F0)
            latent_next = z @ f_t.T + self.latent_noise[:, t]
        else:
            latent_next = baseline_next

        self.state[:, t + 1] = state_next
        self.latent[:, t + 1] = latent_next
        self.proxy_all[:, t + 1] = (
            latent_next @ PROXY_LOADING.T + self.measurement_noise[:, t + 1]
        )
        deterministic_reward = x - 0.25 * x**2 - 0.50 * action**2 + TAU1 * action
        self.original_reward[:, t] = deterministic_reward
        self.reward[:, t] = deterministic_reward
        self.reward_observation[:, t] = deterministic_reward
        self._next_t = t + 1

    def released_reward(self, t: int) -> np.ndarray:
        t = int(t)
        if t < INITIAL_DATE + 1 or t > self._next_t:
            raise IndexError("requested reward is not released")
        return self.reward_observation[:, t - 1].copy()

    def fork(self) -> "PublicForwardHost":
        """Copy host state while preserving the proxy-array/w alias in the fork."""
        return copy.deepcopy(self)

    def __deepcopy__(self, memo: dict[int, object]) -> "PublicForwardHost":
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key != "w":
                setattr(result, key, copy.deepcopy(value, memo))
        result.w = result.proxy_all[:, :, 0]
        return result


__all__ = ["PublicForwardHost", "INITIAL_DATE", "FINAL_DATE", "HORIZON"]
