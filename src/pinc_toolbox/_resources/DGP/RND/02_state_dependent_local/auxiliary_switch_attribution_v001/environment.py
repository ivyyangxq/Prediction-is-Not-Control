"""Bounded RND2 auxiliary switch-attribution environment.

This module is deliberately small and self contained.  It reproduces the
RND2 physical/source law constants while exposing controlled model and latent
switches for the attribution experiment.  ``latent`` and ``proxy_all`` are
evaluator state; :meth:`public` exposes only the state and first proxy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SCENARIOS = ("STABLE", "MODEL_A", "MODEL_M", "LATENT", "JOINT", "BEHAVIOR")
SOURCE_MODULE = (
    "Boundaries/ChangeInformation/EpsilonRepair/source_v001/"
    "run_rnd2_same_parameter_detect_to_repair_v001.py"
)

# Exact source constants (do not round or refit these in the runner).
P_PRE = 0.80
ACTION_SLOPE = 0.80
ACTION_CURVATURE = 2.0
STATE_ACTION_LOADING = 0.50
ETA = np.array([0.08, -0.06, 0.05], dtype=float)
LAMBDA = np.array([0.35, -0.25, 0.20], dtype=float)
F_PRE = np.array([[0.72, 0.08, 0.00], [0.03, 0.65, 0.06], [0.00, 0.05, 0.58]], dtype=float)
LATENT_INNOVATION_SD = np.array([0.040, 0.035, 0.030], dtype=float)
STATE_INNOVATION_SD = 0.020
MEASUREMENT_NOISE_SD = 0.020
PROXY_LOADING = np.array(
    [[1.0, 0.2, -0.1], [0.7, -0.3, 0.4], [-0.4, 1.0, 0.2]], dtype=float
)
ACTION_GRID = np.linspace(0.0, 0.10, 11)
REFERENCE_ACTION = 0.05
DISCOUNT = 0.95
TAU1 = REFERENCE_ACTION - DISCOUNT / (1.0 - DISCOUNT * P_PRE) * (0.8 - 4.0 * REFERENCE_ACTION) + 0.3


def _stationary_latent(rng: np.random.Generator, paths: int) -> np.ndarray:
    z = np.zeros((paths, 3), dtype=float)
    for _ in range(250):
        z = z @ F_PRE.T + rng.normal(size=(paths, 3)) * LATENT_INNOVATION_SD
    return z


def _validate_scenario(scenario: str) -> str:
    scenario = str(scenario).upper()
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    return scenario


def source_law(scenario: str, date: int, switch_date: int = 8, strength: float = 1.0) -> dict[str, Any]:
    """Return the structural law used for transition ``date -> date+1``.

    The switch is applied to transition date ``tau``.  Consequently the first
    changed released observation is date ``tau+1``; no public information at
    date ``tau`` contains the post-switch transition.
    """
    scenario = _validate_scenario(scenario)
    post = int(date) >= int(switch_date) and scenario != "STABLE" and scenario != "BEHAVIOR"
    p = P_PRE
    action_shift = 0.0
    F = F_PRE
    if post and scenario in ("MODEL_A", "JOINT"):
        # Exact registered model-A perturbation.
        action_shift = -0.6 * float(strength)
    if post and scenario == "MODEL_M":
        p = P_PRE + 0.15 * float(strength)
    if post and scenario in ("LATENT", "JOINT"):
        F = F_PRE.copy()
        F[0, 0] = 0.72 - 1.27 * float(strength)
    return {"p": float(p), "action_shift": float(action_shift), "F": np.array(F, copy=True), "post": bool(post)}


@dataclass
class Environment:
    """Paired finite-horizon RND2 host.

    ``state`` and ``proxy`` have H+1 dated observations.  ``latent`` and
    ``proxy_all`` are intentionally separate evaluator-only arrays.  Callers
    should invoke ``step(t, actions)`` exactly once for each t in 0..H-1.
    """

    seed: int
    scenario: str
    paths: int
    horizon: int = 24
    switch_date: int = 8
    strength: float = 1.0

    def __post_init__(self) -> None:
        self.seed = int(self.seed)
        self.paths = int(self.paths)
        self.horizon = int(self.horizon)
        self.switch_date = int(self.switch_date)
        self.strength = float(self.strength)
        self.scenario = _validate_scenario(self.scenario)
        if self.paths < 1 or self.horizon < 1:
            raise ValueError("paths and horizon must be positive")
        if not 0 <= self.switch_date < self.horizon:
            raise ValueError("switch_date must be in [0, horizon)")
        self.rng = np.random.default_rng(self.seed)
        # One tape, generated before any stepping, makes paired scenarios share
        # every initial/state/latent/measurement innovation.
        self.state_noise = self.rng.normal(size=(self.paths, self.horizon)) * STATE_INNOVATION_SD
        self.latent_noise = self.rng.normal(size=(self.paths, self.horizon, 3)) * LATENT_INNOVATION_SD
        self.measurement_noise = self.rng.normal(size=(self.paths, self.horizon + 1, 3)) * MEASUREMENT_NOISE_SD
        self.state = np.empty((self.paths, self.horizon + 1), dtype=float)
        self.latent = np.empty((self.paths, self.horizon + 1, 3), dtype=float)
        self.proxy_all = np.empty((self.paths, self.horizon + 1, 3), dtype=float)
        self.action = np.zeros((self.paths, self.horizon), dtype=float)
        self.reward = np.zeros((self.paths, self.horizon), dtype=float)
        self.latent[:, 0] = _stationary_latent(self.rng, self.paths)
        self.state[:, 0] = 0.13 + 0.035 * self.rng.normal(size=self.paths)
        self.proxy_all[:, 0] = self.latent[:, 0] @ PROXY_LOADING.T + self.measurement_noise[:, 0]
        self.w = self.proxy_all[:, :, 0]
        self.exploration_probability = self.exploration_probability_at(0)
        self.metadata = simulation_metadata(self.scenario, self.switch_date, self.strength, self.horizon)

    def public(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Return learner-visible ``(state, first_proxy)`` at date t only."""
        t = int(t)
        if not 0 <= t <= self.horizon:
            raise IndexError(t)
        return self.state[:, t].copy(), self.w[:, t].copy()

    def exploration_probability_at(self, t: int) -> float:
        """Declared behavior schedule; the environment does not draw actions."""
        return 0.30 if self.scenario == "BEHAVIOR" and int(t) >= self.switch_date else 0.10

    def step(self, t: int, actions: np.ndarray | float) -> None:
        t = int(t)
        if not 0 <= t < self.horizon:
            raise IndexError(t)
        action = np.asarray(actions, dtype=float).reshape(-1)
        if action.size == 1 and self.paths != 1:
            action = np.repeat(action, self.paths)
        if action.size != self.paths or not np.all(np.isfinite(action)):
            raise ValueError("actions must be one finite value per path")
        law = source_law(self.scenario, t, self.switch_date, self.strength)
        s, z = self.state[:, t], self.latent[:, t]
        self.action[:, t] = action
        self.reward[:, t] = s - 0.25 * s**2 - 0.50 * action**2 + TAU1 * action
        self.state[:, t + 1] = (
            law["p"] * s + z @ ETA + ACTION_SLOPE * action - ACTION_CURVATURE * action**2
            + STATE_ACTION_LOADING * s * action + action * (z @ LAMBDA)
            + law["action_shift"] * (action - REFERENCE_ACTION) + self.state_noise[:, t]
        )
        self.latent[:, t + 1] = z @ law["F"].T + self.latent_noise[:, t]
        self.proxy_all[:, t + 1] = self.latent[:, t + 1] @ PROXY_LOADING.T + self.measurement_noise[:, t + 1]

    @property
    def latent_truth(self) -> np.ndarray:
        return self.latent

    @property
    def all_proxies(self) -> np.ndarray:
        return self.proxy_all


def simulation_metadata(scenario: str, switch_date: int = 8, strength: float = 1.0, horizon: int = 24) -> dict[str, Any]:
    scenario = _validate_scenario(scenario)
    structural = scenario not in ("STABLE", "BEHAVIOR")
    return {
        "dgp": "RND2-AUX-SWITCH-001", "scenario": scenario, "horizon": int(horizon),
        "switch_date": int(switch_date), "strength": float(strength),
        "first_affected_release": int(switch_date) + 1 if structural else None,
        "detector_can_first_see": int(switch_date) + 1 if structural else None,
        "latent_F_change_transmission": "latent transition at tau; proxy/state effects may lag one physical observation",
        "source_module": SOURCE_MODULE, "learner_public": ("state", "w"),
        "evaluator_arrays": ("latent_truth", "all_proxies"),
        "behavior_exploration_probability": 0.30 if scenario == "BEHAVIOR" else 0.10,
    }


__all__ = ["Environment", "SCENARIOS", "source_law", "simulation_metadata"]
