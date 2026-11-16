"""Thin source-specific extensions of the source-separation environment."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PARENT_PATH = Path(__file__).resolve().parents[1] / "source_separation_v002" / "environment.py"
_SPEC = importlib.util.spec_from_file_location("model_source_parent", _PARENT_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load parent environment from {_PARENT_PATH}")
_PARENT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PARENT
_SPEC.loader.exec_module(_PARENT)

_PARENT_SCENARIOS = tuple(_PARENT.SCENARIOS)
SCENARIOS = _PARENT_SCENARIOS + ("MODEL_POS", "LATENT_MEAN", "MODEL_LATENT")
BOUNDARY_SCENARIOS = ("LATENT_MATCHED_ACTION",)
PROFILE_NAMES = _PARENT.PROFILE_NAMES
ACTION_GRID = _PARENT.ACTION_GRID.copy()
A0 = float(_PARENT.A0)
F0 = _PARENT.F0.copy()
FSTAR = _PARENT.FSTAR.copy()
ETA = _PARENT.ETA.copy()
LAMBDA = _PARENT.LAMBDA.copy()
C1 = _PARENT.C1.copy()
V = _PARENT.V.copy()
KAPPA = float(_PARENT.KAPPA)
VSTRONG = V * (-0.465 / KAPPA)
P0 = float(_PARENT.P0)
ACTION_SLOPE = float(_PARENT.ACTION_SLOPE)
ACTION_CURVATURE = float(_PARENT.ACTION_CURVATURE)
STATE_ACTION_LOADING = float(_PARENT.STATE_ACTION_LOADING)
TAU1 = float(_PARENT.TAU1)


def _validate_source(source: str) -> str:
    source = str(source).upper()
    if source not in SCENARIOS + BOUNDARY_SCENARIOS:
        raise ValueError(f"unknown source {source!r}; expected one of {SCENARIOS + BOUNDARY_SCENARIOS}")
    return source


class Environment(_PARENT.Environment):
    """Parent fixture plus isolated model-source variants."""

    _PARENT_SOURCE = {"MODEL_POS": "MODEL_A", "LATENT_MEAN": "STABLE", "MODEL_LATENT": "MODEL_A",
                      "LATENT_MATCHED_ACTION": "STABLE"}

    def __init__(self, seed: int, source: str, paths: int = 32, horizon: int = 80,
                 switch_date: int = 24, strength: float = 1.0,
                 profile: str = "step", reward_noise_sd: float = 0.0) -> None:
        chosen = _validate_source(source)
        self._chosen_source = chosen
        parent_source = self._PARENT_SOURCE.get(chosen, chosen)
        super().__init__(seed, parent_source, paths=paths, horizon=horizon,
                         switch_date=switch_date, strength=strength,
                         profile=profile, reward_noise_sd=reward_noise_sd)
        self._parent_source = parent_source
        self.source = chosen
        if chosen == "LATENT_MATCHED_ACTION":
            self.latent[:, 0] = self._baseline_latent[:, 0] + self.g_profile[0] * VSTRONG
            self.proxy_all[:, 0] = self.latent[:, 0] @ _PARENT.PROXY_LOADING.T + self.measurement_noise[:, 0]
        self.metadata["dgp"] = "RND2-MODEL-SOURCE-004"
        self.metadata["source"] = chosen
        if chosen in {"LATENT_MEAN", "LATENT_MATCHED_ACTION"}:
            changed = np.any(self.g_profile != 0)
            self.metadata["first_affected_release"] = self.switch_date + 1 if changed else None

    def _parent_step(self, t: int, actions: np.ndarray) -> None:
        old_source = self.source
        self.source = self._parent_source
        try:
            super().step(t, actions)
        finally:
            self.source = old_source

    def step(self, t: int, actions: np.ndarray | float) -> None:
        t = int(t)
        if self._chosen_source == "LATENT_MATCHED_ACTION":
            return self._step_latent_matched_action(t, actions)
        if self._chosen_source not in {"MODEL_POS", "LATENT_MEAN", "MODEL_LATENT"}:
            return super().step(t, actions)
        a = _PARENT._validate_actions(actions, self.paths)
        old_latent = self.latent[:, t].copy()
        g_t = float(self.g_profile[t])
        self._parent_step(t, a)
        if self._chosen_source == "MODEL_POS":
            self.state[:, t + 1] += 0.93 * g_t * (a - A0)
        else:
            if self._chosen_source == "LATENT_MEAN":
                latent_next = old_latent @ F0.T + self.latent_noise[:, t] + g_t * np.array([0.04, -0.03, 0.02])
            else:
                F_t = F0 + 0.69375 * g_t * (FSTAR - F0)
                latent_next = old_latent @ F_t.T + self.latent_noise[:, t]
            self.latent[:, t + 1] = latent_next
            self.proxy_all[:, t + 1] = latent_next @ _PARENT.PROXY_LOADING.T + self.measurement_noise[:, t + 1]

    def _step_latent_matched_action(self, t: int, actions: np.ndarray | float) -> None:
        if t != self._next_t or t >= self.horizon:
            raise ValueError("nonsequential or out-of-horizon transition")
        a = _PARENT._validate_actions(actions, self.paths)
        z0 = self.latent[:, t]
        x = self.state[:, t]
        self.action[:, t] = a
        self.state[:, t + 1] = (P0 * x + z0 @ ETA + ACTION_SLOPE * a - ACTION_CURVATURE * a**2
                                + STATE_ACTION_LOADING * x * a + a * (z0 @ LAMBDA)
                                + self.state_noise[:, t])
        baseline_next = self._baseline_latent[:, t] @ F0.T + self.latent_noise[:, t]
        self._baseline_latent[:, t + 1] = baseline_next
        latent_next = baseline_next + self.g_profile[t + 1] * VSTRONG
        self.latent[:, t + 1] = latent_next
        self.proxy_all[:, t + 1] = latent_next @ _PARENT.PROXY_LOADING.T + self.measurement_noise[:, t + 1]
        self.original_reward[:, t] = x - 0.25 * x**2 - 0.50 * a**2 + TAU1 * a
        self.reward[:, t] = self.original_reward[:, t]
        self.reward_observation[:, t] = self.reward[:, t] + self.reward_noise[:, t]
        self._next_t = t + 1


def law(source: str, g: float = 0.0) -> dict[str, Any]:
    source = _validate_source(source)
    g = float(g)
    if not np.isfinite(g):
        raise ValueError("g must be finite")
    F = F0 + 0.69375 * g * (FSTAR - F0)
    c = 0.465 * g if source == "MODEL_POS" else (-0.465 * g if source in {"MODEL_A", "MODEL_LATENT"} else 0.0)
    return {"g": g, "c": c, "effective_matched_response": -.465*g if source == "LATENT_MATCHED_ACTION" else None, "F": F if source in {"LATENT_F", "MODEL_LATENT"} else F0.copy(),
            "reward_shift": g if source == "REWARD" else 0.0,
            "latent_mean_shift": g * np.array([0.04, -0.03, 0.02]) if source == "LATENT_MEAN" else np.zeros(3)}


__all__ = ["Environment", "SCENARIOS", "BOUNDARY_SCENARIOS", "PROFILE_NAMES", "ACTION_GRID", "A0", "F0", "FSTAR",
           "ETA", "LAMBDA", "C1", "V", "VSTRONG", "KAPPA", "law"]
