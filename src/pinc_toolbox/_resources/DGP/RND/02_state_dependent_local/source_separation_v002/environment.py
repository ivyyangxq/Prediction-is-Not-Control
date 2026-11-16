"""Temporal source-separation fixtures built on the RND2 ladder.

The mother process, initial draw, and physical innovation tapes are inherited
from ``auxiliary_regime_ladder_v002``.  A separate profile RNG and reward-noise
RNG keep source/profile variation from changing those tapes.  ``public`` is
the dated learner interface; latent arrays, full proxies, original rewards,
and profile arrays are evaluator objects.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np


_MOTHER_PATH = Path(__file__).resolve().parents[1] / "auxiliary_regime_ladder_v002" / "environment.py"
_SPEC = importlib.util.spec_from_file_location("source_separation_mother", _MOTHER_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load mother environment from {_MOTHER_PATH}")
_MOTHER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOTHER
_SPEC.loader.exec_module(_MOTHER)

SCENARIOS = ("STABLE", "MODEL_A", "LATENT_F", "REWARD", "MEASUREMENT", "EQUIV_RESPONSE", "EQUIV_LATENT")
ACTION_GRID = np.asarray(_MOTHER.GRID, dtype=float).copy()
A0 = float(_MOTHER.A0)
P0 = float(_MOTHER.P0)
ETA = np.asarray(_MOTHER.ETA, dtype=float).copy()
LAMBDA = np.asarray(_MOTHER.LAMBDA, dtype=float).copy()
F0 = np.asarray(_MOTHER.F0, dtype=float).copy()
FSTAR = np.array([[-.55, .08, 0.0], [.03, .85, .06], [0.0, .05, .30]], dtype=float)
C1 = np.asarray(_MOTHER.C1, dtype=float).copy()
PROXY_LOADING = np.asarray(_MOTHER.M.PROXY_LOADING, dtype=float).copy()
STATE_INNOVATION_SD = float(_MOTHER.SIGE)
LATENT_INNOVATION_SD = np.asarray(_MOTHER.SU, dtype=float).copy()
MEASUREMENT_NOISE_SD = float(_MOTHER.SIGN)
TAU1 = float(_MOTHER.M.TAU1)
ACTION_SLOPE = float(_MOTHER.M.ACTION_SLOPE)
ACTION_CURVATURE = float(_MOTHER.M.ACTION_CURVATURE)
STATE_ACTION_LOADING = float(_MOTHER.M.STATE_ACTION_LOADING)
V = 1.0e-3 * np.array([-19.0, 279.0, 368.0], dtype=float)
KAPPA = float(V @ LAMBDA)
V_F0 = V @ F0.T
PROFILE_NAMES = ("step", "volatile", "persistent", "ramp")
_PROFILE_SEED = 0x53535052
_REWARD_SEED = 0x52574432


def _validate_source(source: str) -> str:
    source = str(source).upper()
    if source not in SCENARIOS:
        raise ValueError(f"unknown source {source!r}; expected one of {SCENARIOS}")
    return source


def _profile_tape(seed: int, source: str, strength: float, profile: str,
                  horizon: int, switch_date: int) -> np.ndarray:
    """Return dated g[0:H+1]; draws are independent of mother innovations."""
    g = np.zeros(horizon + 1, dtype=float)
    if profile == "step":
        g[switch_date:] = strength
    elif profile == "ramp":
        g[switch_date:] = strength * np.minimum((np.arange(horizon + 1 - switch_date) + 1) / 8.0, 1.0)
    else:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), _PROFILE_SEED]))
        rho = 0.0 if profile == "volatile" else 0.85
        signs = np.empty(horizon + 1 - switch_date, dtype=float)
        signs[0] = 1.0 if rng.random() < .5 else -1.0
        for j in range(1, signs.size):
            signs[j] = signs[j - 1] if rng.random() < (1.0 + rho) / 2.0 else -signs[j - 1]
        g[switch_date:] = strength * (1.0 + 0.4 * signs)
    if not np.isfinite(g).all():
        raise ValueError("profile tape is non-finite")
    return g


def _validate_coefficients(g: np.ndarray) -> None:
    alpha = 0.69375 * g
    if np.any(alpha < 0.0) or np.any(alpha >= 1.0):
        raise ValueError("latent transition interpolation weight outside [0, 1)")
    F = F0[None, :, :] + alpha[:, None, None] * (FSTAR - F0)[None, :, :]
    if not np.isfinite(F).all() or np.max(np.abs(F)) > 1.0:
        raise ValueError("latent transition coefficients outside declared bounds")
    # F_t is a convex combination of F0 and FSTAR with a common contraction
    # bound: ||F0||_2 and ||FSTAR||_2 are both below one.
    if max(np.linalg.norm(f, 2) for f in F) >= 1.0:
        raise ValueError("latent transition lacks the common contraction bound")
    radii = [max(abs(np.linalg.eigvals(f))) for f in F]
    if not np.isfinite(radii).all() or max(radii) >= 1.0:
        raise ValueError("latent transition is not stable")


def _validate_actions(actions: np.ndarray | float, paths: int) -> np.ndarray:
    a = np.asarray(actions, dtype=float).reshape(-1)
    if a.size == 1 and paths != 1:
        a = np.repeat(a, paths)
    if a.size != paths or not np.isfinite(a).all():
        raise ValueError("actions must be one finite value per path")
    if np.any(a < ACTION_GRID[0]) or np.any(a > ACTION_GRID[-1]):
        raise ValueError("actions outside declared domain [0, .1]")
    return a


class Environment:
    """Paired finite-horizon source fixture with release-safe observations."""

    def __init__(self, seed: int, source: str, paths: int = 32, horizon: int = 80,
                 switch_date: int = 24, strength: float = 1.0,
                 profile: str = "step", reward_noise_sd: float = 0.0) -> None:
        self.seed = int(seed)
        self.source = _validate_source(source)
        self.paths = int(paths); self.horizon = int(horizon); self.switch_date = int(switch_date)
        self.strength = float(strength); self.profile = str(profile).lower()
        self.reward_noise_sd = float(reward_noise_sd)
        if self.paths < 1 or self.horizon < 1:
            raise ValueError("paths and horizon must be positive")
        if not 0 <= self.switch_date < self.horizon:
            raise ValueError("switch_date must be in [0, horizon)")
        if not any(np.isclose(self.strength, x) for x in (.5, 1.0)):
            raise ValueError("strength must be exactly 0.5 or 1.0")
        if self.profile not in PROFILE_NAMES:
            raise ValueError(f"unknown profile {profile!r}; expected {PROFILE_NAMES}")
        if not np.isfinite(self.reward_noise_sd) or self.reward_noise_sd < 0:
            raise ValueError("reward_noise_sd must be finite and nonnegative")

        mother = _MOTHER.Environment(self.seed, "STABLE", paths=self.paths,
                                     horizon=self.horizon, switch_date=self.switch_date)
        self.state_noise = np.array(mother.state_noise, copy=True)
        self.latent_noise = np.array(mother.latent_noise, copy=True)
        self.measurement_noise = np.array(mother.measurement_noise, copy=True)
        self.state = np.empty((self.paths, self.horizon + 1), dtype=float)
        self.latent = np.empty((self.paths, self.horizon + 1, 3), dtype=float)
        self.proxy_all = np.empty((self.paths, self.horizon + 1, 3), dtype=float)
        self.state[:, 0] = mother.state[:, 0]
        self._baseline_latent = np.empty_like(self.latent)
        self._baseline_latent[:, 0] = mother.latent[:, 0]
        self.g_profile = _profile_tape(self.seed, self.source, self.strength, self.profile,
                                       self.horizon, self.switch_date)
        self.g_t = self.g_profile
        _validate_coefficients(self.g_profile)
        if self.source == "EQUIV_LATENT":
            self.latent[:, 0] = self._baseline_latent[:, 0] + self.g_profile[0] * V
        else:
            self.latent[:, 0] = self._baseline_latent[:, 0]
        self.proxy_all[:, 0] = self.latent[:, 0] @ PROXY_LOADING.T + self.measurement_noise[:, 0] * (1.0 + 2.0 * self.g_profile[0] if self.source == "MEASUREMENT" else 1.0)
        self.w = self.proxy_all[:, :, 0]
        self.action = np.zeros((self.paths, self.horizon), dtype=float)
        self.reward = np.zeros((self.paths, self.horizon), dtype=float)
        self.original_reward = np.zeros((self.paths, self.horizon), dtype=float)
        self.reward_observation = np.full((self.paths, self.horizon), np.nan, dtype=float)
        reward_rng = np.random.default_rng(np.random.SeedSequence([self.seed, _REWARD_SEED]))
        self.reward_noise = reward_rng.normal(size=(self.paths, self.horizon)) * self.reward_noise_sd
        self._next_t = 0
        changed = self.source != "STABLE" and np.any(self.g_profile != 0)
        self.metadata: dict[str, Any] = {
            "dgp": "RND2-SOURCE-SEPARATION-002", "source": self.source,
            "profile": self.profile, "strength": self.strength, "horizon": self.horizon,
            "switch_date": self.switch_date, "first_affected_release": self.switch_date + 1 if changed else None,
            "learner_public": ("state", "w", "released_reward"),
            "evaluator_only": ("latent_truth", "all_proxies", "original_reward", "g_profile", "fork"),
            "profile_tape_shared_by_seed": True,
            "C1_v": float(C1 @ V), "centered_response_v": float((ETA + A0 * LAMBDA) @ V),
            "kappa": KAPPA,
        }

    def public(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        t = int(t)
        if not 0 <= t <= self._next_t:
            raise IndexError("requested unreleased public date")
        return self.state[:, t].copy(), self.w[:, t].copy()

    def public_extra(self, t: int) -> np.ndarray:
        t = int(t)
        if not 0 <= t <= self._next_t:
            raise IndexError("requested unreleased public date")
        return (self.state[:, t].copy(), self.w[:, t].copy(), self.proxy_all[:, t, 1].copy())

    def step(self, t: int, actions: np.ndarray | float) -> None:
        t = int(t)
        if t != self._next_t or t >= self.horizon:
            raise ValueError("nonsequential or out-of-horizon transition")
        a = _validate_actions(actions, self.paths)
        z0 = self.latent[:, t]
        x = self.state[:, t]
        self.action[:, t] = a
        base_state_next = (P0 * x + z0 @ ETA + ACTION_SLOPE * a - ACTION_CURVATURE * a**2
                           + STATE_ACTION_LOADING * x * a + a * (z0 @ LAMBDA) + self.state_noise[:, t])
        c_t = -1.2 * 0.3875 * self.g_profile[t] if self.source == "MODEL_A" else 0.0
        if self.source == "EQUIV_RESPONSE":
            base_state_next = base_state_next + KAPPA * self.g_profile[t] * (a - A0)
        zbase_next = self._baseline_latent[:, t] @ F0.T + self.latent_noise[:, t]
        self._baseline_latent[:, t + 1] = zbase_next
        if self.source == "LATENT_F":
            F_t = F0 + 0.69375 * self.g_profile[t] * (FSTAR - F0)
            z_next = z0 @ F_t.T + self.latent_noise[:, t]
        elif self.source == "EQUIV_LATENT":
            g_t, g_next = self.g_profile[t], self.g_profile[t + 1]
            z_next = self.latent[:, t] @ F0.T + self.latent_noise[:, t] + g_next * V - g_t * V_F0
        else:
            z_next = zbase_next
        if self.source == "EQUIV_LATENT":
            state_next = (P0 * x + self.latent[:, t] @ ETA + ACTION_SLOPE * a - ACTION_CURVATURE * a**2
                          + STATE_ACTION_LOADING * x * a + a * (self.latent[:, t] @ LAMBDA) + self.state_noise[:, t])
        else:
            state_next = base_state_next + c_t * (a - A0)
        self.state[:, t + 1] = state_next
        self.latent[:, t + 1] = z_next
        measurement_scale = 1.0 + 2.0 * self.g_profile[t] if self.source == "MEASUREMENT" else 1.0
        self.proxy_all[:, t + 1] = z_next @ PROXY_LOADING.T + self.measurement_noise[:, t + 1] * measurement_scale
        self.original_reward[:, t] = x - 0.25 * x**2 - 0.50 * a**2 + TAU1 * a
        self.reward[:, t] = self.original_reward[:, t]
        if self.source == "REWARD":
            self.reward[:, t] += self.g_profile[t] * (a - A0)
        self.reward_observation[:, t] = self.reward[:, t] + self.reward_noise[:, t]
        self._next_t = t + 1

    def released_reward(self, t: int) -> np.ndarray:
        t = int(t)
        if t < 1:
            raise IndexError("no reward is released at date zero")
        if t > self._next_t:
            raise IndexError("requested unreleased reward")
        return self.reward_observation[:, t - 1].copy()

    def fork(self) -> "Environment":
        return copy.deepcopy(self)

    @property
    def latent_truth(self) -> np.ndarray:
        return self.latent

    @property
    def all_proxies(self) -> np.ndarray:
        return self.proxy_all

    @property
    def profile_tape(self) -> np.ndarray:
        return self.g_profile


__all__ = ["Environment", "SCENARIOS", "PROFILE_NAMES", "ACTION_GRID", "A0", "F0", "FSTAR",
           "ETA", "LAMBDA", "C1", "V", "KAPPA", "law"]


def law(source: str, g: float = 0.0) -> dict[str, Any]:
    """Return the bounded per-transition coefficient plug-ins for audits."""
    source = _validate_source(source); g = float(g)
    if not np.isfinite(g):
        raise ValueError("g must be finite")
    F = F0 + 0.69375 * g * (FSTAR - F0)
    return {"g": g, "c": -1.2 * 0.3875 * g if source == "MODEL_A" else 0.0,
            "F": F if source == "LATENT_F" else F0.copy(),
            "reward_shift": g if source == "REWARD" else 0.0}
