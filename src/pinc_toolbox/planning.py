"""Explicit-table finite and infinite discounted MDP solvers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

class MDPTableError(ValueError):
    """The supplied finite MDP table is malformed."""

@dataclass(frozen=True)
class QResult:
    q: np.ndarray
    policy: np.ndarray
    residual: float | None
    iterations: int
    status: str

@dataclass(frozen=True)
class VResult:
    values: np.ndarray
    q: np.ndarray
    policy: np.ndarray
    residual: float | None
    iterations: int
    status: str

def _validate(rewards: Any, transitions: Any, mask: Any, beta: float):
    r = np.asarray(rewards, dtype=float)
    p = np.asarray(transitions, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if r.ndim != 2:
        raise MDPTableError("rewards must have shape (states, actions)")
    n, actions = r.shape
    if p.shape != (n, actions, n):
        raise MDPTableError("transitions must have shape (states, actions, states)")
    if m.shape != (n, actions):
        raise MDPTableError("mask must match rewards")
    if not np.isfinite(beta) or not 0 <= float(beta) < 1:
        raise MDPTableError("beta must satisfy 0 <= beta < 1")
    if not m.any(axis=1).all():
        raise MDPTableError("every state needs a supported action")
    if not np.isfinite(r[m]).all():
        raise MDPTableError("supported rewards must be finite")
    if not np.isfinite(p[m]).all() or (p[m] < 0).any():
        raise MDPTableError("supported transitions must be finite and nonnegative")
    if not np.allclose(p[m].sum(axis=1), 1, rtol=0, atol=1e-12):
        raise MDPTableError("supported transition rows must sum to one")
    return r, p, m, float(beta)

def _residual(q, r, p, m, beta):
    values = np.max(np.where(m, q, -np.inf), axis=1)
    target = r + beta * np.einsum("san,n->sa", p, values)
    return float(np.max(np.abs((target - q)[m])))

def _policy(q, mask):
    out = np.full(mask.shape[0], -1, dtype=int)
    for state in range(len(out)):
        supported = np.flatnonzero(mask[state])
        out[state] = int(supported[np.argmax(q[state, supported])])
    return out

def solve_finite(rewards, transitions, mask, beta, horizon, terminal):
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise MDPTableError("horizon must be a positive integer")
    r, p, m, b = _validate(rewards, transitions, mask, beta)
    t = np.asarray(terminal, dtype=float).reshape(-1)
    if t.shape != (len(r),) or not np.isfinite(t).all():
        raise MDPTableError("terminal must be finite with one value per state")
    values = t.copy()
    q = np.full(r.shape, -np.inf)
    for _ in range(int(horizon)):
        q = r + b * np.einsum("san,n->sa", p, values)
        if not np.isfinite(q[m]).all():
            raise MDPTableError("finite-horizon supported Q is nonfinite")
        q = np.where(m, q, -np.inf)
        values = np.max(q, axis=1)
    return QResult(q, _policy(q, m), None, int(horizon), "finite_terminal_recursion")

def solve_bellman(rewards, transitions, mask, beta, *, tolerance=1e-10, max_iterations=100000):
    r, p, m, b = _validate(rewards, transitions, mask, beta)
    if not np.isfinite(tolerance) or tolerance < 0 or int(max_iterations) < 1:
        raise MDPTableError("invalid tolerance or max_iterations")
    q = np.zeros(r.shape, dtype=float)
    q[~m] = -np.inf
    residual = _residual(q, r, p, m, b)
    iterations = 0
    while iterations < int(max_iterations) and residual > tolerance:
        values = np.max(np.where(m, q, -np.inf), axis=1)
        updated = r + b * np.einsum("san,n->sa", p, values)
        if not np.isfinite(updated[m]).all():
            raise MDPTableError("infinite-horizon supported Q is nonfinite")
        q = np.where(m, updated, -np.inf)
        iterations += 1
        residual = _residual(q, r, p, m, b)
    status = "converged" if residual <= tolerance else "nonconverged"
    return QResult(q, _policy(q, m), residual, iterations, status)

def solve_value_iteration(rewards, transitions, mask, beta, *, tolerance=1e-10, max_iterations=100000):
    """Solve a discounted explicit MDP by iterating the state-value Bellman map.

    This is an alternate numerical representation of the same optimal-control
    problem solved by solve_bellman; it does not add information or identify a
    latent or mechanism source. Rewards, transitions, support and discount must
    therefore be identical for a meaningful solver comparison.
    """
    r, p, m, b = _validate(rewards, transitions, mask, beta)
    if not np.isfinite(tolerance) or tolerance < 0 or int(max_iterations) < 1:
        raise MDPTableError("invalid tolerance or max_iterations")
    values = np.zeros(r.shape[0], dtype=float)
    residual = float("inf")
    iterations = 0
    q = np.full(r.shape, -np.inf, dtype=float)
    while iterations < int(max_iterations):
        target_q = r + b * np.einsum("san,n->sa", p, values)
        if not np.isfinite(target_q[m]).all():
            raise MDPTableError("supported value-iteration Q is nonfinite")
        target_q = np.where(m, target_q, -np.inf)
        updated_values = np.max(target_q, axis=1)
        residual = float(np.max(np.abs(updated_values - values)))
        values = updated_values
        q = target_q
        iterations += 1
        if residual <= tolerance:
            break
    final_q = r + b * np.einsum("san,n->sa", p, values)
    if not np.isfinite(final_q[m]).all():
        raise MDPTableError("supported value-iteration Q is nonfinite")
    q = np.where(m, final_q, -np.inf)
    residual = float(np.max(np.abs(np.max(q, axis=1) - values)))
    status = "converged" if residual <= tolerance else "nonconverged"
    return VResult(values, q, _policy(q, m), residual, iterations, status)
