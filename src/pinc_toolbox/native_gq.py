"""Native RND2 GQ extraction, integrated from the reviewed 2026-09-22 candidate.

The public candidate is RND2GQ.fit/update/q_values/act.  It preserves the
current GQ_main RND2 path; it is neither a general economic-model adapter nor
an autonomous RA-GQ workflow.  It reads no repository file, loads no generator,
and imports only the standard library, NumPy, pandas, scikit-learn and the
installed pinc_toolbox G estimator.  See README.md for scope and provenance.
The source candidate was compared against the native host before integration;
installed-package execution and fresh-environment checks remain separate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .g import IdentificationError, SimpleGQ, SupportError
from .g_estimators import EstimatorConfig, fit_joint_ml

GRID = np.linspace(0.0, 0.10, 11)
NODES_PER_MEMORY = 5
XCOL = ["state", "w", "previous_action"]
MEMORY_STATE = tuple(XCOL)
AUX = {"base": [], "gamma": ["delta_gamma"], "epsilon": ["residual"],
       "full": ["delta_gamma", "delta_m", "residual"]}
# Basis order used by Rmat: [1, x, x^2, a, a^2]. RND2's reward law is
# prescribed; released reward labels are evaluator records, not learner input.
RND2_REWARD_COEFFICIENTS = (0.0, 1.0, -0.25, -2.025, -0.5)


# Selected native infinite_q helpers; solve_td/numba are not needed.

class MDPTableError(ValueError):
    """Raised when the supported finite-MDP table is malformed."""

@dataclass(frozen=True)
class QResult:
    """Common result fields returned by both solvers."""

    q: np.ndarray
    policy: np.ndarray
    residual: float
    iterations: int
    sweeps: int
    elapsed_seconds: float
    status: str
    visits: np.ndarray | None = None

def _validate(
    rewards: Any, transitions: Any, mask: Any, beta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    r = np.asarray(rewards, dtype=float)
    p = np.asarray(transitions, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if r.ndim != 2:
        raise MDPTableError("rewards must have shape (n_states, n_actions)")
    n, a = r.shape
    if p.shape != (n, a, n):
        raise MDPTableError("transitions must have shape (n_states, n_actions, n_states)")
    if m.shape != (n, a):
        raise MDPTableError("mask must have the same first two dimensions as rewards")
    if not np.isfinite(beta) or not 0 <= float(beta) < 1:
        raise MDPTableError("beta must be finite and satisfy 0 <= beta < 1")
    if not m.any(axis=1).all():
        raise MDPTableError("every state must have at least one supported action")
    supported_rewards = r[m]
    supported_transitions = p[m]
    if not np.isfinite(supported_rewards).all():
        raise MDPTableError("supported rewards must be finite")
    if not np.isfinite(supported_transitions).all():
        raise MDPTableError("supported transition rows must be finite")
    if (supported_transitions < 0).any():
        raise MDPTableError("supported transition probabilities must be nonnegative")
    row_sums = supported_transitions.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-12):
        raise MDPTableError("supported transition rows must sum to one")
    pairs = np.argwhere(m)
    return r, p, m, float(beta), pairs

def _residual(q: np.ndarray, r: np.ndarray, p: np.ndarray, mask: np.ndarray, beta: float) -> float:
    v = np.max(np.where(mask, q, -np.inf), axis=1)
    target = r + beta * np.einsum("san,n->sa", p, v)
    return float(np.max(np.abs(target[mask] - q[mask])))

def _policy(q: np.ndarray, mask: np.ndarray) -> np.ndarray:
    policy = np.full(mask.shape[0], -1, dtype=int)
    for state in range(mask.shape[0]):
        policy[state] = int(np.flatnonzero(mask[state])[np.argmax(q[state, mask[state]])])
    return policy

def solve_bellman(
    rewards: Any,
    transitions: Any,
    mask: Any,
    beta: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100_000,
    tol: float | None = None,
    max_iter: int | None = None,
    initial: Any | None = None,
) -> QResult:
    """Solve the supported MDP by synchronous full Bellman iteration.

    Convergence is decided only from the independently recomputed full
    Bellman residual of the returned Q table.  ``max_iterations`` is a hard
    budget; a budget exhaustion returns ``status='nonconverged'``.
    """
    r, p, m, b, _ = _validate(rewards, transitions, mask, beta)
    if tol is not None:
        tolerance = float(tol)
    if max_iter is not None:
        max_iterations = int(max_iter)
    if not np.isfinite(tolerance) or tolerance < 0 or max_iterations < 1:
        raise MDPTableError("tolerance must be nonnegative and max_iterations positive")
    started = time.perf_counter()
    q = np.zeros(r.shape, dtype=float) if initial is None else np.asarray(initial, dtype=float).copy()
    if q.shape != r.shape or not np.isfinite(q[m]).all():
        raise MDPTableError("initial must match rewards and be finite on supported cells")
    q[~m] = -np.inf
    iterations = 0
    residual = _residual(q, r, p, m, b)
    while iterations < int(max_iterations) and residual > float(tolerance):
        v = np.max(np.where(m, q, -np.inf), axis=1)
        updated = r + b * np.einsum("san,n->sa", p, v)
        q[m] = updated[m]
        iterations += 1
        residual = _residual(q, r, p, m, b)
    status = "converged" if residual <= float(tolerance) else "nonconverged"
    q[~m] = -np.inf
    return QResult(q, _policy(q, m), residual, iterations, iterations,
                   time.perf_counter() - started, status, None)

# RND2 dictionaries and local support; G coefficients use package SimpleGQ.

class Reference:
 def predict(self,x):return np.asarray(x)[:,2]

class Transition:
 def __init__(self,core,declared,active):
  self.core=core;self.declared=declared;self.active=active;self.reference=Reference()
 def context(self,x,e=None):
  # Compatibility e argument is intentionally opaque to physical G.
  x=np.asarray(x,float)
  if x.ndim!=2 or x.shape[1]<3:raise ValueError('physical context requires state,w,previous_action')
  return x[:,:3]
 def predict(self,x,a,e=None):
  a=np.asarray(a,float).reshape(-1)
  return np.column_stack((self.core.predict_physical(self.context(x,e),a),a))
 def response(self,x,a,e=None):
  y=self.core.response(self.context(x,e),np.asarray(a,float).reshape(-1))
  # The full controlled contrast includes the exact bookkeeping memory response.
  return np.column_stack((y,np.asarray(a)-np.asarray(x)[:,2]))

class Model:pass

def _fit_model(frame,reference_mode='previous_action',*,mode='GQ',auxiliary_state_cols=(),seed=0,validation=None,estimator_config=None,g_spec='legacy',outer_config=None,policy_columns=()):
 if reference_mode!='previous_action':raise ValueError('Simple mainline requires previous_action')
 if validation is None:validation=frame.iloc[:0].copy()
 if set(frame.path_id)&set(validation.path_id):raise ValueError('training/validation path overlap')
 cols=tuple(auxiliary_state_cols);active=()  # E is downstream only, even when missing/nonfinite
 config=EstimatorConfig() if estimator_config is None else estimator_config
 if not isinstance(config,EstimatorConfig):raise TypeError('estimator_config must be an EstimatorConfig')
 if g_spec not in ('legacy','v012'):raise ValueError('g_spec must be legacy or v012')
 if g_spec=='v012':
  from .g_v012 import fit_v012_g, policy_schema
  if config.estimator!='linear' or config.pipeline!='planning_only':raise ValueError('v012 preserves linear FWL modules; use outer_estimator for ML')
  policies=policy_schema(policy_columns)
  if set(policies)&set(cols):raise ValueError('a declared auxiliary cannot also be a physical policy input')
  physical_names=('path_id','period','release_period',*MEMORY_STATE,*policies,'action','next_state','next_w')
  physical_frame=frame[[c for c in physical_names if c in frame]].copy()
  fitted,diagnostics=fit_v012_g(physical_frame,outer_config=outer_config,policy_columns=policies)
  diagnostics.update(active_auxiliaries=[],excluded_physical_auxiliaries=list(cols))
  m=Model();m.core=fitted;m._train_frame=physical_frame;m._validation_frame=validation[[c for c in physical_names if c in validation]].copy()
  m.state_cols_=(*MEMORY_STATE,*policies);m.diagnostics_=diagnostics
  # RND2 planning has no implicit macro successor law. G-only callers can
  # query core.predict_physical using the explicit conditioning schema.
  if policies:
   m.transition_=None;m.support_=None
  else:
   m.transition_=Transition(fitted,cols,());m.lower_q=.025;m.upper_q=.975
   m.support_=Support(frame[list(MEMORY_STATE)].to_numpy(float),frame.action.to_numpy(float),Reference(),m.lower_q,m.upper_q,30,seed)
  return m
 if policy_columns or outer_config is not None:raise ValueError('outer/macro settings require g_spec=v012')
 x=frame[list(MEMORY_STATE)].to_numpy(float);a=frame.action.to_numpy(float)
 y=frame[['next_state','next_w']].to_numpy(float)
 def B(x):
  s,w,p=x[:,0],x[:,1],x[:,2];out=[np.ones(len(x)),s,w,p,p*p,s*p,w*p]
  return np.column_stack(out)
 def H(x,a):
  s,w=x[:,0],x[:,1];out=[a,a*a,s*a,w*a]
  return np.column_stack(out)
 native_fit=SimpleGQ(B,H,memory_index=2,physical_indices=(0,1)).fit(x,a,y)
 if config.estimator=='linear':
  fitted=native_fit
  fit_diagnostics=dict(fitted.diagnostics_,active_auxiliaries=list(active),
                       excluded_physical_auxiliaries=list(cols),physical_inputs=list(MEMORY_STATE),
                       estimator='SimpleGQ_FWL')
 else:
  fitted,fit_diagnostics=fit_joint_ml(x,a,y,config=config,active_auxiliaries=active)
  fit_diagnostics.update(excluded_physical_auxiliaries=list(cols),physical_inputs=list(MEMORY_STATE))
  fit_diagnostics['native_admissibility']={
   'estimator':'SimpleGQ_FWL',
   'joint_rank':int(native_fit.diagnostics_['joint_rank']),
   'joint_columns':int(native_fit.diagnostics_['joint_columns']),
   'response_rank':int(native_fit.diagnostics_['rank']),
   'response_columns':int(native_fit.diagnostics_['response_columns']),
  }
 m=Model();m.core=fitted;m.transition_=Transition(fitted,cols,active);m._train_frame=frame.copy();m._validation_frame=validation.copy();m.state_cols_=MEMORY_STATE
 m.lower_q=.025;m.upper_q=.975;m.support_=Support(frame[list(MEMORY_STATE)].to_numpy(float),a,Reference(),m.lower_q,m.upper_q,30,seed)
 m.diagnostics_=fit_diagnostics
 return m

class Support:
    """Local observed-action support for a vector state."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        reference: Reference,
        lower_q: float,
        upper_q: float,
        neighbors: int,
        random_state: int,
        max_rows: int = 20_000,
    ) -> None:
        rng = np.random.default_rng(random_state)
        if len(states) > max_rows:
            index = np.sort(rng.choice(len(states), max_rows, replace=False))
            states, actions = states[index], actions[index]
        self.center = states.mean(axis=0)
        self.scale = np.maximum(states.std(axis=0, ddof=1), 1e-10)
        self.states = (states - self.center) / self.scale
        self.actions = np.asarray(actions, dtype=float)
        self.reference = reference
        self.lower_q = float(lower_q)
        self.upper_q = float(upper_q)
        self.neighbors = min(max(10, int(neighbors)), len(states))
        self.search = NearestNeighbors(n_neighbors=self.neighbors).fit(self.states)
        # The 99th percentile of leave-neighborhood distances distinguishes a
        # true extrapolation from a sparse but observed state region.
        probe_k = min(self.neighbors + 1, len(states))
        probe = NearestNeighbors(n_neighbors=probe_k).fit(self.states)
        distances, _ = probe.kneighbors(self.states)
        self.distance_limit = float(np.quantile(distances[:, -1], 0.99) * 1.5 + 1e-12)

    def standardize(self, states: np.ndarray) -> np.ndarray:
        return (states - self.center) / self.scale

    def mask(self, states: np.ndarray, action_grid: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        action_grid = np.asarray(action_grid, dtype=float).reshape(-1)
        distances, indices = self.search.kneighbors(self.standardize(states))
        local = self.actions[indices]
        lower = np.quantile(local, self.lower_q, axis=1)
        upper = np.quantile(local, self.upper_q, axis=1)
        inside = distances[:, -1] <= self.distance_limit
        mask = (action_grid[None, :] >= lower[:, None]) & (
            action_grid[None, :] <= upper[:, None]
        )
        # Empty support is an economic restriction, not a missing grid value
        # to be replaced by an unsupported reference/midpoint action.
        return mask & inside[:, None]

# Only the host functions called by GQ_main.GQ are extracted.

def _fit(frame: pd.DataFrame, cols, seed: int, estimator_config=None, *, g_spec='legacy', outer_config=None):
    cols = tuple(cols)
    f = frame.copy()  # Physical sample never depends on auxiliary availability.
    if f.empty:
        raise ValueError(f"empty fit frame for auxiliary={cols}")
    return _fit_model(f, "previous_action", auxiliary_state_cols=cols,
                        seed=int(seed), validation=f.iloc[:0].copy(),
                        estimator_config=estimator_config,g_spec=g_spec,outer_config=outer_config)

def _select_nodes(frame: pd.DataFrame, old, new, public_mask: np.ndarray,
                  cols, nodes_per_memory: int | None = None,
                  allow_missing_infeasible_memory: bool = False, *, grid=GRID) -> pd.DataFrame:
    nodes_per_memory = NODES_PER_MEMORY if nodes_per_memory is None else int(nodes_per_memory)
    cols = list(cols)
    need = XCOL + cols
    f = frame.dropna(subset=need).copy()
    if f.empty:
        raise ValueError("no finite actual joint records")
    x = f[XCOL].to_numpy(float)
    m0 = old.support_.mask(x, grid)
    m1 = new.support_.mask(x, grid)
    public_mask = np.asarray(public_mask, bool).reshape(-1)
    if public_mask.shape != (len(grid),):
        raise ValueError("public action mask has wrong width")
    eligible = (m0 & m1 & public_mask[None, :]).any(axis=1)
    pool = f.loc[eligible].copy()
    chosen = []
    for memory_index, memory in enumerate(grid):
        sub = pool[np.isclose(pool.previous_action, memory, rtol=0, atol=1e-12)].sort_values(
            ["state", "w", "period", "path_id"], kind="mergesort"
        )
        if sub.empty:
            # A banned future action cannot be successor memory. Retain any
            # observed historical group, but do not manufacture a missing
            # banned group when this option is explicitly requested.
            if allow_missing_infeasible_memory and not public_mask[memory_index]:
                continue
            raise ValueError(f"no common actual node at previous_action={memory:g}")
        count = min(int(nodes_per_memory), len(sub))
        ids = np.linspace(0, len(sub) - 1, count).astype(int)
        chosen.extend(sub.iloc[ids].index.tolist())
    nodes = f.loc[chosen].copy().drop_duplicates(subset=need, keep="first").reset_index(drop=True)
    required_groups = int(public_mask.sum()) if allow_missing_infeasible_memory else len(grid)
    if len(nodes) < required_groups:
        raise ValueError("duplicate selection removed too many actual joint nodes")
    return nodes

def _support(model, nodes: pd.DataFrame, public_mask: np.ndarray, *, grid=GRID) -> np.ndarray:
    m = model.support_.mask(nodes[XCOL].to_numpy(float), grid)
    return m & np.asarray(public_mask, bool)[None, :]

def _reward_matrix(coef: np.ndarray, nodes: pd.DataFrame) -> np.ndarray:
    return Rmat(np.asarray(coef, float), nodes[XCOL].to_numpy(float))

def reward_fit(_frame=None):
 """Return the prescribed RND2 reward coefficients (labels are ignored).

 Kept as a compatibility helper for old callers; the fitted learner never
 estimates reward from observed labels.
 """
 return np.asarray(RND2_REWARD_COEFFICIENTS, float).copy()


def _normalize_reward_coefficients(value=None):
    """Validate and freeze a prescribed RND2 reward vector in Rmat order."""
    coefficients = RND2_REWARD_COEFFICIENTS if value is None else value
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (5,) or not np.isfinite(coefficients).all():
        raise ValueError("reward_coefficients must be five finite values in [1, x, x^2, a, a^2] order")
    return tuple(float(item) for item in coefficients)

def Rmat(coef,X, *, grid=GRID):
 s=np.repeat(X[:,0],len(grid));a=np.tile(grid,len(X));B=np.c_[np.ones(len(s)),s,s*s,a,a*a]
 return (B@coef).reshape(len(X),len(grid))

def _transition_noise(model, cols, noise_nodes):
    f = model._train_frame
    tx = f[["state", "w", "previous_action"]].to_numpy(float)
    te = None  # Physical residual bank is independent of historical E.
    errors = f[["next_state", "next_w"]].to_numpy(float) - model.transition_.predict(
        tx, f.action.to_numpy(float), te)[:, :2]
    order = np.argsort(errors[:, 0], kind="stable")
    ids = order[np.linspace(0, len(order) - 1, int(noise_nodes)).astype(int)]
    return errors[ids]


def _transition_targets(model, base, X, E, action, noise, cols):
    """Build transition targets for source rows independently of destinations."""
    pred = model.transition_.predict(X, np.full(len(X), action))
    bp = base.transition_.predict(X, np.full(len(X), action))
    target = pred[:, None, :2] + noise[None, :, :]
    if model.diagnostics_.get('g_spec') == 'v012':
        from .g_v012 import select_residuals
        if cols:
            xx=np.repeat(X,len(noise),axis=0);aa=np.full(len(xx),action)
            en=select_residuals(model.core,xx,aa,target.reshape(-1,2),cols)
            target=np.concatenate((target,np.full((len(X),len(noise),1),action),
                                   en.reshape(len(X),len(noise),len(cols))),axis=2)
        else:
            target=np.concatenate((target,np.full((len(X),len(noise),1),action)),axis=2)
        return target,pred
    if cols:
        en = np.broadcast_to(E[:, None, :], (len(X), len(noise), E.shape[1])).copy()
        if "residual" in cols:
            en[:, :, cols.index("residual")] = target[:, :, 0] - bp[:, None, 0]
        target = np.concatenate((target, np.broadcast_to(action, (len(X), len(noise), 1)), en), axis=2)
    else:
        target = np.concatenate((target, np.broadcast_to(action, (len(X), len(noise), 1))), axis=2)
    return target, pred


def _projection_neighbors(target, destination_points, scale):
    """Native stable four-neighbor inverse-distance projection weights."""
    d = np.sqrt((((destination_points[None, None, :, :] - target[:, :, None, :]) /
                  scale[None, None, None, :]) ** 2).sum(axis=3))
    near = np.argsort(d, axis=2, kind="stable")[:, :, :min(4, len(destination_points))]
    nearest = np.take_along_axis(d, near, axis=2)
    exact = nearest[:, :, 0] < 1e-12
    w = 1.0 / np.maximum(nearest, 1e-12)
    w /= w.sum(axis=2, keepdims=True)
    if exact.any():
        w[exact] = 0.0
        w[exact, 0] = 1.0
    return near, w, nearest


def kernel(model, base, nodes, cols, mask, scale, *, grid=None, noise_nodes=8):
    grid = np.linspace(0.0, 0.10, 11) if grid is None else np.asarray(grid, float)
    X = nodes[["state", "w", "previous_action"]].to_numpy(float)
    E = nodes[list(cols)].to_numpy(float) if cols else None
    points = np.c_[X, E] if cols else X
    noise = _transition_noise(model, cols, noise_nodes)
    n, k = len(X), len(noise)
    P = np.zeros((n, len(grid), n), float)
    maxdist = 0.0; raw_memory = 0.0
    for j, action in enumerate(grid):
        ii = np.flatnonzero(mask[:, j])
        dest = np.flatnonzero(np.isclose(X[:, 2], action, rtol=0, atol=1e-12))
        if len(ii) == 0: continue
        if len(dest) == 0: raise ValueError("missing exact memory successor nodes")
        target, pred = _transition_targets(model, base, X[ii],
                                           None if E is None else E[ii], action, noise, cols)
        raw_memory = max(raw_memory, float(np.max(np.abs(pred[:, 2] - action))))
        near, w, nearest = _projection_neighbors(target, points[dest], scale)
        maxdist = max(maxdist, float(nearest[:, :, 0].max()))
        rows = np.repeat(ii, k); cols_flat = near.reshape(-1, near.shape[-1])
        vals = (w / k).reshape(-1, w.shape[-1])
        np.add.at(P[:, j, :], (rows[:, None], dest[cols_flat]), vals)
    wrong_mass = float(max(np.sum(P[:, j, ~np.isclose(X[:, 2], action, rtol=0, atol=1e-12)])
                           for j, action in enumerate(grid)))
    if wrong_mass > 1e-12: raise ValueError("nonzero wrong-memory probability")
    memory_error = float(np.max(np.abs((P * X[None, None, :, 2]).sum(2)[mask] -
                                      np.broadcast_to(grid, (n, len(grid)))[mask])))
    if memory_error > 1e-12: raise ValueError("projected memory mismatch")
    return P, dict(wrong_memory_probability=wrong_mass,
                   projected_memory_error=memory_error, raw_memory_error=raw_memory,
                   max_projection_distance=maxdist, residual_nodes=int(noise_nodes))

def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)

def _date(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("as_of must be an integer decision date")
    return int(value)

def _finite_q(rewards, transitions, mask, beta, horizon):
    """Finite-table Bellman recursion, with the existing table validation.

    This is the recurrence used in the online-boundary experiment, with beta
    and horizon supplied explicitly instead of experiment-level constants.
    It introduces no Q regression or interpolation beyond the supplied P.
    """
    r, p, m, b, _ = _validate(rewards, transitions, mask, beta)
    values = np.zeros(len(r))
    by_horizon = {}
    for h in range(1, horizon + 1):
        q = r + b * np.einsum("san,n->sa", p, values)
        q = np.where(m, q, -np.inf)
        if not np.isfinite(q[m]).all():
            raise ValueError("nonfinite finite-horizon Q value")
        by_horizon[h] = q
        values = np.max(q, axis=1)
    return by_horizon

class RND2GQ:
    """Offline or explicitly refreshed GQ for the existing RND2 public panel.

    The default action grid has 11 points from 0 to 0.10. ``horizon``
    is None for infinite-horizon discounted Bellman iteration, or a positive
    integer for a finite objective with zero terminal value. Auxiliary inputs
    must already have been constructed from legally released observations.
    """

    def __init__(self, auxiliary="base", *, beta=.95, horizon=None,
                 nodes_per_memory=5, noise_nodes=8, estimator="linear",
                 estimator_params=None, pipeline="planning_only",g_spec="legacy",
                 outer_estimator="linear",outer_estimator_params=None,policy_columns=(),
                 reward_coefficients=None, reward_equation=None, reward_inputs=None,
                 action_grid=None):
        if g_spec not in ('legacy','v012'):
            raise ValueError('g_spec must be legacy or v012')
        if g_spec=='v012':
            from .g_v012 import AUX as layouts,policy_schema
            self.policy_columns_=policy_schema(policy_columns)
        else:
            layouts=AUX;self.policy_columns_=()
            if policy_columns or outer_estimator!='linear' or outer_estimator_params:
                raise ValueError('outer/macro settings require g_spec=v012')
        if auxiliary not in layouts:
            raise ValueError("auxiliary must be base, gamma, epsilon, or full")
        if not np.isfinite(beta) or not 0 <= float(beta) < 1:
            raise ValueError("beta must satisfy 0 <= beta < 1")
        self.auxiliary = auxiliary
        self.beta = float(beta)
        self.horizon = None if horizon is None else _positive_integer(horizon, "horizon")
        self.nodes_per_memory = _positive_integer(nodes_per_memory, "nodes_per_memory")
        self.noise_nodes = _positive_integer(noise_nodes, "noise_nodes")
        self.estimator_config_ = EstimatorConfig(
            estimator=estimator, pipeline=pipeline,
            estimator_params={} if estimator_params is None else estimator_params)
        self.g_spec=g_spec
        if g_spec=='v012' and (estimator!='linear' or pipeline!='planning_only'):
            raise ValueError('v012 preserves linear FWL modules; use outer_estimator for ML')
        self.outer_config_=(EstimatorConfig(estimator=outer_estimator,
            estimator_params={} if outer_estimator_params is None else outer_estimator_params)
            if g_spec=='v012' else None)
        self.auxiliary_columns_ = tuple(layouts[auxiliary])
        self.action_grid_ = np.asarray(GRID if action_grid is None else action_grid, float).copy()
        grid = self.action_grid_
        if grid.ndim != 1 or not grid.size or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 1e-12):
            raise ValueError("action_grid must be finite and strictly increasing with distinct actions")
        if reward_equation is not None and reward_coefficients is not None:
            raise ValueError("provide a reward equation or reward coefficients")
        self.reward_equation = deepcopy(reward_equation)
        self.reward_inputs = dict(reward_inputs or {})
        if reward_equation is not None and (
                set(self.reward_inputs) != set(reward_equation.variables)
                or not set(self.reward_inputs.values()) <= {"state", "w", "previous_action", "action"}):
            raise ValueError("reward inputs must map the declared variables to native state or action fields")
        self.prescribed_reward_coefficients = _normalize_reward_coefficients(reward_coefficients)

    def _prescribed_rewards(self, states, actions):
        columns = {name: states[:, i, None] for i, name in enumerate(XCOL)}
        columns["action"] = np.asarray(actions)[None, :]
        values = self.reward_equation({name: columns[source] for name, source in self.reward_inputs.items()})
        return np.broadcast_to(values, (len(states), len(actions)))

    # G-step: fit only complete transitions released before the decision.
    def fit(self, history, *, as_of, window=24, feasible=None):
        """Fit an offline snapshot using ``as_of-window <= period < as_of``.

        ``period`` dates the action; its physical successor is released by
        ``as_of``. If a ``release_period`` column is present, it is enforced.
        Reward is evaluated from the prescribed equation or coefficient vector;
        any observed reward column is ignored.
        ``feasible`` is the public action mask maintained throughout this
        planning problem; its default is the complete host action grid.
        Missing initial lag/auxiliary rows are omitted explicitly and counted.
        Fitting or solving failures leave a previously fitted snapshot intact.
        """
        as_of, window = _date(as_of), _positive_integer(window, "window")
        if self.policy_columns_:
            raise ValueError('optional macro inputs are G-only until an explicit macro continuation law is specified; no future policy is simulated')
        if not isinstance(history, pd.DataFrame):
            raise TypeError("history must be a pandas DataFrame")
        required = (list(XCOL) + ["action", "next_state", "next_w",
                    "period", "path_id"] + list(self.auxiliary_columns_))
        missing = set(required).difference(history.columns)
        if missing:
            raise ValueError(f"missing public history columns: {sorted(missing)}")
        periods = history.period.to_numpy(float)
        if not np.isfinite(periods).all() or not np.equal(periods, np.floor(periods)).all():
            raise ValueError("history.period must contain finite integer action dates")
        keep = (periods >= as_of - window) & (periods < as_of)
        selected = history.loc[keep].copy()
        if "release_period" in selected:
            release = selected.release_period.to_numpy(float)
            if (not np.isfinite(release).all()
                    or not np.equal(release, np.floor(release)).all()
                    or (release <= selected.period.to_numpy(float)).any()):
                raise ValueError("release_period must be an integer after its action date")
            selected = selected.loc[release <= as_of].copy()
        if selected.duplicated(["path_id", "period"]).any():
            raise ValueError("duplicate path/date observations")
        physical_required = [c for c in required if c not in self.auxiliary_columns_]
        physical_finite = np.isfinite(selected[physical_required].to_numpy(float)).all(axis=1)
        # Drop the outcome label before any learner preprocessing. In
        # particular, a missing/nonfinite label cannot alter G rows, support,
        # selected Q nodes, or the Bellman reward table.
        physical_frame = selected.loc[physical_finite].drop(columns=["reward"], errors="ignore").copy()
        joint_required = list(self.auxiliary_columns_)
        frame = (physical_frame.loc[np.isfinite(physical_frame[joint_required].to_numpy(float)).all(axis=1)].copy()
                 if joint_required else physical_frame.copy())
        if frame.empty:
            raise IdentificationError("no complete released rows in the requested window")
        if self.g_spec=='v012':
            from .g_v012 import validate_history
            # Validate the selected historical record, not only rows surviving
            # a dropna operation: a partial fixed group must never disappear.
            validate_history(physical_frame,self.auxiliary_columns_)
        ordered = physical_frame.sort_values(["path_id", "period"], kind="mergesort")
        lag_action = ordered.groupby("path_id", sort=False).action.shift(1)
        lag_date = ordered.groupby("path_id", sort=False).period.shift(1)
        consecutive = ordered.period.eq(lag_date + 1)
        if not np.allclose(ordered.loc[consecutive, "previous_action"],
                           lag_action.loc[consecutive], rtol=0, atol=1e-12):
            raise ValueError("previous_action must be the executed action at period-1")
        if self.auxiliary_columns_ and "residual" in self.auxiliary_columns_:
            if "residual_source_period" in frame:
                source = frame.residual_source_period.to_numpy(float)
                if not np.isfinite(source).all() or not np.array_equal(source, frame.period.to_numpy(float) - 1):
                    raise ValueError("residual must come from the previous completed transition")
            if "residual_release_period" in frame:
                released = frame.residual_release_period.to_numpy(float)
                if (not np.isfinite(released).all()
                        or not np.equal(released, np.floor(released)).all()
                        or (released > frame.period.to_numpy(float)).any()):
                    raise ValueError("residual must be released by the action date")
                if "residual_source_period" in frame:
                    source = frame.residual_source_period.to_numpy(float)
                    if (released <= source).any():
                        raise ValueError("residual release must follow its source transition")
        for name in ("gamma_source_period", "auxiliary_as_of"):
            if self.auxiliary_columns_ and name in frame:
                source = frame[name].to_numpy(float)
                if not np.isfinite(source).all() or (source > frame.period.to_numpy(float)).any():
                    raise ValueError(f"{name} contains unavailable auxiliary information")

        public_mask = (np.ones(len(self.action_grid_), dtype=bool) if feasible is None
                       else np.asarray(feasible, bool).copy())
        if public_mask.shape != (len(self.action_grid_),):
            raise ValueError("fit feasible must be a public mask matching the action grid")
        if not public_mask.any():
            raise SupportError("no publicly feasible action in the planning problem")
        if self.g_spec=='v012':
            model = _fit(physical_frame, (), 0, self.estimator_config_,g_spec=self.g_spec,outer_config=self.outer_config_)
        else:
            # Historical isolated backends replace _fit with this exact legacy
            # signature. Do not pass newly added keywords to their callbacks.
            model = _fit(physical_frame, (), 0, self.estimator_config_)
        base_config = (self.estimator_config_ if self.estimator_config_.pipeline == "full_ml"
                       else EstimatorConfig())
        if self.g_spec=='v012':
            base=model  # One physical G, shared by full/simple and the noise bank.
        elif not self.auxiliary_columns_ and base_config == self.estimator_config_:
            base = model
        else:
            base = _fit(physical_frame, (), 0, base_config)
        reward_coefficients = np.asarray(self.prescribed_reward_coefficients, float).copy()

        # Q-step: actual joint records and exact next-action-memory groups.
        grid_options = {} if np.array_equal(self.action_grid_, GRID) else {"grid": self.action_grid_}
        nodes = _select_nodes(frame, model, model, public_mask,
                                    self.auxiliary_columns_, self.nodes_per_memory,
                                    allow_missing_infeasible_memory=True, **grid_options)
        mask = _support(model, nodes, public_mask, **grid_options)
        point_columns = list(XCOL) + list(self.auxiliary_columns_)
        points = nodes[point_columns].to_numpy(float)
        scale = np.maximum(np.std(points, axis=0), 1e-10)
        transitions, kernel_diagnostics = kernel(
            model, base, nodes, list(self.auxiliary_columns_), mask, scale,
            grid=self.action_grid_, noise_nodes=self.noise_nodes)
        if self.reward_equation is not None:
            rewards = self._prescribed_rewards(nodes[XCOL].to_numpy(float), self.action_grid_)
        elif grid_options:
            rewards = Rmat(reward_coefficients, nodes[XCOL].to_numpy(float), **grid_options)
        else:
            rewards = _reward_matrix(reward_coefficients, nodes)
        if self.horizon is None:
            result = solve_bellman(rewards, transitions, mask, self.beta, tolerance=1e-8)
            if result.status != "converged":
                raise ValueError("Bellman solver did not converge")
            q, by_horizon = result.q, None
            solver_diagnostics = {"solver": "infinite_bellman", "bellman_residual": result.residual,
                                  "iterations": result.iterations}
        else:
            by_horizon = _finite_q(rewards, transitions, mask, self.beta, self.horizon)
            q = by_horizon[self.horizon]
            solver_diagnostics = {"solver": "finite_bellman", "horizon": self.horizon,
                                  "terminal_value": 0.0}

        self.model_, self.base_model_ = model, base
        self.nodes_, self.mask_ = nodes, mask
        self.transitions_, self.rewards_, self.q_ = transitions, rewards, q
        self.q_by_horizon_ = by_horizon
        self.reward_coefficients_ = reward_coefficients if self.reward_equation is None else None
        self.planning_feasible_ = public_mask
        self.as_of_, self.window_ = as_of, window
        self._points, self._scale = points, scale
        self.diagnostics_ = {
            "as_of": as_of, "window": window, "training_first_period": int(frame.period.min()),
            "training_last_period": int(frame.period.max()), "training_rows": len(frame),
            "omitted_incomplete_rows": len(selected) - len(frame), "node_count": len(nodes),
            "planning_feasible": public_mask.tolist(),
            "auxiliary_columns": self.auxiliary_columns_, "G": model.diagnostics_,
            "physical_training_rows": len(physical_frame), "reward_training_rows": 0,
            "reward_source": ("canonical_RND2_prescription" if self.prescribed_reward_coefficients == RND2_REWARD_COEFFICIENTS
                              else "prescribed_configuration_independent_of_observed_labels"),
            "prescribed_reward_coefficients": list(self.prescribed_reward_coefficients),
            "reward_basis": ["1", "x", "x_squared", "action", "action_squared"],
            "physical_auxiliary_rule": "excluded_before_fit_preprocessing_selection_and_prediction",
            "auxiliary_continuation": "hold_delta_gamma_delta_m; residual=successor-minus-measurement_forecast",
            "estimator_config": self.estimator_config_.as_dict(),
            "base_measurement_estimator": base.diagnostics_.get("estimator", "SimpleGQ_FWL"),
            **kernel_diagnostics, **solver_diagnostics,
        }
        if self.g_spec=='v012':
            self.diagnostics_.update(g_spec=self.g_spec,outer_config=self.outer_config_.as_dict(),
                auxiliary_continuation='all selected raw own-regression residuals evaluated from one generated tuple and current frozen maps',
                residual_metric='existing coordinatewise std distance; coincident blocks retained separately',
                block_widths=([2,2,2] if self.auxiliary=='full' else [2] if self.auxiliary=='gamma' else []))
        if self.reward_equation is not None:
            self.diagnostics_.update(reward_source="prescribed_equation",
                                     reward_equation=self.reward_equation.expression,
                                     reward_inputs=self.reward_inputs.copy(),
                                     reward_parameters=dict(self.reward_equation.parameters))
            del self.diagnostics_["prescribed_reward_coefficients"], self.diagnostics_["reward_basis"]
        return self

    # Rolling: explicitly rebuild G, reward, nodes, kernel and Q from a new window.
    def update(self, history, *, as_of, window=24, feasible=None):
        """Refresh an existing snapshot; the caller chooses the update cadence.

        This is released-window re-estimation, not detector-triggered repair.
        It does not generate auxiliary records or collect policy-induced data.
        The previous public planning mask is retained unless a new one is given.
        """
        self._require_fitted()
        if _date(as_of) < self.as_of_:
            raise ValueError("a rolling update cannot move the decision date backward")
        public_mask = self.planning_feasible_ if feasible is None else feasible
        return self.fit(history, as_of=as_of, window=window, feasible=public_mask)

    def _require_fitted(self):
        if not hasattr(self, "q_"):
            raise ValueError("fit must be called first")

    def q_values(self, states, auxiliary=None, *, feasible=None, remaining_horizon=None,
                 evaluation="nearest"):
        """Query Q using legacy nearest-node lookup or an explicit Bellman step.

        ``evaluation="nearest"`` preserves the historical lookup and remains
        the default. ``evaluation="bellman"`` evaluates the query's reward and
        projects its transition into the fixed fitted destination nodes, using
        saved H-1 continuation values (or the saved infinite-horizon Q table).
        """
        if evaluation not in ("nearest", "bellman"):
            raise ValueError("evaluation must be 'nearest' or 'bellman'")
        return self._query_values(states, auxiliary, feasible=feasible,
                                  remaining_horizon=remaining_horizon,
                                  evaluation=evaluation)

    def explain_q_values(self, states, auxiliary=None, *, feasible=None, remaining_horizon=None):
        """Return Q and read-only support/nearest-node details for each query.

        Inputs and validation match :meth:`q_values`. ``q_values`` is the same
        ndarray (unsupported cells are -inf); ``rows`` contains per-action
        physical/public masks, node support, table finiteness and blockers.
        Missing-memory node fields are None. Finite Q is not an eta guarantee.
        This performs no fit, solve, support expansion or policy update.
        """
        return self._query_values(states, auxiliary, feasible=feasible,
                                  remaining_horizon=remaining_horizon, explain=True)

    def _query_values(self, states, auxiliary=None, *, feasible=None,
                      remaining_horizon=None, explain=False, evaluation="nearest"):
        self._require_fitted()
        x = np.asarray(states, float)
        if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
            raise ValueError("states must be finite with shape (n, 3)")
        if self.auxiliary_columns_:
            e = np.asarray(auxiliary, float)
            if e.shape != (len(x), len(self.auxiliary_columns_)) or not np.isfinite(e).all():
                raise ValueError("auxiliary must be finite and match auxiliary_columns_")
            query = np.column_stack((x, e))
        else:
            if auxiliary is not None:
                raise ValueError("base GQ has no auxiliary input")
            query = x
        table = self.q_
        h = None
        if remaining_horizon is not None:
            h = _positive_integer(remaining_horizon, "remaining_horizon")
            if evaluation == "nearest" or explain:
                if self.q_by_horizon_ is None or h not in self.q_by_horizon_:
                    raise ValueError("remaining_horizon requires an available finite-horizon Q table")
                table = self.q_by_horizon_[h]
            elif h > 1 and (self.q_by_horizon_ is None or h - 1 not in self.q_by_horizon_):
                raise ValueError("Bellman evaluation requires the saved H-1 Q table")
        support = self.model_.support_.mask(x, self.action_grid_)
        same_memory = np.isclose(x[:, 2, None], self.nodes_.previous_action.to_numpy()[None, :], rtol=0, atol=1e-12).any(axis=1)
        support &= same_memory[:, None]
        if explain:
            physical = support.copy()
            public = np.ones_like(support, dtype=bool)
        if feasible is not None:
            public = np.asarray(feasible, bool)
            if public.shape == (len(self.action_grid_),):
                public = np.broadcast_to(public, support.shape)
            if public.shape != support.shape:
                raise ValueError("feasible must match the action grid or the query-by-action shape")
            support &= public
        if evaluation == "bellman":
            if explain:
                raise ValueError("explain_q_values supports nearest-node evaluation only")
            support &= self.planning_feasible_[None, :]
            return self._direct_bellman_values(x, e if self.auxiliary_columns_ else None,
                                               support, horizon=self.horizon if h is None else h)
        out = np.full((len(x), len(self.action_grid_)), -np.inf)
        memory = self.nodes_.previous_action.to_numpy(float)
        rows = [] if explain else None
        for i in range(len(x)):
            candidates = np.flatnonzero(np.isclose(memory, x[i, 2], rtol=0, atol=1e-12))
            node = None
            nearest_distance = None
            if len(candidates):
                distance = np.linalg.norm((self._points[candidates] - query[i]) / self._scale, axis=1)
                node = int(candidates[int(np.argmin(distance))])
                allowed = support[i] & self.mask_[node] & np.isfinite(table[node])
                out[i, allowed] = table[node, allowed]
                if explain:
                    nearest_distance = float(distance[np.flatnonzero(candidates == node)[0]])
            if explain:
                actions = []
                for j, action in enumerate(self.action_grid_):
                    node_mask = None if node is None else bool(self.mask_[node, j])
                    finite = None if node is None else bool(np.isfinite(table[node, j]))
                    value = None if node is None else float(table[node, j])
                    blockers = []
                    if not physical[i, j]:
                        blockers.append("current_physical_support_refusal")
                    if not public[i, j]:
                        blockers.append("public_feasibility_refusal")
                    if node is None:
                        blockers.append("no_same_memory_node")
                    else:
                        if not node_mask:
                            blockers.append("nearest_node_action_support_refusal")
                        if not finite:
                            blockers.append("nearest_node_q_nonfinite")
                    eligible = bool(np.isfinite(out[i, j]))
                    actions.append(dict(
                        action=float(action), current_physical_mask=bool(physical[i, j]),
                        feasible_mask=bool(public[i, j]), nearest_node_action_mask=node_mask,
                        table_finite=finite, table_value=value if finite else None,
                        table_nonfinite=None if value is None or finite else str(value),
                        eligible=eligible,
                        reason=("eligible" if eligible else "no_same_memory_node"
                                if node is None else blockers[0]),
                        all_blockers=blockers, q_value=float(out[i, j]) if eligible else None))
                rows.append(dict(query_index=i, same_memory_exists=bool(len(candidates)),
                                 same_memory_count=len(candidates), nearest_node_index=node,
                                 nearest_node_distance=nearest_distance, actions=actions))
        if explain:
            return dict(q_values=out, rows=rows, eta_status="not_supplied",
                        method="shared_native_query", support_geometry_reimplemented=False)
        return out

    def _direct_bellman_values(self, x, e, support, *, horizon):
        """Evaluate query-row Bellman values against the original fixed nodes."""
        if self.reward_equation is not None:
            rewards = self._prescribed_rewards(x, self.action_grid_)
        elif np.array_equal(self.action_grid_, GRID):
            rewards = Rmat(self.reward_coefficients_, x)
        else:
            rewards = Rmat(self.reward_coefficients_, x, grid=self.action_grid_)
        if not np.isfinite(rewards[support]).all():
            raise ValueError("prescribed reward is nonfinite at a supported query")
        out = np.full(support.shape, -np.inf)
        if horizon == 1:
            out[support] = rewards[support]
            return out

        if horizon is None:
            if self.horizon is not None:
                raise ValueError("infinite Bellman evaluation requires an infinite-horizon fitted table")
            continuation_table = self.q_
        else:
            if self.q_by_horizon_ is None or horizon - 1 not in self.q_by_horizon_:
                raise ValueError("Bellman evaluation requires the saved H-1 Q table")
            continuation_table = self.q_by_horizon_[horizon - 1]
        continuation = np.max(np.where(self.mask_, continuation_table, -np.inf), axis=1)
        noise = _transition_noise(self.model_, self.auxiliary_columns_, self.noise_nodes)
        X = self.nodes_[ ["state", "w", "previous_action"] ].to_numpy(float)
        E = self.nodes_[list(self.auxiliary_columns_)].to_numpy(float) if self.auxiliary_columns_ else None
        points = self._points
        base = self.base_model_
        for i in range(len(x)):
            for j, action in enumerate(self.action_grid_):
                if not support[i, j]:
                    continue
                dest = np.flatnonzero(np.isclose(X[:, 2], action, rtol=0, atol=1e-12))
                if len(dest) == 0:
                    raise ValueError("missing exact memory successor nodes for Bellman query")
                target, _ = _transition_targets(
                    self.model_, base, x[i:i + 1], None if e is None else e[i:i + 1],
                    float(action), noise, list(self.auxiliary_columns_))
                near, weights, _ = _projection_neighbors(target, points[dest], self._scale)
                row = np.zeros(len(X), float)
                np.add.at(row, dest[near.reshape(-1)], (weights / len(noise)).reshape(-1))
                used = row > 0
                if not np.isfinite(continuation[used]).all():
                    raise ValueError("nonfinite saved continuation at a projected destination")
                value = rewards[i, j] + self.beta * float(row[used] @ continuation[used])
                if not np.isfinite(value):
                    raise ValueError("direct Bellman query produced a nonfinite value")
                out[i, j] = value
        return out

    def act(self, states, auxiliary=None, *, feasible=None, remaining_horizon=None,
            evaluation="nearest"):
        """Return the first maximizing supported action, or NaN if unavailable."""
        q = self.q_values(states, auxiliary, feasible=feasible,
                          remaining_horizon=remaining_horizon, evaluation=evaluation)
        available = np.isfinite(q).any(axis=1)
        actions = np.full(len(q), np.nan)
        actions[available] = self.action_grid_[np.argmax(q[available], axis=1)]
        return actions

__all__ = ["RND2GQ", "IdentificationError", "SupportError"]
