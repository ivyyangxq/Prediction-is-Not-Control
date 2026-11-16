"""Independent unpenalized linear G estimation with a floating action anchor."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np

class IdentificationError(ValueError):
    """The declared finite-dimensional dictionaries are not identified."""
class SupportError(ValueError):
    """The declared action/state support is empty."""

def _finite(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a

def linear_physical_memory_features(states: Any, physical_indices: tuple[int, ...], memory_index: int) -> np.ndarray:
    x = _finite(states, "states", 2)
    return np.column_stack((np.ones(len(x)), x[:, physical_indices], x[:, memory_index]))

def scalar_action_features(states: Any, actions: Any) -> np.ndarray:
    x = _finite(states, "states", 2)
    a = _finite(actions, "actions").reshape(-1)
    if len(x) != len(a):
        raise ValueError("states and actions row counts must agree")
    return a[:, None]


def _column_scales(b: np.ndarray, mode: Any) -> np.ndarray:
    """Return the common training-column scales used by the linear solver."""
    if isinstance(mode, str):
        if mode == "max_abs":
            scale = np.max(np.abs(b), axis=0)
        elif mode == "standard_deviation":
            scale = np.ones(b.shape[1], dtype=float)
            if b.shape[1] > 1:
                scale[1:] = np.std(b[:, 1:], axis=0)
                scale[1:] = np.maximum(scale[1:], 1e-15)
        else:
            raise ValueError("column_scale must be max_abs, standard_deviation, or an array")
    else:
        scale = _finite(mode, "column_scale", 1).reshape(-1)
        if scale.shape[0] != b.shape[1]:
            raise ValueError("column_scale must have one entry per state column")
        if np.any(scale <= 0):
            raise ValueError("column_scale entries must be positive")
    if np.any(scale == 0) or not np.isfinite(scale).all():
        raise IdentificationError("dictionary columns must be nonzero")
    return scale


def _fit_linear_fwl(b: np.ndarray, d: np.ndarray, y: np.ndarray, *,
                    column_scale: Any = "max_abs", allow_empty_action: bool = False) -> dict[str, Any]:
    """Shared unpenalized solver for action and no-action linear models."""
    b = _finite(b, "state_features", 2)
    d = _finite(d, "action_features", 2)
    y = _finite(y, "next_physical", 2)
    if b.shape[0] != d.shape[0] or b.shape[0] != y.shape[0] or b.shape[1] == 0:
        raise ValueError("linear dictionaries and response must have compatible rows")
    if d.shape[1] == 0:
        if not allow_empty_action:
            raise IdentificationError("dictionaries must have columns")
        scale = _column_scales(b, column_scale)
        bn = b / scale
        fit = np.linalg.lstsq(bn, y, rcond=None)
        if fit[2] < b.shape[1]:
            raise IdentificationError("state dictionary is rank deficient")
        c = fit[0] / scale[:, None]
        if not np.isfinite(c).all():
            raise ValueError("nonfinite G estimates")
        return {
            "c": c, "psi": np.empty((0, y.shape[1])), "scale": scale,
            "rank": int(fit[2]), "joint_rank": int(fit[2]),
            "joint_columns": int(b.shape[1]), "response_columns": 0,
            "singular_values": [], "fitted": b @ c,
        }
    if b.shape[0] < b.shape[1] + d.shape[1]:
        raise IdentificationError("fewer observations than declared columns")
    bs = _column_scales(b, column_scale)
    ds = np.max(np.abs(d), axis=0)
    if np.any(ds == 0) or not np.isfinite(ds).all():
        raise IdentificationError("dictionary columns must be nonzero")
    bn, dn = b / bs, d / ds
    fit = np.linalg.lstsq(bn, np.column_stack((dn, y)), rcond=None)
    if fit[2] < bn.shape[1]:
        raise IdentificationError("state dictionary is rank deficient")
    joint = np.column_stack((bn, dn))
    jf = np.linalg.lstsq(joint, np.zeros((len(b), 1)), rcond=None)
    if jf[2] < joint.shape[1]:
        raise IdentificationError("[B,d] is rank deficient")
    dp = dn - bn @ fit[0][:, :dn.shape[1]]
    yp = y - bn @ fit[0][:, dn.shape[1]:]
    rf = np.linalg.lstsq(dp, yp, rcond=None)
    if rf[2] < dp.shape[1]:
        raise IdentificationError("D_perp is rank deficient")
    psi_s = rf[0]
    c_s = np.linalg.lstsq(bn, y - dn @ psi_s, rcond=None)[0]
    psi, c = psi_s / ds[:, None], c_s / bs[:, None]
    if not np.isfinite(psi).all() or not np.isfinite(c).all():
        raise ValueError("nonfinite G estimates")
    return {
        "c": c, "psi": psi, "scale": bs, "rank": int(rf[2]),
        "joint_rank": int(jf[2]), "joint_columns": int(joint.shape[1]),
        "response_columns": int(dp.shape[1]),
        "singular_values": rf[3].tolist(), "fitted": b @ c + d @ psi,
    }

@dataclass
class SimpleGQ:
    state_features: Callable[[np.ndarray], Any]
    action_features: Callable[[np.ndarray, np.ndarray], Any]
    memory_index: int = -1
    physical_indices: tuple[int, ...] | None = None

    def fit(self, states: Any, actions: Any, next_physical: Any) -> "SimpleGQ":
        x = _finite(states, "states", 2)
        a = _finite(actions, "actions").reshape(-1)
        y = _finite(next_physical, "next_physical", 2)
        if len(x) < 1 or len(x) != len(a) or len(x) != len(y):
            raise ValueError("states, actions, and next_physical rows must agree and be positive")
        j = self.memory_index if self.memory_index >= 0 else x.shape[1] + self.memory_index
        if j < 0 or j >= x.shape[1]:
            raise ValueError("memory_index is outside states")
        phys = tuple(i for i in range(x.shape[1]) if i != j) if self.physical_indices is None else tuple(self.physical_indices)
        if (
            not phys
            or len(set(phys)) != len(phys)
            or any(
                not isinstance(i, (int, np.integer))
                or i < 0
                or i >= x.shape[1]
                or i == j
                for i in phys
            )
        ):
            raise ValueError("physical_indices must identify distinct non-memory coordinates")
        if y.shape[1] != len(phys):
            raise ValueError("next_physical width must match physical_indices")
        b = _finite(self.state_features(x), "state_features", 2)
        d0 = _finite(self.action_features(x, a), "action_features", 2)
        da = _finite(self.action_features(x, x[:, j]), "anchor action_features", 2)
        if b.shape[0] != len(x) or d0.shape != da.shape or d0.shape[0] != len(x):
            raise ValueError("feature functions returned incompatible shapes")
        d = d0 - da
        if not np.isfinite(d).all():
            raise ValueError("nonfinite action contrast")
        solved = _fit_linear_fwl(b, d, y, column_scale="max_abs")
        c, psi = solved["c"], solved["psi"]
        self._memory_index = j
        self._physical_indices = phys
        self._auxiliary_indices = tuple(
            i for i in range(x.shape[1]) if i != j and i not in phys
        )
        self._C = c
        self._Psi = psi
        self._state_width = x.shape[1]
        self._b_columns = b.shape[1]
        self._d_columns = d.shape[1]
        self.diagnostics_ = {
            "n": int(len(x)),
            "b_columns": int(b.shape[1]),
            "action_columns": int(d.shape[1]),
            "columns": int(b.shape[1] + d.shape[1]),
            "rank": solved["rank"],
            "joint_rank": solved["joint_rank"],
            "joint_columns": solved["joint_columns"],
            "response_columns": solved["response_columns"],
            "singular_values_d_perp": solved["singular_values"],
            "memory_index": int(j),
            "physical_dimension": int(y.shape[1]),
            "physical_indices": list(phys),
            "auxiliary_indices": list(self._auxiliary_indices),
            "method": "same_sample_fwl_column_normalized_lstsq",
            "penalty": "none",
        }
        return self
    def fit_released_window(self, states, actions, next_physical, release_dates, as_of, window):
        if isinstance(window,bool) or not isinstance(window,(int,np.integer)) or window<=0: raise ValueError("window must be a positive integer")
        dates = _finite(release_dates, "release_dates").reshape(-1)
        as_num = float(as_of)
        if not np.equal(dates,np.floor(dates)).all() or not np.isfinite(as_num) or as_num!=np.floor(as_num): raise ValueError("release_dates/as_of must be finite integers")
        x,a,y=np.asarray(states),np.asarray(actions),np.asarray(next_physical)
        if x.ndim!=2 or a.ndim!=1 or y.ndim!=2 or len(x)!=len(dates) or len(x)!=len(a) or len(x)!=len(y): raise ValueError("release_dates row count must match data")
        keep=(dates>as_num-int(window))&(dates<=as_num)
        if not keep.any(): raise IdentificationError("released window contains no rows")
        self.fit(x[keep], a[keep], y[keep])
        self.diagnostics_.update(
            {
                "release_window_count": int(keep.sum()),
                "release_latest": int(dates[keep].max()),
                "release_as_of": int(as_num),
                "release_window": int(window),
            }
        )
        return self
    def _states(self, states):
        if not hasattr(self,"_C"): raise ValueError("fit must be called first")
        x=_finite(states,"states",2)
        if x.shape[1]!=self._state_width: raise ValueError("states have the wrong width")
        return x
    def reference_motion(self, states):
        x = self._states(states)
        b = _finite(self.state_features(x), "state_features", 2)
        if b.shape!=(len(x),self._b_columns): raise ValueError("state_features returned wrong shape")
        out = b @ self._C
        if not np.isfinite(out).all():
            raise ValueError("nonfinite reference motion")
        return out
    def response(self, states, actions):
        x = self._states(states)
        a = _finite(actions, "actions").reshape(-1)
        if len(x)!=len(a): raise ValueError("states and actions row counts must agree")
        action = _finite(self.action_features(x, a), "action_features", 2)
        anchor = _finite(
            self.action_features(x, x[:, self._memory_index]),
            "anchor action_features",
            2,
        )
        d = action - anchor
        if not np.isfinite(d).all():
            raise ValueError("nonfinite action response contrast")
        if d.shape!=(len(x),self._d_columns): raise ValueError("action_features returned wrong shape")
        out = d @ self._Psi
        if not np.isfinite(out).all():
            raise ValueError("nonfinite action response")
        return out
    def predict_physical(self, states, actions):
        out = self.reference_motion(states) + self.response(states, actions)
        if not np.isfinite(out).all():
            raise ValueError("nonfinite physical prediction")
        return out
    def predict_next(self, states, actions, *, auxiliary_next=None):
        x = self._states(states)
        a = _finite(actions, "actions").reshape(-1)
        p = self.predict_physical(x, a)
        out = np.empty_like(x)
        out[:, self._physical_indices] = p
        out[:, self._memory_index] = a
        if self._auxiliary_indices:
            if auxiliary_next is None: raise ValueError("augmented prediction requires explicit next-auxiliary law")
            z=_finite(auxiliary_next,"auxiliary_next",2)
            if z.shape!=(len(x),len(self._auxiliary_indices)): raise ValueError("auxiliary_next has wrong shape")
            out[:,self._auxiliary_indices]=z
        elif auxiliary_next is not None: raise ValueError("model has no auxiliary coordinates")
        return out


@dataclass
class TransitionG:
    """Unpenalized state transition with no action, reward, or Q function."""
    state_features: Callable[[np.ndarray], Any]
    scaling: Any = "standard_deviation"
    ddof: int | None = None

    def fit(self, states: Any, next_physical: Any) -> "TransitionG":
        x = _finite(states, "states", 2)
        y = _finite(next_physical, "next_physical", 2)
        if len(x) == 0 or len(x) != len(y):
            raise ValueError("states and next_physical rows must agree and be positive")
        b = _finite(self.state_features(x), "state_features", 2)
        if b.shape[0] != len(x):
            raise ValueError("state_features returned incompatible rows")
        solved = _fit_linear_fwl(b, np.empty((len(x), 0)), y,
                                 column_scale=self.scaling,
                                 allow_empty_action=True)
        dof = b.shape[1] if self.ddof is None else self.ddof
        if isinstance(dof, bool) or not isinstance(dof, (int, np.integer)) or dof < 0:
            raise ValueError("ddof must be a nonnegative integer")
        if len(x) <= dof:
            raise ValueError("not enough observations for the requested covariance ddof")
        residuals = y - solved["fitted"]
        self._C = solved["c"]
        self.C_ = self._C
        self._state_width = x.shape[1]
        self._b_columns = b.shape[1]
        self.dof_ = int(len(x) - dof)
        self.residual_covariance_ = residuals.T @ residuals / self.dof_
        self.diagnostics_ = {
            "n": int(len(x)), "b_columns": int(b.shape[1]),
            "action_columns": 0, "columns": int(b.shape[1]),
            "rank": solved["rank"], "joint_rank": solved["joint_rank"],
            "joint_columns": solved["joint_columns"], "response_columns": 0,
            "singular_values_d_perp": [],
            "method": "state_transition_column_normalized_lstsq",
            "penalty": "none", "column_scale": solved["scale"].tolist(),
            "covariance_ddof": int(dof), "dof": self.dof_,
        }
        return self

    def _states(self, states: Any) -> np.ndarray:
        if not hasattr(self, "_C"):
            raise ValueError("fit must be called first")
        x = _finite(states, "states", 2)
        if x.shape[1] != self._state_width:
            raise ValueError("states have the wrong width")
        return x

    def predict(self, states: Any) -> np.ndarray:
        x = self._states(states)
        b = _finite(self.state_features(x), "state_features", 2)
        if b.shape != (len(x), self._b_columns):
            raise ValueError("state_features returned wrong shape")
        out = b @ self._C
        if not np.isfinite(out).all():
            raise ValueError("nonfinite transition prediction")
        return out

    def residuals(self, states: Any, observed: Any) -> np.ndarray:
        y = _finite(observed, "observed", 2)
        pred = self.predict(states)
        if y.shape != pred.shape:
            raise ValueError("observed has the wrong shape")
        return y - pred

    def act(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("TransitionG has no action support")

    def q_values(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("TransitionG has no Q function")
