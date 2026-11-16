"""Optional joint estimators for the native RND2 G-step.

The default linear estimator remains in :mod:`pinc_toolbox.native_gq` and is
the original unpenalized SimpleGQ/FWL fit.  This module imports scikit-learn
model classes only when a nonlinear estimator is selected.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping
import time
import warnings

import numpy as np


DEFAULTS = {
    "linear": {},
    "gradient_boosting": {
        "loss": "squared_error", "n_estimators": 50, "max_depth": 2,
        "learning_rate": .05, "min_samples_leaf": 32, "subsample": 1.0,
        "random_state": 0,
    },
    "extra_trees": {
        "n_estimators": 64, "max_depth": 5, "min_samples_leaf": 20,
        "max_features": 1.0, "random_state": 0, "n_jobs": 1,
    },
    "mlp": {
        "hidden_layer_sizes": (24, 12), "activation": "relu", "solver": "adam",
        "alpha": .01, "learning_rate_init": .003, "max_iter": 350,
        "tol": 1e-5, "random_state": 0, "early_stopping": False,
    },
}


@dataclass(frozen=True, init=False)
class EstimatorConfig:
    """Estimator choice, residual pipeline, and explicit model parameters."""

    estimator: str = "linear"
    pipeline: str = "planning_only"
    _estimator_params: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __init__(self, estimator="linear", pipeline="planning_only", estimator_params=None):
        params = {} if estimator_params is None else estimator_params
        if not isinstance(params, Mapping):
            raise TypeError("estimator_params must be a mapping")
        object.__setattr__(self, "estimator", estimator)
        object.__setattr__(self, "pipeline", pipeline)
        object.__setattr__(self, "_estimator_params", deepcopy(dict(params)))
        self.__post_init__()

    def __post_init__(self):
        if self.estimator not in DEFAULTS:
            raise ValueError(f"unsupported estimator {self.estimator!r}; choose {tuple(DEFAULTS)}")
        if self.pipeline not in ("planning_only", "full_ml"):
            raise ValueError("pipeline must be 'planning_only' or 'full_ml'")
        if self.estimator == "linear" and self._estimator_params:
            raise ValueError("linear is the exact native unpenalized estimator and accepts no overrides")
        unknown = set(self._estimator_params) - set(DEFAULTS[self.estimator])
        if unknown:
            raise ValueError(f"unknown {self.estimator} parameters: {sorted(unknown)}")
        if self.estimator == "mlp" and self._estimator_params.get("early_stopping", False):
            raise ValueError("MLP early_stopping requires a chronological group-respecting holdout")

    @property
    def estimator_params(self):
        """Return a detached copy of the caller-supplied parameter map."""
        return deepcopy(dict(self._estimator_params))

    @property
    def parameters(self):
        return deepcopy({**DEFAULTS[self.estimator], **dict(self._estimator_params)})

    def as_dict(self):
        return {
            "estimator": self.estimator,
            "pipeline": self.pipeline,
            "estimator_params": self.parameters,
            "defaults": deepcopy(DEFAULTS[self.estimator]),
        }


def native_joint_features(states: Any, actions: Any) -> np.ndarray:
    """Fixed physical/action dictionary; trailing legacy E columns are ignored."""
    x = np.asarray(states, dtype=float)
    a = np.asarray(actions, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[1] < 3 or len(x) != len(a):
        raise ValueError("states and actions must be row-aligned with at least three state columns")
    x = x[:, :3]  # Sanitize before finite checks, scaling, encoding or fitting.
    if not np.isfinite(x).all() or not np.isfinite(a).all():
        raise ValueError("states and actions must be finite")
    s, w, previous = x[:, 0], x[:, 1], x[:, 2]
    b = [s, w, previous, previous * previous, s * previous, w * previous]
    h = [a - previous, a * a - previous * previous,
         s * (a - previous), w * (a - previous)]
    return np.column_stack((*b, a, *h))


def _make_estimator(name: str, params: Mapping[str, Any]):
    """Build one estimator using the original Flexible G-Step recipe."""
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**params)
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(**params)
    if name == "mlp":
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        target = TransformedTargetRegressor(
            regressor=MLPRegressor(**params), transformer=StandardScaler())
        return make_pipeline(StandardScaler(), target)
    raise ValueError(f"no nonlinear implementation for {name!r}")


class JointMLCore:
    """Joint conditional mean with a zero-at-anchor response by construction."""

    def __init__(self, estimator, *, separate_outputs, name, params, active_auxiliaries):
        self.estimator = estimator
        self.separate_outputs = bool(separate_outputs)
        self.estimator_name = name
        self.params = dict(params)
        self.active_auxiliaries = tuple(active_auxiliaries)

    def predict_physical(self, states, actions):
        z = native_joint_features(states, actions)
        if self.separate_outputs:
            return np.column_stack([model.predict(z) for model in self.estimator])
        prediction = np.asarray(self.estimator.predict(z), dtype=float)
        return prediction.reshape(len(z), 2)

    def reference_motion(self, states):
        x = np.asarray(states, dtype=float)
        return self.predict_physical(x, x[:, 2])

    def response(self, states, actions):
        x = np.asarray(states, dtype=float)
        return self.predict_physical(x, actions) - self.reference_motion(x)


def fit_joint_ml(states, actions, next_physical, *, config: EstimatorConfig,
                 active_auxiliaries=()):
    """Fit the selected nonlinear G and return its fit metadata."""
    x = np.asarray(states, dtype=float)
    if x.ndim != 2 or x.shape[1] < 3:
        raise ValueError("physical states require at least three columns")
    x = x[:, :3]
    active_auxiliaries = ()
    a = np.asarray(actions, dtype=float).reshape(-1)
    y = np.asarray(next_physical, dtype=float)
    if y.shape != (len(x), 2) or len(a) != len(x) or not len(x):
        raise ValueError("joint ML inputs must align with next_physical shape (n,2)")
    if not np.isfinite(x).all() or not np.isfinite(a).all() or not np.isfinite(y).all():
        raise ValueError("joint ML training data must be finite")
    z = native_joint_features(x, a)
    params = config.parameters
    separate_outputs = config.estimator == "gradient_boosting"
    model = ([_make_estimator(config.estimator, params) for _ in range(2)]
             if separate_outputs else _make_estimator(config.estimator, params))
    convergence = []
    started = time.perf_counter()
    if separate_outputs:
        for j, output_model in enumerate(model):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                output_model.fit(z, y[:, j])
            convergence.extend(str(w.message) for w in caught)
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(z, y)
        convergence.extend(str(w.message) for w in caught)
    elapsed = time.perf_counter() - started
    core = JointMLCore(model, separate_outputs=separate_outputs,
                       name=config.estimator, params=params,
                       active_auxiliaries=active_auxiliaries)
    fitted = core.predict_physical(x, a)
    diagnostics = {
        "estimator": config.estimator,
        "training_objective": ("joint_squared_error_raw_targets_f_of_Z_action"
                                if config.estimator != "mlp" else
                                "joint_squared_error_standardized_targets_f_of_Z_action"),
        "outputs": 2,
        "features": "fixed physical B, raw action and centered action dictionary",
        "physical_inputs": ["state", "w", "previous_action"],
        "feature_count": int(z.shape[1]),
        "active_auxiliaries": list(active_auxiliaries),
        "hyperparameters": config.parameters,
        "training_rows": len(x),
        "fit_seconds": float(elapsed),
        "fit_warnings": convergence,
        "training_rmse_by_output": np.sqrt(np.mean((fitted - y) ** 2, axis=0)).tolist(),
    }
    if config.estimator == "mlp":
        fitted_mlp = model.named_steps["transformedtargetregressor"].regressor_
        diagnostics["convergence"] = {
            "n_iter": int(fitted_mlp.n_iter_),
            "loss": float(fitted_mlp.loss_),
            "max_iter": int(params["max_iter"]),
            "hit_iteration_cap": bool(fitted_mlp.n_iter_ >= params["max_iter"]),
        }
    return core, diagnostics
