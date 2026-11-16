"""Approved v012 modules, outer G and untransformed regression residuals.

The module estimators remain same-sample unpenalized FWL. Outer training uses
those fitted module outputs, deliberately without cross-fitting. Coincident
residuals retain separate sources and separate storage coordinates.
"""
from __future__ import annotations

from copy import deepcopy
import warnings

import numpy as np

from .g import SimpleGQ
from .g_estimators import EstimatorConfig, _make_estimator

SPEC = "v012"
OUTPUTS = ("state", "w")
FULL_COLUMNS = tuple(f"epsilon_{stage}_{name}" for stage in ("m0", "Gamma", "G")
                     for name in OUTPUTS)
GAMMA_COLUMNS = tuple(f"epsilon_Gamma_{name}" for name in OUTPUTS)
AUX = {"base": (), "full": FULL_COLUMNS, "gamma": GAMMA_COLUMNS}
CLOCK_COLUMNS = ("residual_source_period", "residual_release_period",
                 "residual_fit_as_of", "auxiliary_as_of", "residual_warmup")


def dictionaries(states, actions):
    """X=(state,w,optional policy); action memory stays in column 2."""
    x = np.asarray(states, float)
    a = np.asarray(actions, float).reshape(-1)
    if x.ndim != 2 or x.shape[1] < 3 or len(x) != len(a):
        raise ValueError("v012 requires row-aligned physical state and action")
    if not np.isfinite(x).all() or not np.isfinite(a).all():
        raise ValueError("v012 physical state and action must be finite")
    z = x[:, [0, 1, *range(3, x.shape[1])]]
    p = x[:, 2]
    d = a - p
    b = np.column_stack((np.ones(len(x)), z, p, p*p, p[:, None]*z))
    h = np.column_stack((d, p*d, d*d, d[:, None]*z))
    return b, h


def baseline_features(x):
    return dictionaries(x, np.asarray(x)[:, 2])[0]


def action_features(x, a):
    return dictionaries(x, a)[1]


def policy_schema(columns):
    if isinstance(columns, str):
        raise ValueError("policy_columns must be a sequence of separate column names")
    cols = tuple(columns)
    if (len(set(cols)) != len(cols) or any(not isinstance(c, str) or not c.strip() for c in cols)
            or set(cols) & {"state", "w", "previous_action", "action", "next_state", "next_w"}
            or any(any(token in c.lower() for token in
                ('epsilon','residual','auxiliary','delta_gamma','delta_m','lagged_e','lag_e')) for c in cols)):
        raise ValueError("policy columns must be distinct observable inputs, separate from w/action/auxiliary")
    return cols


class TwoModuleOuterCore:
    """Physical G fitted solely to the ordered (m0, Gamma) output blocks."""

    def module_outputs(self, states, actions):
        x = np.asarray(states, float)
        return (self.modules.reference_motion(x), self.modules.response(x, actions))

    def outer_features(self, states, actions):
        return np.column_stack(self.module_outputs(states, actions))

    def predict_physical(self, states, actions):
        z = self.outer_features(states, actions)
        if self.outer_config.estimator == "linear":
            out = np.column_stack((np.ones(len(z)), z)) @ self.outer_coefficients_
        elif self.separate_outputs:
            out = np.column_stack([m.predict(z) for m in self.outer_model_])
        else:
            out = np.asarray(self.outer_model_.predict(z), float).reshape(len(z), 2)
        if out.shape != (len(z), 2) or not np.isfinite(out).all():
            raise ValueError("v012 outer G produced an invalid physical prediction")
        return out

    def reference_motion(self, states):
        x = np.asarray(states, float)
        return self.predict_physical(x, x[:, 2])

    def response(self, states, actions):
        # Outer controlled contrast differs from the inner Gamma under ML.
        return self.predict_physical(states, actions) - self.reference_motion(states)

    def regression_residuals(self, states, actions, outcomes):
        x, a, y = np.asarray(states, float), np.asarray(actions, float), np.asarray(outcomes, float)
        if y.shape != (len(x), 2) or not np.isfinite(y).all():
            raise ValueError("v012 residuals require both original outcome coordinates")
        b, d = dictionaries(x, a)
        m0, gamma = self.module_outputs(x, a)
        # Each expression preserves the response of its own regression.
        epsilon_m0 = (y - gamma) - m0
        r_y = y - b @ self.projection_y_
        r_d = d - b @ self.projection_d_
        epsilon_gamma = r_y - r_d @ self.modules._Psi
        epsilon_g = y - self.predict_physical(x, a)
        return np.column_stack((epsilon_m0, epsilon_gamma, epsilon_g))


def fit_v012_g(frame, *, outer_config=None, policy_columns=()):
    """Fit physical modules and outer G; optional macro inputs condition G only.

    This G-only interface predicts next state/w. Declared common policy inputs
    are not auxiliary and must agree within each decision date. It does not
    invent or simulate a macro transition. RND2 Q rejects macro use until that
    separate continuation interface has been specified.
    """
    policies = policy_schema(policy_columns)
    columns = ("state", "w", "previous_action", *policies)
    if policies:
        if "period" not in frame or any(frame.groupby("period")[c].nunique(dropna=False).max() != 1 for c in policies):
            raise ValueError("common policy inputs must agree across individuals at each date")
    config = EstimatorConfig() if outer_config is None else outer_config
    if not isinstance(config, EstimatorConfig) or config.pipeline != "planning_only":
        raise ValueError("outer configuration must use the independent planning_only interface")
    if config.estimator == "extra_trees" and config.parameters["n_jobs"] != 1:
        raise ValueError("v012 outer ExtraTrees requires n_jobs=1")
    x = frame[list(columns)].to_numpy(float)
    a = frame.action.to_numpy(float)
    y = frame[["next_state", "next_w"]].to_numpy(float)
    modules = SimpleGQ(baseline_features, action_features, memory_index=2,
                       physical_indices=(0, 1)).fit(x, a, y)
    b, d = dictionaries(x, a)
    bs, ds = np.max(abs(b), axis=0), np.max(abs(d), axis=0)
    projection = np.linalg.lstsq(b/bs, np.column_stack((d/ds, y)), rcond=None)[0]
    core = TwoModuleOuterCore()
    core.modules = modules
    core.projection_d_ = projection[:, :d.shape[1]]*ds[None, :]/bs[:, None]
    core.projection_y_ = projection[:, d.shape[1]:]/bs[:, None]
    core.outer_config = deepcopy(config)
    core.separate_outputs = config.estimator == "gradient_boosting"
    z = core.outer_features(x, a)
    fit_warnings = []
    if config.estimator == "linear":
        design = np.column_stack((np.ones(len(z)), z))
        scale = np.maximum(np.max(abs(design), axis=0), np.finfo(float).tiny)
        coef, _, rank, _ = np.linalg.lstsq(design/scale, y, rcond=None)
        core.outer_coefficients_ = coef/scale[:, None]
        outer_rank = int(rank)
    else:
        core.outer_model_ = ([_make_estimator(config.estimator, config.parameters) for _ in OUTPUTS]
                             if core.separate_outputs else _make_estimator(config.estimator, config.parameters))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if core.separate_outputs:
                for j, model in enumerate(core.outer_model_):
                    model.fit(z, y[:, j])
            else:
                core.outer_model_.fit(z, y)
        fit_warnings = [str(w.message) for w in caught]
        outer_rank = None
    diagnostics = dict(modules.diagnostics_, g_spec=SPEC, estimator="v012_two_module_outer",
        physical_inputs=list(columns), physical_outputs=["next_state", "next_w"],
        module_estimator="same_sample_unpenalized_FWL", policy_columns=list(policies),
        policy_output_status="conditioning_only_no_macro_transition",
        outer_config=config.as_dict(), outer_features=["m0_state", "m0_w", "Gamma_state", "Gamma_w"],
        outer_training="same_sample_module_fits_no_crossfit", outer_rank=outer_rank,
        outer_coefficient_identification="minimum_norm_if_rank_deficient_no_penalty",
        outer_warnings=fit_warnings, residual_order=list(FULL_COLUMNS),
        coincident_residual_rule="retain_separate_regression_sources_no_alias_or_deduplication")
    return core, diagnostics


def select_residuals(core, x, a, y, columns):
    columns = tuple(columns)
    if columns not in (FULL_COLUMNS, GAMMA_COLUMNS):
        raise ValueError("v012 requires the fixed full raw group or entire Gamma residual")
    full = core.regression_residuals(x, a, y)
    return full if columns == FULL_COLUMNS else full[:, 2:4].copy()


def validate_history(frame, columns):
    columns = tuple(columns)
    if not columns:
        return
    if columns not in (FULL_COLUMNS, GAMMA_COLUMNS):
        raise ValueError("invalid v012 fixed residual layout")
    missing = set((*CLOCK_COLUMNS, *columns)) - set(frame)
    if missing:
        raise ValueError(f"missing dated v012 residual fields: {sorted(missing)}")
    clock = frame[list(CLOCK_COLUMNS)].to_numpy(float)
    if not np.isfinite(clock).all() or not np.equal(clock, np.floor(clock)).all():
        raise ValueError("v012 residual clocks must be finite integers")
    source, release, fit, available, warm = clock.T
    date = frame.period.to_numpy(float)
    if not ((source == date-1).all() and (release == date).all() and (available == date).all()
            and (fit <= source).all() and np.isin(warm, [0, 1]).all()):
        raise ValueError("v012 residual must use its issued vintage and previous completed release")
    values = frame[list(columns)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("complete raw v012 residual group required")
    if not ((values[warm == 1] == 0).all() and (fit[warm == 1] == source[warm == 1]).all()):
        raise ValueError("warmup zeros must be explicitly marked as unmeasured")


def clocks(time, fit_as_of, *, warmup=False):
    return dict(residual_source_period=time-1, residual_release_period=time,
                residual_fit_as_of=fit_as_of, auxiliary_as_of=time,
                residual_warmup=int(warmup))
