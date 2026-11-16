"""Released physical-history boundary shared by flexible G entry points.

Auxiliary columns are never read here. This API can therefore fit physical G
even when the downstream auxiliary panel is absent or incomplete.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .g import IdentificationError
from .g_estimators import EstimatorConfig

PHYSICAL_COLUMNS = ("state", "w", "previous_action", "action", "next_state", "next_w")


def released_physical_history(history, *, as_of, window=24):
    from .native_gq import _date, _positive_integer
    as_of, window = _date(as_of), _positive_integer(window, "window")
    if not isinstance(history, pd.DataFrame):
        raise TypeError("history must be a pandas DataFrame")
    required = [*PHYSICAL_COLUMNS, "path_id", "period"]
    missing = set(required).difference(history.columns)
    if missing:
        raise ValueError(f"missing physical history columns: {sorted(missing)}")
    periods = history.period.to_numpy(float)
    if not np.isfinite(periods).all() or not np.equal(periods, np.floor(periods)).all():
        raise ValueError("history.period must contain finite integer action dates")
    columns = required + (["release_period"] if "release_period" in history else [])
    frame = history.loc[(periods >= as_of-window) & (periods < as_of), columns].copy()
    if "release_period" in frame:
        release = frame.release_period.to_numpy(float)
        if (not np.isfinite(release).all() or not np.equal(release, np.floor(release)).all()
                or (release <= frame.period.to_numpy(float)).any()):
            raise ValueError("release_period must be an integer after its action date")
        frame = frame.loc[release <= as_of].copy()
    if frame.duplicated(["path_id", "period"]).any():
        raise ValueError("duplicate path/date observations")
    frame = frame.loc[np.isfinite(frame[required].to_numpy(float)).all(axis=1)].copy()
    if frame.empty:
        raise IdentificationError("no complete released physical transitions")
    ordered = frame.sort_values(["path_id", "period"], kind="mergesort")
    lag_action = ordered.groupby("path_id", sort=False).action.shift(1)
    lag_date = ordered.groupby("path_id", sort=False).period.shift(1)
    adjacent = ordered.period.eq(lag_date+1)
    if not np.allclose(ordered.loc[adjacent, "previous_action"], lag_action.loc[adjacent],
                       rtol=0, atol=1e-12):
        raise ValueError("previous_action must be the executed action at period-1")
    return frame


def fit_physical_g(history, *, as_of, window=24, estimator="linear",
                   estimator_params=None, config=None, seed=0):
    """Refit the declared fixed dictionary on completed physical/action rows.

    Call after every release for rolling estimation. The estimator choice is
    explicit, never selected from E or its availability. All preprocessing is
    fitted on these released physical rows only.
    """
    from .native_gq import _fit_model
    selected = (EstimatorConfig(estimator=estimator, estimator_params=estimator_params)
                if config is None else config)
    frame = released_physical_history(history, as_of=as_of, window=window)
    model = _fit_model(frame, estimator_config=selected, seed=seed)
    model.diagnostics_.update(as_of=int(as_of), window=int(window),
        training_first_period=int(frame.period.min()),
        training_last_period=int(frame.period.max()),
        physical_auxiliary_rule="excluded_before_fit_preprocessing_selection_and_prediction")
    return model
