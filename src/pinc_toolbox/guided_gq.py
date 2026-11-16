"""Run the native GQ solver from a named, released panel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .native_gq import RND2GQ, XCOL
from .reward_equation import RewardEquation


MODES = ("Offline", "Online (Observed Inputs)", "Online (Estimated-State Inputs)")


def _clock(column, numeric):
    if pd.api.types.is_numeric_dtype(column) != numeric:
        raise ValueError("panel and macro timestamps must use the same clock: periods or ISO dates")
    if numeric:
        values = column.to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("timestamps must be finite")
        return values
    times = pd.to_datetime(column, utc=True)
    if times.isna().any():
        raise ValueError("timestamps must be complete")
    return pd.DatetimeIndex(times).asi8


def panel_history(table, columns, *, macro=None):
    """Map chosen data columns to the native model's released history."""
    decision_column = table[columns["period"]]
    numeric = pd.api.types.is_numeric_dtype(decision_column)
    decisions = _clock(decision_column, numeric)
    releases = _clock(table[columns["release_period"]], numeric)
    if (releases <= decisions).any():
        raise ValueError("a transition must be released after its decision date")
    history = pd.DataFrame({name: pd.to_numeric(table[source]) for name, source in columns.items()
                            if name not in {"period", "release_period"}})
    if numeric:
        history["period"], history["release_period"] = decisions, releases
    else:
        calendar = np.unique(decisions)
        history["period"] = np.searchsorted(calendar, decisions)
        history["release_period"] = np.searchsorted(calendar, releases)
        history.attrs["decision_calendar"] = pd.to_datetime(calendar, utc=True).astype(str).tolist()

    if macro is not None:
        macro_table, value_column, release_column = macro
        published = _clock(macro_table[release_column], numeric)
        values = pd.to_numeric(macro_table[value_column]).to_numpy(float)
        if not len(published) or not np.isfinite(values).all() or len(np.unique(published)) != len(published):
            raise ValueError("macro W needs finite values and unique publication timestamps")
        order = np.argsort(published)
        published, values = published[order], values[order]
        current = np.searchsorted(published, decisions, side="right") - 1
        if numeric:
            successor = np.searchsorted(published, decisions + 1, side="right") - 1
        else:
            next_index = history.period.to_numpy() + 1
            known = next_index < len(calendar)
            successor = np.full(len(history), -1)
            successor[known] = np.searchsorted(published, calendar[next_index[known]], side="right") - 1
        history["w"] = np.where(current >= 0, values[current], np.nan)
        history["next_w"] = np.where(successor >= 0, values[successor], np.nan)

    if "path_id" not in history:
        history["path_id"] = 0
    for name in ("path_id", "period", "release_period"):
        values = history[name].to_numpy(float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{columns.get(name, name)} must contain integer values")
        history[name] = history[name].astype(int)
    if history.duplicated(["path_id", "period"]).any():
        raise ValueError("the data contain duplicate path/decision dates")
    history = history.sort_values(["path_id", "period"], kind="stable")
    if "previous_action" not in history:
        grouped = history.groupby("path_id", sort=False)
        consecutive = history.period.eq(grouped.period.shift(1) + 1)
        history["previous_action"] = grouped.action.shift(1).where(consecutive)
    return history


def run_panel(history, settings):
    """Fit and query snapshots in release order, with one prescribed reward."""
    mode = settings["mode"]
    if mode not in MODES:
        raise ValueError("unknown GQ mode")
    start = settings["start"]
    window = settings["window"]
    end = settings["end"]
    path = history.loc[history.path_id.eq(settings["path"])].set_index("period")
    if start not in path.index or end < start:
        raise ValueError("initial decision date is missing for the selected path")
    observed_inputs = mode == MODES[1]
    dates = path.loc[start:end].index if observed_inputs else range(start, end + 1)
    exploration = float(settings.get("exploration_probability", 0))
    if not 0 <= exploration <= 1 or (observed_inputs and exploration):
        raise ValueError("exploration must be a probability and applies only to predicted paths")
    rng = np.random.default_rng(settings.get("exploration_seed", 0))

    reward = settings["reward"]
    inputs = reward["inputs"]
    equation = RewardEquation(reward["expression"], inputs, reward["parameters"])
    model = RND2GQ(horizon=settings["horizon"], beta=settings["discount"],
                   action_grid=settings["grid"], reward_equation=equation, reward_inputs=inputs)
    grid = model.action_grid_
    feasible = np.isin(grid, settings["candidates"])
    model.fit(history, as_of=start, window=window, feasible=feasible)

    result = []
    estimated = path.loc[start, XCOL].to_numpy(float)
    for time in dates:
        time = int(time)
        if mode != MODES[0] and time > start:
            model.update(history, as_of=time, window=window)
        entry = {"date": time, "fit_as_of": model.as_of_}
        if "decision_calendar" in history.attrs:
            entry["decision_timestamp"] = history.attrs["decision_calendar"][time]
        if observed_inputs:
            row = path.loc[time]
            state = row[XCOL].to_numpy(float)
            action = float(row.action) if np.isfinite(row.action) else None
            entry["recorded_action"] = action
        else:
            state = estimated
            action = None
        if (state is None or not np.isfinite(state).all()
                or (observed_inputs and action is None)):
            entry.update(status="input_unavailable")
            result.append(entry)
            estimated = None
            continue
        q = model.q_values(state[None, :], evaluation="bellman")[0]
        finite = np.isfinite(q)
        greedy = float(grid[np.argmax(q)]) if finite.any() else None
        action_source = "observed" if observed_inputs else None
        if not observed_inputs and greedy is not None:
            if exploration and rng.random() < exploration:
                action = float(rng.choice(grid[finite]))
                action_source = "exploration"
            else:
                action = greedy
                action_source = "greedy"
        payoff = (model._prescribed_rewards(state[None, :], [action])[0, 0]
                  if action is not None else np.nan)
        reward_value = float(payoff) if np.isfinite(payoff) else None
        entry.update(status="available" if finite.any() else "support_unavailable",
                     state=state.tolist(),
                     state_source="observed" if observed_inputs or time == start else "issued_G_prediction",
                     action_used_for_reward=action, action_source=action_source, reward_value=reward_value,
                     recommended_action=greedy,
                     action_values=[{"action": float(a), "q": float(v) if np.isfinite(v) else None}
                                    for a, v, allowed in zip(grid, q, feasible) if allowed])
        if observed_inputs:
            entry["reward_at_recorded_action"] = reward_value
        else:
            # The current fit issues the forecast at the selected action.
            # The next refit leaves this already-issued state intact.
            estimated = (model.model_.transition_.predict(state[None, :], [action])[0]
                         if action is not None else None)
            entry["issued_next_state"] = estimated.tolist() if estimated is not None else None
        result.append(entry)
    return {"mode": mode, "configuration": settings, "decisions": result}
