"""Released-data G fitting and append-only forecast/error observations.

This optional module is not exported by the active package initializer. It
uses the existing unpenalized SimpleGQ estimator. Forecasts concern observed
physical transitions; they do not certify interventions or a Q map.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from .g import SimpleGQ, linear_physical_memory_features, scalar_action_features


def _clock(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer clock")
    return int(value)


def _record(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("public transition must be a mapping")
    key = row["record_id"]
    if not isinstance(key, str) or not key:
        raise ValueError("record_id must be a nonempty string")
    decision = _clock(row["decision_time"], "decision_time")
    release = _clock(row["outcome_release"], "outcome_release")
    if release <= decision:
        raise ValueError("outcome_release must follow decision_time")
    states = np.asarray(row["states"], dtype=float)
    target = np.asarray(row["next_physical"], dtype=float)
    action = float(row["action"])
    if states.ndim != 1 or target.ndim != 1 or not states.size or not target.size:
        raise ValueError("states and next_physical must be nonempty vectors")
    if not np.isfinite(states).all() or not np.isfinite(target).all() or not np.isfinite(action):
        raise ValueError("public transition values must be finite")
    return dict(record_id=key, decision_time=decision, outcome_release=release,
                states=states.tolist(), action=action, next_physical=target.tolist())


class ReleasedG:
    """A release-by-release rolling G fit with immutable issued predictions.

    Call ``ingest`` when outcomes arrive, ``fit`` on released history, and
    ``issue`` before the target outcome arrives. A later fit never revises a
    previously issued prediction. ``snapshot`` returns copies. The reference
    used for source inference is a separate fixed object in eta_monitor.
    """

    def __init__(self, *, memory_index: int, physical_indices: Sequence[int], window: int):
        if isinstance(memory_index, bool) or not isinstance(memory_index, int):
            raise ValueError("memory_index must be an integer")
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer")
        self.memory_index = memory_index
        self.physical_indices = tuple(physical_indices)
        if not self.physical_indices or any(isinstance(i, bool) or not isinstance(i, int) for i in self.physical_indices):
            raise ValueError("physical_indices must be nonempty integer indices")
        self.window = window
        self._rows: dict[str, dict[str, Any]] = {}
        self._forecasts: dict[str, dict[str, Any]] = {}
        self._errors: dict[str, dict[str, Any]] = {}
        self._model: SimpleGQ | None = None
        self._fit_as_of: int | None = None
        self._latest_clock: int | None = None

    def _advance(self, as_of: int) -> int:
        clock = _clock(as_of, "as_of")
        if self._latest_clock is not None and clock < self._latest_clock:
            raise ValueError("stream clock cannot move backwards")
        return clock

    def ingest(self, rows: Sequence[Mapping[str, Any]], *, as_of: int) -> None:
        clock = self._advance(as_of)
        incoming = [_record(row) for row in rows]
        staged = dict(self._rows)
        errors = dict(self._errors)
        for row in incoming:
            key = row["record_id"]
            if row["outcome_release"] > clock:
                raise ValueError("cannot ingest an unreleased outcome")
            if key in staged and staged[key] != row:
                raise ValueError("cannot revise an already released transition")
            issued = self._forecasts.get(key)
            if issued is not None:
                for name in ("decision_time", "outcome_release", "states", "action"):
                    if issued[name] != row[name]:
                        raise ValueError(f"released transition disagrees with issued {name}")
                observed = np.asarray(row["next_physical"])
                prediction = np.asarray(issued["prediction"])
                if observed.shape != prediction.shape:
                    raise ValueError("outcome dimension differs from issued prediction")
                errors[key] = dict(record_id=key, outcome_release=row["outcome_release"],
                                   observed=row["next_physical"],
                                   prediction_error=(observed-prediction).tolist(),
                                   forecast_issued_at=issued["issued_at"])
            staged[key] = row
        self._rows, self._errors, self._latest_clock = staged, errors, clock

    def fit(self, *, as_of: int) -> dict[str, Any]:
        clock = self._advance(as_of)
        rows = list(self._rows.values())
        if not rows:
            raise ValueError("no released transitions")
        model = SimpleGQ(
            lambda x: linear_physical_memory_features(x, self.physical_indices, self.memory_index),
            scalar_action_features, memory_index=self.memory_index,
            physical_indices=self.physical_indices,
        )
        model.fit_released_window(
            [r["states"] for r in rows], [r["action"] for r in rows],
            [r["next_physical"] for r in rows], [r["outcome_release"] for r in rows],
            as_of=clock, window=self.window,
        )
        self._model, self._fit_as_of, self._latest_clock = model, clock, clock
        return deepcopy(model.diagnostics_)

    def issue(self, *, record_id: str, states: Sequence[float], action: float,
              decision_time: int, outcome_release: int, as_of: int) -> dict[str, Any]:
        clock = self._advance(as_of)
        decision = _clock(decision_time, "decision_time")
        release = _clock(outcome_release, "outcome_release")
        if clock != decision or release <= clock:
            raise ValueError("issue at decision_time before outcome_release")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a nonempty string")
        if record_id in self._forecasts or record_id in self._rows:
            raise ValueError("cannot overwrite an issued or observed record")
        if self._model is None or self._fit_as_of is None or self._fit_as_of > clock:
            raise ValueError("fit released G before issuing a prediction")
        x = np.asarray(states, dtype=float)
        if x.ndim != 1:
            raise ValueError("states must be a vector")
        prediction = self._model.predict_physical(x[None, :], [action])[0]
        result = dict(record_id=record_id, states=x.tolist(), action=float(action),
                      decision_time=decision, outcome_release=release, issued_at=clock,
                      fit_as_of=self._fit_as_of, prediction=prediction.tolist(),
                      interpretation="physical prediction; no true-action/Q certification")
        self._forecasts[record_id] = result
        self._latest_clock = clock
        return deepcopy(result)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(dict(as_of=self._latest_clock, fit_as_of=self._fit_as_of,
                             forecasts=list(self._forecasts.values()),
                             released_errors=list(self._errors.values()),
                             released_transitions=list(self._rows.values())))


def fit_g_offline(rows: Sequence[Mapping[str, Any]], *, as_of: int,
                  memory_index: int, physical_indices: Sequence[int], window: int) -> ReleasedG:
    """Fit from public rows released by as_of, without inspecting future values."""
    clock = _clock(as_of, "as_of")
    released = [r for r in rows if _clock(r["outcome_release"], "outcome_release") <= clock]
    stream = ReleasedG(memory_index=memory_index, physical_indices=physical_indices, window=window)
    stream.ingest(released, as_of=clock)
    stream.fit(as_of=clock)
    return stream
