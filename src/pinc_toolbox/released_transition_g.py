"""Append-only released transition predictions and residual observations.

The caller owns estimation.  This module accepts an already fitted model with
``predict(states)`` (including ``TransitionG``) and applies only release-clock
and issued-prediction bookkeeping.  It does not fit, revise, or certify a
policy/action model.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np


def _clock(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer clock")
    return int(value)


def _vector(value: Any, name: str) -> list[float]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    return array.tolist()


def _transition(row: Mapping[str, Any], *, observed: bool) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("public transition must be a mapping")
    record_id = row.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be a nonempty string")
    decision = _clock(row.get("decision_time"), "decision_time")
    release = _clock(row.get("outcome_release"), "outcome_release")
    if release <= decision:
        raise ValueError("outcome_release must follow decision_time")
    result = dict(record_id=record_id, decision_time=decision,
                  outcome_release=release, states=_vector(row.get("states"), "states"))
    if observed:
        result["next_physical"] = _vector(row.get("next_physical"), "next_physical")
    return result


def _predict(model: Any, states: Sequence[Sequence[float]]) -> np.ndarray:
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise TypeError("fitted model must provide predict(states)")
    values = np.asarray(predict(np.asarray(states, dtype=float)), dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != len(states) or not np.isfinite(values).all():
        raise ValueError("model.predict must return one finite vector per state")
    return values


def prediction_errors(model: Any, predictors: Sequence[Sequence[float]],
                      observed_successors: Sequence[Sequence[float]], *,
                      decision_times: Sequence[int], release_times: Sequence[int],
                      as_of: int, fit_as_of: int) -> dict[str, Any]:
    """Predict a released public prefix in one batch and return its residuals.

    Unreleased rows are excluded before prediction and do not appear in the
    returned prediction/error arrays.  The full boolean mask records which
    supplied rows were eligible at ``as_of``.
    """
    clock = _clock(as_of, "as_of")
    fit_clock = _clock(fit_as_of, "fit_as_of")
    if clock < fit_clock:
        raise ValueError("as_of cannot precede fit_as_of")
    if not hasattr(predictors, "__len__") or not hasattr(observed_successors, "__len__"):
        raise ValueError("predictors and observed_successors must be sized sequences")
    if len(predictors) != len(observed_successors):
        raise ValueError("predictors and observed_successors must be aligned matrices")
    decisions = list(decision_times)
    releases = list(release_times)
    if len(decisions) != len(predictors) or len(releases) != len(predictors):
        raise ValueError("clock arrays must match predictor rows")
    decisions = [_clock(value, "decision_time") for value in decisions]
    releases = [_clock(value, "outcome_release") for value in releases]
    released = [release <= clock for release in releases]
    selected_indices = [index for index, eligible in enumerate(released) if eligible]
    for index in selected_indices:
        if releases[index] <= decisions[index]:
            raise ValueError("outcome_release must follow decision_time")
    if selected_indices and fit_clock > min(decisions[index] for index in selected_indices):
        raise ValueError("fit_as_of cannot follow a selected prediction decision")
    if not selected_indices:
        return {
            "as_of": clock, "fit_as_of": fit_clock,
            "released_mask": released, "selected_indices": [],
            "decision_times": [], "outcome_releases": [],
            "predictions": [], "observed_successors": [], "prediction_errors": [],
            "interpretation": "fixed-reference batch replay of released physical-transition residuals; "
                              "not a historical issued-prediction ledger and no action/Q certification",
        }
    # Future predictor/outcome objects are not converted or inspected.
    selected_states = np.asarray([predictors[index] for index in selected_indices], dtype=float)
    selected_observed = np.asarray([observed_successors[index] for index in selected_indices], dtype=float)
    if selected_states.ndim != 2 or selected_observed.ndim != 2 or not np.isfinite(selected_observed).all():
        raise ValueError("released predictors and successors must be finite matrices")
    predictions = _predict(model, selected_states)
    if predictions.shape != selected_observed.shape:
        raise ValueError("model output dimension differs from observed successors")
    errors = selected_observed - predictions
    if not np.isfinite(errors).all():
        raise ValueError("prediction errors must be finite")
    return {
        "as_of": clock,
        "fit_as_of": fit_clock,
        "released_mask": released,
        "selected_indices": selected_indices,
        "decision_times": [decisions[index] for index in selected_indices],
        "outcome_releases": [releases[index] for index in selected_indices],
        "predictions": predictions.tolist(),
        "observed_successors": selected_observed.tolist(),
        "prediction_errors": errors.tolist(),
        "interpretation": "fixed-reference batch replay of released physical-transition residuals; "
                          "not a historical issued-prediction ledger and no action/Q certification",
    }


class ReleasedTransitionG:
    """Release-clock ledger for a fixed, already fitted transition model."""

    def __init__(self, model: Any, *, fit_as_of: int, fit_id: str | None = None):
        if not callable(getattr(model, "predict", None)):
            raise TypeError("model must provide predict(states)")
        self._model = model
        self.fit_as_of = _clock(fit_as_of, "fit_as_of")
        if fit_id is not None and (not isinstance(fit_id, str) or not fit_id):
            raise ValueError("fit_id must be a nonempty string or None")
        self.fit_id = fit_id
        self._rows: dict[str, dict[str, Any]] = {}
        self._forecasts: dict[str, dict[str, Any]] = {}
        self._errors: dict[str, dict[str, Any]] = {}
        self._latest_clock: int | None = None

    def _advance(self, as_of: int) -> int:
        clock = _clock(as_of, "as_of")
        if self._latest_clock is not None and clock < self._latest_clock:
            raise ValueError("stream clock cannot move backwards")
        if clock < self.fit_as_of:
            raise ValueError("as_of cannot precede fit_as_of")
        return clock

    def issue(self, *, record_id: str, states: Sequence[float], decision_time: int,
              outcome_release: int, as_of: int) -> dict[str, Any]:
        clock = self._advance(as_of)
        row = _transition(dict(record_id=record_id, states=states,
                               decision_time=decision_time,
                               outcome_release=outcome_release), observed=False)
        if clock != row["decision_time"] or row["outcome_release"] <= clock:
            raise ValueError("issue at decision_time before outcome_release")
        if row["record_id"] in self._rows or row["record_id"] in self._forecasts:
            raise ValueError("cannot overwrite an issued or observed record")
        prediction = _predict(self._model, [row["states"]])[0].tolist()
        issued = dict(row, issued_at=clock, fit_as_of=self.fit_as_of,
                      fit_id=self.fit_id, prediction=prediction,
                      interpretation="physical transition prediction; no action/Q certification")
        self._forecasts[row["record_id"]] = issued
        self._latest_clock = clock
        return deepcopy(issued)

    def issue_many(self, rows: Sequence[Mapping[str, Any]], *, as_of: int) -> list[dict[str, Any]]:
        clock = self._advance(as_of)
        parsed = [_transition(row, observed=False) for row in rows]
        if any(row["decision_time"] != clock or row["outcome_release"] <= clock for row in parsed):
            raise ValueError("all batch issues must be at the current decision clock and unreleased")
        ids = [row["record_id"] for row in parsed]
        if len(set(ids)) != len(ids) or any(key in self._rows or key in self._forecasts for key in ids):
            raise ValueError("cannot overwrite or duplicate an issued record")
        predictions = _predict(self._model, [row["states"] for row in parsed])
        issued = [dict(row, issued_at=clock, fit_as_of=self.fit_as_of, fit_id=self.fit_id,
                       prediction=predictions[index].tolist(),
                       interpretation="physical transition prediction; no action/Q certification")
                  for index, row in enumerate(parsed)]
        self._forecasts.update({row["record_id"]: row for row in issued})
        self._latest_clock = clock
        return deepcopy(issued)

    def ingest(self, rows: Sequence[Mapping[str, Any]], *, as_of: int) -> None:
        clock = self._advance(as_of)
        incoming = [_transition(row, observed=True) for row in rows]
        staged_rows = dict(self._rows)
        staged_errors = dict(self._errors)
        for row in incoming:
            if row["outcome_release"] > clock:
                raise ValueError("cannot ingest an unreleased outcome")
            key = row["record_id"]
            if key in staged_rows and staged_rows[key] != row:
                raise ValueError("cannot revise an already released transition")
            issued = self._forecasts.get(key)
            if issued is not None:
                for name in ("decision_time", "outcome_release", "states"):
                    if issued[name] != row[name]:
                        raise ValueError(f"released transition disagrees with issued {name}")
                prediction = np.asarray(issued["prediction"], dtype=float)
                observed = np.asarray(row["next_physical"], dtype=float)
                if prediction.shape != observed.shape:
                    raise ValueError("outcome dimension differs from issued prediction")
                staged_errors[key] = dict(record_id=key, outcome_release=row["outcome_release"],
                                          observed=row["next_physical"],
                                          prediction=issued["prediction"],
                                          prediction_error=(observed - prediction).tolist(),
                                          forecast_issued_at=issued["issued_at"],
                                          fit_as_of=issued["fit_as_of"], fit_id=issued["fit_id"])
            staged_rows[key] = row
        self._rows, self._errors, self._latest_clock = staged_rows, staged_errors, clock

    def snapshot(self) -> dict[str, Any]:
        return deepcopy({"as_of": self._latest_clock, "fit_as_of": self.fit_as_of,
                         "fit_id": self.fit_id, "forecasts": list(self._forecasts.values()),
                         "released_errors": list(self._errors.values()),
                         "released_transitions": list(self._rows.values()),
                         "interpretation": "released physical-transition residuals; no action/Q certification"})


__all__ = ["ReleasedTransitionG", "prediction_errors"]
