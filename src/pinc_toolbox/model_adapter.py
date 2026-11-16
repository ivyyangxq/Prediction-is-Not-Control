"""Generic environment intake for the shared PINC detector.

This module deliberately stops at intake and validation.  It does not choose a
cutoff, fit a model, or manufacture an action comparison.  The validated
description is consumed by the existing GQ/RA-GQ runners.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple


class AdapterError(ValueError):
    """Raised when an environment declaration is incomplete or inconsistent."""


@dataclass(frozen=True)
class EnvironmentSpec:
    """Economic meaning and information timing of one environment."""

    state: Tuple[str, ...]
    action: Optional[str]
    reward: str
    next_state: Tuple[str, ...]
    release_date: str
    decision_date: str
    perceived_model: Any = None
    hidden: Tuple[str, ...] = ()
    q_applicable: bool = True
    action_is_structural: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        for name, value in (("state", self.state), ("next_state", self.next_state), ("hidden", self.hidden)):
            value = tuple(value)
            if ((name != "hidden" and not value)
                    or any(not isinstance(x, str) or not x.strip() for x in value)):
                raise AdapterError(f"{name} must contain nonempty column names")
            if len(set(value)) != len(value):
                raise AdapterError(f"{name} contains duplicate column names")
            object.__setattr__(self, name, value)
        for name in ("reward", "release_date", "decision_date"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise AdapterError(f"{name} must be a nonempty column name")
        if self.action is not None and (not isinstance(self.action, str) or not self.action.strip()):
            raise AdapterError("action must be a column name or None")
        if set(self.state) & set(self.hidden):
            raise AdapterError("hidden columns cannot also be public state columns")
        if not self.q_applicable and self.action_is_structural:
            # An actionless G-only environment may still have an observed action,
            # but the caller must not silently claim a structural Q comparison.
            object.__setattr__(self, "action_is_structural", False)


@dataclass(frozen=True)
class ReleasedData:
    """Column-oriented released observations.

    ``table`` may be a mapping or a dataframe-like object supporting
    ``__getitem__`` and membership.  No rows are fitted or dropped here.
    """

    table: Any
    action_grid: Tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def columns(self) -> set[str]:
        if isinstance(self.table, Mapping):
            return {str(k) for k in self.table.keys()}
        cols = getattr(self.table, "columns", None)
        if cols is not None:
            return {str(k) for k in cols}
        raise AdapterError("table must be a mapping or dataframe-like object with columns")

    def __post_init__(self) -> None:
        grid = tuple(float(x) for x in self.action_grid)
        if any(x != x or x in (float("inf"), float("-inf")) for x in grid):
            raise AdapterError("action_grid must contain finite numbers")
        if len(set(grid)) != len(grid):
            raise AdapterError("action_grid contains duplicate values")
        object.__setattr__(self, "action_grid", grid)


@dataclass(frozen=True)
class DetectorConfig:
    """Execution choices shared by environments; no threshold is implicit."""

    eta: Any
    fit_window: int
    monitoring_window: int
    q_horizon: Optional[int] = None
    q_mode: str = "none"
    detector: str = "declared_gq_ragq"

    def __post_init__(self) -> None:
        if int(self.fit_window) <= 0 or int(self.monitoring_window) <= 0:
            raise AdapterError("fit_window and monitoring_window must be positive")
        if self.q_mode not in {"none", "finite", "infinite"}:
            raise AdapterError("q_mode must be 'none', 'finite', or 'infinite'")
        if self.q_mode == "finite" and (self.q_horizon is None or int(self.q_horizon) <= 0):
            raise AdapterError("finite Q requires a positive q_horizon")
        if self.q_mode == "infinite" and self.q_horizon is not None:
            raise AdapterError("infinite Q must not declare a finite q_horizon")


def validate_environment(spec: EnvironmentSpec, data: ReleasedData, config: DetectorConfig) -> dict[str, Any]:
    """Validate a declaration and return a machine-readable intake report.

    The report is intentionally descriptive.  It does not infer missing
    columns, discard unsupported rows, select eta, or calculate an alarm.
    """
    columns = data.columns()
    required = set(spec.state) | set(spec.next_state) | {spec.reward, spec.release_date, spec.decision_date}
    if spec.action is not None:
        required.add(spec.action)
    missing = sorted(required - columns)
    if missing:
        raise AdapterError("missing required columns: " + ", ".join(missing))
    if spec.q_applicable and not data.action_grid:
        raise AdapterError("q_applicable=True requires an explicit action_grid")
    if spec.q_applicable and spec.action is None:
        raise AdapterError("q_applicable=True requires an action column")
    if config.q_mode != "none" and not spec.q_applicable:
        raise AdapterError("Q execution requested for an environment that declared q_applicable=False")
    return {
        "status": "ready_for_core",
        "state_columns": list(spec.state),
        "action_column": spec.action,
        "reward_column": spec.reward,
        "next_state_columns": list(spec.next_state),
        "release_column": spec.release_date,
        "decision_column": spec.decision_date,
        "hidden_columns": list(spec.hidden),
        "q_applicable": bool(spec.q_applicable),
        "action_is_structural": bool(spec.action_is_structural),
        "action_grid": list(data.action_grid),
        "q_mode": config.q_mode,
        "q_horizon": config.q_horizon,
        "fit_window": int(config.fit_window),
        "monitoring_window": int(config.monitoring_window),
        "detector": config.detector,
        "calibration": "caller_declared; no cutoff selected by intake",
    }


__all__ = ["AdapterError", "EnvironmentSpec", "ReleasedData", "DetectorConfig", "validate_environment"]
