"""Isolated transactional session for the certified eta monitor operation.

This module is intentionally separate from the existing floating
``eta_monitor`` path.  It keeps the append-only public-history/session rules
while delegating initialization to ``select_eta_certified`` and monitoring to
the pending ``monitor_history_certified`` implementation.  Its numerical
scope is exact represented-input construction and source verification,
conditional on the stated model/MGF; it has no kappa (``κ``), real-valued DGP,
or Q certification interface.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any

from .eta_monitor import _integer, _public_row
from .certified_reference import select_eta_certified
from .source_arithmetic import validate_source_arithmetic_rule


_NUMERICAL_SCOPE = (
    "exact represented-input construction and source verification, conditional "
    "on stated model/MGF; no kappa interface, no real-valued DGP or Q certification"
)


def _not_ready_monitor() -> dict[str, Any]:
    return {
        "status": "not_ready", "records": [],
        "numerical_certification": _NUMERICAL_SCOPE,
        "e_eta": None, "D_L_minus_M": None,
        "action_Q_certification": "unavailable",
    }


class PublicCertifiedEtaMonitor:
    """Append-only released-history session with fixed certified initialization.

    Rows, initialization, published records, and the prefix cache are staged
    locally.  They are committed only after selection or monitoring completes;
    malformed rows, late backfills, changed rows, and downstream errors leave
    the prior state untouched.
    """

    def __init__(self, reference_spec, monitor_spec):
        validate_source_arithmetic_rule(monitor_spec.get("source_arithmetic_allowance_rule"))
        source_execution_mode = monitor_spec.get("source_execution_mode", "exhaustive")
        if source_execution_mode not in ("exhaustive", "lazy"):
            raise ValueError("source_execution_mode must be exhaustive or lazy")
        self._reference_spec = deepcopy(reference_spec)
        self._monitor_spec = deepcopy(monitor_spec)
        self._rows: dict[tuple[int, int], dict[str, Any]] = {}
        self._initialization: dict[str, Any] | None = None
        self._published: dict[int, dict[str, Any]] = {}
        self._prefix_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self._prepared_cache: dict[int, dict[str, Any]] = {}
        self._as_of = -1
        self._last: dict[str, Any] | None = None

    def update(self, rows, *, as_of, return_snapshot=True):
        if not isinstance(return_snapshot, bool):
            raise ValueError("return_snapshot must be a boolean")
        clock = _integer(as_of, "as_of")
        if clock < self._as_of:
            raise ValueError("release clock cannot move backwards")
        staged_rows = dict(self._rows)
        for raw in rows:
            row = _public_row(raw)
            key = (_integer(row["period"], "period"), _integer(row["household"], "household"))
            release = _integer(row["outcome_release"], "outcome_release")
            if release > clock:
                raise ValueError("session accepts only released rows")
            if key in staged_rows and staged_rows[key] != row:
                raise ValueError("cannot revise a released public row")
            if key not in staged_rows and release <= self._as_of:
                raise ValueError("late backfill would revise an earlier synchronous history")
            staged_rows[key] = deepcopy(row)

        history = list(staged_rows.values())
        staged_initialization = self._initialization
        staged_cache = dict(self._prefix_cache)
        staged_prepared = dict(self._prepared_cache)
        staged_published = dict(self._published)

        if staged_initialization is None:
            trial = select_eta_certified(
                history, self._reference_spec, as_of=clock,
                prefix_cache=staged_cache,
            )
            if trial["status"] == "ready":
                staged_initialization = trial
            else:
                result = dict(
                    initialization=trial, monitoring=_not_ready_monitor(), as_of=clock,
                    numerical_certification=_NUMERICAL_SCOPE,
                )
                self._rows = staged_rows
                self._as_of = clock
                self._prefix_cache = staged_cache
                self._prepared_cache = staged_prepared
                self._last = result
                return deepcopy(result) if return_snapshot else None

        from .certified_monitor import monitor_history_certified
        monitoring = monitor_history_certified(
            history, staged_initialization, self._monitor_spec,
            self._reference_spec, as_of=clock,
            skip_windows=set(staged_published),
            prepared_cache=staged_prepared,
        )
        pending = []
        for record in monitoring["records"]:
            if record["availability"] != "not_yet_released":
                staged_published[record["window_index"]] = record
            else:
                pending.append(record)
        monitoring["records"] = [staged_published[key] for key in sorted(staged_published)] + pending
        monitoring["episode"] = dict(
            switch=any(record.get("alarm") is True for record in staged_published.values()),
            model=any(record.get("model_required") is True for record in staged_published.values()),
            latent=any(record.get("latent_required") is True for record in staged_published.values()),
        )
        monitoring.setdefault("numerical_certification", _NUMERICAL_SCOPE)
        result = dict(
            initialization=staged_initialization, monitoring=monitoring, as_of=clock,
            numerical_certification=_NUMERICAL_SCOPE,
        )

        self._rows = staged_rows
        self._as_of = clock
        self._initialization = staged_initialization
        self._published = staged_published
        self._prefix_cache = staged_cache
        self._prepared_cache = staged_prepared
        self._last = result
        return deepcopy(result) if return_snapshot else None

    def snapshot(self):
        return deepcopy(self._last)

    def progress(self):
        """Return a detached small progress view without copying full records."""
        if self._last is None:
            return {"status": "empty", "as_of": self._as_of}
        initialization = self._last.get("initialization", {})
        monitoring = self._last.get("monitoring", {})
        records = monitoring.get("records", [])
        return {
            "status": self._last.get("monitoring", {}).get("status", initialization.get("status")),
            "as_of": self._last.get("as_of"),
            "initialization_status": initialization.get("status"),
            "N": initialization.get("N"), "T": initialization.get("T"),
            "eta_exact": initialization.get("eta_exact"),
            "availability": {str(row.get("window_index")): row.get("availability")
                             for row in records},
            "published_window_indices": [row.get("window_index") for row in records
                                          if row.get("availability") != "not_yet_released"],
            "publication_stop": deepcopy(monitoring.get("publication_stop")),
        }


def run_public_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-compatible entry for ``operation='eta_monitor_certified'``."""
    if request.get("operation") != "eta_monitor_certified":
        raise ValueError("eta_monitor_certified operation required")
    monitor_config = request.get("monitor")
    if not isinstance(monitor_config, Mapping):
        raise ValueError("monitor must be an object")
    source_execution_mode = monitor_config.get("source_execution_mode", "exhaustive")
    if source_execution_mode not in ("exhaustive", "lazy"):
        raise ValueError("source_execution_mode must be exhaustive or lazy")
    validate_source_arithmetic_rule(monitor_config.get("source_arithmetic_allowance_rule"))
    if "release_events" in request:
        session = PublicCertifiedEtaMonitor(request["reference"], request["monitor"])
        for event in request["release_events"]:
            session.update(event["public_rows"], as_of=event["as_of"])
        return {
            "operation": "eta_monitor_certified", "result": session.snapshot(),
            "implementation_status": "isolated_certified_candidate",
            "numerical_certification": _NUMERICAL_SCOPE,
        }
    rows = request["public_rows"]
    initialization = select_eta_certified(
        rows, request["reference"], as_of=request["as_of"],
    )
    if initialization["status"] == "ready":
        from .certified_monitor import monitor_history_certified
        monitoring = monitor_history_certified(
            rows, initialization, request["monitor"], request["reference"],
            as_of=request["as_of"],
        )
    else:
        monitoring = _not_ready_monitor()
    return {
        "operation": "eta_monitor_certified",
        "implementation_status": "isolated_certified_candidate",
        "initialization": initialization,
        "monitoring": monitoring,
        "numerical_certification": _NUMERICAL_SCOPE,
        "maintained_assumptions": [
            "predeclared tasks/domain/prefixes/windows/source family",
            "stable linear reference until deployment",
            "known conditional block-subGaussian bounds",
            "features/weights/source loadings predictable before whole-date innovation",
        ],
        "interpretation": "source compatibility/exclusion, not hidden-state recovery or economic action certification",
    }


__all__ = ["PublicCertifiedEtaMonitor", "run_public_request"]
