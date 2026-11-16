"""Lazy, per-class exact source evaluation workflow.

``CertifiedSourceEvaluator`` validates and freezes a source specification in
its constructor without solving anything.  Calling one of ``C0``, ``CL``,
``CM`` or ``CMIX`` evaluates only that class, retains the complete candidate
evidence, and caches the result.  This provider is intentionally separate from
the eager ``diagnose_certified_sources`` adapter; a controller can call it
through a cached router while retaining the raw class evidence here.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
import time
from typing import Any

from . import certified_source_backend as _backend


_CLASSES = ("C0", "CL", "CM", "CMIX")


def _defining_inputs(
    moments: list[Fraction],
    radii: list[Fraction],
    latent: list[list[Fraction]],
    latent_lower: list[Fraction],
    latent_upper: list[Fraction],
    mechanisms: list[tuple[str, list[list[Fraction]], list[Fraction], list[Fraction]]],
) -> dict[str, Any]:
    return {
        "moments_exact": _backend._exact(moments),
        "radii_exact": _backend._exact(radii),
        "latent_loading_exact": _backend._exact(latent),
        "latent_lower_exact": _backend._exact(latent_lower),
        "latent_upper_exact": _backend._exact(latent_upper),
        "mechanisms_exact": {
            name: {
                "loading_exact": _backend._exact(loading),
                "lower_exact": _backend._exact(lower),
                "upper_exact": _backend._exact(upper),
            }
            for name, loading, lower, upper in mechanisms
        },
    }


class CertifiedSourceEvaluator:
    """Validate an exact source specification and evaluate classes lazily."""

    def __init__(self, source_spec: Mapping[str, Any]):
        if not isinstance(source_spec, Mapping):
            raise _backend.CertifiedSourceBackendError("source_spec must be a mapping")
        frozen = deepcopy(dict(source_spec))
        required = {
            "moments", "radii", "latent_loading", "latent_bounds",
            "mechanism_loadings", "mechanism_bounds",
        }
        missing = sorted(required.difference(frozen))
        if missing:
            raise _backend.CertifiedSourceBackendError(
                f"source_spec is missing required keys: {missing!r}"
            )
        unknown = sorted(set(frozen).difference(required | {"solver_options"}))
        if unknown:
            raise _backend.CertifiedSourceBackendError(
                f"source_spec has unknown keys: {unknown!r}"
            )

        moments = _backend._vector(frozen["moments"], name="moments", allow_empty=False)
        radii = _backend._vector(frozen["radii"], name="radii")
        if len(radii) != len(moments):
            raise _backend.CertifiedSourceBackendError(
                "moments and radii must have the same dimension"
            )
        if any(value < 0 for value in radii):
            raise _backend.CertifiedSourceBackendError("radii must be nonnegative")
        rows = len(moments)
        latent = _backend._matrix(
            frozen["latent_loading"], name="latent_loading", rows=rows
        )
        latent_dimension = len(latent[0]) if latent else 0
        latent_lower, latent_upper = _backend._box(
            frozen["latent_bounds"], name="latent", dimension=latent_dimension
        )
        mechanism_values = frozen["mechanism_loadings"]
        if not isinstance(mechanism_values, Mapping) or not mechanism_values:
            raise _backend.CertifiedSourceBackendError(
                "mechanism_loadings must be a nonempty candidate mapping"
            )
        bound_values = frozen["mechanism_bounds"]
        if isinstance(bound_values, Mapping):
            missing_bounds = [name for name in mechanism_values if name not in bound_values]
            if missing_bounds:
                raise _backend.CertifiedSourceBackendError(
                    f"missing mechanism bounds for candidates {missing_bounds!r}"
                )
            bounds_for = bound_values
        else:
            bounds_for = None

        mechanisms = []
        for name, loading_value in mechanism_values.items():
            if not isinstance(name, str) or not name:
                raise _backend.CertifiedSourceBackendError(
                    "mechanism candidate names must be nonempty strings"
                )
            loading = _backend._matrix(
                loading_value,
                name=f"mechanism_loadings[{name!r}]",
                rows=rows,
            )
            dimension = len(loading[0]) if loading else 0
            bound_value = bounds_for[name] if bounds_for is not None else bound_values
            lower, upper = _backend._box(
                bound_value, name=f"mechanism {name!r}", dimension=dimension
            )
            mechanisms.append((name, loading, lower, upper))

        solver_options = frozen.get("solver_options")
        if solver_options is not None and not isinstance(solver_options, Mapping):
            raise _backend.CertifiedSourceBackendError(
                "solver_options must be a mapping or None"
            )
        self._source_spec = frozen
        self._moments = moments
        self._radii = radii
        self._latent = latent
        self._latent_lower = latent_lower
        self._latent_upper = latent_upper
        self._mechanisms = mechanisms
        self._solver_options = deepcopy(solver_options) if solver_options is not None else None
        self._inputs = _defining_inputs(
            moments, radii, latent, latent_lower, latent_upper, mechanisms
        )
        self._cache: dict[str, dict[str, Any]] = {}

    def _candidate(
        self,
        label: str,
        loading: list[list[Fraction]],
        lower: list[Fraction],
        upper: list[Fraction],
    ) -> tuple[dict[str, Any], int]:
        return _backend._candidate(
            label=label,
            moments=self._moments,
            radii=self._radii,
            loading=loading,
            lower=lower,
            upper=upper,
            solver_options=self._solver_options,
        )

    def _finish(self, class_name: str, result: dict[str, Any], started: float, calls: int):
        elapsed = time.monotonic() - started
        if not _backend.isfinite(elapsed):
            elapsed = 0.0
        result = dict(result)
        result["class_name"] = class_name
        result["defining_inputs"] = deepcopy(self._inputs)
        result["solver_calls"] = calls
        result["elapsed_seconds"] = elapsed
        result["numerical_certification"] = (
            "exact rational certification of the supplied represented source region only; "
            "upstream statistical construction and continuation values are outside scope"
        )
        self._cache[class_name] = result
        return deepcopy(result)

    def _evaluate_c0(self):
        started = time.monotonic()
        zero_loading = [[] for _ in self._moments]
        c0 = _backend.raw_box_certificate(
            self._moments, self._radii, zero_loading, [], [], primal=[], threshold=1
        )
        reason = "exact_empty_loading_zero_witness"
        if c0["status"] != "compatible":
            coordinate = next(
                index for index, (moment, radius) in enumerate(zip(self._moments, self._radii))
                if abs(moment) > radius
            )
            direction = [Fraction(0)] * len(self._moments)
            direction[coordinate] = Fraction(1) if self._moments[coordinate] > 0 else Fraction(-1)
            c0 = _backend.raw_box_certificate(
                self._moments, self._radii, zero_loading, [], [],
                dual=direction, threshold=1,
            )
            reason = "coordinate_separating_direction"
        candidate = {
            "candidate": "zero",
            "status": c0["status"],
            "description": _backend._candidate_description("zero", c0["status"]),
            "reason": reason,
            "search": {"status": "not_run", "reason": "C0 uses exact checking without LP"},
            "certificate": c0,
            "moments_exact": _backend._exact(self._moments),
            "radii_exact": _backend._exact(self._radii),
            "loading_exact": _backend._exact(zero_loading),
            "lower_exact": [],
            "upper_exact": [],
            "solver_calls": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
        return {
            "status": candidate["status"],
            "description": _backend._class_description("C0", candidate["status"]),
            "reason": reason,
            "selected_candidate": "zero" if candidate["status"] == "compatible" else None,
            "candidates": [candidate],
        }, 0

    def _evaluate(self, class_name: str) -> dict[str, Any]:
        started = time.monotonic()
        calls = 0
        if class_name == "C0":
            result, calls = self._evaluate_c0()
            return self._finish(class_name, result, started, calls)
        if class_name == "CL":
            candidate, calls = self._candidate(
                "latent", self._latent, self._latent_lower, self._latent_upper
            )
            result = _backend._class_result("CL", [candidate])
            return self._finish(class_name, result, started, calls)
        if class_name == "CM":
            candidates = []
            for name, loading, lower, upper in self._mechanisms:
                candidate, candidate_calls = self._candidate(name, loading, lower, upper)
                calls += candidate_calls
                candidates.append(candidate)
            result = _backend._class_result("CM", candidates)
            return self._finish(class_name, result, started, calls)
        if class_name == "CMIX":
            candidates = []
            for name, loading, lower, upper in self._mechanisms:
                mixed_loading = [
                    latent_row + mechanism_row
                    for latent_row, mechanism_row in zip(self._latent, loading)
                ]
                candidate, candidate_calls = self._candidate(
                    name,
                    mixed_loading,
                    self._latent_lower + lower,
                    self._latent_upper + upper,
                )
                calls += candidate_calls
                candidates.append(candidate)
            result = _backend._class_result("CMIX", candidates)
            return self._finish(class_name, result, started, calls)
        raise ValueError(f"unknown source class {class_name!r}")

    def __call__(self, class_name: str) -> dict[str, Any]:
        if class_name not in _CLASSES:
            raise ValueError(f"unknown source class {class_name!r}")
        if class_name not in self._cache:
            self._evaluate(class_name)
        return deepcopy(self._cache[class_name])

    def snapshot(self) -> dict[str, Any]:
        """Return an independent copy of defining inputs and class cache."""
        return {
            "defining_inputs": deepcopy(self._inputs),
            "computed_classes": list(self._cache),
            "classes": deepcopy(self._cache),
        }

    @classmethod
    def from_request(cls, source_spec: Mapping[str, Any]):
        """Construct an evaluator from a caller-supplied source specification."""
        return cls(source_spec)


def certified_source_workflow_documentation() -> dict[str, Any]:
    """Describe the lazy provider contract without importing numerical code."""
    return {
        "classes": list(_CLASSES),
        "constructor": "CertifiedSourceEvaluator(source_spec)",
        "call": "evaluator(class_name)",
        "lazy": "constructor validates exact arrays; only the selected class searches",
        "cache": "one immutable deep-copied result per class",
        "scope": "exact represented supplied source region only; no Q or e_eta",
    }


__all__ = [
    "CertifiedSourceEvaluator",
    "certified_source_workflow_documentation",
]
