"""Lossless compact/restore codec for lazy certified-source evidence.

Only unevaluated CMIX candidate input arrays are compacted.  Evaluated
certificates, failures, class statuses, catalogue counts, and all provenance
remain inline.  The referenced arrays are reconstructed from the same backend
moments/radii, latent declaration, and matching CM declaration.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_MARKER = "_lazy_source_compact_encoding"
_FIELDS = ("loading_exact", "lower_exact", "upper_exact", "moments_exact", "radii_exact")


def _reference(label: str) -> dict[str, str]:
    return {"kind": "backend_latent_plus_matching_CM",
            "mechanism_candidate": label,
            "region": "backend.moments_exact_and_radii_exact"}


def _inputs(backend: Mapping[str, Any], pure: Mapping[str, Any]) -> dict[str, Any]:
    latent = backend["latent_loading_exact"]
    mechanism = pure["loading_exact"]
    if len(latent) != len(mechanism):
        raise ValueError("source row dimension mismatch")
    if len(pure["lower_exact"]) != len(pure["upper_exact"]):
        raise ValueError("source box dimension mismatch")
    if any(len(row) != len(pure["lower_exact"]) for row in mechanism):
        raise ValueError("mechanism columns mismatch")
    if any(len(row) != len(backend["latent_lower_exact"]) for row in latent):
        raise ValueError("latent columns mismatch")
    if len(backend["latent_lower_exact"]) != len(backend["latent_upper_exact"]):
        raise ValueError("latent box mismatch")
    return {"loading_exact": [left + right for left, right in zip(latent, mechanism)],
            "lower_exact": backend["latent_lower_exact"] + pure["lower_exact"],
            "upper_exact": backend["latent_upper_exact"] + pure["upper_exact"],
            "moments_exact": backend["moments_exact"],
            "radii_exact": backend["radii_exact"]}


def _compact_backend(backend: Mapping[str, Any]) -> dict[str, Any]:
    if _MARKER in backend:
        raise ValueError("payload already compacted")
    out = dict(backend)
    classes = out.get("classes")
    if not isinstance(classes, dict) or not isinstance(classes.get("CMIX"), dict):
        return out
    pure_rows = classes.get("CM", {}).get("candidates", [])
    pure = {row.get("candidate"): row for row in pure_rows}
    mixed = dict(classes["CMIX"])
    compacted = []
    count = 0
    for candidate in mixed.get("candidates", []):
        item = dict(candidate)
        if item.get("status") != "not_evaluated":
            compacted.append(item)
            continue
        label = item.get("candidate")
        if label not in pure or any(field not in item for field in _FIELDS):
            compacted.append(item)
            continue
        inputs = _inputs(out, pure[label])
        if any(item[field] != inputs[field] for field in _FIELDS):
            raise ValueError("unevaluated mixed inputs differ from source components")
        for field in _FIELDS:
            item.pop(field, None)
        item["source_input_reference"] = _reference(label)
        compacted.append(item)
        count += 1
    mixed["candidates"] = compacted
    out["classes"] = dict(classes, CMIX=mixed)
    out[_MARKER] = {"kind": "unevaluated_CMIX_inputs_from_same_backend",
                    "compacted_candidates": count}
    return out


def _restore_backend(backend: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(backend)
    marker = out.pop(_MARKER, None)
    if marker is None:
        return out
    if marker.get("kind") != "unevaluated_CMIX_inputs_from_same_backend":
        raise ValueError("unknown compact source encoding")
    classes = out.get("classes")
    if not isinstance(classes, dict) or not isinstance(classes.get("CMIX"), dict):
        raise ValueError("compact source encoding lacks CMIX class")
    pure_rows = classes.get("CM", {}).get("candidates", [])
    pure = {row.get("candidate"): row for row in pure_rows}
    restored = []
    count = 0
    for candidate in classes["CMIX"].get("candidates", []):
        item = dict(candidate)
        reference = item.pop("source_input_reference", None)
        if reference is None:
            restored.append(item)
            continue
        label = item.get("candidate")
        if reference != _reference(label) or item.get("status") != "not_evaluated":
            raise ValueError("invalid unevaluated CMIX source reference")
        if label not in pure or any(field in item for field in _FIELDS):
            raise ValueError("ambiguous or missing source reference")
        item.update(_inputs(out, pure[label]))
        restored.append(item)
        count += 1
    if marker.get("compacted_candidates") != count:
        raise ValueError("compact source reference count differs")
    out["classes"] = dict(classes, CMIX=dict(classes["CMIX"], candidates=restored))
    return out


def _walk(value: Any, operation):
    if isinstance(value, dict):
        if isinstance(value.get("classes"), dict) and "CMIX" in value["classes"]:
            return operation(value)
        return {key: _walk(item, operation) for key, item in value.items()}
    if isinstance(value, list):
        return [_walk(item, operation) for item in value]
    return value


def compact_source_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _walk(dict(payload), _compact_backend)


def restore_source_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _walk(dict(payload), _restore_backend)


__all__ = ["compact_source_payload", "restore_source_payload"]
