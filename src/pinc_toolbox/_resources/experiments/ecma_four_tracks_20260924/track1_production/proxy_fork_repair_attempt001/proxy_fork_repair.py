"""Opt-in deepcopy repair for RND2's first-proxy view alias.

This module is intentionally external to the historical DGP sources. Load a
fresh environment module, then call :func:`install_for_module` before making
instances. The installed hook preserves only the verified RND2 relationship
``w is proxy_all[:, :, 0]`` when the original arrays share that exact view.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np


_RND2_ROOT = Path(__file__).resolve().parents[4] / "DGP" / "RND" / "02_state_dependent_local"


def _is_verified_first_proxy_view(environment: Any) -> bool:
    """Check alias identity by memory/layout, never by equal values alone."""
    proxy = getattr(environment, "proxy_all", None)
    w = getattr(environment, "w", None)
    if not isinstance(proxy, np.ndarray) or not isinstance(w, np.ndarray):
        return False
    if proxy.ndim != 3 or proxy.shape[2] < 1 or w.shape != proxy.shape[:2]:
        return False
    expected = proxy[:, :, 0]
    return (
        np.shares_memory(w, proxy)
        and w.dtype == expected.dtype
        and w.shape == expected.shape
        and w.strides == expected.strides
        and w.__array_interface__["data"][0] == expected.__array_interface__["data"][0]
    )


def _paired_values(original: Any, cloned: Any) -> Iterable[tuple[Any, Any]]:
    """Walk corresponding object graphs with cycle/shared-node protection."""
    pending = [(original, cloned)]
    visited: set[tuple[int, int]] = set()
    while pending:
        left, right = pending.pop()
        pair = (id(left), id(right))
        if pair in visited:
            continue
        visited.add(pair)
        yield left, right
        if isinstance(left, dict) and isinstance(right, dict):
            for key in left.keys() & right.keys():
                pending.append((left[key], right[key]))
        elif isinstance(left, (list, tuple)) and isinstance(right, type(left)):
            pending.extend(zip(left, right))
        elif isinstance(left, set) and isinstance(right, set):
            # Environment graphs do not use sets for paired child objects.
            continue
        elif type(left) is type(right) and hasattr(left, "__dict__"):
            for name, value in vars(left).items():
                if name in vars(right):
                    pending.append((value, vars(right)[name]))


def _restore_verified_aliases(original: Any, cloned: Any) -> None:
    for left, right in _paired_values(original, cloned):
        if _is_verified_first_proxy_view(left):
            # This creates a view into the clone's own proxy storage.
            right.w = right.proxy_all[:, :, 0]


def _deepcopy_environment(self: Any, memo: dict[int, Any]) -> Any:
    existing = memo.get(id(self))
    if existing is not None:
        return existing
    cloned = type(self).__new__(type(self))
    memo[id(self)] = cloned
    if hasattr(self, "__dict__"):
        for name, value in vars(self).items():
            setattr(cloned, name, copy.deepcopy(value, memo))
    # Preserve slots too, for derived environment classes that add them.
    for cls in type(self).__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in {"__dict__", "__weakref__"} or not hasattr(self, name):
                continue
            setattr(cloned, name, copy.deepcopy(getattr(self, name), memo))
    _restore_verified_aliases(self, cloned)
    return cloned


def repair_environment_class(cls: type) -> type:
    """Install memo-aware deepcopy on one freshly loaded environment class."""
    cls.__deepcopy__ = _deepcopy_environment
    return cls


def install_for_module(module: ModuleType, *, recursive: bool = True) -> ModuleType:
    """Patch Environment classes in a loaded module and nested source modules.

    The RND2 wrappers load their dependencies into globals named ``_PARENT``,
    ``_MODEL``, ``_AUX`` and ``_MOTHER``. Only ``environment.py`` modules
    whose resolved paths remain inside this project's RND2 directory qualify.
    """
    seen: set[int] = set()

    def visit(current: ModuleType) -> None:
        if id(current) in seen:
            return
        seen.add(id(current))
        module_file = getattr(current, "__file__", None)
        if module_file is None:
            return
        resolved_file = Path(module_file).resolve()
        try:
            resolved_file.relative_to(_RND2_ROOT)
        except ValueError:
            return
        if resolved_file.name != "environment.py":
            return
        cls = getattr(current, "Environment", None)
        if isinstance(cls, type):
            repair_environment_class(cls)
        if recursive:
            # Follow only the explicit RND2 source-lineage imports. Walking all
            # module globals would also walk sys, NumPy and unrelated modules.
            for name in ("_PARENT", "_MODEL", "_AUX", "_MOTHER"):
                value = getattr(current, name, None)
                if isinstance(value, ModuleType):
                    visit(value)

    visit(module)
    return module


def load_patched_environment(path: str | Path, module_name: str) -> ModuleType:
    """Load one environment file under an isolated name and patch its lineage."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load environment module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return install_for_module(module)


__all__ = ["repair_environment_class", "install_for_module", "load_patched_environment"]
