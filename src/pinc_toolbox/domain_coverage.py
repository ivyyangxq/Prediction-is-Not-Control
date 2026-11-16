"""Exact deterministic coverage certificates for an axis-aligned domain.

This module is geometry only.  It does not inspect outcomes, spend an alpha
budget, choose a partition, or make a source/noncoverage claim.  A failed
certificate means only that the declared sufficient certificate did not pass.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
import math
from typing import Any, Sequence


def _q(value: Any, name: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite rational")
    try:
        if isinstance(value, Fraction):
            result = value
        elif isinstance(value, int):
            result = Fraction(value, 1)
        elif isinstance(value, str):
            result = Fraction(value.strip())
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError
            result = Fraction.from_float(number)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        raise ValueError(f"{name} must be a finite rational") from error
    return result


def _geometry(domain: Sequence[Sequence[Any]], scale: Sequence[Any], anchors: Sequence[Sequence[Any]]):
    if not isinstance(domain, (list, tuple)) or not domain:
        raise ValueError("domain must be a nonempty axis-aligned bounds list")
    bounds = []
    for index, pair in enumerate(domain):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("each domain coordinate must have lower and upper endpoints")
        low, high = _q(pair[0], f"domain[{index}].lower"), _q(pair[1], f"domain[{index}].upper")
        if high < low:
            raise ValueError("domain upper endpoint must not be below lower endpoint")
        bounds.append((low, high))
    dimension = len(bounds)
    scales = [_q(value, f"scale[{index}]") for index, value in enumerate(scale)]
    if len(scales) != dimension or any(value <= 0 for value in scales):
        raise ValueError("scale must be positive and match domain dimension")
    if not isinstance(anchors, (list, tuple)) or not anchors:
        raise ValueError("anchors must be a nonempty matrix")
    exact_anchors = []
    for row in anchors:
        if not isinstance(row, (list, tuple)) or len(row) != dimension:
            raise ValueError("anchors must be rectangular and match domain dimension")
        exact_anchors.append(tuple(_q(value, "anchor") for value in row))
    return tuple(bounds), tuple(scales), tuple(exact_anchors)


def _distance(bounds, scales, anchor):
    return max(max(abs(bounds[j][0] - anchor[j]), abs(bounds[j][1] - anchor[j])) / scales[j]
               for j in range(len(scales)))


def whole_domain_certificate(domain, scale, anchors):
    """Return the exact cheap single-anchor upper bound for the whole domain."""
    bounds, scales, exact_anchors = _geometry(domain, scale, anchors)
    values = [_distance(bounds, scales, anchor) for anchor in exact_anchors]
    index = min(range(len(values)), key=lambda i: values[i])
    return {
        "method": "whole_domain_single_anchor",
        "upper_bound_exact": str(values[index]),
        "anchor_index": index,
        "anchor_exact": [str(value) for value in exact_anchors[index]],
        "cells_checked": 1,
    }


def cell_certificate(domain, scale, anchors, *, subdivisions=4, max_cells=100000):
    """Return the exact fixed-grid cell upper bound, with auditable witnesses."""
    bounds, scales, exact_anchors = _geometry(domain, scale, anchors)
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError("subdivisions must be a positive integer")
    cell_count = subdivisions ** len(bounds)
    if cell_count > max_cells:
        raise ValueError("declared cell grid exceeds max_cells")
    endpoints = [
        [low + (high - low) * Fraction(index, subdivisions)
         for index in range(subdivisions + 1)]
        for low, high in bounds
    ]
    worst = Fraction(0)
    worst_cell = None
    worst_anchor = None
    for cell_index in product(range(subdivisions), repeat=len(bounds)):
        cell = tuple((endpoints[j][cell_index[j]], endpoints[j][cell_index[j] + 1])
                     for j in range(len(bounds)))
        values = [_distance(cell, scales, anchor) for anchor in exact_anchors]
        anchor_index = min(range(len(values)), key=lambda i: values[i])
        value = values[anchor_index]
        if worst_cell is None or value > worst:
            worst, worst_cell, worst_anchor = value, cell_index, anchor_index
    return {
        "method": f"fixed_{subdivisions}^{len(bounds)}_cell_grid",
        "upper_bound_exact": str(worst),
        "anchor_index": worst_anchor,
        "anchor_exact": [str(value) for value in exact_anchors[worst_anchor]],
        "cell_index": list(worst_cell),
        "cell_bounds_exact": [[str(low), str(high)] for low, high in
                               (tuple((endpoints[j][worst_cell[j]], endpoints[j][worst_cell[j] + 1])
                                      for j in range(len(bounds))))],
        "cells_checked": cell_count,
        "subdivisions": subdivisions,
    }


def certify_eta(domain, scale, anchors, eta, *, subdivisions=4, max_cells=100000):
    """Certify one eta using the whole-domain rule, then fixed cells if needed."""
    eta_exact = _q(eta, "eta")
    if eta_exact < 0:
        raise ValueError("eta must be nonnegative")
    whole = whole_domain_certificate(domain, scale, anchors)
    whole_bound = _q(whole["upper_bound_exact"], "whole upper bound")
    if whole_bound <= eta_exact:
        certificate = whole
    else:
        certificate = cell_certificate(domain, scale, anchors,
                                       subdivisions=subdivisions, max_cells=max_cells)
    bound = _q(certificate["upper_bound_exact"], "certificate upper bound")
    return {
        "eta_exact": str(eta_exact), "upper_bound_exact": str(bound),
        "enough": bound <= eta_exact, "certificate": certificate,
        "whole_domain_certificate": whole,
        "interpretation": "sufficient represented-domain geometry certificate; failure is not proof of noncoverage",
    }


def select_smallest_certified_eta(domain, scale, anchors, eta_grid, *,
                                  subdivisions=4, max_cells=100000):
    """Try candidates in ascending order and return the first sufficient certificate."""
    candidates = sorted({_q(eta, "eta_grid value") for eta in eta_grid})
    if not candidates:
        raise ValueError("eta_grid must be nonempty")
    if any(eta < 0 for eta in candidates):
        raise ValueError("eta_grid values must be nonnegative")
    cell_cache = None
    evaluations = []
    for eta in candidates:
        whole = whole_domain_certificate(domain, scale, anchors)
        if _q(whole["upper_bound_exact"], "whole upper bound") <= eta:
            result = {"eta_exact": str(eta), "upper_bound_exact": whole["upper_bound_exact"],
                      "enough": True, "certificate": whole}
        else:
            if cell_cache is None:
                cell_cache = cell_certificate(domain, scale, anchors,
                                              subdivisions=subdivisions, max_cells=max_cells)
            bound = _q(cell_cache["upper_bound_exact"], "cell upper bound")
            result = {"eta_exact": str(eta), "upper_bound_exact": str(bound),
                      "enough": bound <= eta, "certificate": cell_cache}
        evaluations.append(result)
        if result["enough"]:
            return {"status": "certified", "selected": result,
                    "evaluations": evaluations,
                    "cell_certificate_cached": cell_cache is not None,
                    "interpretation": "smallest ascending grid eta passing the declared sufficient certificate"}
    return {"status": "not_certified", "selected": None, "evaluations": evaluations,
            "cell_certificate_cached": cell_cache is not None,
            "interpretation": "failure is not proof of actual noncoverage"}


__all__ = ["whole_domain_certificate", "cell_certificate", "certify_eta",
           "select_smallest_certified_eta"]
