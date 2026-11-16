"""Deterministic RND2 scalar source rounding allowance.

This covers only finite binary64 scalar multiply/add/divide operations under
the declared source bounds.  It does not certify baseline, noise, or a whole
DGP, and it does not establish mechanism constancy or onset membership.
"""
from fractions import Fraction
import math
import sys


RULE = "rnd2_scalar_binary64"
_U = Fraction(1, 2 ** 53)
_H = Fraction(1, 2 ** 1075)
_GAMMA3 = 3 * _U / (1 - 3 * _U)
_MAX_FLOAT = Fraction.from_float(sys.float_info.max)


def validate_source_arithmetic_rule(rule):
    if rule not in (None, RULE):
        raise ValueError("source_arithmetic_allowance_rule must be None or %s" % RULE)
    return rule


def _operand(value, name):
    # JSON numbers arrive as Python int/float. Strings are deliberately not
    # accepted: arbitrary decimal strings are not a binary64 claim.
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError("%s must be a finite binary64 scalar" % name)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("%s must be a finite binary64 scalar" % name)
    try:
        out = Fraction.from_float(value) if isinstance(value, float) else Fraction(value)
    except (OverflowError, ValueError):
        raise ValueError("%s must be a finite binary64 scalar" % name)
    return out


def _binary64(value, name):
    out = _operand(value, name)
    try:
        represented = Fraction.from_float(float(value))
    except (OverflowError, ValueError):
        raise ValueError("%s must be exactly representable finite binary64" % name)
    if represented != out or not math.isfinite(float(value)):
        raise ValueError("%s must be exactly representable finite binary64" % name)
    return out


def scalar_source_error(a, ell, latent_bound, mechanism_bound):
    """Return exact e_i for one public action/loading pair."""
    a = _binary64(a, "a")
    ell = _binary64(ell, "ell")
    b = _operand(latent_bound, "latent_bound")
    k = _operand(mechanism_bound, "mechanism_bound")
    if b < 0 or k < 0:
        raise ValueError("source bounds must be nonnegative")
    if ell == 0:
        raise ValueError("ell must be nonzero for the declared RND2 rule")
    scale = b * abs(ell) + k * abs(a)
    # This also bounds the sum of the two rounded products before the final
    # addition, rather than checking only their unrounded magnitudes.
    if (1 + _U) * scale + 2 * _H > _MAX_FLOAT:
        raise ValueError("source arithmetic operands exceed finite binary64 applicability range")
    return _GAMMA3 * scale + (4 + abs(ell) / (1 - _U)) * _H


def source_arithmetic_allowance(weights, actions, loadings, *, latent_bound,
                                mechanism_bound, denominator):
    """Return exact E_j for signed row weights and a fixed denominator."""
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
        raise ValueError("denominator must be a positive integer")
    if len(actions) != len(loadings):
        raise ValueError("actions and loadings must have equal length")
    errors = [scalar_source_error(a, ell, latent_bound, mechanism_bound)
              for a, ell in zip(actions, loadings)]
    out = []
    for row in weights:
        if len(row) != len(errors):
            raise ValueError("weight rows must match source operands")
        out.append(sum((abs(_operand(w, "weight")) * e for w, e in zip(row, errors)), Fraction(0))
                   / denominator)
    return out


__all__ = ["RULE", "validate_source_arithmetic_rule", "scalar_source_error",
           "source_arithmetic_allowance"]
