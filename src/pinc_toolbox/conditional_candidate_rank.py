"""Conservative finite-Monte-Carlo rank arithmetic for saved candidate scores.

No coverage is certified here: conditional exchangeability and matching score
construction remain responsibilities of the measurement route.
"""
import math
from fractions import Fraction
from .calibration_rejection_capacity import calibration_rejection_capacity


def conditional_candidate_rank(*, observed_score, observed_complete,
                               calibration_scores, calibration_complete,
                               alpha):
    """Keep the planned CAL denominator; missing/failed CAL contributes +inf.

    Every planned CAL replicate must have one entry. Incomplete observation
    never becomes a retained witness or an exclusion, even if its rank is one.
    ``None`` is the strict-JSON representation of unavailable scores.
    """
    if not isinstance(observed_complete, bool):
        raise ValueError('observed_complete must be boolean')
    if not isinstance(calibration_scores, (list, tuple)) or not calibration_scores:
        raise ValueError('calibration_scores must contain every planned replicate')
    if len(calibration_complete) != len(calibration_scores) or any(
            not isinstance(x, bool) for x in calibration_complete):
        raise ValueError('calibration_complete must match every planned replicate')
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')

    def finite(value):
        return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)

    failed = [not ok or not finite(score)
              for score, ok in zip(calibration_scores, calibration_complete)]
    B = len(calibration_scores)
    valid = observed_complete and finite(observed_score)
    ge = (sum(bad or score >= observed_score for score, bad in zip(calibration_scores, failed))
          if valid else B)
    p = (1 + ge) / (B + 1)
    rejected = bool(valid and Fraction(1 + ge, B + 1) <= Fraction(str(alpha)))
    return {**calibration_rejection_capacity(cal_total_B=B,cal_failed_count=sum(failed),alpha=alpha),
            'status': ('excluded' if rejected else 'compatible') if valid else 'unresolved',
            'description': 'conditional candidate rank arithmetic; assumptions not verified',
            'observed_score': float(observed_score) if finite(observed_score) else None,
            'observed_complete': bool(valid), 'cal_total_B': B,
            'cal_ge_observed_count': ge if valid else None,
            'cal_failed_count': sum(failed), 'rank_numerator': 1 + ge,
            'rank_denominator': B + 1, 'p': p, 'alpha': alpha,
            'rejected': rejected, 'conditional_bound': None,
            'reason': None if valid else 'observed_incomplete_or_nonfinite'}
