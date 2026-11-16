"""Explicit stopped-score ranks; completion and rank validity are distinct."""
import math
from fractions import Fraction
from .calibration_rejection_capacity import calibration_rejection_capacity


def stopped_candidate_rank(*, observed_score, observed_full_window_complete,
        observed_score_valid_for_rank, calibration_scores,
        calibration_full_window_complete, calibration_score_valid_for_rank, alpha):
    B = len(calibration_scores)
    flags = [observed_full_window_complete, observed_score_valid_for_rank]
    flags += list(calibration_full_window_complete) + list(calibration_score_valid_for_rank)
    if not B or len(calibration_full_window_complete) != B or len(calibration_score_valid_for_rank) != B:
        raise ValueError('retain every planned CAL slot and both flags')
    if any(type(x) is not bool for x in flags):
        raise ValueError('completion and rank validity must be booleans')
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    def finite(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    valid = [ok and finite(s) for ok, s in zip(calibration_score_valid_for_rank, calibration_scores)]
    obs_rank_valid = observed_score_valid_for_rank and finite(observed_score)
    usable = observed_full_window_complete and obs_rank_valid
    failed = B - sum(valid)
    ge = sum(not ok or s >= observed_score for ok, s in zip(valid, calibration_scores)) if usable else B
    reject = bool(usable and Fraction(1 + ge, B + 1) <= Fraction(str(alpha)))
    return {**calibration_rejection_capacity(cal_total_B=B, cal_failed_count=failed, alpha=alpha),
        'score_policy': 'stopped_prefix_observed_complete',
        'status': ('excluded' if reject else 'compatible') if usable else 'unresolved',
        'observed_score': float(observed_score) if finite(observed_score) else None,
        'observed_complete': observed_full_window_complete,
        'observed_full_window_complete': observed_full_window_complete,
        'observed_score_valid_for_rank': bool(obs_rank_valid),
        'calibration_full_window_complete': list(calibration_full_window_complete),
        'calibration_score_valid_for_rank': valid,
        'cal_total_B': B, 'cal_failed_count': failed,
        'cal_ge_observed_count': ge if usable else None,
        'rank_numerator': 1 + ge, 'rank_denominator': B + 1,
        'p': (1 + ge) / (B + 1), 'alpha': alpha, 'rejected': reject,
        'conditional_bound': None,
        'description': 'common stopped-score conditional rank; assumptions not verified',
        'reason': None if usable else 'observed_incomplete_or_nonfinite'}
