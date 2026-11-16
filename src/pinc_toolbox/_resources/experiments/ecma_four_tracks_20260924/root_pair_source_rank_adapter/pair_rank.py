"""Experimental arithmetic adapter, not a calibrated source procedure.

Callers must supply independent matched A-and-pair-complete candidate slots.
An absent slot stays absent. This module neither simulates nor selects slots.
"""
import math
from pinc_toolbox.stopped_candidate_rank import stopped_candidate_rank


def pair_contrast(report):
    """The existing main minus fixed-action discounted-return contrast."""
    if report is None:
        return None
    values = []
    for name in ('main', 'fixed_action'):
        branch = report['branches'][name]
        value = branch.get('discounted_return')
        if branch['status'] != 'complete' or isinstance(value, bool):
            return None
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        values.append(value)
    value = values[0] - values[1]
    return value if math.isfinite(value) else None


def rank_pair(*, observed_report, observed_alarm, calibration_reports,
              calibration_alarms, alpha=0.05):
    """Two tails of one existing contrast; no source posterior or law check."""
    if type(observed_alarm) is not bool:
        raise TypeError('observed_alarm must be a boolean')
    if len(calibration_reports) != len(calibration_alarms):
        raise ValueError('retain one flag per planned calibration slot')
    if any(type(a) is not bool for a in calibration_alarms):
        raise TypeError('calibration alarms must be booleans')
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0,1)')
    observed = pair_contrast(observed_report) if observed_alarm else None
    scores = [pair_contrast(r) if a else None
              for r, a in zip(calibration_reports, calibration_alarms)]
    valid = [s is not None for s in scores]
    tails = []
    for sign in (1, -1):
        tails.append(stopped_candidate_rank(
            observed_score=None if observed is None else sign * observed,
            observed_full_window_complete=observed is not None,
            observed_score_valid_for_rank=observed is not None,
            calibration_scores=[None if s is None else sign * s for s in scores],
            calibration_full_window_complete=valid,
            calibration_score_valid_for_rank=valid, alpha=alpha / 2))
    return dict(
        status=('excluded' if any(t['rejected'] for t in tails) else 'compatible')
            if observed is not None else 'unresolved',
        contrast=observed,
        p_value=min(1., 2 * min(t['p'] for t in tails)) if observed is not None else None,
        upper=tails[0], lower=tails[1],
        cal_slots=len(scores), cal_failed=sum(not v for v in valid),
        calibration_assumptions='not_verified_by_arithmetic_adapter',
        scope='one candidate point, conditional matched selection; not a source probability',
        reason='no_usable_observed_alarm_pair' if observed is None else
            ('incomplete_calibration' if not all(valid) else None))
