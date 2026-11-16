"""Explain the rejection floor from a fixed CAL budget and failed CAL slots."""
from fractions import Fraction
from numbers import Integral
import math


def calibration_rejection_capacity(*, cal_total_B, cal_failed_count, alpha):
    """Failure-count lower bound on rank p; does not assert attainability or power.

    Failed CAL scores count as +infinity. Even above every finite CAL score,
    the observed rank p cannot fall below (1+failed)/(B+1). Equality with alpha
    permits rejection; the comparison is exact for the declared decimal alpha.
    """
    for name,value in [('cal_total_B',cal_total_B),('cal_failed_count',cal_failed_count)]:
        if isinstance(value,bool) or not isinstance(value,Integral):
            raise ValueError(name+' must be an integer')
    B=int(cal_total_B);failed=int(cal_failed_count)
    if B<1 or failed<0 or failed>B:
        raise ValueError('require B>=1 and 0<=failed<=B')
    if isinstance(alpha,bool) or not math.isfinite(alpha) or not 0<alpha<1:
        raise ValueError('alpha must be in (0,1)')
    floor=Fraction(1+failed,B+1)
    return {'cal_total_B':B,'cal_failed_count':failed,
            'calibration_failure_p_floor':float(floor),
            'calibration_failure_p_floor_numerator':1+failed,
            'calibration_failure_p_floor_denominator':B+1,
            'calibration_allows_rejection':floor<=Fraction(str(alpha)),
            'calibration_capacity_explanation':
                'Failed CAL slots count as +infinity and remain in original B. '
                'This is a failure-count lower bound on rank p; the statistic range and finite CAL ties may force a higher p. '
                'False means rejection is impossible with this CAL record; retention is not evidence of fit. '
                'True means the failure floor alone does not prohibit rejection; it asserts neither an attainable rejecting score, power nor observed completion.'}
