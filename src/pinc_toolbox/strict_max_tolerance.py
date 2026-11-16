"""Strict-maximum tolerance arithmetic; no simulation or verified coverage.

Unlike an ordinary rank test at risk_target, this mode alarms only above
every planned CAL score. The probability-of-bad-bank bound requires the
caller to establish the matching independent null score law.
"""
from fractions import Fraction
from numbers import Integral, Real
import math


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite real number')
    return float(value)


def _probability(value, name):
    value = _finite(value, name)
    if not 0 < value < 1:
        raise ValueError(name + ' must be in (0,1)')
    return value


def _scope(scope):
    if not isinstance(scope, dict) or set(scope) != {'kind', 'release_times'}:
        raise ValueError('scope requires exactly kind and release_times')
    kind = scope['kind']
    if kind not in ('single_look', 'whole_calendar'):
        raise ValueError('scope must be single_look or whole_calendar; no anytime scope')
    times = scope['release_times']
    if not isinstance(times, (list, tuple)) or not times:
        raise ValueError('scope must declare a nonempty release calendar')
    if any(isinstance(t, bool) or not isinstance(t, Integral) or t < 0 for t in times):
        raise ValueError('release_times must be nonnegative integer dates')
    times = [int(t) for t in times]
    if times != sorted(set(times)) or (kind == 'single_look' and len(times) != 1):
        raise ValueError('release_times must be strictly increasing; single_look has one date')
    return {'kind': kind, 'release_times': times}


def strict_max_tolerance(*, calibration_slots, planned_slot_count,
                         observed_score, observed_available, observed_status,
                         risk_target, cal_failure_target, scope):
    """Return a JSON-safe Stage-1 record without changing or drawing a bank.

    Each slot has slot (0..B-1), status, score_kind and score. Accepted
    slots require a finite score; cap_exhausted slots require score=None
    and score_kind='positive_infinity'. All planned slots are mandatory.
    Explicit software failures raise; they are not support refusals.

    risk_target is the fixed-bank tail-risk target, not a rank cutoff.
    cal_failure_target bounds the probability of a bad random bank.
    An insufficient B is reported, not enlarged or silently certified.
    The strict-max decision is still returned for an insufficient B.

    For whole_calendar, each supplied score must already be the maximum
    over the entire declared stopped null trajectory, including zero for
    no valid look. observed_available means that this maximum is defined,
    not that all windows completed. Single-look refusal encodings are not
    accepted in this scope. This function cannot
    verify that construction, independence, or equality of null laws.
    """
    if (isinstance(planned_slot_count, bool) or
            not isinstance(planned_slot_count, Integral) or planned_slot_count < 1):
        raise ValueError('planned_slot_count must be a positive integer')
    B = int(planned_slot_count)
    if not isinstance(calibration_slots, (list, tuple)) or len(calibration_slots) != B:
        raise ValueError('retain all planned CAL slots; length must equal planned_slot_count')
    risk = _probability(risk_target, 'risk_target')
    delta = _probability(cal_failure_target, 'cal_failure_target')
    declared_scope = _scope(scope)
    scores = []
    failed = 0
    for index, slot in enumerate(calibration_slots):
        if not isinstance(slot, dict):
            raise ValueError('each CAL slot must be a mapping')
        if slot.get('status') == 'software_failure':
            raise RuntimeError('CAL software failure stops tolerance evaluation')
        if not {'slot', 'status', 'score_kind', 'score'} <= slot.keys():
            raise ValueError('CAL slot requires slot, status, score_kind and explicit score')
        slot_id = slot.get('slot')
        if isinstance(slot_id, bool) or not isinstance(slot_id, Integral) or slot_id != index:
            raise ValueError('CAL slots must contain ordered, unique indices 0..B-1')
        status, kind = slot.get('status'), slot.get('score_kind')
        if status == 'accepted' and kind == 'finite':
            score = _finite(slot.get('score'), 'accepted CAL score')
            if declared_scope['kind'] == 'whole_calendar' and score < 0:
                raise ValueError('whole-calendar energy maximum must be nonnegative')
            scores.append(score)
        elif status == 'cap_exhausted' and kind == 'positive_infinity' and slot.get('score') is None:
            scores.append(math.inf)
            failed += 1
        else:
            raise ValueError('CAL slot must be accepted finite or explicitly cap_exhausted infinity')
    if observed_status == 'software_failure':
        raise RuntimeError('observed software failure stops tolerance evaluation')
    if type(observed_available) is not bool:
        raise ValueError('observed_available must be boolean')
    if declared_scope['kind'] == 'whole_calendar' and not observed_available:
        raise ValueError('whole-calendar support exits retain the prior maximum; no valid look uses zero')
    if observed_available:
        if observed_status != 'available':
            raise ValueError('available observation requires available status')
        observed = _finite(observed_score, 'observed score')
        if declared_scope['kind'] == 'whole_calendar' and observed < 0:
            raise ValueError('whole-calendar energy maximum must be nonnegative')
    else:
        if observed_status not in ('main_support_refusal', 'initialization_support_refusal'):
            raise ValueError('unavailable observation requires an explicit support-refusal status')
        if observed_score is not None:
            raise ValueError('unavailable observed score must be None')
        observed = None

    threshold = max(scores)
    reject = bool(observed_available and not failed and observed > threshold)
    reason = ('calibration_cap_exhausted' if failed else
              'observed_support_refusal' if not observed_available else None)
    # Use exact arithmetic on the declared decimal targets for the comparison.
    # A log bound remains informative if its displayed float underflows.
    bad_bank_bound = (1 - Fraction(str(risk))) ** B
    target_met = bad_bank_bound <= Fraction(str(delta))
    return {
        'mode': 'strict_max_tolerance',
        'status': 'unavailable' if reason else ('alarm' if reject else 'no_alarm'),
        'rejected': reject, 'reason': reason,
        'observed_score': observed, 'observed_available': observed_available,
        'observed_status': observed_status,
        'threshold': None if failed else threshold,
        'threshold_kind': 'positive_infinity' if failed else 'finite',
        'comparison': 'observed_score > maximum_of_all_planned_CAL_slots',
        'cal_total_B': B, 'cal_failed_count': failed,
        'risk_target': risk, 'cal_failure_target': delta,
        'bad_bank_probability_bound_under_assumptions': float(bad_bank_bound),
        'log_bad_bank_probability_bound_under_assumptions': B * math.log1p(-risk),
        'cal_confidence_target_met_by_B': target_met,
        'assumptions_status': 'not_verified',
        'statistical_coverage': 'not_established',
        'scope': declared_scope,
        'risk_event': ('single_look_alarm_conditional_on_main_availability'
                       if declared_scope['kind'] == 'single_look' else
                       'any_alarm_over_declared_whole_calendar'),
        'bound_statement': 'P_CAL(fixed-bank null risk > risk_target) <= (1-risk_target)^B',
        'required_assumptions': [
            'independent CAL score streams and an independent future null observation',
            'matching predeclared complete null scoring law and availability/stopping rule',
            'first-available sampling only for the stated single-look availability conditioning',
            'whole-calendar inputs are maxima of matching entire null trajectories, including support stops',
            'bank and score design chosen before CAL; no outcome-based bank selection',
        ],
        'scope_limits': ['not an absolute guarantee for every realized bank',
                         'not indefinite or anytime monitoring',
                         'not source calibration, source accuracy, or eta qualification'],
    }
