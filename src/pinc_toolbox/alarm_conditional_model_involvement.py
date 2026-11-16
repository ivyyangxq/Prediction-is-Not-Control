"""Experimental alarm-conditional finite-catalogue model-involvement rank.

This interface deliberately requires an explicit alarm state. It reuses the
fixed-entry ordered 39/128 bank collector, but applies a two-sided rank to a
signed score and does not inherit fixed-entry one-sided scope or conclusions.
"""
from copy import deepcopy
from fractions import Fraction
import math

from .fixed_entry_model_involvement import (
    ALPHA as DEFAULT_ALPHA,
    CAL_TARGET,
    NO_MODEL_CANDIDATES,
    PROPOSAL_CAP,
    _calibration_bank,
)


def _finite_score(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _standardize_proposal(record):
    """Adapt signed_sum records to the shared ordered-bank score schema."""
    item = deepcopy(record)
    if item.get('status') == 'complete':
        if 'score' not in item and 'signed_sum' in item:
            item['score'] = item['signed_sum']
    elif item.get('status') == 'support_failure':
        item.setdefault('complete', False)
    elif item.get('status') in ('software_error', 'numerical_error', 'other_error', 'not_run'):
        item.setdefault('complete', False)
    # Unknown/inconsistent status is deliberately left for _calibration_bank
    # to preserve as an invalid proposal with its failure reason.
    return item


def _two_sided_rank(observed_score, calibration_scores):
    """Inclusive tie-aware two-tail ranks with the finite-sample +1 rule."""
    n = len(calibration_scores)
    if n == 0:
        raise ValueError('at least one calibration score is required')
    lo_count = 1 + sum(float(score) <= observed_score for score in calibration_scores)
    hi_count = 1 + sum(float(score) >= observed_score for score in calibration_scores)
    denominator = n + 1
    lower = Fraction(lo_count, denominator)
    upper = Fraction(hi_count, denominator)
    p = min(Fraction(1, 1), 2 * min(lower, upper))
    return {
        'lower_tail_count': lo_count,
        'upper_tail_count': hi_count,
        'denominator': denominator,
        'lower_p': float(lower),
        'upper_p': float(upper),
        'p_two_sided': float(p),
        'ties_included_in_both_tails': True,
    }


def alarm_conditional_model_involvement(
    *, alarm_status, calibration_proposals_by_candidate, observed,
    alpha=DEFAULT_ALPHA, no_model_candidates=NO_MODEL_CANDIDATES,
    calibration_bank_size=CAL_TARGET, proposal_cap=PROPOSAL_CAP,
):
    """Return only ``model_involvement`` or ``unresolved``.

    `no_model_candidates` explicitly declares the finite no-model catalogue;
    its default is N/CL. It does not estimate or certify candidate laws.
    Missing banks remain unavailable and never silently remove a candidate.

    `alarm_status` must be ``no_alarm``, ``not_run``, ``support_unavailable``, or ``triggered``. For a
    triggered route, N and CL records are ordered proposal mappings using
    ``status`` (complete/support_failure/software_error/numerical_error/
    other_error/not_run) plus ``signed_sum`` for completed records. Up to 128
    proposals are examined per candidate; the first 39 complete finite scores
    are selected. All input records and failure details are returned unchanged
    under `calibration_inputs` and `observed_input`.
    """
    if (not isinstance(no_model_candidates, (list, tuple)) or not no_model_candidates
            or any(not isinstance(c, str) or not c.strip() for c in no_model_candidates)
            or len(set(no_model_candidates)) != len(no_model_candidates)):
        raise ValueError('no_model_candidates must be a nonempty ordered list or tuple of distinct names')
    candidates = tuple(no_model_candidates)
    if alarm_status not in ('no_alarm', 'not_run', 'support_unavailable', 'triggered'):
        raise ValueError("alarm_status must be 'no_alarm', 'not_run', 'support_unavailable', or 'triggered'")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be finite and in (0, 1)')
    if not isinstance(calibration_proposals_by_candidate, dict):
        raise ValueError('calibration_proposals_by_candidate must be a mapping')
    if not isinstance(observed, dict):
        raise ValueError('observed must be a mapping')

    raw_inputs = {
        candidate: deepcopy(calibration_proposals_by_candidate.get(candidate, ()))
        for candidate in candidates
    }
    result = {
        'scope': 'alarm_conditional_signed_rank_model_involvement',
        'alarm_status': alarm_status,
        'no_model_candidates': list(candidates),
        'decision': 'unresolved',
        'unresolved_category': None,
        'rank_by_candidate': {},
        'calibration_banks': {},
        'calibration_inputs': raw_inputs,
        'observed_input': deepcopy(observed),
        'observed_signed_sum': None,
        'alpha': float(alpha),
        'latent': 'unresolved',
        'eta_status': 'not_supplied',
        'source_probability': None,
        'posterior_probability': None,
        'conditional_assumptions_verified': False,
        'assumptions': 'correct candidate-specific conditional kernels and common qualification/completion rules are required but unverified',
        'guarantee': 'no empirical error, power, or global monitoring guarantee established',
    }
    if alarm_status == 'no_alarm':
        result['unresolved_category'] = 'no_alarm'
        result['reason'] = 'conditional source rank is not run without an alarm'
        result['calibration_banks'] = {c: {'status': 'not_requested'} for c in candidates}
        return result
    if alarm_status == 'support_unavailable':
        result['unresolved_category'] = 'monitor_support_unavailable'
        result['reason'] = 'monitor support is unavailable; absence of an alarm is not established'
        result['calibration_banks'] = {c: {'status': 'not_requested'} for c in candidates}
        return result

    if alarm_status == 'not_run':
        result['unresolved_category'] = 'alarm_not_run'
        result['reason'] = 'alarm status is not run; conditional source rank is unavailable'
        result['calibration_banks'] = {c: {'status': 'not_requested'} for c in candidates}
        return result

    if any(not isinstance(raw_inputs[c], (list, tuple)) for c in candidates):
        raise ValueError('each candidate calibration input must be an ordered list or tuple')
    banks = {}
    if isinstance(calibration_bank_size, bool) or not isinstance(calibration_bank_size, int) or calibration_bank_size <= 0:
        raise ValueError('calibration_bank_size must be a positive integer')
    if isinstance(proposal_cap, bool) or not isinstance(proposal_cap, int) or proposal_cap <= 0:
        raise ValueError('proposal_cap must be a positive integer')
    for candidate in candidates:
        standardized = [_standardize_proposal(record) if isinstance(record, dict) else record
                       for record in raw_inputs[candidate]]
        banks[candidate] = _calibration_bank(standardized, candidate,
                                              target=calibration_bank_size,
                                              proposal_cap=proposal_cap)
    result['calibration_banks'] = banks

    obs = deepcopy(observed)
    obs_status = obs.get('status')
    obs_complete = obs.get('complete', obs.get('complete_A'))
    obs_score = obs.get('signed_sum', obs.get('score'))
    valid_observed = obs_status == 'complete' and obs_complete is True and _finite_score(obs_score)
    if not valid_observed:
        result['unresolved_category'] = (
            'support_failure' if obs_status == 'support_failure' else
            'other_error' if obs_status in ('software_error', 'numerical_error', 'other_error') else
            'observed_not_rankable'
        )
        result['reason'] = obs.get('reason') or obs.get('reasons') or obs.get('failure') or 'observed signed score incomplete or invalid'
        return result
    result['observed_signed_sum'] = float(obs_score)

    for candidate in candidates:
        bank = banks[candidate]
        if bank['status'] != 'complete':
            continue
        rank = _two_sided_rank(float(obs_score), bank['selected_scores'])
        rank.update({'observed_signed_sum': float(obs_score),
                     'calibration_count': len(bank['selected_scores']),
                     'rejected': rank['p_two_sided'] <= float(alpha)})
        result['rank_by_candidate'][candidate] = rank
    if len(result['rank_by_candidate']) != len(candidates):
        result['unresolved_category'] = 'calibration_unavailable'
        result['reason'] = 'one or more ordered conditional calibration banks are not complete'
    elif all(result['rank_by_candidate'][c]['rejected'] for c in candidates):
        result['decision'] = 'model_involvement'
    else:
        result['unresolved_category'] = 'complete_non_rejection'
    return result
