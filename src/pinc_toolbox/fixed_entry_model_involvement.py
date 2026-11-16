"""Fixed-entry, finite-catalogue model-involvement rank decision.

This is separate from alarm-conditional source inference. Calibration proposals are
consumed in their fixed order; only complete finite paired-A scores enter each
39-item bank. An observed path is called model-involving only when both no-model
candidate ranks reject at alpha=.05. Every other outcome is unresolved.
"""
import math
from .stopped_candidate_rank import stopped_candidate_rank

CAL_TARGET = 39
ALPHA = 0.05
PROPOSAL_CAP = 128
NO_MODEL_CANDIDATES = ('N', 'CL')


def _finite_score(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _reason(record):
    reason = record.get('reason')
    if reason is not None:
        return str(reason)
    failure = record.get('failure')
    if isinstance(failure, dict):
        return failure.get('reason') or failure.get('type')
    return None


def _calibration_bank(proposals, candidate, *, target=CAL_TARGET, proposal_cap=PROPOSAL_CAP):
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValueError('target must be a positive integer')
    if isinstance(proposal_cap, bool) or not isinstance(proposal_cap, int) or proposal_cap <= 0:
        raise ValueError('proposal_cap must be a positive integer')
    if not isinstance(proposals, (list, tuple)):
        raise ValueError(f'{candidate} calibration proposals must be ordered')
    received = len(proposals)
    selected = []
    outcomes = []
    support_failures = []
    other_errors = []
    invalid_reason = None
    for index, record in enumerate(proposals[:proposal_cap]):
        if not isinstance(record, dict):
            raise ValueError(f'{candidate} proposal {index} must be a mapping')
        status = record.get('status')
        complete = record.get('complete', record.get('complete_A'))
        score = record.get('score', record.get('A'))
        if status == 'complete' and complete is True:
            if not _finite_score(score):
                invalid_reason = f'proposal_{index}_complete_score_nonfinite_or_missing'
                outcomes.append(dict(index=index, status='other_error', accepted=False,
                                     reason=invalid_reason))
                other_errors.append(index)
                break
            selected.append((index, float(score)))
            outcomes.append(dict(index=index, status='complete', accepted=True, score=float(score)))
            if len(selected) == target:
                break
        elif status == 'support_failure' and complete is not True:
            reason = _reason(record) or 'support_failure_reason_not_supplied'
            support_failures.append(dict(index=index, reason=reason))
            outcomes.append(dict(index=index, status='support_failure', accepted=False, reason=reason))
        elif status in ('software_error', 'numerical_error', 'other_error'):
            reason = _reason(record) or status
            other_errors.append(index)
            outcomes.append(dict(index=index, status='other_error', accepted=False, reason=reason))
            invalid_reason = f'proposal_{index}_{status}'
            break
        else:
            reason = _reason(record) or f'unrecognized_or_inconsistent_status:{status!r}'
            other_errors.append(index)
            outcomes.append(dict(index=index, status='other_error', accepted=False, reason=reason))
            invalid_reason = f'proposal_{index}_invalid_record'
            break
    examined = len(outcomes)
    status = ('invalid' if invalid_reason else
              'complete' if len(selected) == target else 'unavailable')
    return dict(candidate=candidate, status=status, target=target,
                proposal_cap=proposal_cap, proposal_count_received=received,
                proposal_count_examined=examined,
                proposal_count_after_target_or_cap=max(0, received-examined),
                selected_indices=[i for i, _ in selected],
                selected_scores=[v for _, v in selected],
                support_failures=support_failures, other_errors=other_errors,
                proposal_outcomes=outcomes, reason=invalid_reason or
                (None if status == 'complete' else 'proposal_cap_exhausted_or_bank_incomplete'))


def fixed_entry_model_involvement(*, calibration_proposals_by_candidate, observed):
    """Evaluate one preassigned path against fixed N and CL calibration proposals.

    Each proposal/observed record has `status` in complete, support_failure,
    software_error, numerical_error, other_error, or not_run; complete records
    additionally carry `complete` (or `complete_A`) and `score` (or `A`).
    Software/numerical errors remain explicit unresolved errors and are never
    recoded as support failures. No candidate is labeled model-absent.
    """
    if not isinstance(calibration_proposals_by_candidate, dict):
        raise ValueError('calibration_proposals_by_candidate must be a mapping')
    banks = {}
    for candidate in NO_MODEL_CANDIDATES:
        banks[candidate] = _calibration_bank(
            calibration_proposals_by_candidate.get(candidate, ()), candidate)

    if not isinstance(observed, dict):
        raise ValueError('observed must be a mapping')
    obs_status = observed.get('status')
    obs_complete = observed.get('complete', observed.get('complete_A'))
    obs_score = observed.get('score', observed.get('A'))
    rank_valid = obs_status == 'complete' and obs_complete is True and _finite_score(obs_score)
    obs_error = None
    if obs_status == 'support_failure' and obs_complete is not True:
        unresolved_category = 'support_failure'
        obs_error = _reason(observed) or 'support_failure_reason_not_supplied'
    elif obs_status in ('software_error', 'numerical_error', 'other_error'):
        unresolved_category = 'other_error'
        obs_error = _reason(observed) or obs_status
    elif obs_status == 'not_run':
        unresolved_category = 'not_run'
        obs_error = _reason(observed) or 'evaluation_not_run'
    elif rank_valid:
        unresolved_category = None
    else:
        unresolved_category = 'other_error'
        obs_error = _reason(observed) or 'complete_score_nonfinite_or_inconsistent_status'

    ranks = {}
    if all(banks[c]['status'] == 'complete' for c in NO_MODEL_CANDIDATES):
        for candidate in NO_MODEL_CANDIDATES:
            b = banks[candidate]
            rank = stopped_candidate_rank(
                observed_score=obs_score if rank_valid else None,
                observed_full_window_complete=bool(rank_valid),
                observed_score_valid_for_rank=bool(rank_valid),
                calibration_scores=b['selected_scores'],
                calibration_full_window_complete=[True] * CAL_TARGET,
                calibration_score_valid_for_rank=[True] * CAL_TARGET,
                alpha=ALPHA)
            ranks[candidate] = rank

    if rank_valid and len(ranks) == 2:
        decision = ('model_involvement'
                    if all(ranks[c]['rejected'] for c in NO_MODEL_CANDIDATES)
                    else 'unresolved')
        if decision == 'unresolved':
            unresolved_category = 'complete_non_rejection'
    else:
        decision = 'unresolved'
        if rank_valid and not all(banks[c]['status'] == 'complete' for c in NO_MODEL_CANDIDATES):
            unresolved_category = 'calibration_unavailable'
    return dict(scope='fixed_entry_finite_catalogue', decision=decision,
                unresolved_category=unresolved_category,
                reason=obs_error, observed_status=obs_status,
                observed_complete=bool(rank_valid), observed_score=float(obs_score) if rank_valid else None,
                calibration_banks=banks, rank_by_candidate=ranks,
                eta_status='not_supplied', alarm_conditional=False,
                source_classification='not computed', model_absent=False,
                error_guarantee='independent iid histories and common measurable completion event; conditional-on-completion exchangeability only; assumptions not verified')
