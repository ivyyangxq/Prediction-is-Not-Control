"""Exact-law grouped, two-branch HS composite scoring helpers.

Grouping is based on the full supplied conditional F/d/covariance schedules.
Family summaries depend only on group membership and group scoring statuses;
no generating-candidate label is used to select a test or witness.
"""
from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
import numpy as np
from .hs_terminal_active import score_active_terminal
from .stopped_candidate_rank import stopped_candidate_rank

_FAMILY_ORDER = ('C0', 'CL', 'CM', 'CMIX')
_VALID_STATUSES = {'excluded', 'compatible', 'unresolved'}
_COMMON_OVERRIDE_KEYS = {'reference_coefficients', 'reference', 'reference_F', 'reference_d',
    'residual_left_inverse', 'residual_map', 'K', 'support', 'eta'}


def _law(candidate):
    try:
        f = np.asarray(candidate.get('candidate_F', candidate['F'] if 'F' in candidate else None), dtype=float)
        d = np.asarray(candidate.get('candidate_d', candidate['d'] if 'd' in candidate else None), dtype=float)
        q = np.asarray(candidate.get('candidate_covariance', candidate['Q'] if 'Q' in candidate else None), dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('each candidate needs numeric F, d, and covariance schedules') from exc
    if f.ndim != 3 or f.shape[1:] != (2, 2) or not len(f):
        raise ValueError('candidate_F must be (H,2,2)')
    h = len(f)
    if d.shape != (h, 2) or q.shape != (h, 2, 2):
        raise ValueError('candidate schedules have inconsistent horizons or dimensions')
    if not (np.isfinite(f).all() and np.isfinite(d).all() and np.isfinite(q).all()):
        raise ValueError('candidate schedules must be finite')
    if 'candidate_id' not in candidate and 'id' not in candidate:
        raise ValueError('each candidate needs candidate_id or id')
    return f, d, q


def group_hs_conditional_laws(candidates):
    """Group candidate labels by exact equality of full F, d, and Q arrays.

    `candidates` is traversed in its supplied catalogue order. Each record must
    include candidate_id (or id), candidate_F/d/covariance (or F/d/Q), and
    families; catalogue_index is optional and defaults to the input position.
    The returned groups are ordered by their minimum catalogue index.
    """
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError('supply an ordered nonempty candidate list')
    groups = []
    seen_ids, seen_indices = set(), set()
    common_horizon = None
    for input_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError('candidate entries must be dictionaries')
        override = _COMMON_OVERRIDE_KEYS.intersection(candidate)
        if override:
            raise ValueError('candidate-specific reference/map/support/eta overrides are not allowed')
        f, d, q = _law(candidate)
        if common_horizon is None:
            common_horizon = len(f)
        elif len(f) != common_horizon:
            raise ValueError('all candidates must use the same full conditional horizon')
        candidate_id = candidate.get('candidate_id', candidate.get('id'))
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError('candidate IDs must be nonempty strings')
        if candidate_id in seen_ids:
            raise ValueError('candidate IDs must be unique')
        seen_ids.add(candidate_id)
        index = candidate.get('catalogue_index', input_index)
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0:
            raise ValueError('catalogue_index must be a nonnegative integer')
        if int(index) in seen_indices:
            raise ValueError('catalogue_index values must be unique')
        seen_indices.add(int(index))
        families = candidate.get('families')
        if (not isinstance(families, (list, tuple, set)) or not families or
                not set(families) <= set(_FAMILY_ORDER)):
            raise ValueError('families must be a nonempty subset of C0, CL, CM, CMIX')
        member = {key: value for key, value in candidate.items()
                  if key not in ('candidate_F', 'candidate_d', 'candidate_covariance', 'F', 'd', 'Q')}
        member['candidate_id'] = candidate_id
        member['catalogue_index'] = int(index)
        member['families'] = [name for name in _FAMILY_ORDER if name in set(families)]
        match = None
        for group in groups:
            if (np.array_equal(f, group['candidate_F']) and
                    np.array_equal(d, group['candidate_d']) and
                    np.array_equal(q, group['candidate_covariance'])):
                match = group
                break
        if match is None:
            groups.append({'candidate_F': f.copy(), 'candidate_d': d.copy(),
                'candidate_covariance': q.copy(), 'members': [member]})
        else:
            match['members'].append(member)
    for group in groups:
        group['members'].sort(key=lambda m: m['catalogue_index'])
        representative = group['members'][0]
        group['representative_id'] = representative['candidate_id']
        group['canonical_catalogue_index'] = representative['catalogue_index']
        group['family_membership_union'] = [name for name in _FAMILY_ORDER
            if any(name in member['families'] for member in group['members'])]
    family_ids = {name: {m['candidate_id'] for g in groups for m in g['members'] if name in m['families']}
                  for name in _FAMILY_ORDER}
    if not (family_ids['C0'] <= family_ids['CL'] <= family_ids['CMIX']
            and family_ids['C0'] <= family_ids['CM'] <= family_ids['CMIX']):
        raise ValueError('candidate family memberships must satisfy C0 subset CL/CM subset CMIX')
    groups.sort(key=lambda g: g['canonical_catalogue_index'])
    for group_index, group in enumerate(groups):
        group['group_index'] = group_index
    return groups


def _rank_terminal(out, alpha_terminal):
    term = out['terminal']
    rank = stopped_candidate_rank(
        observed_score=term['observed_score'],
        observed_full_window_complete=out['observed_full_window_complete'],
        observed_score_valid_for_rank=out['observed_score_valid_for_rank'],
        calibration_scores=term['calibration_scores'],
        calibration_full_window_complete=out['calibration_full_window_complete'],
        calibration_score_valid_for_rank=out['calibration_score_valid_for_rank'],
        alpha=alpha_terminal)
    rank['score_policy'] = 'terminal_prefix_observed_complete'
    rank['statistic'] = 'terminal_slope_energy'
    term['rank'] = rank
    return rank


def _composite_status(max_rank, terminal_rank):
    if max_rank['status'] == 'excluded' or terminal_rank['status'] == 'excluded':
        return 'excluded'
    if (max_rank['status'] == terminal_rank['status'] == 'compatible' and
        max_rank['observed_full_window_complete'] and
        max_rank['observed_score_valid_for_rank'] and
        terminal_rank['observed_full_window_complete'] and
        terminal_rank['observed_score_valid_for_rank']):
        return 'compatible'
    return 'unresolved'


def score_hs_composite_candidate(*, evaluation_path, calibration_paths,
        reference_coefficients, candidate_F, candidate_d, candidate_covariance,
        residual_left_inverse, support, eta, context,
        alpha_max=0.025, alpha_terminal=0.025):
    """Score a grouped law with max-prefix OR fixed-terminal rejection.

    Both branches use the same retained paths, support rule, and residual map,
    with their own explicit rank levels. A compatibility status requires both
    ranks to be compatible and the observed path complete and valid.
    """
    if (isinstance(alpha_max, bool) or isinstance(alpha_terminal, bool) or
            not np.isfinite(alpha_max) or not np.isfinite(alpha_terminal) or
            not 0 < alpha_max < 1 or not 0 < alpha_terminal < 1):
        raise ValueError('branch alpha values must each be in (0,1)')
    out = score_active_terminal(evaluation_path=evaluation_path,
        calibration_paths=calibration_paths,
        reference_coefficients=reference_coefficients, candidate_F=candidate_F,
        candidate_d=candidate_d, candidate_covariance=candidate_covariance,
        residual_left_inverse=residual_left_inverse, support=support, eta=eta,
        alpha=alpha_max, context=context, include_terminal=True)
    max_rank = {key: out[key] for key in (
        'p', 'status', 'cal_ge_observed_count', 'rank_numerator', 'rank_denominator',
        'cal_total_B', 'cal_failed_count', 'observed_full_window_complete',
        'observed_score_valid_for_rank')}
    terminal_rank = _rank_terminal(out, alpha_terminal)
    short_terminal = {key: terminal_rank[key] for key in max_rank}
    status = _composite_status(max_rank, terminal_rank)
    out.update(composite_status=status, composite_decision_rule='max_prefix_reject OR terminal_reject',
        alpha_max=float(alpha_max), alpha_terminal=float(alpha_terminal),
        max_rank=max_rank, terminal_rank=short_terminal,
        bound_metadata={'combined_level_bound': None, 'calibration_error_bound': None,
            'declared_split': {'max_prefix': float(alpha_max), 'terminal': float(alpha_terminal)},
            'nominal_union_sum': float(alpha_max + alpha_terminal),
            'claim': 'No combined bound or theorem claim is certified by this routine.'})
    return out


def aggregate_hs_grouped_candidates(groups_with_status, *, expected_groups):
    """Aggregate statuses only after verifying complete grouped-catalogue coverage."""
    if not isinstance(groups_with_status, (list, tuple)):
        raise ValueError('groups_with_status must be an ordered list or tuple')
    if not isinstance(expected_groups, (list, tuple)) or not expected_groups:
        raise ValueError('expected_groups must contain the complete grouped catalogue')
    expected = {}
    for group in expected_groups:
        index = group.get('group_index')
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0 or index in expected:
            raise ValueError('expected group indices must be unique nonnegative integers')
        expected[int(index)] = group
    if set(expected) != set(range(len(expected))):
        raise ValueError('expected group_index values must be contiguous from zero')
    actual = {}
    for group in groups_with_status:
        index = group.get('group_index')
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0 or index in actual:
            raise ValueError('scored group indices must be unique nonnegative integers')
        actual[int(index)] = group
    if set(actual) != set(expected):
        raise ValueError('scored groups must cover every expected group exactly once')

    groups = []
    for index in sorted(expected):
        want, got = expected[index], actual[index]
        if (got.get('representative_id') != want.get('representative_id') or
                got.get('canonical_catalogue_index') != want.get('canonical_catalogue_index')):
            raise ValueError('scored group representative/catalogue metadata mismatch')
        def member_signature(m):
            return (m.get('candidate_id'), m.get('catalogue_index'), tuple(m.get('families', ())))
        expected_members = [member_signature(m) for m in want.get('members', [])]
        actual_members = [member_signature(m) for m in got.get('members', [])]
        if len(set(m[0] for m in actual_members)) != len(actual_members) or actual_members != expected_members:
            raise ValueError('scored group member label/index/family metadata mismatch')
        if list(got.get('family_membership_union', [])) != list(want.get('family_membership_union', [])):
            raise ValueError('scored group family union mismatch')
        status = got.get('composite_status', got.get('union_status', got.get('status')))
        if status not in _VALID_STATUSES:
            raise ValueError('each group needs excluded, compatible, or unresolved status')
        groups.append((got, status))

    by_family = {family: [] for family in _FAMILY_ORDER}
    for group, status in groups:
        memberships = set(group['family_membership_union'])
        if not memberships <= set(_FAMILY_ORDER):
            raise ValueError('unknown family membership')
        for family in memberships:
            by_family[family].append((group, status))
    result = {}
    for family, rows in by_family.items():
        statuses = [status for _, status in rows]
        family_status = ('excluded' if rows and all(s == 'excluded' for s in statuses)
                         else 'compatible' if 'compatible' in statuses else 'unresolved')
        result[family] = {'status': family_status, 'intersecting_groups': len(rows),
            'compatible_groups': [g.get('representative_id') for g,s in rows if s == 'compatible'],
            'excluded_group_count': sum(s == 'excluded' for s in statuses),
            'unresolved_group_count': sum(s == 'unresolved' for s in statuses)}
    witness = next((g.get('representative_id') for g,s in by_family['CMIX']
                    if s == 'compatible'), None)
    result['CMIX']['canonical_compatible_witness'] = witness
    result['source_outcome'] = ('maintained_family_inadequate'
        if result['CMIX']['status'] == 'excluded' else
        'both_required' if result['CL']['status'] == 'excluded' and
            result['CM']['status'] == 'excluded' and witness else
        'model_required' if result['CL']['status'] == 'excluded' and witness else
        'latent_required' if result['CM']['status'] == 'excluded' and witness else
        'ambiguous' if witness else 'unresolved')
    return result


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.setflags(write=False)
        return result
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return deepcopy(value)


def score_grouped_hs_source(*, candidates, stage1_status, evaluation_path,
        reference_coefficients, residual_left_inverse, support, eta, context,
        calibration_size, calibration_provider, alpha_max=0.025,
        alpha_terminal=0.025, on_group=None, evidence_sink=None):
    """Score exact-law groups serially, gated by a stage-1 alarm.

    `calibration_provider(group, common_context)` supplies exactly
    ``calibration_size`` original CAL slots for that group. One bank is held at
    a time. `on_group(record)` receives the full scored record immediately.
    Alternatively, `evidence_sink(group, evidence)` can persist raw branch scores
    and return a reference; only that reference is kept in the summary.
    """
    groups = group_hs_conditional_laws(candidates)
    if isinstance(stage1_status, bool):
        alarm = stage1_status
    elif isinstance(stage1_status, dict) and type(stage1_status.get('alarm')) is bool:
        alarm = stage1_status['alarm']
    else:
        raise ValueError('stage1_status must be a boolean or mapping with boolean alarm')
    if not alarm:
        return {'status': 'no_alarm', 'groups': [], 'family_results': None,
                'calibration_groups_scored': 0, 'stage1_alarm': False}
    if isinstance(calibration_size, bool) or not isinstance(calibration_size, (int, np.integer)) or calibration_size < 1:
        raise ValueError('calibration_size must be a positive integer')
    if not callable(calibration_provider):
        raise ValueError('calibration_provider must be callable')
    if on_group is not None and not callable(on_group):
        raise ValueError('on_group must be callable when supplied')
    if evidence_sink is not None and not callable(evidence_sink):
        raise ValueError('evidence_sink must be callable when supplied')
    if on_group is not None and evidence_sink is not None:
        raise ValueError('choose either on_group or evidence_sink')
    try:
        start_state = np.asarray(evaluation_path[0], dtype=float).tolist()
    except (TypeError, ValueError, IndexError):
        start_state = None
    common = _freeze({'source_start_state': start_state,
        'reference_coefficients': reference_coefficients,
        'residual_left_inverse': residual_left_inverse, 'support': support,
        'eta': eta, 'context': context, 'calibration_size': int(calibration_size)})
    summaries = []
    for group in groups:
        # The callback is invoked only after the alarm and once per exact law.
        provider_group = _freeze(group)
        supplied = calibration_provider(provider_group, common)
        if isinstance(supplied, (list, tuple)):
            bank = supplied
        else:
            try:
                bank = list(supplied)
            except TypeError as exc:
                raise ValueError('calibration_provider must return every original CAL slot') from exc
        if len(bank) != int(calibration_size):
            raise ValueError('calibration_provider must retain every planned CAL slot')
        score = score_hs_composite_candidate(evaluation_path=evaluation_path,
            calibration_paths=bank, reference_coefficients=reference_coefficients,
            candidate_F=group['candidate_F'], candidate_d=group['candidate_d'],
            candidate_covariance=group['candidate_covariance'],
            residual_left_inverse=residual_left_inverse, support=support, eta=eta,
            context={**(context if isinstance(context, dict) else {'context': context}),
                'group_index': group['group_index'],
                'representative_id': group['representative_id'],
                'canonical_catalogue_index': group['canonical_catalogue_index']},
            alpha_max=alpha_max, alpha_terminal=alpha_terminal)
        record = {**group, 'scoring_result': score,
                  'composite_status': score['composite_status']}
        evidence_ref = None
        if on_group is not None:
            on_group(record)
        elif evidence_sink is not None:
            evidence_ref = evidence_sink(provider_group, score)
        summaries.append({'group_index': group['group_index'],
            'canonical_catalogue_index': group['canonical_catalogue_index'],
            'representative_id': group['representative_id'],
            'members': group['members'],
            'family_membership_union': group['family_membership_union'],
            'composite_status': score['composite_status'],
            'max_p': score['max_rank']['p'], 'terminal_p': score['terminal_rank']['p'],
            **({'evidence_ref': evidence_ref} if evidence_sink is not None else {})})
        del bank, score, record
    family_input = [{**summary, 'status': summary['composite_status']}
                    for summary in summaries]
    return {'status': 'complete', 'stage1_alarm': True,
        'calibration_groups_scored': len(summaries), 'groups': summaries,
        'family_results': aggregate_hs_grouped_candidates(family_input, expected_groups=groups),
        'bound_metadata': {'combined_level_bound': None,
            'calibration_error_bound': None,
            'declared_split': {'max_prefix': float(alpha_max),
                               'terminal': float(alpha_terminal)},
            'nominal_union_sum': float(alpha_max + alpha_terminal),
            'claim': 'No combined bound or theorem claim is certified by this routine.'}}


# Alternate verb-first spelling retained as a convenience alias.
run_hs_grouped_composite = score_grouped_hs_source
