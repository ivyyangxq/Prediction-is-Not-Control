"""Finite, disjoint source classes; no nested-family inheritance.

Report-only interface. All banks must correspond to the same supplied observed
statistic/selection/qualification program. This interface cannot verify kernels
or prove that supplied records were produced by that program.
"""
from copy import deepcopy
from .alarm_conditional_model_involvement import alarm_conditional_model_involvement

CLASSES = ('N', 'L', 'M', 'ML')


def finite_source_set(*, alarm_status, candidate_classes,
                      calibration_proposals_by_candidate, observed, alpha=.05):
    """Retain untested laws and report implications of a common candidate set.

    Every class must have at least one declared law. Multiple nuisance values
    are allowed. Involvement assertions concern only the declared catalogue.
    The existing 39-qualified/128-proposal and inclusive rank rules are reused.
    Neither retention nor a singleton is a posterior probability.
    """
    if not isinstance(candidate_classes, dict) or not candidate_classes:
        raise ValueError('candidate_classes must be a nonempty mapping')
    if any(v not in CLASSES for v in candidate_classes.values()):
        raise ValueError('class labels must be N, L, M, ML')
    if set(candidate_classes.values()) != set(CLASSES):
        raise ValueError('declare laws for all four classes; missing banks stay unavailable')
    # Reuse bank parsing/ranking, never its no-model decision: this call supplies
    # all candidate laws solely to the common rank engine.
    ranks = alarm_conditional_model_involvement(
        alarm_status=alarm_status, observed=observed, alpha=alpha,
        no_model_candidates=tuple(candidate_classes),
        calibration_proposals_by_candidate=calibration_proposals_by_candidate)
    excluded = [c for c, r in ranks['rank_by_candidate'].items()
                if r['p_two_sided'] <= alpha]
    retained = [c for c in candidate_classes if c not in excluded]
    retained_classes = [k for k in CLASSES if any(candidate_classes[c] == k for c in retained)]
    eligible = alarm_status == 'triggered' and ranks['observed_signed_sum'] is not None
    conflict = eligible and not retained
    return dict(
        scope='finite_disjoint_source_catalogue_common_experiment',
        alarm_status=alarm_status, alpha=alpha,
        candidate_classes=deepcopy(candidate_classes),
        excluded_candidates=excluded, retained_candidates=retained,
        retained_classes=retained_classes,
        model_involvement=bool(eligible and retained and set(retained_classes) <= {'M','ML'}),
        latent_involvement=bool(eligible and retained and set(retained_classes) <= {'L','ML'}),
        singleton_class=retained_classes[0] if eligible and len(retained_classes)==1 else None,
        decision='catalogue_conflict' if conflict else ('retained_set' if eligible else 'unresolved'),
        unresolved_category=ranks['unresolved_category'] if not eligible else None,
        rank_by_candidate=ranks['rank_by_candidate'],
        calibration_banks=ranks['calibration_banks'],
        calibration_inputs=ranks['calibration_inputs'], observed_input=ranks['observed_input'],
        source_probability=None, posterior_probability=None,
        conditional_assumptions_verified=False, eta_status='not_supplied',
        guarantee='Conditional candidate-set coverage requires the true law in the catalogue, correct common-program kernels and exchangeability. No empirical power or global monitoring guarantee is supplied.')
