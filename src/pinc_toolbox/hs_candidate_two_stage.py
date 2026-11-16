"""Saved HS residual-candidate evidence through the strict publication controller.

This is reporting composition, not conditional calibration or a path generator.
"""
from copy import deepcopy

from .residual_candidate_source import route_residual_candidates
from .two_stage_report import run_two_stage_report


HS_LABELS = {"latent": "Environment law (Z observed)",
             "model": "Growth mechanism", "mixed": "Both required"}


def run_hs_candidate_two_stage(*, stage1, candidate_evidence, family_members,
                               active, release, context):
    """Report one saved path/release/eta record; missing candidates stay unknown.

    Candidate evidence must come from the same residual-energy experiment and
    declared finite catalogue. ``context`` preserves provenance, coordinate
    system, eta, fitted G and CAL scope; this function cannot verify provenance
    or turn historical unconditional CAL into post-alarm conditional CAL.
    """
    if not isinstance(active, bool):
        raise ValueError("active must be boolean")
    if not isinstance(context, dict):
        raise ValueError("context must be a mapping")
    # The common router inherits compatibility along nested families. Enforce
    # the declared membership needed for that inference, using ordinary sets.
    families = {}
    for name in ("C0", "CL", "CM", "CMIX"):
        members = family_members.get(name, [])
        if not isinstance(members, list) or any(not isinstance(x, str) for x in members):
            raise ValueError("family members must be lists of candidate IDs")
        if len(members) != len(set(members)):
            raise ValueError("family members must be distinct")
        families[name] = members
    if not (set(families['C0']) <= set(families['CL']) <= set(families['CMIX'])
            and set(families['C0']) <= set(families['CM']) <= set(families['CMIX'])):
        raise ValueError("declared candidate families must be nested")
    saved_context = deepcopy(context)
    saved_context.update(route="hs_saved_residual_candidates", release=release,
                         source_scope="declared_finite_catalogue_only",
                         observation="Z observed; no hidden-state inference",
                         statistic="sequential_slope_energy",
                         family_members=families,
                         conditional_calibration="not_verified_by_reporting_adapter")
    evidence = None

    def evaluate(name):
        nonlocal evidence
        if evidence is None:
            evidence = route_residual_candidates(
                candidate_evidence, families, ordinary_alarm=True,
                active=active, release=release)
        return evidence['family_evidence'][name]

    report = run_two_stage_report(
        stage1, context=saved_context, evaluate=evaluate,
        readiness={"status": "ready" if active else "unavailable",
                   "reason": "saved_support_active" if active else "saved_support_exit"})
    report['source_labels'] = dict(HS_LABELS)
    report['semantic_disclosure'] = (
        "CL is the environment law with observed Z, including induced growth and covariance changes. "
        "CM is the growth mechanism, including induced covariance changes. "
        "Machine field latent denotes environment necessity here; it does not mean hidden Z. "
        "Necessity is relative to the declared finite catalogue; continuous coverage is not established.")
    report['action_comparison'] = {"status": "not_applicable", "reason": "actionless_HS_transition"}
    report['reliability']['numerical_certification'] = 'saved_residual_candidate_evidence_only'
    if evidence is not None:
        report['saved_candidate_route'] = evidence
        report['stage2']['joint_witness_ids'] = [
            name for name in families['CMIX']
            if evidence['candidate_evidence'].get(name, {}).get('status') == 'compatible']
    return report
