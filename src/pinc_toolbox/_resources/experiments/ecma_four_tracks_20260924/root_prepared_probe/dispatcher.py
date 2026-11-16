"""One-step simulator probes from an already prepared public learner state.

This opt-in development executor neither prepares/refits nor selects actions.
The host must implement valid cloning and dated, shared exogenous innovations.
It is not an inference routine or a continuous counterfactual policy.
"""
from copy import deepcopy
import math
import numpy as np
from pinc_toolbox.counterfactual import PublicState, PublicInput, PublicOutcome
from pinc_toolbox.g import SupportError


def _vector(value, shape, label):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(label + ' has invalid shape or nonfinite values')
    return value.copy()


def run_prepared_probes(*, learner, environment, state, public, time,
                        actions=(.02, .08), on_record=None):
    """Return one record per fixed action; never mutate the parent objects.

    Caller supplies the actual prepared state at decision ``time`` and its
    dated public input, before issuing the parent's action. No condition or
    prepare call is repeated. Unsupported endpoints stay in the result.
    Software/clock errors propagate; they must not become support refusals.
    on_record receives completed/refused endpoints immediately, and a failure
    receipt before a software exception propagates, preserving earlier work.
    """
    if isinstance(time, (bool, np.bool_)) or not isinstance(time, (int, np.integer)) or time < 0:
        raise ValueError('time must be a nonnegative integer')
    if not isinstance(state, PublicState) or not isinstance(public, PublicInput):
        raise TypeError('prepared state and dated public input required')
    if isinstance(public.available_at, (bool, np.bool_)) or not isinstance(public.available_at, (int, np.integer)) or public.available_at > time:
        raise ValueError('public input is not available at decision time')
    for value in (state.physical, state.auxiliary, public.values):
        if np.asarray(value).ndim != 1 or not np.isfinite(value).all():
            raise ValueError('public vectors must be finite and one dimensional')
    actions = tuple(actions)
    if on_record is not None and not callable(on_record):
        raise TypeError('on_record must be callable')
    if not actions or any(isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, float, np.integer, np.floating)) or not math.isfinite(a) for a in actions):
        raise ValueError('actions must be finite numeric values')
    if len(set(actions)) != len(actions):
        raise ValueError('planned probe actions must be distinct')
    records = []
    for action in actions:
        # The deepcopy protocol belongs to the reviewed environment adapter;
        # it must preserve its internal proxy views without sharing parents.
        child = deepcopy(dict(learner=learner, environment=environment,
                              state=state, public=public))
        l, e, s, p = (child[k] for k in ('learner', 'environment', 'state', 'public'))
        action = float(action)
        record = dict(time=int(time), action=action, status='pending',
                      state=deepcopy(s), public=deepcopy(p), issued=None,
                      outcome=None, next_state=None, failure=None)
        stage = 'supported'
        try:
            supported = l.supported(time, deepcopy(s), action)
            if not isinstance(supported, (bool, np.bool_)):
                raise TypeError('supported must return bool')
            if not supported:
                raise SupportError('planned action outside native support')
            stage = 'issue'
            record['issued'] = deepcopy(l.issue(time, deepcopy(s), action))
            stage = 'step'
            out = e.step(time, deepcopy(s), action, deepcopy(p))
            if not isinstance(out, PublicOutcome):
                raise TypeError('step must return PublicOutcome')
            if isinstance(out.release_time, (bool, np.bool_)) or not isinstance(out.release_time, (int, np.integer)) or out.release_time != time + 1:
                raise ValueError('probe outcome has wrong release time')
            out = PublicOutcome(_vector(out.next_physical, s.physical.shape, 'next physical'),
                                float(out.reward), int(out.release_time))
            if not math.isfinite(out.reward):
                raise ValueError('nonfinite reward')
            record['outcome'] = deepcopy(out)
            stage = 'observe'
            aux = l.observe(time, deepcopy(s), action, deepcopy(out))
            record['next_state'] = PublicState(out.next_physical.copy(),
                _vector(aux, s.auxiliary.shape, 'next auxiliary'))
            record['status'] = 'complete'
        except Exception as exc:
            admissibility_refusal = isinstance(exc, SupportError) and stage in ('supported', 'issue')
            record['status'] = 'support_refusal' if admissibility_refusal else 'software_failure'
            record['failure'] = dict(stage=stage, type=type(exc).__name__, reason=str(exc))
            if not admissibility_refusal:
                if on_record is not None:
                    on_record(deepcopy(record))
                raise
        records.append(record)
        if on_record is not None:
            on_record(deepcopy(record))
    return records
