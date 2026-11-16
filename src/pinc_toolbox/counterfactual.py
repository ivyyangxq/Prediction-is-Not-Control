"""Synchronous one-main/three-child public counterfactual execution.

Adapters are trusted model-specific boundaries, not discovered simulators.
Only dated public input, state and released outcomes reach the learner. Private
simulation parameters may live inside an environment, never in these records.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol
import math
import numpy as np

from .g import SupportError
from .two_stage_report import run_two_stage_report

BRANCHES = ('main', 'fixed_action', 'fixed_public_environment', 'both_fixed')


@dataclass
class PublicState:
    physical: np.ndarray
    auxiliary: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass
class PublicInput:
    values: np.ndarray
    available_at: int


@dataclass
class PublicOutcome:
    next_physical: np.ndarray
    reward: float
    release_time: int


class PublicEnvironmentAdapter(Protocol):
    """Declare public input names; condition must not freeze endogenous state.

    public_input(t) is called only on the main environment at t. step runs
    after action choice and releases its outcome at t+1. Each branch receives
    a separate deep copy of the adapter, preserving matched initial RNG state.
    Use period-indexed innovations when branch-dependent RNG consumption exists.
    """
    public_input_names: tuple[str, ...]

    def public_input(self, time: int) -> PublicInput: ...
    def condition(self, state: PublicState, public: PublicInput) -> PublicState: ...
    def step(self, time: int, state: PublicState, action: float,
             public: PublicInput) -> PublicOutcome: ...


class BranchLearner(Protocol):
    """Own-history policy. prepare sees only outcomes already released.

    supported must check the actual forced action at this branch's state.
    observe records the executed action and returns next public auxiliaries.
    """
    def prepare(self, time: int, state: PublicState) -> PublicState: ...
    def issue(self, time: int, state: PublicState, action: float) -> dict: ...
    def choose(self, time: int, state: PublicState) -> float: ...
    def supported(self, time: int, state: PublicState, action: float) -> bool: ...
    def observe(self, time: int, state: PublicState, action: float,
                outcome: PublicOutcome) -> np.ndarray: ...


def _date(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    return int(value)


def _vector(value, name):
    x = np.asarray(value, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError(name + ' must be a finite vector')
    return x.copy()


def _state(state):
    if not isinstance(state, PublicState):
        raise TypeError('state must be PublicState')
    return PublicState(_vector(state.physical, 'physical'),
                       _vector(state.auxiliary, 'auxiliary'))


def _state_record(state):
    return dict(physical=state.physical.tolist(), auxiliary=state.auxiliary.tolist())


def run_four_paths(*, learner: BranchLearner, environment: PublicEnvironmentAdapter,
                   initial_state: PublicState, start_time: int, steps: int,
                   beta: float = 1.) -> dict:
    """Run synchronous main, constant-action, constant-public-input and both.

    At the first decision main chooses before its first outcome. That action
    becomes the constant for fixed_action and both_fixed. The public-input
    branch executes each just-issued main action, never its own greedy action.
    No child can change the main policy's support or choice. Support exits and
    software failures are retained per branch, without replacement or retry.
    Outcomes are delivered before prepare at the next decision (and finally
    at start_time+steps); no final-date refit is requested.
    """
    start = _date(start_time, 'start_time'); count = _date(steps, 'steps')
    if count < 1 or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError('positive steps and beta in [0,1] required')
    initial = _state(initial_state)
    names = tuple(environment.public_input_names)
    if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError('adapter must name distinct public input coordinates')
    live = {name: dict(learner=deepcopy(learner), environment=deepcopy(environment),
                      state=deepcopy(initial), pending=None) for name in BRANCHES}
    paths = {name: dict(status='running', trace=[], failures=[], completed_steps=0,
                        partial_discounted_return=0., discounted_return=None,
                        initial_state=_state_record(initial)) for name in BRANCHES}
    tape = []; fixed_public = None; fixed_action = None

    def failed(name, time, exc):
        p = paths[name]
        p['status'] = 'support_refusal' if isinstance(exc, SupportError) else 'software_failure'
        p['failures'].append(dict(time=time, type=type(exc).__name__, reason=str(exc)))

    def release(name, clock):
        branch = live[name]; pending = branch['pending']
        if pending is None:
            return
        time, state, action, outcome = pending
        if outcome.release_time != clock:
            raise ValueError('outcome not available at this release clock')
        aux = branch['learner'].observe(time, deepcopy(state), action, deepcopy(outcome))
        branch['state'] = PublicState(outcome.next_physical.copy(), _vector(aux, 'next auxiliary'))
        if branch['state'].auxiliary.shape != initial.auxiliary.shape:
            raise ValueError('auxiliary dimension changed')
        branch['pending'] = None
        paths[name]['trace'][-1].update(released=True, next_state=_state_record(branch['state']))

    for offset in range(count):
        time = start + offset
        for name in BRANCHES:
            if paths[name]['status'] == 'running':
                try:
                    release(name, time)
                except Exception as exc:
                    failed(name, time, exc)
        if paths['main']['status'] != 'running':
            for name in BRANCHES[1:]:
                if paths[name]['status'] == 'running':
                    paths[name]['status'] = 'main_unavailable'
                    paths[name]['failures'].append(dict(time=time, reason='no_current_main_action'))
            break
        try:
            public = live['main']['environment'].public_input(time)
            if not isinstance(public, PublicInput):
                raise TypeError('public_input must return PublicInput')
            public = PublicInput(_vector(public.values, 'public input'), _date(public.available_at, 'available_at'))
            if public.values.shape != (len(names),) or public.available_at > time:
                raise ValueError('public input has wrong dimension or is not yet available')
            if fixed_public is None:
                fixed_public = deepcopy(public)
        except Exception as exc:
            failed('main', time, exc)
            for name in BRANCHES[1:]:
                if paths[name]['status'] == 'running':
                    paths[name]['status'] = 'main_unavailable'
            break
        main_action = None
        for name in BRANCHES:
            if paths[name]['status'] != 'running':
                continue
            if name != 'main' and main_action is None:
                paths[name]['status'] = 'main_unavailable'
                paths[name]['failures'].append(dict(time=time, reason='no_current_main_action'))
                continue
            b = live[name]; p = paths[name]
            use_public = fixed_public if name in ('fixed_public_environment', 'both_fixed') else public
            try:
                state = _state(b['environment'].condition(deepcopy(b['state']), deepcopy(use_public)))
                if state.physical.shape != initial.physical.shape or state.auxiliary.shape != initial.auxiliary.shape:
                    raise ValueError('condition changed state dimensions')
                state = _state(b['learner'].prepare(time, deepcopy(state)))
                if state.physical.shape != initial.physical.shape or state.auxiliary.shape != initial.auxiliary.shape:
                    raise ValueError('prepare changed state dimensions')
                action = (float(b['learner'].choose(time, deepcopy(state))) if name == 'main'
                          else fixed_action if name in ('fixed_action', 'both_fixed') else main_action)
                if action is None or not math.isfinite(action):
                    raise SupportError('no finite action at current branch state')
                admitted = b['learner'].supported(time, deepcopy(state), action)
                if not isinstance(admitted, (bool, np.bool_)):
                    raise TypeError('supported must return a boolean')
                if not admitted:
                    raise SupportError('executed action outside branch support')
                if name == 'main':
                    main_action = action
                    if fixed_action is None:
                        fixed_action = action
                    tape.append(dict(time=time, available_at=time, action=action))
                issued = deepcopy(b['learner'].issue(time, deepcopy(state), action))
                outcome = b['environment'].step(time, deepcopy(state), action, deepcopy(use_public))
                if not isinstance(outcome, PublicOutcome):
                    raise TypeError('step must return PublicOutcome')
                outcome = PublicOutcome(_vector(outcome.next_physical, 'next physical'),
                                        float(outcome.reward), _date(outcome.release_time, 'release_time'))
                if (outcome.next_physical.shape != initial.physical.shape or
                    outcome.release_time != time + 1 or not math.isfinite(outcome.reward)):
                    raise ValueError('malformed or mistimed public outcome')
                p['trace'].append(dict(time=time, state=_state_record(state), action=action,
                    main_action=main_action, issued=issued, public_input=use_public.values.tolist(),
                    public_input_available_at=use_public.available_at,
                    reward=outcome.reward, release_time=outcome.release_time, released=False))
                p['completed_steps'] += 1
                p['partial_discounted_return'] += beta**offset * outcome.reward
                b['pending'] = (time, state, action, outcome)
            except Exception as exc:
                failed(name, time, exc)
    for name in BRANCHES:
        p = paths[name]
        if p['status'] == 'running':
            try:
                release(name, start + count)
                p['status'] = 'complete'
                p['discounted_return'] = p['partial_discounted_return']
            except Exception as exc:
                failed(name, start + count, exc)
        p['final_state'] = _state_record(live[name]['state'])
    values = {name: p['discounted_return'] for name, p in paths.items()}
    complete = all(p['status'] == 'complete' for p in paths.values())
    differences = None
    if complete:
        m, a, e, both = (values[n] for n in BRANCHES)
        differences = dict(action_with_varying_environment=m-a,
                           action_with_fixed_environment=e-both,
                           environment_with_main_actions=m-e,
                           environment_with_fixed_action=a-both,
                           interaction=m-a-e+both)
    return dict(status='complete' if complete else 'incomplete', mode='synchronous_main_tape',
        start_time=start, final_release_time=start+count, steps=count, beta=float(beta),
        public_input_names=list(names), frozen_public_input=None if fixed_public is None else fixed_public.values.tolist(),
        frozen_action=fixed_action, frozen_action_selected_at=start if fixed_action is not None else None,
        main_action_tape=tape, branches=paths,
        comparison=dict(status='available' if complete else 'unavailable', values=values,
                        signed_differences=differences, reason=None if complete else 'incomplete_branch',
                        scope='diagnostic finite-path returns; no source inference'),
        inference=dict(status='unavailable', p_value=None, source_probability=None,
                       reason='no_counterfactual_calibration_supplied'))


def counterfactual_report(result, *, stage1=None, context=None, existing_report=None,
                          pair_routing=None):
    """Attach diagnostics to the existing two-stage controller without evidence.

    A separately supplied report is retained verbatim, not validated or reused
    as inference for these counterfactual paths. No source callback is invoked.
    pair_routing explicitly opts into a model-specific diagnostic: applicability
    must declare input_only_matched_physical_paths. Generic physical-public-input
    adapters do not establish equivalence of the two action-return contrasts.
    """
    report = run_two_stage_report(
        {'status': 'unavailable', 'reason': 'detector_not_supplied'} if stage1 is None else deepcopy(stage1),
        context={'route': 'four_path_diagnostic'} if context is None else deepcopy(context))
    report['counterfactual'] = deepcopy(result)
    if pair_routing is not None:
        from .counterfactual_routing import route_action_pairs
        report['counterfactual']['routed_action_diagnostic'] = route_action_pairs(
            result, **deepcopy(pair_routing))
    if existing_report is not None:
        report['attached_report'] = dict(scope='separately_supplied_not_validated_for_this_diagnostic',
                                         report=deepcopy(existing_report))
    return report
