"""Pure first-H selection and conditional first39 acquisition orchestration."""
import math
from copy import deepcopy

CELLS=('N','CL','CM','MIX')
UNAVAILABLE={'initialization_support_refusal','initialization_identification_unavailable',
             'main_support_refusal','main_identification_unavailable'}

def evaluate_monitor(value, threshold):
    if value['status'] in UNAVAILABLE:
        return None
    if value['status']!='available' or value['completed_dates']!=8:
        raise ValueError('fatal, unknown or incomplete available monitor')
    score=value['scores']['serial']
    if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or score<0:
        raise ValueError('invalid complete monitor serial score')
    return score>threshold

def blank(plan):
    return dict(status='pending',selected=None,H=None,observed_future_started=False,
        monitors=[dict(**p,status='not_run',H=None) for p in plan],
        CAL=[dict(candidate=c,replicate_id=i,proposal_id=f'CAL:{c}:{i:03d}',status='not_run')
             for i in range(128) for c in CELLS],banks=None,observed=None,inference=None,failure=None)

def slim(row):
    return {k:deepcopy(v) for k,v in row.items() if k!='endpoints'}

def execute(*, plan, threshold, branch, monitor, condition, calibration, observed,
            progress=lambda kind,row:None, state=None):
    """Observed callback is reachable only after every candidate is done/capped.

    Callbacks carry public-conditioning objects, never receive monitor truth
    labels in CAL/observed interfaces. Fatal errors propagate to outer accounting.
    """
    state=blank(plan) if state is None else state
    banks={c:branch.CandidateBanks(c) for c in CELLS}
    state['_live_banks']=banks
    for index,p in enumerate(plan):
        row=state['monitors'][index];row['status']='running'
        value=monitor(p)
        h=evaluate_monitor(value,threshold)
        row.update(status=value['status'],H=h,completed_dates=value['completed_dates'],
                   scores=value.get('scores'),failure=value.get('failure'),counts=value.get('counts',{}))
        progress('monitor',row)
        if h is True:
            state.update(selected=index,H=True)
            for later in state['monitors'][index+1:]:later['status']='not_run_first_alarm_selected'
            break
    if state['selected'] is None:
        state.update(status='no_alarm_within_cap',H=None)
        for row in state['CAL']:row['status']='not_run_no_alarm_selected'
        return state
    public=condition(state['selected'])
    for row in state['CAL']:
        bank=banks[row['candidate']]
        reason=bank.next_slot_status()
        if reason!='attempt':
            row['status']=reason
            continue
        row['status']='running'
        raw=calibration(public,row['candidate'],row['replicate_id'],row['proposal_id'])
        compact=branch.compact(raw)
        recorded=bank.record(slim(compact))
        row.update(status=compact['status'],C_A=compact['C_A'],C_B=compact['C_B'],
                   A=compact['A'],B=compact['B'],counts=compact.get('counts',{}),
                   branch=recorded['branch'],failure=compact.get('failure'))
        progress('CAL',row)
        if bank.fatal:raise RuntimeError('fatal candidate continuation; no observed future or inference')
    if any(bank.next_slot_status()=='attempt' for bank in banks.values()):
        raise RuntimeError('observed future attempted before CAL acquisition ended')
    state['banks']={c:b.status() for c,b in banks.items()}
    # Publish bounded-CAL completion before touching the observed future.
    progress('CAL_complete',dict(banks=state['banks']))
    state['observed_future_started']=True
    raw=observed(public)
    obs=branch.compact(raw)
    state['observed']=slim(obs)
    if obs['status']=='fatal':raise RuntimeError('fatal observed future; inference invalid')
    state['inference']=branch.infer_eval(obs,banks)
    state['status']='complete'
    progress('observed',state['observed'])
    return state

def invalidate(state, failure):
    state.update(status='invalid',failure=failure,inference=None)
    for group in ('monitors','CAL'):
        for row in state[group]:
            if row['status']=='running':row['status']='fatal'
            elif row['status']=='not_run':row['status']='not_run_after_fatal'
    banks=state.get('_live_banks')
    if banks:state['banks']={c:b.status() for c,b in banks.items()}
    return state
