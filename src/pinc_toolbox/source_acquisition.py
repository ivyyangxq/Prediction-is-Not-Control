"""Post-alarm supported-extreme diagnostic action recommendations.

No action is executed here; this module does not change fitted Q or certify
eta-restricted continuation, economic Q, welfare or source probabilities.
"""
from numbers import Integral
import math
from .eta_domain import eta_qualified_native_action


def _time(value,name):
    if isinstance(value,bool) or not isinstance(value,Integral):
        raise ValueError(name+' must be an integer')
    return int(value)


def recommend_source_action(*, action_grid, qualified, mode='passive',
        stage1_status, alarm_release, decision_time, source_start_decision,
        completed_action_count, source_end_decision=None, behavior_action=None):
    """One query snapshot, one decision, same parity as the native eta driver.

    ``qualified`` is one row returned by eta_qualified_native_action. Every mask
    is checked; no missing/unknown mask becomes support. Caller supplies the
    pre-innovation behavior action (e.g. its existing epsilon draw), if any.
    ``completed_action_count`` counts all prior executed decisions, not probes.
    The returned action is a recommendation, never an executed outcome.
    """
    if mode not in {'passive','alternate_supported_extremes'}:
        raise ValueError('unknown acquisition mode')
    if stage1_status not in {'alarm','no_alarm','unavailable','unresolved'}:
        raise ValueError('stage1_status must be explicit')
    decision=_time(decision_time,'decision_time');start=_time(source_start_decision,'source_start_decision')
    count=_time(completed_action_count,'completed_action_count')
    if count<0:raise ValueError('completed_action_count must be nonnegative')
    end=None if source_end_decision is None else _time(source_end_decision,'source_end_decision')
    if end is not None and end<start:raise ValueError('source end precedes source start')
    alarm=None if alarm_release is None else _time(alarm_release,'alarm_release')
    result={'status':'unavailable','reason':None,'mode':mode,'probe':False,
            'q_suggested_action':None,'behavior_action':behavior_action,'action':None,
            'action_status':'recommendation_only','supported_actions':[],
            'decision_time':decision,'source_start_decision':start,'source_end_decision':end,
            'alarm_release':alarm,'stage1_status':stage1_status,'completed_action_count':count,
            'source_probability':None,'economic_Q_comparison':None}
    try:
        grid=list(action_grid)
        if not grid or any(isinstance(x,bool) or not math.isfinite(x) for x in grid) or len(set(grid))!=len(grid):
            raise ValueError('action_grid must contain distinct finite values')
        if qualified.get('status') not in {'ready','partial'}:
            raise ValueError('qualified_query_unavailable')
        masks=[]
        for key in ['eta_mask','feasible_mask','qualified_mask','native_mask']:
            value=qualified.get(key)
            if not isinstance(value,list) or len(value)!=1 or len(value[0])!=len(grid) or any(type(x) is not bool for x in value[0]):
                raise ValueError(key+'_missing_or_unknown')
            masks.append(value[0])
        support=[float(a) for j,a in enumerate(grid) if all(m[j] for m in masks)]
        result['supported_actions']=support
        actions=qualified.get('actions',[])
        if len(actions)!=1 or actions[0] not in support:
            raise ValueError('native_Q_suggestion_not_supported')
        suggested=float(actions[0]);result['q_suggested_action']=suggested
        base=suggested if behavior_action is None else behavior_action
        if isinstance(base,bool) or base not in support:
            raise ValueError('behavior_action_not_supported')
    except (ValueError,TypeError,KeyError,AttributeError) as exc:
        result['reason']=str(exc);return result
    reason=('passive' if mode=='passive' else
            'no_affirmative_alarm' if stage1_status!='alarm' else
            'alarm_release_unavailable' if alarm is None else
            'alarm_after_source_start' if alarm>start else
            'before_source_start' if decision<start else
            'after_source_end' if end is not None and decision>end else 'post_alarm_source_acquisition')
    probe=reason=='post_alarm_source_acquisition'
    chosen=(min(support) if count%2==0 else max(support)) if probe else float(base)
    result.update(status='ready',reason=reason,probe=probe,behavior_action=float(base),action=chosen)
    return result


def recommend_native_source_action(policy, state, domain, *, feasible=None,
        remaining_horizon=None, auxiliary=None, **decision):
    """Query actual fitted native Q with eta support, then apply acquisition."""
    qualified=eta_qualified_native_action(policy,[state],domain,feasible=feasible,
        remaining_horizon=remaining_horizon,auxiliary=auxiliary)
    result=recommend_source_action(action_grid=policy.action_grid_.tolist(),qualified=qualified,**decision)
    result['qualified_query']=qualified
    return result
