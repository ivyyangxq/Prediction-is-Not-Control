"""Parameterised original prepared future loop; called only after selected H.
Derived from independently reviewed four-future DEV without changing numerics.
"""
from pathlib import Path
import copy
import importlib.util
import json
import signal
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
CANDIDATES = ('N', 'CL', 'CM', 'MIX')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def select_supported_high(model,physical,auxiliary):
    """Largest pre-outcome grid action in (.02,.08] with physical/public support and finite Bellman Q."""
    import numpy as np
    grid=model.action_grid_
    values=model.q_values(physical[None],auxiliary[None],remaining_horizon=10,evaluation='bellman')[0]
    physical_mask=model.model_.support_.mask(physical[None],grid)[0]
    public_mask=np.asarray(model.planning_feasible_,dtype=bool)
    candidates=[]
    for j,a in enumerate(grid):
        if .02 < float(a) <= .08:
            candidates.append(dict(action=float(a),physical_support=bool(physical_mask[j]),
                public_support=bool(public_mask[j]),finite_direct_Q=bool(np.isfinite(values[j])),
                direct_Q=float(values[j]) if np.isfinite(values[j]) else None))
    eligible=[r for r in candidates if r['physical_support'] and r['public_support'] and r['finite_direct_Q']]
    return (max((r['action'] for r in eligible),default=None),candidates)


def pair_actual(low,high):
    import numpy as np
    out=dict(time=low['time'],low_action=low['action'],high_action=high['action'],status='unavailable',
        actual_x_difference=None,predicted_x_difference=None,x_residual_difference=None,x_identity_error=None,
        actual_w_difference=None,predicted_w_difference=None,w_residual_difference=None,w_identity_error=None)
    if low['time']!=high['time']:raise ValueError('pair dates differ')
    if high['action'] is None or low['status']!='complete' or high['status']!='complete':return out
    if not low['action']<high['action']<=.08:raise ValueError('actual paired action labels invalid')
    for j,key in ((0,'x'),(1,'w')):
        actual=high['next_physical'][j]-low['next_physical'][j]
        pred=high['issued']['main']['full_physical_prediction'][j]-low['issued']['main']['full_physical_prediction'][j]
        residual=high['full_'+key+'_error']-low['full_'+key+'_error'];identity=residual-(actual-pred)
        if not np.isfinite([actual,pred,residual,identity]).all() or abs(identity)>1e-12:
            raise ValueError('paired residual identity failure')
        out.update({f'actual_{key}_difference':actual,f'predicted_{key}_difference':pred,
            f'{key}_residual_difference':residual,f'{key}_identity_error':identity})
    if out['actual_w_difference']!=0:raise ValueError('same-date probes changed actual proxy')
    out['status']='available';return out

def episode(spec, panel, env, folder, fixed, b, dispatch, measurements):
    import numpy as np
    from pinc_toolbox.counterfactual import PublicState
    from pinc_toolbox.g import SupportError
    folder.mkdir()
    candidate = spec['cell']
    if spec.get('H') is not True: raise ValueError('selected history H=true required')
    endpoints = [fixed.blank_endpoint(spec,t,a) for t in range(40,48) for a in (.02,None)]
    date_selections=[]
    b.ACTIVE_PROBE_ENDPOINTS = endpoints
    before = b.COUNTS.copy()
    mainrows = []
    status, failure, t = 'pending', None, None
    b.ARCHIVE = b.EpisodeWriter(folder/'records.xz', b.BUDGET)
    def save(name, value): b.save(folder/name, value)
    try:
        l = panel.learner
        save('BOUNDARY.json', dict(release_time=40, H=True, H_status='fixed_public_history_alarm',
             history_selection='first predeclared complete monitor with strict H=true',
             candidate=candidate, physical=panel.physical.tolist(), auxiliary=panel.auxiliary.tolist()))
        b.snap(folder/'released_learner_40.pkl.gz', l)
        for t in range(40, 48):
            b.front.CURRENT.update(seed=None, folder=folder, date=t, stage='conditional_H_future',
                                   raw_validated=b.front.validate_history(l.history))
            panel.prepare(t)
            b.snap(folder/f'snapshot_{t}.pkl.gz', l)
            if l.model.as_of_ != t or l.pending is not None or panel.pending is not None:
                raise ValueError('MAIN not prepared before reset probes')
            save(f'states_{t}.json', dict(time=t, physical=panel.physical.tolist(), auxiliary=panel.auxiliary.tolist()))
            qs = PublicState(panel.physical[0].copy(), panel.auxiliary[0].copy())
            planned = float(l.choose(t, qs))
            if not l.supported(t, qs, planned): raise SupportError('MAIN native action unsupported')
            high_action,candidate_mask=select_supported_high(l.model,qs.physical,qs.auxiliary)
            selection=dict(time=t,low_action=.02,candidates=candidate_mask,chosen_high_action=high_action,action_gap=(high_action-.02 if high_action is not None else None))
            date_selections.append(selection);save(f'probe_{t}_SELECTION.json',selection)
            pl = b.PanelBranchLearner(l)
            pe = b.PanelEnvironmentAdapter(b.OriginalEnvironmentHost(env))
            ps = b.pack_state(panel.physical, panel.auxiliary)
            public = pe.public_input(t)
            if public.available_at != t or public.values[0] != panel.physical[0, 1]:
                raise ValueError('probe public input mismatch')
            history_before = l.history.copy(deep=True)
            physical_before, aux_before = panel.physical.copy(), panel.auxiliary.copy()
            core_before = (l.model.model_.core._C.copy(), l.model.model_.core._Psi.copy())
            public_before = env.public(t)
            date_actions=(.02,high_action)
            for ai,a in enumerate(date_actions):
                row=endpoints[(t-40)*2+ai]
                row.update(action=a,candidate_mask=candidate_mask,selected_high_action=high_action,action_gap=selection['action_gap'],evaluation='bellman')
                if a is None:
                    row.update(status='support_refusal',failure={'type':'NoSupportedDistinctHighAction','reason':'no action in (.02,.08] passed physical/public support and finite Bellman Q'},
                        physical_action_supported=False,finite_executed_Q=False,executed_Q=None,
                        query_physical=ps.physical.tolist(),query_auxiliary=ps.auxiliary.tolist())
                    continue
                row['status']='pending'
                j=int(np.flatnonzero(l.model.action_grid_==a)[0])
                value=l.model.q_values(qs.physical[None],qs.auxiliary[None],remaining_horizon=10,evaluation='bellman')[0,j]
                physical_supported=bool(l.model.model_.support_.mask(qs.physical[None],l.model.action_grid_)[0,j])
                public_supported=bool(l.model.planning_feasible_[j])
                row.update(physical_action_supported=physical_supported,public_action_supported=public_supported,
                           finite_executed_Q=bool(np.isfinite(value)),executed_Q=float(value) if np.isfinite(value) else None,
                           query_physical=ps.physical.tolist(),query_auxiliary=ps.auxiliary.tolist())
                save(f'probe_{t}_{round(a*100):02d}_PLAN.json',dict(time=t,action=a,prepared_as_of=l.model.as_of_,evaluation='bellman',qualification=row.copy()))
                science_before=b.COUNTS.copy()
                raw=fixed.execute_endpoint(dispatch,row=row,
                    persist=lambda value:save(f'probe_{t}_{round(a*100):02d}.json',value),
                    learner=pl,environment=pe,state=ps,public=public,time=t,actions=(a,))
                if raw['status']=='complete':
                    if not row['physical_action_supported'] or not row['public_action_supported'] or not row['finite_executed_Q']:
                        raise ValueError('probe completed outside Bellman support')
                    issued=raw['issued']['main']
                    if raw['issued']['time']!=t or issued['fit_as_of']!=t or issued['action']!=a:
                        raise ValueError('probe issue vintage/action mismatch')
                elif raw['status']!='support_refusal':raise ValueError('unknown probe status')
                for key in ('prefix_G', 'native_G', 'Q', 'reward', 'constructors'):
                    for suffix in ('attempted', 'succeeded'):
                        if b.COUNTS[key+'_'+suffix] != science_before[key+'_'+suffix]:
                            raise ValueError('probe fitted or solved')
                if env._next_t != t or l.pending is not None or panel.pending is not None or not l.history.equals(history_before):
                    raise ValueError('probe contaminated MAIN')
                if not np.shares_memory(env.w, env.proxy_all) or not np.shares_memory(pe.host._env.w, pe.host._env.proxy_all):
                    raise ValueError('host proxy alias lost')
                for old, new in ((physical_before, panel.physical), (aux_before, panel.auxiliary),
                     (core_before[0], l.model.model_.core._C), (core_before[1], l.model.model_.core._Psi), *zip(public_before, env.public(t))):
                    np.testing.assert_array_equal(old, new)
                save(f'probe_{t}_{round(a*100):02d}_VALIDATION.json', dict(no_extra_fit_or_solve=True, parent_unchanged=True))
            save(f'PAIR_{t}.json',pair_actual(*endpoints[(t-40)*2:(t-40)*2+2]))
            issued, actions = panel.issue(t)
            if actions[0] != planned: raise ValueError('probe changed MAIN action')
            save(f'ISSUED_{t}.json', list(issued.values()))
            env.step(t, actions)
            nx, nw = env.public(t+1)
            actual, rewards = np.column_stack([nx, nw, actions]), env.released_reward(t+1)
            save(f'RELEASED_{t+1}.json', dict(release_time=t+1, physical=actual.tolist(), reward=rewards.tolist()))
            ne = panel.observe(t, actual, rewards, t+1)
            save(f'AUXILIARY_{t+1}.json', dict(release_time=t+1, auxiliary=ne.tolist()))
            b.snap(folder/f'released_learner_{t+1}.pkl.gz', l)
            mainrows.append(dict(time=t, status='complete',
                full_x_error=float(actual[0, 0]-issued[0]['issued']['full_physical_prediction'][0]),
                full_w_error=float(actual[0, 1]-issued[0]['issued']['full_physical_prediction'][1])))
        status = 'complete'
    except Exception as exc:
        classified = dict(status='fatal', reason='probe_software_failure') if isinstance(exc, fixed.ProbeExecutionFailure) else b.front.classification(exc)
        status = 'fatal' if classified['status'] == 'fatal' else 'MAIN_unavailable'
        failure = dict(time=t, type=type(exc).__name__, reason=str(exc), classification=classified, traceback=traceback.format_exc())
        for row in endpoints:
            if row['status'] in ('main_not_available', 'pending'):
                row.update(status='not_run_after_fatal' if status == 'fatal' else 'main_unavailable', failure=failure)
        try:
            save('FAILURE.json', failure)
        except BaseException as recording_error:
            # Keep the original timeout/write fault as the primary exception.
            # An unhealthy archive cannot accept a second failure record.
            exc.original_error = failure
            exc.recording_error = dict(type=type(recording_error).__name__, reason=str(recording_error))
            try:
                b.ARCHIVE.abort()
            except BaseException as abort_error:
                exc.archive_abort_error = dict(type=type(abort_error).__name__, reason=str(abort_error))
            finally:
                b.ARCHIVE = None
            raise exc from recording_error
    result = dict(spec=spec, candidate=candidate, replication=spec['replicate_id'], status=status, failure=failure,
        H=True, H_status='fixed_public_history_alarm', source_inference='deferred_until_all_CAL_and_observed_complete',
        monitor_rerun=0, prefix_rerun=0, monitor_completed=8, main_source_completed=len(mainrows),
        main_score=measurements.score(mainrows),
        probe_scores={'0.02':measurements.score([r for r in endpoints if r['action']==.02])},
        endpoints=endpoints,date_selections=date_selections,pairs=[pair_actual(*endpoints[i:i+2]) for i in range(0,16,2)],
        counts={k:b.COUNTS[k]-before[k] for k in b.COUNTS})
    try:
        save('RESULT.json', result)
        b.ARCHIVE.finish()
    except BaseException as primary:
        try:
            b.ARCHIVE.abort()
        except BaseException as secondary:
            primary.archive_abort_error = dict(type=type(secondary).__name__, reason=str(secondary))
        raise
    finally:
        b.ARCHIVE = None
    return result

