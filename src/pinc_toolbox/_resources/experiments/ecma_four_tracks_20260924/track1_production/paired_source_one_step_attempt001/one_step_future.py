"""One-date conditional source episode; same H10 Q and paired reset acquisition at date40."""
from pathlib import Path
import importlib.util,sys,traceback
HERE=Path(__file__).resolve().parent
TRACK=HERE.parent
SOURCE=TRACK/'direct_bellman_conditional_source_attempt001/future_episode.py'
_spec=importlib.util.spec_from_file_location('paired_one_step_original_future',SOURCE)
_original=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_original;_spec.loader.exec_module(_original)
select_supported_high=_original.select_supported_high
pair_actual=_original.pair_actual


def signed_step(endpoints):
    """Return high-minus-low FULL-x residual for date40, or None if unavailable."""
    if not isinstance(endpoints,list) or len(endpoints)!=2:return None
    lows=[e for e in endpoints if e.get('action')==.02]
    highs=[e for e in endpoints if e.get('action') is not None and e.get('action')>.02]
    if len(lows)!=1 or len(highs)!=1:return None
    low,high=lows[0],highs[0]
    if low.get('time')!=40 or high.get('time')!=40 or low.get('status')!='complete' or high.get('status')!='complete':return None
    if low.get('release_time')!=41 or high.get('release_time')!=41:return None
    lo=low.get('full_x_error');hi=high.get('full_x_error')
    if not isinstance(lo,(int,float)) or not isinstance(hi,(int,float)):return None
    import math
    if not math.isfinite(lo) or not math.isfinite(hi):raise ValueError('nonfinite completed date40 residual')
    return float(hi)-float(lo)


def episode(spec,panel,env,folder,fixed,b,dispatch,measurements):
    """Run one conditional date, keeping native MAIN and H10 probes unchanged."""
    import numpy as np
    from pinc_toolbox.counterfactual import PublicState
    from pinc_toolbox.g import SupportError
    folder.mkdir();candidate=spec['cell']
    if spec.get('H') is not True:raise ValueError('selected history H=true required')
    t=40;endpoints=[fixed.blank_endpoint(spec,t,a) for a in (.02,None)]
    b.ACTIVE_PROBE_ENDPOINTS=endpoints;before=b.COUNTS.copy();failure=None;status='pending';main=None
    b.ARCHIVE=b.EpisodeWriter(folder/'records.xz',b.BUDGET)
    def save(name,value):b.save(folder/name,value)
    try:
        l=panel.learner
        save('BOUNDARY.json',dict(release_time=40,H=True,candidate=candidate,physical=panel.physical.tolist(),auxiliary=panel.auxiliary.tolist(),
            source_window='date40 only; original H10 direct-Bellman Q retained'))
        b.snap(folder/'released_learner_40.pkl.gz',l)
        b.front.CURRENT.update(seed=None,folder=folder,date=t,stage='conditional_H_future',raw_validated=b.front.validate_history(l.history))
        panel.prepare(t);b.snap(folder/'snapshot_40.pkl.gz',l)
        if l.model.as_of_!=t or l.pending is not None or panel.pending is not None:raise ValueError('MAIN not prepared before probes')
        save('states_40.json',dict(time=t,physical=panel.physical.tolist(),auxiliary=panel.auxiliary.tolist()))
        qs=PublicState(panel.physical[0].copy(),panel.auxiliary[0].copy());planned=float(l.choose(t,qs))
        if not l.supported(t,qs,planned):raise SupportError('MAIN native action unsupported')
        high,mask=select_supported_high(l.model,qs.physical,qs.auxiliary)
        selection=dict(time=t,low_action=.02,candidates=mask,chosen_high_action=high,action_gap=(high-.02 if high is not None else None))
        save('probe_40_SELECTION.json',selection)
        ps=b.pack_state(panel.physical,panel.auxiliary);history_before=l.history.copy(deep=True)
        physical_before,aux_before=panel.physical.copy(),panel.auxiliary.copy()
        core=(l.model.model_.core._C.copy(),l.model.model_.core._Psi.copy());public_before=env.public(t)
        date_actions=(.02,high)
        for i,a in enumerate(date_actions):
            row=endpoints[i];row.update(action=a,candidate_mask=mask,selected_high_action=high,action_gap=selection['action_gap'],evaluation='bellman')
            if a is None:
                row.update(status='support_refusal',failure={'type':'NoSupportedDistinctHighAction','reason':'no eligible high action'},
                    physical_action_supported=False,finite_executed_Q=False,executed_Q=None,
                    query_physical=ps.physical.tolist(),query_auxiliary=ps.auxiliary.tolist())
                continue
            j=int(np.flatnonzero(l.model.action_grid_==a)[0]);values=l.model.q_values(qs.physical[None],qs.auxiliary[None],remaining_horizon=10,evaluation='bellman')[0]
            row.update(physical_action_supported=bool(l.model.model_.support_.mask(qs.physical[None],l.model.action_grid_)[0,j]),
                public_action_supported=bool(l.model.planning_feasible_[j]),finite_executed_Q=bool(np.isfinite(values[j])),
                executed_Q=float(values[j]) if np.isfinite(values[j]) else None,
                query_physical=ps.physical.tolist(),query_auxiliary=ps.auxiliary.tolist(),status='pending')
            b.front.CURRENT.update(seed=None,folder=folder,date=t,stage='conditional_H_future',raw_validated=b.front.validate_history(l.history))
            pl=b.PanelBranchLearner(l);pe=b.PanelEnvironmentAdapter(b.OriginalEnvironmentHost(env));public=pe.public_input(t)
            if public.available_at!=t or public.values[0]!=panel.physical[0,1]:raise ValueError('probe public vintage mismatch')
            science_before=b.COUNTS.copy()
            raw=fixed.execute_endpoint(dispatch,row=row,persist=lambda value,aa=a:save(f'probe_40_{round(aa*100):02d}.json',value),
                learner=pl,environment=pe,state=ps,public=public,time=t,actions=(a,))
            row.update(status=raw['status'],failure=raw['failure'],issued=raw['issued'])
            if raw['status']=='complete':
                if row['physical_action_supported'] is not True or row['public_action_supported'] is not True or row['finite_executed_Q'] is not True:
                    raise ValueError('completed endpoint outside H10 direct-Bellman support')
                issued_probe=raw['issued']['main']
                if raw['issued']['time']!=t or issued_probe['fit_as_of']!=t or issued_probe['action']!=a:
                    raise ValueError('probe issue vintage/action mismatch')
                outcome=raw['outcome'];row.update(next_physical=outcome.next_physical.tolist(),next_auxiliary=raw['next_state'].auxiliary.tolist(),
                    reward=outcome.reward,release_time=outcome.release_time)
                if not np.isfinite(row.get('full_x_error',np.nan)):raise ValueError('completed endpoint has nonfinite FULL-x residual')
            elif raw['status']!='support_refusal':raise ValueError('unknown one-step endpoint state')
            if not np.shares_memory(env.w,env.proxy_all) or not np.shares_memory(pe.host._env.w,pe.host._env.proxy_all):
                raise ValueError('host proxy alias lost')
            for key in ('prefix_G','native_G','Q','reward','constructors'):
                for suffix in ('attempted','succeeded'):
                    if b.COUNTS[key+'_'+suffix]!=science_before[key+'_'+suffix]:raise RuntimeError('probe refitted/solved/constructed')
            if env._next_t!=t or l.pending is not None or panel.pending is not None or not l.history.equals(history_before):raise RuntimeError('probe contaminated MAIN')
            np.testing.assert_array_equal(physical_before,panel.physical);np.testing.assert_array_equal(aux_before,panel.auxiliary)
            np.testing.assert_array_equal(core[0],l.model.model_.core._C);np.testing.assert_array_equal(core[1],l.model.model_.core._Psi)
            for old,new in zip(public_before,env.public(t)):np.testing.assert_array_equal(old,new)
            save(f'probe_40_{round(a*100):02d}_VALIDATION.json',dict(no_extra_fit_or_solve=True,parent_unchanged=True))
        pair=pair_actual(*endpoints);signed=signed_step(endpoints)
        save('PAIR_40.json',dict(pair=pair,signed_step=signed,scope='one date only; no eight-date statistic'))
        issued,actions=panel.issue(t)
        if actions[0]!=planned:raise ValueError('probe changed MAIN action')
        save('ISSUED_40.json',list(issued.values()));env.step(t,actions);nx,nw=env.public(41)
        actual=np.column_stack([nx,nw,actions]);rewards=env.released_reward(41)
        save('RELEASED_41.json',dict(release_time=41,physical=actual.tolist(),reward=rewards.tolist()))
        ne=panel.observe(40,actual,rewards,41);save('AUXILIARY_41.json',dict(release_time=41,auxiliary=ne.tolist()))
        b.snap(folder/'released_learner_41.pkl.gz',l)
        main=dict(time=40,release_time=41,action=float(actions[0]),next_physical=actual[0].tolist(),
            full_x_error=float(actual[0,0]-issued[0]['issued']['full_physical_prediction'][0]))
        save('MAIN_40.json',main);status='complete'
    except Exception as exc:
        failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
        status='fatal' if isinstance(exc,fixed.ProbeExecutionFailure) else ('MAIN_unavailable' if b.front.classification(exc)['status']!='fatal' else 'fatal')
        for row in endpoints:
            if row.get('status') in ('pending','main_not_available'):row.update(status='not_run_after_fatal' if status=='fatal' else 'main_unavailable',failure=failure)
        save('FAILURE.json',failure)
    result=dict(spec=spec,candidate=candidate,status=status,failure=failure,H=True,H_status='fixed_public_history_alarm',
        source_window='date40 only',q_evaluation='bellman',remaining_horizon=10,main_completed=main is not None,
        endpoints=endpoints,pair=pair_actual(*endpoints),signed_step=signed_step(endpoints),counts={k:b.COUNTS[k]-before[k] for k in b.COUNTS})
    try:save('RESULT.json',result);b.ARCHIVE.finish()
    except BaseException:b.ARCHIVE.abort();raise
    finally:b.ARCHIVE=None
    return result
