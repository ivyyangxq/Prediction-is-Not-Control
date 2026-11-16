"""Optional monitor acquisition adapter; original MAIN loop/order/scoring stay intact."""
from pathlib import Path
import importlib.util, sys
HERE=Path(__file__).resolve().parent
TRACK=HERE.parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def endpoint_information(row, state_x):
    """Canonical expanded-transcript row, including a completed single endpoint."""
    return dict(path=0,branch='paired_reset_probe',time=row.get('time'),release_time=row.get('release_time'),
        state_x=float(state_x),action=row.get('action'),status=row.get('status'),
        next_x=(float(row['next_physical'][0]) if row.get('next_physical') is not None else None),
        next_w=(float(row['next_physical'][1]) if row.get('next_physical') is not None else None),
        full_x_error=row.get('full_x_error'),failure=row.get('failure'))


def install(monitor_module, b, dispatch, fixed, future):
    """Patch only the loaded campaign module; its episode and first-H logic are unchanged."""
    def loop(b_arg,l,panel,env,folder,events,position):
        if b_arg is not b: raise RuntimeError('unexpected backend for paired monitor')
        return monitor_loop(b,l,panel,env,folder,events,position,dispatch,fixed,future)
    monitor_module.monitor_loop=loop


def monitor_loop(b,l,panel,env,folder,events,position,dispatch,fixed,future):
    import numpy as np
    from pinc_toolbox.counterfactual import PublicState
    from pinc_toolbox.g import SupportError
    def save(name,value): b.save(folder/name,value)
    all_endpoints=[]; expanded=[]; pairs=[]; main_observations=[]
    spec=dict(seed=b.front.CURRENT.get('seed'),cell='monitor',source='monitor')
    for t in range(32,40):
        position.update(phase='prepare',time=t)
        b.front.CURRENT.update(folder=folder,date=t,stage='monitor',raw_validated=b.front.validate_history(l.history))
        if not np.isfinite(panel.physical).all() or not np.isfinite(panel.auxiliary).all(): raise ValueError('nonfinite pre-prepare panel')
        panel.prepare(t);position['phase']='recording';b.snap(folder/f'snapshot_{t}.pkl.gz',l)
        if l.model.as_of_!=t or l.pending is not None or panel.pending is not None: raise ValueError('MAIN not prepared before issue')
        save(f'states_{t}.json',dict(time=t,physical=panel.physical.tolist(),auxiliary=panel.auxiliary.tolist()))
        position['phase']='choose_support';qs=PublicState(panel.physical[0].copy(),panel.auxiliary[0].copy())
        planned=float(l.choose(t,qs))
        if not l.supported(t,qs,planned): raise SupportError('MAIN native action unsupported')

        # Acquire reset probes against the prepared MAIN state before the untouched MAIN issue.
        high,candidates=future.select_supported_high(l.model,qs.physical,qs.auxiliary)
        state_x=float(qs.physical[0]); ps=b.pack_state(panel.physical,panel.auxiliary)
        pe=b.PanelEnvironmentAdapter(b.OriginalEnvironmentHost(env)); public=pe.public_input(t)
        if public.available_at!=t: raise ValueError('probe public vintage mismatch')
        date_rows=[]
        for action in (.02,high):
            row=fixed.blank_endpoint(spec,t,action)
            row.update(path=0,branch='paired_reset_probe',state_x=state_x,selected_high_action=high,
                       candidate_mask=candidates,evaluation='bellman',release_time=None)
            if action is None:
                row.update(status='support_refusal',failure=dict(type='NoSupportedDistinctHighAction',reason='no eligible high action'))
            else:
                j=int(np.flatnonzero(l.model.action_grid_==action)[0])
                vals=l.model.q_values(qs.physical[None],qs.auxiliary[None],remaining_horizon=10,evaluation='bellman')[0]
                row.update(physical_action_supported=bool(l.model.model_.support_.mask(qs.physical[None],l.model.action_grid_)[0,j]),
                    public_action_supported=bool(l.model.planning_feasible_[j]),finite_executed_Q=bool(np.isfinite(vals[j])),
                    executed_Q=float(vals[j]) if np.isfinite(vals[j]) else None,
                    query_physical=ps.physical.tolist(),query_auxiliary=ps.auxiliary.tolist())
                pl=b.PanelBranchLearner(l)
                before=l.history.copy(deep=True);physical_before=panel.physical.copy();aux_before=panel.auxiliary.copy()
                core=(l.model.model_.core._C.copy(),l.model.model_.core._Psi.copy());public_before=env.public(t);counts=b.COUNTS.copy()
                try:
                    raw=fixed.execute_endpoint(dispatch,row=row,
                        persist=lambda value,tt=t,aa=action:save(f'PAIR_ENDPOINT_{tt}_{"LOW" if aa==.02 else "HIGH"}.json',value),
                        learner=pl,environment=pe,state=ps,public=public,time=t,actions=(action,))
                    row.update(status=raw['status'],failure=raw['failure'],issued=raw['issued'])
                    if raw['status']=='complete':
                        out=raw['outcome'];row.update(next_physical=out.next_physical.tolist(),release_time=out.release_time,
                            reward=out.reward,next_auxiliary=raw['next_state'].auxiliary.tolist())
                    elif raw['status']!='support_refusal': raise RuntimeError('unknown probe endpoint status')
                except fixed.ProbeExecutionFailure:
                    raise
                if not l.history.equals(before) or env._next_t!=t or l.pending is not None or panel.pending is not None:
                    raise RuntimeError('paired probe contaminated MAIN')
                np.testing.assert_array_equal(physical_before,panel.physical);np.testing.assert_array_equal(aux_before,panel.auxiliary)
                np.testing.assert_array_equal(core[0],l.model.model_.core._C);np.testing.assert_array_equal(core[1],l.model.model_.core._Psi)
                for key in ('prefix_G','native_G','Q','reward','constructors'):
                    for suffix in ('attempted','succeeded'):
                        if b.COUNTS[key+'_'+suffix]!=counts[key+'_'+suffix]: raise RuntimeError('probe fitted/solved/constructed')
                for old,new in zip(public_before,env.public(t)):np.testing.assert_array_equal(old,new)
            date_rows.append(row);all_endpoints.append(row);expanded_row=endpoint_information(row,state_x);expanded.append(expanded_row)
            save(f'EXPANDED_ENDPOINT_{t}_{"LOW" if action==.02 else "HIGH"}.json',expanded_row)
        low,hi=date_rows
        pair=dict(path=0,branch='paired_reset_probe',time=t,state_x=state_x,low_action=low.get('action'),high_action=hi.get('action'),
                  low_status=low.get('status'),high_status=hi.get('status'),
                  low_next_x=(low['next_physical'][0] if low.get('next_physical') is not None else None),
                  high_next_x=(hi['next_physical'][0] if hi.get('next_physical') is not None else None),
                  release_time=(low.get('release_time') if low.get('status')=='complete' else hi.get('release_time')),
                  status='complete' if low.get('status')=='complete' and hi.get('status')=='complete' else 'partial_or_unavailable')
        pairs.append(pair);save(f'PAIR_{t}.json',dict(pair=pair,endpoints=date_rows))

        # Existing MAIN sequence below is retained verbatim in operation and score input.
        position['phase']='issue';issued,actions=panel.issue(t)
        if actions[0]!=planned: raise ValueError('probe changed MAIN action')
        position['phase']='recording';save(f'ISSUED_{t}.json',list(issued.values()))
        position['phase']='host';env.step(t,actions);nx,nw=env.public(t+1)
        actual=np.column_stack([nx,nw,actions]);rewards=env.released_reward(t+1)
        position['phase']='recording';save(f'RELEASED_{t+1}.json',dict(release_time=t+1,physical=actual.tolist(),reward=rewards.tolist()))
        position['phase']='observe';ne=panel.observe(t,actual,rewards,t+1);position['phase']='recording'
        save(f'AUXILIARY_{t+1}.json',dict(release_time=t+1,auxiliary=ne.tolist()));b.snap(folder/f'released_learner_{t+1}.pkl.gz',l)
        main_observations.append(dict(path=0,branch='MAIN',time=t,release_time=t+1,state_x=state_x,action=float(actions[0]),next_x=float(actual[0,0]),full_x_error=float(actual[0,0]-issued[0]['issued']['full_physical_prediction'][0])))
        events.append(dict(time=t,release_time=t+1,issued=issued[0]['issued'],physical=issued[0]['physical'],
            auxiliary=issued[0]['auxiliary'],action=float(actions[0]),next_physical=actual[0].tolist(),
            next_auxiliary=ne[0].tolist(),reward=float(rewards[0])))
    position['phase']='scoring';scores=b.front.score_main(events,8)
    save('MONITOR_SCORE.json',scores);save('MONITOR_EVENTS.json',events)
    save('PAIRED_PROBE_ENDPOINTS.json',expanded);save('PAIRED_PROBE_PAIRS.json',pairs);save('MAIN_OBSERVATIONS.json',main_observations)
    return scores
