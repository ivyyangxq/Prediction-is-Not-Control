"""Original reviewed H8 monitor loop, without future selection.
Copied unchanged from fresh_h8 monitor_adapter functions.
"""
import traceback

def episode(spec, out, dgp, b, threshold):
    """Return public monitor receipt and a private selected-only evaluator pair."""
    folder=out/f"monitor_{spec['ordinal']:03d}";folder.mkdir()
    before=b.COUNTS.copy();events=[];scores=None;failure=None;held=None
    position=dict(phase='prefix',time=None)
    b.front.CURRENT.update(seed=spec['seed'],folder=folder,date=None,stage='prefix',raw_validated=False)
    b.ARCHIVE=b.EpisodeWriter(folder/'records.xz',b.BUDGET)
    try:
        try:
            b.storage();history,env=b.prefix._prefix(dgp,spec['seed'],spec['source'],1.)
            if env.horizon!=64 or env.switch_date!=32 or env._next_t!=32:
                raise ValueError('prefix host clock/config mismatch')
            b.ARCHIVE.raw('prefix.csv',history.to_csv(index=False,float_format='%.17g').encode())
            position['phase']='initialization'
            learner,x,e,initial=b.initialize_checked(history)
            position['phase']='recording';b.save(folder/'PREFIX_INITIALIZATION.json',initial)
            panel=b.front.Panel(learner,x,e)
            scores=monitor_loop(b,learner,panel,env,folder,events,position)
            status='available'
            # Selection uses only the completed public statistic. No future query.
            if scores['serial']>threshold:held=(panel,env)
        except Exception as exc:
            status,cause=exception_status(b,exc,position['phase'])
            failure=dict(**position,type=type(exc).__name__,reason=str(exc),classification=cause,traceback=traceback.format_exc())
            b.save(folder/'FAILURE.json',failure)
        result=dict(status=status,scores=scores if status=='available' else None,
            completed_dates=len(events),failure=failure,archive=str(folder/'records.xz'),
            counts={k:b.COUNTS[k]-before[k] for k in b.COUNTS})
        b.save(folder/'RESULT.json',result);b.ARCHIVE.finish()
        return result,held
    except BaseException as primary:
        try:b.ARCHIVE.abort()
        except BaseException as secondary:
            primary.archive_abort_error=dict(type=type(secondary).__name__,reason=str(secondary))
        raise
    finally:b.ARCHIVE=None

def monitor_loop(b, l, panel, env, folder, events, position):
    """Shared path retained from fixed_action_probe_dev_attempt001.episode."""
    import numpy as np
    from pinc_toolbox.counterfactual import PublicState
    from pinc_toolbox.g import SupportError
    def save(name,value):b.save(folder/name,value)
    for t in range(32,40):
        position.update(phase='prepare',time=t)
        b.front.CURRENT.update(folder=folder,date=t,stage='monitor',raw_validated=b.front.validate_history(l.history))
        if not np.isfinite(panel.physical).all() or not np.isfinite(panel.auxiliary).all():
            raise ValueError('nonfinite pre-prepare panel')
        panel.prepare(t)
        position['phase']='recording';b.snap(folder/f'snapshot_{t}.pkl.gz',l)
        if l.model.as_of_!=t or l.pending is not None or panel.pending is not None:
            raise ValueError('MAIN not prepared before issue')
        save(f'states_{t}.json',dict(time=t,physical=panel.physical.tolist(),auxiliary=panel.auxiliary.tolist()))
        position['phase']='choose_support'
        qs=PublicState(panel.physical[0].copy(),panel.auxiliary[0].copy())
        planned_main=float(l.choose(t,qs))
        if not l.supported(t,qs,planned_main):raise SupportError('MAIN native action unsupported')
        position['phase']='issue'
        issued,actions=panel.issue(t)
        if actions[0]!=planned_main:raise ValueError('MAIN action changed before issue')
        position['phase']='recording';save(f'ISSUED_{t}.json',list(issued.values()))
        position['phase']='host'
        env.step(t,actions);nx,nw=env.public(t+1)
        actual=np.column_stack([nx,nw,actions]);rewards=env.released_reward(t+1)
        position['phase']='recording'
        save(f'RELEASED_{t+1}.json',dict(release_time=t+1,physical=actual.tolist(),reward=rewards.tolist()))
        position['phase']='observe';ne=panel.observe(t,actual,rewards,t+1)
        position['phase']='recording'
        save(f'AUXILIARY_{t+1}.json',dict(release_time=t+1,auxiliary=ne.tolist()))
        b.snap(folder/f'released_learner_{t+1}.pkl.gz',l)
        events.append(dict(time=t,release_time=t+1,issued=issued[0]['issued'],
                           physical=issued[0]['physical'],auxiliary=issued[0]['auxiliary'],
                           action=float(actions[0]),next_physical=actual[0].tolist(),
                           next_auxiliary=ne[0].tolist(),reward=float(rewards[0])))
    position['phase']='scoring'
    scores=b.front.score_main(events,8)
    save('MONITOR_SCORE.json',scores)
    save('MONITOR_EVENTS.json',events)
    return scores

def exception_status(b, exc, phase):
    # SupportError raised by a host/recorder is not scientific unavailability.
    if phase not in ('initialization','prepare','choose_support','issue'):
        return 'fatal',dict(status='fatal',reason='software_clock_nonfinite_or_resource')
    cause=b.front.classification(exc)
    if cause['status'] not in ('support_unavailable','identification_unavailable'):
        return 'fatal',cause
    suffix='support_refusal' if cause['status']=='support_unavailable' else 'identification_unavailable'
    return ('initialization_' if phase=='initialization' else 'main_')+suffix,cause
