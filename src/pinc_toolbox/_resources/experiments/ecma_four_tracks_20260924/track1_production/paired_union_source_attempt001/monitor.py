"""Original reviewed H8 monitor loop, without future selection.
Copied unchanged from fresh_h8 monitor_adapter functions.
"""
import traceback
import math

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
            if union_alarm(scores,threshold):held=(panel,env)
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

def exception_status(b, exc, phase):
    # SupportError raised by a host/recorder is not scientific unavailability.
    if phase not in ('initialization','prepare','choose_support','issue'):
        return 'fatal',dict(status='fatal',reason='software_clock_nonfinite_or_resource')
    cause=b.front.classification(exc)
    if cause['status'] not in ('support_unavailable','identification_unavailable'):
        return 'fatal',cause
    suffix='support_refusal' if cause['status']=='support_unavailable' else 'identification_unavailable'
    return ('initialization_' if phase=='initialization' else 'main_')+suffix,cause


def paired_rank(value,bank):
    if value is None or len(bank)!=119:return None
    if not math.isfinite(value) or not all(math.isfinite(x) for x in bank):
        raise ValueError("nonfinite paired rank input")
    return min(1.,2*min((1+sum(x<=value for x in bank))/120,
                       (1+sum(x>=value for x in bank))/120))

def union_alarm(scores,threshold):
    p=paired_rank(scores.get('paired_signed_sum'),threshold.get('paired_bank',[]))
    return (scores['serial']>threshold['serial'] or scores['scalar']>threshold['scalar']
            or (p is not None and p<=1/60))

def install(acquisition,b,dispatch,fixed,future):
    global monitor_loop
    def run(b_arg,l,panel,env,folder,events,position):
        captured={};original_save=b.save
        def save(path,value):
            if path.name=='PAIRED_PROBE_ENDPOINTS.json':captured['endpoints']=value
            return original_save(path,value)
        b.save=save
        try: scores=acquisition.monitor_loop(b_arg,l,panel,env,folder,events,position,dispatch,fixed,future)
        finally:b.save=original_save
        endpoints=captured.get('endpoints',[])
        value=None
        if len(endpoints)==16 and all(r['status']=='complete' for r in endpoints):
            value=sum(endpoints[2*j+1]['full_x_error']-endpoints[2*j]['full_x_error'] for j in range(8))
        if value is not None and not math.isfinite(value):raise ValueError('nonfinite paired residual sum')
        scores=dict(scores,paired_signed_sum=value)
        b.save(folder/'PAIRED_MONITOR_SCORE.json',dict(signed_sum=value,complete=value is not None))
        return scores
    monitor_loop=run
