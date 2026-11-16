"""Finite predictable SimpleGQ detector followed by heldout-reference source work."""
from .two_stage_report import run_two_stage_report
from .certified_source_workflow_backend import CertifiedSourceEvaluator


def run_online_gq_two_stage(*, detection_reference, detection_windows,
                           source_reference, source_window, sigma, source_deadline,
                           alpha_detection_reference=.005, alpha_detection_noise=.005,
                           alpha_source_reference=.025, alpha_source_noise=.025,
                           latent_bound=.6, model_bound=2., assumptions=None):
    """One first-alarm episode; all supplied detection windows are predeclared.

    Uses the same actual SimpleGQ joint-design radius for detection and source.
    Source old reference must never influence detector policies, eta or inputs.
    Array IDs reject obvious reuse but cannot verify that scientific premise.
    """
    from .predictable_gq_two_stage import prepare_predictable_gq_region, run_predictable_gq_two_stage
    if not isinstance(detection_windows, list) or not detection_windows:
        raise ValueError('a nonempty predeclared detection calendar is required')
    times = [max(w['dates']) for w in detection_windows]
    if any(t >= u for t,u in zip(times,times[1:])):
        raise ValueError('detection endpoints must be strictly increasing')
    old_ids = detection_reference['record_ids']
    detector_ids = [i for w in detection_windows for i in w['record_ids']]
    if set(source_reference['record_ids']) & (set(old_ids) | set(detector_ids)):
        raise ValueError('source reference overlaps the full detection calendar')
    if set(old_ids) & set(detector_ids):
        raise ValueError('detector reference overlaps current detector rows')
    if max(detection_reference['dates']) >= min(min(w['dates']) for w in detection_windows):
        raise ValueError('detector reference must precede every detection window')
    records, failures = [], []
    alarm_time = None
    for time, window in zip(times, detection_windows):
        try:
            measured = prepare_predictable_gq_region(
                detection_reference, window, sigma=sigma,
                alpha_reference=alpha_detection_reference,
                alpha_noise=alpha_detection_noise, calendar_size=len(times),
                latent_bound=latent_bound, model_bound=model_bound)
            if measured['readiness']['status'] != 'ready':
                records.append({'time':time,'status':'unavailable','measurement':measured})
                failures.append({'time':time,'reason':'detector_region_unavailable'})
                continue
            c0 = CertifiedSourceEvaluator(measured['source_spec'])('C0')
            status = {'excluded':'alarm','compatible':'no_alarm'}.get(c0['status'],'unresolved')
            records.append({'time':time,'status':status,'C0':c0,'measurement':measured})
            if status == 'unresolved':
                failures.append({'time':time,'reason':'detector_C0_unresolved'})
            if status == 'alarm':
                alarm_time=time
                break
        except Exception as exc:
            records.append({'time':time,'status':'unavailable','reason':str(exc)})
            failures.append({'time':time,'reason':str(exc)})
    stage1 = {'status':'alarm' if alarm_time is not None else 'unavailable' if failures else 'no_alarm',
              'alarm_time':alarm_time, 'first_alarm':alarm_time,
              'event':'first certified C0 exclusion on predeclared predictable GQ calendar',
              'records':records, 'calendar':times,
              'not_visited_times':times[len(records):],
              'provenance':{'record_ids':detector_ids,'detector_reference_ids':old_ids}}
    context={'route':'online_predictable_simple_gq_two_stage','source_deadline':source_deadline}
    if alarm_time is None:
        report=run_two_stage_report(stage1,context=context)
    else:
        try:
            report=run_predictable_gq_two_stage(
                stage1=stage1,reference=source_reference,future=source_window,sigma=sigma,
                alpha_reference=alpha_source_reference,alpha_noise=alpha_source_noise,
                calendar_size=1,source_deadline=source_deadline,
                latent_bound=latent_bound,model_bound=model_bound,assumptions=assumptions)
            report['context']['route']='online_predictable_simple_gq_two_stage'
        except Exception as exc:
            report=run_two_stage_report(stage1,context=context,
                readiness={'status':'unavailable','reason':str(exc)})
    report['completion']['failures'] = failures + report['completion']['failures']
    report['errors']['detection']['type_i']['target_budget']=alpha_detection_reference+alpha_detection_noise
    # The source assumption list does not automatically validate the detector.
    report['reliability']['detection_theorem_scope']=(
        'fixed Gaussian detector reference; predictable conditional subGaussian current noise; '
        'correct joint GQ dictionary; entire supplied finite calendar; sound numerics')
    if (assumptions or {}).get('detector_predictable_region_assumptions') is True:
        report['errors']['detection']['type_i'].update(
            status='theorem_under_declared_assumptions',
            proved_bound=alpha_detection_reference+alpha_detection_noise)
    detector_bound=report['errors']['detection']['type_i']['proved_bound']
    source_bound=report['errors'].get('source_joint',{}).get('type_i',{}).get('proved_bound')
    joint=report['errors']['joint']['type_i']
    joint['target_budget']=alpha_detection_reference+alpha_detection_noise+alpha_source_reference+alpha_source_noise
    if detector_bound is not None and source_bound is not None:
        joint.update(status='theorem_under_declared_assumptions',
                     proved_bound=min(1.,detector_bound+source_bound))
    else:
        joint.update(status='not_established',proved_bound=None)
    return report
