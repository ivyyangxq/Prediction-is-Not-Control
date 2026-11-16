"""Research-checkout adapter for the registered round/switch32 H10 workflow.

This is deliberately scoped to the controlled common-innovation benchmark.
It is not an ordinary-panel source procedure. Generic panel runs do not import
this module. The original monitor/calibration and action-curve qualifier are
executed, not replaced by a detector threshold or stored MODEL label.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from functools import wraps
import importlib.util
import io
import gzip
import json
import lzma
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from .product import PanelSession, clean, digest, frame_digest


def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


R0_REWARD={'expression':'x - .25*x*x - 2.025*a - .5*a*a','inputs':{'x':'state','a':'action'},'parameters':{}}


def canonical_panel_digest(history,cutoff=40):
    columns=['path_id','period','release_period','state','w','previous_action','action','next_state','next_w']
    return frame_digest(history.loc[history.release_period<=cutoff,columns].sort_values(['path_id','period']).reset_index(drop=True))


def _extend_original_host(env,module,end):
    """Append independent same-law primitives without redrawing the 64 prefix."""
    horizon=end+1;original=env.horizon
    if horizon<=original:return
    parent=module._PARENT
    rng=np.random.default_rng(np.random.SeedSequence([env.seed,871901]))
    extra=horizon-original
    for name,scale in [('state_noise',parent.STATE_INNOVATION_SD),
                       ('latent_noise',parent.LATENT_INNOVATION_SD),
                       ('measurement_noise',parent.MEASUREMENT_NOISE_SD),
                       ('reward_noise',env.reward_noise_sd)]:
        old=getattr(env,name);shape=list(old.shape);shape[1]=extra
        tail=rng.normal(size=shape)*scale
        setattr(env,name,np.concatenate([old,tail],axis=1))
    for name in ('state','latent','proxy_all','_baseline_latent','action','reward','original_reward','reward_observation'):
        old=getattr(env,name);shape=list(old.shape);shape[1]=extra
        setattr(env,name,np.concatenate([old,np.full(shape,np.nan)],axis=1))
    env.g_profile=np.r_[env.g_profile,np.repeat(env.g_profile[-1],extra)]
    env.g_t=env.g_profile;env.w=env.proxy_all[:,:,0];env.horizon=horizon
    env.metadata.update(horizon=horizon,original_prefix_horizon=original,
        extension='independent same-law tail; all original primitive tapes unchanged')


def _public_archive(path):
    # Only trusted, locally generated scientific archives. Never user pickle.
    result={}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',DeprecationWarning)
        with lzma.open(path,'rb') as stream:
            while True:
                try:event=pickle.load(stream)
                except EOFError:break
                if event[0]=='end':break
                if event[0]=='raw' and (event[1]=='prefix.csv' or event[1].endswith('.json')):
                    result[event[1]]=event[2]
    return result


def _isolated_instrumentation(function):
    @wraps(function)
    def run(*args,**kwargs):
        from . import native_gq, rnd2_counterfactual, g
        modules=(native_gq,rnd2_counterfactual,g)
        saved=[dict(vars(m)) for m in modules]
        try:
            return function(*args,**kwargs)
        finally:
            # Historical monitoring installs archive/count/audit wrappers in
            # shared package globals. They must not leak into PanelSession.
            for module,before in zip(modules,saved):
                for key,value in before.items():
                    if not key.startswith('__'):setattr(module,key,value)
    return run


class LiveCurveEvidence:
    """Internally generated live source result, with actual alarm and endpoints."""
    def __init__(self,result,request,binding):
        self.result=deepcopy(result);self.request=deepcopy(request)
        self.binding=deepcopy(binding)
        self.publication_utc=datetime.now(timezone.utc).isoformat()
        self.applicability={'scope':'controlled RND2 same-state/latent/innovation reset benchmark',
            'premises':['original H10 first-layer protocol and matching calibration before release40',
                        'same actual state, latent record and innovation for three supported reset actions',
                        'every admitted no-model law has action curvature -2',
                        'endpoint arithmetic error <=2.5e-10 (conditional numerical premise)'],
            'repair_applicable':True}

    def snapshot(self):
        r=self.result
        return clean(dict(binding=self.binding,applicability=self.applicability,
            as_of=40,request_digest=digest(self.request),stage1=r['stage1'],
            model_involvement=r['source_outcome']=='model_involvement',
            source_outcome=r['source_outcome'],status='resolved' if r['source_outcome']=='model_involvement' else 'unresolved',
            native_snapshot=r,publication_kind='fresh_controlled_history',
            publication_utc=self.publication_utc))


@_isolated_instrumentation
def fresh_round_monitor(root,row,out,*,end=95):
    """Existing actual monitor, original CAL, and live action-curve source test."""
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    round_module=_load('product_round_protocol',root/'experiments/round_coefficient_precision128_20261110/run_round.py')
    precision,driver=round_module.build_worker()
    base=driver._load('paired16_science_adapter',driver.SOURCE_RUN/'science_adapter.py')
    base.HERE=driver.TRACK/'direct_bellman_conditional_source_attempt001'
    base.monitor=driver._load('paired16_monitor',driver.HERE/'monitor.py')
    base.OUTPUT_BYTES=128*1024**2;base.TEMP_BYTES=0;base.RESERVE_BYTES=2*1024**3
    base.LIMIT=dict(prefix_G=3200,native_G=8192,Q=4096,reward=4096,constructors=128,
                    host_calls=32768,transitions=1048576,conditional_initializations=256)
    alloc=precision.allocation_for(row)
    thresholds=driver.load_corrected_stage1()
    expanded=driver._load('paired16_expanded',driver.HERE/'expanded_condition.py')
    science=expanded.science_class(base)(out,alloc,thresholds)
    acquisition=driver._load('paired16_acquisition',driver.TRACK/'paired_monitor_acquisition_attempt001/paired_monitor.py')
    base.monitor.install(acquisition,science.b,science.dispatch,science.fixed,science.future)
    monitored=science.monitor(alloc['monitor_proposals'][0])
    if monitored['status']=='fatal':raise RuntimeError(str(monitored['failure']))
    alarm=None if monitored['status']!='available' else base.monitor.union_alarm(monitored['scores'],thresholds)
    archive=Path(monitored['archive'])
    curve=_load('product_registered_curve',root/'experiments/round_coefficient_precision128_20261110/source_curvature_continuation_20261112/saved_readback.py')
    raw=curve.archive_raw(archive)
    qualified=[];unavailable=[]
    if alarm:
        for date in range(32,40):
            point,reason=curve.qualify(raw,date)
            if point:qualified.append(point)
            else:unavailable.append({'date':date,'reason':reason})
    violation=[x['date'] for x in qualified if abs(Decimal(x['discrepancy']))>curve.TOL]
    source=('not_entered_no_alarm' if alarm is False else 'source_unavailable_monitor' if alarm is None else
            'model_involvement' if violation else 'unresolved' if qualified else 'source_unavailable_no_qualified_triples')
    result=clean(dict(stage1={'status':monitored['status'],'alarm':alarm,'scores':monitored.get('scores'),
                             'failure':monitored.get('failure'),'thresholds':thresholds,
                             'method':'original scalar/serial/paired H10 union; original accepted CAL banks'},
        source_outcome=source,qualified_triples=qualified,unavailable_dates=unavailable,
        violating_dates=violation,last_input_release=40 if alarm else None,
        method='existing common_innovation_action_curve_exclusion',arithmetic_allowance=str(curve.TOL),
        latent_component='not_executed',archive=str(archive),
        scope='controlled reset benchmark, not ordinary-panel identification'))
    (out/'LIVE_SOURCE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    # Reconstruct an evaluator from the assigned primitive stream, replaying only
    # recorded executed actions. No private state is sent to the learner.
    public=_public_archive(archive)
    history=pd.read_csv(io.BytesIO(public['prefix.csv']),float_precision='round_trip')
    original=_load('product_fresh_evaluator',root/'DGP/RND/02_state_dependent_local/model_source_v004/environment.py')
    dgp=round_module.round_environment(original)
    env=dgp.Environment(row['seed'],base.SOURCES[row['candidate']],paths=32,horizon=64,switch_date=32,strength=1.,profile='step')
    if row['candidate']=='CL_Q100':
        # The retained source catalogue defines Q100 as FSTAR, whereas the
        # raw LATENT_F host's default interpolation coefficient is .69375.
        env.g_profile[:]=np.asarray(env.g_profile)/.69375
        env.g_t=env.g_profile
    # Restore the public first-proxy view explicitly, never a serialized copy.
    env.w=env.proxy_all[:,:,0]
    _extend_original_host(env,dgp,end)
    for date in range(32):
        rows=history.loc[history.period==date].sort_values('path_id')
        np.testing.assert_allclose(env.state[:,date],rows.state,rtol=0,atol=1e-12)
        env.step(date,rows.action.to_numpy(float))
        np.testing.assert_allclose(env.state[:,date+1],rows.next_state,rtol=0,atol=1e-12)
        np.testing.assert_allclose(env.w[:,date+1],rows.next_w,rtol=0,atol=1e-12)
    extra=[]
    # If deployment refused early, external logging still continues; source
    # remains unavailable. Its subsequent records do not refill the detector.
    private_rng=np.random.default_rng(np.random.SeedSequence([row['seed'],571901]))
    for date in range(32,end+1):
        issued=public.get(f'ISSUED_{date}.json') if date<40 else None
        records=json.loads(issued) if issued is not None else []
        recorded={r['path_id']:r for r in records}
        if len(recorded)==32:
            actions=np.array([recorded[i]['action'] for i in range(32)])
        else:
            actions=private_rng.choice(np.linspace(0,.1,11),size=32)
        state=env.state[:,date].copy();w=env.w[:,date].copy();previous=env.action[:,date-1].copy()
        env.step(date,actions)
        released=public.get(f'RELEASED_{date+1}.json') if date<40 else None
        if released is not None:
            physical=np.asarray(json.loads(released)['physical'])
            np.testing.assert_allclose(physical[:,:2],np.column_stack([env.state[:,date+1],env.w[:,date+1]]),rtol=0,atol=1e-12)
        for i in range(32):
            extra.append(dict(path_id=i,period=date,release_period=date+1,state=state[i],w=w[i],
                previous_action=previous[i],action=actions[i],next_state=env.state[i,date+1],next_w=env.w[i,date+1]))
    full=pd.concat([history.drop(columns=['reward'],errors='ignore'),pd.DataFrame(extra)],ignore_index=True)
    full=full[['path_id','period','release_period','state','w','previous_action','action','next_state','next_w']]
    full.attrs={'input_semantics':'original RND2 path-specific public proxy w; not a common macro W',
                'source_scope':'controlled common-innovation reset benchmark',
                'post40_collection':'independent external randomized logged actions; private random stream excluded from learner inputs'}
    result.update(public_stream_prefix_digest=canonical_panel_digest(full),reward_digest=digest(R0_REWARD),
        publication_kind='fresh_controlled_history',last_generated_release=end+1,
        prefix_preservation='original horizon64 primitives retained; independent same-law extension after64')
    result['runtime_code_files']=sorted({str(Path(module.__file__).resolve().relative_to(root.resolve()))
        for module in tuple(sys.modules.values()) if getattr(module,'__file__',None)
        and str(module.__file__).endswith('.py') and Path(module.__file__).resolve().is_relative_to(root.resolve())})
    (out/'LIVE_SOURCE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result,full


def run_controlled_history(root,row,out,*,end=95,value_bank=None,compact_output=False):
    """Full fresh detect/source -> explicit update -> support, one assigned history.

    All branches use PanelSession, the same engine as the notebook. The source
    window is unchanged; only post40 additional action readability differs.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    source,raw=fresh_round_monitor(root,row,out/'detect_source',end=max(63,end))
    raw.to_csv(out/('EXTERNAL_PUBLIC_STREAM.csv.gz' if compact_output else 'EXTERNAL_PUBLIC_STREAM.csv'),
               index=False,float_format='%.17g',compression='gzip' if compact_output else None)
    initial=raw.loc[(raw.period>=0)&(raw.period<24)]
    scale=initial[['state','w','previous_action','action']].std().to_numpy(float)
    scale=np.maximum(scale,1e-8).tolist()
    target=raw.loc[(raw.path_id==0)&(raw.period==24)].iloc[0]
    branches={}
    reports={}
    for arm in ('read_new_actions','withhold_new_actions','fixed_history_reference'):
        history=raw.copy(deep=True)
        if arm=='withhold_new_actions':
            history.loc[(history.path_id!=0)&(history.period>=40),['action','previous_action']]=np.nan
        # Neither raw seed nor the private logging realization/seed is a learner input.
        settings=dict(mode='Online (Observed Inputs)',update_policy='report_guided',
            learn_start=0,learn_end=24,start=24,end=end,path=0,window=24,warmup_as_of=8,initial_full=True,retain_warmup_rows=True,
            horizon=10,discount=.95,grid=np.linspace(0,.1,11).tolist(),candidates=np.linspace(0,.1,11).tolist(),
            reward=deepcopy(R0_REWARD),
            reference_scale=scale,reference_query=[target.state,target.w,target.previous_action],eta=None,
            nodes_per_memory=5,noise_nodes=4)
        session=PanelSession(history,settings)
        evidence=LiveCurveEvidence(source,{'method':source['method'],'initial_learning_digest':frame_digest(initial),
            'public_stream_prefix_digest':source['public_stream_prefix_digest']},session.identity)
        session.evidence=evidence
        session.source_reports.append(evidence.snapshot())
        while session.next_date<40:session.advance()
        if evidence.snapshot()['model_involvement'] and arm!='fixed_history_reference':
            session.decide_repair(True,method='rolling_full_RA_GQ')
        else:
            session.decide_repair(False)
        report=session.run()
        reports[arm]=report
        session.export(out/arm,compress=compact_output)
        branches[arm]={'decisions':len(report['timeline']),
            'available':sum(r['status']=='available' for r in report['timeline']),
            'available_post40':sum(r['status']=='available' and r['date']>=40 for r in report['timeline']),
            'statuses':[r['status'] for r in report['timeline']],
            'recommendations':[r['recommended_action'] for r in report['timeline']],
            'reward_digest':session.identity['reward'],
            'value_qualification':'unavailable; no counterfactual economic interval supplied'}
    value_scores=None
    if value_bank is not None:
        from .controlled_value import score_recommendations
        value_scores=score_recommendations(root,raw,source['archive'],row['candidate'],reports,value_bank)
        if compact_output:
            with gzip.open(out/'EVALUATOR_ONLY_VALUE_SCORES.json.gz','wt',encoding='utf-8',compresslevel=6) as handle:
                json.dump(value_scores,handle,allow_nan=False,separators=(',',':'))
        else:
            (out/'EVALUATOR_ONLY_VALUE_SCORES.json').write_text(json.dumps(value_scores,indent=2,allow_nan=False)+'\n')
        for arm in branches:
            rows=value_scores['arms'][arm]['rows']
            branches[arm].update(first_maintained_12_date=value_scores['arms'][arm]['first_maintained_12_date'],
                score_available_dates=sum(r['status']=='scored_numerical_bounds' for r in rows),
                loss_upper_below_005_dates=sum(r.get('upper_below_005',False) for r in rows),
                loss_lower_above_005_dates=sum(r.get('lower_above_005',False) for r in rows),
                pre24_31_loss_lower_above_005_dates=sum(r.get('lower_above_005',False) and r['date']<32 for r in rows),
                pre24_31_loss_upper_below_005_dates=sum(r.get('upper_below_005',False) and r['date']<32 for r in rows),
                post32_39_loss_lower_above_005_dates=sum(r.get('lower_above_005',False) and 32<=r['date']<40 for r in rows),
                post40_loss_lower_above_005_dates=sum(r.get('lower_above_005',False) and r['date']>=40 for r in rows),
                right_censored=value_scores['arms'][arm]['right_censored'],
                value_qualification='evaluator-only known-law floating bounds; not a learner causal qualification')
    return clean({'assignment':row,'source':source,'branches':branches,
                 'status':'complete','scope':'controlled-source action-observation benchmark; no ordinary-panel source claim',
                 'economic_recovery':'evaluator-only known-law numerical H10 bounds' if value_scores is not None else 'not_established; evaluator not requested'})


def evidence_for_panel(root,directory,session):
    """Automatically verify a controlled acquisition against imported CSV history.

    No manually prepared research JSON is required. Ordinary data lacking this
    actual reset transcript cannot use this route.
    """
    root=Path(root);directory=Path(directory).resolve()
    result=json.loads((directory/'LIVE_SOURCE.json').read_text())
    if result.get('publication_kind')!='fresh_controlled_history':
        raise ValueError('a fresh controlled acquisition transcript is required, not an old MODEL handoff')
    if result.get('public_stream_prefix_digest')!=canonical_panel_digest(session.history):
        raise ValueError('imported panel/proxy prefix does not match the actual diagnostic acquisition')
    if result.get('reward_digest')!=session.identity['reward']:
        raise ValueError('the calibrated acquisition requires the identical fixed RND2 R0 equation/input/parameter specification')
    archive=Path(result['archive']).resolve()
    if not archive.is_relative_to(directory):raise ValueError('controlled archive must belong to the supplied acquisition directory')
    archived=_public_archive(archive)
    actual=json.loads(archived['RESULT.json'])
    if actual.get('scores')!=result['stage1'].get('scores') or actual['status']!=result['stage1']['status']:
        raise ValueError('monitor receipt mismatch')
    calibration=_load('product_verified_original_calibration',root/'note/middle_loop_deployment_20261108/fixed_reward_review_20261109/behavior_replay/run_matched.py')
    thresholds=calibration.replayed_thresholds()
    monitor=_load('product_verified_original_monitor',root/'experiments/ecma_four_tracks_20260924/track1_production/paired_union_source_attempt001/monitor.py')
    alarm=None if actual['status']!='available' else monitor.union_alarm(actual['scores'],thresholds)
    curve=_load('product_verified_original_curve',root/'experiments/round_coefficient_precision128_20261110/source_curvature_continuation_20261112/saved_readback.py')
    raw=curve.archive_raw(archive);qualified=[];refusals=[]
    if alarm:
        for date in range(32,40):
            point,reason=curve.qualify(raw,date)
            if point:qualified.append(point)
            else:refusals.append({'date':date,'reason':reason})
    violations=[r['date'] for r in qualified if abs(Decimal(r['discrepancy']))>curve.TOL]
    outcome=('not_entered_no_alarm' if alarm is False else 'source_unavailable_monitor' if alarm is None else
        'model_involvement' if violations else 'unresolved' if qualified else 'source_unavailable_no_qualified_triples')
    if outcome!=result['source_outcome']:raise ValueError('live curve recomputation disagrees with stored receipt')
    verified={**result,'qualified_triples':qualified,'unavailable_dates':refusals,'violating_dates':violations,
              'stage1':{**result['stage1'],'alarm':alarm,'thresholds':thresholds},
              'revalidation':'imported CSV/proxy prefix, native monitor receipt, registered CAL and current public curve endpoints checked'}
    return LiveCurveEvidence(verified,{'directory':str(directory),'public_stream_prefix_digest':result['public_stream_prefix_digest']},session.identity)
