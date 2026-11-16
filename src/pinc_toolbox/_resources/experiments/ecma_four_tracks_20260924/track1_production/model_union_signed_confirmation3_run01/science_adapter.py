"""Root-only prospective first-H conditional source campaign; import is inert."""
from pathlib import Path
import copy, io, json, math, shutil, signal, sys, time, traceback
import numpy as np
REPAIR=Path(__file__).resolve().parent.parent/'conditional_h_failure_repair_attempt001'
ORIGINAL=Path(__file__).resolve().parent.parent/'conditional_h_first_history_attempt001'
sys.path.insert(0,str(ORIGINAL));sys.path.insert(0,str(REPAIR))
from resource_budget import ResourceBudget
from future_episode import load
import monitor
import protocol

HERE=Path(__file__).resolve().parent
BellmanLearner=None
TRACK=HERE.parent
ROOT=TRACK.parent
CELLS=protocol.CELLS
SOURCES={'N':'STABLE','CL':'LATENT_F','CM':'MODEL_A','MIX':'MODEL_LATENT'}
LIMIT=dict(prefix_G=3200,native_G=10256,Q=5128,reward=5128,constructors=128,
           host_calls=17432,transitions=557824,conditional_initializations=512)
OUTPUT_BYTES=1536*1024**2
TEMP_BYTES=256*1024**2
RESERVE_BYTES=512*1024**2
TIME_SECONDS=5400
ALLOCATION=HERE/'ALLOCATION.json'
DETECTOR=TRACK/'direct_bellman_h8_cal_attempt001/execution001/RESULT.json'

def error(exc):
    result=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
    for key in ('archive_abort_error','original_error','recording_error'):
        if hasattr(exc,key):result[key]=getattr(exc,key)
    return result

def allocation(path):
    config=json.loads(Path(path).read_text());plan=config['monitor_proposals']
    if len(plan)!=128 or len({p['seed'] for p in plan})!=128:
        raise ValueError('need 128 unique preassigned monitor seeds')
    if any(p['ordinal']!=i or p['candidate']!=CELLS[i%4] or
           type(p['seed']) is not int or not 0<=p['seed']<2**32 for i,p in enumerate(plan)):
        raise ValueError('monitor order/seed allocation invalid')
    if config['history_index']!=0 or config['replicate_ids']!=list(range(128)):
        raise ValueError('unexpected conditional stream allocation')
    entropy=config['conditional_future_campaign_entropy']
    if type(entropy) is not int or entropy<0:raise ValueError('invalid Root entropy')
    return config,plan

def detector_threshold():
    bank=json.loads(DETECTOR.read_text());slots=bank['slots']
    if bank['status']!='complete' or not bank['A_all_statistical_slots_complete'] or not bank['R_execution_complete']:
        raise ValueError('fixed fresh H8 detector bank not complete')
    if len(slots)!=59 or any(s['status']!='accepted' or s['score_kind']!='finite' or
                            not math.isfinite(s['score']) for s in slots):
        raise ValueError('fixed detector does not contain 59 finite accepted slots')
    threshold=max(s['score'] for s in slots)
    if threshold!=bank['threshold']:raise ValueError('fixed bank maximum mismatch')
    return threshold

def setup_bellman():
    project=HERE.parents[3]
    import pinc_toolbox
    working_src=Path(pinc_toolbox.__file__).resolve().parent.parent
    sys.path.insert(0,str(working_src))
    import pinc_toolbox.native_gq as working_native_gq
    import pinc_toolbox.rnd2_counterfactual as working_rnd2_counterfactual
    fixed=load('direct_bellman_fixed_runner',TRACK/'fixed_action_probe_dev_attempt001/runner.py')
    b,dispatch,measurements=fixed.setup()
    if b.native_gq is not working_native_gq or b.rnd2_counterfactual is not working_rnd2_counterfactual:
        raise RuntimeError('fixed scientific backend did not retain working Bellman package')
    import inspect
    if Path(inspect.getsourcefile(b.native_gq)).resolve()!=(working_src/'pinc_toolbox/native_gq.py').resolve():
        raise RuntimeError(f'working native_gq path mismatch: {Path(inspect.getsourcefile(b.native_gq)).resolve()} != {(working_src/"pinc_toolbox/native_gq.py").resolve()}')
    if inspect.signature(b.native_gq.RND2GQ.q_values).parameters['evaluation'].default!='nearest':
        raise RuntimeError('legacy nearest default changed')
    class BellmanLearner(b.rnd2_counterfactual.RND2BranchLearner):
        def _q(self,state):
            if state.physical.shape!=(3,) or state.auxiliary.shape!=(3,):
                raise ValueError('RND2 Bellman route requires three physical and auxiliary coordinates')
            return self.model.q_values(state.physical[None],state.auxiliary[None],
                remaining_horizon=10,evaluation='bellman')[0]
    if BellmanLearner._q is b.rnd2_counterfactual.RND2BranchLearner._q:
        raise RuntimeError('explicit Bellman route missing')
    BellmanLearner.__module__=__name__
    BellmanLearner.__qualname__='BellmanLearner'
    globals()['BellmanLearner']=BellmanLearner
    b.BellmanLearner=BellmanLearner
    original_initialize=b.initialize_checked
    def initialize_bellman(history):
        learner,x,e,receipt=original_initialize(history)
        routed=BellmanLearner(learner.model,learner.history,path_id=learner.path_id,refit_every=learner.refit_every)
        return routed,x,e,receipt
    b.initialize_checked=initialize_bellman
    return b,dispatch,measurements,fixed

class Science:
    """Evaluator keeps original environment; CAL interfaces receive public f only."""
    def __init__(self,out,config,threshold):
        self.out=out;self.config=config;self.threshold=threshold
        b,self.dispatch,self.measurements,self.fixed=setup_bellman();self.b=b
        b.ResourceBudget=ResourceBudget
        b.LIMIT=LIMIT.copy();b.COUNTS={k+'_'+s:0 for k in LIMIT for s in ('attempted','succeeded')}
        b.front.COUNTS=b.COUNTS
        b.BUDGET=b.ResourceBudget(out,allowance=OUTPUT_BYTES,reserve=RESERVE_BYTES+TEMP_BYTES)
        self.capacity=dict(free=shutil.disk_usage(out).free,required=OUTPUT_BYTES+TEMP_BYTES+RESERVE_BYTES)
        if self.capacity['free']<self.capacity['required']:
            raise OSError('need 1.5GiB output +256MiB temporary +512MiB reserve')
        self.dgp=b.load_dgp();b.install(self.dgp)
        self.forward=load('first_h_forward',ROOT/'track2_errors/public_posterior_forward_initializer_attempt001/public_forward_host.py')
        self.sampler=load('first_h_sampler',ROOT/'track2_errors/conditional_sampler_interface_attempt001/sampler.py')
        self.filter=load('first_h_filter',ROOT/'track2_errors/public_history_gaussian_filter_attempt001/public_posterior.py')
        self.extractor=load('first_h_extractor',TRACK/'public_transcript_extract_attempt001/extractor.py')
        self.restore=load('first_h_restore',TRACK/'public_learner_restore_attempt001/restore.py')
        self.branch=load('direct_bellman_branch',HERE/'branch_acquisition.py')
        self.future=load('direct_bellman_future_episode',HERE/'future_episode.py')
        self.forward.PublicForwardHost.__init__=b.counted(self.forward.PublicForwardHost.__init__,'conditional_initializations')
        original=self.forward.PublicForwardHost.step
        def forward_step(host,t,actions):
            if b.COUNTS['host_calls_attempted']>=LIMIT['host_calls'] or b.COUNTS['transitions_attempted']+host.paths>LIMIT['transitions']:
                raise RuntimeError('campaign host/path transition budget exceeded')
            b.COUNTS['host_calls_attempted']+=1;b.COUNTS['transitions_attempted']+=host.paths
            result=original(host,t,actions)
            b.COUNTS['host_calls_succeeded']+=1;b.COUNTS['transitions_succeeded']+=host.paths
            return result
        self.forward.PublicForwardHost.step=forward_step
        self._observed_private=None;self.selected_archive=None
        self.filter_calls=0;self.normal_variates=0
        self.sampler_calls_attempted=0;self.sampler_calls_completed=0
        for name in ('monitors','CAL','cal_draws'): (out/name).mkdir()
        for candidate in CELLS:(out/'CAL'/candidate).mkdir()

    def write(self,path,data):
        if isinstance(data,str):data=data.encode()
        self.b.BUDGET.check(len(data))
        with Path(path).open('xb') as f:f.write(data)
        self.b.BUDGET.check()

    def write_json(self,path,data):self.write(path,json.dumps(data,indent=2,allow_nan=False)+'\n')
    def write_npz(self,path,arrays):
        stream=io.BytesIO();np.savez_compressed(stream,**arrays);self.write(path,stream.getvalue())

    def monitor(self,proposal):
        spec=dict(proposal,source=SOURCES[proposal['candidate']])
        result,held=monitor.episode(spec,self.out/'monitors',self.dgp,self.b,self.threshold)
        if held is not None:
            if self._observed_private is not None:raise RuntimeError('second selected observed history')
            self._observed_private=held
            self.selected_archive=Path(result['archive'])
            # Opaque evaluator checkpoint retains the original prospective tape.
            # It is never opened by the public extractor/filter/CAL code.
            import cloudpickle
            self.write(self.out/'EVALUATOR_ONLY_ORIGINAL_BOUNDARY.pkl',cloudpickle.dumps(held,protocol=5))
        return result

    def condition(self,selected_index):
        if self._observed_private is None:raise RuntimeError('selected original boundary absent')
        directory=self.out/'selected_public';directory.mkdir()
        records,history,readback=self.extractor.read_public_until40(self.selected_archive)
        arrays,table,learner_history,issued,boundary,checks=self.extractor.assemble(records,history)
        boundary.update(H=True,H_status='fixed_H8_scalar_or_serial_union_alarm')
        self.write_npz(directory/'FILTER_INPUT.npz',arrays)
        self.write(directory/'PUBLIC_TRANSITIONS.csv',table.to_csv(index=False,float_format='%.17g'))
        self.write(directory/'PUBLIC_LEARNER_HISTORY.csv',learner_history.to_csv(index=False,float_format='%.17g'))
        self.write_json(directory/'PUBLIC_MONITOR_ISSUED.json',issued)
        self.write_json(directory/'PUBLIC_BOUNDARY.json',boundary)
        self.write_json(directory/'PROVENANCE_NOT_FILTER_INPUT.json',dict(selected_index=selected_index,
            archive=str(self.selected_archive),selection='first preassigned strict H=true; no future/support selection',**readback,**checks))
        panel=self.restore.restore_boundary(self.selected_archive,directory/'PUBLIC_BOUNDARY.json',self.b.front.Panel)
        if type(panel.learner)._q is not self.b.BellmanLearner._q:
            raise RuntimeError('restored observed learner lost explicit Bellman route')
        posteriors={};saved={}
        for candidate in CELLS:
            posterior=self.filter.filter_public_history(candidate=candidate,**arrays)
            self.filter_calls+=1
            if posterior.final_date!=40:raise ValueError('posterior boundary is not date40')
            posteriors[candidate]=(posterior.means.copy(),posterior.covariances.copy())
            saved[candidate+'_means']=posterior.means;saved[candidate+'_covariances']=posterior.covariances
        self.write_npz(directory/'POSTERIORS.npz',saved)
        # No observed source label, DGP seed, original latent/tape or env is returned.
        return dict(panel=panel,posteriors=posteriors)

    def calibration(self,public,candidate,replicate_id,proposal_id):
        means,covariances=public['posteriors'][candidate]
        self.sampler_calls_attempted+=1
        sample=self.sampler.draw_replica_inputs(means=means,covariances=covariances,
            campaign_entropy=self.config['conditional_future_campaign_entropy'],history_index=0,
            candidate=candidate,replicate_id=replicate_id)
        self.sampler_calls_completed+=1;self.normal_variates+=1888
        draws=sample['host_inputs']
        self.write_npz(self.out/'cal_draws'/f'{candidate}_{replicate_id:03d}.npz',draws)
        self.write_json(self.out/'cal_draws'/f'{candidate}_{replicate_id:03d}.json',sample['stream_metadata'])
        panel=self.restore.clone_boundary(public['panel'])
        if type(panel.learner)._q is not self.b.BellmanLearner._q:raise RuntimeError('CAL clone lost Bellman route')
        env=self.forward.PublicForwardHost(candidate=candidate,x40=panel.physical[:,0],
            w40=panel.physical[:,1],previous_action39=panel.physical[:,2],**draws)
        # Legacy receipt key "seed" is a string proposal identity, never RNG input.
        spec=dict(seed=proposal_id,cell=candidate,H=True,replicate_id=replicate_id,role='CAL')
        return self.future.episode(spec,panel,env,self.out/'CAL'/candidate/f'proposal_{replicate_id:03d}',
            self.fixed,self.b,self.dispatch,self.measurements)

    def observed(self,public):
        if not (self.out/'CAL_COMPLETE.json').exists():raise RuntimeError('CAL not finished before observed future')
        panel,env=self._observed_private
        if type(panel.learner)._q is not self.b.BellmanLearner._q:raise RuntimeError('observed learner lost Bellman route')
        if env._next_t!=40 or panel.learner.model.as_of_!=39:
            raise ValueError('original observed boundary changed before release')
        np.testing.assert_array_equal(panel.physical,public['panel'].physical)
        np.testing.assert_array_equal(panel.auxiliary,public['panel'].auxiliary)
        # Neutral receipt category serves schema only; inference never reads truth.
        spec=dict(seed='OBSERVED:0',cell='N',H=True,replicate_id=0,role='observed',
                  cell_meaning='neutral schema placeholder, not observed truth')
        return self.future.episode(spec,panel,env,self.out/'observed',self.fixed,self.b,self.dispatch,self.measurements)

    def progress(self,kind,row):
        if kind=='CAL_complete':self.write_json(self.out/'CAL_COMPLETE.json',row)
        else:self.b.append_execution(self.out/'PROGRESS.jsonl',json.dumps(dict(kind=kind,**row),allow_nan=False)+'\n')

def run(out,allocation_path=ALLOCATION):
    out=Path(out);out.mkdir(exist_ok=False)
    start=time.monotonic();science=None;state=None;old_handler=None;installed=False
    try:
        config,plan=allocation(allocation_path);state=protocol.blank(plan)
        with (out/'PREDECLARED.json').open('x') as f:json.dump(dict(allocation=config,limits=LIMIT,
            output_bytes=OUTPUT_BYTES,temp_bytes=TEMP_BYTES,reserve_bytes=RESERVE_BYTES,time_seconds=TIME_SECONDS),f,indent=2)
        threshold=detector_threshold()
        if signal.getitimer(signal.ITIMER_REAL)!=(0.,0.):raise RuntimeError('use dedicated process without existing timer')
        def deadline(*args):raise TimeoutError('90 minute technical repeat deadline; no automatic retry')
        old_handler=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,TIME_SECONDS);installed=True
        science=Science(out,config,threshold)
        import sklearn,scipy
        science.write_json(out/'CONFIG.json',dict(threshold=threshold,detector_path=str(DETECTOR),
            detector_comparison='complete eight-date serial > fixed maximum',capacity=science.capacity,
            versions=dict(numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__),
            scope='same-allocation technical repeat from scratch, not independent confirmation; one selected public H history, bounded conditional CAL, then one original observed future'))
        protocol.execute(plan=plan,threshold=threshold,branch=science.branch,monitor=science.monitor,
            condition=science.condition,calibration=science.calibration,observed=science.observed,
            progress=science.progress,state=state)
    except BaseException as exc:
        if state is None:state=dict(status='invalid',failure=error(exc),inference=None)
        else:protocol.invalidate(state,error(exc))
        if science is not None and science.b.ARCHIVE is not None:
            try:science.b.ARCHIVE.abort()
            except BaseException as secondary:state['failure']['archive_abort_error']=str(secondary)
            finally:science.b.ARCHIVE=None
    finally:
        if installed:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
    state.pop('_live_banks',None)
    state['elapsed_seconds']=time.monotonic()-start
    if science is not None:
        state.update(counts=science.b.COUNTS,posterior_filter_calls=science.filter_calls,
            conditional_normal_variates_in_completed_calls=science.normal_variates,
            sampler_calls_attempted=science.sampler_calls_attempted,sampler_calls_completed=science.sampler_calls_completed,
            completed_original_latent_initialization_updates=8000*science.b.COUNTS['constructors_succeeded'])
        science.b.BUDGET.reporting=True
        science.write_json(out/'RESULT.json',state)
    else:
        with (out/'RESULT.json').open('x') as f:json.dump(state,f,indent=2)
    return state
