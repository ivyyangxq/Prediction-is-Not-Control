"""Prospective matched MAIN campaign. Prepared only; Root has not authorized execution."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import sys,json,csv,gzip,copy,time,argparse,importlib.util,traceback,shutil,io,inspect
HERE=Path(__file__).resolve().parent;P=HERE.parent/'persistent_panel_frontier_attempt001';W=HERE.parent/'panel_four_branch_wrapper_attempt001'
sys.path[:0]=[str(HERE),str(HERE.parent/'direct_source_contrast_dev_attempt001'),str(W),str(P/'runtime_source')]
import numpy as np,cloudpickle
from pinc_toolbox import native_gq,rnd2_counterfactual
from pinc_toolbox.counterfactual import run_four_paths,BRANCHES
from panel_wrapper import PanelBranchLearner,PanelEnvironmentAdapter,pack_state
from live_host import OriginalEnvironmentHost
from design import streams,detector_flags,LIMIT,SOURCE_BETA,dispatch_source
from compact import EpisodeWriter,ResourceBudget
from inference import branch_score,candidate_decision
ARCHIVE=None
BUDGET=None

def load_module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
front=load_module('source_dev_frontier',P/'runner.py')
prefix=load_module('source_dev_prefix',P/'dependencies/prefix_driver.py')
repair=load_module('source_dev_proxy_fork_repair',HERE.parent/'proxy_fork_repair_attempt001/proxy_fork_repair.py')
pair=load_module('source_dev_pair',HERE.parents[1]/'root_pair_source_rank_adapter/pair_rank.py')
COUNTS={k+'_'+s:0 for k in LIMIT for s in ('attempted','succeeded')};CURRENT={};BRANCH_INDEX=0

def storage():
 if BUDGET is not None:BUDGET.check()
 elif shutil.disk_usage(HERE).free<512*1024**2:raise OSError("disk free below512MiB")
def save(path,obj):
 data=(json.dumps(obj,indent=2,allow_nan=False)+'\n').encode()
 if ARCHIVE is not None and path.is_relative_to(ARCHIVE.path.parent):ARCHIVE.raw(str(path.relative_to(ARCHIVE.path.parent)),data)
 else:
  if BUDGET is not None:BUDGET.check(len(data))
  else:storage()
  with path.open('xb') as f:f.write(data)
  storage()
def append_execution(path,text):
 data=text.encode()
 if BUDGET is not None:BUDGET.check(len(data))
 else:storage()
 with path.open('ab') as f:f.write(data);f.flush()
 storage()
def snap(path,obj):
 storage()
 ARCHIVE.snapshot(str(path.relative_to(ARCHIVE.path.parent)),obj)
def counted(fun,key):
 def wrapped(*a,**kw):
  if COUNTS[key+'_attempted']>=LIMIT[key]:raise RuntimeError('authorized '+key+' budget exhausted')
  COUNTS[key+'_attempted']+=1;value=fun(*a,**kw);COUNTS[key+'_succeeded']+=1;return value
 return wrapped

def install(dgp):
 rnd2_counterfactual._fit_model=counted(rnd2_counterfactual._fit_model,'prefix_G')
 native_gq._fit=counted(native_gq._fit,'native_G');native_gq._finite_q=counted(native_gq._finite_q,'Q');native_gq.reward_fit=counted(native_gq.reward_fit,'reward')
 dgp.Environment.__init__=counted(dgp.Environment.__init__,'constructors')
 original=dgp.Environment.step
 def step(self,*a,**kw):
  n=int(self.paths)
  if COUNTS['transitions_attempted']+n>LIMIT['transitions'] or COUNTS['host_calls_attempted']>=LIMIT['host_calls']:raise RuntimeError('authorized transitions exhausted')
  COUNTS['transitions_attempted']+=n;COUNTS['host_calls_attempted']+=1
  value=original(self,*a,**kw);COUNTS['transitions_succeeded']+=n;COUNTS['host_calls_succeeded']+=1;return value
 dgp.Environment.step=step
 install_audit_recording()
 front.install_fit_audit();front.install_availability_capture()

def install_audit_recording():
 """Change only audit sinks; numeric fitting/classification code remains exact."""
 def raw(path,data,append=False):
  name=str(path.relative_to(ARCHIVE.path.parent))
  (ARCHIVE.append_raw if append else ARCHIVE.raw)(name,data)
 def solver_arrays(path,**arrays):
  stream=io.BytesIO();np.savez_compressed(stream,**arrays);raw(path,stream.getvalue())
 front.save=save;front.record_raw=raw;front.record_arrays=solver_arrays
 source=inspect.getsource(front.install_fit_audit)
 old="frame.to_csv(CURRENT['folder']/('FAILED_FIT_%d_%s.csv'%(info['as_of'],info['stage'])),index=False)"
 new="record_raw(CURRENT['folder']/('FAILED_FIT_%d_%s.csv'%(info['as_of'],info['stage'])),frame.to_csv(index=False).encode())"
 assert source.count(old)==1;source=source.replace(old,new)
 old="with (CURRENT['folder']/'FIT_AUDITS.jsonl').open('a') as f:f.write(json.dumps(info,allow_nan=False)+'\\n');f.flush()"
 new="record_raw(CURRENT['folder']/'FIT_AUDITS.jsonl',(json.dumps(info,allow_nan=False)+'\\n').encode(),append=True)"
 assert source.count(old)==1;source=source.replace(old,new)
 exec(compile(source,str(P/'runner.py')+'::audit_sink','exec'),front.__dict__)
 source=inspect.getsource(front.install_availability_capture)
 old='np.savez_compressed(folder/'
 assert source.count(old)==1;source=source.replace(old,'record_arrays(folder/')
 exec(compile(source,str(P/'runner.py')+'::availability_sink','exec'),front.__dict__)

class RecordedPanelLearner(PanelBranchLearner):
 def __deepcopy__(self,memo):
  global BRANCH_INDEX
  out=object.__new__(type(self));memo[id(self)]=out
  for k,v in self.__dict__.items():setattr(out,k,copy.deepcopy(v,memo))
  out.branch=BRANCHES[BRANCH_INDEX];BRANCH_INDEX+=1;return out
 def prepare(self,t,state):
  folder=CURRENT['source_folder']/self.branch;folder.mkdir(exist_ok=True)
  front.CURRENT.update(folder=folder,date=t,stage='source_'+self.branch,raw_validated=front.validate_history(self.learner.history))
  try:out=super().prepare(t,state)
  except Exception as exc:
   info=dict(time=t,classification=front.classification(exc),error_type=type(exc).__name__,reason=str(exc));save(folder/('PREPARE_%d_FAILURE.json'%t),info);CURRENT['source_errors'][self.branch]=info;raise
  snap(folder/('snapshot_%d.pkl.gz'%t),self.learner);return out
 def observe(self,t,state,action,outcome):
  out=super().observe(t,state,action,outcome)
  snap(CURRENT['source_folder']/self.branch/('released_learner_%d.pkl.gz'%(t+1)),self.learner);return out

def initialize_checked(history):
 front.CURRENT.update(raw_validated=False,date=32,stage='initialization')
 valid=front.validate_history(history)
 front.CURRENT.update(raw_validated=valid)
 return front.initialize_panel(history)

def oracle_receipt(env,start,stop):
 return dict(oracle_only=True,start=start,end=stop,initial_Z=env.latent[:,start].tolist(),state_innovations=env.state_noise[:,start:stop].tolist(),latent_innovations=env.latent_noise[:,start:stop].tolist(),measurement_innovations=env.measurement_noise[:,start:stop+1].tolist(),g_profile=env.g_profile[start:stop+1].tolist())

def run_history(spec,out,dgp):
 global BRANCH_INDEX,ARCHIVE
 folder=out/('episode_'+str(spec['seed']));folder.mkdir();before=COUNTS.copy();phase='prefix';events=[];detector=None;source_report=None;failure=None;status='pending';t=None
 ARCHIVE=EpisodeWriter(folder/'records.xz',BUDGET)
 front.CURRENT.update(seed=spec['seed'],folder=folder,date=None,stage='prefix',raw_validated=False)
 try:
  storage()
  history,env=prefix._prefix(dgp,spec['seed'],spec['source'],spec['strength']);ARCHIVE.raw('prefix.csv',history.to_csv(index=False,float_format='%.17g').encode())
  save(folder/'MONITOR_ORACLE_AUDIT_ONLY.json',oracle_receipt(env,32,40))
  l,x,e,init=initialize_checked(history);save(folder/'PREFIX_INITIALIZATION.json',init);panel=front.Panel(l,x,e);phase='monitor'
  for t in range(32,40):
   front.CURRENT.update(folder=folder,date=t,stage='monitor',raw_validated=front.validate_history(l.history))
   panel.prepare(t);snap(folder/('snapshot_%d.pkl.gz'%t),l)
   save(folder/('states_%d.json'%t),dict(time=t,physical=panel.physical.tolist(),auxiliary=panel.auxiliary.tolist()))
   issued,actions=panel.issue(t);save(folder/('ISSUED_%d.json'%t),list(issued.values()))
   env.step(t,actions);nx,nw=env.public(t+1);actual=np.column_stack([nx,nw,actions]);rewards=env.released_reward(t+1)
   save(folder/('RELEASED_%d.json'%(t+1)),dict(release_time=t+1,physical=actual.tolist(),reward=rewards.tolist()))
   ne=panel.observe(t,actual,rewards,t+1);save(folder/('AUXILIARY_%d.json'%(t+1)),dict(release_time=t+1,auxiliary=ne.tolist()))
   events.append(dict(time=t,release_time=t+1,issued=issued[0]['issued'],physical=issued[0]['physical'],auxiliary=issued[0]['auxiliary'],action=float(actions[0]),next_physical=actual[0].tolist(),next_auxiliary=ne[0].tolist(),reward=float(rewards[0])))
  detector=detector_flags(front.score_main(events,8));save(folder/'DETECTOR_DEV.json',detector)
  if not dispatch_source(released_monitor_steps=len(events),monitor_available=True):status='monitor_complete_not_scheduled'
  else:
   phase='source';sf=folder/'source';sf.mkdir();CURRENT.update(source_folder=sf,source_errors={});BRANCH_INDEX=0
   save(sf/'FORK_INPUT.json',dict(release_time=40,physical=panel.physical.tolist(),auxiliary=panel.auxiliary.tolist(),last_model_as_of=l.model.as_of_,prepare40_required=True))
   snap(sf/'fork_released_learner_40.pkl.gz',l)
   # Evaluator-only receipt. Learner and wrapper never receive this object.
   save(sf/'ORACLE_AUDIT_ONLY.json',oracle_receipt(env,40,48))
   source_report=run_four_paths(learner=RecordedPanelLearner(l),environment=PanelEnvironmentAdapter(OriginalEnvironmentHost(env)),initial_state=pack_state(panel.physical,panel.auxiliary),start_time=40,steps=8,beta=SOURCE_BETA)
   save(sf/'FOUR_PATHS.json',source_report);contrast=pair.pair_contrast(source_report)
   save(sf/'PAIR_DIAGNOSTIC.json',dict(pair_available=contrast is not None,contrast=contrast,full_four_comparison=source_report['comparison'],source_inference='unavailable',scope='uncalibrated DEV return contrast',beta=SOURCE_BETA,selection_event='native_reach40_and_main_fixed_action_pair_complete',prepare_failures=CURRENT['source_errors']))
   status='source_pair_available' if contrast is not None else 'source_pair_unavailable'
   # Keep raw engine statuses. Only verified finite-input identification is an
   # ordinary unavailable outcome; unknown/software failures halt the pilot.
   for branch,b in source_report['branches'].items():
    if b['status']=='software_failure' and CURRENT['source_errors'].get(branch,{}).get('classification',{}).get('status')!='identification_unavailable':
     status='fatal';failure=dict(phase='source',branch=branch,raw_failures=b['failures']);break
 except Exception as exc:
  cause=front.classification(exc);status='fatal' if cause['status']=='fatal' else 'monitor_or_initialization_unavailable'
  failure=dict(phase=phase,time=t,error_type=type(exc).__name__,reason=str(exc),classification=cause,traceback=traceback.format_exc())
 result=dict(spec=spec,status=status,failure=failure,monitor_completed=len(events),detector=detector,source_scheduled=bool(detector and dispatch_source(released_monitor_steps=len(events),monitor_available=True)),source_entered=source_report is not None,pair_available=None if source_report is None else pair.pair_contrast(source_report) is not None,all_four_available=None if source_report is None else source_report['status']=='complete',source_branches=None if source_report is None else {name:dict(status=b['status'],completed_steps=b['completed_steps'],failures=b['failures']) for name,b in source_report['branches'].items()},counts={k:COUNTS[k]-before[k] for k in COUNTS})
 result['branch_energies']={name:branch_score(None if source_report is None else source_report['branches'][name]) for name in BRANCHES}
 try:
  save(folder/'RESULT.json',result)
  for raw in sorted(folder.rglob('*')):
   if raw.is_file() and raw!=ARCHIVE.path:ARCHIVE.raw(str(raw.relative_to(folder)),raw.read_bytes())
  ARCHIVE.finish()
 except BaseException:
  ARCHIVE.abort();raise
 finally:ARCHIVE=None
 return result

def load_dgp():
 return repair.load_patched_environment(prefix.ENV_FILE,'source_dev_repaired_model_source_v004')

def run(out):
 global BUDGET,ARCHIVE
 out.mkdir(exist_ok=False);BUDGET=ResourceBudget(out);storage()
 dgp=load_dgp();install(dgp);start=time.time()
 save(out/'CONFIG.json',dict(pid=os.getpid(),source_return_beta=SOURCE_BETA,plan=streams(),limits=LIMIT,scope='MAIN residual finite-catalogue benchmark',environment_file=str(prefix.ENV_FILE),output_allowance_bytes=BUDGET.allowance,reserve_bytes=BUDGET.reserve))
 results=[];failure=None
 for spec in streams():
  before=COUNTS.copy()
  try:
   r=run_history(spec,out,dgp);results.append(r)
   append_execution(out/'PROGRESS.jsonl',json.dumps(r,allow_nan=False)+'\n')
   append_execution(out/'RUN.log','history '+str(spec['seed'])+' '+r['status']+'\n')
  except BaseException as exc:
   failure=dict(seed=spec['seed'],error_type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
   if ARCHIVE is not None:
    try:ARCHIVE.abort()
    except BaseException as close_error:failure['archive_close_error']=str(close_error)
    ARCHIVE=None
   if results and results[-1]['spec']['seed']==spec['seed']:
    results[-1].update(status='fatal',failure=failure)
   else:results.append(dict(spec=spec,status='fatal',failure=failure,branch_energies={name:branch_score(None) for name in BRANCHES},counts={k:COUNTS[k]-before[k] for k in COUNTS}))
   break
  if r['status']=='fatal':failure=r['failure'];break
 BUDGET.reporting=True
 complete={r['spec']['seed']:r for r in results}
 slots=[]
 for spec in streams():
  r=complete.get(spec['seed'])
  slots.append(dict(spec=spec,status='not_run' if r is None else r['status'],main=branch_score(None) if r is None else r['branch_energies']['main'],history=r))
 decisions=[];evaluations=[]
 campaign_valid=len(results)==348 and failure is None
 for row in slots[316:]:
  retained=[]
  for cell in ('N','CL','CM','MIX'):
   bank=[r['main'] for r in slots[:316] if r['spec']['cell']==cell]
   decision=candidate_decision(row['main'],bank) if campaign_valid else dict(excluded=False,unresolved=True,ranks=None,reason='campaign_invalid_no_inference')
   if not decision['excluded']:retained.append(cell)
   decisions.append(dict(seed=row['spec']['seed'],true_source=row['spec']['cell'],candidate=cell,campaign_valid=campaign_valid,**decision))
  evaluations.append(dict(spec=row['spec'],execution_status=row['status'],retained_set=retained,set_size=len(retained),true_source_retained=row['spec']['cell'] in retained,unresolved=not campaign_valid or not row['main']['full_window_complete'],campaign_valid=campaign_valid))
 table=[]
 for cell in ('N','CL','CM','MIX'):
  ev=[r for r in evaluations if r['spec']['cell']==cell]
  cellslots=[r for r in slots[316:] if r['spec']['cell']==cell]
  table.append(dict(true_source=cell,assigned=8,executed=sum(r['status']!='not_run' for r in cellslots),reached40=sum(bool((r['history'] or {}).get('source_entered')) for r in cellslots),main_complete=sum(r['main']['full_window_complete'] for r in cellslots),unresolved=sum(r['unresolved'] for r in ev),true_source_retained=sum(r['true_source_retained'] for r in ev),empty_sets=sum(r['set_size']==0 for r in ev),all_four_sets=sum(r['set_size']==4 for r in ev),exclusion_counts={candidate:sum(d['excluded'] for d in decisions if d['true_source']==cell and d['candidate']==candidate) for candidate in ('N','CL','CM','MIX')},frequency_denominator=8,campaign_valid=campaign_valid))
 save(out/'CAL_SLOTS.json',slots[:316]);save(out/'EVAL_SLOTS.json',slots[316:]);save(out/'CANDIDATE_DECISIONS.json',decisions);save(out/'EVALUATIONS.json',evaluations)
 save(out/'EXCLUSION_TABLE.json',table)
 save(out/'RESULT.json',dict(status='complete' if len(results)==348 and failure is None else 'stopped_fatal',failure=failure,histories=results,counts=COUNTS,elapsed_seconds=time.time()-start,scope='MAIN residual finite-catalogue benchmark; four branches descriptive only'))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
 if not args.run:parser.error('design-only unless subsequent Root execution authorization and --run')
 run(HERE/'execution001')
