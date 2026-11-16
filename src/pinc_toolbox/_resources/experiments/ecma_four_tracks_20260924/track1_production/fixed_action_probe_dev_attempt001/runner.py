"""Prepared eight-history fixed-action DEV. Science requires separate Root approval."""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import sys,json,time,signal,copy,importlib.util,argparse,traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from experiment import ACTIONS,LIMIT,OUTPUT_BYTES,TIME_SECONDS,streams,configuration

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def setup():
 # Reuse the reviewed scientific runtime and compact recorder; never modify them.
 backend=load('fixed_probe_scientific_backend',HERE.parent/'matched_main_candidate_cal_attempt001/runner.py')
 backend.LIMIT=LIMIT.copy();backend.COUNTS={k+'_'+v:0 for k in LIMIT for v in ('attempted','succeeded')}
 dispatch=load('root_prepared_probe_dispatcher',HERE.parents[1]/'root_prepared_probe/dispatcher.py')
 measurements=load('fixed_probe_measurements',HERE/'measurements.py')
 return backend,dispatch,measurements

def blank_endpoint(spec,t,a,reason='main_not_available'):
 return dict(seed=spec['seed'],cell=spec['cell'],time=t,action=a,status=reason,physical_action_supported=None,finite_executed_Q=None,executed_Q=None,issued=None,next_physical=None,next_auxiliary=None,reward=None,release_time=None,full_x_error=None,full_w_error=None,failure=None,eta_status='not_supplied',mode='one_step_reset_from_prepared_MAIN')

class ProbeExecutionFailure(RuntimeError):
 pass

def execute_endpoint(dispatch,*,row,persist,**kwargs):
 """Persist dispatcher receipts immediately; never relabel host faults as MAIN U."""
 submitted=[]
 def on_record(raw):
  row.update(status=raw['status'],failure=raw['failure'],issued=raw['issued'],query_physical=raw['state'].physical.tolist(),query_auxiliary=raw['state'].auxiliary.tolist())
  if raw['outcome'] is not None:
   outcome=raw['outcome'];row.update(next_physical=outcome.next_physical.tolist(),reward=outcome.reward,release_time=outcome.release_time)
  if raw['next_state'] is not None:row['next_auxiliary']=raw['next_state'].auxiliary.tolist()
  if raw['issued'] is not None and raw['outcome'] is not None:
   pred=raw['issued']['main']['full_physical_prediction'];row.update(full_x_error=float(raw['outcome'].next_physical[0]-pred[0]),full_w_error=float(raw['outcome'].next_physical[1]-pred[1]))
  submitted.append(True);persist(row)
 try:return dispatch.run_prepared_probes(**kwargs,on_record=on_record)[0]
 except Exception as exc:
  if not submitted:
   row.update(status='software_failure',failure=dict(type=type(exc).__name__,reason=str(exc)));persist(row)
  raise ProbeExecutionFailure('probe software/clock/resource failure: '+str(exc)) from exc

def episode(spec,out,dgp,b,dispatch,measurements,probe_counts):
 import numpy as np
 from pinc_toolbox.counterfactual import PublicState
 from pinc_toolbox.g import SupportError
 folder=out/f"episode_{spec['seed']}";folder.mkdir();before=b.COUNTS.copy()
 endpoints=[blank_endpoint(spec,t,a) for t in range(40,48) for a in ACTIONS]
 b.ACTIVE_PROBE_ENDPOINTS=endpoints
 mainrows=[];monitor=[];failure=None;status='pending';phase='prefix';t=None
 b.ARCHIVE=b.EpisodeWriter(folder/'records.xz',b.BUDGET)
 b.front.CURRENT.update(seed=spec['seed'],folder=folder,date=None,stage=phase,raw_validated=False)
 def save(name,value):b.save(folder/name,value)
 try:
  b.storage();history,env=b.prefix._prefix(dgp,spec['seed'],spec['source'],1.)
  b.ARCHIVE.raw('prefix.csv',history.to_csv(index=False,float_format='%.17g').encode())
  save('ORACLE_AUDIT_ONLY.json',b.oracle_receipt(env,32,48))
  l,x,e,init=b.initialize_checked(history);save('PREFIX_INITIALIZATION.json',init);panel=b.front.Panel(l,x,e)
  for t in range(32,48):
   phase='monitor' if t<40 else 'MAIN_source'
   b.front.CURRENT.update(folder=folder,date=t,stage=phase,raw_validated=b.front.validate_history(l.history))
   panel.prepare(t);b.snap(folder/f'snapshot_{t}.pkl.gz',l)
   if l.model.as_of_!=t or l.pending is not None or panel.pending is not None:raise ValueError('MAIN not prepared before forks')
   save(f'states_{t}.json',dict(time=t,physical=panel.physical.tolist(),auxiliary=panel.auxiliary.tolist()))
   qs=PublicState(panel.physical[0].copy(),panel.auxiliary[0].copy())
   planned_main=float(l.choose(t,qs))
   if not l.supported(t,qs,planned_main):raise SupportError('MAIN native action unsupported')
   if t>=40:
    # All clones are rooted before MAIN issue/step; original MAIN objects remain live.
    pl=b.PanelBranchLearner(l);pe=b.PanelEnvironmentAdapter(b.OriginalEnvironmentHost(env))
    ps=b.pack_state(panel.physical,panel.auxiliary);public=pe.public_input(t)
    if public.available_at!=t or float(public.values[0])!=float(panel.physical[0,1]):raise ValueError('probe public input mismatch')
    history_before=l.history.copy(deep=True);c=l.model.model_.core._C.copy();psi=l.model.model_.core._Psi.copy()
    physical_before=panel.physical.copy();aux_before=panel.auxiliary.copy();env_public_before=env.public(t)
    for ai,a in enumerate(ACTIONS):
     row=endpoints[(t-40)*2+ai];row['status']='pending'
     probe_counts['endpoints_attempted']+=1
     if probe_counts['endpoints_attempted']>128:raise RuntimeError('128 endpoint budget exceeded')
     j=int(np.flatnonzero(l.model.action_grid_==a)[0]);value=l.model.q_values(qs.physical[None],qs.auxiliary[None])[0,j]
     row.update(physical_action_supported=bool(l.model.model_.support_.mask(qs.physical[None],l.model.action_grid_)[0,j]),finite_executed_Q=bool(np.isfinite(value)),executed_Q=float(value) if np.isfinite(value) else None)
     save(f'probe_{t}_{round(a*100):02d}_PLAN.json',dict(time=t,action=a,prepared_as_of=l.model.as_of_,qualification=row.copy()))
     science_before=b.COUNTS.copy()
     raw=execute_endpoint(dispatch,row=row,persist=lambda value:save(f'probe_{t}_{round(a*100):02d}.json',value),learner=pl,environment=pe,state=ps,public=public,time=t,actions=(a,))
     row.update(status=raw['status'],failure=raw['failure'],issued=raw['issued'])
     if raw['status']=='complete':
      outcome=raw['outcome'];next_state=raw['next_state'];issued=raw['issued']['main']
      if not row['physical_action_supported'] or not row['finite_executed_Q']:raise ValueError('probe completed outside native support')
      if raw['issued']['time']!=t or issued['fit_as_of']!=t or issued['action']!=a:raise ValueError('probe issue vintage/action mismatch')
      row.update(next_physical=outcome.next_physical.tolist(),next_auxiliary=next_state.auxiliary.tolist(),reward=outcome.reward,release_time=outcome.release_time,full_x_error=float(outcome.next_physical[0]-issued['full_physical_prediction'][0]),full_w_error=float(outcome.next_physical[1]-issued['full_physical_prediction'][1]))
      probe_counts['endpoints_completed']+=1
     elif raw['status']!='support_refusal':raise ValueError('unknown probe status')
     for key in ('prefix_G','native_G','Q','reward','constructors'):
      for suffix in ('attempted','succeeded'):
       if b.COUNTS[key+'_'+suffix]!=science_before[key+'_'+suffix]:raise ValueError('probe refitted/solved/constructed')
     if env._next_t!=t or l.pending is not None or panel.pending is not None or not l.history.equals(history_before):raise ValueError('probe contaminated MAIN')
     if not np.shares_memory(env.w,env.proxy_all) or not np.shares_memory(pe.host._env.w,pe.host._env.proxy_all):raise ValueError('repaired proxy view lost')
     for old,new in ((physical_before,panel.physical),(aux_before,panel.auxiliary),(c,l.model.model_.core._C),(psi,l.model.model_.core._Psi),*zip(env_public_before,env.public(t))):np.testing.assert_array_equal(old,new)
     save(f'probe_{t}_{round(a*100):02d}_VALIDATION.json',dict(no_extra_fit_or_solve=True,parent_unchanged=True))
    save(f'PAIR_{t}.json',measurements.pair(*endpoints[(t-40)*2:(t-40)*2+2]))
   # Actual MAIN follows the unchanged original panel path and native action.
   issued,actions=panel.issue(t)
   if actions[0]!=planned_main:raise ValueError('probe changed MAIN action')
   save(f'ISSUED_{t}.json',list(issued.values()))
   env.step(t,actions);nx,nw=env.public(t+1);actual=np.column_stack([nx,nw,actions]);rewards=env.released_reward(t+1)
   save(f'RELEASED_{t+1}.json',dict(release_time=t+1,physical=actual.tolist(),reward=rewards.tolist()))
   ne=panel.observe(t,actual,rewards,t+1);save(f'AUXILIARY_{t+1}.json',dict(release_time=t+1,auxiliary=ne.tolist()));b.snap(folder/f'released_learner_{t+1}.pkl.gz',l)
   event=dict(time=t,release_time=t+1,issued=issued[0]['issued'],physical=issued[0]['physical'],auxiliary=issued[0]['auxiliary'],action=float(actions[0]),next_physical=actual[0].tolist(),next_auxiliary=ne[0].tolist(),reward=float(rewards[0]))
   if t<40:monitor.append(event)
   else:mainrows.append(dict(time=t,status='complete',full_x_error=float(actual[0,0]-event['issued']['full_physical_prediction'][0]),full_w_error=float(actual[0,1]-event['issued']['full_physical_prediction'][1])))
   if t==39:save('MONITOR_SCORE.json',b.front.score_main(monitor,8))
  status='complete'
 except Exception as exc:
  classified=dict(status='fatal',reason='probe_software_failure') if isinstance(exc,ProbeExecutionFailure) else b.front.classification(exc)
  failure=dict(phase=phase,time=t,type=type(exc).__name__,reason=str(exc),classification=classified,traceback=traceback.format_exc())
  status='fatal' if classified['status']=='fatal' else 'MAIN_unavailable'
  for row in endpoints:
   if row['status'] in ('main_not_available','pending'):row.update(status='not_run_after_fatal' if status=='fatal' else 'main_unavailable',failure=failure)
  save('FAILURE.json',failure)
 result=dict(spec=spec,status=status,failure=failure,monitor_completed=len(monitor),main_source_completed=len(mainrows),main_score=measurements.score(mainrows),probe_scores={str(a):measurements.score([r for r in endpoints if r['action']==a]) for a in ACTIONS},endpoints=endpoints,pairs=[measurements.pair(*endpoints[i:i+2]) for i in range(0,16,2)],counts={k:b.COUNTS[k]-before[k] for k in b.COUNTS})
 try:save('RESULT.json',result);b.ARCHIVE.finish()
 except BaseException:b.ARCHIVE.abort();raise
 finally:b.ARCHIVE=None
 return result

def run(out):
 out.mkdir(exist_ok=False);b,dispatch,measurements=setup();b.BUDGET=b.ResourceBudget(HERE,allowance=OUTPUT_BYTES)
 b.storage();dgp=b.load_dgp();b.install(dgp)
 start=time.monotonic();probe_counts=dict(endpoints_attempted=0,endpoints_completed=0);results=[];failure=None
 def deadline(*args):raise TimeoutError('1800 second scientific execution limit')
 signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,TIME_SECONDS)
 try:
  config=configuration();config.update(pid=os.getpid(),science_execution_authorization='Root-authorized invocation required externally',dispatcher=str(HERE.parents[1]/'root_prepared_probe/dispatcher.py'))
  b.save(out/'CONFIG.json',config)
  for spec in streams():
   b.ACTIVE_PROBE_ENDPOINTS=[blank_endpoint(spec,t,a,'not_run') for t in range(40,48) for a in ACTIONS]
   try:
    r=episode(spec,out,dgp,b,dispatch,measurements,probe_counts);results.append(r)
    b.append_execution(out/'PROGRESS.jsonl',json.dumps(r,allow_nan=False)+'\n')
   except BaseException as exc:
    failure=dict(seed=spec['seed'],type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
    if b.ARCHIVE is not None:
     try:b.ARCHIVE.abort()
     except BaseException:pass
     b.ARCHIVE=None
    if results and results[-1]['spec']['seed']==spec['seed']:results[-1].update(status='fatal',failure=failure)
    else:results.append(dict(spec=spec,status='fatal',failure=failure,endpoints=copy.deepcopy(getattr(b,'ACTIVE_PROBE_ENDPOINTS',[blank_endpoint(spec,t,a,'not_available_after_fatal') for t in range(40,48) for a in ACTIONS]))))
    break
   if r['status']=='fatal':failure=r['failure'];break
 except BaseException as exc:
  failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 b.BUDGET.reporting=True;byseed={r['spec']['seed']:r for r in results}
 allrows=[]
 for spec in streams():
  r=byseed.get(spec['seed'])
  if r is None:r=dict(spec=spec,status='not_run',failure=None,endpoints=[blank_endpoint(spec,t,a,'not_run') for t in range(40,48) for a in ACTIONS])
  allrows.append(r)
 observable={cell:any(r.get('probe_scores',{}).get(str(ACTIONS[0]),{}).get('complete') and r.get('probe_scores',{}).get(str(ACTIONS[1]),{}).get('complete') for r in allrows if r['spec']['cell']==cell) for cell in ('N','CL','CM','MIX')}
 b.save(out/'ALL_ASSIGNED.json',allrows)
 b.save(out/'RESULT.json',dict(status='complete' if len(results)==8 and failure is None else 'stopped_fatal',failure=failure,counts=b.COUNTS,probe_counts=probe_counts,assigned_histories=8,assigned_probe_endpoints=128,assigned_pair_dates=64,elapsed_seconds=time.monotonic()-start,double_complete_observable=observable,CM_MIX_window_comparison_observable=observable['CM'] and observable['MIX'],initial_latent_updates_by_original_program=8000*b.COUNTS['constructors_succeeded'],scope='DEV descriptive reset probes, not source test',new_p_values=False,eta_status='not_supplied'))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
 if args.prepare:
  with (HERE/'PREDECLARED_CONFIG.json').open('x') as f:json.dump(configuration(),f,indent=2);f.write('\n')
 elif args.run:run(HERE/'execution001')
 else:parser.error('use --prepare; --run only after Root science authorization')
