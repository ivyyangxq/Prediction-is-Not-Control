"""Authorized persistent32-path panel engineering pilot. Import performs no fits/draws."""
from pathlib import Path
import os,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):os.environ[key]='1'
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'DGP').is_dir() and (p/'experiments').is_dir())
sys.path.insert(0,str(HERE/'runtime_source'))
import argparse,copy,csv,gzip,importlib.util,json,math,shutil,time,traceback
import cloudpickle,numpy as np
from pinc_toolbox import native_gq,rnd2_counterfactual
from pinc_toolbox.counterfactual import PublicState
from pinc_toolbox.g import SupportError
from pinc_toolbox.residual_window_energy import window_slope_energy
from pinc_toolbox.strict_max_tolerance import strict_max_tolerance
COUNTS={k+'_'+s:0 for k in ('prefix_G','native_G','Q','reward','host_calls','transitions') for s in ('attempted','succeeded')}
LIMIT={'prefix_G':6825,'native_G':17472,'Q':8736,'reward':8736,'host_calls':17472,'transitions':559104}

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def save(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def write(path,obj):
 with Path(path).open('w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def storage():
 if shutil.disk_usage(HERE).free<512*1024**2:raise OSError('disk free below 512MiB; preserving partial run')
def counted(fun,key):
 def wrapped(*a,**kw):
  if COUNTS[key+'_attempted']>=LIMIT[key]:raise RuntimeError('authorized '+key+' budget exhausted')
  COUNTS[key+'_attempted']+=1;value=fun(*a,**kw);COUNTS[key+'_succeeded']+=1;return value
 return wrapped

def install_counters(dgp):
 rnd2_counterfactual._fit_model=counted(rnd2_counterfactual._fit_model,'prefix_G')
 native_gq._fit=counted(native_gq._fit,'native_G');native_gq._finite_q=counted(native_gq._finite_q,'Q');native_gq.reward_fit=counted(native_gq.reward_fit,'reward')
 step=dgp.Environment.step
 def counted_step(self,*a,**kw):
  n=int(self.paths)
  if COUNTS['host_calls_attempted']>=LIMIT['host_calls'] or COUNTS['transitions_attempted']+n>LIMIT['transitions']:raise RuntimeError('authorized DGP budget exhausted')
  COUNTS['host_calls_attempted']+=1;COUNTS['transitions_attempted']+=n
  value=step(self,*a,**kw);COUNTS['host_calls_succeeded']+=1;COUNTS['transitions_succeeded']+=n;return value
 dgp.Environment.step=counted_step

# Scientific plan and streams are read separately below.
CURRENT={};FIT_AUDITS=[]

def matrix_check(frame,cols):
 # Audit the declared physical dictionary without reading auxiliary values.
 active=[]
 x=frame[list(native_gq.MEMORY_STATE)].to_numpy(float);s,w,p=x[:,:3].T;a=frame.action.to_numpy(float)
 b=[np.ones(len(x)),s,w,p,p*p,s*p,w*p];d=[a-p,a*a-p*p,s*(a-p),w*(a-p)]
 for j in range(len(active)):b.extend([x[:,3+j],p*x[:,3+j]]);d.append((a-p)*x[:,3+j])
 B=np.column_stack(b);D=np.column_stack(d);bs=np.max(abs(B),axis=0);ds=np.max(abs(D),axis=0)
 def stat(v):
  sv=np.linalg.svd(v,compute_uv=False);tol=np.finfo(float).eps*max(v.shape)*sv[0]
  return dict(rank=int(sum(sv>tol)),columns=v.shape[1],smin=float(sv[-1]),tolerance=float(tol))
 result=dict(distinct_actions=np.unique(a).tolist(),distinct_action_count=int(len(np.unique(a))),rows=len(frame),period_min=int(frame.period.min()),period_max=int(frame.period.max()),prefix_rows=int((frame.period<32).sum()),online_rows=int((frame.period>=32).sum()),active=active,B=stat(B),D=stat(D),joint=stat(np.column_stack([B,D])),action_coverage=[dict(action=float(a),count=int(np.isclose(frame.action,a,rtol=0,atol=1e-14).sum())) for a in native_gq.GRID])
 if np.all(bs>0) and np.all(ds>0):
  bn=B/bs;dn=D/ds;dp=dn-bn@np.linalg.lstsq(bn,dn,rcond=None)[0]
  result.update(scaled_B=stat(bn),scaled_joint=stat(np.column_stack([bn,dn])),D_perp=stat(dp))
 return result

def install_fit_audit():
 old=native_gq._fit
 def audited(frame,cols,seed,estimator_config=None):
  CURRENT.update(fit_date=int(frame.period.max())+1,fit_kind='full' if cols else 'base')
  info=dict(history_seed=CURRENT['seed'],stage='full' if cols else 'base',as_of=int(frame.period.max())+1,**matrix_check(frame,cols))
  try:model=old(frame,(),seed) if estimator_config is None else old(frame,(),seed,estimator_config)
  except Exception as exc:info.update(status='failure',error=str(exc),error_type=type(exc).__name__);frame.to_csv(CURRENT['folder']/('FAILED_FIT_%d_%s.csv'%(info['as_of'],info['stage'])),index=False);raise
  else:
   rank=getattr(model.core,'diagnostics_',{}).get('joint_rank')
   info.update(status='success',fitted_joint_rank=None if rank is None else int(rank),
     fitted_rank_status='available' if rank is not None else 'not_applicable_to_selected_estimator',
     physical_inputs=list(native_gq.MEMORY_STATE),estimator=getattr(estimator_config,'estimator','linear'));return model
  finally:
   FIT_AUDITS.append(info)
   with (CURRENT['folder']/'FIT_AUDITS.jsonl').open('a') as f:f.write(json.dumps(info,allow_nan=False)+'\n');f.flush()
 native_gq._fit=audited

import pandas as pd

def logging_actions(t):return native_gq.GRID[(np.arange(31)+(t-32))%11].copy()

def initialize_panel(history):
 last=history.loc[history.period==31].sort_values('path_id');assert last.path_id.tolist()==list(range(32))
 targetx=last[native_gq.XCOL].to_numpy(float);targeta=last.action.to_numpy(float);captured=[]
 original=rnd2_counterfactual._fit_model
 def fit(frame,*a,**kw):
  model=original(frame,*a,**kw)
  if int(frame.period.max())+1==31:
   predict=model.transition_.predict
   def recording(x,actions,*args,**kwargs):
    y=predict(x,actions,*args,**kwargs)
    if np.array_equal(x,targetx) and np.array_equal(actions,targeta):captured.append(np.array(y,copy=True))
    return y
   model.transition_.predict=recording
  return model
 rnd2_counterfactual._fit_model=fit
 try:l,main=rnd2_counterfactual.prepare_rnd2_learner(history,as_of=32,warmup_as_of=8,window=24,path_id=0,refit_every=1,beta=.95,horizon=10,nodes_per_memory=5,noise_nodes=4)
 finally:rnd2_counterfactual._fit_model=original
 if len(captured)!=1:raise RuntimeError('prefix31 issuance capture not unique')
 physical=last[['next_state','next_w','action']].to_numpy(float);aux=np.tile(main.auxiliary,(32,1));aux[:,2]=physical[:,0]-captured[0][:,0]
 np.testing.assert_array_equal(physical[0],main.physical);np.testing.assert_array_equal(aux[0],main.auxiliary)
 receipt=dict(provenance='reconstructed_prefix_not_live_forecast',issue_period=31,release_period=32,fit_as_of=31,path_ids=list(range(32)),issued_base_physical=captured[0][:,:2].tolist(),actual_next_physical=physical.tolist(),initial_auxiliary=aux.tolist())
 return l,physical,aux,receipt

class Panel:
 def __init__(self,learner,physical,auxiliary):self.learner=learner;self.physical=physical.copy();self.auxiliary=auxiliary.copy();self.pending=None
 def prepare(self,t):
  if self.pending is not None:raise ValueError('panel previous release still pending')
  main=self.learner.prepare(t,PublicState(self.physical[0].copy(),self.auxiliary[0].copy()))
  self.auxiliary[:,:2]=main.auxiliary[:2];self.auxiliary[0]=main.auxiliary
 def issue(self,t):
  if self.pending is not None:raise ValueError('panel duplicate issue')
  l=self.learner;main=PublicState(self.physical[0].copy(),self.auxiliary[0].copy());a=float(l.choose(t,main))
  if not l.supported(t,main,a):raise SupportError('main action not supported')
  actions=np.r_[a,logging_actions(t)];mainreceipt=l.issue(t,main,a)
  x=self.physical[1:];aux=self.auxiliary[1:];base=l.model.base_model_.transition_.predict(x,actions[1:])[:,:2];full=l.model.model_.transition_.predict(x,actions[1:],aux)[:,:2]
  if not np.isfinite(base).all() or not np.isfinite(full).all():raise ValueError('nonfinite logging forecast')
  q=l.model.q_values(x,aux);physicalmask=l.model.model_.support_.mask(x,native_gq.GRID)
  pending={0:dict(path_id=0,role='main_greedy',time=t,physical=self.physical[0].tolist(),auxiliary=self.auxiliary[0].tolist(),action=a,issued=mainreceipt)}
  for i in range(1,32):
   index=int(np.flatnonzero(native_gq.GRID==actions[i])[0]);value=q[i-1,index]
   receipt=dict(time=t,fit_as_of=l.model.as_of_,base_fit_as_of=l.model.as_of_,full_fit_as_of=l.model.as_of_,base_state_prediction=float(base[i-1,0]),full_state_prediction=float(full[i-1,0]),base_physical_prediction=base[i-1].tolist(),full_physical_prediction=full[i-1].tolist(),action=float(actions[i]),qualification='exogenous_logging_not_a_Q_recommendation',physical_action_supported=bool(physicalmask[i-1,index]),finite_executed_q=bool(np.isfinite(value)),executed_action_q=float(value) if np.isfinite(value) else None,q_values=[float(v) if np.isfinite(v) else None for v in q[i-1]],qualified_action_mask=np.isfinite(q[i-1]).tolist(),auxiliary_at_issue=self.auxiliary[i].tolist())
   pending[i]=dict(path_id=i,role='exogenous_logging',time=t,physical=self.physical[i].tolist(),auxiliary=self.auxiliary[i].tolist(),action=float(actions[i]),issued=receipt)
  self.pending=pending
  return copy.deepcopy(pending),actions
 def observe(self,t,physical,rewards,release_time):
  if self.pending is None or release_time!=t+1 or set(self.pending)!=set(range(32)):raise ValueError('unpaired panel release')
  pending=self.pending
  if any(p['time']!=t or physical[i,2]!=p['action'] for i,p in pending.items()):raise ValueError('panel action/clock mismatch')
  actual=np.asarray(physical);rewards=np.asarray(rewards)
  if actual.shape!=(32,3) or rewards.shape!=(32,) or not np.isfinite(actual).all() or not np.isfinite(rewards).all():raise ValueError('invalid released panel')
  p=pending[0];mainaux=self.learner.observe(t,PublicState(np.array(p['physical']),np.array(p['auxiliary'])),p['action'],__import__('pinc_toolbox.counterfactual',fromlist=['PublicOutcome']).PublicOutcome(actual[0],float(rewards[0]),release_time))
  nextaux=self.auxiliary.copy();rows=[]
  for i,p in pending.items():
   state=np.array(p['physical']);aux=np.array(p['auxiliary']);nextaux[i,:2]=aux[:2];nextaux[i,2]=actual[i,0]-p['issued']['base_state_prediction']
   # aux is the state captured at issue t, released from transition t-1.
   # nextaux is released now at t+1 and belongs to the next action state.
   if i:rows.append(dict(path_id=i,period=t,release_period=release_time,state=state[0],w=state[1],previous_action=state[2],action=p['action'],next_state=actual[i,0],next_w=actual[i,1],reward=rewards[i],delta_gamma=aux[0],delta_m=aux[1],residual=aux[2],residual_source_period=t-1,residual_release_period=p['time'],auxiliary_as_of=t))
  np.testing.assert_array_equal(nextaux[0],mainaux)
  self.learner.history=pd.concat([self.learner.history,pd.DataFrame(rows)],ignore_index=True)
  self.physical=actual.copy();self.auxiliary=nextaux;self.pending=None
  return nextaux.copy()

HORIZONS=(8,16,32);NAMES=('scalar','serial')
policy=load('panel_frontier_availability',HERE/'availability_policy.py')
from pinc_toolbox import g as g_module


def validate_history(history):
 columns=['path_id','period','release_period','state','w','previous_action','action','next_state','next_w','reward']
 columns +=[x for x in ('delta_gamma','delta_m','residual','residual_source_period','auxiliary_as_of') if x in history]
 if not np.isfinite(history[columns].to_numpy(float)).all():raise ValueError('nonfinite released training history')
 if history.duplicated(['path_id','period']).any() or not np.array_equal(history.release_period,history.period+1):raise ValueError('duplicate or mistimed training history')
 if not np.equal(history.period,np.floor(history.period)).all():raise ValueError('invalid period')
 if 'residual_source_period' in history and not np.array_equal(history.residual_source_period,history.period-1):raise ValueError('invalid residual clock')
 return True

def install_availability_capture():
 prefix_original=rnd2_counterfactual._fit_model
 def prefix_with_context(frame,*args,**kwargs):
  CURRENT.update(fit_date=int(frame.period.max())+1,fit_kind='prefix',raw_validated=validate_history(frame))
  return prefix_original(frame,*args,**kwargs)
 rnd2_counterfactual._fit_model=prefix_with_context
 original=g_module._fit_linear_fwl
 def actual(b,d,y,**kwargs):
  try:return original(b,d,y,**kwargs)
  except Exception as exc:
   classification=policy.classify(exc,raw_inputs_finite=CURRENT.get('raw_validated',False),B=b,D=d,Y=y)
   exc.panel_availability=classification
   index=COUNTS['prefix_G_attempted']+COUNTS['native_G_attempted']
   folder=CURRENT['folder'];np.savez_compressed(folder/('FAILED_SOLVER_INPUT_%05d.npz'%index),B=np.array(b,copy=True),D=np.array(d,copy=True),Y=np.array(y,copy=True))
   save(folder/('FAILED_SOLVER_CAUSE_%05d.json'%index),dict(date=CURRENT.get('fit_date'),stage=CURRENT.get('fit_kind'),classification=classification))
   raise
 g_module._fit_linear_fwl=actual

def classification(exc):
 if isinstance(exc,SupportError):return policy.classify(exc,raw_inputs_finite=CURRENT.get('raw_validated',False))
 return getattr(exc,'panel_availability',{'status':'fatal','reason':'software_or_resource_failure','exception_type':type(exc).__name__,'message':str(exc)})

def score_main(events,H):
 if len(events)!=H:raise ValueError('incomplete scoring window')
 x=[];w=[];weights=[];lag=0.
 for i,event in enumerate(events):
  t=32+i;assert event['time']==t and event['release_time']==t+1 and event['issued']['fit_as_of']==t
  x.append(event['next_physical'][0]-event['issued']['full_state_prediction']);w.append(event['next_physical'][1]-event['issued']['full_physical_prediction'][1]);weights.append(lag);lag=w[-1]
 def energy(e,h):return float(window_slope_energy(np.array(e)[None,:,None],np.array(h)[None],residual_left_inverse=np.eye(1),windows=(H,),eligible=np.ones((1,H,1),bool))['scores'][0,0])
 score=dict(scalar=energy(x,np.ones(H)),serial=energy(w,weights),signed_serial_numerator=float(np.dot(w,weights)),serial_denominator=float(np.dot(weights,weights)))
 if not all(math.isfinite(v) for v in score.values()):raise ValueError('nonfinite score')
 return score

def streams():
 with (HERE/'PLANNED_STREAMS.csv').open() as f:rows=list(csv.DictReader(f))
 for row in rows:
  for k in ('seed','slot','attempt','index'):row[k]=int(row[k]) if row[k]!='' else None
  row['strength']=float(row['strength']);row['source']='STABLE' if row['cell']=='N' else 'LATENT_F';row.update(executed=False,status='not_run',accepted_horizons=[])
 assert len(rows)==273 and len({r['seed'] for r in rows})==273
 return rows

def evaluate(window,slots,H,name):
 return strict_max_tolerance(calibration_slots=slots[str(H)][name],planned_slot_count=59,observed_score=window['scores'][name] if window['available'] else None,observed_available=window['available'],observed_status=window['status'],risk_target=.05,cal_failure_target=.05,scope={'kind':'single_look','release_times':[32+H]})

def protocol(episode,progress=lambda x:None,commit_writer=lambda x:None):
 rows=streams();slots={str(H):{name:[] for name in NAMES} for H in HORIZONS};commits=[];attempts=[];evaluations=[]
 def call(row,target,callback):
  row['executed']=True;row['target_horizon']=target
  try:
   e=episode(copy.deepcopy(row),target,callback)
   if e['status'] not in ('available','main_support_refusal','initialization_support_refusal','main_identification_unavailable','initialization_identification_unavailable'):raise RuntimeError('episode software failure or unknown status')
   row['status']=e['status'];attempts.append({**copy.deepcopy(row),**e});return e
  except Exception:row['status']='software_failure';raise
 try:
  for j in range(59):
   accepted={};candidates=[x for x in rows if x['role']=='CAL' and x['slot']==j]
   for row in candidates:
    unresolved=[H for H in HORIZONS if H not in accepted]
    if not unresolved:row['status']='unused_after_all_horizons_accepted';continue
    def commit(H,scores,trace):
     if H not in unresolved:return
     if H in accepted:raise RuntimeError('duplicate horizon commit')
     if not all(math.isfinite(scores[n]) for n in NAMES):raise ValueError('nonfinite committed score')
     record=dict(slot=j,attempt=row['attempt'],seed=row['seed'],H=H,release_time=32+H,scores=scores,trace_length=len(trace))
     if len(trace)!=H or trace[-1]['release_time']!=32+H:raise ValueError('mistimed CAL commit')
     # Persist immediately before allowing further episode execution.
     commit_writer(record);accepted[H]=record;commits.append(record);row['accepted_horizons'].append(H)
    e=call(row,max(unresolved),commit)
    for H in unresolved:
     window=e['windows'][str(H)]
     if window['available']!=(H in accepted):raise RuntimeError('callback availability mismatch')
    progress(dict(event='episode',role='CAL',slot=j,seed=row['seed'],status=e['status'],accepted_horizons=list(row['accepted_horizons']),counts=dict(COUNTS)))
   for H in HORIZONS:
    rec=accepted.get(H)
    for name in NAMES:slots[str(H)][name].append(dict(slot=j,status='accepted' if rec else 'cap_exhausted',score_kind='finite' if rec else 'positive_infinity',score=rec['scores'][name] if rec else None,seed=rec['seed'] if rec else None,attempt=rec['attempt'] if rec else None))
  valid=[H for H in HORIZONS if all(s['status']=='accepted' for s in slots[str(H)]['scalar'])]
  for row in rows:
   if row['role']!='EVAL':continue
   if not valid:row['status']='not_run_all_calibration_unavailable';continue
   e=call(row,max(valid),lambda H,s,t:None)
   decisions={str(H):{name:evaluate(e['windows'][str(H)],slots,H,name) for name in NAMES} for H in valid}
   evaluations.append({**copy.deepcopy(row),**e,'decisions':decisions})
   progress(dict(event='episode',role='EVAL',cell=row['cell'],strength=row['strength'],seed=row['seed'],status=e['status'],alarms={h:{n:v['rejected'] for n,v in d.items()} for h,d in decisions.items()},counts=dict(COUNTS)))
  return dict(status='complete' if valid else 'all_calibration_unavailable',valid_horizons=valid,slots=slots,commits=commits,attempts=attempts,evaluations=evaluations,ledger=rows)
 except Exception as exc:return dict(status='software_failure',error=str(exc),traceback=traceback.format_exc(),slots=slots,commits=commits,attempts=attempts,evaluations=evaluations,ledger=rows)

def execute(out):
 out=Path(out);out.mkdir(exist_ok=False);start=time.time()
 save(out/'CONFIG.json',dict(pid=os.getpid(),runtime=json.loads((HERE/'RUNTIME_AND_STORAGE.json').read_text()),source=rnd2_counterfactual.__file__,started_epoch=start,limits=LIMIT,horizons=HORIZONS,constructor_horizon=64,availability_policy='typed finite-input verified identification failures and native support reasons; all other faults fatal'))
 driver=load('frontier_panel_prefix',HERE/'dependencies/prefix_driver.py');dgp=driver._load_dgp();install_counters(dgp);install_fit_audit();install_availability_capture()
 def progress(event):
  event['elapsed_seconds']=time.time()-start;line=json.dumps(event,allow_nan=False)
  with (out/'PROGRESS.jsonl').open('a') as f:f.write(line+'\n');f.flush()
  print(line,flush=True)
 def commit_writer(record):
  save(out/('slot_%02d_H%02d_COMMIT.json'%(record['slot'],record['H'])),record)
  with (out/'SLOT_COMMITS.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
 def episode(row,target,callback):
  storage();folder=out/('episode_'+str(row['seed']));folder.mkdir();CURRENT.update(seed=row['seed'],folder=folder,date=None,stage='prefix',raw_validated=False);before=dict(COUNTS)
  events=[];windows={};phase='initialization';failure=None;status='available';t=None
  try:
   history,env=driver._prefix(dgp,row['seed'],row['source'],row['strength']);assert env.horizon==64 and env.switch_date==32 and env._next_t==32
   CURRENT['raw_validated']=validate_history(history);CURRENT['date']=32
   history.to_csv(folder/'prefix.csv',index=False,float_format='%.17g')
   save(folder/'oracle_truth.json',dict(oracle_only=True,constructor_horizon=64,switch_date=32,initial_Z=env.latent[:,32].tolist(),latent_innovations=env.latent_noise[:,32:64].tolist(),state_innovations=env.state_noise[:,32:64].tolist(),measurement_innovations=env.measurement_noise[:,32:65].tolist()))
   l,physical,aux,capture=initialize_panel(history);save(folder/'PREFIX_INITIALIZATION.json',capture);panel=Panel(l,physical,aux);phase='main'
   for t in range(32,32+target):
    CURRENT.update(date=t,stage='online',raw_validated=validate_history(l.history))
    if not np.isfinite(panel.physical).all() or not np.isfinite(panel.auxiliary).all():CURRENT['raw_validated']=False;raise ValueError('nonfinite panel before prepare')
    panel.prepare(t);storage()
    with gzip.open(folder/('snapshot_%02d.pkl.gz'%t),'xb',compresslevel=3) as f:cloudpickle.dump(l,f)
    save(folder/('states_%02d.json'%t),dict(time=t,physical=panel.physical.tolist(),auxiliary=panel.auxiliary.tolist()))
    issued,actions=panel.issue(t);save(folder/('ISSUED_%02d.json'%t),list(issued.values()))
    assert np.array_equal(panel.physical[:,0],env.state[:,t]) and np.array_equal(panel.physical[:,1],env.proxy_all[:,t,0])
    env.step(t,actions);x,w=env.public(t+1);actual=np.column_stack([x,w,actions]);rewards=env.released_reward(t+1)
    save(folder/('RELEASED_%02d.json'%(t+1)),dict(release_time=t+1,physical=actual.tolist(),reward=rewards.tolist()))
    nextaux=panel.observe(t,actual,rewards,t+1);save(folder/('AUXILIARY_%02d.json'%(t+1)),dict(release_time=t+1,auxiliary=nextaux.tolist()))
    events.append(dict(time=t,release_time=t+1,issued=issued[0]['issued'],physical=issued[0]['physical'],auxiliary=issued[0]['auxiliary'],action=float(actions[0]),next_physical=actual[0].tolist(),next_auxiliary=nextaux[0].tolist(),reward=float(rewards[0])))
    H=t-31
    if H in HORIZONS:
     scores=score_main(events,H);windows[str(H)]=dict(status='available',available=True,scores=scores)
     save(folder/('WINDOW_H%02d.json'%H),dict(H=H,release_time=32+H,scores=scores,main_events=events));callback(H,scores,copy.deepcopy(events))
  except Exception as exc:
   cause=classification(exc);failure=dict(time=t if t is not None else 32,stage=phase,classification=cause,type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
   if cause['status']=='fatal':
    save(folder/'FAILURE.json',dict(status='software_failure',failure=failure,counts={k:COUNTS[k]-before[k] for k in COUNTS}));raise
   status=('initialization_' if phase=='initialization' else 'main_')+('identification_unavailable' if cause['status']=='identification_unavailable' else 'support_refusal')
  for H in HORIZONS:
   if H<=target and str(H) not in windows:windows[str(H)]=dict(status=status,available=False,scores=None,unavailability_reason=failure['classification'] if failure else None)
  result=dict(status=status,windows=windows,failure=failure,completed_dates=len(events),target_horizon=target,main_events=events,counts={k:COUNTS[k]-before[k] for k in COUNTS})
  save(folder/'RESULT.json',result);return {k:v for k,v in result.items() if k!='main_events'}|{'episode_dir':folder.name}
 result=protocol(episode,progress,commit_writer);result.update(counts=COUNTS,pid=os.getpid(),elapsed_seconds=time.time()-start)
 save(out/'RESULT.json',result);print(json.dumps(dict(final_status=result['status'],valid_horizons=result.get('valid_horizons'),counts=COUNTS,elapsed_seconds=result['elapsed_seconds'])),flush=True)
 if result['status']=='software_failure':raise SystemExit(1)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--output',type=Path,default=HERE/'execution001');args=parser.parse_args()
 if not args.run:parser.error('--run required')
 execute(args.output)
