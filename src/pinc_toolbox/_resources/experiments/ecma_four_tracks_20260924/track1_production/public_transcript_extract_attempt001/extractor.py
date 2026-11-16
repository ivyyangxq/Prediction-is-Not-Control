"""Public-only archive prefix extraction; no scientific object or future decoding."""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import json,pickle,lzma,io,copy,traceback
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
BASE_COLUMNS=['path_id','period','release_period','state','w','previous_action','action','next_state','next_w','reward']
AUX_COLUMNS=['delta_gamma','delta_m','residual','residual_source_period','auxiliary_as_of']
RELEASE_METADATA='residual_release_period'
LEGACY_HISTORY_COLUMNS=BASE_COLUMNS+AUX_COLUMNS
HISTORY_COLUMNS=BASE_COLUMNS+AUX_COLUMNS[:-1]+[RELEASE_METADATA]+AUX_COLUMNS[-1:]
TAU1=.05-.95/(1.-.95*.80)*(.8-4.*.05)+.3
LIMIT=50*1024**2
class PublicInputError(ValueError):pass

def validate_history_dates(frame,*,as_of):
 """Accept only the two declared public schemas; never infer a missing date."""
 columns=list(frame.columns)
 expected=HISTORY_COLUMNS if RELEASE_METADATA in columns else LEGACY_HISTORY_COLUMNS
 if not frame.columns.is_unique or set(columns)!=set(expected):
  raise PublicInputError('nonpublic or incomplete learner-history column')
 clocks=['period','release_period','residual_source_period','auxiliary_as_of']
 if RELEASE_METADATA in columns:clocks.append(RELEASE_METADATA)
 for col in clocks:
  values=frame[col].to_numpy(float)
  if not np.isfinite(values).all() or not np.equal(values,np.floor(values)).all():
   raise PublicInputError('noninteger or unknown learner-history clock')
 if (not frame.release_period.eq(frame.period+1).all() or
     not frame.residual_source_period.eq(frame.period-1).all() or
     not frame.auxiliary_as_of.eq(frame.period).all() or
     not frame.release_period.le(as_of).all()):
  raise PublicInputError('learner auxiliary clock or future physical release')
 if RELEASE_METADATA in columns:
  release=frame[RELEASE_METADATA]
  if (not release.eq(frame.period).all() or
      not release.eq(frame.auxiliary_as_of).all() or
      not frame.residual_source_period.lt(release).all() or
      not release.lt(frame.release_period).all()):
   raise PublicInputError('residual release must precede its action-row target')
 return expected

def read_public_until40(path):
 records={};histories={};boundary_history=None;names=[];decoded=[];snapshots=[];metadata_rows=0
 wanted={'prefix.csv'}|{f'{kind}_{t}.json' for kind in ('states','ISSUED') for t in range(32,40)}|{f'{kind}_{t}.json' for kind in ('RELEASED','AUXILIARY') for t in range(33,41)}
 snapshot_dates={**{f'snapshot_{t}.pkl.gz':t for t in range(32,40)},
                 **{f'released_learner_{t}.pkl.gz':t for t in range(33,41)}}
 with lzma.open(path,'rb') as stream:
  while True:
   r=pickle.load(stream)
   if r[0]=='end':raise PublicInputError('monitor boundary not reached')
   kind,name=r[:2]
   if name in names and kind!='append':raise PublicInputError('duplicate archived record name')
   names.append(name)
   # The container transports unused bytes; only explicitly allowed public JSON is decoded.
   # Oracle payloads and all learner-shell pickle payloads remain opaque and unused.
   if kind=='raw' and name in wanted:
    records[name]=pd.read_csv(io.BytesIO(r[2]),float_precision='round_trip') if name=='prefix.csv' else json.loads(r[2]);decoded.append(name)
   elif kind=='snapshot':
    _,name,chain,base,delta,payload=r
    if name not in snapshot_dates:raise PublicInputError('unexpected or future learner-history snapshot')
    columns=validate_history_dates(delta,as_of=snapshot_dates[name])
    if base is not None and base not in histories:raise PublicInputError('missing public history delta base')
    if base is not None and list(histories[base].columns)!=columns:raise PublicInputError('public history delta schema changed')
    delta=delta[columns]
    if RELEASE_METADATA in columns:metadata_rows+=len(delta)
    hist=delta.copy() if base is None else pd.concat([histories[base],delta],ignore_index=True)
    histories[chain]=hist;snapshots.append(name)
    if name=='released_learner_40.pkl.gz':boundary_history=hist.copy();break
 if set(records)!=wanted:raise PublicInputError('incomplete public prefix/monitor records')
 return records,boundary_history,dict(decoded_public_record_names=decoded,public_history_snapshots=snapshots,stop_record='released_learner_40.pkl.gz',model_payloads_decoded=0,oracle_payloads_decoded=0,future_records_decoded=0,residual_release_metadata_rows_validated=metadata_rows,archive_terminal_checked=False,scope='intentional public prefix only; full archive terminal was separately audited')

def close(x,y,label):
 x=np.asarray(x);y=np.asarray(y)
 if x.shape!=y.shape or not np.allclose(x,y,atol=1e-14,rtol=0):raise PublicInputError(label+' mismatch')

def validate_table(frame,*,dates,columns):
 if list(frame.columns)!=columns:raise PublicInputError('unexpected public table schema')
 for col in ('path_id','period','release_period'):
  a=frame[col].to_numpy(float)
  if not np.isfinite(a).all() or not np.equal(a,np.floor(a)).all():raise PublicInputError('noninteger public clock/ID')
 if frame.duplicated(['path_id','period']).any():raise PublicInputError('duplicate public path/date')
 expected={(i,t) for t in dates for i in range(32)}
 if set(zip(frame.path_id,frame.period))!=expected:raise PublicInputError('missing/extra public path/date')
 if not np.array_equal(frame.release_period,frame.period+1):raise PublicInputError('mistimed public release')
 if not np.isfinite(frame.to_numpy(float)).all():raise PublicInputError('nonfinite public field')

def assemble(records,history):
 history_columns=validate_history_dates(history,as_of=40)
 prefix=records['prefix.csv'].copy();validate_table(prefix,dates=range(32),columns=BASE_COLUMNS)
 rows=prefix.to_dict('records');monitor_rows=[];issued_public=[]
 for t in range(32,40):
  states=records[f'states_{t}.json'];issued=records[f'ISSUED_{t}.json'];released=records[f'RELEASED_{t+1}.json'];aux=records[f'AUXILIARY_{t+1}.json']
  if states['time']!=t or released['release_time']!=t+1 or aux['release_time']!=t+1:raise PublicInputError('monitor record clock')
  x=np.asarray(states['physical'],float);e=np.asarray(states['auxiliary'],float);nx=np.asarray(released['physical'],float);ne=np.asarray(aux['auxiliary'],float);reward=np.asarray(released['reward'],float)
  if any(v.shape!=(32,3) for v in (x,e,nx,ne)) or reward.shape!=(32,) or not all(np.isfinite(v).all() for v in (x,e,nx,ne,reward)):raise PublicInputError('monitor panel shape/finite')
  if len(issued)!=32 or {r['path_id'] for r in issued}!=set(range(32)):raise PublicInputError('duplicate/missing issued path')
  for row in sorted(issued,key=lambda z:z['path_id']):
   i=row['path_id'];p=row['issued'];a=row['action']
   if row['time']!=t or p['time']!=t or any(p[k]!=t for k in ('fit_as_of','base_fit_as_of','full_fit_as_of')) or p['action']!=a:raise PublicInputError('issued forecast clock/action')
   close(row['physical'],x[i],'issued physical');close(row['auxiliary'],e[i],'issued auxiliary')
   if 'auxiliary_at_issue' in p:close(p['auxiliary_at_issue'],e[i],'forecast auxiliary')
   close(nx[i,2],a,'released action memory')
   close(ne[i,:2],e[i,:2],'within-step auxiliary carry')
   close(ne[i,2],nx[i,0]-p['base_state_prediction'],'released public residual')
   r=dict(path_id=i,period=t,release_period=t+1,state=x[i,0],w=x[i,1],previous_action=x[i,2],action=a,next_state=nx[i,0],next_w=nx[i,1],reward=reward[i]);rows.append(r)
   monitor_rows.append({**r,'delta_gamma':e[i,0],'delta_m':e[i,1],'residual':e[i,2],
                        'residual_source_period':t-1,'residual_release_period':t,'auxiliary_as_of':t})
   issued_public.append(dict(path_id=i,time=t,physical=x[i].tolist(),auxiliary=e[i].tolist(),action=a,fit_as_of=p['fit_as_of'],base_physical_prediction=p['base_physical_prediction'],full_physical_prediction=p['full_physical_prediction']))
 public=pd.DataFrame(rows,columns=BASE_COLUMNS);validate_table(public,dates=range(40),columns=BASE_COLUMNS)
 x=np.empty((32,41));w=np.empty((32,41));actions=np.empty((32,40));rewards=np.empty((32,40));prev=np.empty((32,40))
 for i in range(32):
  f=public[public.path_id==i].sort_values('period')
  close(f.next_state.to_numpy()[:-1],f.state.to_numpy()[1:],'public x recurrence')
  close(f.next_w.to_numpy()[:-1],f.w.to_numpy()[1:],'public w recurrence')
  close(f.action.to_numpy()[:-1],f.previous_action.to_numpy()[1:],'previous-action recurrence')
  x[i]=np.r_[f.state.to_numpy(),f.next_state.iloc[-1]];w[i]=np.r_[f.w.to_numpy(),f.next_w.iloc[-1]];actions[i]=f.action;rewards[i]=f.reward;prev[i]=f.previous_action
 close(prev[:,0],np.full(32,.05),'prefix initial previous action')
 if not np.all(np.any(np.isclose(actions[:,:,None],np.linspace(0,.1,11),atol=1e-14,rtol=0),axis=2)):raise PublicInputError('action outside public grid')
 for t in range(32,40):close(actions[1:,t],np.linspace(0,.1,11)[(np.arange(31)+t-32)%11],'companion public logging rule')
 expected_reward=x[:,:40]-.25*x[:,:40]**2-.50*actions**2+TAU1*actions
 close(rewards,expected_reward,'deterministic public reward')
 # The saved learner starts at warmup8. Preserve that actual annotated public history,
 # and independently verify all its physical records against the complete transcript.
 history=history[history_columns].copy();validate_table(history,dates=range(8,40),columns=history_columns)
 expected=public[public.period>=8].sort_values(['path_id','period']).reset_index(drop=True)
 actual=history.sort_values(['path_id','period']).reset_index(drop=True)
 close(actual[BASE_COLUMNS].to_numpy(),expected[BASE_COLUMNS].to_numpy(),'learner/public physical history')
 monitor=pd.DataFrame(monitor_rows,columns=history_columns).sort_values(['path_id','period']).reset_index(drop=True)
 close(actual[actual.period>=32].reset_index(drop=True).to_numpy(),monitor.to_numpy(),'learner monitor annotations')
 if not np.array_equal(actual.residual_source_period,actual.period-1) or not np.array_equal(actual.auxiliary_as_of,actual.period):raise PublicInputError('learner auxiliary clock')
 boundary=dict(release_time=40,next_decision_time=40,last_prepared_as_of=39,pending_issue=False,panel_physical=records['RELEASED_40.json']['physical'],panel_auxiliary=records['AUXILIARY_40.json']['auxiliary'],learner_history_first_period=8,learner_history_last_period=39,learner_history_rows=len(actual),model_payload_exported=False,policy=dict(auxiliary='full',refit_every=1,window=24,horizon=10,beta=.95,nodes_per_memory=5,noise_nodes=4,action_grid=np.linspace(0,.1,11).tolist()),H=None,H_status='not_calibrated',scope='public released panel and annotated learner-history boundary; fitted learner model remains in source archive, not a standalone executable learner')
 arrays=dict(x=x,w=w,actions=actions,rewards=rewards,dates=np.arange(41,dtype=int))
 return arrays,public,actual,issued_public,boundary,dict(path_count=32,state_dates=41,action_dates=40,public_transition_rows=1280,learner_history_rows=len(actual),max_reward_identity_error=float(np.max(np.abs(rewards-expected_reward))),earliest_public_date=0,latest_public_release=40)

def run():
 out=P/'output001';out.mkdir(exist_ok=False)
 def write(name,data):
  if isinstance(data,str):data=data.encode()
  used=sum(f.stat().st_size for f in P.rglob('*') if f.is_file())
  if used+len(data)>LIMIT:raise OSError('50MiB output limit')
  with (out/name).open('xb') as f:f.write(data)
 try:
  campaign=ROOT/'root_paired_source_campaign_attempt001';config=json.loads((campaign/'PREDECLARED_CONFIG.json').read_text())
  first=next(s for s in config['plan'] if s['cell']=='N' and s['phase']=='CAL')
  archive=campaign/'execution001'/('episode_'+str(first['seed']))/'records.xz'
  records,history,readback=read_public_until40(archive);arrays,table,lh,issued,boundary,checks=assemble(records,history)
  b=io.BytesIO();np.savez_compressed(b,**arrays);write('FILTER_INPUT.npz',b.getvalue())
  write('PUBLIC_TRANSITIONS.csv',table.to_csv(index=False,float_format='%.17g'))
  write('PUBLIC_LEARNER_HISTORY.csv',lh.to_csv(index=False,float_format='%.17g'))
  write('PUBLIC_MONITOR_ISSUED.json',json.dumps(issued,indent=2)+'\n')
  write('PUBLIC_BOUNDARY.json',json.dumps(boundary,indent=2)+'\n')
  provenance=dict(source_archive=str(archive),selection='first N CAL proposal in predeclared order; public monitor completeness only, no score or source-window inspection',selected_position=first['index'],path_ids=list(range(32)),filter_input_keys=list(arrays),filter_receives_no_recordID=True,**readback)
  write('PROVENANCE_NOT_FILTER_INPUT.json',json.dumps(provenance,indent=2)+'\n')
  report=dict(status='PASS',**checks,source_histories=1,model_queries=0,new_fits=0,new_Q_solves=0,new_DGP=0,posterior_filter_calls=0,hidden_or_future_inputs=False,output_bytes=sum(f.stat().st_size for f in out.iterdir()),scope='public transcript through40 only; no posterior or future generation')
  write('RESULT.json',json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 except BaseException as exc:
  write('FAILURE.json',json.dumps(dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc()),indent=2)+'\n');raise
if __name__=='__main__':run()
