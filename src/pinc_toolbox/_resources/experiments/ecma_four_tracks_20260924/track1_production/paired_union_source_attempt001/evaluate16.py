"""Paired16 H10/direct-Bellman controller. Science starts only with --execute."""
from __future__ import annotations
import argparse, json, os, shutil, signal, subprocess, sys, time, traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
TRACK=HERE.parent
SOURCE_RUN=TRACK/'model_union_signed_confirmation3_run01'
TOTAL_OUTPUT=2*1024**3
RESERVE=512*1024**2
SINGLE_WORKER_CAP=1536*1024**2
TOTAL_SECONDS=7200
META_RESERVE=2*1024**2
FAMILIES=('N','CL','CM','MIX')


def plan_rows():
    rows=[]
    for ordinal in range(16):
        rows.append(dict(ordinal=ordinal, candidate=FAMILIES[ordinal%4],
                         seed=3252770000+ordinal,
                         conditional_future_campaign_entropy=202609263400+ordinal))
    return rows


def validate_plan(rows):
    assert len(rows)==16
    assert [x['ordinal'] for x in rows]==list(range(16))
    assert [sum(x['candidate']==f for x in rows) for f in FAMILIES]==[4]*4
    assert len({x['seed'] for x in rows})==16
    assert [x['seed'] for x in rows]==list(range(3252770000,3252770016))
    assert [x['conditional_future_campaign_entropy'] for x in rows]==list(range(202609263400,202609263416))
    return True


def allocation_for(row):
    return dict(
        target='paired16_one_monitor_per_assigned_history',
        monitor_proposals=[dict(ordinal=row['ordinal'],seed=row['seed'],candidate=row['candidate'])],
        conditional_future_campaign_entropy=row['conditional_future_campaign_entropy'],
        source_candidates=['N','CL'], target_complete=39, proposal_cap=128, source_alpha=.05,
        source_statistic='signed sum of high-minus-low FULL-x residuals over dates 40..47',
        tail='two-sided conservative rank: min(1,2*min(lower,upper))',
        route='H10 direct Bellman; fixed remaining_horizon=10')


def summarize(plan, results):
    """Fixed assignment denominators; never reassign or drop an outcome."""
    by={f:dict(assigned=0,model_involvement=0,unresolved=0,no_alarm=0,
               monitor_support_unavailable=0,technical_invalid=0,not_run=0) for f in FAMILIES}
    for row in plan:
        cell=by[row['candidate']];cell['assigned']+=1
        result=results.get(row['ordinal'])
        if result is None or result.get('status')=='not_run':
            cell['not_run']+=1;cell['unresolved']+=1;continue
        if result.get('execution_status')=='invalid':
            cell['technical_invalid']+=1;cell['unresolved']+=1;continue
        if result.get('monitor_status')=='no_alarm':cell['no_alarm']+=1
        if result.get('monitor_status')=='support_unavailable':cell['monitor_support_unavailable']+=1
        if result.get('decision')=='model_involvement':cell['model_involvement']+=1
        else:cell['unresolved']+=1
    assert all(v['assigned']==4 for v in by.values())
    return by


def used_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def enough_resources(free_bytes, remaining_output, remaining_seconds):
    return remaining_output > META_RESERVE + 16*1024**2 and free_bytes >= remaining_output + RESERVE and remaining_seconds > 0


def _load(name, path):
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def worker(row, out, output_allowance, time_allowance):
    """Execute one allocated history; this is dormant unless called via --worker."""
    import math, traceback
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    ROOT=HERE.parents[3]
    sys.path.insert(0,str(ROOT/'software/pinc_toolbox/src'))
    base=_load('paired16_science_adapter',SOURCE_RUN/'science_adapter.py')
    base.HERE=TRACK/'direct_bellman_conditional_source_attempt001'
    base.monitor=_load('paired16_monitor',HERE/'monitor.py')
    base.OUTPUT_BYTES=min(SINGLE_WORKER_CAP,max(0,int(output_allowance)-META_RESERVE))
    base.TEMP_BYTES=0;base.RESERVE_BYTES=RESERVE
    base.TIME_SECONDS=max(1,int(time_allowance))
    base.LIMIT=dict(prefix_G=3200,native_G=8192,Q=4096,reward=4096,constructors=128,
                    host_calls=32768,transitions=1048576,conditional_initializations=256)
    alloc=allocation_for(row)
    xread=json.loads((TRACK/'h8_xscalar_readback_attempt001/READBACK.json').read_text())
    thresholds=dict(serial=base.detector_threshold(),scalar=xread['scalar_threshold'])
    if thresholds['serial']!=xread['serial_threshold']:
        raise RuntimeError('fixed H8 serial/scalar threshold source mismatch')
    cal=json.loads((TRACK/'paired_monitor_calibration_attempt001/execution001/RESULT.json').read_text())
    if cal['status']!='bank_complete' or len(cal['bank'])!=119:raise RuntimeError('paired null calibration unavailable')
    thresholds['paired_bank']=cal['bank']
    started=time.monotonic();state=dict(ordinal=row['ordinal'],candidate=row['candidate'],seed=row['seed'],
        entropy=row['conditional_future_campaign_entropy'],route='H10 direct Bellman, remaining_horizon=10',
        execution_status='running',monitor_status=None,decision='unresolved',latent='unresolved',
        thresholds=thresholds,monitor=None,banks={},observed=None,inference=None)
    def progress():
        (out/'PROGRESS.json').write_text(json.dumps(dict(ordinal=row['ordinal'],monitor_status=state['monitor_status'],banks={c:dict(proposals=len(b['attempts']),accepted=len(b['values'])) for c,b in state['banks'].items()},elapsed_seconds=time.monotonic()-started))+'\n')
    science=None
    def deadline(*_): raise TimeoutError('paired16 total time budget')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,max(1,int(time_allowance)))
    try:
        (out/'PREDECLARED.json').write_text(json.dumps(dict(allocation=alloc,thresholds=thresholds,
            output_allowance=base.OUTPUT_BYTES,reserve_bytes=RESERVE,remaining_seconds=time_allowance),indent=2)+'\n')
        expanded=_load('paired16_expanded',HERE/'expanded_condition.py')
        science=expanded.science_class(base)(out,alloc,thresholds)
        acquisition=_load('paired16_acquisition',TRACK/'paired_monitor_acquisition_attempt001/paired_monitor.py')
        base.monitor.install(acquisition,science.b,science.dispatch,science.fixed,science.future)
        proposal=alloc['monitor_proposals'][0]
        monitor=science.monitor(proposal)
        if monitor['status']=='fatal':raise RuntimeError('monitor technical failure: '+str(monitor.get('failure')))
        h=None if monitor['status']!='available' else base.monitor.union_alarm(monitor['scores'],thresholds)
        state.update(monitor=monitor,monitor_alarm=h)
        if monitor['status']!='available':
            state['monitor_status']='support_unavailable'
        elif not h:
            state['monitor_status']='no_alarm'
        else:
            state['monitor_status']='alarm'
            progress()
            public=science.condition(row['ordinal'])
            for candidate in alloc['source_candidates']:
                bank=dict(values=[],attempts=[]);state['banks'][candidate]=bank
                for rep in range(alloc['proposal_cap']):
                    raw=science.calibration(public,candidate,rep,f'{candidate}:{row["ordinal"]}:{rep}')
                    if raw['status']=='fatal':raise RuntimeError(f'{candidate} CAL technical failure: '+str(raw.get('failure')))
                    receipt=science.branch.compact(raw,science.measurements)
                    score=base_signed_sum(receipt)
                    attempt=dict(replicate_id=rep,complete=bool(receipt['C_A']),signed_sum=score,
                                 status=raw['status'],reasons=receipt['reasons']['A'])
                    bank['attempts'].append(attempt)
                    if attempt['complete']:bank['values'].append(score)
                    progress()
                    if len(bank['values'])==39:break
                bank['status']='complete' if len(bank['values'])==39 else 'unavailable'
            science.write_json(out/'CAL_COMPLETE.json',dict(banks=state['banks']))
            raw=science.observed(public)
            if raw['status']=='fatal':raise RuntimeError('observed technical failure: '+str(raw.get('failure')))
            receipt=science.branch.compact(raw,science.measurements);observed=base_signed_sum(receipt)
            state['observed']=dict(complete=bool(receipt['C_A']),signed_sum=observed,
                                   status=raw['status'],reasons=receipt['reasons']['A'])
            ranks={}
            if receipt['C_A'] and all(b['status']=='complete' for b in state['banks'].values()):
                for c,b in state['banks'].items():
                    lower=(1+sum(v<=observed for v in b['values']))/40
                    upper=(1+sum(v>=observed for v in b['values']))/40
                    ranks[c]=min(1.0,2*min(lower,upper))
            state['inference']=dict(rank_p=ranks,tail='two-sided',alpha=.05,
                decision='model_involvement' if len(ranks)==2 and all(p<=.05 for p in ranks.values()) else 'unresolved',
                latent='unresolved',statistic='signed_sum_paired_FULL_x_residual')
            state['decision']=state['inference']['decision']
        state['execution_status']='complete'
    except BaseException as exc:
        state.update(execution_status='invalid',failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc()),decision='unresolved',inference=None)
        if science is not None and science.b.ARCHIVE is not None:
            try:science.b.ARCHIVE.abort()
            except BaseException:pass
            science.b.ARCHIVE=None
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        state['elapsed_seconds']=time.monotonic()-started
        if science is not None:
            state['counts']=science.b.COUNTS
            science.b.BUDGET.reporting=True
        # Terminal receipt is written outside the scientific archive; keep it small.
        (out/'RESULT.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    return state


def base_signed_sum(receipt):
    if not receipt['C_A']:return None
    total=0.
    for t in range(40,48):
        es=[e for e in receipt['endpoints'] if e['time']==t]
        lo=next(e for e in es if e['action']==.02)
        hi=next(e for e in es if e['action']!=.02)
        total+=hi['full_x_error']-lo['full_x_error']
    return total


def run_batch(execution, execute=False):
    rows=plan_rows();validate_plan(rows);execution=Path(execution)
    if not execute:
        return dict(status='design_only',assigned=rows,method='one fixed H10/direct-Bellman monitor per history; no alarm -> unresolved, no next alarm search')
    if execution.exists():raise FileExistsError(execution)
    execution.mkdir(parents=True)
    free=shutil.disk_usage(execution).free
    if not enough_resources(free,TOTAL_OUTPUT, TOTAL_SECONDS):
        rec=dict(status='not_run',reason='preflight requires 2 GiB output allowance plus 512 MiB reserve and positive 7200-second budget',
                 free_bytes=free,assigned=16,not_run_ordinals=list(range(16)))
        (execution/'NOT_RUN.json').write_text(json.dumps(rec,indent=2)+'\n');return rec
    plan=dict(status='predeclared',assigned=rows,per_family=8,total_output_bytes=TOTAL_OUTPUT,
              reserve_bytes=RESERVE,total_seconds=TOTAL_SECONDS,
              route='H10 direct Bellman; remaining_horizon=10; no next-alarm search')
    (execution/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    results={};start=time.monotonic()
    for row in rows:
        remain_time=int(TOTAL_SECONDS-(time.monotonic()-start))
        remain_output=TOTAL_OUTPUT-used_bytes(execution)-META_RESERVE
        free=shutil.disk_usage(execution).free
        if not enough_resources(free,max(0,remain_output),remain_time):
            for rest in rows[row['ordinal']:]:results[rest['ordinal']]=dict(status='not_run')
            break
        trial=execution/f'history_{row["ordinal"]:02d}'
        cfg=allocation_for(row);trial.mkdir()
        (trial/'ALLOCATION.json').write_text(json.dumps(cfg,indent=2)+'\n')
        cap=min(SINGLE_WORKER_CAP,max(0,remain_output))
        worker_args=[sys.executable,str(Path(__file__).resolve()),'--worker',str(row['ordinal']),
                     '--out',str(trial/'execution001'),'--output-allowance',str(cap),
                     '--time-allowance',str(max(1,remain_time-5))]
        try:
            proc=subprocess.run(worker_args,timeout=remain_time,check=False)
            result_path=trial/'execution001/RESULT.json'
            result=json.loads(result_path.read_text()) if result_path.exists() else dict(execution_status='invalid',failure='worker produced no terminal receipt')
            results[row['ordinal']]=result
            if proc.returncode!=0 or result.get('execution_status')!='complete':
                for rest in rows[row['ordinal']+1:]:results[rest['ordinal']]=dict(status='not_run')
                break
        except subprocess.TimeoutExpired:
            results[row['ordinal']]=dict(execution_status='invalid',failure='global time limit expired')
            for rest in rows[row['ordinal']+1:]:results[rest['ordinal']]=dict(status='not_run')
            break
        summary=dict(status='running',completed=len(results),results=results,by_family=summarize(rows,results))
        (execution/'PROGRESS.json').write_text(json.dumps(summary,indent=2)+'\n')
    final=dict(status='complete' if len(results)==16 and all(r.get('execution_status')=='complete' for r in results.values()) else 'incomplete',
               assigned=rows,results=results,by_family=summarize(rows,results),elapsed_seconds=time.monotonic()-start,
               no_replacement=True)
    (execution/'SUMMARY.json').write_text(json.dumps(final,indent=2)+'\n')
    return final


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--execute',action='store_true',help='start the 16 fixed scientific histories')
    ap.add_argument('--out',type=Path,default=HERE/'execution001')
    ap.add_argument('--worker',type=int,choices=range(16))
    ap.add_argument('--output-allowance',type=int,default=0)
    ap.add_argument('--time-allowance',type=int,default=0)
    args=ap.parse_args()
    if args.worker is not None:
        return worker(plan_rows()[args.worker],args.out,args.output_allowance,args.time_allowance)
    receipt=run_batch(args.out,execute=args.execute)
    print(json.dumps({k:v for k,v in receipt.items() if k!='assigned' and k!='results'},indent=2))
    return receipt

if __name__=='__main__': main()
