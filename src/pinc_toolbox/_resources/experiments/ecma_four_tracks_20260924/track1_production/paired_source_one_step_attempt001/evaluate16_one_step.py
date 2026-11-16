"""Fresh fixed16 one-step source-window driver; adapts evaluate16 in memory, never on import."""
from pathlib import Path
import argparse,json,math,sys,types,importlib.util
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
BASE_DRIVER=PARENT/'paired_union_source_attempt001/evaluate16.py'
ONE_STEP=HERE/'one_step_future.py'
FAMILIES=('N','CL','CM','MIX')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def plan_rows():
    return [dict(ordinal=i,candidate=FAMILIES[i%4],seed=3252780000+i,
                 conditional_future_campaign_entropy=202609263500+i) for i in range(16)]


def validate_plan(rows):
    assert len(rows)==16 and [r['ordinal'] for r in rows]==list(range(16))
    assert [sum(r['candidate']==f for r in rows) for f in FAMILIES]==[4]*4
    assert [r['seed'] for r in rows]==list(range(3252780000,3252780016))
    assert [r['conditional_future_campaign_entropy'] for r in rows]==list(range(202609263500,202609263516))
    return True


def compact_one_step(record, score_module=None):
    """Adapter from the saved date40 episode to the existing worker's A-rank interface."""
    endpoints=record.get('endpoints',[]);spec=record.get('spec',{})
    if record.get('status')=='fatal':
        return dict(C_A=None,A=None,endpoints=endpoints,reasons={'A':['fatal']})
    if record.get('H') is not True or record.get('status') not in ('complete','MAIN_unavailable','main_unavailable'):
        return dict(C_A=False,A=None,endpoints=endpoints,reasons={'A':['history_unavailable']})
    if len(endpoints)!=2 or any(e.get('time')!=40 for e in endpoints):
        raise ValueError('one-step receipt must contain exactly two date40 endpoint slots')
    lows=[e for e in endpoints if e.get('action')==.02]
    highs=[e for e in endpoints if e.get('action') is None or e.get('action')>.02]
    if len(lows)!=1 or len(highs)!=1:raise ValueError('missing fixed low or selected high endpoint')
    low,high=lows[0],highs[0]
    if any(e.get('cell')!=spec.get('cell') or e.get('seed')!=spec.get('seed') for e in endpoints):
        raise ValueError('one-step endpoint allocation identity mismatch')
    if low.get('status')=='complete' and low.get('action')!=.02:raise ValueError('low action mismatch')
    if high.get('action') is not None and high.get('selected_high_action')!=high.get('action'):
        raise ValueError('selected high action mismatch')
    for e in (low,high):
        if e.get('status') not in ('complete','support_refusal','MAIN_unavailable','main_unavailable','main_not_available','not_run_after_fatal'):
            raise ValueError('unknown one-step endpoint state')
        if e.get('status')=='complete':
            if e.get('release_time')!=41 or e.get('physical_action_supported') is not True or e.get('public_action_supported') is not True or e.get('finite_executed_Q') is not True:
                raise ValueError('completed endpoint lacks date40 H10 support receipt')
            issued=e.get('issued',{});main=issued.get('main',{})
            prediction=main.get('full_physical_prediction');actual=e.get('next_physical')
            if issued.get('time')!=40 or issued.get('fit_as_of')!=40 or main.get('fit_as_of')!=40 or main.get('action')!=e.get('action'):
                raise ValueError('completed endpoint vintage/action mismatch')
            if not isinstance(prediction,(list,tuple)) or not isinstance(actual,(list,tuple)):
                raise ValueError('completed endpoint missing physical prediction/outcome')
            val=float(actual[0])-float(prediction[0])
            if not math.isfinite(val) or not math.isclose(val,float(e.get('full_x_error',float('nan'))),rel_tol=0.,abs_tol=1e-12):
                raise ValueError('completed endpoint FULL-x residual mismatch')
    if record.get('status')=='complete' and record.get('main_completed') is True and low.get('status')=='complete' and high.get('status')=='complete':
        value=float(high['full_x_error'])-float(low['full_x_error'])
        return dict(C_A=True,A=value,endpoints=endpoints,reasons={'A':['complete_one_step_date40']})
    why='support_refusal' if any(e.get('status')=='support_refusal' for e in (low,high)) else 'incomplete_one_step_pair'
    return dict(C_A=False,A=None,endpoints=endpoints,reasons={'A':[why]})


def _patched_driver():
    d=load('paired16_one_step_base_driver',BASE_DRIVER)
    d.plan_rows=plan_rows
    d.validate_plan=validate_plan
    original_alloc=d.allocation_for
    def alloc(row):
        result=original_alloc(row)
        result.update(source_window='date40 only',source_statistic='date40 high-minus-low FULL-x residual',
                      route='H10 direct Bellman; remaining_horizon=10; source future window=one date')
        return result
    d.allocation_for=alloc
    d.base_signed_sum=lambda receipt: receipt.get('A') if receipt.get('C_A') is True else None
    original_load=d._load
    one_step=load('paired16_one_step_future',ONE_STEP)
    def patched_load(name,path):
        module=original_load(name,path)
        if name=='paired16_expanded':
            factory=module.science_class
            def science_class(base):
                parent=factory(base)
                class OneStepScience(parent):
                    def __init__(self,*args,**kwargs):
                        super().__init__(*args,**kwargs)
                        self.future=one_step
                        # Local receipt adapter; original shared eight-date module is untouched.
                        self.branch=types.SimpleNamespace(compact=compact_one_step)
                return OneStepScience
            module.science_class=science_class
        return module
    d._load=patched_load
    return d


def worker(index,out,output_allowance,time_allowance):
    d=_patched_driver();row=plan_rows()[index]
    return d.worker(row,out,output_allowance,time_allowance)


def run_batch(execution,execute=False):
    d=_patched_driver();rows=plan_rows();validate_plan(rows);execution=Path(execution)
    if not execute:return dict(status='design_only',assigned=rows,per_family_denominator=4,source_window='date40 only',route='H10 unchanged')
    if execution.exists():raise FileExistsError(execution)
    execution.mkdir(parents=True)
    free=__import__('shutil').disk_usage(execution).free
    if not d.enough_resources(free,d.TOTAL_OUTPUT,d.TOTAL_SECONDS):
        rec=dict(status='not_run',reason='preflight requires 2 GiB output allowance plus 512 MiB reserve and positive 7200-second budget',
                 free_bytes=free,assigned=16,per_family_denominator=4,not_run_ordinals=list(range(16)))
        (execution/'NOT_RUN.json').write_text(json.dumps(rec,indent=2)+'\n');return rec
    plan=dict(status='predeclared',assigned=rows,per_family=4,total_output_bytes=d.TOTAL_OUTPUT,
              reserve_bytes=d.RESERVE,total_seconds=d.TOTAL_SECONDS,
              route='H10 direct Bellman; remaining_horizon=10; one source date at40; no next-alarm search')
    (execution/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    results={};started=__import__('time').monotonic()
    for row in rows:
        remain_time=int(d.TOTAL_SECONDS-(__import__('time').monotonic()-started))
        remain_output=d.TOTAL_OUTPUT-d.used_bytes(execution)-d.META_RESERVE
        free=__import__('shutil').disk_usage(execution).free
        if not d.enough_resources(free,max(0,remain_output),remain_time):
            for rest in rows[row['ordinal']:]:results[rest['ordinal']]=dict(status='not_run')
            break
        trial=execution/f'history_{row["ordinal"]:02d}';trial.mkdir()
        cfg=d.allocation_for(row);(trial/'ALLOCATION.json').write_text(json.dumps(cfg,indent=2)+'\n')
        cap=min(d.SINGLE_WORKER_CAP,max(0,remain_output))
        args=[sys.executable,str(Path(__file__).resolve()),'--worker',str(row['ordinal']),
              '--out',str(trial/'execution001'),'--output-allowance',str(cap),
              '--time-allowance',str(max(1,remain_time-5))]
        try:
            proc=__import__('subprocess').run(args,timeout=remain_time,check=False)
            result_path=trial/'execution001/RESULT.json'
            result=json.loads(result_path.read_text()) if result_path.exists() else dict(execution_status='invalid',failure='worker produced no terminal receipt')
            results[row['ordinal']]=result
            if proc.returncode!=0 or result.get('execution_status')!='complete':
                for rest in rows[row['ordinal']+1:]:results[rest['ordinal']]=dict(status='not_run')
                break
        except __import__('subprocess').TimeoutExpired:
            results[row['ordinal']]=dict(execution_status='invalid',failure='global time limit expired')
            for rest in rows[row['ordinal']+1:]:results[rest['ordinal']]=dict(status='not_run')
            break
        progress=dict(status='running',completed=len(results),results=results,by_family=d.summarize(rows,results),per_family_denominator=4)
        (execution/'PROGRESS.json').write_text(json.dumps(progress,indent=2)+'\n')
    final=dict(status='complete' if len(results)==16 and all(r.get('execution_status')=='complete' for r in results.values()) else 'incomplete',
        assigned=rows,results=results,by_family=d.summarize(rows,results),elapsed_seconds=__import__('time').monotonic()-started,
        per_family_denominator=4,source_window='date40 only',no_replacement=True)
    (execution/'SUMMARY.json').write_text(json.dumps(final,indent=2)+'\n');return final


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--out',type=Path,default=HERE/'execution001')
    ap.add_argument('--worker',type=int,choices=range(16));ap.add_argument('--output-allowance',type=int,default=0);ap.add_argument('--time-allowance',type=int,default=0)
    a=ap.parse_args()
    if a.worker is not None:return worker(a.worker,a.out,a.output_allowance,a.time_allowance)
    result=run_batch(a.out,execute=a.execute)
    print(json.dumps({k:v for k,v in result.items() if k not in ('assigned','results')},indent=2))

if __name__=='__main__':main()
