"""Independent 112-history finite-latent confirmation; science requires --execute."""
from __future__ import annotations
import argparse, importlib.util, inspect, json, math, os, shutil, subprocess, sys, time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRACK = ROOT / 'experiments/ecma_four_tracks_20260924/track1_production'
BASE_DRIVER = TRACK / 'paired_union_source_attempt001/evaluate16.py'
sys.path.insert(0,str(ROOT/'codex/physical_g_repair_20261017'))
from stage1_calibration import patch_worker_source, load_thresholds as load_corrected_stage1
EXPANDED = TRACK / 'paired_union_source_attempt001/expanded_condition.py'
ADAPTER = ROOT / 'experiments/ecma_four_tracks_20260924/track3_identification/latent_nuisance_frontier_design_attempt001/finite_latent_catalogue.py'
FAMILIES = ('N', 'CL_Q100', 'CM', 'MIX')
NO_MODEL = ('N', 'CL', 'CL_Q025', 'CL_Q050', 'CL_Q100')
SEED0 = 3260040000
ENTROPY0 = 202610040000
TOTAL_OUTPUT = 5_900_000_000  # conservative campaign cap; leaves HS 1 GiB + reserve/planning 512 MiB each
RESERVE = 512 * 1024**2
META_RESERVE = 2 * 1024**2
SINGLE_WORKER_CAP = 256 * 1024**2
TOTAL_SECONDS = 5 * 60 * 60
FIRST_BLOCK_HISTORIES = 64
FIRST_BLOCK_OUTPUT = 3 * 1024**3


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def plan_rows():
    rows=[]
    for j in range(128):
        for k, family in enumerate(FAMILIES):
            rows.append(dict(ordinal=4*j+k, candidate=family,
                seed=3260040000+j,
                conditional_future_campaign_entropy=202610040000+j,
                pair_group=j))
    return rows


def validate_plan(rows):
    if len(rows) != 512 or [r['ordinal'] for r in rows] != list(range(512)):
        raise ValueError('plan must have exactly ordinals 0..511')
    if [sum(r['candidate'] == f for r in rows) for f in FAMILIES] != [128] * 4:
        raise ValueError('plan must assign 128 histories to each fixed evaluation family')
    if any(rows[4*j+k]['seed'] != 3260040000+j or
           rows[4*j+k]['conditional_future_campaign_entropy'] != 202610040000+j
           for j in range(128) for k in range(4)):
        raise ValueError('unexpected fixed paired DGP seed/entropy allocation')
    return True


def allocation_for(row):
    return dict(target='one_fixed_H10_three-branch-union_monitor_per_history',
        monitor_proposals=[dict(ordinal=row['ordinal'], seed=row['seed'], candidate=row['candidate'])],
        conditional_future_campaign_entropy=row['conditional_future_campaign_entropy'],
        source_candidates=list(NO_MODEL), target_complete=39, proposal_cap=128, source_alpha=.05,
        source_statistic='high-minus-low FULL-x residual at date40 only',
        tail='two-sided conservative rank: min(1,2*min(lower,upper))',
        route='H10 direct Bellman; remaining_horizon=10; paired source window date40 only')


def model_claim(rank_p, required=5, alpha=.05):
    if not isinstance(rank_p, dict):
        raise TypeError('rank_p must be a candidate-to-p-value mapping')
    values = []
    for p in rank_p.values():
        value = float(p)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError('rank p-values must be finite values in [0,1]')
        values.append(value)
    if len(values) == required and set(rank_p) != set(NO_MODEL):
        raise ValueError('complete rank map must match the five predeclared null candidates')
    return len(values) == required and all(p <= alpha for p in values)


def summarize(rows, results):
    out = {f: dict(assigned=0, alarm=0, no_alarm=0, support_unavailable=0,
                   model_involvement=0, unresolved=0, technical_invalid=0, not_run=0)
           for f in FAMILIES}
    for row in rows:
        cell = out[row['candidate']]
        cell['assigned'] += 1
        result = results.get(row['ordinal'])
        if result is None or result.get('status') == 'not_run':
            cell['not_run'] += 1; cell['unresolved'] += 1; continue
        if result.get('execution_status') == 'invalid':
            cell['technical_invalid'] += 1; cell['unresolved'] += 1; continue
        status = result.get('monitor_status')
        if status == 'alarm': cell['alarm'] += 1
        elif status == 'no_alarm': cell['no_alarm'] += 1
        elif status == 'support_unavailable': cell['support_unavailable'] += 1
        if result.get('decision') == 'model_involvement': cell['model_involvement'] += 1
        else: cell['unresolved'] += 1
    if any(v['assigned'] != 128 for v in out.values()):
        raise ValueError('fixed family denominator changed')
    return out


def _patch_driver():
    """Load the reviewed worker in memory and generalize only its rank count."""
    module = types.ModuleType('frontier16_base_worker')
    module.__file__ = str(BASE_DRIVER)
    sys.modules[module.__name__] = module
    source = BASE_DRIVER.read_text()
    token = "len(ranks)==2 and all(p<=.05 for p in ranks.values())"
    init_limit = 'conditional_initializations=256'
    if source.count(token) != 1 or source.count(init_limit) != 1:
        raise RuntimeError('base worker rank or initialization-budget seam changed')
    source = source.replace(token, "model_claim(ranks, len(alloc['source_candidates']))")
    source = source.replace(init_limit, 'conditional_initializations=640')
    source = patch_worker_source(source)
    module.load_corrected_stage1 = load_corrected_stage1
    module.model_claim = model_claim
    exec(compile(source, str(BASE_DRIVER), 'exec'), module.__dict__)
    d = module
    d.FAMILIES = FAMILIES
    d.plan_rows = plan_rows
    d.validate_plan = validate_plan
    d.allocation_for = allocation_for
    d.TOTAL_OUTPUT = TOTAL_OUTPUT
    d.RESERVE = RESERVE
    d.META_RESERVE = META_RESERVE
    d.SINGLE_WORKER_CAP = SINGLE_WORKER_CAP
    d.TOTAL_SECONDS = TOTAL_SECONDS
    d.base_signed_sum = lambda receipt: receipt.get('A') if receipt.get('C_A') is True else None
    return d


def _install_science_adapters(d):
    base_load = d._load
    def adapted_load(name, path):
        module = base_load(name, path)
        if name == 'paired16_science_adapter':
            module.CELLS = list(NO_MODEL)
            module.SOURCES.update({n: 'LATENT_F' for n in ('CL_Q025', 'CL_Q050', 'CL_Q100')})
            return module
        if name != 'paired16_expanded':
            return module
        frontier = load('frontier16_latent_catalogue', ADAPTER)
        module.filter_public_history = frontier.filter_public_history
        original_factory = module.science_class
        def science_class(base):
            ParentScience = original_factory(base)
            class FrontierScience(ParentScience):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    # Bind to the already-instrumented host class. This keeps
                    # constructor/step accounting inside the worker budget.
                    frontier.bind_forward_module(self.forward)
                    one_step_dir = TRACK/'paired_source_one_step_attempt001'
                    self.future = load('frontier16_one_step_future', one_step_dir/'one_step_future.py')
                    one_step_driver = load('frontier16_one_step_driver_helpers', one_step_dir/'evaluate16_one_step.py')
                    self.branch = types.SimpleNamespace(compact=one_step_driver.compact_one_step)
                def monitor(self, proposal):
                    candidate = proposal['candidate']
                    if candidate not in {'CL_Q025', 'CL_Q050', 'CL_Q100'}:
                        return super().monitor(proposal)
                    original_dgp = self.dgp
                    self.dgp = frontier.prefix_dgp_factory(original_dgp, candidate)
                    try:
                        return super().monitor(proposal)
                    finally:
                        self.dgp = original_dgp
                def calibration(self, public, candidate, replicate_id, proposal_id):
                    means, covariances = public['posteriors'][candidate]
                    self.sampler_calls_attempted += 1
                    sample = frontier.draw_replica_inputs(
                        means=means, covariances=covariances,
                        campaign_entropy=self.config['conditional_future_campaign_entropy'],
                        history_index=0, candidate=candidate, replicate_id=replicate_id)
                    self.sampler_calls_completed += 1
                    self.normal_variates += 1888
                    draws = sample['host_inputs']
                    self.write_npz(self.out/'cal_draws'/f'{candidate}_{replicate_id:03d}.npz', draws)
                    self.write_json(self.out/'cal_draws'/f'{candidate}_{replicate_id:03d}.json', sample['stream_metadata'])
                    panel = self.restore.clone_boundary(public['panel'])
                    if type(panel.learner)._q is not self.b.BellmanLearner._q:
                        raise RuntimeError('CAL clone lost direct-Bellman learner')
                    env = frontier.make_forward_host(candidate, x40=panel.physical[:, 0],
                        w40=panel.physical[:, 1], previous_action39=panel.physical[:, 2], **draws)
                    spec = dict(seed=proposal_id, cell=candidate, H=True,
                                replicate_id=replicate_id, role='CAL')
                    return self.future.episode(spec, panel, env,
                        self.out/'CAL'/candidate/f'proposal_{replicate_id:03d}',
                        self.fixed, self.b, self.dispatch, self.measurements)
            return FrontierScience
        module.science_class = science_class
        return module
    d._load = adapted_load
    return d


def build_worker():
    d = _install_science_adapters(_patch_driver())
    return d


def worker(index, out, output_allowance, time_allowance):
    d = build_worker()
    return d.worker(plan_rows()[index], Path(out), output_allowance, time_allowance)


def used_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def enough_resources(free, remaining_output, remaining_time):
    return (remaining_output > META_RESERVE + 16 * 1024**2 and
            free >= remaining_output + RESERVE and remaining_time > 0)


def run_batch(execution, *, execute=False, resume=False):
    rows = plan_rows(); validate_plan(rows); execution = Path(execution)
    if not execute:
        return dict(status='design_only', assigned=rows, per_family_denominator=128,
                    conditional_candidates=list(NO_MODEL), branches=['scalar','serial','paired'],
                    route='H10 direct Bellman; paired source window date40 only; monitor union unchanged; no next-alarm search')
    previous = None
    if execution.exists():
        if not resume: raise FileExistsError(execution)
        previous = json.loads((execution/'SUMMARY.json').read_text())
        stored_plan = json.loads((execution/'PLAN.json').read_text())
        if len(stored_plan.get('assigned', [])) != 512 or stored_plan.get('per_family_denominator') != 128:
            raise RuntimeError('resume plan does not match fixed 512-slot assignment')
        if previous.get('status') == 'complete': raise RuntimeError('campaign is already complete')
    else:
        if resume: raise FileNotFoundError('cannot resume without the original execution directory')
        execution.mkdir(parents=True)
    free = shutil.disk_usage(execution).free
    initial_need = min(FIRST_BLOCK_OUTPUT, TOTAL_OUTPUT)
    if not resume and not enough_resources(free, initial_need, TOTAL_SECONDS):
        rec = dict(status='not_run', reason='preflight requires first-block output plus shared reserve and positive wall budget',
                   free_bytes=free, assigned=512, per_family_denominator=128, not_run_ordinals=list(range(512)))
        (execution/'NOT_RUN.json').write_text(json.dumps(rec, indent=2)+'\n'); return rec
    plan = dict(status='predeclared', assigned=rows, per_family_denominator=128,
        conditional_candidates=list(NO_MODEL), bank_rule='39 qualified within 128; support refusals retained; no replacement',
        source_alpha=.05, branches=['scalar','serial','paired'],
        route='H10 direct Bellman; paired source window date40 only; all five null ranks must reject',
        total_output_bytes=TOTAL_OUTPUT, first_block_output_bytes=FIRST_BLOCK_OUTPUT,
        first_block_histories=FIRST_BLOCK_HISTORIES, worker_output_cap_bytes=SINGLE_WORKER_CAP,
        reserve_bytes=RESERVE, total_seconds=TOTAL_SECONDS,
        conditional_initialization_limit=640)
    if not resume:
        (execution/'PLAN.json').write_text(json.dumps(plan, indent=2)+'\n')
    else:
        prior_results = previous.get('results', {})
        results = {int(k): v for k, v in prior_results.items()
                   if v.get('execution_status') in ('complete', 'invalid')}
        if any(v.get('execution_status') == 'invalid' for v in results.values()):
            raise RuntimeError('prior assigned invalid slot requires investigation; do not continue')
        base_elapsed = float(previous.get('elapsed_seconds', 0.0))
        prior_resumes = list(execution.glob('RESUME_*.json'))
        resume_path = execution/f'RESUME_{len(prior_resumes)+1:03d}.json'
        (resume_path).write_text(json.dumps(dict(
            status='started', fixed_plan='PLAN.json', prior_complete=len(results),
            first_free_bytes=free, additional_reserve_bytes=2*1024**3,
            campaign_cap_bytes=TOTAL_OUTPUT, first_block_cap_bytes=FIRST_BLOCK_OUTPUT,
            worker_cap_bytes=SINGLE_WORKER_CAP, prior_elapsed_seconds=base_elapsed), indent=2)+'\n')
    if previous is None:
        results = {}; base_elapsed = 0.0; resume_path = None
    started = time.monotonic()
    for row in rows:
        if row['ordinal'] in results: continue
        elapsed_total = base_elapsed + (time.monotonic()-started)
        remain_time = int(TOTAL_SECONDS - elapsed_total)
        remain_output = TOTAL_OUTPUT - used_bytes(execution) - META_RESERVE
        free = shutil.disk_usage(execution).free
        if row['ordinal'] < FIRST_BLOCK_HISTORIES and elapsed_total >= 3*60*60:
            for rest in rows[row['ordinal']:]: results[rest['ordinal']] = dict(status='not_run')
            break
        if remain_time <= 0 or remain_output <= META_RESERVE + 16*1024**2:
            for rest in rows[row['ordinal']:]: results[rest['ordinal']] = dict(status='not_run')
            break
        trial = execution/f'history_{row["ordinal"]:02d}'
        if trial.exists(): raise RuntimeError(f'planned slot already has a directory but no terminal result: {row["ordinal"]}')
        if row['ordinal'] < FIRST_BLOCK_HISTORIES:
            first_used = sum(used_bytes(execution/f'history_{i:02d}') for i in range(FIRST_BLOCK_HISTORIES)
                             if (execution/f'history_{i:02d}').exists())
            block_remaining = FIRST_BLOCK_OUTPUT - first_used - META_RESERVE
        else:
            block_remaining = remain_output
        cap = min(SINGLE_WORKER_CAP, max(0, remain_output), max(0, block_remaining))
        if cap <= META_RESERVE or free < cap + 2*1024**3:
            results[row['ordinal']] = dict(status='not_run')
            for rest in rows[row['ordinal']+1:]: results[rest['ordinal']] = dict(status='not_run')
            break
        trial.mkdir()
        (trial/'ALLOCATION.json').write_text(json.dumps(allocation_for(row), indent=2)+'\n')
        args = [sys.executable, str(Path(__file__).resolve()), '--worker', str(row['ordinal']),
                '--out', str(trial/'execution001'), '--output-allowance', str(cap),
                '--time-allowance', str(max(1, remain_time-5))]
        try:
            proc = subprocess.run(args, timeout=remain_time, check=False, stdout=subprocess.DEVNULL)
            path = trial/'execution001/RESULT.json'
            result = json.loads(path.read_text()) if path.exists() else dict(execution_status='invalid', failure='worker produced no terminal receipt')
            results[row['ordinal']] = result
            if proc.returncode or result.get('execution_status') != 'complete':
                for rest in rows[row['ordinal']+1:]: results[rest['ordinal']] = dict(status='not_run')
                break
        except subprocess.TimeoutExpired:
            results[row['ordinal']] = dict(execution_status='invalid', failure='global time limit expired')
            for rest in rows[row['ordinal']+1:]: results[rest['ordinal']] = dict(status='not_run')
            break
        (execution/'PROGRESS.json').write_text(json.dumps(dict(status='running', completed=len(results),
            by_family=summarize(rows, results)), indent=2)+'\n')
    for row in rows:
        results.setdefault(row['ordinal'], dict(status='not_run'))
    final = dict(status='complete' if len(results)==512 and all(r.get('execution_status')=='complete' for r in results.values()) else 'incomplete',
        assigned=rows, results=results, by_family=summarize(rows, results),
        elapsed_seconds=base_elapsed+time.monotonic()-started, per_family_denominator=128, no_replacement=True)
    if resume:
        rlog=json.loads(resume_path.read_text())
        rlog.update(status='stopped_or_finished', final_free_bytes=shutil.disk_usage(execution).free,
                    completed_new=sum(v.get('execution_status')=='complete' for k,v in results.items() if k not in {int(x) for x in previous.get('results',{})}),
                    final_status=final['status'])
        resume_path.write_text(json.dumps(rlog,indent=2)+'\n')
    (execution/'SUMMARY.json').write_text(json.dumps(final, indent=2)+'\n')
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', help='explicitly run the preassigned histories')
    parser.add_argument('--resume', action='store_true', help='continue unrun slots in the existing fixed assignment')
    parser.add_argument('--out', type=Path, default=HERE/'execution001')
    parser.add_argument('--worker', type=int, choices=range(512))
    parser.add_argument('--output-allowance', type=int, default=0)
    parser.add_argument('--time-allowance', type=int, default=0)
    args = parser.parse_args()
    if args.worker is not None:
        result = worker(args.worker, args.out, args.output_allowance, args.time_allowance)
    else:
        result = run_batch(args.out, execute=args.execute, resume=args.resume)
    print(json.dumps({k:v for k,v in result.items() if k not in ('assigned','results')}, indent=2))
    return result


if __name__ == '__main__':
    main()
