"""JSON/Dynare entry to the same full RND2 synchronous four-path executor.

--host-file is explicitly trusted Python implementing create_environment(config).
Only environment_config is supplied to that factory; it is never given learner
history, detector reports or private labels by this entry point.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import pandas as pd
from .counterfactual import run_four_paths, counterfactual_report
from .rnd2_counterfactual import prepare_rnd2_learner


def report_request(request):
    """Format an existing result without loading a host, fitting or stepping."""
    return counterfactual_report(request['counterfactual_result'],
        stage1=request.get('stage1'), context=request.get('context'),
        existing_report=request.get('existing_report'),
        pair_routing=request.get('pair_routing'))


def run_request(request, environment):
    settings = dict(request['learner'])
    learner, state = prepare_rnd2_learner(pd.DataFrame(request['history']), **settings)
    result = run_four_paths(learner=learner, environment=environment,
        initial_state=state, start_time=settings['as_of'], steps=request['steps'],
        beta=settings.get('beta', .95))
    report = counterfactual_report(result, stage1=request.get('stage1'),
        context=request.get('context'), existing_report=request.get('existing_report'),
        pair_routing=request.get('pair_routing'))
    report['initialization'] = dict(auxiliary='full', provenance='released_prefix_reconstruction',
        auxiliary_order=['delta_gamma','delta_m','residual'],
        qualification='native_local_support_and_finite_Q', eta_status='not_supplied',
        prefix_refit_audits=learner.refit_audits, fitted_as_of=learner.model.as_of_,
        estimator_config=(None if not hasattr(learner.model, 'estimator_config_')
                          else learner.model.estimator_config_.as_dict()),
        G=learner.model.diagnostics_, online_refit_every=learner.refit_every)
    if getattr(learner.model,'g_spec','legacy')=='v012':
        report['initialization'].update(g_spec='v012',auxiliary=learner.model.auxiliary,
            auxiliary_order=list(learner.model.auxiliary_columns_),
            outer_config=learner.model.outer_config_.as_dict())
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--host-file');mode.add_argument('--report-only',action='store_true')
    parser.add_argument('--output',required=True);args=parser.parse_args(argv)
    output=Path(args.output)
    if output.exists():
        parser.error('output already exists')
    request=json.loads(Path(args.input).read_text())
    if args.report_only:
        report=report_request(request)
    else:
        spec=importlib.util.spec_from_file_location('pinc_counterfactual_user_host',Path(args.host_file).resolve())
        if spec is None or spec.loader is None:
            raise ValueError('cannot load explicitly supplied host file')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        environment=module.create_environment(request['environment_config'])
        report=run_request(request,environment)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as file:
        json.dump(report,file,indent=2,allow_nan=False);file.write('\n')


if __name__=='__main__':
    main()
