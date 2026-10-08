"""Run an actual full RA-GQ four-path example using only the installed package.

PYTHONPATH=software/pinc_toolbox/src python rnd2_four_paths.py --output result.json
Optional --request writes the identical JSON request for the CLI/Dynare wrapper.
"""
from pathlib import Path
import argparse
import json
import importlib.util
import numpy as np
from pinc_toolbox.counterfactual_cli import run_request
_spec=importlib.util.spec_from_file_location('pinc_example_environment',Path(__file__).with_name('rnd2_environment.py'))
_host=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_host)
create_environment=_host.create_environment


def make_request():
    rng=np.random.default_rng(20260925);grid=np.linspace(0,.1,11);history=[]
    for path in range(16):
        x=float(rng.normal(.5,.15));w=float(rng.normal(0,.04));prev=float(rng.choice(grid))
        for time in range(-40,0):
            action=float(rng.choice(grid));nxt=.3+.4*x+.2*w+.1*prev+.5*(action-prev)+rng.normal(0,.012)
            nw=.65*w+rng.normal(0,.025);reward=1-.3*x*x+2*action-30*action*action+10*w*action
            history.append(dict(path_id=path,period=time,release_period=time+1,state=x,w=w,
                previous_action=prev,action=action,next_state=float(nxt),next_w=float(nw),reward=float(reward)))
            x,w,prev=float(nxt),float(nw),action
    initial_w=next(r['next_w'] for r in history if r['path_id']==0 and r['period']==-1)
    return dict(history=history,learner=dict(as_of=0,warmup_as_of=-20,window=16,path_id=0,
        refit_every=1,beta=.95,horizon=3,nodes_per_memory=5,noise_nodes=4,
        q_evaluation="bellman",query_horizon=3),steps=4,
        environment_config=dict(initial_w=initial_w,drift=.004,seed=812),
        stage1=dict(status='unavailable',reason='engineering_example_without_detection_calibration'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--request');args=parser.parse_args()
    request=make_request();report=run_request(request,create_environment(request['environment_config']))
    with Path(args.output).open('x') as file:json.dump(report,file,indent=2,allow_nan=False)
    if args.request:
        with Path(args.request).open('x') as file:json.dump(request,file,indent=2,allow_nan=False)
    print(json.dumps(dict(status=report['counterfactual']['status'],values=report['counterfactual']['comparison']['values'])))


if __name__=='__main__':main()
