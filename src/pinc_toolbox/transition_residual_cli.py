"""Shared fixed-reference G transition/residual monitoring for public arrays.

Data generation and economic truth stay outside this module. The estimator is
pinc_toolbox.g's shared no-action specialization; the statistic is the existing
sequential slope energy. This operation does not supply source certification.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from . import g, residual_energy, released_transition_g


def _integer(value,name,minimum=0):
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _json_safe(value):
    if isinstance(value,np.ndarray): return _json_safe(value.tolist())
    if isinstance(value,np.generic): return _json_safe(value.item())
    if isinstance(value,dict): return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_json_safe(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def run_request(request):
    if request.get('operation')!='transition_residual_monitor': raise ValueError('unsupported operation')
    source=Path(request['data_path']);artifact=Path(request['artifact_output'])
    if not source.is_absolute() or not artifact.is_absolute(): raise ValueError('data and artifact paths must be absolute')
    if artifact.exists(): raise ValueError('artifact_output must be fresh')
    settings=request['settings'];start=_integer(settings['monitor_start'],'monitor_start')
    horizon=_integer(settings['horizon'],'horizon',1);weight_index=_integer(settings['weight_index'],'weight_index')
    fit_as_of=_integer(settings.get('fit_as_of',0),'fit_as_of')
    etas=np.asarray(settings['etas'],dtype=float)
    if etas.ndim!=1 or not len(etas) or not np.isfinite(etas).all() or np.any(etas<0): raise ValueError('etas must be finite and nonnegative')
    return_trace=settings.get('return_trace',False)
    if not isinstance(return_trace,bool): raise ValueError('return_trace must be boolean')
    alpha=float(settings.get('alpha',.05));started=time.perf_counter()
    with np.load(source,allow_pickle=False) as data:
        train=np.array(data['training_states']);target=np.array(data['next_physical'])
        calibration=np.array(data['calibration_full_paths']);evaluation=np.array(data['evaluation_full_paths'])
    if train.ndim!=2 or target.shape!=train.shape or not len(train): raise ValueError('training states and next_physical must be aligned nonempty matrices')
    dimension=train.shape[1]
    if weight_index>=dimension: raise ValueError('weight_index outside public predictors')
    if calibration.ndim!=3 or evaluation.ndim<3 or calibration.shape[-1]!=dimension or evaluation.shape[-1]!=dimension:
        raise ValueError('calibration/evaluation must end with path-time-state axes')
    if calibration.shape[-2]<start+horizon+1 or evaluation.shape[-2]<start+horizon+1: raise ValueError('requested public horizon is not present')
    # Both controlled SimpleGQ and this no-action model call the same fitter.
    model=g.TransitionG(lambda x:np.column_stack((np.ones(len(x)),x)),scaling='standard_deviation')
    model.fit(train,target)
    support=residual_energy.predictor_envelope(train)
    covariance=model.residual_covariance_
    inverse=np.linalg.inv(np.linalg.cholesky(covariance))
    arrays={};results={};times={}
    for label,full in [('calibration',calibration),('evaluation',evaluation)]:
        phase=time.perf_counter();leading=full.shape[:-2]
        paths=full.reshape((-1,full.shape[-2],dimension))[:,start:start+horizon+1]
        predictors=paths[:,:-1];observed=paths[:,1:]
        errors=np.empty_like(observed,dtype=float);predictions=np.empty_like(observed,dtype=float)
        for begin in range(0,len(paths),32):
            end=min(begin+32,len(paths)); shape=predictors[begin:end].shape
            decisions=np.broadcast_to(np.arange(start,start+horizon),shape[:2]).reshape(-1)
            releases=decisions+1
            batch=released_transition_g.prediction_errors(model,predictors[begin:end].reshape(-1,dimension),observed[begin:end].reshape(-1,dimension),decision_times=decisions,release_times=releases,as_of=start+horizon,fit_as_of=fit_as_of)
            errors[begin:end]=np.asarray(batch['prediction_errors'],float).reshape(shape)
            predictions[begin:end]=np.asarray(batch['predictions'],float).reshape(shape)
            del batch
        eligible=residual_energy.envelope_eligibility(predictors,support,etas)
        scored=residual_energy.sequential_slope_energy(errors,predictors[:,:,weight_index],residual_left_inverse=inverse,eligible=eligible,denominator_floor=1e-15,return_trace=return_trace and label=='evaluation')
        for key in ('scores','terminal_energy','valid_count','first_exit'):
            arrays[label+'_'+key]=scored[key].reshape(leading+(len(etas),))
        if return_trace and label=='evaluation':
            for key in ('score_trace','eligible_trace'):
                arrays[label+'_'+key]=scored[key].reshape(leading+(horizon,len(etas)))
        arrays[label+'_predictions']=predictions.reshape(leading+(horizon,dimension))
        arrays[label+'_residuals']=errors.reshape(leading+(horizon,dimension))
        results[label]={'path_shape':list(leading),'planned_paths':int(np.prod(leading)),
                        'released_transitions_per_path':horizon,'prediction_mode':'fixed_reference_batch_replay',
                        'scores':arrays[label+'_scores'],'valid_count':arrays[label+'_valid_count'],'first_exit':arrays[label+'_first_exit']}
        times[label]=time.perf_counter()-phase
    thresholds=residual_energy.rank_cutoff(arrays['calibration_scores'],alpha=alpha)
    arrays['thresholds']=thresholds;arrays['etas']=etas
    arrays['evaluation_alarm']=arrays['evaluation_scores']>thresholds
    if return_trace:
        crossing=arrays['evaluation_score_trace']>thresholds
        arrays['evaluation_alarm_trace']=crossing
        arrays['evaluation_first_alarm']=np.where(crossing.any(axis=-2),crossing.argmax(axis=-2)+start+1,-1)
        arrays['evaluation_first_exit_release']=np.where(arrays['evaluation_first_exit']<horizon,arrays['evaluation_first_exit']+start+1,-1)
        arrays['release_times']=np.arange(start+1,start+horizon+1)
        results['evaluation']['trace_semantics']={'time_axis':'release_times, absolute outcome release', 'first_alarm':'first strict crossing absolute release, -1 if never', 'first_exit_release':'first unscored outcome release, -1 if never; exit rule uses preceding decision predictors','eligible_trace':'permanently active after current predictor support check','alarm_trace':'cumulative strict crossing retained after support exit; no new score consumed after exit'}
    arrays['coefficient']=model._C;arrays['covariance']=covariance
    for key,value in zip(('support_low','support_high','support_width'),support): arrays[key]=value
    arrays['residual_left_inverse']=inverse
    artifact.parent.mkdir(parents=True,exist_ok=True)
    with artifact.open('xb') as f: np.savez_compressed(f,**arrays)
    result={'operation':'transition_residual_monitor','status':'completed','data_path':str(source),'artifact_path':str(artifact),
            'method':'shared_G_transition_residual_sequential_slope_energy','core_modules':{m.__name__:m.__file__ for m in (g,residual_energy,released_transition_g)},
            'package_version':importlib.metadata.version('pinc-toolbox'),
            'fit':{'coefficient':model._C,'F':model._C[1:].T,'d':model._C[0],'covariance':covariance,'diagnostics':model.diagnostics_},
            'settings':settings,'calibration':results['calibration'],'evaluation':results['evaluation'],
            'thresholds':thresholds,'calibration_rank':int(np.ceil((len(calibration)+1)*(1-alpha))),
            'alarm':arrays['evaluation_alarm'],'strict_comparison':'score > cutoff',
            'source':{'status':'unavailable','reason':'same_residual_candidate_images_and_candidate_calibration_not_supplied'},
            'action':'not_applicable','Q':'not_applicable','scope':'ordinary residual alarm; each eta marginal, no simultaneous-across-eta claim',
            'timing':dict(times,total_elapsed_seconds=time.perf_counter()-started),
            'nonfinite_json':'null represents no eligible score (-inf) or unattainable rank cutoff (+inf); exact arrays preserved in NPZ'}
    return _json_safe(result)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists(): parser.error('output must be fresh')
    request=json.loads(args.input.read_text())
    if Path(request.get('artifact_output','')).resolve()==args.output.resolve(): parser.error('artifact and JSON output must differ')
    result=run_request(request);args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'output':str(args.output)}));return 0


if __name__=='__main__':raise SystemExit(main())
