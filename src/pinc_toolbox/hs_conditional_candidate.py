"""HS conditional paths scored by the existing residual-energy implementation.

The caller supplies fresh complete-law conditional CAL; the adapter checks the
common start but cannot verify independence or simulator provenance from arrays.
"""
import numpy as np
from .conditional_candidate_rank import conditional_candidate_rank
from .residual_candidate_source import score_residual_candidate


def score_hs_conditional_candidate(*, evaluation_path, calibration_paths,
        reference_coefficients, candidate_F, candidate_d, candidate_covariance,
        residual_left_inverse, support, eta, alpha, context):
    """One candidate, one terminal source report, fixed planned CAL list.

    Coordinates are actionless growth (DeltaY,Z). None CAL entries retain their
    planned slots. Observed or CAL support exits use conservative noncompletion.
    Full covariance is retained as simulator metadata, not used as a likelihood.
    """
    f=np.asarray(candidate_F,dtype=float); d=np.asarray(candidate_d,dtype=float)
    cov=np.asarray(candidate_covariance,dtype=float)
    if f.ndim!=3 or f.shape[1:]!=(2,2) or len(f)<1 or d.shape!=(len(f),2):
        raise ValueError('candidate mean schedule must have shapes (H,2,2) and (H,2)')
    if cov.shape!=(len(f),2,2) or not np.isfinite(cov).all():
        raise ValueError('candidate covariance must cover the full horizon')
    if not np.allclose(cov,np.swapaxes(cov,1,2)) or np.min(np.linalg.eigvalsh(cov)) < -1e-12:
        raise ValueError('candidate covariance must be symmetric positive semidefinite')
    if not isinstance(calibration_paths,(list,tuple)) or not calibration_paths:
        raise ValueError('supply every planned CAL path including missing entries')
    H=len(f); B=len(calibration_paths); expected=(H+1,2)
    obs=np.asarray(evaluation_path,dtype=float)
    scores=[None]*B; complete=[False]*B; raw={}; error=None
    counts=[None]*B; exits=[None]*B; reasons=['missing_or_invalid_path']*B
    observed_count=None; observed_exit=None
    observed_score=None; observed_complete=False
    if obs.shape==expected and np.isfinite(obs).all():
        valid=[]; positions=[]
        for i,path in enumerate(calibration_paths):
            if path is None:continue
            array=np.asarray(path,dtype=float)
            if array.shape!=expected or not np.isfinite(array).all():continue
            if not np.array_equal(array[0],obs[0]):
                raise ValueError('every CAL must start at the actual observed source-start state')
            valid.append(array); positions.append(i)
        # A placeholder only obtains the observed score if all planned CAL failed;
        # it never enters the planned rank denominator as a completed draw.
        cal=np.asarray(valid) if valid else obs[None,:,:]
        try:
            raw=score_residual_candidate(obs[None,:,:],cal,
                reference_coefficients=reference_coefficients,candidate_F=f,candidate_d=d,
                monitor_start=0,horizon=H,residual_left_inverse=residual_left_inverse,
                support=support,etas=[eta],weight_index=1,alpha=alpha,return_trace=False)
            observed_score=float(raw['evaluation_scores'][0,0])
            observed_count=int(raw['evaluation_valid_count'][0,0])
            observed_exit=int(raw['evaluation_first_exit'][0,0])
            observed_complete=bool(observed_count==H)
            for j,i in enumerate(positions):
                scores[i]=float(raw['calibration_scores'][j,0])
                counts[i]=int(raw['arrays']['calibration_valid_count'][j,0])
                exits[i]=int(raw['arrays']['calibration_first_exit'][j,0])
                complete[i]=bool(counts[i]==H)
                reasons[i]=None if complete[i] else 'support_stop'
        except Exception as exc:
            error={'type':type(exc).__name__,'message':str(exc)}
            observed_complete=False
    out=conditional_candidate_rank(observed_score=observed_score,
        observed_complete=observed_complete,calibration_scores=scores,
        calibration_complete=complete,alpha=alpha)
    out.update(context=context,statistic='sequential_slope_energy',
        coordinates=['DeltaY','Z'],candidate_F=f.tolist(),candidate_d=d.tolist(),
        candidate_covariance=cov.tolist(),
        source_start_state=obs[0].tolist() if obs.ndim==2 and len(obs) else None,
        calibration_scores=[x if x is None or np.isfinite(x) else None for x in scores],
        calibration_complete=complete,calibration_valid_count=counts,
        calibration_first_exit=exits,calibration_failure_reason=reasons,
        observed_valid_count=observed_count,observed_first_exit=observed_exit,
        horizon=H,eta=eta,implementation_error=error)
    return out
