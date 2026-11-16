"""Exact Gaussian observation update for a matched finite-DGP simulator.
Not a detector, classifier, likelihood score, or production default.
"""
import numpy as np

def condition_exact(mean,covariance,loading,observation,*,atol=1e-12):
    mean=np.asarray(mean,dtype=float);covariance=np.asarray(covariance,dtype=float);loading=np.asarray(loading,dtype=float)
    if mean.ndim!=1 or loading.shape!=mean.shape or covariance.shape!=(mean.size,mean.size):
        raise ValueError('incompatible Gaussian dimensions')
    if not all(np.isfinite(x).all() for x in (mean,covariance,loading)) or not np.isfinite(observation):
        raise ValueError('nonfinite Gaussian input')
    if not np.isfinite(atol) or atol<=0:raise ValueError('positive finite tolerance required')
    if not np.allclose(covariance,covariance.T,rtol=0,atol=atol) or np.linalg.eigvalsh(covariance).min() < -atol:
        raise ValueError('covariance must be symmetric positive semidefinite')
    cross=covariance@loading;v=float(loading@cross);innovation=float(observation-loading@mean)
    variance_tol=64*np.finfo(float).eps*np.linalg.norm(covariance,ord=2)*float(loading@loading)
    if v<=variance_tol:
        if abs(innovation)>atol:raise ValueError('observation incompatible with degenerate conditional law')
        return mean.copy(),covariance.copy()
    updated=mean+cross*(innovation/v)
    projection=np.eye(mean.size)-np.outer(cross,loading)/v
    cov=projection@covariance@projection.T
    return updated,(cov+cov.T)/2
