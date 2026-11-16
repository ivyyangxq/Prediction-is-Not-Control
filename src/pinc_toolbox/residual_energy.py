"""Existing sequential slope-energy residual statistic, shared by adapters.

This extracts the statistic in HS2022 run_confirmation.slope_energy_scores
and eta_boundary.score_paths_explicit. It is not a likelihood ratio, source
certificate, or Q bound. Inputs are already released prediction errors and
predictable weights/eligibility; no estimator or economic model is fitted here.
"""
from __future__ import annotations
import math
import numpy as np


def sequential_slope_energy(residuals, weights, *, residual_left_inverse,
                            eligible=None, denominator_floor=1e-15,
                            return_trace=False):
    """Freeze each path/eligibility rule permanently before its first exit.

    residuals: (paths, released transitions, response coordinates).
    weights: (paths, transitions), known at each prediction time.
    eligible: (paths, transitions, rules), determined from predictors only.
    Returned first_exit is zero-based, or transitions if there is no exit.
    Scores start at -inf when no eligible transition was ever released.
    """
    errors=np.asarray(residuals,dtype=float); w=np.asarray(weights,dtype=float)
    inv=np.asarray(residual_left_inverse,dtype=float)
    if errors.ndim!=3 or w.shape!=errors.shape[:2] or inv.shape!=(errors.shape[2],errors.shape[2]):
        raise ValueError('residuals, weights, and whitening dimensions disagree')
    if not np.isfinite(errors).all() or not np.isfinite(w).all() or not np.isfinite(inv).all():
        raise ValueError('residuals, weights, and whitening must be finite')
    floor=float(denominator_floor)
    if not np.isfinite(floor) or floor<0: raise ValueError('denominator_floor must be finite and nonnegative')
    n,steps,width=errors.shape
    if eligible is None: supported=np.ones((n,steps,1),dtype=bool)
    else:
        supported=np.asarray(eligible)
        if supported.dtype!=np.bool_ or supported.ndim!=3 or supported.shape[:2]!=(n,steps) or supported.shape[2]<1:
            raise ValueError('eligible must be a boolean paths x transitions x rules array')
    rules=supported.shape[2]
    scores=np.full((n,rules),-np.inf); terminal=np.full((n,rules),-np.inf)
    count=np.zeros((n,rules),dtype=np.int64); first=np.full((n,rules),steps,dtype=np.int64)
    if not isinstance(return_trace, (bool, np.bool_)):
        raise ValueError('return_trace must be boolean')
    score_trace = np.full((n, steps, rules), -np.inf) if return_trace else None
    eligible_trace = np.zeros((n, steps, rules), dtype=bool) if return_trace else None
    # A separate accumulator per declared rule exactly follows the old explicit
    # freezing implementation. It never consumes post-exit residuals.
    for rule in range(rules):
        active=np.ones(n,dtype=bool); numer=np.zeros((n,width)); denom=np.zeros(n)
        for t in range(steps):
            exits=active & ~supported[:,t,rule];first[exits,rule]=t;active[exits]=False
            count[active,rule]+=1
            r=errors[active,t] @ inv.T;z=w[active,t]
            numer[active]+=z[:,None]*r;denom[active]+=z*z
            value=np.divide((numer[active]**2).sum(axis=1),denom[active],out=np.zeros(active.sum()),where=denom[active]>floor)
            if not np.isfinite(value).all(): raise ValueError('nonfinite accumulated residual energy')
            terminal[active,rule]=value;scores[active,rule]=np.maximum(scores[active,rule],value)
            if return_trace:
                # An unsupported current release freezes the prior prefix max;
                # it never consumes that release's innovation.
                score_trace[:, t, rule] = scores[:, rule]
                eligible_trace[:, t, rule] = active
    out = {'scores':scores,'terminal_energy':terminal,'valid_count':count,'first_exit':first,
           'statistic':'sequential_slope_energy','method_module':__name__,
           'source_certification':'not_supplied','Q':'not_applicable'}
    if return_trace:
        out['score_trace'] = score_trace
        out['eligible_trace'] = eligible_trace
    return out


def sequential_slope_energy_compact(residuals, weights, *, residual_left_inverse,
                                    eligible=None, denominator_floor=1e-15,
                                    cutoffs=None):
    """Use one chronological accumulator for rules differing only by first exit.

    Each rule consumes the same initial prefix of a path. The shared accumulator
    advances while any rule remains active; each rule freezes its own maximum,
    terminal energy and count before its first unsupported release. No time
    trace is retained. Optional fixed cutoffs produce first strict-crossing
    indices (``steps`` means never), plus the score at that crossing.

    This is an alternative implementation of ``sequential_slope_energy``;
    callers still supply the same residuals, weights, whitening and eligibility.
    It does not represent missing or failed candidate evidence.
    """
    errors=np.asarray(residuals,dtype=float); w=np.asarray(weights,dtype=float)
    inv=np.asarray(residual_left_inverse,dtype=float)
    if errors.ndim!=3 or w.shape!=errors.shape[:2] or inv.shape!=(errors.shape[2],errors.shape[2]):
        raise ValueError('residuals, weights, and whitening dimensions disagree')
    if not np.isfinite(errors).all() or not np.isfinite(w).all() or not np.isfinite(inv).all():
        raise ValueError('residuals, weights, and whitening must be finite')
    floor=float(denominator_floor)
    if not np.isfinite(floor) or floor<0: raise ValueError('denominator_floor must be finite and nonnegative')
    n,steps,width=errors.shape
    if eligible is None: supported=np.ones((n,steps,1),dtype=bool)
    else:
        supported=np.asarray(eligible)
        if supported.dtype!=np.bool_ or supported.ndim!=3 or supported.shape[:2]!=(n,steps) or supported.shape[2]<1:
            raise ValueError('eligible must be a boolean paths x transitions x rules array')
    rules=supported.shape[2]
    thresholds=None if cutoffs is None else np.asarray(cutoffs,dtype=float)
    if thresholds is not None and (thresholds.shape!=(rules,) or np.isnan(thresholds).any()):
        raise ValueError('cutoffs must have one non-NaN value per rule')
    scores=np.full((n,rules),-np.inf); terminal=np.full((n,rules),-np.inf)
    count=np.zeros((n,rules),dtype=np.int64); first=np.full((n,rules),steps,dtype=np.int64)
    active=np.ones((n,rules),dtype=bool); numer=np.zeros((n,width)); denom=np.zeros(n)
    crossing=np.full((n,rules),steps,dtype=np.int64)
    crossing_score=np.full((n,rules),-np.inf)
    for t in range(steps):
        exits=active & ~supported[:,t,:]; first[exits]=t; active[exits]=False
        count+=active
        live=active.any(axis=1)
        if not live.any(): break
        r=errors[live,t] @ inv.T; z=w[live,t]
        numer[live]+=z[:,None]*r; denom[live]+=z*z
        value=np.divide((numer[live]**2).sum(axis=1),denom[live],out=np.zeros(live.sum()),where=denom[live]>floor)
        if not np.isfinite(value).all(): raise ValueError('nonfinite accumulated residual energy')
        current=np.full(n,-np.inf); current[live]=value
        np.copyto(terminal,current[:,None],where=active)
        np.maximum(scores,np.where(active,current[:,None],-np.inf),out=scores)
        if thresholds is not None:
            hit=active & (crossing==steps) & (scores>thresholds)
            crossing[hit]=t; crossing_score[hit]=scores[hit]
    out={'scores':scores,'terminal_energy':terminal,'valid_count':count,'first_exit':first,
         'statistic':'sequential_slope_energy','method_module':__name__,
         'implementation':'shared_prefix_compact','source_certification':'not_supplied','Q':'not_applicable'}
    if thresholds is not None:
        out.update(first_crossing=crossing,first_crossing_score=crossing_score)
    return out


def sequential_slope_energy_blocked(residuals, weights, *, residual_left_inverse,
                                    eligible=None, denominator_floor=1e-15,
                                    cutoffs=None, block_size=128):
    """Bounded path blocks with chronological NumPy prefix accumulations.

    All rules use the same residual stream and differ only in their permanent
    first exit. Residuals and weights after the last active rule are zeroed
    before whitening or products. Prefix sums run along chronological time;
    stopped maxima and terminal energies are gathered at each rule's last
    consumed release. This alternative is checked against the explicit and
    shared-loop implementations; bitwise equivalence is not presumed.
    """
    errors=np.asarray(residuals,dtype=float); w=np.asarray(weights,dtype=float)
    inv=np.asarray(residual_left_inverse,dtype=float)
    if errors.ndim!=3 or w.shape!=errors.shape[:2] or inv.shape!=(errors.shape[2],errors.shape[2]):
        raise ValueError('residuals, weights, and whitening dimensions disagree')
    if not np.isfinite(errors).all() or not np.isfinite(w).all() or not np.isfinite(inv).all():
        raise ValueError('residuals, weights, and whitening must be finite')
    floor=float(denominator_floor)
    if not np.isfinite(floor) or floor<0: raise ValueError('denominator_floor must be finite and nonnegative')
    if isinstance(block_size,bool) or not isinstance(block_size,(int,np.integer)) or block_size<1:
        raise ValueError('block_size must be a positive integer')
    n,steps,width=errors.shape
    if eligible is None: supported=np.ones((n,steps,1),dtype=bool)
    else:
        supported=np.asarray(eligible)
        if supported.dtype!=np.bool_ or supported.ndim!=3 or supported.shape[:2]!=(n,steps) or supported.shape[2]<1:
            raise ValueError('eligible must be a boolean paths x transitions x rules array')
    rules=supported.shape[2]
    thresholds=None if cutoffs is None else np.asarray(cutoffs,dtype=float)
    if thresholds is not None and (thresholds.shape!=(rules,) or np.isnan(thresholds).any()):
        raise ValueError('cutoffs must have one non-NaN value per rule')
    scores=np.full((n,rules),-np.inf); terminal=np.full((n,rules),-np.inf)
    first=np.full((n,rules),steps,dtype=np.int64)
    crossing=np.full((n,rules),steps,dtype=np.int64); crossing_score=np.full((n,rules),-np.inf)
    if steps:
        times=np.arange(steps)[None,:]
        for begin in range(0,n,int(block_size)):
            end=min(begin+int(block_size),n); size=end-begin
            unsupported=~supported[begin:end]
            exits=np.where(unsupported.any(axis=1),unsupported.argmax(axis=1),steps)
            first[begin:end]=exits
            consumed=times<exits.max(axis=1)[:,None]
            # Mask before multiplication, including whitening: unused finite
            # extreme values must not overflow a computation the rule stops.
            masked_errors=np.where(consumed[:,:,None],errors[begin:end],0.0)
            masked_weights=np.where(consumed,w[begin:end],0.0)
            whitened=(masked_errors.reshape(-1,width) @ inv.T).reshape(size,steps,width)
            numerator=np.cumsum(masked_weights[:,:,None]*whitened,axis=1)
            denominator=np.cumsum(masked_weights*masked_weights,axis=1)
            energy=np.divide((numerator**2).sum(axis=2),denominator,
                out=np.zeros((size,steps)),where=denominator>floor)
            if not np.isfinite(energy).all(): raise ValueError('nonfinite accumulated residual energy')
            maxima=np.maximum.accumulate(energy,axis=1)
            rows=np.arange(size)[:,None];last=np.maximum(exits-1,0)
            scores[begin:end]=np.where(exits>0,maxima[rows,last],-np.inf)
            terminal[begin:end]=np.where(exits>0,energy[rows,last],-np.inf)
            if thresholds is not None:
                for rule,cut in enumerate(thresholds):
                    crossed=energy>cut
                    first_hit=crossed.argmax(axis=1)
                    valid=crossed.any(axis=1)&(first_hit<exits[:,rule])
                    crossing[begin:end,rule]=np.where(valid,first_hit,steps)
                    crossing_score[begin:end,rule]=np.where(valid,energy[np.arange(size),first_hit],-np.inf)
    out={'scores':scores,'terminal_energy':terminal,'valid_count':first.copy(),'first_exit':first,
         'statistic':'sequential_slope_energy','method_module':__name__,
         'implementation':'numpy_prefix_blocks','source_certification':'not_supplied','Q':'not_applicable'}
    if thresholds is not None:out.update(first_crossing=crossing,first_crossing_score=crossing_score)
    return out


def prepare_slope_energy_prefixes(weights, eligible, *, denominator_floor=1e-15):
    """Prepare candidate-independent support stops and chronological denominators.

    Reuse only with the same actual predictor paths, weights and support rules.
    Different candidate CAL paths require their own preparation.
    """
    w=np.asarray(weights,dtype=float); supported=np.asarray(eligible)
    if w.ndim!=2 or not np.isfinite(w).all(): raise ValueError('weights must be a finite paths x time matrix')
    if supported.dtype!=np.bool_ or supported.ndim!=3 or supported.shape[:2]!=w.shape or supported.shape[2]<1:
        raise ValueError('eligible must be a boolean paths x time x rules array')
    floor=float(denominator_floor)
    if not np.isfinite(floor) or floor<0: raise ValueError('denominator_floor must be finite and nonnegative')
    n,steps=w.shape; rules=supported.shape[2]
    if steps:
        unsupported=~supported
        first=np.where(unsupported.any(axis=1),unsupported.argmax(axis=1),steps)
    else:first=np.zeros((n,rules),dtype=np.int64)
    consumed=np.arange(steps)[None,:]<first.max(axis=1)[:,None]
    masked_weights=np.where(consumed,w,0.0)
    denominator=np.cumsum(masked_weights*masked_weights,axis=1)
    return {'weights':masked_weights,'consumed':consumed,'first_exit':first,
            'denominator':denominator,'denominator_floor':floor}


def sequential_slope_energy_prepared(residuals, *, prepared, residual_left_inverse,
                                     cutoffs=None, block_size=128):
    """Score residuals using a matching prepared support/weight object."""
    errors=np.asarray(residuals,dtype=float);inv=np.asarray(residual_left_inverse,dtype=float)
    w=np.asarray(prepared['weights']);first=np.asarray(prepared['first_exit'])
    consumed=np.asarray(prepared['consumed']);denominator=np.asarray(prepared['denominator'])
    floor=float(prepared['denominator_floor'])
    if errors.ndim!=3 or w.shape!=errors.shape[:2] or inv.shape!=(errors.shape[2],errors.shape[2]):
        raise ValueError('residuals, prepared weights, and whitening dimensions disagree')
    if not np.isfinite(errors).all() or not np.isfinite(inv).all(): raise ValueError('residuals and whitening must be finite')
    if consumed.shape!=w.shape or consumed.dtype!=np.bool_ or denominator.shape!=w.shape:
        raise ValueError('invalid prepared time arrays')
    n,steps,width=errors.shape
    if first.ndim!=2 or first.shape[0]!=n or first.shape[1]<1 or first.dtype.kind not in 'iu' or np.any(first<0) or np.any(first>steps):
        raise ValueError('invalid prepared first exits')
    if isinstance(block_size,bool) or not isinstance(block_size,(int,np.integer)) or block_size<1:
        raise ValueError('block_size must be a positive integer')
    rules=first.shape[1];thresholds=None if cutoffs is None else np.asarray(cutoffs,dtype=float)
    if thresholds is not None and (thresholds.shape!=(rules,) or np.isnan(thresholds).any()):
        raise ValueError('cutoffs must have one non-NaN value per rule')
    scores=np.full((n,rules),-np.inf);terminal=np.full((n,rules),-np.inf)
    crossing=np.full((n,rules),steps,dtype=np.int64);crossing_score=np.full((n,rules),-np.inf)
    if steps:
        for begin in range(0,n,int(block_size)):
            end=min(begin+int(block_size),n);size=end-begin;exits=first[begin:end]
            masked_errors=np.where(consumed[begin:end,:,None],errors[begin:end],0.0)
            whitened=(masked_errors.reshape(-1,width) @ inv.T).reshape(size,steps,width)
            numerator=np.cumsum(w[begin:end,:,None]*whitened,axis=1)
            denom=denominator[begin:end]
            energy=np.divide((numerator**2).sum(axis=2),denom,out=np.zeros((size,steps)),where=denom>floor)
            if not np.isfinite(energy).all():raise ValueError('nonfinite accumulated residual energy')
            maxima=np.maximum.accumulate(energy,axis=1);rows=np.arange(size)[:,None];last=np.maximum(exits-1,0)
            scores[begin:end]=np.where(exits>0,maxima[rows,last],-np.inf)
            terminal[begin:end]=np.where(exits>0,energy[rows,last],-np.inf)
            if thresholds is not None:
                for rule,cut in enumerate(thresholds):
                    crossed=energy>cut;first_hit=crossed.argmax(axis=1)
                    valid=crossed.any(axis=1)&(first_hit<exits[:,rule])
                    crossing[begin:end,rule]=np.where(valid,first_hit,steps)
                    crossing_score[begin:end,rule]=np.where(valid,energy[np.arange(size),first_hit],-np.inf)
    out={'scores':scores,'terminal_energy':terminal,'valid_count':first.copy(),'first_exit':first.copy(),
         'statistic':'sequential_slope_energy','method_module':__name__,
         'implementation':'prepared_numpy_prefix_blocks','source_certification':'not_supplied','Q':'not_applicable'}
    if thresholds is not None:out.update(first_crossing=crossing,first_crossing_score=crossing_score)
    return out


def rank_cutoff(scores, *, alpha=0.05):
    """Old held-out calibration rank rule; paths are the fixed denominator."""
    values=np.asarray(scores,dtype=float)
    if values.ndim<1 or len(values)<1 or np.isnan(values).any(): raise ValueError('calibration scores require nonempty paths and no NaN')
    alpha=float(alpha)
    if not np.isfinite(alpha) or not 0<alpha<1: raise ValueError('alpha must lie in (0,1)')
    k=int(math.ceil((len(values)+1)*(1-alpha)))
    return np.full(values.shape[1:],np.inf) if k>len(values) else np.sort(values,axis=0)[k-1]


def predictor_envelope(training_predictors):
    """Raw training-predictor rectangle and the old machine-epsilon floor."""
    x=np.asarray(training_predictors,dtype=float)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all(): raise ValueError('training_predictors must be a finite matrix')
    low=x.min(axis=0);high=x.max(axis=0)
    return low,high,np.maximum(high-low,np.finfo(float).eps)


def envelope_eligibility(predictors, support, etas):
    x=np.asarray(predictors,dtype=float);low,high,width=map(lambda v:np.asarray(v,dtype=float),support)
    eta=np.asarray(etas,dtype=float)
    if x.ndim!=3 or low.shape!=(x.shape[2],) or high.shape!=low.shape or width.shape!=low.shape:
        raise ValueError('predictor and envelope dimensions disagree')
    if not np.isfinite(x).all() or not all(np.isfinite(v).all() for v in (low,high,width)) or np.any(width<=0) or np.any(high<low):
        raise ValueError('invalid predictor envelope')
    if eta.ndim!=1 or not len(eta) or np.isnan(eta).any() or np.any(eta<0): raise ValueError('etas must be nonnegative')
    distance=np.maximum(np.maximum((low-x)/width,(x-high)/width),0).max(axis=-1)
    return distance[:,:,None]<=eta
