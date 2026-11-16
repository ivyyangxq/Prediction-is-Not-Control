"""Evaluator-only expected H10 bounds for the registered controlled benchmark.

Reuse the existing LP/PSD supersolution and open-loop lower-value engine. The
known law and focal public+probe posterior are evaluation inputs, never learner
inputs. Binary64 bounds are conditional numerical results, not certified
interval arithmetic or ordinary-panel causal intervals.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .controlled_product import _load, _public_archive


def _law(family):
    return (.8,1.,1.6,-1.,.02) if family in {'CM','MIX'} else (.8,.5,.8,-2.,0.)


def evaluator(root,family):
    root=Path(root)
    old=_load('product_value_'+family,root/'Outer Loop/rnd2_repair/reported_model_repair_20261112/evaluator/score_public_probe_repair_20261113.py')
    c0,cx,ca,caa,intercept=_law(family)
    F=old.FSTAR if family=='CL_Q100' else old.F_MIX if family=='MIX' else old.F0
    def branch(prior,a,latent_F):
        A=np.zeros((4,4));A[0,0]=c0+cx*a;A[0,1:]=old.ETA+a*old.LAM;A[1:,1:]=latent_F
        b=np.r_[intercept+ca*a+caa*a*a,np.zeros(3)]
        T=np.zeros((old.DIM,old.DIM));T[0,0]=1
        T[1:,0]=(A@np.r_[old.CENTER,np.zeros(3)]+b-np.r_[old.CENTER,np.zeros(3)])/old.SCALE;T[1:,1:]=A
        N=np.zeros((old.DIM,old.DIM));N[1:,1:]=old.QNOISE/old.SCALE**2
        B=old.reward_matrix(a)+old.BETA*(T.T@prior@T);B[0,0]+=old.BETA*float(np.sum(prior*N))
        return .5*(B+B.T)
    def sequences(x,m,P,seq,latent_F):
        seq=np.asarray(seq,float);n=len(seq)
        mu=np.zeros((n,4));mu[:,0]=float(x);mu[:,1:]=m
        cov=np.zeros((n,4,4));cov[:,1:,1:]=P
        value=np.zeros(n)
        for j in range(seq.shape[1]):
            a=seq[:,j];mx=mu[:,0];vx=cov[:,0,0]
            value+=old.BETA**j*(mx-.25*(mx*mx+vx)-2.025*a-.5*a*a)
            A=np.zeros((n,4,4));A[:,0,0]=c0+cx*a;A[:,0,1:]=old.ETA[None]+a[:,None]*old.LAM[None];A[:,1:,1:]=latent_F[None]
            b=np.zeros((n,4));b[:,0]=intercept+ca*a+caa*a*a
            mu=(A@mu[...,None])[...,0]+b
            cov=(A@cov)@np.swapaxes(A,1,2)+old.QNOISE
        return value
    old.branch_matrix=branch;old.seqvals=sequences
    return old,F


def build_upper_bank(root,out):
    """Fixed identity objective, chosen before confirmation outcomes are read."""
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    bank={}
    for family in ('N','CL_Q100','CM','MIX'):
        engine,F=evaluator(root,family)
        matrices,diagnostics=engine.fit_upper(F,np.eye(engine.DIM))
        bank[family]={'matrices':[m.tolist() for m in matrices],'diagnostics':diagnostics,
                      'objective':'fixed identity, no outcome-dependent refinement','physical_law':list(_law(family)),'F':F.tolist()}
    out.write_text(json.dumps(bank,indent=2,allow_nan=False)+'\n')
    return bank


def posterior_timeline(root,history,archive,family):
    directory=Path(root)/'experiments/ecma_four_tracks_20260924/track2_errors/paired_monitor_conditioning_attempt001'
    exact=_load('exact_update',directory/'exact_update.py')
    # The paired filter admits PSD covariances after an exact public slope.
    # The older MAIN-only filter requires positive definiteness and correctly
    # refuses that singular intermediate representation.
    post=_load('product_known_law_posterior',directory/'public_posterior_with_pairs.py')
    raw=_public_archive(archive)
    probes=json.loads(raw.get('PAIRED_PROBE_ENDPOINTS.json',b'[]'))
    groups={}
    for p in probes:
        if p.get('path')==0 and p.get('status')=='complete' and p.get('release_time')==p['time']+1:
            groups.setdefault(p['time'],[]).append(p)
    focal=history.loc[history.path_id==0].sort_values('period').set_index('period')
    m=np.zeros(3);P=post.finite_burn_covariance()
    m,P=post._scalar_update(m,P,post.H,float(focal.loc[0].w))
    output={}
    for time,row in focal.iterrows():
        time=int(time);output[time]=(m.copy(),P.copy())
        coefficients=_law(family) if time>=32 else _law('N')
        c0,cx,ca,caa,intercept=coefficients
        points=groups.get(time,[])
        if len(points)>=2:
            low,high=sorted(points,key=lambda p:p['action'])[::len(points)-1]
            if high['action']>low['action']:
                if abs(low['state_x']-row.state)>1e-12 or abs(high['state_x']-row.state)>1e-12:
                    raise ValueError('public probe starting state mismatch')
                value=(high['next_x']-low['next_x'])/(high['action']-low['action'])-(ca+caa*(high['action']+low['action'])+cx*row.state)
                m,P=exact.condition_exact(m,P,post.LAMBDA,float(value))
        offset=(c0+cx*row.action)*row.state+ca*row.action+caa*row.action**2+intercept
        m,P=post._scalar_update(m,P,post.ETA+row.action*post.LAMBDA,float(row.next_state-offset))
        F=(post.FSTAR if family=='CL_Q100' else post.F0+.69375*(post.FSTAR-post.F0)) if time>=32 and family in {'CL_Q100','MIX'} else post.F0
        m=F@m;P=F@P@F.T+post.Q
        m,P=post._scalar_update(m,.5*(P+P.T),post.H,float(row.next_w))
    return output


def score_recommendations(root,history,archive,family,reports,bank):
    """One public posterior and matched value target for both information arms."""
    engine,F=evaluator(root,family);posterior=posterior_timeline(root,history,archive,family)
    matrices=np.asarray(bank[family]['matrices'])
    pre_engine,pre_F=evaluator(root,'N');pre_matrices=np.asarray(bank['N']['matrices'])
    focal=history.loc[history.path_id==0].set_index('period')
    cache={};scored={}
    for arm,report in reports.items():
        rows=[]
        for entry in report['timeline']:
            time=entry['date'];action=entry['recommended_action']
            if action is None:
                rows.append({'date':time,'status':'recommendation_unavailable','loss_interval':None});continue
            m,P=posterior[time];x=float(focal.loc[time].state)
            if time not in cache:
                current,current_F,current_matrices=(pre_engine,pre_F,pre_matrices) if time<32 else (engine,F,matrices)
                low,_,passes=current.lower_values(x,m,P,current_F)
                K=current.public_K(x,m,P)
                up=np.array([float(np.trace(B@K)) for B in current_matrices])
                if min(up-low)<-1e-7:raise FloatingPointError('upper below feasible lower')
                cache[time]=(low,up,passes)
            low,up,passes=cache[time];index=int(np.argmin(abs(engine.GRID-action)))
            competitors=np.arange(len(engine.GRID))!=index
            lower=max(0.,float(np.max(low[competitors])-up[index]))
            upper=max(0.,float(np.max(up[competitors])-low[index]))
            rows.append({'date':time,'status':'scored_numerical_bounds','recommended_action':action,
                'loss_interval':[lower,upper],'upper_below_005':upper<=.005,'lower_above_005':lower>.005,
                'posterior_mean':m.tolist(),'posterior_covariance':P.tolist(),
                'value_target':'persistent pre-switch law H10 (initial competence target)' if time<32 else 'persistent current post-switch law H10',
                'lower_values':low.tolist(),'upper_values':up.tolist(),'coordinate_passes':passes})
        maintained=[]
        for start in range(max(0,len(rows)-11)):
            block=rows[start:start+12]
            if len(block)==12 and block[0]['date']>=40 and all(r.get('upper_below_005') for r in block):maintained.append(block[0]['date'])
        scored[arm]={'rows':rows,'first_maintained_12_date':min(maintained) if maintained else None,
            'right_censored':not bool(maintained),'censor_date':rows[-1]['date'] if rows else None}
    return {'method':'existing open-loop feasible lower / global quadratic Bellman supersolution upper, generalized to declared N/CL and CM/MIX coefficients',
        'scope':'evaluator only; true known law; focal public MAIN+historical reset probes; companion reduction requires stated independence/nonanticipation premises',
        'numerical_scope':'binary64 LP/eigenvalue repair; exact-probe posterior uses point endpoints, not formal interval certificates',
        'not_learner_information':['true family','known transition coefficients','posterior latent distribution','upper bank'],
        'target':'same observed state and public focal+probe posterior, expected H10, zero terminal, grid actions',
        'arms':scored}
