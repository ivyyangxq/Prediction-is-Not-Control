"""True-law expected H10 score for executed actions in reported-model repair.

Input is a JSON object with four fields per decision row: family (CM/MIX),
arm, date, actual_x, executed_action, posterior_mean, posterior_covariance.
Posterior fields must be computed from the legal public transcript only. This
module contains no private-state or future-tape interface.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

GRID=np.linspace(0.,.10,11); BETA=.95; H=10; CENTER=.25; SCALE=.1; DIM=5
ETA=np.array([.08,-.06,.05]); LAM=np.array([.35,-.25,.20])
F0=np.array([[.72,.08,0.],[.03,.65,.06],[0.,.05,.58]])
FSTAR=np.array([[-.55,.08,0.],[.03,.85,.06],[0.,.05,.30]])
F_MIX=F0+.69375*(FSTAR-F0)
QNOISE=np.diag(np.r_[.020**2,np.array([.040,.035,.030])**2])
I,J=np.triu_indices(DIM); MULT=np.where(I==J,1.,2.)

def form(M): return np.asarray(M)[I,J]*MULT
def unform(v):
    M=np.zeros((DIM,DIM)); M[I,J]=v; M[J,I]=v; return M

def reward_matrix(a):
    R=np.zeros((DIM,DIM)); R[0,0]=CENTER-.25*CENTER**2-2.025*a-.5*a*a
    R[0,1]=R[1,0]=.5*SCALE*(1.-.5*CENTER); R[1,1]=-.25*SCALE*SCALE
    return R

def branch_matrix(prior,a,F):
    A=np.zeros((4,4)); A[0,0]=.8+a; A[0,1:]=ETA+a*LAM; A[1:,1:]=F
    b=np.r_[.02+1.6*a-a*a,np.zeros(3)]
    T=np.zeros((DIM,DIM)); T[0,0]=1.; T[1:,0]=(A@np.r_[CENTER,np.zeros(3)]+b-np.r_[CENTER,np.zeros(3)])/SCALE; T[1:,1:]=A
    N=np.zeros((DIM,DIM)); N[1:,1:]=QNOISE/SCALE**2
    B=reward_matrix(a)+BETA*(T.T@prior@T); B[0,0]+=BETA*float(np.sum(prior*N))
    return .5*(B+B.T)

def cuts_initial(W):
    cuts=[]
    for k in range(DIM):
        e=np.zeros(DIM); e[k]=1.; cuts += [e,-e]
    for k in range(DIM):
        for l in range(k+1,DIM):
            for s in (-1.,1.):
                e=np.zeros(DIM); e[k]=1/np.sqrt(2); e[l]=s/np.sqrt(2); cuts.append(e)
    vals,vecs=np.linalg.eigh(W)
    cuts.extend(v for v in vecs[:,vals>1e-12].T)
    return cuts

def repair_lp(prior,branches,W,max_rounds=100):
    cuts=cuts_initial(W); obj=form(W); hist=[]; result=None
    for _ in range(max_rounds):
        Aub=[]; bub=[]
        for B in branches:
            for v in cuts: Aub.append(-form(np.outer(v,v))); bub.append(-float(v@B@v))
        result=linprog(obj,A_ub=np.asarray(Aub),b_ub=np.asarray(bub),bounds=[(None,None)]*len(obj),method='highs')
        if not result.success: raise RuntimeError('HiGHS: '+result.message)
        M=unform(result.x); eig=[float(np.linalg.eigvalsh(.5*((M-B)+(M-B).T))[0]) for B in branches]
        hist.append(min(eig))
        if min(eig)>=-1e-8: break
        for B in branches:
            R=.5*((M-B)+(M-B).T); vals,vecs=np.linalg.eigh(R)
            if vals[0]<-1e-8:
                v=vecs[:,0]; v=v/np.linalg.norm(v)
                if all(abs(float(c@v))<1.-1e-10 for c in cuts): cuts.append(v)
    raw=unform(result.x); pre=[float(np.linalg.eigvalsh(.5*((raw-B)+(raw-B).T))[0]) for B in branches]
    shift=max(0.,-min(pre))+1e-9; M=raw+shift*np.eye(DIM)
    post=[float(np.linalg.eigvalsh(.5*((M-B)+(M-B).T))[0]) for B in branches]
    if min(post)<-5e-9: raise FloatingPointError(f'PSD check failed {min(post)}')
    return M,dict(rounds=len(hist),cuts=len(cuts),converged=bool(hist and hist[-1]>=-1e-8),min_pre_eig=min(pre),min_post_eig=min(post),diagonal_shift=shift)

def public_K(x,m,P):
    u=np.r_[1.,(float(x)-CENTER)/SCALE,np.asarray(m,float)/SCALE]
    K=np.outer(u,u); K[2:,2:]+=np.asarray(P,float)/SCALE**2; return K

def fit_upper(F,W):
    U=reward_matrix(0.); stages=[]
    # U is V1; recursive supersolutions yield V2,...,V9.
    for h in range(2,H):
        branches=[branch_matrix(U,float(a),F) for a in GRID]
        U,d=repair_lp(U,branches,W)
        stages.append(dict(horizon=h,**d))
    B=[branch_matrix(U,float(a),F) for a in GRID]
    return B,stages

def seqvals(x,m,P,seq,F):
    seq=np.asarray(seq,float); n=len(seq)
    mu=np.zeros((n,4)); mu[:,0]=float(x); mu[:,1:]=m
    cov=np.zeros((n,4,4)); cov[:,1:,1:]=P
    value=np.zeros(n)
    for j in range(seq.shape[1]):
        a=seq[:,j]; mx=mu[:,0]; vx=cov[:,0,0]
        value+=BETA**j*(mx-.25*(mx*mx+vx)-2.025*a-.5*a*a)
        A=np.zeros((n,4,4)); A[:,0,0]=.8+a; A[:,0,1:]=ETA[None,:]+a[:,None]*LAM[None,:]; A[:,1:,1:]=F[None,:,:]
        b=np.zeros((n,4)); b[:,0]=.02+1.6*a-a*a
        mu=np.einsum('nij,nj->ni',A,mu)+b
        cov=np.einsum('nij,njk,nlk->nil',A,cov,A)+QNOISE
    return value

def lower_values(x,m,P,F,passes=16):
    starts=np.array([0.,.05,.10]); acts=np.repeat(np.tile(starts,len(GRID))[:,None],H,axis=1)
    acts[:,0]=np.repeat(GRID,len(starts))
    used=0
    for it in range(passes):
        changed=False
        for k in range(1,H):
            tr=np.repeat(acts[:,None,:],len(GRID),axis=1); tr[:,:,k]=GRID[None,:]
            vv=seqvals(x,m,P,tr.reshape(-1,H),F).reshape(len(acts),len(GRID))
            aa=GRID[np.argmax(vv,axis=1)]; changed|=bool(np.any(aa!=acts[:,k])); acts[:,k]=aa
        used=it+1
        if not changed: break
    val=seqvals(x,m,P,acts,F); lows=[]; tapes=[]
    for i in range(len(GRID)):
        ix=slice(3*i,3*i+3); j=int(np.argmax(val[ix]))+3*i
        lows.append(float(val[j])); tapes.append(acts[j].tolist())
    return np.array(lows),tapes,used

def evaluate_row(row,matrices):
    fam=row['family'].upper(); F=F0 if fam=='CM' else F_MIX
    x=float(row['actual_x']); a=float(row['executed_action']); m=np.asarray(row['posterior_mean'],float); P=np.asarray(row['posterior_covariance'],float)
    low,tapes,npass=lower_values(x,m,P,F); K=public_K(x,m,P); up=np.array([float(np.trace(B@K)) for B in matrices[fam]])
    gaps=up-low; idx=int(np.argmin(abs(GRID-a)))
    if abs(GRID[idx]-a)>1e-10: raise ValueError('executed action is off grid')
    if np.min(gaps)<-1e-7: status='numerical_upper_below_feasible_lower'
    else: status='scored'
    competitors=np.arange(len(GRID))!=idx
    regret=[max(0.,float(np.max(low[competitors])-up[idx])),max(0.,float(np.max(up[competitors])-low[idx]))]
    return dict(family=fam,arm=row['arm'],date=int(row['date']),actual_x=x,executed_action=a,status=status,
        posterior_mean=m.tolist(),posterior_covariance=P.tolist(),feasible_lower_values=low.tolist(),global_quadratic_upper_values=up.tolist(),
        feasible_sequences=tapes,coordinate_passes=npass,upper_minus_lower_min=float(np.min(gaps)),upper_minus_lower_max=float(np.max(gaps)),
        executed_action_loss_interval=regret,upper_possible_opt_actions=[float(GRID[i]) for i in range(len(GRID)) if up[i]>=float(np.max(low))-1e-12])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    source=json.loads(args.input.read_text()); rows=source['rows']
    matrices={}; diagnostics={}
    # One objective refinement: average posterior second moments at the actual
    # focal x inputs. This changes LP tightness only, not PSD-bound validity.
    W=np.mean([public_K(r['actual_x'],r['posterior_mean'],r['posterior_covariance']) for r in rows],axis=0)+1e-8*np.eye(DIM)
    objective_eigs=np.linalg.eigvalsh(W)
    for fam,F in [('CM',F0),('MIX',F_MIX)]: matrices[fam],diagnostics[fam]=fit_upper(F,W)
    scored=[evaluate_row(r,matrices) for r in rows]
    output=dict(method='expected H10 under the exact persistent post-32 CM/MIX transition law; open-loop feasible lower and global quadratic Bellman supersolution upper; public focal-plus-historical-probe posterior integrated at actual physical x',
      conditioning_information='known true law plus focal MAIN x,w,actions and all historical released probe endpoints through40; future MAIN/proxy releases only; companion reduction under CORE_BRIDGE_COMPANION_AUDIT_20261113 assumptions',
      reward='x-.25*x^2-2.025*a-.5*a^2',horizon=10,beta=.95,grid=GRID.tolist(),objective='mean actual-query public posterior second moment plus 1e-8 I; affects tightness only, not validity',objective_eigenvalues=objective_eigs.tolist(),
      laws={k:dict(family=k,F=F.tolist(),source='environment.py canonical source chain',stages=diagnostics[k],min_eig_by_horizon=[s['min_post_eig'] for s in diagnostics[k]]) for k,F in [('CM',F0),('MIX',F_MIX)]},
      numerical_scope='Floating point LP plus eigenvalue verification and isotropic shift; numerical bounds, not formal interval certificates. No realized future shocks; target is expected true-law H10.',rows=scored)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'rows':len(scored),'scored':sum(r['status']=='scored' for r in scored),'unresolved_bounds':sum(r['executed_action_loss_interval'][1]-r['executed_action_loss_interval'][0]>.005 for r in scored),'worst_interval_width':max(r['executed_action_loss_interval'][1]-r['executed_action_loss_interval'][0] for r in scored)}))
if __name__=='__main__': main()
