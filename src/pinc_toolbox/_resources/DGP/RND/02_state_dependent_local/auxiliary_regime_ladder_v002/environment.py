"""Nested RND2 structural plug-ins; no learner or GQ imports."""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np
SOURCE=Path(__file__).resolve().parents[1]/'auxiliary_switch_attribution_v001/environment.py'
spec=importlib.util.spec_from_file_location('ladder_mother_environment',SOURCE)
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)
P0=M.P_PRE;ETA=M.ETA;LAMBDA=M.LAMBDA;F0=M.F_PRE;C1=M.PROXY_LOADING[0]
SU=M.LATENT_INNOVATION_SD;SIGE=M.STATE_INNOVATION_SD;SIGN=M.MEASUREMENT_NOISE_SD
GRID=M.ACTION_GRID;A0=.05;FSTAR=np.array([[-.55,.08,0],[.03,.85,.06],[0,.05,.30]])
SCENARIOS=('STABLE','BEHAVIOR','MODEL_A','MODEL_M','LATENT_F','LATENT_MEAN','MEASUREMENT','JOINT')

def _intensity(theta):
    theta=float(theta)
    if not np.isfinite(theta) or not 0<=theta<=1:raise ValueError('theta outside [0,1]')
    return theta

def law(scenario,theta=0.,joint_theta=None,date=0,switch_date=16):
    sc=str(scenario).upper();th=_intensity(theta)
    if sc not in SCENARIOS:raise ValueError(f'unknown scenario: {sc}')
    jt={} if joint_theta is None else {k:_intensity(v) for k,v in joint_theta.items()}
    if sc=='JOINT' and set(jt)!= {'MODEL_A','LATENT_F'}:raise ValueError('JOINT requires both component intensities')
    post=int(date)>=int(switch_date)
    p=P0;c=0.;F=F0.copy();b=np.zeros(3);sn=SIGN
    if post:
        if sc in ('MODEL_A','JOINT'):c=-1.2*(jt['MODEL_A'] if sc=='JOINT' else th)
        if sc=='MODEL_M':p=P0-.4*th
        if sc in ('LATENT_F','JOINT'):F=F0+(jt['LATENT_F'] if sc=='JOINT' else th)*(FSTAR-F0)
        if sc=='LATENT_MEAN':b=th*np.array([.04,-.03,.02])
        if sc=='MEASUREMENT':sn=SIGN*(1+3*th)
    return {'p':p,'c':c,'F':F,'b':b,'sigma_nu':sn,'post':post}
coefficient_law=law

def _burn_cov():
    P=np.zeros((3,3));Q=np.diag(SU**2)
    for _ in range(250):P=F0@P@F0.T+Q
    return P

class Environment:
    def __init__(self,seed,scenario,theta=0.,paths=8,horizon=40,switch_date=16,joint_theta=None):
        config=(str(scenario).upper(),_intensity(theta),joint_theta)
        law(config[0],config[1],config[2],0,switch_date)
        mother=M.Environment(int(seed),'STABLE',paths=int(paths),horizon=int(horizon),switch_date=int(switch_date))
        self.__dict__.update(mother.__dict__)
        self.scenario,self.theta,self.joint_theta=config;self._next_t=0
        self.metadata={'dgp':'RND2-AUX-LADDER-002','scenario':self.scenario,'theta':self.theta,'joint_theta':joint_theta,'first_affected_release':self.switch_date+1 if self.scenario not in ('STABLE','BEHAVIOR') and (self.theta>0 or self.scenario=='JOINT') else None}
    def public(self,t):
        t=int(t)
        if not 0<=t<=self._next_t:raise IndexError('requested unreleased public date')
        return self.state[:,t].copy(),self.w[:,t].copy()
    def step(self,t,actions):
        t=int(t)
        if t!=self._next_t or t>=self.horizon:raise ValueError('nonsequential or out-of-horizon transition')
        a=np.asarray(actions,float).reshape(-1)
        if len(a)==1 and self.paths!=1:a=np.repeat(a,self.paths)
        if len(a)!=self.paths or not np.isfinite(a).all():raise ValueError('one finite action per path required')
        l=law(self.scenario,self.theta,self.joint_theta,t,self.switch_date);x=self.state[:,t];z=self.latent[:,t]
        self.action[:,t]=a;self.reward[:,t]=x-.25*x**2-.5*a**2+M.TAU1*a
        self.state[:,t+1]=(l['p']*x+z@ETA+M.ACTION_SLOPE*a-M.ACTION_CURVATURE*a**2+M.STATE_ACTION_LOADING*x*a+a*(z@LAMBDA)+l['c']*(a-A0)+self.state_noise[:,t])
        self.latent[:,t+1]=z@l['F'].T+l['b']+self.latent_noise[:,t]
        self.proxy_all[:,t+1]=self.latent[:,t+1]@M.PROXY_LOADING.T+self.measurement_noise[:,t+1]*(l['sigma_nu']/SIGN)
        self._next_t=t+1
    @property
    def latent_truth(self):return self.latent
    @property
    def all_proxies(self):return self.proxy_all

def action_tape(seed,paths,horizon=40,policy='broad',switch_date=16):
    rng=np.random.default_rng(int(seed)+991)
    a=GRID[rng.integers(0,len(GRID),size=(int(horizon),int(paths)))]
    coins=rng.random(size=a.shape)
    if policy=='behavior':a[int(switch_date):]=np.where(coins[int(switch_date):]<.8,.05,a[int(switch_date):])
    elif policy=='reference':a[:]=.05
    elif policy!='broad':raise ValueError(policy)
    return a
