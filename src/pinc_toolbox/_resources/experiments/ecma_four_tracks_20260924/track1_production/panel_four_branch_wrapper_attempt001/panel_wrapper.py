"""Experimental opt-in panel transport for existing run_four_paths; no source inference.

PublicState.physical transports 96 actual state entries plus 32 released rewards.
PublicState.auxiliary transports 96 own auxiliary entries plus conditioned main w.
The last coordinate is an algorithm input, never a latent state intervention.
"""
from copy import deepcopy
import numpy as np
import pandas as pd
from pinc_toolbox.counterfactual import PublicState, PublicInput, PublicOutcome, run_four_paths
from pinc_toolbox.native_gq import GRID
from pinc_toolbox.g import SupportError

N=32

def logging_actions(time):
    return GRID[(np.arange(31)+(time-32))%11].copy()

def pack_state(physical, auxiliary, rewards=None, conditioned_w=None):
    x=np.asarray(physical,float);e=np.asarray(auxiliary,float)
    r=np.zeros(N) if rewards is None else np.asarray(rewards,float)
    if x.shape!=(N,3) or e.shape!=(N,3) or r.shape!=(N,):
        raise ValueError('panel dimensions must be 32x3, 32x3 and 32')
    cw=x[0,1] if conditioned_w is None else float(conditioned_w)
    state=PublicState(np.r_[x.ravel(),r],np.r_[e.ravel(),cw])
    unpack(state)
    return state

def unpack(state):
    if state.physical.shape!=(128,) or state.auxiliary.shape!=(97,):
        raise ValueError('invalid panel transport dimensions')
    if not np.isfinite(state.physical).all() or not np.isfinite(state.auxiliary).all():
        raise ValueError('nonfinite panel transport')
    return state.physical[:96].reshape(N,3).copy(),state.auxiliary[:96].reshape(N,3).copy(),state.physical[96:].copy(),float(state.auxiliary[96])

class PanelBranchLearner:
    """Wrap one own-history RND2BranchLearner; deep-copy ownership is explicit."""
    def __init__(self, learner):
        self.learner=deepcopy(learner)
        if self.learner.path_id!=0 or self.learner.refit_every!=1:
            raise ValueError('requires main path0 and per-date refresh')
        self.pending=None

    def prepare(self,time,state):
        if self.pending is not None:raise ValueError('unreleased panel issue')
        x,e,r,cw=unpack(state)
        main=self.learner.prepare(time,PublicState(x[0].copy(),e[0].copy()))
        # A post-release fork must really refresh, not reuse the prior issue model.
        if self.learner.model.as_of_!=time:raise ValueError('fork/decision model not prepared at actual date')
        e[:,:2]=main.auxiliary[:2];e[0]=main.auxiliary
        return pack_state(x,e,r,cw)

    def _main_query(self,state):
        x,e,_,cw=unpack(state);x[0,1]=cw
        return PublicState(x[0],e[0])

    def choose(self,time,state):return self.learner.choose(time,self._main_query(state))
    def supported(self,time,state,action):return self.learner.supported(time,self._main_query(state),action)

    def issue(self,time,state,action):
        if self.pending is not None:raise ValueError('duplicate panel issue')
        if not self.supported(time,state,action):raise SupportError('forced main action unsupported on own branch')
        x,e,_,cw=unpack(state);qstate=self._main_query(state);l=self.learner
        actions=np.r_[action,logging_actions(time)]
        issued=l.issue(time,qstate,action)
        base=l.model.base_model_.transition_.predict(x[1:],actions[1:])[:,:2]
        full=l.model.model_.transition_.predict(x[1:],actions[1:],e[1:])[:,:2]
        if not np.isfinite(base).all() or not np.isfinite(full).all():raise ValueError('nonfinite companion forecast')
        q=l.model.q_values(x[1:],e[1:]);mask=l.model.model_.support_.mask(x[1:],GRID)
        companions=[]
        for j in range(31):
            aindex=int(np.flatnonzero(GRID==actions[j+1])[0]);value=q[j,aindex]
            companions.append(dict(path_id=j+1,action=float(actions[j+1]),base_physical_prediction=base[j].tolist(),full_physical_prediction=full[j].tolist(),physical_action_supported=bool(mask[j,aindex]),finite_executed_q=bool(np.isfinite(value)),executed_action_q=float(value) if np.isfinite(value) else None,qualification='exogenous_logging_not_a_Q_recommendation'))
        self.pending=dict(time=time,physical=x,auxiliary=e,actions=actions,base=base,issued=deepcopy(issued))
        return dict(time=time,fit_as_of=l.model.as_of_,actual_physical=x.tolist(),own_auxiliary=e.tolist(),conditioned_main_input_w=cw,main_query_physical=qstate.physical.tolist(),main=issued,companions=companions,action_scope='main_only',proxy_scope='main_input_only',source_inference='unavailable')

    def observe(self,time,state,action,outcome):
        p=self.pending
        if p is None or p['time']!=time or p['actions'][0]!=action:raise ValueError('release without matching panel issue')
        if outcome.release_time!=time+1:raise ValueError('mistimed release')
        if outcome.next_physical.shape!=(128,) or not np.isfinite(outcome.next_physical).all():raise ValueError('bad panel release')
        actual=outcome.next_physical[:96].reshape(N,3);rewards=outcome.next_physical[96:]
        if not np.array_equal(actual[:,2],p['actions']):raise ValueError('released action memory mismatch')
        if outcome.reward!=rewards[0]:raise ValueError('main reward mismatch')
        if not np.array_equal(unpack(state)[0],p['physical']):raise ValueError('issued actual panel mismatch')
        x=p['physical'];e=p['auxiliary'];l=self.learner
        # Training records retain actual observed w; forecasts retain conditioned input.
        mainaux=l.observe(time,PublicState(x[0].copy(),e[0].copy()),action,PublicOutcome(actual[0].copy(),float(rewards[0]),time+1))
        nextaux=e.copy();nextaux[0]=mainaux
        nextaux[1:,2]=actual[1:,0]-p['base'][:,0]
        # e was captured at the matched issue; its residual was released at time.
        # The residual just calculated in nextaux is for the next action state.
        rows=[dict(path_id=i,period=time,release_period=time+1,state=x[i,0],w=x[i,1],previous_action=x[i,2],action=p['actions'][i],next_state=actual[i,0],next_w=actual[i,1],reward=rewards[i],delta_gamma=e[i,0],delta_m=e[i,1],residual=e[i,2],residual_source_period=time-1,residual_release_period=p['time'],auxiliary_as_of=time) for i in range(1,N)]
        l.history=pd.concat([l.history,pd.DataFrame(rows)],ignore_index=True)
        self.pending=None
        return np.r_[nextaux.ravel(),actual[0,1]]

class PanelEnvironmentAdapter:
    """Host owns actual panel, never receives conditioned proxy/auxiliary.

    host.public_main_proxy(time) -> actual released scalar w
    host.step_panel(time, actual_physical, actions) -> (next32x3, reward32, release_time)
    Host law must be declared separately; this interface does not identify it.
    """
    public_input_names=('observed_main_proxy_w',)
    def __init__(self,host):self.host=deepcopy(host)
    def public_input(self,time):return PublicInput(np.array([self.host.public_main_proxy(time)],float),time)
    def condition(self,state,public):
        x,e,r,_=unpack(state)
        return pack_state(x,e,r,float(public.values[0]))
    def step(self,time,state,action,public):
        x,_,_,_=unpack(state);actions=np.r_[action,logging_actions(time)]
        nx,rewards,release=self.host.step_panel(time,x.copy(),actions.copy())
        nx=np.asarray(nx,float);rewards=np.asarray(rewards,float)
        if nx.shape!=(N,3) or rewards.shape!=(N,) or not np.isfinite(nx).all() or not np.isfinite(rewards).all():raise ValueError('malformed host panel')
        if release!=time+1 or not np.array_equal(nx[:,2],actions):raise ValueError('host release/action clock mismatch')
        return PublicOutcome(np.r_[nx.ravel(),rewards],float(rewards[0]),release)

def run_panel_four_paths(*,learner,host,physical,auxiliary,start_time,steps,beta=1.):
    result=run_four_paths(learner=PanelBranchLearner(learner),environment=PanelEnvironmentAdapter(host),initial_state=pack_state(physical,auxiliary),start_time=start_time,steps=steps,beta=beta)
    result['panel_transport']=dict(paths=32,physical_layout='actual32x3_then_released_rewards32',auxiliary_layout='own32x3_then_conditioned_main_w',fixed_action_scope='main_only',fixed_proxy_scope='main_input_only',companion_schedule='GRID[(path_id-1+time-32)%11]',training_w='actual_released',scope='experimental diagnostic; no calibrated source adapter')
    return result
