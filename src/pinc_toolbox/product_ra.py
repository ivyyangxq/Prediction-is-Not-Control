"""Incremental cache of the existing chronological legacy RA reconstruction.

Uses the same G fits and common_rnd2_deltas as prepare_rnd2_learner. The cache
avoids refitting every historical vintage again at every new release.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .native_gq import RND2GQ, XCOL, _fit_model
from .rnd2_counterfactual import prepare_rnd2_learner, common_rnd2_deltas
from .g import SupportError


class ReleasedRACache:
    def __init__(self,prefix,time,settings,equation):
        self.settings=settings;self.equation=equation
        learner,boundary=prepare_rnd2_learner(prefix,as_of=time,warmup_as_of=settings['warmup_as_of'],
            window=settings['window'],path_id=settings['path'],beta=settings['discount'],horizon=settings['horizon'],
            q_evaluation='bellman',nodes_per_memory=settings.get('nodes_per_memory',5),noise_nodes=settings.get('noise_nodes',8),
            reward_equation=equation,reward_inputs=settings['reward']['inputs'],
            retain_warmup_rows=settings.get('retain_warmup_rows',False))
        self.annotated=learner.history.copy();self.base=learner.model.base_model_;self.date=time
        self.auxiliary=boundary.auxiliary.copy();self.delta=boundary.auxiliary[:2].copy();self.initial_model=learner.model
        previous=self._base_at(prefix,time-1)
        last=prefix.loc[prefix.period==time-1]
        errors=last.next_state.to_numpy(float)-previous.transition_.predict(last[XCOL].to_numpy(float),last.action.to_numpy(float))[:,0]
        self.incoming={int(pid):np.r_[boundary.auxiliary[:2],error] for pid,error in zip(last.path_id,errors)}
        self.audits=list(learner.refit_audits)

    def _base_at(self,prefix,time):
        window=self.settings['window']
        return _fit_model(prefix.loc[(prefix.period>=time-window)&(prefix.period<time)&(prefix.release_period<=time)])

    def advance(self,prefix,time):
        if time==self.date:return
        if time!=self.date+1:raise ValueError('incremental RA requires consecutive release dates')
        rows=prefix.loc[prefix.period==self.date].copy()
        for j,name in enumerate(('delta_gamma','delta_m','residual')):
            rows[name]=[self.incoming.get(int(pid),np.r_[self.delta,np.nan])[j] for pid in rows.path_id]
        rows['residual_source_period']=self.date-1
        rows['residual_release_period']=self.date
        rows['auxiliary_as_of']=self.date
        # Old paired raw auxiliary records are never rewritten.
        annotated=pd.concat([self.annotated,rows],ignore_index=True)
        newbase=self._base_at(prefix,time)
        errors=rows.next_state.to_numpy(float)-self.base.transition_.predict(rows[XCOL].to_numpy(float),rows.action.to_numpy(float))[:,0]
        try:
            delta,audit=common_rnd2_deltas(self.base,newbase)
        except SupportError as error:
            delta=np.full(2,np.nan);audit={'status':'unavailable','reason':str(error)}
        incoming={int(pid):np.r_[delta,error] for pid,error in zip(rows.path_id,errors)}
        self.annotated=annotated;self.base=newbase;self.date=time;self.incoming=incoming;self.delta=delta.copy()
        self.auxiliary=incoming.get(self.settings['path'],np.full(3,np.nan)).copy()
        self.audits.append(dict(time=time,provenance='reconstructed_cached_issuing_vintage_not_live_receipt',**audit))

    def fit(self,feasible):
        s=self.settings
        model=RND2GQ('full',beta=s['discount'],horizon=s['horizon'],action_grid=s['grid'],
            nodes_per_memory=s.get('nodes_per_memory',5),noise_nodes=s.get('noise_nodes',8),
            reward_equation=self.equation,reward_inputs=s['reward']['inputs'])
        return model.fit(self.annotated,as_of=self.date,window=s['window'],feasible=feasible)
