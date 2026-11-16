"""Released-history RA-GQ adapter using the installed RND2 G/Q implementation.

Auxiliary definitions are extracted from RND2 simple_adapter.common_deltas:
RMS change of both physical response coordinates on common action/reference
support; RMS first-physical reference-motion change; preissued base-G error.
No HS economic equations, latent state or independent planner are used.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import pandas as pd
from .counterfactual import PublicState, _date
from .native_gq import RND2GQ, GRID, XCOL, _fit_model, _positive_integer
from .g_estimators import EstimatorConfig
from .g import SupportError


def common_rnd2_deltas(old, new):
    """Use up to27 sorted actual common-reference probes, as in RND2 core."""
    probes = (pd.concat([old._train_frame, new._train_frame], ignore_index=True)
              [XCOL].drop_duplicates().sort_values(XCOL, kind='mergesort').to_numpy(float))
    eligible = np.ones(len(probes), bool)
    def masks(model, x):
        support = model.support_
        distances, ids = support.search.kneighbors(support.standardize(x))
        local = support.actions[ids]
        low, high = np.quantile(local, [.025, .975], axis=1)
        inside = distances[:, -1] <= support.distance_limit
        anchor = inside & (x[:, 2] >= low) & (x[:, 2] <= high)
        cells = inside[:, None] & (GRID >= low[:, None]) & (GRID <= high[:, None])
        return cells, anchor
    for model in (old, new):
        eligible &= masks(model, probes)[1]
    probes = probes[eligible]
    if not len(probes):
        raise SupportError('no common supported RND2 reference probes')
    probes = probes[np.linspace(0, len(probes)-1, min(27, len(probes)), dtype=int)]
    om, oa = masks(old, probes); nm, na = masks(new, probes)
    cells = om & nm & (oa & na)[:, None]
    if not cells.any():
        raise SupportError('no common supported RND2 action/reference pairs')
    x = np.repeat(probes, len(GRID), axis=0); actions = np.tile(GRID, len(probes))
    old_a = old.transition_.predict(x, actions)[:, :2]
    new_a = new.transition_.predict(x, actions)[:, :2]
    old_r = old.transition_.predict(x, x[:, 2])[:, :2]
    new_r = new.transition_.predict(x, x[:, 2])[:, :2]
    gamma = float(np.sqrt(np.mean(((new_a-new_r)-(old_a-old_r))[cells.ravel()]**2)))
    motion = float(np.sqrt(np.mean((new_r-old_r).reshape(len(probes), len(GRID), 2)[cells.any(1), 0, 0]**2)))
    return np.array([gamma, motion]), dict(probe_count=len(probes), supported_cells=int(cells.sum()),
        delta_gamma=gamma, delta_m=motion,
        definition='RND2 common_deltas: response RMS over state,w; reference RMS over state')


class RND2BranchLearner:
    """Full RA-GQ with own-history refresh and actual issued base-G receipts."""
    def __init__(self, model, history, *, path_id, refit_every=1,
                 q_evaluation="nearest", query_horizon=None):
        if not isinstance(model, RND2GQ) or (model.auxiliary != 'full' and not
                (model.g_spec=='v012' and model.auxiliary=='gamma')):
            raise ValueError('a fitted full RND2GQ is required')
        model._require_fitted()
        if isinstance(refit_every, bool) or not isinstance(refit_every, int) or refit_every < 1:
            raise ValueError('refit_every must be positive integer')
        if q_evaluation not in ('nearest', 'bellman'):
            raise ValueError("evaluation must be 'nearest' or 'bellman'")
        if query_horizon is not None:
            query_horizon = _positive_integer(query_horizon, 'remaining_horizon')
        self.model = deepcopy(model); self.history = history.drop(columns=['reward'], errors='ignore').copy(deep=True)
        self.path_id = int(path_id); self.refit_every = refit_every
        self.q_evaluation = q_evaluation; self.query_horizon = query_horizon
        self.start_time = model.as_of_; self.receipts = []; self.refit_audits = []; self.pending = None

    def prepare(self, time, state):
        if getattr(self.model,'g_spec','legacy')=='v012' and self.pending is not None:
            raise ValueError('release the pending v012 issue before refitting')
        if time < self.model.as_of_:
            raise ValueError('cannot run before fitted vintage')
        if time > self.model.as_of_ and (time-self.start_time) % self.refit_every == 0:
            candidate = deepcopy(self.model)
            try:
                candidate.update(self.history, as_of=time, window=self.model.window_)
            except ValueError as exc:
                if str(exc).startswith(('no common actual node at previous_action=', 'duplicate selection removed too many actual joint nodes')):
                    raise SupportError(str(exc)) from exc
                raise
            if getattr(self.model,'g_spec','legacy')=='v012':
                self.refit_audits.append(dict(time=time,g_spec='v012',
                    auxiliary_update='retain previous released raw record; refit does not rewrite it'))
            else:
                delta, audit = common_rnd2_deltas(self.model.base_model_, candidate.base_model_)
                state.auxiliary[:2] = delta
                self.refit_audits.append(dict(time=time, **audit))
            self.model = candidate
        return state

    def _q(self, state):
        expected=len(self.model.auxiliary_columns_)
        if state.physical.shape != (3,) or state.auxiliary.shape != (expected,):
            raise ValueError('RND2 requires state,w,previous_action and delta_gamma,delta_m,residual')
        return self.model.q_values(state.physical[None], state.auxiliary[None],
                                   remaining_horizon=getattr(self, 'query_horizon', None),
                                   evaluation=getattr(self, 'q_evaluation', 'nearest'))[0]

    def choose(self, time, state):
        q = self._q(state)
        if not np.isfinite(q).any():
            raise SupportError('no supported full RA-GQ action')
        return float(GRID[np.argmax(q)])

    def supported(self, time, state, action):
        match = np.flatnonzero(GRID == action)
        return bool(len(match) == 1 and np.isfinite(self._q(state)[match[0]]))

    def issue(self, time, state, action):
        if self.pending is not None:
            raise ValueError('previous forecast has not been released')
        if getattr(self.model,'g_spec','legacy')=='v012':
            time=_date(time,'time')
            if time<self.model.as_of_:
                raise ValueError('v012 issue time cannot precede its fitted vintage')
            if not hasattr(self,'residual_clock_'):
                raise ValueError('v012 issue requires a dated incoming residual from prepare_rnd2_learner')
            from .g_v012 import validate_history
            validate_history(pd.DataFrame([dict(period=time,**self.residual_clock_,
                **dict(zip(self.model.auxiliary_columns_,state.auxiliary)))]),self.model.auxiliary_columns_)
        base_prediction = self.model.base_model_.transition_.predict(state.physical[None], [action])[0]
        full_prediction = self.model.model_.transition_.predict(state.physical[None], [action])[0]
        base = float(base_prediction[0])
        full = float(full_prediction[0])
        q = self._q(state)
        record = dict(time=time, fit_as_of=self.model.as_of_, base_fit_as_of=self.model.as_of_, full_fit_as_of=self.model.as_of_,
                      q_values=[float(v) if np.isfinite(v) else None for v in q],
                      qualified_action_mask=np.isfinite(q).tolist(),
                      executed_action_q=float(q[np.flatnonzero(GRID==action)[0]]),
                      base_state_prediction=base,
                      full_state_prediction=full, action=action,
                      # Additive receipt fields: w is the observed proxy, not latent Z.
                      # Exact action memory is not an estimated physical response.
                      base_physical_prediction=base_prediction[:2].tolist(),
                      full_physical_prediction=full_prediction[:2].tolist(),
                      same_base_as_Q_kernel=True,
                      qualification='native_local_support_and_finite_Q', eta_status='not_supplied',
                      auxiliary_at_issue=state.auxiliary.tolist(),
                      active_auxiliaries=list(self.model.model_.transition_.active))
        self.pending = deepcopy(record)
        if getattr(self.model,'g_spec','legacy')=='v012':
            m0,gamma=self.model.model_.core.module_outputs(state.physical[None],[action])
            record.update(g_spec='v012',outer_config=self.model.outer_config_.as_dict(),
                module_m0=m0[0].tolist(),module_Gamma=gamma[0].tolist(),
                auxiliary_columns=list(self.model.auxiliary_columns_),
                auxiliary_provenance='previous_completed_release',
                physical_g_auxiliary_inputs=[])
            self.pending=deepcopy(record)
            # Deep-copy the issuing maps; later prepare/refit cannot alter them.
            self.pending['_issued_core']=deepcopy(self.model.model_.core)
            self.pending['_issued_x']=state.physical.copy()
            self.pending['_issued_e']=state.auxiliary.copy()
            self.pending['_incoming_clock']=deepcopy(self.residual_clock_)
        return record

    def observe(self, time, state, action, outcome):
        issued = self.pending
        if issued is None or issued['time'] != time or issued['action'] != action:
            raise ValueError('release lacks its issued forecast')
        if outcome.release_time != time+1 or outcome.next_physical[2] != action:
            raise ValueError('release must update previous_action to the executed action')
        x = state.physical; y = outcome.next_physical
        if getattr(self.model,'g_spec','legacy')=='v012':
            from .g_v012 import select_residuals,clocks
            x=issued['_issued_x']
            selected=select_residuals(issued['_issued_core'],x[None],[action],y[None,:2],
                                      self.model.auxiliary_columns_)[0]
            row=dict(path_id=self.path_id,period=time,release_period=time+1,
                state=x[0],w=x[1],previous_action=x[2],action=action,
                next_state=y[0],next_w=y[1],
                **dict(zip(self.model.auxiliary_columns_,issued['_issued_e'])),
                **deepcopy(issued['_incoming_clock']))
            self.history=pd.concat([self.history,pd.DataFrame([row])],ignore_index=True)
            receipt={k:deepcopy(v) for k,v in issued.items() if not k.startswith('_')}
            receipt.update(release_time=time+1,actual_physical=y[:2].tolist(),
                raw_residuals=selected.tolist(),residual_provenance='next_release',
                residual_source_period=time,residual_fit_as_of=issued['fit_as_of'])
            self.receipts.append(receipt)
            self.residual_clock_=clocks(time+1,issued['fit_as_of'])
            self.pending=None
            return selected
        row = dict(path_id=self.path_id, period=time, release_period=time+1,
                   state=x[0], w=x[1], previous_action=x[2], action=action,
                   next_state=y[0], next_w=y[1],
                   delta_gamma=state.auxiliary[0], delta_m=state.auxiliary[1], residual=state.auxiliary[2],
                   residual_source_period=time-1, residual_release_period=time, auxiliary_as_of=time)
        self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)
        residual = float(y[0]-issued['base_state_prediction'])
        self.receipts.append(dict(**issued, release_time=time+1, actual_state=float(y[0]), residual=residual))
        self.pending = None
        return np.r_[state.auxiliary[:2], residual]


def prepare_rnd2_learner(history, *, as_of, warmup_as_of, window,
                         path_id, refit_every=1, beta=.95, horizon=3,
                         nodes_per_memory=5, noise_nodes=4,
                         q_evaluation="nearest", query_horizon=None,
                         estimator="linear", estimator_params=None,
                         pipeline="planning_only",g_spec="legacy",outer_estimator="linear",
                         outer_estimator_params=None,auxiliary="full",policy_columns=(),
                         reward_coefficients=None, reward_equation=None, reward_inputs=None,
                         retain_warmup_rows=False):
    """Reconstruct public prefix auxiliaries, then fit actual full RND2 G/Q.

    Warmup forecasts are labeled reconstructed; online forecasts are issued
    before outcomes. Initial delta/residual are zero at warmup_as_of only.
    The caller supplies all released data; no fixture or environment is loaded.
    """
    if g_spec=='v012':
        if retain_warmup_rows:
            raise ValueError('neutral early warmup retention is a legacy full-RA initialization only')
        if reward_equation is not None:
            raise ValueError('prescribed equation prefix reconstruction currently supports legacy full RA only')
        return _prepare_v012_learner(history,as_of=as_of,warmup_as_of=warmup_as_of,window=window,
            path_id=path_id,refit_every=refit_every,beta=beta,horizon=horizon,
            nodes_per_memory=nodes_per_memory,noise_nodes=noise_nodes,
            q_evaluation=q_evaluation,query_horizon=query_horizon,estimator=estimator,
            estimator_params=estimator_params,pipeline=pipeline,outer_estimator=outer_estimator,
            outer_estimator_params=outer_estimator_params,auxiliary=auxiliary,policy_columns=policy_columns,
            reward_coefficients=reward_coefficients)
    if g_spec!='legacy' or auxiliary!='full' or policy_columns or outer_estimator!='linear' or outer_estimator_params:
        raise ValueError('new outer/auxiliary/macro settings require g_spec=v012')
    as_of=_date(as_of,'as_of');warmup_as_of=_date(warmup_as_of,'warmup_as_of')
    window=_positive_integer(window,'window');path_id=_date(path_id,'path_id')
    if not warmup_as_of <= as_of-window and not retain_warmup_rows:
        raise ValueError('warmup must precede as_of; positive window required')
    raw = history.drop(columns=['reward'], errors='ignore').copy(deep=True)
    # Reward is an evaluator outcome. RND2's learner uses the prescribed
    # reward law and only requires complete physical transitions here.
    required = XCOL + ['path_id','period','action','next_state','next_w']
    if not set(required) <= set(raw):
        raise ValueError('missing RND2 public transition columns')
    if 'release_period' not in raw:
        raw['release_period'] = raw.period+1
    if (not np.isfinite(raw[['period','release_period']].to_numpy(float)).all()
        or not np.equal(raw.period,np.floor(raw.period)).all()
        or (raw.release_period != raw.period+1).any()) or raw.duplicated(['path_id','period']).any():
        raise ValueError('one-period unique public transitions required')
    # Future data are excluded before any reconstruction or probe selection.
    raw = raw.loc[raw.release_period <= as_of].sort_values(['period','path_id'],kind='stable').copy()
    if not np.isfinite(raw[required].to_numpy(float)).all():
        raise ValueError('public transitions must be finite')
    selected_config = EstimatorConfig(
        estimator=estimator, pipeline=pipeline,
        estimator_params={} if estimator_params is None else estimator_params)
    base_config = (selected_config if selected_config.pipeline == 'full_ml'
                   else EstimatorConfig())
    def fit_base(date):
        frame = raw.loc[(raw.period >= date-window)&(raw.period < date)&(raw.release_period <= date)]
        return _fit_model(frame, estimator_config=base_config)
    base = fit_base(warmup_as_of); delta=np.zeros(2); residuals={pid:0. for pid in raw.path_id.unique()}
    residual_dates={pid:warmup_as_of-1 for pid in raw.path_id.unique()}
    annotated=[]; audits=[]
    if retain_warmup_rows:
        early=raw.loc[raw.period<warmup_as_of].copy()
        for name in ('delta_gamma','delta_m','residual'):early[name]=0.
        early['residual_source_period']=early.period-1
        early['residual_release_period']=early.period
        early['auxiliary_as_of']=early.period
        early['auxiliary_initialization']='neutral_warmup_not_measured_residual'
        annotated.append(early)
    for time in range(warmup_as_of, as_of):
        rows=raw.loc[raw.period==time].copy()
        if rows.empty:
            raise ValueError('warmup requires released rows at each date')
        rows['delta_gamma']=delta[0];rows['delta_m']=delta[1]
        rows['residual']=[residuals[pid] if residual_dates[pid]==time-1 else np.nan for pid in rows.path_id]
        rows['residual_source_period']=time-1;rows['residual_release_period']=time;rows['auxiliary_as_of']=time
        issued=base.transition_.predict(rows[XCOL].to_numpy(float),rows.action.to_numpy(float))[:,0]
        residuals.update(zip(rows.path_id,rows.next_state.to_numpy(float)-issued))
        residual_dates.update({pid:time for pid in rows.path_id})
        annotated.append(rows);new=fit_base(time+1);delta,audit=common_rnd2_deltas(base,new)
        audits.append(dict(time=time+1,provenance='reconstructed_prefix_not_live_forecast',**audit));base=new
    frame=pd.concat(annotated,ignore_index=True)
    model=RND2GQ('full',beta=beta,horizon=horizon,nodes_per_memory=nodes_per_memory,
                 noise_nodes=noise_nodes, estimator=selected_config.estimator,
                 estimator_params=selected_config.estimator_params,
                 pipeline=selected_config.pipeline,
                 reward_coefficients=reward_coefficients, reward_equation=reward_equation,
                 reward_inputs=reward_inputs).fit(frame,as_of=as_of,window=window)
    # The warmup's final base vintage must be the base used by full Q's residual successor.
    if selected_config.estimator == 'linear' and selected_config.pipeline == 'planning_only':
        np.testing.assert_allclose(base.core._C,model.base_model_.core._C,rtol=0,atol=1e-10)
        np.testing.assert_allclose(base.core._Psi,model.base_model_.core._Psi,rtol=0,atol=1e-10)
    else:
        check = model.base_model_._train_frame
        probe_states = check[XCOL].to_numpy(float)
        probe_actions = check.action.to_numpy(float)
        issued_base = base.transition_.predict(probe_states, probe_actions)
        fitted_base = model.base_model_.transition_.predict(probe_states, probe_actions)
        np.testing.assert_allclose(issued_base, fitted_base, rtol=1e-10, atol=1e-10,
                                   err_msg='warmup base forecasts differ from full-Q residual successor forecasts')
    last=raw.loc[raw.path_id==path_id].sort_values('period').iloc[-1]
    if last.period != as_of-1:
        raise ValueError('target path must have a final prefix release at as_of')
    state=PublicState(np.array([last.next_state,last.next_w,last.action]),np.r_[delta,residuals[path_id]])
    learner=RND2BranchLearner(model,frame,path_id=path_id,refit_every=refit_every,
                            q_evaluation=q_evaluation,query_horizon=query_horizon)
    learner.refit_audits=audits
    return learner,state


def _prepare_v012_learner(history, *, as_of,warmup_as_of,window,path_id,refit_every,
        beta,horizon,nodes_per_memory,noise_nodes,q_evaluation,query_horizon,
        estimator,estimator_params,pipeline,outer_estimator,outer_estimator_params,
        auxiliary,policy_columns,reward_coefficients):
    """Chronological prefix replay with independent raw records per path."""
    from .g_v012 import AUX,clocks,select_residuals
    model=RND2GQ(auxiliary,beta=beta,horizon=horizon,nodes_per_memory=nodes_per_memory,
        noise_nodes=noise_nodes,estimator=estimator,estimator_params=estimator_params,
        pipeline=pipeline,g_spec='v012',outer_estimator=outer_estimator,
        outer_estimator_params=outer_estimator_params,policy_columns=policy_columns,
        reward_coefficients=reward_coefficients)
    if auxiliary not in ('full','gamma'):
        raise ValueError('chronological v012 learner requires full raw group or Gamma-only')
    if model.policy_columns_:
        raise ValueError('macro Q requires an explicit future macro law; no future policy is simulated')
    as_of=_date(as_of,'as_of');warmup_as_of=_date(warmup_as_of,'warmup_as_of')
    window=_positive_integer(window,'window');path_id=_date(path_id,'path_id')
    if warmup_as_of>as_of-window:
        raise ValueError('warmup must precede as_of; positive window required')
    # Reward labels remain optional evaluator metadata, outside G/Q fitting.
    required=XCOL+['path_id','period','action','next_state','next_w']
    raw=history.drop(columns=['reward'],errors='ignore').copy(deep=True)
    if not set(required)<=set(raw):raise ValueError('missing RND2 public transition columns')
    if 'release_period' not in raw:raw['release_period']=raw.period+1
    if (not np.isfinite(raw[['period','release_period']].to_numpy(float)).all()
            or not np.equal(raw.period,np.floor(raw.period)).all()
            or (raw.release_period!=raw.period+1).any()
            or raw.duplicated(['path_id','period']).any()):
        raise ValueError('one-period unique public transitions required')
    raw=raw.loc[raw.release_period<=as_of].sort_values(['period','path_id'],kind='stable').copy()
    if not np.isfinite(raw[required].to_numpy(float)).all():raise ValueError('public transitions must be finite')
    cols=AUX[auxiliary];ids=tuple(raw.path_id.unique())
    latest={pid:np.zeros(len(cols)) for pid in ids}
    vintages={pid:warmup_as_of-1 for pid in ids}
    annotated=[];audits=[]
    for time in range(warmup_as_of,as_of):
        complete=raw.loc[(raw.period>=time-window)&(raw.period<time)&(raw.release_period<=time)]
        # The exact same continuity check as native fit, on realized history.
        ordered=complete.sort_values(['path_id','period'],kind='stable')
        lag=ordered.groupby('path_id').action.shift(1);date=ordered.groupby('path_id').period.shift(1)
        mask=ordered.period.eq(date+1)
        if not np.allclose(ordered.loc[mask,'previous_action'],lag.loc[mask],rtol=0,atol=1e-12):
            raise ValueError('previous_action must be the executed action at period-1')
        fitted=_fit_model(complete,g_spec='v012',outer_config=model.outer_config_)
        rows=raw.loc[raw.period==time].copy()
        if set(rows.path_id)!=set(ids):raise ValueError('v012 prefix requires every path at each date')
        for j,c in enumerate(cols):rows[c]=rows.path_id.map(lambda pid:latest[pid][j])
        for key in clocks(time,0):
            rows[key]=[clocks(time,vintages[pid],warmup=time==warmup_as_of)[key] for pid in rows.path_id]
        annotated.append(rows)
        values=select_residuals(fitted.core,rows[XCOL].to_numpy(float),rows.action.to_numpy(float),
                                 rows[['next_state','next_w']].to_numpy(float),cols)
        for i,pid in enumerate(rows.path_id):latest[pid]=values[i].copy();vintages[pid]=time
        audits.append(dict(time=time,fit_as_of=time,source_period=time,release_period=time+1,
            provenance='reconstructed_prefix_not_live_forecast',g_spec='v012',outer_config=model.outer_config_.as_dict()))
    frame=pd.concat(annotated,ignore_index=True)
    model.fit(frame,as_of=as_of,window=window)
    target=raw.loc[raw.path_id==path_id].sort_values('period')
    if target.empty or target.iloc[-1].period!=as_of-1:raise ValueError('target path must have a final prefix release at as_of')
    last=target.iloc[-1]
    state=PublicState(np.array([last.next_state,last.next_w,last.action]),latest[path_id].copy())
    learner=RND2BranchLearner(model,frame,path_id=path_id,refit_every=refit_every,
        q_evaluation=q_evaluation,query_horizon=query_horizon)
    learner.residual_clock_=clocks(as_of,vintages[path_id])
    learner.refit_audits=audits
    boundary=raw.loc[raw.period==as_of-1].sort_values('path_id')
    learner.prefix_boundary_=dict(path_ids=boundary.path_id.to_numpy().copy(),
        physical=boundary[['next_state','next_w','action']].to_numpy(float),
        auxiliary=np.stack([latest[pid] for pid in boundary.path_id]),
        fit_as_of=np.asarray([vintages[pid] for pid in boundary.path_id],int))
    return learner,state
