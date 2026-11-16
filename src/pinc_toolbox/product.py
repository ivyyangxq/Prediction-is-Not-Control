"""Released-panel product engine used by GQ_tool.ipynb and experiment callers.

No simulator, oracle, reward fitting, automatic intervention or source assertion
is imported here. Native GQ and certified public-history workflows are reused.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import gzip
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd

from .guided_gq import panel_history, MODES, _clock
from .native_gq import RND2GQ, XCOL
from .reward_equation import RewardEquation


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def clean(value):
    """Strict JSON, preserving missing measurements as null, never zero."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def frame_digest(frame):
    # Includes all original columns and row order, including optional labels.
    return hashlib.sha256(frame.to_csv(index=False, float_format='%.17g').encode()).hexdigest()


def prepare_panel(panel, macro, mapping, macro_value, macro_release, *, macro_unit=None):
    """Named CSV columns -> native history, using actual publication timestamps.

    Unit labels can be strings. Explicit previous actions are validated against
    within-unit recorded actions; otherwise a consecutive lag is constructed.
    """
    required = {'path_id', 'period', 'release_period', 'state', 'action', 'next_state'}
    if not required <= set(mapping):
        raise ValueError(f'mapping needs {sorted(required)}')
    panel = panel.copy(deep=True)
    original_panel_digest=frame_digest(panel)
    unit_keys=panel[mapping['path_id']].astype(str).to_numpy()
    ids, labels = pd.factorize(panel[mapping['path_id']], sort=True)
    if (ids < 0).any():
        raise ValueError('unit identifiers must be complete')
    panel[mapping['path_id']] = ids
    if macro_unit is None:
        history = panel_history(panel, mapping, macro=(macro, macro_value, macro_release))
    else:
        # A keyed path proxy is a separate physical input, never relabeled
        # a common macro series. Retain one global decision calendar.
        numeric=pd.api.types.is_numeric_dtype(panel[mapping['period']])
        decisions=_clock(panel[mapping['period']],numeric)
        calendar=np.unique(decisions)
        successors=decisions+1 if numeric else np.array([calendar[i+1] if i+1<len(calendar) else np.iinfo(np.int64).max for i in np.searchsorted(calendar,decisions)])
        current=np.full(len(panel),np.nan);following=np.full(len(panel),np.nan)
        for key in np.unique(unit_keys):
            selected=macro.loc[macro[macro_unit].astype(str).eq(key)]
            if selected.empty:continue
            published=_clock(selected[macro_release],numeric);values=pd.to_numeric(selected[macro_value]).to_numpy(float)
            if len(np.unique(published))!=len(published) or not np.isfinite(values).all():
                raise ValueError('unit proxy needs finite values and unique publication timestamps per unit')
            order=np.argsort(published);published,values=published[order],values[order]
            rows=np.flatnonzero(unit_keys==key)
            for times,destination in ((decisions,current),(successors,following)):
                index=np.searchsorted(published,times[rows],side='right')-1
                destination[rows]=np.where(index>=0,values[index],np.nan)
        panel['__pinc_public_w']=current;panel['__pinc_next_public_w']=following
        if not numeric:
            panel.loc[decisions==calendar[-1],'__pinc_next_public_w']=np.nan
        mapping={**mapping,'w':'__pinc_public_w','next_w':'__pinc_next_public_w'}
        history=panel_history(panel,mapping)
    ordered = history.sort_values(['path_id', 'period'])
    previous = ordered.groupby('path_id').action.shift(1)
    consecutive = ordered.period.eq(ordered.groupby('path_id').period.shift(1) + 1)
    # A provided actual anchor can be checked only against a known logged action.
    # Missing logged actions stay missing even when the next anchor is supplied.
    check_previous = consecutive & previous.notna()
    if not np.allclose(ordered.loc[check_previous, 'previous_action'], previous[check_previous],
                       rtol=0, atol=1e-12, equal_nan=True):
        raise ValueError('previous_action must equal that unit\'s previous recorded executed action')
    history.attrs.update(unit_labels={int(i): str(v) for i, v in enumerate(labels)},
                         macro_semantics='common observable W; latest publication by decision' if macro_unit is None else 'unit-specific observable path proxy; NOT common macro W',
                         raw_panel_digest=original_panel_digest, raw_macro_digest=frame_digest(macro))
    return history


def released_window(history, as_of, window, learn_start, learn_end, *, updating=True):
    """Only released transitions, never incomplete records with fabricated actions."""
    end = as_of if updating else learn_end
    first = max(learn_start, end-window)
    rows = history.loc[(history.period >= first) & (history.period < end)
                       & (history.release_period <= as_of)].copy()
    # Reward labels cannot change sample selection, geometry or G.
    physical = XCOL + ['action', 'next_state', 'next_w']
    complete = np.isfinite(rows[physical].to_numpy(float)).all(axis=1)
    return rows.loc[complete].drop(columns='reward', errors='ignore'), {
        'first_allowed_period': first, 'last_allowed_period_exclusive': end,
        'released_rows': len(rows), 'usable_rows': int(complete.sum()),
        'incomplete_rows': int((~complete).sum()),
        'missing_action_rows': int(rows.action.isna().sum())}


class SupportTracker:
    """Fixed scaled L-infinity joint-cloud measurement, distinct from native kNN.

    eta=None means not supplied. This does not gate or widen the controller.
    """
    def __init__(self, query, grid, scale, eta):
        self.query = np.asarray(query, float).copy()
        self.grid = np.asarray(grid, float).copy()
        self.scale = np.asarray(scale, float).copy()
        self.eta = eta
        if self.query.shape != (3,) or not np.isfinite(self.query).all():
            raise ValueError('fixed public query needs three finite coordinates')
        if self.scale.shape != (4,) or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError('reference scales need four fixed positive values: state,W,anchor,action')
        if eta is not None and (not np.isfinite(eta) or eta < 0):
            raise ValueError('eta must be nonnegative or None')
        self.previous_mask = None
        self.previous_rows = set()
        self.native_reference=None
        self.previous_native_common=None

    def set_native_reference(self,support,as_of):
        if self.native_reference is None:
            self.native_reference={'center':support.center.copy(),'scale':support.scale.copy(),
                'neighbors':support.neighbors,'neighbor_rule':'min(frozen maximum_k, currently available rows)',
                'distance_limit':support.distance_limit,
                'lower_q':support.lower_q,'upper_q':support.upper_q,'fit_as_of':as_of}

    def fixed_native(self,rows,state):
        reference=self.native_reference
        if reference is None or rows.empty or state is None or not np.isfinite(state).all():
            return {'status':'unavailable','mask':None}
        distances=np.linalg.norm((rows[XCOL].to_numpy(float)-state)/reference['scale'],axis=1)
        k=min(reference['neighbors'],len(rows));indices=np.argsort(distances,kind='stable')[:k]
        lower,upper=np.quantile(rows.action.to_numpy(float)[indices],[reference['lower_q'],reference['upper_q']])
        inside=distances[indices[-1]]<=reference['distance_limit']
        mask=(self.grid>=lower)&(self.grid<=upper)&inside
        return clean({'status':'measured','mask':mask,'neighbor_radius':distances[indices[-1]],
            'boundary_margin':reference['distance_limit']-distances[indices[-1]],
            'action_quantiles':[lower,upper],'neighbors_used':k})

    def distances(self, rows, state):
        if state is None or not np.isfinite(state).all() or rows.empty:
            return None
        cloud = rows[XCOL + ['action']].to_numpy(float)
        points = np.column_stack([np.tile(state, (len(self.grid), 1)), self.grid])
        return np.min(np.max(np.abs(points[:, None, :] - cloud[None, :, :]) /
                             self.scale, axis=2), axis=1)

    def measure(self, rows, state):
        common = self.distances(rows, self.query)
        own = self.distances(rows, state)
        mask = common <= self.eta if common is not None and self.eta is not None else None
        ids = set(zip(rows.path_id.astype(int), rows.period.astype(int)))
        transitions = None
        if mask is not None and self.previous_mask is not None:
            old = self.previous_mask
            transitions = {key: self.grid[which].tolist() for key, which in {
                'added': mask & ~old, 'retained': mask & old,
                'lost': ~mask & old, 'unsupported_both': ~mask & ~old}.items()}
            transitions['net_change'] = len(transitions['added']) - len(transitions['lost'])
        self.previous_mask = None if mask is None else mask.copy()
        entrants, exits = sorted(ids-self.previous_rows), sorted(self.previous_rows-ids)
        self.previous_rows = ids
        native_common=self.fixed_native(rows,self.query)
        native_own=self.fixed_native(rows,state)
        native_changes=None
        current=native_common['mask']
        if current is not None and self.previous_native_common is not None:
            current=np.asarray(current,bool);old=self.previous_native_common
            native_changes={key:self.grid[which].tolist() for key,which in {
                'added':current&~old,'retained':current&old,'lost':~current&old}.items()}
            native_changes['net_change']=len(native_changes['added'])-len(native_changes['lost'])
        self.previous_native_common=None if current is None else np.asarray(current,bool)
        return clean(dict(
            metric='fixed scaled L-infinity distance to released joint state-action cloud; section15',
            controller_mapping='measurement only; native kNN quantile rule is unchanged',
            eta=self.eta, eta_status='not_supplied' if self.eta is None else 'user_supplied_geometric_tolerance',
            reference_query=self.query, reference_scale=self.scale,
            common_query_distance=common,
            common_query_margin=None if mask is None else self.eta-common,
            common_query_mask=mask, changes=transitions,
            own_query_same_scale_distance=own,
            own_query_same_scale_margin=None if own is None or self.eta is None else self.eta-own,
            fixed_native_reference=self.native_reference,
            fixed_native_common_query=native_common,fixed_native_own_query=native_own,
            fixed_native_common_changes=native_changes,
            fixed_native_scope='frozen first measured native metric/limit/k/quantiles, current released cloud; geometric readback only, not the L-infinity eta tube or economic qualification',
            entrants=[list(x) for x in entrants], exits=[list(x) for x in exits],
            economic_value_interval={'status':'unavailable', 'reason':'observational identification, continuation and uncertainty not established'}))


class EvidenceWorkflow:
    """Live existing certified workflow with binding and explicit applicability.

    A JSON source assertion or an old report cannot activate repair. The actual
    workflow is executed here. Applicability remains a declared mathematical
    premise, recorded in full; this adapter does not prove it from panel/W.
    """
    def __init__(self, request, *, binding, applicability):
        from .public_history_workflow import PublicHistoryWorkflow
        self.request = deepcopy(request)
        self.binding = deepcopy(binding)
        self.applicability = deepcopy(applicability)
        self.workflow = PublicHistoryWorkflow(request)

    def step(self, action, node=None):
        self.workflow.step(action, node=node)
        return self.snapshot()

    def snapshot(self):
        snapshot = self.workflow.snapshot()
        prepared = snapshot.get('prepared') or {}
        record = prepared.get('region') or {}
        source = snapshot.get('source_workflow') or {}
        router = source.get('router') or {}
        outcome = router.get('outcome')
        tested=(source.get('statuses') or {}).get('tested') or {}
        stage1=deepcopy(record)
        if 'C0' in tested:
            stage1['alarm']=True if tested['C0']=='excluded' else False if tested['C0']=='compatible' else None
            stage1['alarm_method']='existing certified no-change-family exclusion'
        # The certified monitor exposes readiness/support; first-stage alarm is
        # reported only if present in the native selected-window record.
        return clean(dict(binding=self.binding, applicability=self.applicability,
                          request_digest=digest(self.request), as_of=self.request['as_of'],
                          stage1=stage1, source_outcome=outcome,
                          model_involvement=outcome in {'model_required', 'both_required'},
                          status=snapshot['status'], native_snapshot=snapshot))


def demo_data(seed=1901):
    """Small external logging panel with common released macro; engineering only."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, .1, 11)
    dates = np.arange(-32, 13)
    w = np.zeros(len(dates)+1)
    for j in range(1, len(w)):
        w[j] = .6*w[j-1] + rng.normal(0, .03)
    macro = pd.DataFrame({'publication': np.arange(-32, 14), 'macro_W': w})
    records = []
    for unit in range(16):
        x, previous = rng.normal(.5, .1), float(rng.choice(grid))
        for j, date in enumerate(dates):
            action = float(rng.choice(grid))
            successor = .3 + .4*x + .2*w[j] + .1*previous + .5*(action-previous) + rng.normal(0, .012)
            records.append(dict(unit=f'unit_{unit:02}', date=int(date), released=int(date+1),
                                x=x, executed=action, anchor=previous, next_x=successor))
            x, previous = successor, action
    mapping = dict(path_id='unit', period='date', release_period='released', state='x',
                   action='executed', previous_action='anchor', next_state='next_x')
    return pd.DataFrame(records), macro, mapping


def demo_settings():
    return dict(mode=MODES[1], update_policy='rolling', learn_start=-32, learn_end=0,
                start=0, end=8, path=0, window=24, horizon=3, discount=.95,
                grid=np.linspace(0,.1,11).tolist(), candidates=np.linspace(0,.1,11).tolist(),
                reward={'expression':'x - curvature*x*x - cost*a - .5*a*a',
                        'inputs':{'x':'state','a':'action'},
                        'parameters':{'curvature':.25,'cost':2.025}},
                reference_scale=[.15,.05,.05,.05], eta=None,
                nodes_per_memory=3, noise_nodes=2)


class PanelSession:
    """One release clock, fixed reward, support timeline, and explicit repair state."""
    def __init__(self, history, settings):
        self.history = history.copy(deep=True)
        self.settings = deepcopy(settings)
        s = self.settings
        if s['mode'] not in MODES or s['update_policy'] not in {'fixed','rolling','report_guided'}:
            raise ValueError('invalid mode or update policy')
        for key in ('learn_start','learn_end','start','end','window','horizon','path'):
            if type(s[key]) is not int:
                raise ValueError(f'{key} must be an integer model period')
        if not s['learn_start'] < s['learn_end'] <= s['start'] <= s['end']:
            raise ValueError('need learn_start < learn_end <= start <= end')
        if s['window'] < 1 or s['horizon'] < 1:
            raise ValueError('window and horizon must be positive')
        if 'decision_calendar' in self.history.attrs and s['end']>=len(self.history.attrs['decision_calendar']):
            raise ValueError('calendar monitoring end must be an observed decision-calendar index')
        self.equation = RewardEquation(s['reward']['expression'], s['reward']['inputs'], s['reward']['parameters'])
        self.grid = np.asarray(s['grid'], float)
        if self.grid.ndim != 1 or not len(self.grid) or not np.isfinite(self.grid).all() or (np.diff(self.grid)<=1e-12).any():
            raise ValueError('grid must be finite, unique and increasing')
        if not s['candidates'] or not np.isin(s['candidates'], self.grid).all():
            raise ValueError('candidates must be a nonempty subset of the fixed grid')
        self.feasible = np.isin(self.grid,s['candidates'])
        # Validate input mapping and discount through the actual native constructor.
        self._new_model()
        self.identity = {'history':frame_digest(self.history), 'settings':digest(s), 'reward':digest(s['reward'])}
        self.evidence = None
        self.source_reports = []
        self.repair = {'status':'not_requested', 'method':None, 'activation_date':None}
        self.model = None
        self.query_auxiliary = None
        self.ra_audit = None
        self.ra_cache = None
        self.estimated = None
        self.broken_estimated_path = False
        self.timeline = []
        self.tracker = None
        self.next_date = s['start']
        self.last_update_keys = set()
        self.pending_forecasts=[]

    def _new_model(self):
        s = self.settings
        return RND2GQ(beta=s['discount'], horizon=s['horizon'], action_grid=s['grid'],
                      reward_equation=self.equation, reward_inputs=s['reward']['inputs'],
                      nodes_per_memory=s.get('nodes_per_memory',5), noise_nodes=s.get('noise_nodes',8))

    @classmethod
    def from_report(cls, history, report):
        """Resume from JSON by deterministic reconstruction, without unpickling.

        Already issued predictions are reproduced and checked; the recorded
        timeline is retained. Input history/config changes require a new run.
        """
        obj=cls(history,report['configuration'])
        if obj.identity != report['identity']:
            raise ValueError('resume requires identical history, settings and reward')
        evidence=report.get('live_evidence')
        if evidence is not None:
            from .public_history_workflow import PublicHistoryWorkflow
            native=evidence['native_snapshot']
            if 'request' not in native:
                raise ValueError('controlled-source JSON replay is not supported; use the controlled acquisition directory for a new session or resume the registered batch allocation')
            request=native['request']
            obj.attach_evidence(request,applicability=evidence['applicability'])
            obj.evidence.workflow=PublicHistoryWorkflow.from_state(native,request)
        saved=deepcopy(report['timeline'])
        for expected in saved:
            repair=expected['repair']
            if repair['status']=='activated' and obj.repair['status']!='activated':
                obj.decide_repair(True,method=repair['method'])
            elif repair['status']=='declined':
                obj.repair=deepcopy(repair)
            actual=obj.advance()
            for key in ('date','status','fit_as_of','training_digest','query_state',
                        'recorded_action','recommended_action','reward_value','issued_next_state'):
                if actual[key]!=expected[key]:
                    raise ValueError(f'resume reconstruction differs at date {expected["date"]}: {key}')
        obj.timeline=saved
        obj.source_reports=deepcopy(report.get('source_reports',[]))
        obj.repair=deepcopy(report['repair'])
        return obj

    def attach_evidence(self, request, *, applicability):
        """Execute declared certified inputs; never translate generic panel into claims."""
        if not applicability.get('scope') or not applicability.get('premises'):
            raise ValueError('evidence needs scope and explicit applicability premises')
        if request.get('application_binding') != self.identity:
            raise ValueError('request.application_binding must match this session history/settings/reward identity; old or unrelated requests cannot be relabeled')
        self.evidence = EvidenceWorkflow(request,binding=self.identity,applicability=applicability)
        return self.evidence.snapshot()

    def evidence_step(self, action, node=None):
        if self.evidence is None:
            raise ValueError('attach the current evidence request first')
        result = self.evidence.step(action,node)
        self.source_reports.append(result)
        return result

    def decide_repair(self, accept, *, method='rolling_base_GQ', as_of=None):
        """Explicit report-to-update handoff; no economic recovery assertion."""
        if as_of is None:
            as_of = self.next_date
        if as_of != self.next_date:
            raise ValueError('repair decisions apply at the next unprocessed release only')
        report = None if self.evidence is None else self.evidence.snapshot()
        if accept:
            if self.settings['mode']==MODES[0]:
                raise ValueError('Offline keeps its initial snapshot; use an online mode for a MODEL-authorized rolling repair')
            if method not in {'rolling_base_GQ','rolling_full_RA_GQ'}:
                raise ValueError('choose rolling_base_GQ or rolling_full_RA_GQ')
            if method == 'rolling_full_RA_GQ' and 'warmup_as_of' not in self.settings:
                raise ValueError('full RA requires an explicit chronological warmup_as_of before start-window')
            if (report is None or not report['model_involvement'] or report['as_of'] > as_of
                    or report['binding'] != self.identity or report['status'] != 'resolved'
                    or self.evidence.applicability.get('repair_applicable') is not True):
                raise ValueError('a current, released, resolved MODEL workflow with declared applicable premises is required')
        self.repair = dict(status='activated' if accept else 'declined',method=method if accept else None,
                           activation_date=as_of, report_digest=None if report is None else digest(report),
                           scope=None if report is None else report['applicability'],
                           economic_recovery='not_established')
        return deepcopy(self.repair)

    def _evidence_status(self, time):
        if self.evidence is None:
            return {'stage1':'unavailable','source':'unavailable',
                    'reason':'panel/W alone do not specify a calibrated measurement, reference model, admitted latent alternatives or conditional source law'}
        result = self.evidence.snapshot()
        if result['as_of'] > time:
            return {'stage1':'not_yet_released','source':'not_yet_released'}
        return {'stage1':result['stage1'], 'source':result['source_outcome'] or result['status'],
                'report_digest':digest(result), 'scope':result['applicability']}

    def advance(self):
        s = self.settings
        time = self.next_date
        if time > s['end']:
            raise StopIteration('monitoring interval is complete')
        started = perf_counter()
        focal = self.history.loc[(self.history.path_id==s['path']) & (self.history.period==time)]
        observed = None if focal.empty else focal.iloc[0][XCOL].to_numpy(float)
        recorded = None if focal.empty or not np.isfinite(focal.iloc[0].action) else float(focal.iloc[0].action)
        if self.tracker is None:
            reference = s.get('reference_query',observed)
            self.tracker = SupportTracker(reference, self.grid, s['reference_scale'], s.get('eta'))
            self.estimated = None if observed is None else observed.copy()
        observed_mode = s['mode']==MODES[1]
        state = observed if observed_mode else (None if self.broken_estimated_path else self.estimated)
        updating = s['update_policy']=='rolling' or self.repair['status']=='activated'
        # Offline keeps its initial snapshot even if rolling was chosen in UI.
        updating = updating and s['mode']!=MODES[0]
        rows, receipt = released_window(self.history,time,s['window'],s['learn_start'],s['learn_end'],updating=updating)
        support_rows, support_receipt = released_window(self.history,time,s['window'],s['learn_start'],s['learn_end'],updating=True)
        # Initial historical learning must exclude dates outside the declared interval.
        if not self.timeline and self.repair['status']!='activated':
            rows, receipt = released_window(self.history,time,s['window'],s['learn_start'],s['learn_end'],updating=False)
            receipt['usable_rows'] = len(rows)
        # An initial/frozen fit has its declared learning vintage even when
        # monitoring starts later or a refused fit is retried.
        initial_snapshot = not self.timeline and self.repair['status']!='activated'
        fit_clock = time if updating and not initial_snapshot else s['learn_end']
        if not updating or initial_snapshot:
            rows, receipt = released_window(self.history,fit_clock,s['window'],s['learn_start'],s['learn_end'],updating=False)
        fit_error = None
        fit_due = self.model is None or (updating and self.model.as_of_ < time)
        if fit_due:
            candidate = self._new_model()
            try:
                use_full = (self.repair['method']=='rolling_full_RA_GQ' if self.repair['status']=='activated'
                            else s.get('initial_full',False))
                if use_full:
                    from .product_ra import ReleasedRACache
                    prefix = self.history.loc[(self.history.period>=s['learn_start'])
                        & (self.history.period<fit_clock) & (self.history.release_period<=fit_clock)].copy()
                    prefix=prefix.loc[np.isfinite(prefix[XCOL+['action','next_state','next_w']].to_numpy(float)).all(axis=1)].copy()
                    if self.ra_cache is None or self.ra_cache.date not in {fit_clock-1,fit_clock}:
                        self.ra_cache=ReleasedRACache(prefix,fit_clock,s,self.equation)
                    else:
                        self.ra_cache.advance(prefix,fit_clock)
                    candidate=self.ra_cache.fit(self.feasible)
                    self.query_auxiliary=self.ra_cache.auxiliary.copy()
                    self.ra_audit={'method':'existing chronological prepare_rnd2_learner',
                        'provenance':'reconstructed_prefix_not_live_forecast',
                        'warmup_as_of':s['warmup_as_of'],'reconstruction':self.ra_cache.audits,
                        'continuation':'native auxiliary projection approximation; no sufficiency guarantee'}
                else:
                    self.query_auxiliary=None
                    fit_window = max(s['window'],fit_clock-s['learn_start']) if not updating or not self.timeline else s['window']
                    candidate.fit(rows,as_of=fit_clock,window=fit_window,feasible=self.feasible)
                self.model = candidate
            except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError) as error:
                fit_error = {'type':type(error).__name__,'message':str(error)}
                # Never query an obsolete model as though the current update succeeded.
                self.model = None
                self.query_auxiliary=None
                self.ra_audit=None
        if (self.model is not None and self.query_auxiliary is not None
                and not updating and time>s['learn_end']):
            previous=self.history.loc[(self.history.path_id==s['path']) & (self.history.period==time-1)
                & (self.history.release_period<=time)]
            if len(previous)==1 and np.isfinite(previous[XCOL+['action','next_state']].to_numpy(float)).all():
                row=previous.iloc[0]
                prediction=self.model.base_model_.transition_.predict(row[XCOL].to_numpy(float)[None],[row.action])[0,0]
                self.query_auxiliary=np.array([0.,0.,row.next_state-prediction])
            else:
                state=None
        q = None
        native = {'status':'unavailable','reason':'fit_unavailable' if self.model is None else 'input_unavailable'}
        auxiliary_available=self.query_auxiliary is None or np.isfinite(self.query_auxiliary).all()
        if self.model is not None and state is not None and np.isfinite(state).all() and auxiliary_available:
            model = self.model
            aux = None if self.query_auxiliary is None else self.query_auxiliary[None]
            explanation = model.explain_q_values(state[None],aux,feasible=self.feasible)
            q = model.q_values(state[None],aux,evaluation='bellman',feasible=self.feasible)[0]
            physical = model.model_.support_
            distances, _ = physical.search.kneighbors(physical.standardize(state[None]))
            native = clean(dict(status='measured',
                geometric_mask=physical.mask(state[None],self.grid)[0],
                action_memory=explanation['rows'][0]['same_memory_exists'],
                nearest_explanation=explanation['rows'][0],
                native_metric=dict(center=physical.center,scale=physical.scale,
                                   neighbor_radius=distances[0,-1],distance_limit=physical.distance_limit,
                                   boundary_margin=physical.distance_limit-distances[0,-1]),
                finite_direct_bellman_mask=np.isfinite(q),
                planning_continuation={'status':'numerical_native_projection','horizon':s['horizon'],
                    'data_qualification':'unavailable; no observational continuation error bound'},
                economic_value_interval='unavailable'))
        finite = q is not None and np.isfinite(q).any()
        recommendation = float(self.grid[np.argmax(q)]) if finite else None
        reward_action = recorded if observed_mode else recommendation
        reward_value = None
        if state is not None and np.isfinite(state).all() and reward_action is not None:
            inputs = dict(zip(XCOL,state));inputs['action']=reward_action
            value = np.asarray(self.equation({key:inputs[field] for key,field in s['reward']['inputs'].items()}))
            if not np.isfinite(value).all():
                raise ValueError('prescribed reward is nonfinite at the current input')
            reward_value = float(value)
        issued = None
        if not observed_mode:
            if recommendation is None or self.model is None or state is None:
                self.broken_estimated_path = True
                self.estimated = None
            else:
                issued = self.model.model_.transition_.predict(state[None],[recommendation])[0]
                self.estimated = issued.copy()
        status = ('input_unavailable' if state is None or not np.isfinite(state).all() else
                  'auxiliary_unavailable' if not auxiliary_available else
                  'fit_unavailable' if self.model is None else 'available' if finite else 'support_unavailable')
        fit_rows = None if self.model is None else self.model.model_._train_frame
        if self.model is not None:
            self.tracker.set_native_reference(self.model.model_.support_,self.model.as_of_)
        support = self.tracker.measure(support_rows,state)
        support['released_window'] = support_receipt
        matured=[];waiting=[]
        for forecast in self.pending_forecasts:
            released=self.history.loc[(self.history.path_id==s['path']) &
                (self.history.period==forecast['decision_date']) & (self.history.release_period<=time)]
            if len(released)==1 and np.isfinite(released.iloc[0].next_state):
                actual=float(released.iloc[0].next_state)
                matured.append({**forecast,'release_date':int(released.iloc[0].release_period),
                    'actual_next_state':actual,'residual':actual-forecast['predicted_next_state']})
            else:waiting.append(forecast)
        self.pending_forecasts=waiting
        issued_observed_forecast=None
        if observed_mode and self.model is not None and state is not None and np.isfinite(state).all() and recorded is not None and auxiliary_available:
            aux=None if self.query_auxiliary is None else self.query_auxiliary[None]
            prediction=self.model.model_.transition_.predict(state[None],[recorded],aux)[0,0]
            if np.isfinite(prediction):
                issued_observed_forecast=dict(decision_date=time,fit_as_of=self.model.as_of_,
                    recorded_action=recorded,predicted_next_state=float(prediction),
                    scope='issued prediction for logged action; diagnostic, not an intervention or economic certificate')
                self.pending_forecasts.append(issued_observed_forecast)
        fit_keys = set() if fit_rows is None else set(zip(fit_rows.path_id.astype(int),fit_rows.period.astype(int)))
        entry = clean(dict(date=time,status=status,fit_as_of=None if self.model is None else self.model.as_of_,
            fit_error=fit_error,window=receipt,training_digest=None if fit_rows is None else frame_digest(fit_rows),
            fit_rows=len(fit_keys),fit_entrants=sorted(fit_keys-self.last_update_keys),fit_exits=sorted(self.last_update_keys-fit_keys),
            query_state=state,recorded_action=recorded,recommended_action=recommendation,
            state_source='observed' if observed_mode else 'initial_observed' if not self.timeline else 'previously_issued_G_prediction',
            action_used_for_reward=reward_action,reward_value=reward_value,reward_digest=self.identity['reward'],
            auxiliary=self.query_auxiliary,auxiliary_audit=self.ra_audit,
            observed_action_monitoring={'released_issued_residuals':matured,
                'issued_forecast':issued_observed_forecast,'calibrated_alarm':None,
                'alarm_scope':'descriptive residual until an applicable calibrated monitor is supplied; source workflow reported separately'},
            recommendation_executed=False,issued_next_state=issued,
            action_values=None if q is None else [dict(action=a,q=v) for a,v in zip(self.grid,q)],
            support=support,native=native,detection=self._evidence_status(time),repair=self.repair,
            elapsed_seconds=perf_counter()-started))
        self.last_update_keys = fit_keys
        self.timeline.append(entry)
        self.next_date += 1
        return deepcopy(entry)

    def run(self):
        while self.next_date <= self.settings['end']:
            self.advance()
        return self.report()

    def report(self):
        return clean(dict(schema='pinc.panel_product.v1',identity=self.identity,
            configuration=self.settings,data_provenance=self.history.attrs,
            environment={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
            timeline=self.timeline,source_reports=self.source_reports,
            live_evidence=None if self.evidence is None else self.evidence.snapshot(),repair=self.repair,
            capabilities={'GQ':'native base GQ, direct Bellman',
                'detection_source':'existing certified finite-history workflow when its required inputs and premises are supplied',
                'repair':'explicit MODEL-conditioned rolling base or chronological full RA-GQ; no automatic success claim',
                'full_RA_panel_repair':'existing legacy full RA reconstruction; requires complete one-period histories, adequate warmup and common supported probes; future auxiliary transition remains approximate',
                'ordinary_panel_source_identification':'not_established',
                'counterfactual_value_intervals':'unavailable'}))

    def export(self, directory, *, compress=False):
        """New directory only; preserve earlier experiments and review snapshots."""
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=False)
        report = self.report()
        if compress:
            with gzip.open(directory/'report.json.gz','wt',encoding='utf-8',compresslevel=6) as handle:
                json.dump(report,handle,allow_nan=False,separators=(',',':'))
            pd.json_normalize(self.timeline,sep='.').to_csv(directory/'timeline.csv.gz',index=False,compression={'method':'gzip','compresslevel':6})
        else:
            (directory/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            pd.json_normalize(self.timeline,sep='.').to_csv(directory/'timeline.csv',index=False)
        (directory/'README.md').write_text(
            '# GQ tool run\n\nNative base GQ on released panel records; reward fixed. '
            'See report.json (or report.json.gz) for data/config/reward identity, vintages, source scope and missing qualifications. '
            'Finite fitted Q and geometry are not causal or economic recovery certificates.\n',encoding='utf-8')
        return directory
