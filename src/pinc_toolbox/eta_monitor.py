"""Finite public-panel initialization, fixed eta, and source measurements.

Restricted candidate implementation of PROOFS.md Lemma C/Theorem D. The public
JSON uses a synchronous clock: period=decision_time, outcome_release=period+1.
All statistical statements maintain known block-MGF bounds and a correct
stable reference while initialization is active. No action-Q error is supplied.
"""
from __future__ import annotations
from copy import deepcopy
from collections.abc import Mapping
from typing import Any
import math
import numpy as np


_PUBLIC_FIELDS = frozenset((
    'household', 'period', 'decision_time', 'outcome_release',
    'features_available_at', 'features', 'outcome', 'noise_loading',
    'support_point', 'weights', 'weights_available_at',
    'source_loadings_available_at', 'latent_loading', 'mechanism_loading',
))


def _public_row(row):
    if not isinstance(row, Mapping):
        raise ValueError('public row must be a mapping')
    return {key: row[key] for key in _PUBLIC_FIELDS if key in row}


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _vector(value, name, size=None):
    out = np.asarray(value, dtype=float)
    if out.ndim != 1 or not np.isfinite(out).all() or (size is not None and len(out) != size):
        raise ValueError(f'{name} must be a finite vector of the declared size')
    return out


def _probability(value, name):
    out = float(value)
    if not math.isfinite(out) or not 0 < out < 1:
        raise ValueError(f'{name} must lie in (0,1)')
    return out


def _distance(points, anchors, scale):
    return np.max(np.abs(points[:, None, :] - anchors[None, :, :]) / scale, axis=2).min(axis=1)


def _reference_error(directions, reference):
    return np.abs(directions @ np.asarray(reference['gram_inverse'])) @ np.asarray(reference['score_bounds'])


def select_eta(public_rows, spec, *, as_of):
    """First completed qualified prefix, then the smallest qualifying eta.

    ``spec`` declares prefixes, bounds/noise, eta_grid, scale_floor, tasks
    (support_point/direction/max_error), and required_task_count before data.
    The fixed denominator is the full task list. Support is nearest-record
    standardized max distance in supplied physical/action coordinates.
    """
    from .panel_reference import fit_prefix
    clock = _integer(as_of, 'as_of')
    public_rows = [_public_row(row) for row in public_rows]
    prefixes = [(_integer(x[0], 'N', 1), _integer(x[1], 'T', 1)) for x in spec['prefixes']]
    if not prefixes or len(set(prefixes)) != len(prefixes):
        raise ValueError('prefixes must be nonempty and distinct')
    eta = sorted(set(float(x) for x in spec['eta_grid']))
    if not eta or any(not math.isfinite(x) or x < 0 for x in eta):
        raise ValueError('eta_grid must be finite and nonnegative')
    bounds = _vector(spec['feature_bounds'], 'feature_bounds')
    if not len(bounds) or np.any(bounds < 0):
        raise ValueError('feature_bounds must be nonnegative and nonempty')
    floor = _vector(spec['scale_floor'], 'scale_floor')
    if not len(floor) or np.any(floor <= 0):
        raise ValueError('scale_floor must be strictly positive')
    tasks = spec['tasks']
    if not tasks:
        raise ValueError('a predeclared nonempty task list is required')
    points = np.asarray([_vector(t['support_point'], 'task.support_point', len(floor)) for t in tasks])
    directions = np.asarray([_vector(t['direction'], 'task.direction', len(bounds)) for t in tasks])
    tolerances = _vector([t['max_error'] for t in tasks], 'task.max_error', len(tasks))
    if np.any(tolerances < 0):
        raise ValueError('task max_error must be nonnegative')
    required = _integer(spec['required_task_count'], 'required_task_count', 1)
    if required > len(tasks):
        raise ValueError('required_task_count exceeds the fixed task denominator')
    alpha = _probability(spec['alpha_reference'], 'alpha_reference')
    diagnostics = []
    # Stable sort uses declaration order for ties across household counts.
    for n, t in sorted(prefixes, key=lambda x: x[1]):
        if t > clock:
            continue
        reference = fit_prefix(public_rows, households=n, periods=t, total_prefixes=len(prefixes),
                               feature_bounds=bounds.tolist(), sigma_common=spec['sigma_common'],
                               sigma_idiosyncratic=spec['sigma_idiosyncratic'],
                               loading_bound=spec['loading_bound'], alpha=alpha, as_of=clock)
        if reference['status'] != 'ready':
            diagnostics.append(dict(N=n, T=t, status='not_ready', reasons=reference['reasons'], eta=None))
            continue
        rows = reference['selected_rows']
        anchors = np.asarray([_vector(r['support_point'], 'support_point', len(floor)) for r in rows])
        scale = np.maximum(np.ptp(anchors, axis=0), floor)
        distances = _distance(points, anchors, scale)
        error = _reference_error(directions, reference)
        if not np.isfinite(error).all():
            raise ArithmeticError('nonfinite projected reference error')
        selected_eta = None
        for candidate in eta:
            supported = distances <= candidate
            precise = error <= tolerances
            qualified = supported & precise
            detail = [dict(task=i, distance=float(distances[i]), supported=bool(supported[i]),
                           reference_error=float(error[i]), max_error=float(tolerances[i]),
                           precise=bool(precise[i]), qualified=bool(qualified[i]),
                           reasons=([] if supported[i] else ['outside_eta_support']) +
                                   ([] if precise[i] else ['reference_precision_insufficient'])) for i in range(len(tasks))]
            ready = int(qualified.sum()) >= required
            diagnostics.append(dict(N=n, T=t, eta=candidate, status='ready' if ready else 'not_ready',
                                    task_denominator=len(tasks), required_task_count=required,
                                    qualified_tasks=int(qualified.sum()), tasks=detail,
                                    reasons=[] if ready else ['required_tasks_not_qualified']))
            if ready and selected_eta is None:
                selected_eta = candidate
        if selected_eta is not None:
            # Stop here. Never fit or inspect future initialization prefixes.
            # Strict integer-clock convention: cutoff T is BEFORE start T+1.
            # Period T is deliberately unused; no same-day subclock is claimed.
            return dict(status='ready', N=n, T=t, eta=selected_eta, eta_fit_cutoff=t,
                        monitoring_start=t+1, reference=reference,
                        scale=scale.tolist(), anchors=anchors.tolist(), candidates=diagnostics,
                        task_denominator=len(tasks), distance_rule='nearest_joint_record_scaled_Linf',
                        scale_rule='reference_range_with_predeclared_positive_floor',
                        fixed_after_start=['eta', 'scale', 'anchors', 'reference', 'tasks'],
                        task_scope='linear_reference_projection_and_observable_geometry_only',
                        e_eta=None, D_L_minus_M=None, action_Q_certification='unavailable')
    return dict(status='not_ready', reasons=['no_completed_qualified_prefix'], candidates=diagnostics,
                eta=None, monitoring_start=None, eta_fit_cutoff=None,
                task_scope='linear_reference_projection_and_observable_geometry_only',
                e_eta=None, D_L_minus_M=None, action_Q_certification='unavailable')


def monitor_history(public_rows, initialization, spec, reference_spec, *, as_of, skip_windows=()):
    """Evaluate predeclared finite windows using a fixed initialized reference.

    Eta suppresses publication permanently at the first out-of-support row.
    No row-selection renormalization occurs. Source images are computed from
    the same full window and weights as its actual residual moments.
    """
    from .source_backend import diagnose_linear_sources, router_evidence
    from .diagnosis import diagnose_sources
    clock = _integer(as_of, 'as_of')
    if initialization['status'] != 'ready':
        return dict(status='not_ready', records=[], episode=dict(switch=False, model=False, latent=False))
    horizon = _integer(spec['horizon'], 'horizon', 1)
    windows = [tuple(_integer(t, 'relative window date', 1) for t in w) for w in spec['windows']]
    if not windows or len(windows) > horizon or any(not w or len(set(w)) != len(w) or max(w) > horizon for w in windows):
        raise ValueError('declare 1..H nonempty distinct-date windows within 1..H')
    m = _integer(spec['measurement_dimension'], 'measurement_dimension', 1)
    alpha_noise = _probability(spec['alpha_noise'], 'alpha_noise')
    alpha_ref = _probability(reference_spec['alpha_reference'], 'alpha_reference')
    if alpha_ref + alpha_noise >= 1:
        raise ValueError('reference and monitoring budgets must sum to less than one')
    start, n = initialization['monitoring_start'], initialization['N']
    reference = initialization['reference']
    theta = _vector(reference['theta_hat'], 'theta_hat')
    bounds = _vector(reference_spec['feature_bounds'], 'feature_bounds', len(theta))
    scale = _vector(initialization['scale'], 'scale')
    anchors = np.asarray(initialization['anchors'], dtype=float)
    sigma_u, sigma_v, load_bound = (float(reference_spec[k]) for k in ('sigma_common', 'sigma_idiosyncratic', 'loading_bound'))
    if any(not math.isfinite(x) or x < 0 for x in (sigma_u, sigma_v, load_bound)):
        raise ValueError('known noise proxies and loading bound must be finite and nonnegative')
    latent_bound, mechanism_bound = float(spec['latent_bound']), float(spec['mechanism_bound'])
    if any(not math.isfinite(x) or x < 0 for x in (latent_bound, mechanism_bound)):
        raise ValueError('source amplitude bounds must be finite and nonnegative')
    family = spec['latent_family']
    if family not in ('rowwise', 'common_date'):
        raise ValueError('latent_family must be rowwise or common_date')
    onsets = [_integer(x, 'candidate onset') for x in spec['mechanism_onsets']]
    if not onsets or len(set(onsets)) != len(onsets):
        raise ValueError('mechanism_onsets must be a predeclared finite distinct list')
    # Index clocks only; future outcomes/features are not inspected.
    dated = {}
    for row in public_rows:
        h, date = _integer(row['household'], 'household'), _integer(row['period'], 'period')
        release = _integer(row['outcome_release'], 'outcome_release')
        if h >= n or date < start or date >= start+horizon or release > clock:
            continue
        if (date, h) in dated:
            raise ValueError('duplicate public date/household')
        if _integer(row['decision_time'], 'decision_time') != date or release != date+1:
            raise ValueError('current implementation requires the synchronous declared clock')
        dated[date, h] = row
    usable, stop = {}, None
    last_relative = min(horizon, max(0, clock-start))
    for relative in range(1, last_relative+1):
        date = start+relative-1
        block = [dated.get((date, h)) for h in range(n)]
        if any(r is None for r in block):
            stop = dict(relative_date=relative, decision_time=date, reason='incomplete_public_date',
                        timing='post_release_availability_abstention')
            break
        prepared = []
        for row in block:
            for name in ('features_available_at', 'weights_available_at', 'source_loadings_available_at'):
                if _integer(row[name], name) > date:
                    raise ValueError(f'{name} must precede the whole date innovation')
            phi = _vector(row['features'], 'features', len(theta))
            point = _vector(row['support_point'], 'support_point', len(scale))
            weight = _vector(row['weights'], 'weights', m)
            values = [float(row[k]) for k in ('outcome', 'noise_loading', 'latent_loading', 'mechanism_loading')]
            if not all(math.isfinite(x) for x in values) or np.any(np.abs(weight) > 1):
                raise ValueError('outcome/loadings must be finite and weights bounded by one')
            distance = float(_distance(point[None, :], anchors, scale)[0])
            if np.any(np.abs(phi) > bounds) or abs(values[1]) > load_bound or distance > initialization['eta']:
                stop = dict(relative_date=relative, decision_time=date, reason='predictable_domain_or_eta_exit',
                            timing='rule_uses_only_pre_noise_inputs', distance=distance)
                break
            prepared.append((row, phi, weight, values))
        if stop is not None:
            break
        usable[relative] = prepared
    output, episode = [], dict(switch=False, model=False, latent=False)
    for index, window in enumerate(windows):
        if index in skip_windows:
            continue
        release = start+max(window)
        common = dict(window_index=index, relative_dates=list(window), as_of=clock,
                      decision_time=start+max(window)-1, outcome_release=release,
                      monitoring_start=start, eta_fit_cutoff=initialization['eta_fit_cutoff'],
                      eta=initialization['eta'], N=n, planned_rows=n*len(window))
        if release > clock:
            output.append(dict(**common, availability='not_yet_released', alarm=None, source_status='not_run'))
            continue
        if any(date not in usable for date in window):
            output.append(dict(**common, availability='unavailable', stop=stop, alarm=None, source_status='not_run'))
            continue
        selected = [item for date in window for item in usable[date]]
        phi = np.asarray([item[1] for item in selected])
        weight = np.asarray([item[2] for item in selected]).T
        y = np.asarray([item[3][0] for item in selected])
        latent = np.asarray([item[3][2] for item in selected])
        mechanism = np.asarray([item[3][3] for item in selected])
        dates = np.asarray([item[0]['period'] for item in selected])
        denominator = n*len(window)
        weighted = weight/denominator
        moments = weighted @ (y-phi@theta)
        projection = weighted@phi
        reference_error = _reference_error(projection, reference)
        noise_radius = math.sqrt(2*(sigma_u*sigma_u*load_bound*load_bound+sigma_v*sigma_v/n)*
                                 math.log(2*m*horizon/alpha_noise)/len(window))
        radii = reference_error+noise_radius
        if not np.isfinite(radii).all() or not np.isfinite(moments).all():
            raise ArithmeticError('nonfinite region construction')
        if np.any(radii <= 0):
            output.append(dict(**common, availability='unavailable', reason='zero_radius_not_supported_by_box_backend',
                               alarm=None, source_status='not_run', moments=moments.tolist(),
                               reference_error=reference_error.tolist(), noise_radius=noise_radius))
            continue
        row_latent = weighted*latent[None, :]
        if family == 'rowwise':
            latent_matrix = row_latent
        else:
            latent_matrix = np.column_stack([row_latent[:, dates==d].sum(axis=1) for d in sorted(set(dates.tolist()))])
        mechanism_matrices = {str(onset): (weighted@(mechanism*(dates>=onset)))[:, None] for onset in onsets}
        latent_box = ([-latent_bound]*latent_matrix.shape[1], [latent_bound]*latent_matrix.shape[1])
        solver_tolerance = float(spec.get('solver_tolerance', 1e-8))
        backend = diagnose_linear_sources(moments, radii, latent_matrix, latent_box,
                                          mechanism_matrices, ([-mechanism_bound], [mechanism_bound]),
                                          tolerance=solver_tolerance)
        thin = router_evidence(backend)
        alarm = backend['classes']['C0']['status'] == 'excluded'
        router = diagnose_sources(thin.__getitem__, alarm={'release': release, 'detected': alarm})
        classes = backend['classes']
        model = classes['CL']['status'] == 'excluded' and classes['CMIX']['status'] == 'compatible'
        latent_required = classes['CM']['status'] == 'excluded' and classes['CMIX']['status'] == 'compatible'
        episode['switch'] |= alarm
        episode['model'] |= model
        episode['latent'] |= latent_required
        output.append(dict(**common, availability='available', used_rows=denominator, fixed_denominator=denominator,
                           alarm=alarm, moments=moments.tolist(), reference_error=reference_error.tolist(),
                           noise_radius=noise_radius, source_radius=radii.tolist(),
                           source_decision_radius=((1+solver_tolerance)*radii).tolist(),
                           reference_projection=projection.tolist(),
                           source_latent_loading=latent_matrix.tolist(),
                           source_mechanism_loadings={k:v.tolist() for k,v in mechanism_matrices.items()},
                           backend=backend, router=router,
                           source_status={k:v['status'] for k,v in classes.items()},
                           source_target='component absence over all weighted rows of this window',
                           model_required=model, latent_required=latent_required,
                           numerical_certification='floating_point_not_outward_rounded',
                           e_eta=None, D_L_minus_M=None))
    return dict(status='monitoring', records=output, publication_stop=stop,
                predictable_stop=stop if stop and stop['reason']=='predictable_domain_or_eta_exit' else None,
                episode=episode,
                type_I_budget=alpha_ref+alpha_noise, alpha_reference=alpha_ref, alpha_noise=alpha_noise,
                type_I_scope='unconditional full finite history, under declared model/MGF and sound numerical certificates',
                numerical_guarantee='whole pipeline floating arithmetic not rigorously enclosed',
                region='rectangle outer approximation of joint parameter-score and noise region',
                horizon=horizon, posterior_source_probability=None, action_Q_certification='unavailable')


def run_public_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-compatible public raw-history entry, also used by Dynare wrapper."""
    if request.get('operation') != 'eta_monitor':
        raise ValueError('eta_monitor operation required')
    if 'release_events' in request:
        session = PublicEtaMonitor(request['reference'], request['monitor'])
        for event in request['release_events']:
            session.update(event['public_rows'], as_of=event['as_of'])
        return dict(operation='eta_monitor', result=session.snapshot(),
                    implementation_status='candidate_numerically_unvalidated')
    rows = request['public_rows']
    initialization = select_eta(rows, request['reference'], as_of=request['as_of'])
    monitoring = monitor_history(rows, initialization, request['monitor'], request['reference'], as_of=request['as_of'])
    return dict(operation='eta_monitor', implementation_status='candidate_numerically_unvalidated',
                initialization=initialization, monitoring=monitoring,
                maintained_assumptions=['predeclared tasks/domain/prefixes/windows/source family',
                    'stable linear reference until deployment', 'known conditional block-subGaussian bounds',
                    'features/weights/source loadings predictable before whole-date innovation'],
                interpretation='source compatibility/exclusion, not hidden-state recovery or economic action certification')


class PublicEtaMonitor:
    """Append-only released-history session with fixed declared program.

    The first successful initialization is retained. Subsequent releases never
    reselect eta or rewrite published window records. Errors are transactional:
    callers may retry a failed update with the same public rows.
    """
    def __init__(self, reference_spec, monitor_spec):
        self._reference_spec = deepcopy(reference_spec)
        self._monitor_spec = deepcopy(monitor_spec)
        self._rows = {}
        self._initialization = None
        self._published = {}
        self._as_of = -1
        self._last = None

    def update(self, rows, *, as_of):
        clock = _integer(as_of, 'as_of')
        if clock < self._as_of:
            raise ValueError('release clock cannot move backwards')
        staged = deepcopy(self._rows)
        for row in rows:
            row = _public_row(row)
            key = (_integer(row['period'], 'period'), _integer(row['household'], 'household'))
            if _integer(row['outcome_release'], 'outcome_release') > clock:
                raise ValueError('session accepts only released rows')
            if key in staged and staged[key] != row:
                raise ValueError('cannot revise a released public row')
            if key not in staged and row['outcome_release'] <= self._as_of:
                raise ValueError('late backfill would revise an earlier synchronous history')
            staged[key] = deepcopy(row)
        history = list(staged.values())
        initialization = deepcopy(self._initialization)
        if initialization is None:
            trial = select_eta(history, self._reference_spec, as_of=clock)
            if trial['status'] == 'ready':
                initialization = trial
            else:
                result = dict(initialization=trial, monitoring=dict(status='not_ready', records=[]))
                self._rows, self._as_of, self._last = staged, clock, result
                return deepcopy(result)
        # Previously published windows remain stored verbatim. Recalculated
        # outputs are not substituted for them; all raw rows/specs are fixed.
        monitoring = monitor_history(history, initialization, self._monitor_spec,
                                     self._reference_spec, as_of=clock,
                                     skip_windows=set(self._published))
        published = deepcopy(self._published)
        pending = []
        for record in monitoring['records']:
            # Unavailability is final under this synchronous no-backfill policy.
            if record['availability'] != 'not_yet_released':
                published[record['window_index']] = deepcopy(record)
            else:
                pending.append(record)
        monitoring['records'] = [published[k] for k in sorted(published)] + pending
        monitoring['episode'] = dict(
            switch=any(r.get('alarm') is True for r in published.values()),
            model=any(r.get('model_required') is True for r in published.values()),
            latent=any(r.get('latent_required') is True for r in published.values()))
        result = dict(initialization=initialization, monitoring=monitoring, as_of=clock)
        self._rows, self._as_of, self._initialization = staged, clock, initialization
        self._published, self._last = published, result
        return deepcopy(result)

    def snapshot(self):
        return deepcopy(self._last)
