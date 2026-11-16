"""Exact represented-input finite-history monitor.

This adapter keeps the public clock and publication semantics of the float
monitor while making every support, moment, reference-error, and source-region
decision on the represented rational inputs.  It certifies only the supplied
finite construction under its declared MGF assumptions; it has no kappa, Q,
or real-DGP certification.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
import math

from .certified_construction import (
    _q,
    _q_matrix,
    _q_vector,
    _solve_fraction,
    _transpose,
    monitor_region,
)
from .source_arithmetic import source_arithmetic_allowance, validate_source_arithmetic_rule


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _display(value):
    exact = _q(value)
    try:
        converted = float(exact)
    except OverflowError:
        return None
    return converted if math.isfinite(converted) else None


def _display_list(values):
    return [_display(value) for value in values]


def _raw_row(row):
    if not isinstance(row, Mapping):
        raise ValueError("public row must be a mapping")
    return row


def _support(point, anchors, scale, eta):
    """Return exact support, stopping on the first exact witness."""
    minimum = None
    for anchor in anchors:
        distance = max((abs(x - a) / s for x, a, s in zip(point, anchor, scale)), default=Fraction(0))
        if distance <= eta:
            return True, distance
        if minimum is None or distance < minimum:
            minimum = distance
    return False, minimum


def _reference_errors(reference, directions):
    if reference.get('region_method') == 'joint_score':
        from .joint_score_reference import project_joint_reference
        projected = project_joint_reference(reference['joint_score_region'], directions)
        return [_q(row['minimum']['radius_exact']) for row in projected['rows']]
    V = _q_matrix(reference['V_exact'], 'reference.V_exact')
    c = _q_vector(reference['c_exact'], 'reference.c_exact')
    theta = _q_vector(reference['candidate_theta_exact'], 'reference.candidate_theta_exact')
    bounds = _q_vector(reference['score_bounds_exact'], 'reference.score_bounds_exact')
    directions = _q_matrix(directions, 'window.directions_exact')
    if len(V) != len(c) or any(len(row) != len(theta) for row in V):
        raise ValueError('stored reference dimensions do not agree')
    if len(bounds) != len(theta) or any(len(row) != len(theta) for row in directions):
        raise ValueError('stored reference bounds or directions have wrong dimension')
    d = [ci - sum((v * t for v, t in zip(row, theta)), Fraction(0)) for ci, row in zip(c, V)]
    solved = _solve_fraction(V, _transpose(directions)) if directions else []
    errors = []
    for index in range(len(directions)):
        z = [solved[row][index] for row in range(len(theta))]
        errors.append(abs(sum((zi * di for zi, di in zip(z, d)), Fraction(0))) +
                      sum((abs(zi) * bound for zi, bound in zip(z, bounds)), Fraction(0)))
    return errors


def _thin_classes(backend):
    return {
        name: {key: item[key] for key in ('status', 'description') if key in item}
        for name, item in backend['classes'].items()
    }


def monitor_history_certified(public_rows, initialization, spec, reference_spec, *, as_of,
                              skip_windows=(), region_only=False, prepared_cache=None):
    """Monitor finite windows with exact represented-input source regions."""
    if not isinstance(region_only, bool):
        raise ValueError('region_only must be a boolean')
    source_execution_mode = spec.get('source_execution_mode', 'exhaustive')
    if source_execution_mode not in ('exhaustive', 'lazy'):
        raise ValueError('source_execution_mode must be exhaustive or lazy')
    arithmetic_rule = validate_source_arithmetic_rule(spec.get('source_arithmetic_allowance_rule'))
    clock = _integer(as_of, 'as_of')
    if initialization['status'] != 'ready':
        return dict(
            status='not_ready', records=[],
            episode=None if region_only else dict(switch=False, model=False, latent=False),
            **({'source_evaluation': 'not_run'} if region_only else {}),
        )
    horizon = _integer(spec['horizon'], 'horizon', 1)
    windows = [tuple(_integer(t, 'relative window date', 1) for t in w) for w in spec['windows']]
    if not windows or len(windows) > horizon or any(not w or len(set(w)) != len(w) or max(w) > horizon for w in windows):
        raise ValueError('declare 1..H nonempty distinct-date windows within 1..H')
    m = _integer(spec['measurement_dimension'], 'measurement_dimension', 1)
    alpha_noise = _q(spec['alpha_noise'])
    alpha_ref = _q(reference_spec['alpha_reference'])
    if not 0 < alpha_noise < 1 or not 0 < alpha_ref < 1 or alpha_ref + alpha_noise >= 1:
        raise ValueError('reference and monitoring budgets must sum to less than one')
    start = _integer(initialization['monitoring_start'], 'monitoring_start')
    n = _integer(initialization['N'], 'N', 1)
    reference = initialization['reference']
    theta = _q_vector(reference['candidate_theta_exact'], 'candidate_theta_exact')
    caps = _q_vector(reference_spec['feature_bounds'], 'feature_bounds')
    if len(caps) != len(theta) or any(cap < 0 for cap in caps):
        raise ValueError('feature bounds must be nonnegative and match theta dimension')
    scale_raw = initialization.get('scale_exact') or reference.get('scale_exact')
    anchors_raw = initialization.get('anchors_exact') or reference.get('anchors_exact')
    if scale_raw is None or anchors_raw is None:
        raise ValueError('authoritative exact scale and anchors are required')
    scale = _q_vector(scale_raw, 'scale_exact')
    anchors = _q_matrix(anchors_raw, 'anchors_exact')
    if not anchors or len(scale) != len(anchors[0]):
        raise ValueError('scale and anchors dimensions do not agree')
    if any(value <= 0 for value in scale) or any(len(row) != len(scale) for row in anchors):
        raise ValueError('scale must be positive and anchors rectangular')
    anchor_widths = [max(row[j] for row in anchors) - min(row[j] for row in anchors)
                     for j in range(len(scale))]
    if any(width > scale[j] for j, width in enumerate(anchor_widths)):
        raise ValueError('scale must cover every anchor range')
    eta_raw = initialization.get('eta_exact', initialization.get('selected_eta_exact'))
    if eta_raw is None:
        raise ValueError('authoritative exact eta is required')
    eta = _q(eta_raw)
    if eta < 0:
        raise ValueError('eta must be nonnegative')
    sigma_u = _q(reference_spec['sigma_common'])
    sigma_v = _q(reference_spec['sigma_idiosyncratic'])
    loading_bound = _q(reference_spec['loading_bound'])
    latent_bound = _q(spec['latent_bound'])
    mechanism_bound = _q(spec['mechanism_bound'])
    if any(x < 0 for x in (sigma_u, sigma_v, loading_bound, latent_bound, mechanism_bound)):
        raise ValueError('noise/loading bounds must be nonnegative')
    family = spec['latent_family']
    if family not in ('rowwise', 'common_date'):
        raise ValueError('latent_family must be rowwise or common_date')
    onsets = [_integer(x, 'candidate onset') for x in spec['mechanism_onsets']]
    if not onsets or len(set(onsets)) != len(onsets):
        raise ValueError('mechanism_onsets must be a predeclared finite distinct list')

    dated = {}
    for raw in public_rows:
        row = _raw_row(raw)
        h = _integer(row['household'], 'household')
        date = _integer(row['period'], 'period')
        release = _integer(row['outcome_release'], 'outcome_release')
        if h >= n or date < start or date >= start + horizon or release > clock:
            continue
        if (date, h) in dated:
            raise ValueError('duplicate public date/household')
        if _integer(row['decision_time'], 'decision_time') != date or release != date + 1:
            raise ValueError('current implementation requires the synchronous declared clock')
        dated[date, h] = row

    domain = reference_spec.get('monitoring_domain')
    domain_lower = _q_vector(domain['lower'], 'monitoring_domain.lower') if domain is not None else None
    domain_upper = _q_vector(domain['upper'], 'monitoring_domain.upper') if domain is not None else None
    if domain is not None and (len(domain_lower) != len(scale) or len(domain_upper) != len(scale) or any(a > b for a, b in zip(domain_lower, domain_upper))):
        raise ValueError('monitoring_domain dimensions or bounds are invalid')
    cache_signature = (
        start, n, horizon, tuple(caps), tuple(scale), tuple(tuple(row) for row in anchors), eta,
        tuple(domain_lower) if domain_lower is not None else None,
        tuple(domain_upper) if domain_upper is not None else None,
        tuple(theta), m, loading_bound,
    )
    if prepared_cache is not None and not isinstance(prepared_cache, dict):
        raise ValueError('prepared_cache must be a mutable mapping')
    usable, stop = {}, None
    last_relative = min(horizon, max(0, clock - start))
    for relative in range(1, last_relative + 1):
        date = start + relative - 1
        block = [dated.get((date, h)) for h in range(n)]
        if any(row is None for row in block):
            stop = dict(relative_date=relative, decision_time=date, reason='incomplete_public_date',
                        timing='post_release_availability_abstention')
            break
        cached = prepared_cache.get(date) if prepared_cache is not None else None
        if (isinstance(cached, dict) and cached.get('signature') == cache_signature
                and cached.get('rows') == block):
            if cached.get('status') == 'stop':
                stop = deepcopy(cached['stop'])
                break
            usable[relative] = cached['prepared']
            continue
        prepared = []
        date_stop = None
        for row in block:
            for name in ('features_available_at', 'weights_available_at', 'source_loadings_available_at'):
                if _integer(row[name], name) > date:
                    raise ValueError(f'{name} must precede the whole date innovation')
            phi = _q_vector(row['features'], 'features')
            point = _q_vector(row['support_point'], 'support_point')
            if domain is not None:
                if len(point) != len(domain_lower):
                    raise ValueError('support_point dimension differs from monitoring_domain')
                if any(x < lo or x > hi for x, lo, hi in zip(point, domain_lower, domain_upper)):
                    date_stop = dict(relative_date=relative, decision_time=date, reason='monitoring_domain_exit', timing='rule_uses_only_pre_noise_inputs')
                    break
            weight = _q_vector(row['weights'], 'weights')
            values = [_q(row[key]) for key in ('outcome', 'noise_loading', 'latent_loading', 'mechanism_loading')]
            if len(phi) != len(theta) or len(point) != len(scale) or len(weight) != m:
                raise ValueError('row dimensions do not agree with the declaration')
            if any(abs(x) > 1 for x in weight):
                raise ValueError('weights or noise loading violate declared bounds')
            supported, distance = _support(point, anchors, scale, eta)
            if (not supported or any(abs(x) > cap for x, cap in zip(phi, caps)) or
                    abs(values[1]) > loading_bound):
                date_stop = dict(relative_date=relative, decision_time=date,
                            reason='predictable_domain_or_eta_exit',
                            timing='rule_uses_only_pre_noise_inputs', distance=str(distance))
                break
            prepared.append((row, phi, point, weight, values))
        if date_stop is not None:
            stop = date_stop
            if prepared_cache is not None:
                prepared_cache[date] = {'signature': cache_signature, 'rows': deepcopy(block),
                                        'status': 'stop', 'stop': deepcopy(date_stop)}
            break
        if prepared_cache is not None:
            prepared_cache[date] = {'signature': cache_signature, 'rows': deepcopy(block),
                                    'status': 'ok', 'prepared': prepared}
        usable[relative] = prepared

    output, episode = [], dict(switch=False, model=False, latent=False)
    eta_display = _display(eta)
    for index, window in enumerate(windows):
        if index in skip_windows:
            continue
        release = start + max(window)
        common = dict(window_index=index, relative_dates=list(window), as_of=clock,
                      decision_time=start + max(window) - 1, outcome_release=release,
                      monitoring_start=start, eta_fit_cutoff=initialization['eta_fit_cutoff'],
                      eta=eta_display, eta_exact=str(eta), N=n, planned_rows=n * len(window),
                      reference_region_method=reference.get('region_method', 'coordinate_score_box'))
        if release > clock:
            output.append(dict(**common, availability='not_yet_released', alarm=None, source_status='not_run'))
            continue
        if any(date not in usable for date in window):
            output.append(dict(**common, availability='unavailable', stop=stop, alarm=None, source_status='not_run'))
            continue
        selected = [item for date in window for item in usable[date]]
        rows = [item[0] for item in selected]
        features = [item[1] for item in selected]
        weights = [[item[3][j] for item in selected] for j in range(m)]
        outcomes = [item[4][0] for item in selected]
        dates = [item[0]['period'] for item in selected]
        denominator = n * len(window)
        latent_values = [item[4][2] for item in selected]
        mechanism_values = [item[4][3] for item in selected]
        try:
            region = monitor_region(
                features, outcomes, weights, theta, [[] for _ in selected],
                sigma_common=sigma_u, sigma_idiosyncratic=sigma_v,
                loading_bound=loading_bound, households=n, window_dates=len(window),
                measurement_dimension=m, horizon=horizon, alpha_noise=alpha_noise,
            )
            moments_exact = _q_vector(region['moments_exact'], 'moments_exact')
            noise_radius = _q(region['noise_radius_exact'])
            projection = _q_matrix(region['reference_projection_matrix_exact'], 'reference_projection_matrix_exact')
            reference_errors = _reference_errors(reference, projection)
            if len(reference_errors) != m:
                raise ValueError('stored reference directions do not match measurement dimension')
            radii_exact = [reference_errors[i] + noise_radius for i in range(m)]
            # monitor_region already computed exact B and m from the original rows;
            # only the source images are formed here without a dense identity matrix.
            rowwise = [[weights[j][i] * latent_values[i] / denominator for i in range(len(selected))]
                       for j in range(m)]
            if family == 'rowwise':
                latent_matrix = rowwise
            else:
                unique_dates = sorted(set(dates))
                latent_matrix = [[sum((rowwise[j][i] for i, date in enumerate(dates) if date == d), Fraction(0))
                                  for d in unique_dates] for j in range(m)]
            mechanism_matrices = {
                str(onset): [[sum((weights[j][i] * mechanism_values[i] *
                                   (dates[i] >= onset) for i in range(len(selected))), Fraction(0)) / denominator]
                             for j in range(m)] for onset in onsets
            }
            if arithmetic_rule is not None:
                arithmetic_exact = source_arithmetic_allowance(
                    weights, mechanism_values, latent_values,
                    latent_bound=latent_bound, mechanism_bound=mechanism_bound,
                    denominator=denominator)
                base_radii_exact = list(radii_exact)
                radii_exact = [base + extra for base, extra in zip(radii_exact, arithmetic_exact)]
            else:
                arithmetic_exact = None
                base_radii_exact = None
            radius_extra = ({
                'source_base_radius_exact': [str(x) for x in base_radii_exact],
                'source_arithmetic_allowance_exact': [str(x) for x in arithmetic_exact],
                'source_arithmetic_allowance_rule': arithmetic_rule,
            } if arithmetic_rule is not None else {})
            if region_only:
                source_spec = {
                    'moments': [str(x) for x in moments_exact],
                    'radii': [str(x) for x in radii_exact],
                    'latent_loading': [[str(x) for x in row] for row in latent_matrix],
                    'latent_bounds': [
                        [str(-latent_bound)] * len(latent_matrix[0]),
                        [str(latent_bound)] * len(latent_matrix[0]),
                    ],
                    'mechanism_loadings': {
                        key: [[str(x) for x in row] for row in value]
                        for key, value in mechanism_matrices.items()
                    },
                    'mechanism_bounds': {
                        str(onset): [[str(-mechanism_bound)], [str(mechanism_bound)]]
                        for onset in onsets
                    },
                }
                moments_display = _display_list(moments_exact)
                reference_display = _display_list(reference_errors)
                noise_display = _display(noise_radius)
                source_display = _display_list(radii_exact)
                projection_display = [[_display(x) for x in row] for row in projection]
                display_unavailable = (
                    any(value is None for value in moments_display + reference_display + source_display)
                    or noise_display is None or eta_display is None
                    or any(value is None for row in projection_display for value in row)
                )
                output.append(dict(
                    **common,
                    availability='available',
                    used_rows=denominator,
                    fixed_denominator=denominator,
                    alarm=None,
                    moments=moments_display,
                    moments_exact=[str(x) for x in moments_exact],
                    reference_error=reference_display,
                    reference_error_exact=[str(x) for x in reference_errors],
                    noise_radius=noise_display,
                    noise_radius_exact=str(noise_radius),
                    source_radius=source_display,
                    source_radius_exact=[str(x) for x in radii_exact],
                    reference_projection=projection_display,
                    reference_projection_exact=[[str(x) for x in row] for row in projection],
                    source_latent_loading=[[str(x) for x in row] for row in latent_matrix],
                    source_mechanism_loadings=source_spec['mechanism_loadings'],
                    source_spec=source_spec,
                    backend=None,
                    router=None,
                    source_status='not_run',
                    source_target=None,
                    model_required=None,
                    latent_required=None,
                    display_status='unavailable' if display_unavailable else 'available',
                    display_unavailable=display_unavailable,
                    numerical_certification='exact represented-input region prepared; source evaluation not run',
                    e_eta=None,
                    D_L_minus_M=None,
                    **radius_extra,
                ))
                continue
            from .certified_source_backend import diagnose_certified_sources
            from .diagnosis import diagnose_sources
            backend = diagnose_certified_sources(
                moments_exact, radii_exact, latent_matrix,
                ([-latent_bound] * len(latent_matrix[0]),
                 [latent_bound] * len(latent_matrix[0])), mechanism_matrices,
                {str(onset): ([-mechanism_bound], [mechanism_bound]) for onset in onsets},
                execution_mode=source_execution_mode,
            )
        except (MemoryError, OverflowError) as error:
            output.append(dict(**common, availability='unavailable', reason='resource_exhaustion',
                               reason_detail=type(error).__name__, alarm=None, source_status='not_run',
                               fixed_denominator=denominator))
            continue
        classes = backend['classes']
        thin = _thin_classes(backend)
        router = diagnose_sources(lambda name: thin[name], alarm={'release': release,
                                                                    'detected': classes['C0']['status'] == 'excluded'})
        alarm = classes['C0']['status'] == 'excluded'
        model = classes['CL']['status'] == 'excluded' and classes['CMIX']['status'] == 'compatible'
        latent_required = classes['CM']['status'] == 'excluded' and classes['CMIX']['status'] == 'compatible'
        episode['switch'] |= alarm
        episode['model'] |= model
        episode['latent'] |= latent_required
        moments_display = [_display(x) for x in moments_exact]
        reference_display = [_display(x) for x in reference_errors]
        noise_display = _display(noise_radius)
        source_display = [_display(x) for x in radii_exact]
        projection_display = [[_display(x) for x in row] for row in projection]
        display_unavailable = (
            any(value is None for value in moments_display + reference_display + source_display)
            or noise_display is None or eta_display is None
            or any(value is None for row in projection_display for value in row)
        )
        output.append(dict(**common, availability='available', used_rows=denominator,
                           fixed_denominator=denominator, alarm=alarm,
                           moments=moments_display, moments_exact=[str(x) for x in moments_exact],
                           reference_error=reference_display,
                           reference_error_exact=[str(x) for x in reference_errors],
                           noise_radius=noise_display, noise_radius_exact=str(noise_radius),
                           source_radius=source_display,
                           source_radius_exact=[str(x) for x in radii_exact],
                           reference_projection=projection_display,
                           reference_projection_exact=[[str(x) for x in row] for row in projection],
                           source_latent_loading=[[str(x) for x in row] for row in latent_matrix],
                           source_mechanism_loadings={k: [[str(x) for x in row] for row in value]
                                                      for k, value in mechanism_matrices.items()},
                           backend=backend, router=router,
                           source_status={k: v['status'] for k, v in classes.items()},
                           source_target='component absence over all weighted rows of this window',
                           model_required=model, latent_required=latent_required,
                           display_status='unavailable' if display_unavailable else 'available',
                           display_unavailable=display_unavailable,
                           numerical_certification='exact represented-input region under declared MGF assumptions',
                           e_eta=None, D_L_minus_M=None, **radius_extra))
    result = dict(status='regions_prepared' if region_only else 'monitoring', records=output, publication_stop=stop,
                predictable_stop=stop if stop and stop['reason'] in ('predictable_domain_or_eta_exit', 'monitoring_domain_exit') else None,
                episode=None if region_only else episode,
                type_I_budget=_display(alpha_ref + alpha_noise),
                type_I_budget_exact=str(alpha_ref + alpha_noise), alpha_reference=_display(alpha_ref),
                alpha_reference_exact=str(alpha_ref), alpha_noise=_display(alpha_noise),
                alpha_noise_exact=str(alpha_noise),
                type_I_scope='unconditional full finite history, under declared model/MGF and exact represented-input certificates',
                numerical_guarantee='exact represented-input construction only; upstream assumptions and real-DGP conversion are outside scope',
                region='exact raw rational outer approximation of joint parameter-score and noise region',
                horizon=horizon, posterior_source_probability=None, action_Q_certification='unavailable')
    if region_only:
        result['source_evaluation'] = 'not_run'
    return result


__all__ = ['monitor_history_certified']
