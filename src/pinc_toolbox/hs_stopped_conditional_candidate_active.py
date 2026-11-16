"""Active-row ndarray implementation of the opt-in HS stopped score.

The support scan is time-major: only currently active rows are inspected, and
outcomes are gathered only for the rows that pass support at that time.
"""
import numpy as np
from .residual_energy import sequential_slope_energy
from .stopped_candidate_rank import stopped_candidate_rank
from .hs_stopped_conditional_candidate import score_hs_stopped_conditional_candidate as _fallback


def score_hs_stopped_conditional_candidate_active(*, evaluation_path, calibration_paths,
        reference_coefficients, candidate_F, candidate_d, candidate_covariance,
        residual_left_inverse, support, eta, alpha, context):
    # Preserve the established route for accessors, subclasses, and malformed
    # path containers. Do not coerce them to arrays: that could touch suffixes.
    if not isinstance(calibration_paths, (list, tuple)) or not calibration_paths:
        return _fallback(evaluation_path=evaluation_path, calibration_paths=calibration_paths,
            reference_coefficients=reference_coefficients, candidate_F=candidate_F,
            candidate_d=candidate_d, candidate_covariance=candidate_covariance,
            residual_left_inverse=residual_left_inverse, support=support, eta=eta,
            alpha=alpha, context=context)
    paths = [evaluation_path, *calibration_paths]
    if any(type(p) is not np.ndarray or p.ndim != 2 or p.shape[1:] != (2,) or
           p.dtype.kind != 'f' or p.shape[0] < 1 for p in paths):
        return _fallback(evaluation_path=evaluation_path, calibration_paths=calibration_paths,
            reference_coefficients=reference_coefficients, candidate_F=candidate_F,
            candidate_d=candidate_d, candidate_covariance=candidate_covariance,
            residual_left_inverse=residual_left_inverse, support=support, eta=eta,
            alpha=alpha, context=context)

    def array(x, shape, name):
        a = np.asarray(x, dtype=float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(name + ' has invalid dimensions or nonfinite values')
        return a
    f = np.asarray(candidate_F, dtype=float)
    if f.ndim != 3 or f.shape[1:] != (2, 2) or not len(f) or not np.isfinite(f).all():
        raise ValueError('candidate_F must be finite (H,2,2)')
    H = len(f)
    d = array(candidate_d, (H, 2), 'candidate_d')
    cov = array(candidate_covariance, (H, 2, 2), 'candidate_covariance')
    if not np.allclose(cov, cov.swapaxes(1, 2)) or np.min(np.linalg.eigvalsh(cov)) < -1e-12:
        raise ValueError('candidate covariance must be symmetric positive semidefinite')
    fr = array(reference_coefficients['F'], (2, 2), 'reference F')
    dr = array(reference_coefficients['d'], (2,), 'reference d')
    inv = array(residual_left_inverse, (2, 2), 'residual_left_inverse')
    low, high, width = [array(v, (2,), 'support') for v in support]
    if np.any(width <= 0) or np.any(high < low) or not np.isfinite(eta) or eta < 0:
        raise ValueError('invalid support or eta')

    def fail(reason, n=0):
        return dict(n=n, complete=False, valid=False, exit=None, reason=reason,
                    failed=True, xp=None, yp=None)

    def scalar_meta(path):
        out = _fallback(evaluation_path=evaluation_path, calibration_paths=[path],
            reference_coefficients=reference_coefficients, candidate_F=candidate_F,
            candidate_d=candidate_d, candidate_covariance=candidate_covariance,
            residual_left_inverse=residual_left_inverse, support=support, eta=eta,
            alpha=alpha, context=context)
        score = out['calibration_scores'][0]
        return dict(n=out['calibration_valid_count'][0],
            complete=out['calibration_complete'][0], valid=score is not None,
            exit=out['calibration_first_exit'][0],
            reason=out['calibration_failure_reason'][0], failed=score is None,
            xp=None, yp=None, score=score)

    # Read only the evaluation start for the global reference state.
    try:
        start = array(evaluation_path[0], (2,), 'path row')
    except (ValueError, TypeError, IndexError):
        start = None
    count = len(paths)
    meta = [None] * count
    predictors = [[] for _ in paths]
    outcomes = [[] for _ in paths]
    active = np.zeros(count, dtype=bool)
    # Initial rows are the only rows read before the chronological scan.
    starts = [None] * count
    for i, path in enumerate(paths):
        try:
            x = array(path[0], (2,), 'path row')
            starts[i] = x
            if start is None or not np.array_equal(x, start):
                meta[i] = fail('missing_or_mismatched_start')
            else:
                active[i] = True
        except (ValueError, TypeError, IndexError) as exc:
            meta[i] = fail('numeric_or_missing_supported_path:' + type(exc).__name__)

    for t in range(H):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        x = np.asarray([starts[i] if t == 0 else outcomes[i][-1] for i in ids])
        # Nonfinite current predictors and support arithmetic overflow are
        # exceptional per-row cases. Scalarize only affected paths.
        bad = ~np.isfinite(x).all(axis=1)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                dist = np.maximum(np.maximum((low-x)/width, (x-high)/width), 0).max(axis=1)
        except FloatingPointError:
            dist = np.zeros(len(ids))
            for j in range(len(ids)):
                if bad[j]:
                    continue
                try:
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        dist[j] = np.maximum(np.maximum((low-x[j])/width, (x[j]-high)/width), 0).max()
                except FloatingPointError:
                    bad[j] = True
        for j in np.flatnonzero(bad):
            i = ids[j]; meta[i] = scalar_meta(paths[i]); active[i] = False
        exited = (dist > eta) & ~bad
        for j in np.flatnonzero(exited):
            i = ids[j]; n = len(predictors[i])
            meta[i] = dict(n=n, complete=False, valid=n > 0, exit=t,
                reason='support_stop' if n else 'empty_supported_prefix', failed=False,
                xp=None, yp=None)
            active[i] = False
        passed = ~(bad | exited)
        if not passed.any():
            continue
        live_ids = ids[passed]
        live_x = x[passed]
        # Gather only supported outcomes, then perform one vectorized conversion
        # and finiteness check for the live rows.
        raw = []
        gathered = []
        for j, i in enumerate(live_ids):
            try:
                raw.append(paths[i][t+1]); gathered.append((j, i))
            except IndexError as exc:
                meta[i] = fail('numeric_or_missing_supported_path:IndexError', len(predictors[i]))
                active[i] = False
        if raw:
            try:
                next_rows = np.asarray(raw, dtype=float)
                if next_rows.shape != (len(raw), 2):
                    raise ValueError('path row has invalid dimensions or nonfinite values')
                finite_rows = np.isfinite(next_rows).all(axis=1)
            except (ValueError, TypeError, OverflowError):
                # This only occurs for irregular inputs; leave exact diagnosis to scalar paths.
                for _, i in gathered:
                    meta[i] = scalar_meta(paths[i]); active[i] = False
                continue
            for k, (j, i) in enumerate(gathered):
                if not finite_rows[k]:
                    meta[i] = scalar_meta(paths[i]); active[i] = False
                    continue
                predictors[i].append(live_x[j].copy())
                outcomes[i].append(next_rows[k])
                starts[i] = next_rows[k]

    for i in range(count):
        if meta[i] is None:
            n = len(predictors[i])
            meta[i] = dict(n=n, complete=n == H, valid=n > 0, exit=H,
                reason=None, failed=False, xp=np.asarray(predictors[i]), yp=np.asarray(outcomes[i]))
        elif 'score' not in meta[i] and meta[i].get('xp') is None and meta[i]['n'] > 0:
            # stopped/failed paths still retain their already-read prefix arrays
            meta[i]['xp'] = np.asarray(predictors[i]); meta[i]['yp'] = np.asarray(outcomes[i])

    residuals = np.zeros((count, H, 2), dtype=float)
    weights = np.zeros((count, H), dtype=float)
    eligible = np.zeros((count, H, 1), dtype=bool)
    scores = [m.get('score') for m in meta]
    for i, m in enumerate(meta):
        n = m['n']
        if not n or m['failed'] or 'score' in m:
            continue
        try:
            xp, yp = m['xp'], m['yp']
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                residual = (yp - np.einsum('tp,pq->tq', xp, fr.T) - dr) - (
                    np.einsum('tp,tqp->tq', xp, f[:n] - fr) + (d[:n] - dr))
            if not np.isfinite(residual).all():
                raise FloatingPointError('nonfinite prefix residual')
            residuals[i, :n] = residual; weights[i, :n] = xp[:, 1]
            eligible[i, :n, 0] = True
        except (ValueError, TypeError, IndexError, FloatingPointError) as exc:
            meta[i].update(scalar_meta(paths[i]))
            scores[i] = meta[i]['score']

    rows = [i for i,m in enumerate(meta) if m['n'] and not m['failed'] and 'score' not in m]
    if rows:
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                result = sequential_slope_energy(residuals[rows], weights[rows],
                    residual_left_inverse=inv, eligible=eligible[rows], return_trace=False)
            for j,i in enumerate(rows):
                val = float(result['scores'][j,0])
                if not np.isfinite(val): raise FloatingPointError('nonfinite prefix score')
                scores[i] = val
        except (ValueError, FloatingPointError):
            # Retry each row alone, so one overflow cannot alter peer results.
            for i in rows:
                n = meta[i]['n']
                try:
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        result = sequential_slope_energy(residuals[i:i+1,:n], weights[i:i+1,:n],
                            residual_left_inverse=inv, return_trace=False)
                    val = float(result['scores'][0,0])
                    if not np.isfinite(val): raise FloatingPointError('nonfinite prefix score')
                    scores[i] = val
                except (ValueError, TypeError, IndexError, FloatingPointError) as exc:
                    meta[i].update(scalar_meta(paths[i]))
                    scores[i] = meta[i]['score']

    obs = meta[0]; cal = meta[1:]; cs = scores[1:]
    out = stopped_candidate_rank(observed_score=scores[0],
        observed_full_window_complete=obs['complete'],
        observed_score_valid_for_rank=bool(obs['valid'] and scores[0] is not None),
        calibration_scores=cs,
        calibration_full_window_complete=[m['complete'] for m in cal],
        calibration_score_valid_for_rank=[bool(m['valid'] and s is not None) for m,s in zip(cal,cs)], alpha=alpha)
    out.update(context=context, statistic='sequential_slope_energy', coordinates=['DeltaY','Z'],
        candidate_F=f.tolist(), candidate_d=d.tolist(), candidate_covariance=cov.tolist(),
        source_start_state=None if start is None else start.tolist(), calibration_scores=cs,
        calibration_complete=[m['complete'] for m in cal], calibration_valid_count=[m['n'] for m in cal],
        calibration_first_exit=[m['exit'] for m in cal], calibration_failure_reason=[m['reason'] for m in cal],
        observed_valid_count=obs['n'], observed_first_exit=obs['exit'], observed_failure_reason=obs['reason'],
        horizon=H, eta=eta, effective_invalid_rank_score='+infinity', implementation_error=None)
    return out
