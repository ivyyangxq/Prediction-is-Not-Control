"""Batched opt-in HS stopped-score route with prefix-only path reads.

Only outcomes whose predictors pass support are read. Residuals are formed only
for those prefixes; zero padding is paired with false eligibility in the shared
sequential energy kernel.
"""
import numpy as np
from .residual_energy import sequential_slope_energy
from .stopped_candidate_rank import stopped_candidate_rank


def score_hs_stopped_conditional_candidate_batch(*, evaluation_path, calibration_paths,
        reference_coefficients, candidate_F, candidate_d, candidate_covariance,
        residual_left_inverse, support, eta, alpha, context):
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
    if not isinstance(calibration_paths, (list, tuple)) or not calibration_paths:
        raise ValueError('supply every planned CAL slot')

    def row(path, t):
        return array(path[t], (2,), 'path row')

    try:
        start = row(evaluation_path, 0)
    except (ValueError, TypeError, IndexError):
        start = None

    paths = [evaluation_path, *calibration_paths]
    # Metadata records semantic path status independently from numeric score.
    meta = []
    prefixes = []
    for path in paths:
        n = 0; first_exit = H; reason = None; failure = None
        predictors, outcomes = [], []
        try:
            x = row(path, 0)
            if start is None or not np.array_equal(x, start):
                meta.append(dict(n=0, complete=False, valid=False, exit=None,
                                 reason='missing_or_mismatched_start', failed=True))
                prefixes.append(None)
                continue
            for t in range(H):
                # This check intentionally precedes any access to path[t+1].
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    distance = np.maximum(np.maximum((low-x)/width, (x-high)/width), 0).max()
                if distance > eta:
                    first_exit = t
                    reason = 'support_stop' if n else 'empty_supported_prefix'
                    break
                xn = row(path, t + 1)
                predictors.append(x); outcomes.append(xn); n += 1; x = xn
            if n:
                xp, yp = np.asarray(predictors), np.asarray(outcomes)
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    # Preserve the reference-residual then candidate-adjustment order.
                    residual = (yp - np.einsum('tp,pq->tq', xp, fr.T) - dr) - (
                        np.einsum('tp,tqp->tq', xp, f[:n] - fr) + (d[:n] - dr))
                if not np.isfinite(residual).all():
                    raise FloatingPointError('nonfinite prefix residual')
                prefixes.append((xp, residual))
                meta.append(dict(n=n, complete=n == H, valid=True, exit=first_exit,
                                 reason=reason, failed=False))
            else:
                prefixes.append((np.empty((0, 2)), np.empty((0, 2))))
                meta.append(dict(n=0, complete=False, valid=False, exit=first_exit,
                                 reason=reason, failed=False))
        except (ValueError, TypeError, IndexError, FloatingPointError) as exc:
            prefixes.append(None)
            meta.append(dict(n=n, complete=False, valid=False, exit=None,
                             reason='numeric_or_missing_supported_path:' + type(exc).__name__, failed=True))

    # Build one H-wide batch. Padding has no residual calculation and is marked
    # unsupported, so sequential_slope_energy only consumes each real prefix.
    count = len(paths)
    residuals = np.zeros((count, H, 2), dtype=float)
    weights = np.zeros((count, H), dtype=float)
    eligible = np.zeros((count, H, 1), dtype=bool)
    batch_rows = []
    for i, (item, m) in enumerate(zip(prefixes, meta)):
        if item is None or m['n'] == 0:
            continue
        xp, residual = item
        residuals[i, :m['n']] = residual
        weights[i, :m['n']] = xp[:, 1]
        eligible[i, :m['n'], 0] = True
        batch_rows.append(i)

    scores = [None] * count
    if batch_rows:
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                result = sequential_slope_energy(residuals[batch_rows], weights[batch_rows],
                    residual_left_inverse=inv, eligible=eligible[batch_rows], return_trace=False)
            for j, i in enumerate(batch_rows):
                value = float(result['scores'][j, 0])
                if not np.isfinite(value):
                    raise FloatingPointError('nonfinite prefix score')
                scores[i] = value
        except (ValueError, FloatingPointError):
            # Kernel accumulation overflow can be path-specific. Re-run rows in
            # isolation so a single failed path cannot invalidate its peers.
            for i in batch_rows:
                try:
                    item = prefixes[i]; n = meta[i]['n']
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        one = sequential_slope_energy(item[1][None], item[0][None, :, 1],
                            residual_left_inverse=inv, return_trace=False)
                    value = float(one['scores'][0, 0])
                    if not np.isfinite(value):
                        raise FloatingPointError('nonfinite prefix score')
                    scores[i] = value
                except (ValueError, TypeError, IndexError, FloatingPointError) as exc:
                    meta[i].update(valid=False, complete=False, exit=None,
                        reason='numeric_or_missing_supported_path:' + type(exc).__name__, failed=True)
                    scores[i] = None

    # None is the strict-JSON sentinel for effective +infinity.
    # Empty and numeric failures remain separately identified in failure_reason.
    obs_meta = meta[0]
    cal = meta[1:]
    cal_scores = scores[1:]
    out = stopped_candidate_rank(observed_score=scores[0],
        observed_full_window_complete=obs_meta['complete'],
        observed_score_valid_for_rank=bool(obs_meta['valid'] and scores[0] is not None),
        calibration_scores=cal_scores,
        calibration_full_window_complete=[m['complete'] for m in cal],
        calibration_score_valid_for_rank=[bool(m['valid'] and s is not None) for m, s in zip(cal, cal_scores)],
        alpha=alpha)
    out.update(context=context, statistic='sequential_slope_energy', coordinates=['DeltaY', 'Z'],
        candidate_F=f.tolist(), candidate_d=d.tolist(), candidate_covariance=cov.tolist(),
        source_start_state=None if start is None else start.tolist(),
        calibration_scores=cal_scores, calibration_complete=[m['complete'] for m in cal],
        calibration_valid_count=[m['n'] for m in cal], calibration_first_exit=[m['exit'] for m in cal],
        calibration_failure_reason=[m['reason'] for m in cal], observed_valid_count=obs_meta['n'],
        observed_first_exit=obs_meta['exit'], observed_failure_reason=obs_meta['reason'],
        horizon=H, eta=eta, effective_invalid_rank_score='+infinity', implementation_error=None)
    return out
