"""Opt-in HS stopped-score route. The complete-only adapter is unchanged.

Only supported transitions are read into residual computation. A semantic stop
with a nonempty finite prefix can rank CAL; observed noncompletion is unresolved.
"""
import numpy as np
from pinc_toolbox.residual_energy import sequential_slope_energy
from pinc_toolbox.stopped_candidate_rank import stopped_candidate_rank


def score_scalar_terminal(*, evaluation_path, calibration_paths,
        reference_coefficients, candidate_F, candidate_d, candidate_covariance,
        residual_left_inverse, support, eta, alpha, context, include_terminal=False):
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

    def score(path):
        n = 0; first_exit = H; reason = None; value = None; terminal = None
        predictors = []; outcomes = []
        try:
            x = row(path, 0)
            if start is None or not np.array_equal(x, start):
                return None, False, False, 0, None, 'missing_or_mismatched_start', None
            for t in range(H):
                # Check the predictor BEFORE reading the next outcome. No suffix
                # conversion, residual, weight, or finiteness check after exit.
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    distance = np.maximum(np.maximum((low-x)/width, (x-high)/width), 0).max()
                if distance > eta:
                    first_exit = t
                    reason = 'support_stop' if n else 'empty_supported_prefix'
                    break
                xn = row(path, t + 1)
                predictors.append(x); outcomes.append(xn); n += 1; x = xn
            if n:
                x = np.asarray(predictors); xn = np.asarray(outcomes)
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    # Existing explicit reference residual minus candidate drift.
                    residual = (xn - np.einsum('tp,pq->tq', x, fr.T) - dr) - (
                        np.einsum('tp,tqp->tq', x, f[:n] - fr) + (d[:n] - dr))
                    result = sequential_slope_energy(residual[None], x[None, :, 1],
                        residual_left_inverse=inv, return_trace=False)
                value = float(result['scores'][0, 0])
                terminal = float(result['terminal_energy'][0, 0])
                if not np.isfinite(value):
                    raise FloatingPointError('nonfinite prefix score')
            return value, n == H, n > 0, n, first_exit, reason, terminal
        except (ValueError, TypeError, IndexError, FloatingPointError) as exc:
            return None, False, False, n, None, 'numeric_or_missing_supported_path:' + type(exc).__name__, None

    obs = score(evaluation_path)
    cal = [score(p) for p in calibration_paths]
    scores, full, valid, counts, exits, reasons, terminal_scores = map(list, zip(*cal))
    out = stopped_candidate_rank(observed_score=obs[0], observed_full_window_complete=obs[1],
        observed_score_valid_for_rank=obs[2], calibration_scores=scores,
        calibration_full_window_complete=full, calibration_score_valid_for_rank=valid, alpha=alpha)
    out.update(context=context, statistic='sequential_slope_energy', coordinates=['DeltaY', 'Z'],
        candidate_F=f.tolist(), candidate_d=d.tolist(), candidate_covariance=cov.tolist(),
        source_start_state=None if start is None else start.tolist(),
        calibration_scores=scores, calibration_complete=full,
        calibration_valid_count=counts, calibration_first_exit=exits,
        calibration_failure_reason=reasons, observed_valid_count=obs[3],
        observed_first_exit=obs[4], observed_failure_reason=obs[5], horizon=H, eta=eta,
        effective_invalid_rank_score='+infinity', implementation_error=None)
    if include_terminal:
        from .hs_terminal_helpers import attach_terminal
        attach_terminal(out, obs[6], terminal_scores)
    return out
