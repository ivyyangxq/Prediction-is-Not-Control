"""Observed HS Y/Z likelihood source backend, extracted candidate, untested.

Scientific algorithm: HS2022/source_diagnosis_closure_20260921. This is distinct
from eta_monitor's panel moment region. Both Y and Z are observed. Source
names are growth exposure and mean reversion, never hidden-latent recovery.
No research-folder imports, environment configuration, or simulated truth is
used by this module. Ordinary floating envelopes are not outward-rounded.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

@dataclass(frozen=True)
class HSParameters:
    """Parameters for one member of equation (26).

    ``alpha_y`` and ``beta_y`` are the coefficients inside the ``.01``-scaled
    Y drift.  ``alpha_z`` and ``beta_z`` are the level and mean-reversion
    coefficients of Z.  Volatility vectors are the *unscaled* source-paper
    vectors; the exact transition applies the ``.01`` factor to the Y row.
    """
    alpha_y: float = 0.484
    beta_y: float = 1.0
    alpha_z: float = 0.0
    beta_z: float = 0.014
    sigma_y: tuple[float, float] = (0.477, 0.0)
    sigma_z: tuple[float, float] = (0.011, 0.025)
    y_scale: float = 0.01

    def sigma_unscaled(self) -> np.ndarray:
        return np.asarray((self.sigma_y, self.sigma_z), dtype=float)

    def volatility(self) -> np.ndarray:
        """The diffusion matrix in (25)/(26), with Y's .01 scaling applied."""
        return np.asarray((self.y_scale * np.asarray(self.sigma_y), self.sigma_z), dtype=float)

def _validate_beta(beta_z: float) -> None:
    if beta_z <= 0.0:
        raise ValueError('beta_z must be positive for the stationary OU formulas')

def transition_matrices(params: HSParameters, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact affine transition ``X' = F X + d + eps`` and covariance Q.

    Here ``X=(Y,Z)'`` and ``eps`` is zero-mean Gaussian.  The formulas are the
    closed-form integral of ``exp(A s) B B' exp(A' s)`` and therefore do not
    use an Euler approximation.
    """
    if dt < 0.0:
        raise ValueError('dt must be non-negative')
    _validate_beta(params.beta_z)
    h = float(dt)
    b = float(params.beta_z)
    eb = np.exp(-b * h)
    g1 = -np.expm1(-b * h) / b
    g2 = -np.expm1(-2.0 * b * h) / (2.0 * b)
    c = params.y_scale * params.beta_y
    F = np.array([[1.0, c * g1], [0.0, eb]], dtype=float)
    d = np.array([params.y_scale * params.alpha_y * h + params.y_scale * params.beta_y * params.alpha_z / b * (h - g1), params.alpha_z * g1], dtype=float)
    B = params.volatility()
    qyy = float(B[0] @ B[0])
    qyz = float(B[0] @ B[1])
    qzz = float(B[1] @ B[1])
    ig = (h - g1) / b
    ig2 = (h - 2.0 * g1 + g2) / (b * b)
    q_yz = qyz * g1 + c * qzz * ((g1 - g2) / b)
    q_yy = qyy * h + 2.0 * c * qyz * ig + c * c * qzz * ig2
    Q = np.array([[q_yy, q_yz], [q_yz, qzz * g2]], dtype=float)
    Q = 0.5 * (Q + Q.T)
    return (F, d, Q)

def _add(x, y):
    return (x[0] + y[0], x[1] + y[1])

def _sub(x, y):
    return (x[0] - y[1], x[1] - y[0])

def _mul(x, y):
    z = [x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]]
    return (min(z), max(z))

def _scale(x, a):
    return _mul(x, (float(a), float(a)))

def _div(x, y):
    if y[0] <= 0 <= y[1]:
        return (-np.inf, np.inf)
    return _mul(x, (1.0 / y[1], 1.0 / y[0]))

def _sq(x):
    if x[0] <= 0 <= x[1]:
        return (0.0, max(x[0] * x[0], x[1] * x[1]))
    return (min(x[0] * x[0], x[1] * x[1]), max(x[0] * x[0], x[1] * x[1]))

def _integral_values(b: float) -> tuple[float, float, float, float, float]:
    g1 = -np.expm1(-b) / b
    g2 = -np.expm1(-2.0 * b) / (2.0 * b)
    i1 = (1.0 - g1) / b
    i2 = (1.0 - 2.0 * g1 + g2) / (b * b)
    j = (g1 - g2) / b
    return (g1, g2, i1, i2, j)

class HSObservedLikelihood:
    """Known-reference, known-common-diffusion, fully observed Y/Z program.

    The declared mixture endpoints, horizon/onset prior, alpha, query calendar
    and numerical search budget define the procedure. No eta is inferred.
    """
    def __init__(self, *, horizon, mixture_betas, alpha=.05, weight_unchanged=.1,
                 interval_budget=4096):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        if isinstance(interval_budget, bool) or not isinstance(interval_budget, int) or interval_budget < 1:
            raise ValueError('interval_budget must be a positive integer')
        alpha, weight_unchanged = float(alpha), float(weight_unchanged)
        if not 0 < alpha < 1 or not 0 < weight_unchanged < 1:
            raise ValueError('alpha and weight_unchanged must lie in (0,1)')
        endpoints = np.asarray(mixture_betas, dtype=float)
        if endpoints.ndim != 2 or endpoints.shape[1] != 2 or len(endpoints) == 0:
            raise ValueError('declared mixture_betas must be a nonempty (K,2) array')
        if not np.isfinite(endpoints).all() or np.any(endpoints[:,1] <= 0):
            raise ValueError('mixture endpoints must be finite with positive beta_z')
        self.reference = HSParameters()
        self.horizon, self.alpha, self.weight_unchanged = horizon, alpha, weight_unchanged
        self.endpoints = endpoints.copy()
        self.log_threshold = float(np.log(1.0/alpha))
        self.interval_budget, self.interval_tolerance = interval_budget, 1e-8
        self.dt, self.family_grid = 1., 33

    def params(self, beta_y: float, beta_z: float) -> HSParameters:
        return HSParameters(alpha_y=self.reference.alpha_y, beta_y=float(beta_y), alpha_z=self.reference.alpha_z, beta_z=float(beta_z), sigma_y=self.reference.sigma_y, sigma_z=self.reference.sigma_z, y_scale=self.reference.y_scale)

    def ll_increments(self, path: np.ndarray, beta_y: float, beta_z: float) -> np.ndarray:
        """Full joint Gaussian transition log density, one released transition at a time."""
        (F, d, Q) = transition_matrices(self.params(beta_y, beta_z), self.dt)
        (sign, logdet) = np.linalg.slogdet(Q)
        if sign <= 0:
            raise ValueError('non-positive transition covariance')
        invQ = np.linalg.inv(Q)
        residual = path[1:] - path[:-1] @ F.T - d
        quad = np.einsum('ti,ij,tj->t', residual, invQ, residual)
        return -0.5 * (2.0 * np.log(2.0 * np.pi) + logdet + quad)

    def mixture_log_ratio(self, path: np.ndarray, endpoint_betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return log(q_t/p_ref,t) and singleton alarm flags for t=1,...,T."""
        ref = self.ll_increments(path, self.reference.beta_y, self.reference.beta_z)
        log_a = np.full(len(endpoint_betas), -np.inf, dtype=float)
        n = path.shape[0] - 1
        log_ratio = np.empty(n, dtype=float)
        endpoint_llr = np.asarray([self.ll_increments(path, b[0], b[1]) - ref for b in endpoint_betas])
        for t in range(n):
            injection = -np.log(self.horizon) if t + 1 <= self.horizon else -np.inf
            log_a = endpoint_llr[:, t] + np.logaddexp(log_a, injection)
            future_mass = max(0.0, (self.horizon - (t + 1)) / self.horizon)
            future = np.log(future_mass) if future_mass > 0 else -np.inf
            terms = np.logaddexp(log_a, future)
            log_ratio[t] = np.logaddexp(np.log(self.weight_unchanged), np.log1p(-self.weight_unchanged) - np.log(len(endpoint_betas)) + logsumexp(terms))
        return (log_ratio, log_ratio >= self.log_threshold)

    def max_onset_suffix(self, delta: np.ndarray, t: int) -> tuple[float, int]:
        """Maximize p(beta, onset)/p(ref) over all onset dates including future onset."""
        if t <= 0:
            return (0.0, t)
        suffix = np.cumsum(delta[:t][::-1])[::-1]
        idx = int(np.argmax(np.r_[suffix, 0.0]))
        if idx == t:
            return (0.0, t)
        return (float(suffix[idx]), idx)

    def candidate_profile(self, path: np.ndarray, t: int, family: str, threshold: float | None=None) -> dict[str, float | bool | str]:
        """Grid-seed plus bounded refinements; result is explicitly candidate-only."""
        if family == 'pure_beta_y':
            (lo, hi) = (-1.0, 3.0)

            def beta(v: float) -> tuple[float, float]:
                return (float(v), self.reference.beta_z)
        elif family == 'pure_beta_z':
            (lo, hi) = (0.001, 0.1)

            def beta(v: float) -> tuple[float, float]:
                return (self.reference.beta_y, float(v))
        else:
            raise ValueError(family)
        ref = self.ll_increments(path, self.reference.beta_y, self.reference.beta_z)
        grid = np.linspace(lo, hi, self.family_grid)
        values = []
        for v in grid:
            b = beta(float(v))
            values.append(self.max_onset_suffix(self.ll_increments(path, *b) - ref, t)[0])
        values = np.asarray(values)
        candidates: list[tuple[float, float, float]] = [(float(values[0]), lo, float(grid[1])), (float(values[-1]), float(grid[-2]), hi)]
        for i in range(1, len(grid) - 1):
            if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
                candidates.append((float(values[i]), float(grid[i - 1]), float(grid[i + 1])))
        grid_best_i = int(np.argmax(values))
        grid_onset = self.max_onset_suffix(self.ll_increments(path, *beta(float(grid[grid_best_i]))) - ref, t)[1]
        if threshold is not None and float(values[grid_best_i]) >= threshold:
            return {'profile_log_ratio': float(values[grid_best_i]), 'profile_value': float(grid[grid_best_i]), 'profile_onset': grid_onset, 'domain_low': lo, 'domain_high': hi, 'grid_log_ratio': float(values[grid_best_i]), 'grid_gap': 0.0, 'converged': True, 'boundary': bool(grid_best_i in (0, len(grid) - 1)), 'failure': '', 'status': 'candidate_point_witness_early_return'}
        best = (float(values[grid_best_i]), float(grid[grid_best_i]), grid_onset)
        converged = True
        failures: list[str] = []
        for (_seed, a, b) in candidates:
            try:
                result = minimize_scalar(lambda v: -self.max_onset_suffix(self.ll_increments(path, *beta(float(v))) - ref, t)[0], bounds=(a, b), method='bounded', options={'xatol': 1e-07, 'maxiter': 200})
                val = -float(result.fun)
                onset_val = self.max_onset_suffix(self.ll_increments(path, *beta(float(result.x))) - ref, t)[1]
                if val > best[0]:
                    best = (val, float(result.x), onset_val)
                converged = converged and bool(result.success)
                if not result.success:
                    failures.append(str(result.message))
            except Exception as exc:
                converged = False
                failures.append(f'{type(exc).__name__}: {exc}')
        fine_idx = int(np.argmax(values))
        fine_best = float(values[fine_idx])
        boundary = bool(abs(best[1] - lo) < 1e-05 or abs(best[1] - hi) < 1e-05)
        return {'profile_log_ratio': best[0], 'profile_value': best[1], 'profile_onset': best[2], 'domain_low': lo, 'domain_high': hi, 'grid_log_ratio': fine_best, 'grid_gap': best[0] - fine_best, 'converged': converged, 'boundary': boundary, 'failure': '; '.join(failures), 'status': 'candidate_only_unavailable_for_exclusion'}

    def _suffix_statistics(self, path: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        z = path[:t, 1]
        u = path[1:t + 1, 0] - path[:t, 0] - 0.01 * self.reference.alpha_y
        v = path[1:t + 1, 1] - path[:t, 1]
        cols = np.column_stack((z * z, u * u, v * v, z * u, z * v, u * v))
        pref = np.vstack((np.zeros(6), np.cumsum(cols, axis=0)))
        stats = pref[t] - pref[:t]
        ref_ll = np.cumsum(self.ll_increments(path, self.reference.beta_y, self.reference.beta_z))
        suffix_ref = ref_ll[t - 1] - np.r_[0.0, ref_ll[:t - 1]]
        return (stats, suffix_ref)

    def interval_loglr_upper(self, stats: np.ndarray, suffix_ref: np.ndarray, family: str, lo: float, hi: float) -> tuple[float, int]:
        """Conservative upper bound over one beta interval and every onset suffix."""
        by = (lo, hi) if family == 'pure_beta_y' else (self.reference.beta_y, self.reference.beta_y)
        bz = (self.reference.beta_z, self.reference.beta_z) if family == 'pure_beta_y' else (lo, hi)
        vals_lo = _integral_values(bz[0])
        vals_hi = _integral_values(bz[1])

        def bounds(k):
            return (min(vals_lo[k], vals_hi[k]), max(vals_lo[k], vals_hi[k]))
        (g1, g2, i1, i2, j) = (bounds(k) for k in range(5))
        c = _scale(by, 0.01)
        D = (np.exp(-bz[1]) - 1.0, np.exp(-bz[0]) - 1.0)
        B = self.reference.volatility()
        a = float(B[0] @ B[0])
        d = float(B[0] @ B[1])
        f = float(B[1] @ B[1])
        Vz = _scale(g2, f)
        Qyz = _add(_scale(g1, d), _mul(_scale(c, f), j))
        Qyy = _add((a, a), _add(_scale(_mul(_mul(c, (d, d)), i1), 2.0), _mul(_mul(_sq(c), (f, f)), i2)))
        K = _div(Qyz, Vz)
        C = _sub(_mul(c, g1), _mul(K, D))
        Vy = _sub(Qyy, _div(_sq(Qyz), Vz))
        if Vz[0] <= 0 or Vy[0] <= 0 or (not np.isfinite(Vz[1] + Vy[1])):
            return (np.inf, -1)

        def imul(x, y):
            z = np.stack((x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]))
            return (np.min(z, axis=0), np.max(z, axis=0))

        def iadd(x, y):
            return (x[0] + y[0], x[1] + y[1])

        def iscale(x, a):
            return imul(x, (a, a))

        def isq(x):
            return (np.zeros_like(x[0]), np.maximum(x[0] * x[0], x[1] * x[1])) if np.all(x[0] <= 0) and np.all(x[1] >= 0) else imul(x, x)
        (zz, uu, vv, zu, zv, uv) = (stats[:, i] for i in range(6))
        rssz = iadd((vv, vv), iadd(iscale(imul(D, (zv, zv)), -2.0), imul(isq(D), (zz, zz))))
        rssy = iadd((uu, uu), iscale(imul(C, (zu, zu)), -2.0))
        rssy = iadd(rssy, iscale(imul(K, (uv, uv)), -2.0))
        rssy = iadd(rssy, imul(isq(C), (zz, zz)))
        rssy = iadd(rssy, iscale(imul(imul(C, K), (zv, zv)), 2.0))
        rssy = iadd(rssy, imul(isq(K), (vv, vv)))
        rssz = (np.maximum(0.0, rssz[0]), rssz[1])
        rssy = (np.maximum(0.0, rssy[0]), rssy[1])
        n = np.arange(len(stats), 0, -1, dtype=float)
        upper = -n * np.log(2 * np.pi) - 0.5 * n * (np.log(Vz[0]) + np.log(Vy[0])) - 0.5 * rssz[0] / Vz[1] - 0.5 * rssy[0] / Vy[1]
        vals = upper - suffix_ref
        best_s = int(np.argmax(vals))
        return (max(0.0, float(vals[best_s])), best_s)

    def continuous_source_decision(self, path: np.ndarray, t: int, family: str, logq: float, candidate: dict) -> dict:
        threshold = logq - self.log_threshold
        if threshold <= 0:
            return {'bound_log_ratio': np.nan, 'bound_margin': np.nan, 'source_decision': 'compatible_stable_witness', 'interval_evals': 0, 'unavailable_reason': 'threshold_nonpositive'}
        if np.isfinite(candidate['profile_log_ratio']) and candidate['profile_log_ratio'] >= threshold:
            return {'bound_log_ratio': np.nan, 'bound_margin': candidate['profile_log_ratio'] - threshold, 'source_decision': 'retained_point_witness', 'interval_evals': 0, 'unavailable_reason': ''}
        (stats, suffix_ref) = self._suffix_statistics(path, t)
        (lo, hi) = (-1.0, 3.0) if family == 'pure_beta_y' else (0.001, 0.1)
        queue = []
        serial = 0

        def push(a, b):
            nonlocal serial
            (upper, onset) = self.interval_loglr_upper(stats, suffix_ref, family, a, b)
            heapq.heappush(queue, (-upper, serial, a, b, upper, onset))
            serial += 1
        push(lo, hi)
        evaluations = 1
        max_upper = np.inf
        separated_upper = -np.inf
        while queue and evaluations < self.interval_budget:
            (_neg, _serial, a, b, upper, onset) = heapq.heappop(queue)
            max_upper = upper
            if upper < threshold - self.interval_tolerance:
                separated_upper = max(separated_upper, upper)
                continue
            if b - a < 1e-07:
                return {'bound_log_ratio': upper, 'bound_margin': upper - threshold, 'source_decision': 'unavailable', 'interval_evals': evaluations, 'unavailable_reason': 'interval_width_budget'}
            mid = 0.5 * (a + b)
            push(a, mid)
            push(mid, b)
            evaluations += 2
        if queue:
            max_upper = max(max_upper, -queue[0][0])
            return {'bound_log_ratio': max_upper, 'bound_margin': max_upper - threshold, 'source_decision': 'unavailable', 'interval_evals': evaluations, 'unavailable_reason': 'interval_budget'}
        max_upper = max(separated_upper, max_upper if np.isfinite(max_upper) else -np.inf)
        return {'bound_log_ratio': max_upper, 'bound_margin': max_upper - threshold, 'source_decision': 'excluded_continuous_bound', 'interval_evals': evaluations, 'unavailable_reason': ''}

    def diagnose(self, public_path, *, as_of, query_times):
        """Consume released Y/Z levels only; index k is release clock k.

        public_path[0] is the initial observed state. At clock t only
        public_path[:t+1] is used. Query times are declared finite dates and
        source families keep the old confirmation domains unchanged.
        """
        if isinstance(as_of, bool) or not isinstance(as_of, int) or not 1 <= as_of <= self.horizon:
            raise ValueError('as_of must be an integer in [1,horizon]')
        if len(public_path) < as_of+1:
            raise ValueError('not enough released Y/Z states for as_of')
        path = np.asarray(public_path[:as_of+1], dtype=float)
        if path.shape != (as_of+1,2) or not np.isfinite(path).all():
            raise ValueError('released path must have finite (as_of+1,2) Y/Z levels')
        queries = list(query_times)
        if not queries or len(set(queries)) != len(queries) or any(
            isinstance(t,bool) or not isinstance(t,int) or not 1 <= t <= self.horizon for t in queries):
            raise ValueError('query_times must be distinct declared dates in [1,horizon]')
        logq, alarm = self.mixture_log_ratio(path, self.endpoints)
        if not np.isfinite(logq).all():
            raise ArithmeticError('nonfinite HS likelihood ratio; no source conclusion issued')
        records = []
        for t in sorted(queries):
            if t > as_of:
                continue
            prefix = path[:t+1]
            for family, name in (('pure_beta_y','pure_growth_exposure'),('pure_beta_z','pure_mean_reversion')):
                ordinary = bool(alarm[t-1])
                if ordinary:
                    candidate = self.candidate_profile(prefix,t,family,float(logq[t-1]-self.log_threshold))
                    decision = self.continuous_source_decision(prefix,t,family,float(logq[t-1]),candidate)
                else:
                    candidate = dict(profile_log_ratio=None,profile_value=None,profile_onset=None,
                                     status='compatible_stable_witness')
                    decision = dict(bound_log_ratio=None,bound_margin=None,
                                    source_decision='compatible_stable_witness',interval_evals=0,
                                    unavailable_reason='')
                records.append(dict(release=t,family=family,named_source_family=name,
                    ordinary_log_q_over_ref=float(logq[t-1]),ordinary_alarm=ordinary,
                    candidate=candidate,decision=decision))
        return _strict_json(dict(
            procedure='hs_observed_joint_likelihood_continuous_source',
            implementation_status='extracted_candidate_numerically_unvalidated',as_of=as_of,
            observation='both Y and Z observed',release_clock='path index; t completed transitions at clock t',
            configuration=dict(horizon=self.horizon,alpha=self.alpha,weight_unchanged=self.weight_unchanged,
                mixture_betas=self.endpoints.tolist(),query_times=queries,
                family_domains=dict(pure_growth_exposure=[-1.,3.],pure_mean_reversion=[.001,.10]),
                interval_budget=self.interval_budget,interval_tolerance=self.interval_tolerance),
            alarm_log_ratio=logq.tolist(),alarm_flags=alarm.tolist(),source_records=records,
            scope='known fixed HS reference and diffusion; no measurement noise; at most one permanent parameter change',
            numerical_guarantee='ordinary floating envelope, not outward-rounded verified arithmetic',
            eta_selection='not part of this likelihood program',joint_family_compatibility='not evaluated',
            hidden_latent_recovery='not claimed',action_Q_certification='unavailable',
            inherited_experimental_rates=None))


def _strict_json(value):
    if isinstance(value, dict):
        return {key:_strict_json(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [_strict_json(item) for item in value]
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):
        return int(value)
    if isinstance(value,np.bool_):
        return bool(value)
    return value


def run_public_request(request):
    """JSON entry distinct from the panel eta/source moment program."""
    if request.get('operation') != 'hs_observed_likelihood':
        raise ValueError('hs_observed_likelihood operation required')
    specification = request['likelihood']
    model = HSObservedLikelihood(
        horizon=specification['horizon'],mixture_betas=specification['mixture_betas'],
        alpha=specification.get('alpha',.05),weight_unchanged=specification.get('weight_unchanged',.1),
        interval_budget=specification.get('interval_budget',4096))
    return model.diagnose(request['public_path'],as_of=request['as_of'],
                          query_times=specification['query_times'])
