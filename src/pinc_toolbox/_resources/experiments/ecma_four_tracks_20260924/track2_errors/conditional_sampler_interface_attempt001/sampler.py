"""Posterior/future array transforms and explicit streams; importing draws nothing.

The pure transform is ordinary-tested. draw_replica_inputs implements the
maintained numerical Gaussian simulation model, not machine-exact Gaussian law.
No original environment, seed, or future tape is consumed.
"""
import numpy as np

CANDIDATES = ('N', 'CL', 'CM', 'MIX')
CHANNELS = ('posterior', 'state', 'latent', 'proxy')
SHAPES = ((32, 3), (32, 8), (32, 8, 3), (32, 8, 3))
LATENT_SD = np.array([.04, .035, .03])


def _array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(name + ' has wrong shape or nonfinite entries')
    return array.copy()


def posterior_factors(*, means, covariances):
    """Return copied means and lower factors; no jitter/clipping/symmetrization.

    The audited saved posterior arrays are exactly symmetric. Reject a changed
    asymmetric input rather than silently choosing a different covariance.
    """
    m = _array(means, (32, 3), 'means')
    p = _array(covariances, (32, 3, 3), 'covariances')
    if not np.array_equal(p, p.transpose(0, 2, 1)):
        raise ValueError('covariances must be exactly symmetric saved arrays')
    lower = np.linalg.cholesky(p)
    if not np.isfinite(lower).all():
        raise FloatingPointError('nonfinite Cholesky factor')
    return m, lower


def _transform(m, lower, z, e, u, v):
    outputs = dict(posterior_draw=m + np.einsum('pij,pj->pi', lower, z),
                   state_noise=.02 * e, latent_noise=u * LATENT_SD,
                   measurement_noise=.02 * v)
    if any(not np.isfinite(value).all() for value in outputs.values()):
        raise FloatingPointError('nonfinite posterior/innovation transformation')
    return outputs


def transform_normals(*, means, covariances, z_normals, state_normals,
                      latent_normals, proxy_normals):
    """Pure transformation of supplied standard-normal arrays (or test fixtures).

    Supplied array provenance is the caller's responsibility. This function
    cannot certify that arrays are independent or standard normally distributed.
    Returned noises are already scaled additive innovations, dated 40..47 for
    state/latent and 41..48 for measurement; w40 is never resampled here.
    """
    m, lower = posterior_factors(means=means, covariances=covariances)
    values = [z_normals, state_normals, latent_normals, proxy_normals]
    arrays = [_array(value, shape, name) for value, shape, name in zip(values, SHAPES, CHANNELS)]
    return _transform(m, lower, *arrays)


def _index(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if not 0 <= value <= 2**32 - 1:
        raise ValueError(name + ' must be in [0,2**32-1]')
    return value


def stream_keys(*, history_index, candidate, replicate_id):
    """Return plain key tuples; does not instantiate a generator or seed sequence."""
    if candidate not in CANDIDATES:
        raise ValueError('candidate must be N, CL, CM, or MIX')
    h = _index(history_index, 'history_index')
    r = _index(replicate_id, 'replicate_id')
    return {name: (1, h, CANDIDATES.index(candidate), r, channel)
            for channel, name in enumerate(CHANNELS)}


def draw_replica_inputs(*, means, covariances, campaign_entropy,
                        history_index, candidate, replicate_id):
    """Generate one candidate/replicate's host-ready inputs when explicitly called.

    The caller must preassign fresh campaign entropy independently of historical
    DGP/detector/source data. Never pass the original DGP seed. Repeating the same
    key deliberately reproduces the same proposal, not an independent replicate.
    Failures propagate; this function never retries, tops up, or drops a draw.
    """
    m, lower = posterior_factors(means=means, covariances=covariances)
    keys = stream_keys(history_index=history_index, candidate=candidate,
                       replicate_id=replicate_id)
    if isinstance(campaign_entropy, (bool, np.bool_)) or not isinstance(campaign_entropy, (int, np.integer)) or campaign_entropy < 0:
        raise ValueError('campaign_entropy must be a preassigned nonnegative integer')
    entropy = int(campaign_entropy)
    arrays = []
    for name, shape in zip(CHANNELS, SHAPES):
        sequence = np.random.SeedSequence(entropy=entropy, spawn_key=keys[name])
        generator = np.random.Generator(np.random.PCG64(sequence))
        arrays.append(_array(generator.standard_normal(size=shape), shape, name))
    return dict(host_inputs=_transform(m, lower, *arrays),
                stream_metadata=dict(campaign_entropy=entropy,
                    history_index=int(history_index), candidate=candidate,
                    replicate_id=int(replicate_id), spawn_keys={k:list(v) for k,v in keys.items()},
                    bit_generator='PCG64', numpy_version=np.__version__,
                    standard_normal_shapes={k:list(s) for k,s in zip(CHANNELS,SHAPES)},
                    normal_variates=1888, innovation_units='already_scaled_additive',
                    scope='numerical iid primitive simulation assumption; not a machine-exact Gaussian certificate'))
