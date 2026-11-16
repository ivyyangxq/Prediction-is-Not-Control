"""Isolated finite-q adapter for the existing paired-source pipeline.

This is an ordinary-use adapter only. It does not launch histories, fit a law,
or alter the current experiment import chain. Filter patching is scoped and
restored in a finally block; callers must not invoke the same module concurrently.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
TRACK = HERE.parents[1]
HOST_FILE = TRACK / "track2_errors/public_posterior_forward_initializer_attempt001/public_forward_host.py"
FILTER_FILE = TRACK / "track2_errors/paired_monitor_conditioning_attempt001/public_posterior_with_pairs.py"
SAMPLER_FILE = TRACK / "track2_errors/conditional_sampler_interface_attempt001/sampler.py"
DGP_FILE = HERE.parents[3] / "DGP/RND/02_state_dependent_local/model_source_v004/environment.py"
_HOST_MODULE = None
_FILTER_MODULE = None
_SAMPLER_MODULE = None
_DGP_MODULE = None

Q_POINTS = (0.0, 0.25, 0.50, 0.69375, 1.0)
_Q_BY_ID = {
    "N": 0.0,
    "CL_Q025": 0.25,
    "CL_Q050": 0.50,
    "CL": 0.69375,
    "CL_Q100": 1.0,
}
# Existing N/CL stream indices are 0/1; added candidate IDs occupy fresh slots.
_STREAM_INDEX = {"N": 0, "CL": 1, "CM": 2, "MIX": 3,
                 "CL_Q025": 4, "CL_Q050": 5, "CL_Q100": 6}


def candidate_q(candidate: str) -> float:
    try:
        return _Q_BY_ID[str(candidate).upper()]
    except KeyError as exc:
        raise ValueError(f"candidate must be one of {tuple(_Q_BY_ID)}") from exc


def latent_matrix(q: float, *, f0=None, fstar=None) -> np.ndarray:
    """Return the declared interpolation matrix; reject silent extrapolation."""
    q = float(q)
    if not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("q must be finite and in [0,1]")
    if f0 is None or fstar is None:
        host = _host()
        f0 = host.F0 if f0 is None else f0
        fstar = host.FSTAR if fstar is None else fstar
    f0 = np.asarray(f0, dtype=float)
    fstar = np.asarray(fstar, dtype=float)
    if f0.shape != (3, 3) or fstar.shape != (3, 3) or not np.isfinite(f0).all() or not np.isfinite(fstar).all():
        raise ValueError("F0 and FSTAR must be finite 3x3 matrices")
    return f0 + q * (fstar - f0)


def latent_transition(z, innovation, q: float, *, transition_date: int) -> np.ndarray:
    """Apply the finite-q source transition in the host's row-vector convention."""
    z = np.asarray(z, dtype=float)
    innovation = np.asarray(innovation, dtype=float)
    if z.shape[-1:] != (3,) or innovation.shape != z.shape:
        raise ValueError("z and innovation must have matching final dimension 3")
    if not np.isfinite(z).all() or not np.isfinite(innovation).all():
        raise ValueError("z and innovation must be finite")
    f = _host().F0 if int(transition_date) < 32 else latent_matrix(q)
    return z @ f.T + innovation


def _load(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    # Sibling helper imports (notably exact_update.py) resolve through this path.
    source_dir = str(path.parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _host():
    global _HOST_MODULE
    if _HOST_MODULE is None:
        _HOST_MODULE = _load("latent_frontier_existing_forward_host", HOST_FILE)
    return _HOST_MODULE


def bind_forward_module(module) -> None:
    """Bind wrappers to a caller's already-instrumented host module."""
    global _HOST_MODULE
    if not hasattr(module, "PublicForwardHost"):
        raise TypeError("forward module must expose PublicForwardHost")
    _HOST_MODULE = module


def _filter():
    global _FILTER_MODULE
    if _FILTER_MODULE is None:
        _FILTER_MODULE = _load("latent_frontier_existing_paired_filter", FILTER_FILE)
    return _FILTER_MODULE


def _sampler():
    global _SAMPLER_MODULE
    if _SAMPLER_MODULE is None:
        _SAMPLER_MODULE = _load("latent_frontier_existing_sampler", SAMPLER_FILE)
    return _SAMPLER_MODULE


def _dgp():
    global _DGP_MODULE
    if _DGP_MODULE is None:
        _DGP_MODULE = _load("latent_frontier_actual_model_source_v004", DGP_FILE)
    return _DGP_MODULE


def stream_keys(*, history_index: int, candidate: str, replicate_id: int) -> dict[str, tuple[int, ...]]:
    """Use the established key layout while assigning new, noncolliding IDs."""
    candidate = str(candidate).upper()
    if candidate not in _STREAM_INDEX:
        raise ValueError(f"candidate must be one of {tuple(_STREAM_INDEX)}")
    base = _sampler()
    history_index = base._index(history_index, "history_index")
    replicate_id = base._index(replicate_id, "replicate_id")
    return {
        name: (1, history_index, _STREAM_INDEX[candidate], replicate_id, channel)
        for channel, name in enumerate(base.CHANNELS)
    }


def draw_replica_inputs(*, means, covariances, campaign_entropy,
                        history_index, candidate, replicate_id):
    """Draw with the explicit candidate key, then call the original transform."""
    candidate = str(candidate).upper()
    sampler = _sampler()
    m, _lower = sampler.posterior_factors(means=means, covariances=covariances)
    keys = stream_keys(history_index=history_index, candidate=candidate,
                       replicate_id=replicate_id)
    if isinstance(campaign_entropy, (bool, np.bool_)) or not isinstance(campaign_entropy, (int, np.integer)) or campaign_entropy < 0:
        raise ValueError("campaign_entropy must be a preassigned nonnegative integer")
    entropy = int(campaign_entropy)
    arrays = []
    for name, shape in zip(sampler.CHANNELS, sampler.SHAPES):
        sequence = np.random.SeedSequence(entropy=entropy, spawn_key=keys[name])
        generator = np.random.Generator(np.random.PCG64(sequence))
        arrays.append(sampler._array(generator.standard_normal(size=shape), shape, name))
    transformed = sampler.transform_normals(
        means=means, covariances=covariances,
        z_normals=arrays[0], state_normals=arrays[1],
        latent_normals=arrays[2], proxy_normals=arrays[3])
    return dict(host_inputs=transformed,
                stream_metadata=dict(campaign_entropy=entropy,
                    history_index=int(history_index), candidate=candidate,
                    replicate_id=int(replicate_id), spawn_keys={k:list(v) for k,v in keys.items()},
                    bit_generator='PCG64', numpy_version=np.__version__,
                    standard_normal_shapes={k:list(s) for k,s in zip(sampler.CHANNELS,sampler.SHAPES)},
                    normal_variates=sum(int(np.prod(shape)) for shape in sampler.SHAPES),
                    innovation_units='already_scaled_additive',
                    scope='numerical iid primitive simulation assumption; not a machine-exact Gaussian certificate'))


def prefix_dgp_factory(dgp, candidate: str):
    """Return an Environment factory suitable for the existing `_prefix`.

    q is installed into the actual source environment's g-profile before its
    first transition. The source constructor still performs its ordinary
    initialization and draws; no completed latent path is edited or replayed.
    """
    candidate = str(candidate).upper()
    q = candidate_q(candidate)
    source = "STABLE" if candidate == "N" else "LATENT_F"

    def Environment(seed, requested_source, *, paths=32, horizon=64,
                    switch_date=32, strength=1.0, profile="step", **kwargs):
        if str(requested_source).upper() != source:
            raise ValueError("prefix source does not match the registered frontier candidate")
        if float(strength) != 1.0:
            raise ValueError("prefix strength must remain the predeclared base value 1.0")
        env = dgp.Environment(seed, source, paths=paths, horizon=horizon,
                              switch_date=switch_date, strength=1.0,
                              profile=profile, **kwargs)
        if source == "LATENT_F":
            # The existing DGP transition is F0 + .69375*g_t*(FSTAR-F0).
            # Rescale g before any transitions so its effective coefficient is q.
            scale = q / 0.69375
            env.g_profile[:] = np.asarray(env.g_profile, dtype=float) * scale
            env.g_t = env.g_profile
        env.frontier_candidate = candidate
        env.frontier_q = q
        env.metadata["frontier_candidate"] = candidate
        env.metadata["latent_interpolation_q"] = q
        env.metadata["frontier_g_profile_scale"] = 1.0 if source == "STABLE" else q / 0.69375
        return env

    # Preserve the full environment module API used by `_prefix` (for example
    # ACTION_GRID and A0), replacing only the constructor seam.
    proxy = SimpleNamespace(**vars(dgp))
    proxy.Environment = Environment
    return proxy


def make_forward_host(candidate: str, **host_inputs):
    """Initialize existing forward host and bind its latent interpolation to q.

    The wrapped host keeps its native N/CL branch and physical equation. Its
    g-profile is rescaled so the host's `.69375 * g_t` equals the requested q.
    """
    candidate = str(candidate).upper()
    q = candidate_q(candidate)
    if candidate == "N":
        base_candidate = "N"
    else:
        base_candidate = "CL"
    host = _host().PublicForwardHost(candidate=base_candidate, **host_inputs)
    if candidate == "N":
        host.g_profile[:] = 0.0
    else:
        host.g_profile[:] = 0.0
        host.g_profile[host.switch_date:] = q / 0.69375
    host.g_t = host.g_profile
    host.frontier_candidate = candidate
    host.frontier_q = q
    host.metadata["frontier_candidate"] = candidate
    host.metadata["latent_interpolation_q"] = q
    return host


@contextmanager
def _finite_q_source_law(module, candidate: str, q: float):
    previous = module._source_law
    base_candidate = "N" if candidate == "N" else "CL"

    def source_law(_candidate, transition_date):
        if int(transition_date) < module.SWITCH_DATE:
            return module.F0, 0.0
        return latent_matrix(q, f0=module.F0, fstar=module.FSTAR), 0.0

    module._source_law = source_law
    try:
        yield base_candidate
    finally:
        module._source_law = previous


def filter_public_history(*, candidate: str, **public_history):
    """Run the existing expanded filter under one explicit finite-q law."""
    candidate = str(candidate).upper()
    q = candidate_q(candidate)
    module = _filter()
    with _finite_q_source_law(module, candidate, q) as base_candidate:
        result = module.filter_public_history(candidate=base_candidate, **public_history)
    return replace(result, candidate=candidate)
