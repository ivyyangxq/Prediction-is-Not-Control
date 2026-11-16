"""JSON CLI: python -m pinc_toolbox.cli --input INPUT --output OUTPUT."""
from __future__ import annotations
import argparse
import json
import sys
from numbers import Integral
import numpy as np
from .g import SimpleGQ, linear_physical_memory_features, scalar_action_features
from .planning import MDPTableError, solve_bellman, solve_finite, solve_value_iteration

def _arr(value, ndim=None):
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected ndim={ndim}, got {array.ndim}")
    return array

def _index_list(value, name):
    array = np.asarray([value] if np.isscalar(value) else value, dtype=float).reshape(-1)
    if len(array) == 0 or not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name} must contain finite integer indices")
    return tuple(int(item) for item in array)

def _q_json(q, mask):
    array = np.asarray(q, dtype=float)
    support = np.asarray(mask, dtype=bool)
    if array.shape != support.shape:
        raise ValueError("Q and mask shapes do not agree")
    if not np.isfinite(array[support]).all():
        raise ValueError("supported Q contains a nonfinite value")
    return [[None if not support[s, a] else float(array[s, a]) for a in range(array.shape[1])] for s in range(array.shape[0])]

def _finite_json(value, name):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return array.tolist()

def _ordinary(value):
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("nonfinite output outside explicitly unsupported Q cells")
        return value.tolist()
    if isinstance(value, np.generic):
        return _ordinary(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("nonfinite scalar output")
        return value
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _ordinary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ordinary(item) for item in value]
    raise TypeError(f"cannot serialize {type(value).__name__}")

def run(payload):
    if not isinstance(payload, dict) or "g" not in payload:
        raise ValueError("input must contain top-level g")
    g = payload["g"]
    required = ["states", "actions", "next_physical", "release_dates", "as_of", "window", "memory_index", "physical_indices", "dictionary"]
    missing = [key for key in required if key not in g]
    if missing:
        raise ValueError(f"g missing fields: {missing}")
    if g["dictionary"] != "linear_physical_memory":
        raise ValueError("only dictionary=linear_physical_memory is supported")
    states = _arr(g["states"], 2)
    actions = _arr(g["actions"]).reshape(-1)
    next_physical = _arr(g["next_physical"])
    if next_physical.ndim == 1:
        next_physical = next_physical[:, None]
    if next_physical.ndim != 2:
        raise ValueError("next_physical must be one- or two-dimensional")
    physical_indices = _index_list(g["physical_indices"], "physical_indices")
    memory_indices = _index_list(g["memory_index"], "memory_index")
    if len(memory_indices) != 1:
        raise ValueError("memory_index must be one integer")
    memory_index = memory_indices[0]
    model = SimpleGQ(
        lambda x: linear_physical_memory_features(x, physical_indices, memory_index),
        scalar_action_features,
        memory_index=memory_index,
        physical_indices=physical_indices,
    )
    model.fit_released_window(states, actions, next_physical, g["release_dates"], g["as_of"], g["window"])
    query_states = _arr(g.get("query_states", states), 2)
    query_actions = _arr(g.get("query_actions", actions)).reshape(-1)
    if len(query_states) != len(query_actions):
        raise ValueError("query_states and query_actions rows must agree")
    gout = {
        "reference_motion": model.reference_motion(query_states),
        "response": model.response(query_states, query_actions),
        "prediction": model.predict_physical(query_states, query_actions),
        "diagnostics": model.diagnostics_,
    }
    out = {
        "g": gout,
        "planning": {},
        "assumptions": {
            "dictionary": "B=[1, physical coordinates, previous action], h=a",
            "estimator": "unpenalized same-sample FWL with floating previous-action anchor",
            "detection": "not_implemented",
            "source_certification": "not_implemented",
            "true_action_certification": "not_requested",
            "q_origin": "explicit user-supplied MDP; not extracted from arbitrary .mod",
        },
    }
    if "mdp" not in payload:
        return out
    mdp = payload["mdp"]
    required_mdp = ["rewards", "transitions", "mask", "beta"]
    missing_mdp = [key for key in required_mdp if key not in mdp]
    if missing_mdp:
        raise ValueError(f"mdp missing fields: {missing_mdp}")
    rewards = mdp["rewards"]
    transitions = mdp["transitions"]
    mask = np.asarray(mdp["mask"], dtype=bool)
    if "finite" in mdp:
        finite = mdp["finite"]
        horizon = finite["horizon"]
        terminal = finite.get("terminal", np.zeros(len(rewards)))
    elif "horizon" in mdp:
        horizon = mdp["horizon"]
        terminal = mdp.get("terminal", np.zeros(len(rewards)))
    else:
        horizon = None
    if horizon is not None:
        finite_result = solve_finite(rewards, transitions, mask, mdp["beta"], horizon, terminal)
        out["planning"]["q_finite"] = _q_json(finite_result.q, mask)
        out["planning"]["finite_policy"] = finite_result.policy
        out["planning"]["finite_status"] = finite_result.status
        out["planning"]["finite_bellman_residual"] = None
    infinite_config = mdp.get("infinite", {})
    method = infinite_config.get("method", "q_iteration")
    if method not in {"q_iteration", "value_iteration"}:
        raise ValueError("mdp.infinite.method must be 'q_iteration' or 'value_iteration'")
    solver = solve_value_iteration if method == "value_iteration" else solve_bellman
    infinite_result = solver(rewards, transitions, mask, mdp["beta"],
                             tolerance=infinite_config.get("tolerance", 1e-10),
                             max_iterations=infinite_config.get("max_iterations", 100000))
    if infinite_result.status != "converged":
        raise MDPTableError(
            f"infinite Bellman solver did not converge within {infinite_result.iterations} iterations; residual={infinite_result.residual}"
        )
    out["planning"].update({
        "q_infinite": _q_json(infinite_result.q, mask),
        "infinite_policy": infinite_result.policy,
        "infinite_method": method,
        "infinite_status": infinite_result.status,
        "bellman_residual": infinite_result.residual,
        "infinite_iterations": infinite_result.iterations,
        "mask": mask,
        "status": "ok",
    })
    if method == "value_iteration":
        out["planning"]["values_infinite"] = _finite_json(infinite_result.values, "values_infinite")
    return out

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        with open(args.input, encoding="utf-8") as handle:
            payload = json.load(handle)
        result = _ordinary(run(payload))
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, allow_nan=False, indent=2)
    except Exception as error:
        print(f"pinc_toolbox error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
