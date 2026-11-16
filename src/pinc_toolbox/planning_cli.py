"""JSON bridge for explicit discounted state-value iteration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .planning import solve_value_iteration


def run(request):
    if not isinstance(request, dict) or not isinstance(request.get("mdp"), dict):
        raise ValueError("request must contain an mdp object")
    mdp = request["mdp"]
    required = ("rewards", "transitions", "mask", "beta")
    missing = [name for name in required if name not in mdp]
    if missing:
        raise ValueError(f"mdp missing fields: {missing}")
    config = mdp.get("infinite", {})
    result = solve_value_iteration(
        mdp["rewards"], mdp["transitions"], mdp["mask"], mdp["beta"],
        tolerance=config.get("tolerance", 1e-10),
        max_iterations=config.get("max_iterations", 100000),
    )
    if result.status != "converged":
        raise RuntimeError(
            f"value iteration did not converge in {result.iterations} iterations; "
            f"Bellman residual={result.residual}"
        )
    mask = np.asarray(mdp["mask"], dtype=bool)
    q = np.asarray(result.q, dtype=float)
    return {
        "method": "value_iteration",
        "values": result.values.tolist(),
        "q": [[float(q[s, a]) if mask[s, a] else None for a in range(q.shape[1])]
              for s in range(q.shape[0])],
        "policy": result.policy.tolist(),
        "mask": mask.tolist(),
        "bellman_residual": result.residual,
        "iterations": result.iterations,
        "status": result.status,
        "interpretation": "Decision evaluator for the supplied MDP; not a source or latent-switch detector.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.input.read_text(encoding="utf-8"))
    result = run(request)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
