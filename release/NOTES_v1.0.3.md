# GQ Tool 1.0.3

A Python workbench for economists studying dynamic decisions with reinforcement learning and online learning. Define a fixed reward, learn from panel histories as observations become available, and inspect recommendations together with the checks that determine whether they can be computed.

## When to use it

Use the notebook to study the gap between predicting observed outcomes and choosing actions under a specified objective. Load records of states, executed actions, subsequent outcomes, and observation times, together with an observed outside input. Choose the action grid and planning horizon, then inspect decisions over time, export reports, and save sessions for continuation. A small example is included.

The package also supports evidence-based investigation after residual alarms, applicable model updates, and comparisons of alternative action paths in compatible controlled simulations. A residual alarm does not by itself distinguish model misspecification from regime change. Diagnosis requires calibration and identifying evidence; simulated alternatives require an evaluator that can execute them. Ordinary observational data alone do not supply either.

Recommendations and simulated returns are conditional on the objective, model, support, and supplied evidence. They do not establish causal effects or gains in welfare or profits. See the [README](https://github.com/ivyyangxq/Prediction-is-Not-Control#readme) for the full scope and an illustrative investment example, and the [user guide](https://github.com/ivyyangxq/Prediction-is-Not-Control/blob/master/docs/START_HERE.md) to run the package.

## Install

Use Python 3.10 or newer. Download the wheel, then run:

```sh
python -m pip install gq_tool-1.0.3-py3-none-any.whl
gq-tool --check
gq-tool
```

The release includes a wheel, source distribution, and SHA256SUMS.txt. MIT licensed. Not published to PyPI.

## Technical validation

Local checks passed for builds, package metadata, installation, actual Jupyter startup, and one simulated run with 24 decisions, including session restoration and comparison tools. These tests verify the tested workflow; they do not establish performance across datasets or real economic gains. See docs/ONE_SEED_TEST.md for detailed conditions.

GitHub Actions was not configured because the existing push credential lacks permission to upload workflow files. No remote automated-test success is claimed.
