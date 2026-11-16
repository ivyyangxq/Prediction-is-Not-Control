# GQ Tool 1.0.3

A Python notebook app for researchers studying repeated decisions from records of actual actions and outcomes. Load data, define a numerical objective, inspect recommended actions over time, export reports, and save sessions to continue later. A small example is included.

## Inputs and results

Provide a CSV table of units, dates, current conditions, actual and previous actions, next observed conditions, and when observations became available. A second table supplies an observed outside input and its publication time. You choose the actions to consider and confirm the objective and its parameters.

The workbench reports recommendations where its checks pass, and explains refusals where they do not. Optional tools for investigating changes and comparing simulated action paths require additional evidence or a compatible controlled simulation. Recommendations do not establish that an action will improve real outcomes or identify the effect of an action never taken.

For an illustrative investment example and feature limits, see the [README](https://github.com/ivyyangxq/Prediction-is-Not-Control#readme). Follow the [user guide](https://github.com/ivyyangxq/Prediction-is-Not-Control/blob/master/docs/START_HERE.md) to try the included example.

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
