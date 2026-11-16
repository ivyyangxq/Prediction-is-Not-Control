# GQ Tool

GQ Tool is a Python app for researchers studying repeated decisions from data. It helps you work through a question such as: **given what was known at the time, which action does the learned model recommend for the objective I specified?** You use a Jupyter notebook with menus to load data, define the objective, inspect recommendations over time, and save the results.

It is intended for economists and other researchers who have records of the same units over several periods, including the actions actually taken and what happened afterward. The package implements the GQ learning method used in the *Prediction is Not Control* research project. You can start with the included small example before preparing your own data.

## What you provide

- A CSV table with a unit identifier, time period, current condition, actual action, previous action, and next observed condition. Include when each observation became available, so later information is not used for an earlier decision.
- A second CSV table with an observed outside input, such as an economic indicator, and its publication time.
- A numerical objective, the actions to consider, and how far ahead to evaluate them. You confirm the objective's parameters and keep them fixed during a run.

## What you get

The workbench reports recommended actions when its data and numerical checks pass. When it cannot provide a recommendation, it reports the reason. You can export reports and tables as JSON or CSV, save a session, resume it later, and add newly available observations.

Additional tools can investigate why a learned relationship has changed and compare alternative action paths in a compatible controlled simulation. Those features require extra evidence or a simulation that can evaluate the alternatives; importing a CSV alone does not enable them.

## A concrete example

Suppose you have a simulated panel of firms. Each row records a firm's output `x`, investment `a`, output in the next period, and when that information became available. A separate table contains an observed economic indicator. You could enter `x - cost*a*a` as an illustrative objective, set `cost` to `2`, and choose the investment levels to consider.

You can then step through decision periods, inspect the recommended investment at each period where the checks pass, and export the results. Recommendations remain separate from the investments recorded in the data. This example describes how you could use the inputs; it is not a claim that the package includes a validated model of firms or that following the recommendations raises profits.

## Scope and limits

This is a research tool for studying decisions under a stated objective and model assumptions. A recommendation means the tool could compute an action under its checks. It does not prove that the action will improve real outcomes.

Historical records alone do not establish what would have happened under an action that was never taken, or explain why a relationship changed. Those conclusions require additional assumptions and evidence. The tool can leave a question unresolved or decline to recommend an action when the required information is missing. See the [user guide](docs/START_HERE.md) for the conditions behind each feature.

## Install and try it

Use Python 3.10 or newer. Download the wheel from [GitHub Releases](https://github.com/ivyyangxq/Prediction-is-Not-Control/releases/latest), then run:

```sh
python -m pip install /path/to/gq_tool-1.0.3-py3-none-any.whl
gq-tool --check
gq-tool
```

The last command opens the notebook workbench. Follow the [first-run instructions](docs/START_HERE.md#first-run-the-small-example) to try the included example. To copy the notebook, example CSV files, and settings into a folder you can keep:

```sh
gq-tool --copy-example my_gq_example
```

For a source checkout, run `python -m pip install .`. The distribution name is `gq-tool`; the Python import name is `pinc_toolbox`. This project has not been published to PyPI.

## Guides and validation

- [User guide and data-column requirements](docs/START_HERE.md)
- [Installation details](docs/INSTALL.md)
- [Jupyter notebook](notebooks/GQ_tool.ipynb)
- [Technical test results and their limits](docs/ONE_SEED_TEST.md)
- [Contributing and running tests](CONTRIBUTING.md)

Local tests checked installation, notebook startup, saving and restoring sessions, and one simulated run with 24 decisions. These checks show that the tested workflow runs; they do not establish how well the method performs across datasets. Remote automated tests are not configured in this publication.

## License

MIT. See [LICENSE](LICENSE).
