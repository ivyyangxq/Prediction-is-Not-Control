# GQ Tool

GQ Tool is a Python workbench for economists studying **dynamic decisions with reinforcement learning and online learning**. It brings together learning from observed state transitions, checking whether a proposed decision is supported by the available data, and evaluating evidence about changes in the learned environment. You specify the reward function and keep it fixed while the learner processes new observations.

The motivation is a familiar problem in quantitative economics: a model can predict reasonably well on observed histories yet provide a weak basis for choosing a different action. After a residual alarm or an apparent regime change, a researcher also needs to decide whether to continue learning, investigate the source of the change, or revise the model. This package provides a notebook workflow for studying those decisions under explicit data and model requirements. It implements the GQ learning method used in the *Prediction is Not Control* project.

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

## Scope: learning, diagnosis, and model updates

**Use the package when the object of interest is a sequence of decisions, not just a forecast.** Starting from panel histories of states, executed actions, and subsequent outcomes, you define a reward, a candidate action grid, and a planning horizon. The learner processes observations as they become available and computes recommendations where its support and numerical conditions hold. The workbench records the decision history and reasons for declining a recommendation, so you can inspect where the learning procedure can operate and where it runs out of usable information.

**Treat residual evidence as a reason to investigate.** In an online learning problem, a change in prediction errors can raise questions about model misspecification, changes in the environment, or both. The package includes detection and source-diagnosis workflows for evaluating specified explanations when the required calibration, candidate models, and identifying evidence are available. A residual alarm alone does not distinguish regime change from misspecification. With ordinary observational panels, that distinction may remain unresolved; importing data does not create the missing identification assumptions.

**Make model updates conditional on the evidence.** Where an implemented update is applicable and the evidence requirements are met, you can explicitly accept or decline it and continue the session under the same reward function. Keeping the objective fixed lets you study a change in the learner without simultaneously changing what it is trying to achieve. An accepted update does not establish that the economic problem has been solved, and it does not waive the checks required for subsequent recommendations.

**Evaluate alternative action paths only where they can be executed.** The controlled-simulation comparison tools run four paths that vary the action choice and whether a specified outside input is held fixed. They help inspect how these choices contribute to simulated returns. They require a compatible evaluator and supported actions. Observational CSV files alone do not identify returns under actions never taken; holding an observed proxy fixed also does not hold an unobserved economic state fixed.

This makes the workbench useful for computational experiments and for organizing empirical learning exercises with explicit limits on what the data support. A computed recommendation is conditional on the chosen objective, model, and available histories. Neither that recommendation nor a successful diagnostic establishes a causal effect, an optimal economic policy, or a gain in welfare or profits. See the [user guide](docs/START_HERE.md) for the input requirements and feature-specific conditions.

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
