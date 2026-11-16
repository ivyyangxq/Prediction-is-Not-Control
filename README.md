# GQ Tool

A Python workbench for fixed-objective GQ learning from released panel data, with explicit reward inputs and parameters, support checks, evidence workflows, checkpoints and controlled ABCD comparisons.

## Install

Download the wheel from GitHub Releases, then run:

```sh
python -m pip install /path/to/gq_tool-1.0.3-py3-none-any.whl
gq-tool --check
gq-tool
```

For a source checkout: `python -m pip install .`.

The distribution name is `gq-tool`; the Python import namespace remains `pinc_toolbox`. No research checkout or PYTHONPATH configuration is required. This release has not been published to PyPI; do not assume `pip install gq-tool` resolves to this project.

## Start here

- [User guide](docs/START_HERE.md)
- [Installation](docs/INSTALL.md)
- [Jupyter workbench](notebooks/GQ_tool.ipynb)
- [One-seed validation](docs/ONE_SEED_TEST.md)
- [Contributing and tests](CONTRIBUTING.md)

Copy runnable starter files with `gq-tool --copy-example NEW_DIRECTORY`.

## Scope

Ordinary panel data do not automatically identify detector calibration, conditional laws or sources. Missing premises remain unresolved. Available recommendations and finite ABCD returns do not establish economic recovery. Actual actions must remain distinct from recommendations. See the guide for release, support and fixed-objective requirements.

The fixed CM seed integration test covers fresh acquisition, 24 decisions, source verification, explicit repair, checkpoints and executed ABCD paths. It is not a statistical power or economic-improvement study. CI and cross-platform status must be assessed from actual workflow results.

## License

MIT. See [LICENSE](LICENSE).

This is the initial installation README; a fuller research overview and API documentation will be developed separately.
