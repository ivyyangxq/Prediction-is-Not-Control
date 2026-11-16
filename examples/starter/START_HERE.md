# GQ Tool for Python

This is an independently installable Python package. No research checkout, PYTHONPATH setting or manual extension copying is required. The `gq-tool` command opens a Jupyter notebook containing the unified workbench.

## Install and launch

Use Python 3.10 or newer. Create an environment and install the downloaded wheel:

```sh
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install /path/to/gq_tool-1.0.3-py3-none-any.whl
gq-tool --check
gq-tool
```

pip resolves the dependencies declared in pyproject.toml. The wheel does not include a Python interpreter or an offline dependency bundle. `--check` constructs the menus; it does not run a scientific validation.

Run the notebook's launch cell with Shift+Enter. If a browser does not open automatically, use the local URL printed by Jupyter. Keep the launching terminal open. The default notebook uses a temporary directory and a kernel bound to the launcher's Python environment. Export results to a directory you own.

To keep a starter folder with example CSV files, settings, notebook and this guide:

```sh
gq-tool --copy-example my_gq_example
python -m notebook my_gq_example/GQ_tool.ipynb
```

For notebooks opened manually, select the environment containing gq-tool. If necessary, register that environment with `python -m ipykernel install --user --name gq-tool --display-name "Python (GQ Tool)"`, then select it. Direct `gq-tool` launches configure the kernel automatically.

## First run: the small example

Keep **Use the built-in small example** selected. Click **1 Load data and configure columns**, then **2 Confirm mappings and timeline**. Keep the default learning interval and grid.

1. Confirm the supplied inputs and that action has the same meaning as the logged actual action.
2. Enter a reward form, such as `x-cost*a*a`.
3. Classify `cost` as a fixed numerical parameter, explicitly enter `2`, and confirm.
4. Click **4 Create fixed-objective session**, then **Advance one period** or **Run remaining periods**.

This example validates operation; it does not automatically supply source identification evidence. Without applicable evidence, do not expect MODEL or diagnosis-guided repair. Refusal and unresolved results can be correct outcomes.

## Import your data

Clear the built-in example selection and provide a panel CSV and a public-input CSV. Map columns explicitly. The files produced by `--copy-example` show the required format.

| File | Example column | Meaning |
|---|---|---|
| Panel | unit | Complete unit identifier |
| Panel | date | Decision period t |
| Panel | released | Period when the actual transition becomes visible |
| Panel | x | Current physical state |
| Panel | executed | Actual executed action, distinct from a recommendation |
| Panel | anchor | Previous actual action; otherwise construct only from consecutive logs |
| Panel | next_x | Actual successor state |
| Public | publication | Actual publication period |
| Public | macro_W | Observed public input |

A common-W table has one value per publication period. A unit-specific proxy also requires a unit key; select that scope explicitly. The importer joins current and successor values by publication time. Neither input is a latent truth.

Check `learn_start < learn_end <= start <= end`. Integer model periods are easiest to audit. With date labels, review the displayed period mapping and release ordering. `path` selects the query unit. `window` controls the learning window; full RA also requires the actual complete dated prefix. Confirm action grid, discount, horizon and every reward parameter before creating the session.

## Interpret results

- **available**: current support, memory and numerical conditions permit a recommendation. This does not establish causal or economic improvement.
- **unavailable**: inspect the reason. Missing actual actions, inadequate support, absent exact action-memory nodes, rank failure or an incomplete full-RA prefix can cause refusal. Never fill missing actual actions with recommendations.
- **unresolved**: current evidence does not settle source classification. Do not replace this with MODEL.

Eta and reference scales are fixed measurement choices. Do not widen them during execution to regain support. Physical G excludes auxiliaries; auxiliaries enter downstream Q. Reward remains fixed throughout the session. Create a new session if the objective changes.

## Detection, source evidence and repair

Ordinary panel data do not automatically supply valid detector calibration, conditional proposal laws or source models. Evidence controls execute real requests and reject copied MODEL labels. Missing identification, noise, coverage or calibration premises remain unresolved.

The controlled acquisition directory route applies only to a matching reset/common-innovation benchmark. It rechecks the panel, monitoring receipt, calibration and curve endpoints. Do not reuse the internal test's source directory for unrelated data.

Accept a selected update only when evidence is valid and applicable to the current session. You may also decline repair. Repair preserves reward and does not bypass support refusal. ABCD executes four counterfactual paths. Public-proxy hold is different from latent hold; returns and interaction are not regret or source evidence.

## Save, resume and ingest releases

JSON/CSV exports are review artifacts, not resumable checkpoints. Choose a new export directory. Use the persistence controls for checkpoint save/restore, and load only trusted checkpoints.

For new releases, supply the updated full CSV using existing mappings and unit identifiers. Known values cannot be revised. Only unprocessed missing future records and new dates may be added. Ingestion invalidates old source authorization.

## Reproduce validation

The source and ready archives include tests:

```sh
python tests/test_installed_product.py
python tests/test_installed_launcher.py
python tests/test_one_seed_end_to_end.py --output NEW_OUTPUT_DIRECTORY
```

The one-seed test uses seed 3260040000 and the CM controlled history. It executes fresh acquisition, detection, source verification, actual widget callbacks, optional repair, export, checkpoint and ABCD. It produces acquisition files and takes longer than constructing menus. See ONE_SEED_TEST.md for results and scope.

The validation environment reused existing host dependencies in a fresh venv; this is not an offline bootstrap test. A single controlled seed validates one integration path, not statistical power, test size or economic recovery. Widget callbacks and a real Jupyter kernel were tested; browser layout was not screenshot-audited.
