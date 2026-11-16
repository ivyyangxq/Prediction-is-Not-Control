Start with **START_HERE.md** for the step-by-step English guide.

# Installed Python GQ Tool

Create an environment with Python 3.10 or later and install the wheel (pip resolves the declared dependencies):

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install /path/to/gq_tool-1.0.3-py3-none-any.whl
gq-tool --check
gq-tool
```

Windows activation: `.venv\Scripts\activate`. `gq-tool` copies the packaged notebook to a new temporary location and starts Jupyter Notebook. Run its launch cell, then follow import → variable mapping → reward form → explicit parameters → fixed session. To keep your own notebook: `gq-tool --copy-notebook my_gq_tool.ipynb`. The installed package contains product_v2, reward/ABCD extensions, reference host assets, the English user guide and persistence controls. No checkout, researcher path or PYTHONPATH is needed.

The persistence panel saves/resumes a checkpoint and ingests an updated full CSV using the existing mappings/unit IDs. Previously known values cannot be revised; only not-yet-processed missing records and new dates may arrive. New ingestion invalidates old source authorization. Load only your own trusted Python checkpoints (cloudpickle contains executable Python objects).

Installed validation used a fresh venv with existing host dependencies (`--system-site-packages`) and local wheel `--no-deps`, from outside the checkout with empty PYTHONPATH. Real CSV import, reward wizard, decisions, checkpoint continuation and four executed ABCD branches passed. A fully offline dependency bundle is not supplied. Dependencies are in pyproject.toml.

The same scientific release, support, actual-action and identification requirements apply as in the English user guide. Arbitrary ordinary panels do not automatically identify source or conditional proposal laws. Native source fixtures/test null banks are engineering demonstrations, not additional scientific results.
