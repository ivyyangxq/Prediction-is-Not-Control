# Contributing

Create a virtual environment, then install the project and development tools:

```sh
python -m pip install -e ".[dev]"
python tests/test_installed_product.py
python tests/test_installed_launcher.py
python tests/test_one_seed_end_to_end.py --output NEW_OUTPUT_DIRECTORY
python -m build
python -m twine check dist/*
```

Test wheel installations outside the checkout as well. The one-seed test runs the controlled CM benchmark; it is not a broad scientific campaign. Keep outputs outside the repository. Preserve actual-action, release-time, support-refusal, fixed-reward and source-premise guards. Include the runtime and exact scope when reporting validation. Do not commit credentials, personal paths, private data, checkpoints or generated acquisition logs.

Open an issue before major API or scientific changes. Submit changes through a pull request with relevant tests. Documentation and user-facing text must be English.
