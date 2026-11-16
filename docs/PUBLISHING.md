# Publication checklist

The target repository is https://github.com/ivyyangxq/Prediction-is-Not-Control. This URL is publication metadata; repository creation and upload must be verified separately.

1. Build: `python -m build`.
2. Validate metadata: `python -m twine check dist/*`.
3. Install the wheel in a new environment and run tests outside the checkout.
4. Push the reviewed source tree to the public GitHub repository.
5. Create a version tag and GitHub Release for v1.0.3.
6. Attach `gq_tool-1.0.3-py3-none-any.whl`, `gq_tool-1.0.3.tar.gz` and `SHA256SUMS.txt`. Use release/NOTES_v1.0.3.md as release notes.
7. Inspect actual GitHub Actions results; configured CI is not evidence of a successful remote run.

Do not commit build directories, local checkpoints, raw acquisition logs, credentials or environment files. The GitHub source tree and release assets serve different purposes; release binaries do not belong in normal source commits.

PyPI publication is separate. It requires a controlled PyPI project name, an account and a suitable publishing authorization or trusted publisher. No PyPI upload is configured or claimed here. Configure that only when the owner is ready. Keep repository metadata and documentation explicit about where installation artifacts are available.

The initial README contains installation and scope information only. A fuller research overview and API narrative will be written separately.
