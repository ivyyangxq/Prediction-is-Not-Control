# Software delivery — 7 October 2026

Current delivery: `0.2.0.dev2026100702`, Python and Dynare-facing archives in `software/pinc_toolbox/dist/release_0.2.0.dev2026100702/`. Both READMEs now describe the current scientific scope and two-stage outputs. All previous packages and scientific outputs are retained. Version0701 and its initial saved-export outputs are retained as a failed denominator check;0702 is the corrected delivery.

| Theory / research object | Actual interface | Output | Verified / unsupported |
|---|---|---|---|
| RND2 actual-state Bellman | `RND2GQ.q_values(evaluation='bellman', remaining_horizon=10)` explicitly selected by the scientific H10 adapter; JSON native query exposes those settings | Q values, support mask, chosen action and method label | Current-state evaluation distributed and Python/Octave tested on3 Oct. Packaged four-path engineering example isH3, not theH10 detector experiment. Future auxiliary continuation is approximated; true economicQ/optimal repair unsupported. |
| Released dated auxiliary information | `prepare_rnd2_learner`, branch issue/observe; historical joint nodes retain dated pairing | Updated auxiliary records and preissued base forecast receipts | Rolling full-RA engineering mechanics tested; no latent recovery or general sufficient-state claim. |
| Conditional two-stage model-involvement result | Alarm first, finite no-model-catalogue follow-up under declared matched calibration and entry law | First alarm, final involvement, unresolved, no alarm and unavailable are distinct | Conditional rank/error theory requires true-law catalogue coverage and follow-up uniform conditional validity over full entry history. Execution cannot verify those assumptions; source posterior/full classification unsupported. |
| KS attempt017 application | Research runner `experiments/ks_main_application_20261005/attempt017_fixed128_macro/run.py`, importing earlier research adapter/host dependencies | FULL employment residual measured with native fitted-Q support and raw previous-action anchor; monitor70–77 and alarm-entered source78–85; N/PUBLIC_MACRO79-slot banks | Frozen pre-event G/Q, causally updatedaux; **not full release-by-release G/Q refitting RA**. Scientific KS adapter/host/sampler **is not inside either software package**. KS fixed-price PE, not solvedGE/welfare/repair or direct household causal localization. |
| Completed KS terminal readback | NEW installed `pinc_toolbox.ks_saved_exhibits_cli`; Dynare-facing `pinc_ks_saved_exhibits.m` | Recomputed15-row main table and outcome summary, paired economic means; three PNG/PDF pairs; receipt | Reads only four saved readback files, no runner import,0fits/0simulation/0CAL draws.128 macro seeds retained,127 economic paths available. No scientific128 rerun. |

## Reliable saved-output command

Install the0702 wheel in a Python environment with its declared dependencies and optional `matplotlib>=3.6` (`reporting` extra). The source input is the already corrected terminal readback, including eighteen fixed calibration repairs and one retained fatal macro seed.

```sh
/absolute/installed/python -I -m pinc_toolbox.ks_saved_exhibits_cli \
  --input-dir "$PWD/experiments/ks_main_application_20261005/attempt017_fixed128_macro" \
  --output-dir /absolute/new/ks_saved_exhibits
```

Required caller files: `017_OUTCOMES.csv`, `017_ECONOMIC_RESPONSES.csv`, `017_BOUNDARY_READBACK.csv`, `017_READBACK_STATUS.json`. These are not bundled. Outputs are written to a new directory so original results remain intact. A global terminal fatal row is carried into every declared case/readout cell as technical unavailable. Its scientific alarm remains unknown; no failure is silently treated as no alarm. Pointwise Wilson intervals describe final-report delivery with128 planned economies. Economic means condition on127 available paths.

For Octave/MATLAB, add the new archive's `matlab` directory and call:

```matlab
receipt = pinc_ks_saved_exhibits(saved_csv_directory, new_output_directory, absolute_installed_python);
```

The bridge invokes the same installed Python exporter. It does not contain a KS simulator or rerun the scientific adapter. Original research `final_saved_readback_017.py` / `plot_017_saved_outputs.py` still exist, but write in the scientific source directory; the new entry is the output-directory-preserving alternative for already completed readback. Cosmetic figure layout can differ from the original three exhibits.

## Actual verification this turn

- Necessary existing code checks run once:10passed in3.61s (`test_counterfactual.py`, `check_native_gq_cli.py`). No repeated broad audit.
- The0702 wheel is actually installed in `/tmp/pinc_ks_saved_20261007`. `python -I` loads its `ks_saved_exhibits_cli` from installed `site-packages`; system-site dependencies were reused, not freshly downloaded.
- Python saved-only regeneration completes:15case/readout cells,128terminal macro seeds,127paired economic paths. Counts and Wilson intervals match the original main table (absolute tolerance2e-15); every outcome-summary count matches. The initial0701 check caught a dropped global fatal row; both the failed outputs/package and corrected output are retained.
- Actual Octave9.4.0 calls the new bridge from independently unpacked0702 archive. Main table, outcome summary and economic mean CSVs are exactly equal to installed Python outputs. Each path writes all three PNG/PDF pairs. MATLAB and Dynare`.mod` execution remain untested.
- Receipt/check: `SOFTWARE_CHECK.json`, `ks_saved_python_corrected/RECEIPT.json`, `ks_saved_octave/RECEIPT.json`; reproduction driver: `ks_saved_octave.m`. No scientific data generation or calibration is executed.

Updated source: both READMEs, `docs/KS_SAVED_DELIVERY_20261007.md`, `ks_saved_exhibits_cli.py`, its `.m` bridge, project version / optional reporting dependency and lightweight builder documentation inclusion. Scientific Framing/Theory/Matrix manuscripts and128results are untouched.
