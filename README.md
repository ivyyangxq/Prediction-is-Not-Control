# Prediction is Not Control

Python and Dynare-facing packages for the GQ / RA-GQ toolbox.

## Downloads

The current research development release is **0.2.0.dev2026100702**. Download the Python ZIP, Dynare ZIP, or Python wheel from [Releases](https://github.com/ivyyangxq/Prediction-is-Not-Control/releases).

- [Python package and instructions](python/package/README.md)
- [Dynare-facing package and instructions](dynare/package/README.md)

The Dynare-facing package uses MATLAB/Octave bridges to the installed Python core. It is not a separate Dynare-native solver.

## Main algorithm

This release retains the existing floating previous-action anchor and auxiliary state `(delta_gamma, delta_m, e_t)`. The first two coordinates measure changes in fitted action response and baseline predictions between G vintages; the third is the observed first state coordinate minus its previously issued BASE forecast. Historical auxiliary records remain paired with their dated observations.

The new stagewise epsilon auxiliary-state trial from 8 October 2026 is experimental and is not included. Experimental repair routines are not included.

## Validation and scope

The installed four-path Bellman engineering example completed on 8 October 2026. Earlier Python/Octave package checks and saved KS exhibit regeneration are described in [delivery notes](SOFTWARE_DELIVERY.md). MATLAB and Dynare model execution were not tested. The KS scientific simulator and research data are not bundled; the package includes a saved-exhibit exporter.

This is a research development release. Detection guarantees require the stated support, calibration and candidate-law assumptions; empirical proportions are not posterior source probabilities.
