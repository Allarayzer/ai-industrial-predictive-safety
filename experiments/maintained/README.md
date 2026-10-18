# Maintained scientific runners — protocol revision 2026-10-13

These runners are maintained copies derived from the frozen PeerJ sources at
`e083192c4adbe516650c153dbcaf542e6a910de2`. The original sources, manifest and
reference CSVs in `../peerj/` are unchanged. Outputs here are new validation
runs, not replacements for published numbers.

Run from the repository root after preparing the relevant local datasets:

```sh
python experiments/maintained/cmapss_rul_multiseed.py --help
python experiments/maintained/e2_cmapss_calib_split.py --help
python experiments/maintained/fd001_validation.py --help
```

Both write to `experiments/maintained/results/`. The maintained Bosch, E5, and
normalized RUL entry points under `benchmarks/` now write there too, so their
new outputs cannot overwrite `benchmarks/results/` or PeerJ reference CSVs.
Use `--data-dir` and a separate `--output-dir` for each run. The multi-seed and
E2 runners save a provenance manifest covering arguments, seeds, engine roles,
dataset bytes, source files, Python, package versions and numerical environment.
Resume rejects a changed identity or a previous CSV without its manifest before
fitting models. This detects changes; it does not make different hardware
bitwise equivalent or independently authenticate the original acquisition.
Frozen reference results are never replacements for missing new predictions.

## Bounded FD001 validation

`fd001_validation.py` is a separate one-seed diagnostic, with the same 32-cycle,
14-sensor input context for Ridge, Random Forest, LSTM-MSE and LSTM-quantile.
Fit/early stopping uses 85/15 whole training engines; official test endpoints
are held out. Hyperparameters are fixed, with at most 20 neural epochs and no
test-based tuning. The 300-second budget is checked between neural batches;
it is a cooperative limit, not a hard process timeout. Limit numerical threads
explicitly when running on a shared machine.

```sh
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src \
python experiments/maintained/fd001_validation.py \
  --data-dir "$CMAPSS_DIR" --data-manifest "$CMAPSS_MANIFEST" \
  --output-dir /tmp/fd001-validation --seed 42 --epochs 20
```

The acquisition manifest must contain file paths and SHA-256 values in a
`files` list; this runner verifies the three FD001 files. Use an empty output
directory. Saved predictions include every official test engine, raw and
sorted/clipped quantiles, and capped/uncapped targets. Source snapshots and
package versions identify the actual execution. Engine bootstrap intervals
condition on this single trained fit; they do not measure training-seed or
new-factory uncertainty. Nominal Q10–Q90 coverage is 80%, with no automatic
empirical coverage guarantee. A single FD001 run is not a full study replay.

## RUL

`cmapss_rul_multiseed.py` splits official training engines before fitting
operational normalization. Models fit on 85%, and the remaining 15% control
early stopping. Windows remain inside engine boundaries. An explicit
validation tuple reaches the neural models; the scaler fits on fitting data.
The official test split is untouched. Classical models still use pointwise
features, whereas neural models use windows; this is not equal-context evidence
of superiority. Seeds summarize variability, not independent industrial trials.
The ±20% endpoint metric is named accordingly and is not alpha-lambda accuracy.
Statistical comparisons reject duplicate rows, join matching seed IDs and report
the number of unmatched rows instead of pairing different seeds by position.
Fewer than two pairs yield no inferential p-value; all-tied differences yield
p=1. Holm correction covers six predeclared comparisons within each subset,
not all subsets. Cliff's delta is a marginal effect size, not a paired one.

`benchmarks/run_cmapss_rul_normalized.py` shares these preprocessing helpers.
Its optional ABC ensemble retains equal weights. Validation controls neural
early stopping, not ensemble weight selection.
Its `--aggregate-only` mode rebuilds the combined CSV from all per-subset CSVs
in `--output-dir`, preserving subsets not evaluated in the current invocation.
Schema, subset names, duplicates and finite values are checked before the
aggregate is replaced. Each write is atomic; the set of multiple CSV writes is
not a transaction. Keep different protocols/configurations in separate folders.

## Risk fusion

`e2_cmapss_calib_split.py` retains the outer 60/20/20 engine roles and reserves
15% of the initial 60% for early stopping (51 fit / 9 early / 20 calibration /
20 test for FD001). All windows remain inside engines. Each engine's first 15
cycles are excluded from every channel's comparison population. Early-stopping
engines never fit the model scaler. Weight calibration and final evaluation
remain disjoint. Lead time measures the distance to the last observed cycle of
a complete run-to-failure engine, including warnings inside the final 30 cycles.
It must not be applied to truncated official test histories. Missed warnings
are omitted from the median and counted separately with warning coverage;
a large median alone does not imply useful alerts. All stochastic components
receive the requested seed. Failed weight optimization aborts that seed instead
of silently reporting the fallback as an optimized ensemble.
These remain heuristic score channels with a fixed decision threshold.

## Bosch and E5

Bosch uses parsed calendar months and reserves whole later months near the
nominal 70/30 normal split. Older faults are excluded from its test. Unknown
dates fail explicitly. Its threshold remains the fitting-score percentile;
there is no independent threshold-calibration role. Pooled counts use exact
integers. Machine-AUC means have explicit `macro_*` names; pooled AUC is left
undefined because separate machines' score scales are not jointly calibrated.

E5 dates each decision at the last observed sample of a trailing window. The
first monitoring window ends after the reference period plus 12 hours, and
false-alert rates use this explicit monitoring exposure. The named ADWIN
approximation remains a historical KS surrogate, not River ADWIN. Periodic
retraining remains a schedule comparator, not a statistical drift detector.
E5 timestamps advance in minutes. Unobserved events are censored at the last
observed sample, with blank delay and an explicit event indicator. A scheduled
retraining beyond the observation period is not an observed event. Summaries
separate event/detection fractions from delay conditional on an observed event;
`--drift-multiplier 0` supplies a null control.

## Separate corrected study protocols

The additional `benchmarks/run_bosch_path_order_corrected.py` and
`benchmarks/run_experiment_e5_causal.py` preserve the reviewed study protocols
as separate entrypoints. Their split, censoring, output paths and schemas
differ from the calendar Bosch and null-control E5 described above. See
[their contracts and source identity](../../docs/bosch-e5-protocols.md).
The maintained library is retained; no new empirical results are implied by
this source integration.
