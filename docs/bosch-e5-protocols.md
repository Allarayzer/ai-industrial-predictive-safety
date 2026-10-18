# Bosch and E5 protocol entrypoints

Choose the protocol explicitly. These entrypoints coexist because their data
roles and output contracts differ; their results are not interchangeable.

| Entrypoint under `benchmarks/` | Protocol and output contract |
|---|---|
| `run_bosch_benchmark.py` | Maintained calendar-month split; whole later normal months form the test and older faults are excluded. Writes under `experiments/maintained/results`. |
| `run_bosch_path_order_corrected.py` | Reviewed study protocol: lexicographic path order, first 70% of normal recordings for fitting, remaining normals plus all faults for test. Path order is not validated acquisition chronology. Saves exact counts, raw recording IDs/scores/labels/thresholds and split metadata. |
| `run_experiment_e5.py` | Existing maintained E5 API and output schema, including `--drift-multiplier 0` null control. Its unobserved-event horizon is the observation end. |
| `run_experiment_e5_causal.py` | Reviewed causal-observation protocol. Decisions are dated at the last available sample, and an unobserved detector event is censored at the last evaluated post-onset decision. Observation end and trailing unchecked samples remain distinct. Saves streams and decision traces. No null-control CLI parameter. |

## Using the separate study entrypoints

Use a source checkout and its local library. Bosch additionally needs `h5py`;
the lightweight `bosch` extra declares it without TensorFlow; `benchmarks` and
`all` also include it. E5 requires only the core dependencies. No data are included and no download is performed. The source distribution
includes the two new script entrypoints and this protocol document; the wheel
contains the installable `ai_cta` library, not repository benchmark scripts.

```bash
python -m pip install -e '.[bosch]'
PYTHONPATH=src python benchmarks/run_bosch_path_order_corrected.py --help
PYTHONPATH=src python benchmarks/run_experiment_e5_causal.py --help

# Explicit future run examples; both destination directories must be new.
PYTHONPATH=src python benchmarks/run_bosch_path_order_corrected.py \
  --data-dir /path/to/bosch/data --seed 0 --ordering-policy lexicographic \
  --output-dir ./results/bosch-path-order-seed0
PYTHONPATH=src python benchmarks/run_experiment_e5_causal.py \
  --n-seeds 10 --pre-drift-days 40 --post-drift-days 20 \
  --output-dir ./results/e5-causal-series
```

Bosch defaults to `benchmarks/results/bosch-corrected` if `--output-dir` is
omitted. It atomically reserves a new directory before fitting and refuses an
existing directory, including an empty one. E5 requires an explicit new output
directory. Neither entrypoint overwrites historical CSVs. Bosch uses
`IsolationForest(n_jobs=-1)`; set an explicit worker/native-thread budget in the
launch environment before a real run. These commands do not impose a CPU or
memory quota.

Bosch pools threshold decisions through exact TP/FP/TN/FN. Its machine ROC-AUC
and average precision (AP) are reported separately and macro averaged over
machines with defined metrics; their contributing-machine count is retained.
Raw scores from separately fitted machine detectors have no common calibrated
ranking scale, so a pooled raw-score AUC is not reported. The threshold is the
95th percentile of fitting scores, not independent threshold calibration or a
conformal guarantee.

Both E5 versions compare KS with an ADWIN-style KS heuristic, not River ADWIN.
Periodic retraining is a calendar action, not a drift detector. The causal
protocol reports delays in an abstract clock of 60 samples per labelled hour;
relabelled timestamps do not establish physical sampling units. Delay means are
conditional on observed events, with censoring and exposure reported separately.
It evaluates a covariate mean shift, not a complete deployed safety system.

## Source identity and evidence boundary

The two separate entrypoints are byte-identical copies of the reviewed source
snapshot with tree SHA256
`af432cd1d1e97e5d24b92808754a2e060bd458a4e1449d464c94f005fdb65112`.
This identifier is a snapshot checksum, not a Git commit. Their original
filenames survive inside the copied source docstrings; use the filenames in
this page when invoking the integrated copies.

| Source file | SHA256 |
|---|---|
| `run_bosch_path_order_corrected.py` | `f8b80b8a5bee02058c2cbccdaababeb1abae52ca7bb2dc6a353cb4831f124e48` |
| `run_experiment_e5_causal.py` | `f2f710ae214a9aa4b710bc61c86eb6545ed495a4ffbcda876f65b9ce94a90a3a` |
| `src/ai_cta/data.py`, same in reviewed and maintained sources | `54a3444cac7074e8b9106b4e1cd2b043f86cdd0d7393b50c9cca07b71f802c81` |
| Reviewed snapshot `src/ai_cta/drift_detector.py` | `464de3cc04d6cd9a16bf252e61546f457cbdf5ce4a63041fd860a02b7f5b1a80` |
| Maintained `src/ai_cta/drift_detector.py` retained here | `75a0d406437eae5b0008e9d2d6e0bd54c6a3bdba5d78914c6e335a3ec4f6a518` |

The maintained DriftDetector retains the KS calculation and adds finite/empty
input validation plus unrelated PSI corrections. Its different source hash is
part of this integration's identity; the old library is not copied over it.
The E5 metadata hashes the local on-disk runner and dependency files. Those
hashes alone do not prove which installed module was imported; use the local
`PYTHONPATH=src` launch shown above and preserve the environment record.

Saved numerical results from the reviewed snapshot remain evidence for that
snapshot and its recorded dependencies. This integration adds bounded software
regressions and a package build; it does not generate a new scientific series,
prove numerical equivalence across dependency versions, or establish system
superiority. The separate FD001 validation and frozen PeerJ 72-job evidence
retain their own source and protocol identities.
