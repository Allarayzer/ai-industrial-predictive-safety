# PeerJ study: corrected experiment protocols

This directory makes the corrected scripts from the article workspace available
inside the repository. Only repository/output paths were adapted; algorithms and
reference CSVs were not edited. See `manifest.json` for source and result SHA-256
hashes and [version notes](../../docs/reproduction.md) before comparing runs.

## Protocol map

| Runner | Protocol / purpose | Reference output |
|---|---|---|
| `e2_cmapss_calib_split.py` | FD001: 60/20/20 disjoint engine groups for model fitting, weight calibration, evaluation | `experiment_e2_cmapss_calib_split.csv` |
| `cwru_leakage_safe.py` | Train normal 97/98, calibrate 99, test normal 100 plus fault recordings; non-overlapping windows | `cwru_leakage_safe.csv` |
| `ablation_calib_split.py` | Synthetic fusion ablation with separate weight-calibration/evaluation data | `ablation_calib_split.csv` |
| `e2_fusion_rul_variants.py` | Point, quantile and conformalized RUL channels | `e2_fusion_rul_variants.csv` |
| `e2_cost_ratio_sensitivity.py` | Cost-ratio sensitivity | `e2_cost_ratio_sensitivity.csv` |
| `e2_fusion_cost_objective.py` | Held-out BCE, Brier score and weighted classification cost | `e2_fusion_cost_objective.csv` |
| `conformal_coverage.py` | Marginal normal-score FAR experiment | `conformal_coverage.csv` |
| `conformal_conditional.py` | Per-regime FAR diagnostic | `conformal_conditional.csv` |
| `conformal_stratified_recalib.py` | Stratified thresholds and post-shift recalibration | `conformal_stratified_recalib.csv` |
| `cmapss_rul_multiseed.py` | Official C-MAPSS splits, eight-seed RUL comparison | `cmapss_rul_multiseed.csv` |

Additional study components retain their existing generators under `benchmarks`
(e.g. Bosch, classical baselines, detector/drift experiments). This directory is
not a new run or a complete republication of the journal's supplemental archive.

## Run from the repository root

```bash
pip install -e ".[benchmarks]"
python experiments/peerj/verify_manifest.py
python experiments/peerj/e2_cmapss_calib_split.py --help
python experiments/peerj/cwru_leakage_safe.py --help
# After downloading the relevant data:
python experiments/peerj/e2_cmapss_calib_split.py --n-seeds 10
python experiments/peerj/cwru_leakage_safe.py --n-seeds 10 --window 2048 --alphas 0.01,0.05,0.10
python experiments/peerj/ablation_calib_split.py --n-seeds 10 --n-samples 5000
python experiments/peerj/cmapss_rul_multiseed.py --subsets FD001,FD002,FD003,FD004 --seeds 8 --epochs 20
```

Data live under `benchmarks/data/`; see [download instructions](../../docs/benchmarks.md).
The recorded CWRU experiment excluded unreadable recording `175.mat` (80 fault
recordings used). Dataset changes, including availability of that recording,
change the replication conditions and must be reported. Cross-load calibration
and testing do not satisfy identical-distribution assumptions automatically.

Generated CSVs go to `results/` (gitignored). Reference CSVs are in
`reference_results/`, separate from new outputs. The SHA-256 check verifies file
integrity; it does not prove the scientific conclusions or rerun the experiments.
