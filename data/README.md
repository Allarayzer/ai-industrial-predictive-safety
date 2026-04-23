# Data
This directory is reserved for user-supplied or generated datasets.
It is intentionally kept empty in the repository; benchmark datasets
(C-MAPSS, CWRU) are not redistributed here because of their respective
licensing terms.
See [`../docs/benchmarks.md`](../docs/benchmarks.md) for download
instructions and expected layout.
For quick experimentation without downloading anything, use
`generate_synthetic_stream` and `inject_anomalies` from
`ai_cta.utils`:
```python
from ai_cta.utils import generate_synthetic_stream, inject_anomalies
df = generate_synthetic_stream(n_samples=1000, random_state=0)
contaminated, labels = inject_anomalies(df, n_anomalies=20, random_state=0)
```
