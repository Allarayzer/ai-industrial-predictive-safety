# Examples
End-to-end demonstrations of the `ai_cta` package.
## `quick_start.py`
Full-loop demo using synthetic sensor data:
1. Generate a multivariate stream.
2. Inject three kinds of anomalies (spikes, drifts, oscillations).
3. Fit an `IsolationForestDetector`.
4. Calibrate a conformal threshold at 5% target FAR.
5. Combine with rule-based `ChannelLimits`.
6. Run the `SafetyPipeline` with an alert callback.
Run it from the repository root:
```bash
python examples/quick_start.py
```
Expected output (metrics will vary slightly with random seed):
```
Generating synthetic training and test streams ...
  Train: 2000 samples
  Test : 1000 samples (342 anomalous)
Fitting IsolationForestDetector ...
  Engineered feature count: 27
  Training windows:         61
Calibrating threshold at alpha=0.05 ...
  Calibrated threshold: 0.888
  F1=0.800  precision=0.714  recall=0.909  FAR=0.211
Running SafetyPipeline on the test stream ...
  Events emitted: 937
  Alerts raised:  1
```
## Extending
Good next steps once the quick start works:
- Swap `IsolationForestDetector` for `HybridDetector` (requires
  `tensorflow`).
- Run the full C-MAPSS benchmark: `python benchmarks/run_cmapss_benchmark.py`
  after downloading the dataset (see `docs/benchmarks.md`).
- Plug a real webhook URL into
  `SafetyPipeline.make_webhook_callback` to drive an n8n or Node-RED
  automation flow.
