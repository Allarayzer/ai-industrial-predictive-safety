# n8n workflows

This directory holds exported [n8n](https://n8n.io) workflow definitions
that orchestrate the AI-CTA predictive-safety pipeline.

The monograph (Chapter 11) describes the following reference workflows:

| File | Purpose |
|------|---------|
| `alert_on_high_risk.json` | Fires when the `/score` endpoint returns `alert_level >= Critical`, creates an incident ticket in the EAM system, and notifies the on-call engineer via Slack/Telegram. |
| `scheduled_retrain.json`  | Triggers `OnlineCalibrator` recalibration on a cron schedule (by default daily at 02:00 UTC). |
| `shadow_mode_compare.json` | Forwards telemetry to both the legacy SCADA alarm logic and the AI-CTA pipeline for comparison (used during Phase 2 of the rollout protocol, § 11.2). |

## How to import

1. Install n8n (see [docs.n8n.io/hosting](https://docs.n8n.io/hosting/)).
2. In the n8n UI, select **Workflows → Import from File**.
3. Choose the `.json` file, review the credentials placeholders, and
   attach your environment-specific secrets (API keys, SMTP, database
   connection strings).
4. Activate the workflow.

## How to export

After authoring a workflow in n8n:

```
n8n export:workflow --output=./workflow.json --id=<workflow_id>
```

and commit the JSON file to this directory.

## Security notes

- **Never commit credentials** alongside workflow JSON. Credentials must
  reside in the n8n credential store or in environment variables, never
  in version control.
- Workflows that expose webhooks should be behind an authenticated
  reverse proxy and rate-limited.
- For workflows that mutate external state (e.g. open EAM tickets),
  always use `workflowExecutionTimeout` and an idempotency key to
  prevent duplicate actions on retry.

Reference: monograph § 11.5 (Security of n8n in an industrial context).
