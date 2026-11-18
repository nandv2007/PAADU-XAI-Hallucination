# Independent Evaluation Benchmark Test Suite
## PAADU/CHARM + SVA-GRAG + NLI External Validation

This directory contains external benchmark incidents and independently adjudicated ground-truth claim annotations for scientific reproducibility audits.

### Structure:
- `external_incidents.json`: Independent incident ground-truth records containing exact model anomaly scores, decision thresholds, trigger policies, and top XAI feature attributions.
- `external_claims.json`: Test claims with reference `gold_label` adjudications (`SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`) created independently of the production verifier.

### Isolation Policy:
This dataset is strictly isolated from development unit tests and internal regression test fixtures to evaluate out-of-distribution generalization and prevent prompt/grammar overfitting.
