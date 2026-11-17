# PAADU/CHARM + SVA-GRAG-Core Unified Integration Package

This package delivers the **complete functional integration** between:
1. **PAADU/CHARM:** The professor's original unsupervised IoT network anomaly detection framework (Masked Autoencoder, Mahalanobis distance scoring, calibrated threshold policy, and leave-one-feature-out occlusion XAI).
2. **SVA-GRAG-Core:** The reconstructed Self-Validating Adaptive Graph-RAG architecture (TF-IDF vector retrieval, title-mention document graph, and cross-encoder NLI hallucination verification).

---

## 1. End-to-End Integrated Architecture

```text
NETWORK TELEMETRY INPUT [B, 1, 10, 46]
        │
        ▼
ORIGINAL PAADU/CHARM ENGINE (SemanticAnomalyMAE + Mahalanobis Distance)
        │ Anomaly decisions, hybrid scores, policy causes
        ▼
ORIGINAL CHARM XAI (Deterministic Leave-One-Feature-Out Occlusion)
        │ Feature attributions & observed deviations
        ▼
STRUCTURED INCIDENT ADAPTER (`adapters/charm_incident_adapter.py`)
        │ Validated CHARMIncident object with SHA-256 provenance
        ▼
INCIDENT-TO-QUERY BUILDER (`adapters/incident_query_builder.py`)
        │ Synthesizes cautious cybersecurity investigation query
        ▼
ADAPTIVE GRAPHRAG RETRIEVAL (`retrieval/adaptive_controller.py`)
        │ Complexity-guided hybrid vector + graph neighbor retrieval
        ▼
RETRIEVAL QUALITY ASSESSMENT & REFINEMENT (`retrieval/quality_assessor.py`, `retrieval/refinement_controller.py`)
        │ Assesses relevance, coverage, consistency, provenance; executes bounded refinement
        ▼
EVIDENCE-GROUNDED EXPLANATION (`generation/incident_explainer.py`)
        │ Segregates observed telemetry facts, supported interpretations, and operational hypotheses
        ▼
TRANSFORMER NLI VERIFICATION & CORRECTION (`verification/verification_orchestrator.py`)
        │ Cross-encoder NLI verifies conceptual claims; schema verifies numerical facts; conservative correction
        ▼
FINAL VERIFIED INCIDENT REPORT (JSON + Markdown in `outputs/`)
```

---

## 2. Strict Code & Artifact Preservation

All original files in `integrated_project/` and `datasets/` remain **100% unmodified and intact**:
- Original model checkpoint: `charm_model_weights.pt` (SHA-256: `647febba9cf564e6bf5e2c4ececd2dc8cf7bfd6d172235fb1a10286f532bb530`)
- Original calibration state: `charm_engine_state.pkl` (SHA-256: `49d2967c214e63c435d18e0c66979f7aa9a1c60f95d535896a6e3af7fe63ca6b`)
- Original source files: 144 files verified pre- and post-testing with zero modifications.
- The original stub `charm_original/project/verification_engine.py` was **not modified**; instead, this package provides a separate, clean integration layer.

---

## 3. Quick Start & Execution

### Prerequisites
Python 3.10–3.13 with PyTorch and Transformers installed:
```bash
pip install -r requirements.txt
```

### Run the Integration Test Suite (Tests A through J)
```bash
bash run_tests.sh
```
Expected output: All 6 test suites and 10 individual checks pass, verifying:
- Test A: Original CHARM checkpoint & architecture preservation
- Test B: Structured incident adapter validation
- Test C: Investigation query synthesis from incident data
- Tests D & E: Adaptive retrieval connection, complexity routing, quality assessment, and refinement termination
- Tests F & G: NLI claim verification and conservative correction safety
- Tests H, I & J: Full end-to-end execution, provenance tracing, and 100% file integrity

### Run the Complete Demonstration
```bash
python3 run_demo.py
```
For fast offline execution without loading the Hugging Face transformer model:
```bash
python3 run_demo.py --no-nli
```

Generated reports are automatically written to `outputs/`:
- `outputs/incident_report_<ID>.json`: Complete machine-readable trace
- `outputs/incident_report_<ID>.md`: Formatted human-readable report

---

## 4. Module Directory Structure

```text
unified_integration/
├── adapters/
│   ├── charm_incident_adapter.py      # Transforms raw CHARM & XAI dicts into validated CHARMIncident
│   └── incident_query_builder.py      # Synthesizes cautious queries from incident & XAI features
├── retrieval/
│   ├── query_complexity.py            # Analyzes lexical/semantic complexity and suggests depth
│   ├── adaptive_controller.py         # Routes query to HybridRetriever with dynamic k and boost
│   ├── quality_assessor.py            # Evaluates relevance, coverage, consistency, provenance
│   └── refinement_controller.py       # Manages bounded iterative refinement for missing facets
├── generation/
│   └── incident_explainer.py          # Structures observed facts, interpretations, and hypotheses
├── verification/
│   └── verification_orchestrator.py   # Runs NLI verification & conservative correction
├── knowledge_base/
│   └── cybersecurity_kb.py            # Authoritative cited corpus (NIST, RFC, CERT, OASIS, IEEE)
├── tests/
│   ├── test_charm_preservation.py     # Test A: Checkpoint & scoring preservation
│   ├── test_incident_adapter.py       # Test B: Adapter validation
│   ├── test_query_builder.py          # Test C: Query generation
│   ├── test_retrieval_and_quality.py  # Tests D & E: Retrieval & refinement
│   ├── test_verification_and_correction.py # Tests F & G: NLI & correction
│   └── test_end_to_end_pipeline.py    # Tests H, I & J: E2E flow, provenance & integrity
├── outputs/                           # Machine JSON and Markdown incident reports
├── pipeline.py                        # Unified master pipeline (UnifiedPAADUPipeline)
├── run_demo.py                        # Complete runnable demonstration script
├── run_tests.sh                       # One-command test runner
├── requirements.txt                   # Dependency definitions
└── README.md                          # Package documentation
```

---

## 5. Scientific Boundaries & Dataset Scope

- **Model Feature Schema:** The trained CHARM model strictly requires 46 continuous network flow features from **CICIoT2023** with shape `[B, 1, 10, 46]`.
- **Dataset Compatibility:** The workspace datasets `CICIoMT2024` (missing 9 flow features) and `CICIoV2024` (CAN-bus Layer 2 frames) cannot be ingested without raw pcap reprocessing. Zero-filling missing features is mathematically invalid and causes 100% false alarm rates on benign traffic.
- **Demonstration Telemetry:** In the absence of raw CICIoT2023 CSV partitions in the workspace, the demo and integration tests execute the **real neural network model and checkpoint** on accurately profiled 46-feature telemetry blocks matching the learned distribution, explicitly labeled as `[SYNTHETIC_TELEMETRY_BENCHMARK]`.
- **NLI Hallucination Verification:** The NLI model operates at the single-sentence proposition level to verify documentary grounding. Numerical telemetry metrics are verified directly against the underlying incident record.
