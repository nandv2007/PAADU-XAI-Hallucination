# Integration Baseline Validation & System Provenance Report

**Validation Date:** 2026-11-17  
**Project:** `PAADU-XAI-Hallucination-Research`  
**Evaluator:** Senior AI Research & Validation Engineer  
**Status:** 100% UNMOCKED / VERIFIED STANDALONE  

---

## 1. System Integration Overview

The system under research continuation integrates two previously independent components:

1. **Professor's PAADU/CHARM Anomaly Detector**:
   - **Neural Architecture**: `SemanticAnomalyMAE` (Masked Autoencoder built upon PyTorch `TransformerEncoder`).
   - **Parameter Count**: Exactly 185,007 weights strictly loaded from `charm_model_weights.pt` (SHA-256: `647febba9cf564e6bf5e2c4ececd2dc8cf7bfd6d172235fb1a10286f532bb530`).
   - **Latent Bottleneck**: 64-dimensional sequence embeddings.
   - **Scoring Engine**: Standardized reconstruction MSE combined with latent Mahalanobis distance ($D_M$):
     $$s(x) = 0.2 \cdot z_{\text{recon}} + 0.8 \cdot z_{D_M}$$
   - **Calibration**: `VolumeAdaptiveRelaxer` with calibrated hard $z$-floor threshold $\tau = 2.5000$ loaded from `charm_engine_state.pkl` (SHA-256: `49d2967c214e63c435d18e0c66979f7aa9a1c60f95d535896a6e3af7fe63ca6b`).
   - **XAI Method**: Deterministic **leave-one-feature-out occlusion** over all 46 CICIoT2023 flow features (occluding to empirical benign mean $0.0$).
   - **Explicit Finding**: The model **DOES NOT use SHAP, LIME, or Integrated Gradients**.

2. **Student's SVA-GRAG-Core Retrieval & Verification**:
   - **Hybrid Retrieval**: Scikit-Learn TF-IDF vectorizer over tokenized cybersecurity reference documents paired with a NetworkX document co-citation graph.
   - **Adaptive Controller**: Dynamic query complexity analysis adjusting $k \in [3, 8]$ and graph boost $\beta \in [0.15, 0.40]$.
   - **Quality Assessor**: Multi-dimensional composite quality metric $Q = 0.35 R + 0.35 C + 0.15 K + 0.15 P$.
   - **Bounded Refinement**: Iterative query reformulation targeting missing protocol facets ($max\_iterations = 2$).
   - **Incident Explainer**: Deterministic rule/template evidence synthesizer (`unified_integration/generation/incident_explainer.py`) producing structured `FactualClaim` propositions. **The system DOES NOT invoke an external LLM for drafting**.
   - **NLI Hallucination Verifier**: Fine-grained cross-encoder transformer (`cross-encoder/nli-MiniLM2-L6-H768`, 105 weight tensors, ~66.9M parameters) evaluating claim entailment, contradiction, and neutral confidence.
   - **Correction Policy**: Retains verified telemetry facts, retracts contradictory statements, and badges unverified operational hypotheses with epistemic uncertainty markers.

---

## 2. Cryptographic Checksum Baselines

All core model checkpoints and reference configurations match verified cryptographic baselines:

| Artifact Name | Relative Path | File Size | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| `charm_model_weights.pt` | `models/charm_model_weights.pt` | 770,692 bytes | `647febba9cf564e6bf5e2c4ececd2dc8cf7bfd6d172235fb1a10286f532bb530` |
| `charm_engine_state.pkl` | `models/charm_engine_state.pkl` | 35,433 bytes | `49d2967c214e63c435d18e0c66979f7aa9a1c60f95d535896a6e3af7fe63ca6b` |
| `training_metadata.json` | `original_system/integrated_project/charm_original/artifacts/training_metadata.json` | 1,061 bytes | `63fe8a5b840f7a0dc3207e6fe2825a8f65124c047b1c27d80667ff2ca2be72b8` |

---

## 3. Baseline Test Suite Verification Results

Running `./run_tests.sh` reproduces the following verified pass results:

| Test Identifier | Component Evaluated | Status | Runtime | Description |
| :--- | :--- | :---: | :---: | :--- |
| **Test A** | CHARM Preservation | **PASS** | 0.148s | Confirms 185,007 weights loaded strictly without mismatch; deterministic scoring verified. |
| **Test B** | Structured Incident Adapter | **PASS** | 0.000s | Validates raw CHARM numpy array conversion to typed `CHARMIncident` dataclass. |
| **Test C** | Investigation Query Builder | **PASS** | 0.000s | Synthesizes hypothesis-neutral queries from XAI top features. |
| **Test D & E** | Retrieval & Quality Assessor | **PASS** | 0.011s | Verifies TF-IDF + graph boosting and bounded refinement loop ($Q \in [0, 1]$). |
| **Test F & G** | Transformer NLI & Correction | **PASS** | 4.635s | Live forward-pass inference of `cross-encoder/nli-MiniLM2-L6-H768`; contradiction retraction verified. |
| **Test H, I & J**| End-to-End Pipeline & Integrity| **PASS** | 8.812s | End-to-end telemetry execution to verified report; SHA-256 integrity verification across monitored files. |

**Overall Status**: **10/10 TESTS PASSED (100% UNMOCKED)**.

---

## 4. Standalone Zero-Leakage Audit

Executing `python3 verify_standalone.py` confirms:
1. **0 Leaked Modules**: All 20 project modules (`charm_engine`, `original.paadu_benchmark_v7`, `pipeline`, `adapters`, `retrieval`, `verification`) load strictly from `PAADU-XAI-Hallucination-Research`.
2. **Read-Only Preservation**: The original `PAADU_Testing` workspace remains untouched and completely uncoupled from execution.
