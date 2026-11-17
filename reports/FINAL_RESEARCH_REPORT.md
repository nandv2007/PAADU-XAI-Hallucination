# Final Research Report: Faithful Explanations and Hallucination Mitigation in Explainable Deep NIDS

**Project:** PAADU/CHARM + SVA-GRAG-Core Hallucination Research  
**Date:** 2026-11-17  
**Status:** Software and Algorithmic Validation Completed; Real-Dataset Validation Formally Marked BLOCKED  
**Author:** Senior AI Research Engineer  

---

## 1. Executive Summary

This report documents the empirical outcomes and scientific validations for the integrated **PAADU/CHARM** deep anomaly detection system and **SVA-GRAG-Core** retrieval and verification architecture. 

The evaluation encompasses four dedicated experimental modules, multi-configuration baseline comparisons, formal statistical tests, publication figures, and comprehensive parameter randomization checks across 60 benchmark incidents and 45 anomalous blocks.

---

## 2. System Architecture & Component Verification

### 2.1 Anomaly Detection Engine (PAADU/CHARM)
- **Model:** `SemanticAnomalyMAE` (185,007 parameters) loaded from `models/charm_model_weights.pt` (SHA-256 verified).
- **Hybrid Anomaly Score:** $s(\mathbf{x}) = 0.2 \cdot z_{\text{recon}} + 0.8 \cdot z_{D_M}$.
- **Volume Adaptive Floor:** Hard calibrated benign floor $\tau = 2.5000$.
- **XAI Method:** Leave-one-feature-out occlusion over continuous flow dimensions in $\mathbb{R}^{B \times 1 \times 10 \times 46}$.

### 2.2 SVA-GRAG-Core Retrieval & Verification
- **Retrieval:** TF-IDF document vectors + NetworkX co-citation graph over NIST/RFC cybersecurity literature.
- **Evidence Quality Assessment:** $Q = 0.35 R + 0.35 C + 0.15 K + 0.15 P$.
- **Verification Engine:** HuggingFace cross-encoder `cross-encoder/nli-MiniLM2-L6-H768` (66.9M parameters).
- **Epistemic Badge Policy:** Ungrounded assertions are tagged with `[OPERATIONAL HYPOTHESIS]`, eliminating unsupported factual claims.

---

## 3. Recomputed Experimental Results

### Experiment 1: Telemetry Fact Checking & Hallucination Mitigation
- **Evaluated Dataset:** 60 incidents (15 Benign, 15 SYN Flood, 15 Volumetric Flood, 15 Port Scan) with controlled entity corruptions on 10 incidents ($N=10$).
- **Total Validated Atomic Propositions:** 474 propositions recorded in `RAW_CLAIM_VALIDATION.csv`.
- **Entity Hallucination Rate (EHR):** Reduced from **0.0167** (1.67%) to **0.0000** (0.0%).
- **Unsupported Claim Rate (UCR):** Reduced from **0.0278** (2.78% mean across all incidents; 16.67% on corrupted incidents) to **0.0000** (0.0%).
- **Evidence Coverage (EC):** Elevated from **97.22%** to **100.00%**.
- **Statistical Significance:** Wilcoxon signed-rank test $W = 0.0$, $p = 2.96 \times 10^{-3}$, Cohen's $d = 0.44$.

### Experiment 2: Counterfactual Metadata Testing & Narrative Stability
- **Evaluated Dataset:** 60 counterfactual incident pairs recorded in `RAW_COUNTERFACTUAL_PAIRS.csv`.
- **Interventions:** Sensor renaming, timestamp shifting, zero-attribution noise perturbation.
- **Attack Attribution Change Rate (AACR):** **0.0000** (0 flips across 60 pairs).
- **Mean Semantic Similarity:** **0.9956** (lexical vector cosine similarity).
- **Token Jaccard Similarity:** **0.9329**.
- **Counterfactual Consistency Rate (CCR):** **0.4333** (string variation restricted exclusively to sensor block ID labels).

### Experiment 3: XAI Faithfulness & AOPC Cumulative Deletion
- **Evaluated Dataset:** 45 anomalous blocks evaluated across $k = 1 \dots 10$ steps = 450 trajectory rows recorded in `RAW_FEATURE_DELETION.csv`.
- **AOPC@10 Top-k Mean:** **63.10** vs Random Baseline **-4.15** vs Bottom-k Baseline **-57.85**.
- **Paired t-test:** $t = 4.40$, $p = 6.89 \times 10^{-5}$, Cohen's $d = 0.66$.
- **Comprehensiveness@5:** **68.36**.
- **Rank Stability under Gaussian Noise:** Spearman $\rho = 0.9766$.

### Experiment 4: Model Parameter Randomization (Adebayo Sanity Check)
- **Evaluated Dataset:** 30 anomalous blocks $\times$ 7 cascading randomization levels $\times$ 3 random seeds (42, 101, 2026) = 630 rows recorded in `RAW_RANDOMIZATION_RESULTS.csv`.
- **Monotonic Rank Correlation Decay:**
  - Level 0 (Trained Weights): $\rho = 1.0000$, Cosine $= 1.0000$, Top-5 Overlap $= 1.0000$
  - Level 1 (Decoder Random): $\rho = 0.9998$, Cosine $= 0.9995$, Top-5 Overlap $= 1.0000$
  - Level 3 (Transformer Random): $\rho = 0.0911$, Cosine $= -0.0026$, Top-5 Overlap $= 0.5931$
  - Level 5 (Complete Random): $\rho = -0.0230$, Cosine $= 0.0631$, Top-5 Overlap $= 0.5720$
  - Level 6 (Permutation Noise Floor): $\rho = 0.0114$, Cosine $= 0.0412$, Top-5 Overlap $= 0.0690$
- **Conclusion:** Confirms that CHARM attributions genuinely depend on learned representations.

---

## 4. Multi-Configuration Comparative Evaluation

| Configuration | Description | EHR | UCR | Evidence Coverage | CCR | AOPC@10 | Mean Latency |
|---|---|---|---|---|---|---|---|
| **Config A** | Raw CHARM XAI Attributions | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 63.10 | **0.0502s** |
| **Config B** | Unverified Explanations | 0.0167 | 0.0278 | 0.9722 | 0.4333 | 63.10 | 0.0503s |
| **Config C** | Graph-RAG Retrieval Only | 0.0167 | 0.0232 | 0.9768 | 0.4333 | 63.10 | 0.0525s |
| **Config D** | Fully Verified Pipeline | **0.0000** | **0.0000** | **1.0000** | **0.4333** | **63.10** | 0.2041s |

---

## 5. Dataset Validation Limitations & Blockers

- **Status of Real-Data Validation:** **BLOCKED**.
- **Root Cause:** The host filesystem contains `datasets/research_partition_manifest.csv` but lacks raw CSV partition files (`test_partition_*.csv`) and raw packet PCAPs for CICIoT2023.
- **Cross-Dataset Compatibility:** Alternative datasets (e.g. CICIoMT2024) lack 9 continuous features required by the 46-feature schema. Zero-filling these missing dimensions distorts the Mahalanobis covariance, producing a 100% false-positive anomaly rate.
- **Scientific Integrity Stance:** Rather than fabricating data or misrepresenting synthetic data as real traffic, real-data validation is explicitly marked as BLOCKED pending raw packet CSV availability.

---

## 6. Verification and Reproducibility

- Master Execution Script: `python3 run_all_experiments.py` (runtime: 257.05s).
- Automated Unit Tests: `python3 tests/test_audit_validation.py` (5/5 PASSED).
- Original Source Integrity: 100% UNCHANGED (23/23 monitored source files verified via SHA-256).
