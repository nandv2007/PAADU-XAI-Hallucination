# Experiment 1 Report: Ground-Truth Telemetry Fact Checking & Hallucination Audit

**Evaluation Date:** 2026-11-17  
**Evaluated Incidents:** 60 paired evaluations across 4 traffic profiles  
**Total Raw Claims Audited:** 768 atomic propositions  
**Status:** Completed (Dynamically Computed from Raw Observations)  

## 1. Research Question & Objective
> **Does the incident explanation contain technical claims that are unsupported by the original network telemetry?**

This experiment quantitatively assesses the hallucination rate of generated incident narratives against ground-truth network telemetry records and authoritative cited evidence. We evaluate both standard generation and controlled entity perturbation injections.

## 2. Technical Clarification: Generator Architecture
The Incident Explanation Generator in `unified_integration/generation/incident_explainer.py` employs a deterministic, rule-based proposition segmentation architecture that transforms numerical telemetry and retrieved evidence into typed `FactualClaim` propositions. **It does not invoke an external autoregressive Large Language Model (such as GPT-4 or LLaMA)**. In unverified generation, operational hypotheses and injected entity distortions lack verification checks. Under Configuration D, fine-grained cross-encoder NLI (`cross-encoder/nli-MiniLM2-L6-H768`) and structured telemetry schema checks verify each proposition.

## 3. Quantitative Results Matrix

### Macro Incident-Level Metrics (N = 60 incidents)
| Metric | Configuration B (Unverified Explanations) | Configuration D (SVA-GRAG + NLI Verified) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Entity Hallucination Rate (EHR)** | 0.0319 (±0.0448) | **0.0000** (±0.0000) | **-100.0%** |
| **Unsupported Claim Rate (UCR)** | 0.0641 (±0.0848) | **0.0000** (±0.0000) | **-100.0%** |
| **Contradiction Rate (CR)** | 0.0232 | **0.0000** | **-100.0%** |
| **Evidence Coverage (EC)** | 0.6212 | **0.6212** | **+0.0%** |

### Pooled Metrics Across All Assertions
| Dimension | Unverified Denominator | Unverified Numerator | Unverified Pooled Rate | Verified Denominator | Verified Numerator | Verified Pooled Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EHR (Entity Errors / Total Entities)** | 1054 | 35 | **0.0332** | 1044 | 0 | **0.0000** |
| **UCR (Unsupported / Total Claims)** | 384 | 25 | **0.0651** | 384 | 0 | **0.0000** |

### Displayed Assertion Breakdown (Configuration D)
- **Total Displayed Claims**: 384
- **Supported Factual Assertions**: 239
- **Retracted Contradictions**: 10
- **Flagged Uncertainty Assertions**: 135
- **Contradicted Assertions**: 0
- **Unsupported Assertions**: 0
- **Unverifiable Assertions**: 0
- **Displayed with Verified Badge**: 239
