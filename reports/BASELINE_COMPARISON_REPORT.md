# Scientific Baseline Comparison Report: Configurations A, B, C, and D

**Evaluation Date:** 2026-11-17  
**Evaluated Sample Size:** 60 paired incidents across 4 distinct traffic profiles  
**Evaluation Mode:** Paired evaluation under identical deterministic conditions (Dynamically Computed)  
**Status:** Completed  

## 1. Evaluated Configurations
- **Configuration A: CHARM + Original XAI**: Raw mathematical telemetry anomaly scores, threshold triggers, and deterministic occlusion feature attributions.
- **Configuration B: CHARM + XAI + Unverified Explanation**: Factual claim generation without external knowledge retrieval or hallucination verification.
- **Configuration C: CHARM + XAI + SVA-GRAG Retrieval**: Retrieval-augmented explanation referencing authoritative RFC/NIST literature, but without NLI verification.
- **Configuration D: CHARM + XAI + SVA-GRAG + NLI Verification & Correction**: Complete pipeline featuring transformer cross-encoder claim verification, contradiction retraction, and conservative uncertainty badging.

## 2. Comparative Evaluation Matrix (Dynamically Measured)

| Metric | Config A (Raw XAI) | Config B (Unverified) | Config C (Retrieval Only) | Config D (Fully Verified) |
| :--- | :---: | :---: | :---: | :---: |
| **Entity Hallucination Rate (EHR)** | **0.0000** (N/A) | 0.0303 | 0.0518 | **0.0000** |
| **Unsupported Claim Rate (UCR)** | **0.0000** (N/A) | 0.0278 | 0.0641 | **0.0000** |
| **Evidence Coverage (EC)** | 1.0000 (telemetry only) | 0.5833 | 0.6212 | **0.5111** |
| **Counterfactual Consistency** | 1.0000 | 0.3333 | 0.3333 | **0.3333** |
| **XAI Faithfulness (AOPC@10)** | 92.48 | 92.48 | 92.48 | **92.48** |
| **Mean Latency (sec/incident)** | **0.0491s** | 0.0492s | 0.0514s | 0.2022s |

## 3. Important Scientific Clarification: XAI Faithfulness vs NLI Verification
A critical conceptual finding of this research validation is that **NLI verification operates strictly on textual claim validity and knowledge grounding; it does not alter the underlying mathematical faithfulness of CHARM's XAI attributions**.
- The feature attribution scores are computed directly by the leave-one-feature-out occlusion over the Masked Autoencoder ($\text{AOPC@10} = 92.48$).
- SVA-GRAG and NLI verification ensure that the *narrative claims presented to human analysts* accurately reflect those mathematical outputs without hallucinating non-existent features or making ungrounded causal claims.
