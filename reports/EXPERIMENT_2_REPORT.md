# Experiment 2 Report: Counterfactual Metadata Testing & Explanation Stability

**Evaluation Date:** 2026-11-17  
**Evaluated Sample Pairs:** 60 counterfactual pairs across 3 non-causal intervention types  
**Stable Pairs Analyzed:** 60 (0 excluded due to decision boundary flips)  
**Status:** Completed  

## 1. Research Question & Objective
> **Does changing irrelevant network metadata cause the explanation to invent or change its threat narrative?**

This experiment evaluates the invariance of incident explanations under controlled, non-causal metadata shifts (sensor IDs, block labels, timestamps, and zero-attribution feature perturbations).

## 2. Quantitative Metric Definitions
- **Narrative Variance Score (NVS)**:
  $$\text{NVS}(E, E') = 1 - \frac{|Claims(E) \cap Claims(E')|}{|Claims(E) \cup Claims(E')|}$$
- **Attack Attribution Change Rate (AACR)**: Fraction of pairs where the inferred attack classification flips despite stable model prediction.
- **Counterfactual Consistency Rate (CCR)**: Fraction of pairs where explanation claims remain 100% consistent (NVS == 0.0).
- **Unsupported Claim Introduction Rate (UCIR)**: Fraction of counterfactuals that introduced an unsupported claim not in the original explanation.

## 3. Results Summary Table

| Metric | Evaluated Value | Scientific Interpretation |
| :--- | :---: | :--- |
| **Attack Attribution Change Rate (AACR)** | **0.0000** | **Zero Spurious Attribution Flips**: Attack family remained 100% stable across all pairs. |
| **Exact Narrative Match Rate** | 0.3333 | Fraction of pairs with character-identical full narratives. |
| **Mean Token Jaccard Similarity** | 0.9834 | Lexical overlap between original and counterfactual texts. |
| **Mean Semantic Similarity** | 0.9915 | Vector space cosine similarity of narrative terms. |
| **Counterfactual Consistency Rate (CCR)** | 0.3333 | Fraction of pairs with 100% standardized claim identity. |
| **Narrative Variance Score (NVS)** | 0.2329 (±0.1996) | Mean claim set Jaccard distance between pairs. |

## 4. Scientific Findings & Audit Disclosures
1. **Preserved Attack Invariance**: Because CHARM's occlusion engine operates on continuous flow features and the query builder synthesizes neutral queries, non-causal metadata interventions caused **zero threat attribution flips (AACR = 0.0000)** across all 60 evaluated pairs.
2. **Deterministic Architecture Clarification**: The high semantic similarity (0.9915) and zero attribution flips are primarily structural properties of deterministic proposition generation rather than proof of generalized LLM hallucination resistance. When evaluated under genuine token-sampling LLMs, metadata shifts can introduce non-deterministic variance.
3. **Audited Trajectories**: Complete paired records (including exact scores, decisions, token Jaccards, semantic similarities, and claim differences) are preserved in `RAW_COUNTERFACTUAL_PAIRS.csv` for independent audit.
