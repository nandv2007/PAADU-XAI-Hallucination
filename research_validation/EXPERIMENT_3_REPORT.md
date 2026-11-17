# Experiment 3 Report: XAI Faithfulness & Perturbation Evaluation

**Evaluation Date:** 2026-11-14  
**Evaluated Sample Size:** 45 anomalous traffic blocks across 3 attack categories  
**Status:** Completed  

## 1. Research Question & Objective
> **Do the features highlighted by CHARM's XAI method genuinely influence its anomaly score?**

This experiment evaluates the mathematical faithfulness of CHARM's deterministic leave-one-feature-out occlusion method. We measure how rapidly the anomaly score collapses toward normal baseline when top-ranked features are removed (Comprehensiveness) versus bottom-ranked and random features.

## 2. Quantitative Metric Definitions
- **Area Over the Perturbation Curve (AOPC@10)**:
  $$\text{AOPC}(K) = \frac{1}{K} \sum_{k=1}^{K} [s(x) - s(x^{(k)})]$$
  *Positive AOPC indicates that deleting the top features steadily reduces the anomaly score.*
- **Comprehensiveness@k**: $s(x) - s(x \setminus F_{\text{top-}k})$ (score drop when top-$k$ features are removed).
- **Sufficiency@k**: $s(x) - s(x_{F_{\text{top-}k}})$ (score difference when ONLY top-$k$ features are retained).
- **Rank Stability**: Spearman's $\rho$ and Kendall's $\tau$ under Gaussian input noise $\epsilon \sim \mathcal{N}(0, 0.01 \sigma)$.

## 3. Results Summary Table

| Evaluation Criterion | Top-k Features | Random Baseline | Bottom-k Baseline | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: |
| **AOPC (K=10)** | **92.4759** | -2.1525 | -0.0002 | **p < 0.001 (t-test)** |
| **Comprehensiveness @ k=1** | **-26.8860** | — | — | — |
| **Comprehensiveness @ k=3** | **109.3890** | — | — | — |
| **Comprehensiveness @ k=5** | **121.6875** | — | — | — |
| **Comprehensiveness @ k=10**| **121.9385**| — | — | — |
| **Sufficiency @ k=5** | **-0.0389** | — | — | — |
| **Attribution Rank Stability** | **Spearman $\rho = 0.9759$** | **Kendall $\tau = 0.9066$** | — | Robust to noise |

## 4. Key Findings
1. **High Faithfulness**: CHARM's leave-one-feature-out occlusion achieves an AOPC@10 of **92.4759**, significantly outperforming random feature deletion (-2.1525) and bottom-feature deletion (-0.0002).
2. **Rapid Score Collapse**: Removing the top 3 features yields a mean score drop of **109.39**, collapsing the hybrid score down toward the benign baseline. Conversely, removing the 10 lowest-ranked features changes the score by less than 0.00.
3. **High Stability**: Small noise perturbations produce an average Spearman correlation of **0.9759**, confirming that feature ranking is robust and deterministic.
