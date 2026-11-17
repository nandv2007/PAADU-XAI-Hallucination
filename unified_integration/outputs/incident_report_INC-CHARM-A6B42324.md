# Incident Investigation Report: INC-CHARM-A6B42324
**Input Reference:** `sample_volumetric_rate_spike` | **Timestamp:** `2026-11-10T10:19:08.275333+00:00`  
**Model:** PAADU/CHARM + SVA-GRAG-Core Integrated Pipeline  

---

## 1. PAADU/CHARM Anomaly Detection & XAI
- **Anomaly Decision:** 🚨 **MALICIOUS / ANOMALY DETECTED**
- **Hybrid Anomaly Score:** `77.0994` (Calibrated Decision Threshold: `2.5000`)
- **Reconstruction MSE:** `8.2712` | **Latent Mahalanobis Distance:** `287.6181`
- **Active Trigger Policies:** `hybrid_threshold, latent_evidence, burst+drift`
- **Top Contributing Features (Occlusion):** Tot sum (-109.66), Rate (+32.06), fin_flag_number (+0.16), Std (+0.15)

---

## 2. Synthesized Cybersecurity Investigation Query
- **Generated Query:** *"What network behaviors and attack patterns are associated with unusual deviations in Tot sum, Rate, fin_flag_number, Std consistent with possible aggregate byte volume surge and transmission packet rate flood, and what evidence helps distinguish such anomalies from legitimate network burst traffic or large bulk file transfer?"*
- **Query Complexity:** `MODERATE` (Score: `0.66`)
- **Retrieval Strategy:** `Adaptive Hybrid (Category=MODERATE, k=4, Boost=0.22, Domains=1)`

---

## 3. Evidence Retrieval & Quality Assessment
- **Retrieved Evidence Documents:** 6 documents
- **Quality Score:** `0.371` (`INSUFFICIENT`)
  - Relevance: `0.20` | Coverage: `0.00` | Provenance: `1.00`
- **Refinement Termination:** Bounded limit reached (2 iterations). Proceeding with best-effort evidence.

---

## 4. Evidence-Grounded Verified Incident Narrative
[VERIFIED TELEMETRY] PAADU/CHARM evaluated sample_volumetric_rate_spike as ANOMALOUS with a hybrid anomaly score of 77.099 against the calibrated baseline threshold of 2.500.

[VERIFIED TELEMETRY] The alert was triggered by policies: [hybrid_threshold, latent_evidence, burst+drift], with a reconstruction error of 8.2712 and a latent Mahalanobis distance of 287.6181.

[VERIFIED TELEMETRY] Explainable AI leave-one-feature-out occlusion identified the most influential features as: Tot sum (contribution: -109.66, observed mean: 144944.66 vs benign mean: 6502.30); Rate (contribution: +32.06, observed mean: 308116.50 vs benign mean: 1929.59); fin_flag_number (contribution: +0.16, observed mean: 0.00 vs benign mean: 0.00).

[VERIFIED EVIDENCE] In a UDP flood attack, adversaries send massive volumes of high-rate datagrams to random ports on the target.

[OPERATIONAL HYPOTHESIS] The observed telemetry deviation cannot be conclusively attributed to malicious adversarial action solely from statistical flow metrics without inspecting packet payload headers or host process logs.

[OPERATIONAL HYPOTHESIS] Further investigation is recommended to confirm whether the detected telemetry shift reflects a true attack or a benign non-malicious burst, such as scheduled sensor synchronization or client reconnection.

---

## 5. Authoritative Document Citations
- **[UDP Flood and Reflection Amplification Attacks]**: CISA Alert TA14-017A: UDP-Based Amplification Attacks / RFC 768 (1980)

---

## 6. Execution Trace & Runtimes
| Stage | Execution Time |
|---|---|
| CHARM Telemetry Inference | 0.0034s |
| XAI Feature Attribution | 0.1449s |
| Incident Adaptation & Query Generation | 0.0000s |
| Adaptive GraphRAG Retrieval & Refinement | 0.0021s |
| Evidence-Grounded Explanation & NLI Verification | 0.1291s |
| **Total Pipeline Latency** | **0.2795s** |
