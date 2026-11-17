# Incident Investigation Report: INC-CHARM-7D4BB03C
**Input Reference:** `sample_benign_iot_flow` | **Timestamp:** `2026-11-10T10:19:07.604756+00:00`  
**Model:** PAADU/CHARM + SVA-GRAG-Core Integrated Pipeline  

---

## 1. PAADU/CHARM Anomaly Detection & XAI
- **Anomaly Decision:** 🚨 **MALICIOUS / ANOMALY DETECTED**
- **Hybrid Anomaly Score:** `10.5905` (Calibrated Decision Threshold: `2.5000`)
- **Reconstruction MSE:** `0.0198` | **Latent Mahalanobis Distance:** `46.0592`
- **Active Trigger Policies:** `hybrid_threshold, latent_evidence, burst`
- **Top Contributing Features (Occlusion):** fin_flag_number (+0.34), rst_flag_number (+0.33), psh_flag_number (+0.22), ARP (-0.20)

---

## 2. Synthesized Cybersecurity Investigation Query
- **Generated Query:** *"What network behaviors and attack patterns are associated with unusual deviations in fin_flag_number, rst_flag_number, psh_flag_number, ARP consistent with possible telemetry deviation in fin_flag_number and telemetry deviation in rst_flag_number, and what evidence helps distinguish such anomalies from DHCP address reassignment or subnet scan?"*
- **Query Complexity:** `MODERATE` (Score: `0.66`)
- **Retrieval Strategy:** `Adaptive Hybrid (Category=MODERATE, k=4, Boost=0.22, Domains=1)`

---

## 3. Evidence Retrieval & Quality Assessment
- **Retrieved Evidence Documents:** 6 documents
- **Quality Score:** `0.514` (`PARTIAL`)
  - Relevance: `0.21` | Coverage: `0.40` | Provenance: `1.00`
- **Refinement Termination:** Bounded limit reached (2 iterations). Proceeding with best-effort evidence.

---

## 4. Evidence-Grounded Verified Incident Narrative
[VERIFIED TELEMETRY] PAADU/CHARM evaluated sample_benign_iot_flow as ANOMALOUS with a hybrid anomaly score of 10.591 against the calibrated baseline threshold of 2.500.

[VERIFIED TELEMETRY] The alert was triggered by policies: [hybrid_threshold, latent_evidence, burst], with a reconstruction error of 0.0198 and a latent Mahalanobis distance of 46.0592.

[VERIFIED TELEMETRY] Explainable AI leave-one-feature-out occlusion identified the most influential features as: fin_flag_number (contribution: +0.34, observed mean: 0.00 vs benign mean: 0.00); rst_flag_number (contribution: +0.33, observed mean: 0.00 vs benign mean: 0.00); psh_flag_number (contribution: +0.22, observed mean: 0.03 vs benign mean: 0.02).

[UNVERIFIED - INSUFFICIENT EVIDENCE] Retrieved cybersecurity reference ARP Spoofing and Layer 2 Cache Poisoning (IETF RFC 826: An Ethernet Address Resolution Protocol (1982) / NIST SP 800-115) indicates that unsupervised statistical telemetry anomalies reflect operational shifts that require external corroboration.

[OPERATIONAL HYPOTHESIS] The observed telemetry deviation cannot be conclusively attributed to malicious adversarial action solely from statistical flow metrics without inspecting packet payload headers or host process logs.

[OPERATIONAL HYPOTHESIS] Further investigation is recommended to confirm whether the detected telemetry shift reflects a true attack or a benign non-malicious burst, such as scheduled sensor synchronization or client reconnection.

---

## 5. Authoritative Document Citations
- **[ARP Spoofing and Layer 2 Cache Poisoning]**: IETF RFC 826: An Ethernet Address Resolution Protocol (1982) / NIST SP 800-115

---

## 6. Execution Trace & Runtimes
| Stage | Execution Time |
|---|---|
| CHARM Telemetry Inference | 0.0034s |
| XAI Feature Attribution | 0.1449s |
| Incident Adaptation & Query Generation | 0.0001s |
| Adaptive GraphRAG Retrieval & Refinement | 0.0032s |
| Evidence-Grounded Explanation & NLI Verification | 0.5582s |
| **Total Pipeline Latency** | **0.7098s** |
