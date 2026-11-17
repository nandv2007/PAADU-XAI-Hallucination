# Incident Investigation Report: INC-CHARM-7DC7953A
**Input Reference:** `sample_syn_flood_burst` | **Timestamp:** `2026-11-10T07:43:39.910631+00:00`  
**Model:** PAADU/CHARM + SVA-GRAG-Core Integrated Pipeline  

---

## 1. PAADU/CHARM Anomaly Detection & XAI
- **Anomaly Decision:** 🚨 **MALICIOUS / ANOMALY DETECTED**
- **Hybrid Anomaly Score:** `61.2298` (Calibrated Decision Threshold: `2.5000`)
- **Reconstruction MSE:** `7.3786` | **Latent Mahalanobis Distance:** `229.7543`
- **Active Trigger Policies:** `hybrid_threshold, latent_evidence, burst+drift`
- **Top Contributing Features (Occlusion):** syn_count (-108.16), syn_flag_number (+24.15), TCP (-2.22), cwr_flag_number (+0.12)

---

## 2. Synthesized Cybersecurity Investigation Query
- **Generated Query:** *"What network behaviors and attack patterns are associated with unusual deviations in syn_count, syn_flag_number, TCP, cwr_flag_number consistent with possible elevated TCP SYN connection requests and disproportionate SYN flag ratio, and what evidence helps distinguish such anomalies from rapid client reconnect storms or high-frequency microservice initiation?"*
- **Query Complexity:** `COMPLEX` (Score: `0.82`)
- **Retrieval Strategy:** `Adaptive Hybrid (Category=COMPLEX, k=6, Boost=0.3, Domains=2)`

---

## 3. Evidence Retrieval & Quality Assessment
- **Retrieved Evidence Documents:** 6 documents
- **Quality Score:** `0.440` (`INSUFFICIENT`)
  - Relevance: `0.23` | Coverage: `0.17` | Provenance: `1.00`
- **Refinement Termination:** Bounded limit reached (2 iterations). Proceeding with best-effort evidence.

---

## 4. Evidence-Grounded Verified Incident Narrative
[VERIFIED TELEMETRY] PAADU/CHARM evaluated sample_syn_flood_burst as ANOMALOUS with a hybrid anomaly score of 61.230 against the calibrated baseline threshold of 2.500.

[VERIFIED TELEMETRY] The alert was triggered by policies: [hybrid_threshold, latent_evidence, burst+drift], with a reconstruction error of 7.3786 and a latent Mahalanobis distance of 229.7543.

[VERIFIED TELEMETRY] Explainable AI leave-one-feature-out occlusion identified the most influential features as: syn_count (contribution: -108.16, observed mean: 17.75 vs benign mean: 0.80); syn_flag_number (contribution: +24.15, observed mean: 0.04 vs benign mean: 0.00); TCP (contribution: -2.22, observed mean: 2.27 vs benign mean: 0.86).

[VERIFIED EVIDENCE] A TCP SYN flood is a denial-of-service attack exploiting the TCP three-way handshake.

[OPERATIONAL HYPOTHESIS] The observed telemetry deviation cannot be conclusively attributed to malicious adversarial action solely from statistical flow metrics without inspecting packet payload headers or host process logs.

[OPERATIONAL HYPOTHESIS] Further investigation is recommended to confirm whether the detected telemetry shift reflects a true attack or a benign non-malicious burst, such as scheduled sensor synchronization or client reconnection.

---

## 5. Authoritative Document Citations
- **[TCP SYN Flood and Handshake Exhaustion]**: CERT Advisory CA-1996-21 / RFC 793: Transmission Control Protocol Specification (1981)

---

## 6. Execution Trace & Runtimes
| Stage | Execution Time |
|---|---|
| CHARM Telemetry Inference | 0.0300s |
| XAI Feature Attribution | 0.1424s |
| Incident Adaptation & Query Generation | 0.0001s |
| Adaptive GraphRAG Retrieval & Refinement | 0.0023s |
| Evidence-Grounded Explanation & NLI Verification | 0.1042s |
| **Total Pipeline Latency** | **0.2790s** |
