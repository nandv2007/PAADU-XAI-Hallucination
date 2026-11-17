# Incident Investigation Report: INC-CHARM-3E083879
**Input Reference:** `test_e2e_block_0` | **Timestamp:** `2026-11-10T07:42:52.647145+00:00`  
**Model:** PAADU/CHARM + SVA-GRAG-Core Integrated Pipeline  

---

## 1. PAADU/CHARM Anomaly Detection & XAI
- **Anomaly Decision:** 🚨 **MALICIOUS / ANOMALY DETECTED**
- **Hybrid Anomaly Score:** `35.2284` (Calibrated Decision Threshold: `2.5000`)
- **Reconstruction MSE:** `4.7799` | **Latent Mahalanobis Distance:** `135.1863`
- **Active Trigger Policies:** `hybrid_threshold, latent_evidence, burst`
- **Top Contributing Features (Occlusion):** TCP (-3.84), syn_count (-3.56), Rate (+0.14), fin_flag_number (-0.12)

---

## 2. Synthesized Cybersecurity Investigation Query
- **Generated Query:** *"What network behaviors and attack patterns are associated with unusual deviations in TCP, syn_count, Rate, fin_flag_number consistent with possible telemetry deviation in TCP and elevated TCP SYN connection requests, and what evidence helps distinguish such anomalies from rapid client reconnect storms or legitimate network burst traffic?"*
- **Query Complexity:** `COMPLEX` (Score: `0.98`)
- **Retrieval Strategy:** `Adaptive Hybrid (Category=COMPLEX, k=6, Boost=0.3, Domains=3)`

---

## 3. Evidence Retrieval & Quality Assessment
- **Retrieved Evidence Documents:** 6 documents
- **Quality Score:** `0.435` (`INSUFFICIENT`)
  - Relevance: `0.22` | Coverage: `0.17` | Provenance: `1.00`
- **Refinement Termination:** Bounded limit reached (2 iterations). Proceeding with best-effort evidence.

---

## 4. Evidence-Grounded Verified Incident Narrative
[VERIFIED TELEMETRY] PAADU/CHARM evaluated test_e2e_block_0 as ANOMALOUS with a hybrid anomaly score of 35.228 against the calibrated baseline threshold of 2.500.

[VERIFIED TELEMETRY] The alert was triggered by policies: [hybrid_threshold, latent_evidence, burst], with a reconstruction error of 4.7799 and a latent Mahalanobis distance of 135.1863.

[VERIFIED TELEMETRY] Explainable AI leave-one-feature-out occlusion identified the most influential features as: TCP (contribution: -3.84, observed mean: 2.26 vs benign mean: 0.86); syn_count (contribution: -3.56, observed mean: 16.18 vs benign mean: 0.80); Rate (contribution: +0.14, observed mean: 491.83 vs benign mean: 1929.59).

[VERIFIED EVIDENCE] A TCP SYN flood is a denial-of-service attack exploiting the TCP three-way handshake.

[VERIFIED EVIDENCE] In a UDP flood attack, adversaries send massive volumes of high-rate datagrams to random ports on the target.

[VERIFIED EVIDENCE] In a UDP flood attack, adversaries send massive volumes of high-rate datagrams to random ports on the target.

[VERIFIED EVIDENCE] In a UDP flood attack, adversaries send massive volumes of high-rate datagrams to random ports on the target.

[OPERATIONAL HYPOTHESIS] The observed telemetry deviation cannot be conclusively attributed to malicious adversarial action solely from statistical flow metrics without inspecting packet payload headers or host process logs.

[OPERATIONAL HYPOTHESIS] Further investigation is recommended to confirm whether the detected telemetry shift reflects a true attack or a benign non-malicious burst, such as scheduled sensor synchronization or client reconnection.

---

## 5. Authoritative Document Citations
- **[TCP SYN Flood and Handshake Exhaustion]**: CERT Advisory CA-1996-21 / RFC 793: Transmission Control Protocol Specification (1981)
- **[Normal IoT Baseline Telemetry and Rate Patterns]**: NIST SP 800-213: IoT Device Cybersecurity Guidance for the Federal Government (2021)
- **[MQTT Protocol Security and IoT Publish/Connect Floods]**: OASIS Standard: MQTT Version 5.0 Specification (2019)
- **[UDP Flood and Reflection Amplification Attacks]**: CISA Alert TA14-017A: UDP-Based Amplification Attacks / RFC 768 (1980)

---

## 6. Execution Trace & Runtimes
| Stage | Execution Time |
|---|---|
| CHARM Telemetry Inference | 0.0034s |
| XAI Feature Attribution | 0.0483s |
| Incident Adaptation & Query Generation | 0.0001s |
| Adaptive GraphRAG Retrieval & Refinement | 0.0034s |
| Evidence-Grounded Explanation & NLI Verification | 0.4541s |
| **Total Pipeline Latency** | **0.5093s** |
