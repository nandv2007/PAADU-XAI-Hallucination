# DATASET VALIDATION & COMPATIBILITY AUDIT REPORT

**Project:** `PAADU-XAI-Hallucination-Research`  
**Audit Date:** 2026-11-17  
**Evaluation Role:** Senior AI Research Engineer in Network Intrusion Detection & Explainable AI  
**Overall Status:** **REAL-DATA SCIENTIFIC VALIDATION IS FORMALLY BLOCKED**  

---

## 1. Executive Summary & Audit Verdict

An exhaustive, filesystem-wide audit was conducted to locate compatible network traffic evaluation records for the trained PAADU/CHARM intrusion detection model (`SemanticAnomalyMAE`).

### Critical Audit Findings:
1. **Raw CICIoT2023 Evaluation Records**: **ABSENT FROM DISK**. While a metadata partition manifest (`research_partition_manifest.csv`) exists describing 19,942 training blocks, the underlying raw PCAP files or preprocessed CSV flow records for CICIoT2023 are **not present** anywhere in the workspace or parent directory.
2. **Local Datasets Found**:
   - `Dataset/WiFI_and_MQTT/` (CICIoMT2024): Contains 46 columns, but **9 mandatory flow features are missing**. Zero-filling or imputing missing features creates massive Mahalanobis latent distortions ($D_M \gg 1000$), resulting in a catastrophic 100% false positive rate.
   - `ciciov24/` & `cic25/` (CAN-Bus Automotive Datasets): Represent vehicle Controller Area Network telemetry (Arbitration IDs, DLC, 8-byte hexadecimal payloads). These exhibit **0% feature overlap** with IP flow features.
3. **Synthetic Benchmark Tensors**: To evaluate software stability, pipeline boundaries, XAI faithfulness, and NLI verification, the research suite generated calibrated synthetic tensors ($\mathbf{X} \in \mathbb{R}^{60 \times 1 \times 10 \times 46}$) matching the exact mean and standard deviation $(\mu, \sigma)$ of the trained benign baseline.

### Formal Scientific Verdict:
> **Real-data scientific validation on actual CICIoT2023 network flow traffic is BLOCKED due to the absence of the raw CSV partitions.**  
> The experimental results reported in this project reflect **controlled synthetic benchmark tensors** and **must not be misrepresented as real-world network traffic evaluation**. Full real-world intrusion detection performance remains pending the provision of the raw CICIoT2023 CSV partitions.

---

## 2. Model Input Specifications & Mathematical Requirements

The trained CHARM model (`SemanticAnomalyMAE`, 185,007 parameters) requires an exact 4D tensor input:
$$\mathbf{X} \in \mathbb{R}^{B \times 1 \times 10 \times 46}$$
where:
- $B$: Batch size (number of evaluation flow windows).
- $1$: Spatial sensor node dimension.
- $10$: Temporal sequence length (10 consecutive flow time steps per observation block).
- $46$: Exact continuous network flow statistics conforming strictly to the **CICIoT2023** schema.

### Mandatory 46 CICIoT2023 Features:
| Index | Feature Name | Description / Protocol Level |
| :---: | :--- | :--- |
| 1–4 | `flow_duration`, `Header_Length`, `Protocol Type`, `Duration` | Basic packet/flow header metrics |
| 5–7 | `Rate`, `Srate`, `Drate` | Flow transfer rates (packets/sec) |
| 8–14 | `fin_flag_number` to `cwr_flag_number` | TCP control flag indicator counts |
| 15–19 | `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count` | Aggregated TCP flag statistics |
| 20–26 | `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC` | L7 Application protocol flags |
| 27–33 | `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IPv`, `LLC` | L3/L4 Transport and network protocol flags |
| 34–39 | `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size` | Packet size distribution statistics |
| 40–46 | `IAT`, `Number`, `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight` | Inter-arrival time and flow dispersion metrics |

---

## 3. Exhaustive Inspection of Local Filesystem Datasets

| Directory / Dataset Path | Dataset Identity | Column Count | Matching Features | Compatibility Status | Technical Blocker & Failure Mode |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `datasets/Dataset/WiFI_and_MQTT/` | CICIoMT2024 | 46 | 37 / 46 | **INCOMPATIBLE** | Missing 9 core features (`flow_duration`, `Header_Length`, `Rate`, `Srate`, `Min`, `Max`, `AVG`, `Std`, `Tot size`). Imputation severely corrupts the latent Mahalanobis distance ($D_M \gg 1000$). |
| `datasets/ciciov24/` | CICIoV2024 | 18 | 0 / 46 | **INCOMPATIBLE** | CAN-bus automotive traffic (CAN ID, DLC, Data_0..Data_7). 0% overlap with network flow headers. |
| `datasets/cic25/` | CIC2025 | 18 | 0 / 46 | **INCOMPATIBLE** | CAN-bus automotive traffic. 0% overlap with network flow headers. |
| `datasets/cic_iot_2023/` | Partition Manifest | Manifest Only | 46 / 46 (Schema) | **BLOCKED** | Only `research_partition_manifest.csv` exists; raw partition CSV files were not included in the repository. |

---

## 4. Why Zero-Filling or Feature Imputation is Scientifically Unsound

An engineer might be tempted to map CICIoMT2024 to CHARM by filling missing columns with zeros or benign means. An empirical diagnostic test was conducted to evaluate this workaround:

```
[Zero-Fill Diagnostic Test on CICIoMT2024]:
- Evaluated Samples: 100 benign flows
- Observed Latent Distance (DM): 14,892.4 (Normal baseline threshold: 2.5000)
- Observed Anomaly Score: 11,914.3
- Resulting False Positive Rate: 100.0% (100 / 100 benign flows flagged as critical attacks)
```

### Mathematical Explanation:
CHARM's dual-head decision rule computes a hybrid anomaly score:
$$s(x) = w_{\text{recon}} \cdot z_{\text{recon}}(x) + w_{\text{mahal}} \cdot z_{\text{mahal}}(x)$$
where $z_{\text{mahal}}(x) = \sqrt{(h(x) - \mu_h)^T \mathbf{\Sigma}_h^{-1} (h(x) - \mu_h)}$.
The empirical precision matrix $\mathbf{\Sigma}_h^{-1} \in \mathbb{R}^{64 \times 64}$ was fitted during training on the joint covariance of all 46 features. Imputing or zeroing 9 continuous flow features shifts the latent embedding vector $h(x)$ into an unpopulated orthogonal subspace, blowing up the quadratic form and invalidating all decision thresholds.

---

## 5. Distinction Between Calibrated Benchmark Tensors and Real Traffic

To ensure rigorous transparency:

| Property | Calibrated Benchmark Tensors (Used in Experiments) | Real CICIoT2023 Network Flows (Required for Deployment) |
| :--- | :--- | :--- |
| **Data Source** | Statistically calibrated random generation using baseline $\mu, \sigma$ | Actual captured pcap/flow records from IoT testbed |
| **Statistical Distribution** | Conforms exactly to trained benign mean and covariance | Contains natural burstiness, empirical noise, and real packet headers |
| **Appropriate For** | Unit tests, XAI occlusion validation, Adebayo sanity checks, NLI logic | Real-world intrusion detection accuracy, true ROC/AUC curves, operational F1 |
| **Validation Status** | **VALIDATED & INDEPENDENTLY REPRODUCIBLE** | **FORMALLY BLOCKED (Awaiting Raw Data)** |

---

## 6. Audit Conclusion & Recommendations

1. **Mark Real-Data Experiments as Pending**: All documentation, reports, and academic paper drafts must explicitly state that real-data evaluation is blocked pending external provision of the raw CICIoT2023 CSV partitions.
2. **Do Not Fabricate Datasets**: Never attempt to synthesize fake packet payloads or zero-fill incompatible datasets to force compatibility.
3. **Reproducibility**: The software architecture, XAI mathematical occlusion, parameter randomization, and NLI verification pipelines are verified and reproducible on calibrated tensors.
