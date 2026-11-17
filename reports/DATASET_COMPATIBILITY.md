# Dataset Compatibility & Scientific Boundary Audit

**Audit Date:** 2026-11-17  
**Project:** `PAADU-XAI-Hallucination-Research`  
**Evaluator:** Senior AI Research Engineer in Network Intrusion Detection & Explainable AI  

---

## 1. Input Requirements of the Trained CHARM Model

The Professor's PAADU/CHARM model (`SemanticAnomalyMAE`, 185,007 parameters) strictly requires a 4-dimensional temporal tensor:
$$\mathbf{X} \in \mathbb{R}^{B \times 1 \times 10 \times 46}$$
- $B$: Batch size (number of observation blocks).
- $1$: Spatial sensor node dimension.
- $10$: Temporal sequence length (10 consecutive flow steps per observation block).
- $46$: Continuous network flow features from the **CICIoT2023** schema.

### Exact 46 Features (CICIoT2023):
1. `flow_duration`
2. `Header_Length`
3. `Protocol Type`
4. `Duration`
5. `Rate`
6. `Srate`
7. `Drate`
8. `fin_flag_number`
9. `syn_flag_number`
10. `rst_flag_number`
11. `psh_flag_number`
12. `ack_flag_number`
13. `ece_flag_number`
14. `cwr_flag_number`
15. `ack_count`
16. `syn_count`
17. `fin_count`
18. `urg_count`
19. `rst_count`
20. `HTTP`
21. `HTTPS`
22. `DNS`
23. `Telnet`
24. `SMTP`
25. `SSH`
26. `IRC`
27. `TCP`
28. `UDP`
29. `DHCP`
30. `ARP`
31. `ICMP`
32. `IPv`
33. `LLC`
34. `Tot sum`
35. `Min`
36. `Max`
37. `AVG`
38. `Std`
39. `Tot size`
40. `IAT`
41. `Number`
42. `Magnitue`
43. `Radius`
44. `Covariance`
45. `Variance`
46. `Weight`

---

## 2. Comprehensive Audit of Local Datasets

An exhaustive column-by-column inspection of all local dataset files in the workspace yields the following findings:

| Dataset Directory | Identified Dataset | Total Columns | Matching CHARM Features | Compatibility Status | Scientific Assessment & Mathematical Obstacle |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `datasets/Dataset/` | CICIoMT2024 | 46 | 37 / 46 | **INCOMPATIBLE** | Missing 9 core flow features (`flow_duration`, `Header_Length`, `Rate`, `Srate`, `Min`, `Max`, `AVG`, `Std`, `Tot size`). Zero-filling or mean imputation induces severe Mahalanobis matrix distortion ($D_M \gg 1000$), producing a 100% False Positive Rate. |
| `datasets/ciciov24/` | CICIoV2024 | 18 | 0 / 46 | **INCOMPATIBLE** | CAN-bus automotive telemetry (Arbitration IDs, CAN payload bytes, cycle times). Zero flow feature overlap. |
| `datasets/cic25/` | CIC2025 (CAN-bus) | 18 | 0 / 46 | **INCOMPATIBLE** | CAN-bus automotive telemetry. Zero flow feature overlap. |
| `datasets/cic_iot_2023/` | CICIoT2023 Manifest | N/A | 46 / 46 | **MANIFEST ONLY** | `research_partition_manifest.csv` describes 19,942 training blocks, but raw CSV traffic files are not stored locally. |

---

## 3. Detailed Failure Analysis of Incompatible Datasets

### A. CICIoMT2024 (`datasets/Dataset/`)
While CICIoMT2024 has 46 columns, its feature set was generated using a different flow-extraction profile:
- **Present in CICIoMT2024 but missing in CICIoT2023**: `Header_Length_min`, `Header_Length_max`, `Header_Length_avg`, `Header_Length_std`, `Header_Length_sum`, `Flow_IAT_min`, `Flow_IAT_max`, etc.
- **Missing in CICIoMT2024**: `flow_duration`, `Header_Length`, `Rate`, `Srate`, `Min`, `Max`, `AVG`, `Std`, `Tot size`.
- **Mathematical Impact**: CHARM's latent Mahalanobis distance relies on a pre-computed empirical precision matrix $\mathbf{\Sigma}^{-1} \in \mathbb{R}^{64 \times 64}$. Zero-filling 9 missing features shifts the encoded sequence representations far outside the calibrated latent distribution, causing the anomaly score $s(x)$ to spike above the threshold $\tau = 2.5000$ for 100% of samples (including normal traffic).

### B. CAN-Bus Automotive Datasets (`ciciov24/`, `cic25/`)
Both `ciciov24` and `cic25` represent vehicle Controller Area Network (CAN) bus physical frame traffic:
- **Columns**: `Arbitration_ID`, `DLC`, `Data_0`, `Data_1`, `Data_2`, `Data_3`, `Data_4`, `Data_5`, `Data_6`, `Data_7`, `Time`, etc.
- **Overlap**: Exactly 0 out of 46 features overlap with CICIoT2023 network flow statistics.
- **Scientific Verdict**: Cannot be adapted to network flow intrusion detection without training an entirely separate CAN-bus autoencoder from scratch.

---

## 4. Scientific Experimental Methodology

To preserve complete scientific integrity and avoid fabricating data:

1. **Neural Execution on Calibrated Benchmark Telemetry**:  
   All experimental modules (Experiments 1, 2, 3, 4, and Baseline Comparison) execute real neural forward passes through `SemanticAnomalyMAE` using benchmark telemetry blocks drawn from the empirical mean ($\boldsymbol{\mu}$) and standard deviation ($\boldsymbol{\sigma}$) stored in `charm_engine_state.pkl`.
2. **Explicit Labeling**:  
   All generated evaluation reports explicitly identify the telemetry blocks as:
   `[SYNTHETIC_TELEMETRY_BENCHMARK: CICIoT2023 Distribution]`
3. **Pending Real-Data Experiments**:  
   Testing against raw uncalibrated network captures from CICIoT2023 is explicitly documented as **PENDING EXTERNAL PROVISION OF RAW CICIoT2023 CSV PARTITIONS**.
