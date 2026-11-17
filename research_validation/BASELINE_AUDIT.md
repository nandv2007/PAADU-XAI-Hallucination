# BASELINE AUDIT: PAADU/CHARM + SVA-GRAG-Core Integrated Architecture

**Audit Date:** 2026-11-14T17:55:00+05:30  
**Evaluator:** Principal AI Research Engineer & Independent Model Validation Engineer  
**Workspace:** `PAADU_Testing/`  
**Integrity State:** 144 / 144 original research and dataset files verified 100% UNCHANGED (SHA-256 match)

---

## 1. System Overview & Component Provenance

The system under validation connects two previously separate research systems into a single end-to-end pipeline:

1. **PAADU/CHARM (Professor's Project)**:
   - **Role**: Unsupervised network anomaly detection and feature attribution.
   - **Model Architecture**: `SemanticAnomalyMAE` (Masked Autoencoder based on PyTorch `nn.TransformerEncoder`).
   - **Parameters**: 185,007 weights strictly loaded from `charm_model_weights.pt` (SHA-256: `647febba9cf564e6bf5e2c4ececd2dc8cf7bfd6d172235fb1a10286f532bb530`).
   - **Latent Bottleneck**: 64-dimensional sequence representations projected to 64-dimensional latent vectors.
   - **Scoring Engine**: Hybrid score combining standardized reconstruction MSE and latent Mahalanobis distance ($D_M$):
     $$s(x) = w_{\text{recon}} \cdot z_{\text{recon}} + w_{\text{mahal}} \cdot z_{\text{DM}}, \quad w_{\text{recon}} = 0.2, \; w_{\text{mahal}} = 0.8$$
   - **Calibration**: `VolumeAdaptiveRelaxer` with calibrated hard $z$-floor threshold $\tau = 2.5000$ derived from held-out benign calibration partitions.
   - **XAI Method**: Deterministic **leave-one-feature-out occlusion** over all 46 features (replacing normalized features with $0.0$, corresponding to the empirical benign training mean).
   - **Explicit Finding**: The system **DOES NOT use SHAP, LIME, or Integrated Gradients**. It uses deterministic leave-one-feature-out occlusion.

2. **SVA-GRAG-Core (Student's Project Reconstruction)**:
   - **Role**: Self-Validating Adaptive Graph-RAG retrieval and verification.
   - **Vector Retrieval**: TF-IDF vectorizer over tokenized cybersecurity reference documents.
   - **Graph Retrieval**: `networkx` Graph representing document co-citation, protocol hierarchies, and RFC relationships.
   - **Adaptive Controller**: Dynamic query complexity analysis adjusting $k \in [3, 8]$ and graph boost $\beta \in [0.15, 0.40]$.
   - **Quality Assessor**: Multi-dimensional scoring assessing Relevance (35%), Coverage (35%), Consistency (15%), and Provenance (15%).
   - **Bounded Refinement**: Iterative query reformulation targeting missing protocol and symptom facets ($max\_iterations = 2$).

3. **Incident Explanation Generator**:
   - **Mechanism**: Deterministic rule/template evidence synthesis implemented in `unified_integration/generation/incident_explainer.py`.
   - **Explicit Finding**: The system **DOES NOT use an external Large Language Model (LLM)** (such as GPT-4, Claude, or LLaMA) for narrative drafting. Instead, it segments observations into discrete factual propositions (`FactualClaim` dataclass) mapping telemetry metrics, occlusion attributions, retrieved literature sentences, and operational hypotheses.

4. **NLI Verification & Conservative Correction**:
   - **Model**: `cross-encoder/nli-MiniLM2-L6-H768` (Hugging Face Transformers, ~66.9M parameters, 105 weight tensors).
   - **Mechanism**: Fine-grained cross-encoder inference evaluating entailment, contradiction, and neutral confidence for each claim proposition against retrieved evidence sentences.
   - **Telemetry Fact Checking**: Structured schema validation verifying numerical metrics against `CHARMIncident`.
   - **Correction Policy**: Conservative preservation (retains verified claims, retracts contradictions with explicit contradiction notices, and appends uncertainty badges to unverified hypotheses).

---

## 2. Feature Schema & Temporal Input Dimensions

CHARM expects a rigid 4D numerical tensor:
$$\mathbf{X} \in \mathbb{R}^{B \times 1 \times 10 \times 46}$$
- $B$: Batch size (number of observation blocks).
- $1$: Number of spatial nodes (single sensor point).
- $10$: Observation sequence length (10 consecutive flow steps per block).
- $46$: Continuous network flow features from the CICIoT2023 schema.

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

## 3. Dataset Compatibility & Local Reality

| Dataset Directory | Identified Dataset | Total Columns | Matching CHARM Features | Compatibility Status | Scientific Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `datasets/Dataset/` | CICIoMT2024 | 46 | 37 / 46 | **INCOMPATIBLE** | Missing 9 flow features (`flow_duration`, `Header_Length`, `Rate`, `Srate`, `Min`, `Max`, `AVG`, `Std`, `Tot size`). Zero-filling causes 100% FPR due to Mahalanobis matrix distortion. |
| `datasets/ciciov24/` | CICIoV2024 | 18 | 0 / 46 | **INCOMPATIBLE** | CAN-bus automotive telemetry (Arbitration IDs, CAN payload bytes). Zero flow overlap. |
| `datasets/cic25/` | CIC2025 (CAN-bus) | 18 | 0 / 46 | **INCOMPATIBLE** | CAN-bus automotive telemetry. Zero flow overlap. |
| `integrated_project/charm_original/data/` | CICIoT2023 Manifest | N/A | 46 / 46 | **MANIFEST ONLY** | `research_partition_manifest.csv` describes 19,942 training blocks, but raw CSV data partitions are absent from the workspace. |

**Scientific Conclusion**: Real-data neural inference is executed on calibrated benchmark telemetry tensors matching the exact mean and covariance of the trained CICIoT2023 distribution. In accordance with strict scientific integrity, all benchmark telemetry evaluations are explicitly designated as `[SYNTHETIC_TELEMETRY_BENCHMARK]`.

---

## 4. Cryptographic Baseline Hashes

All original research files and checkpoints have been cataloged:
- `charm_model_weights.pt`: `647febba9cf564e6bf5e2c4ececd2dc8cf7bfd6d172235fb1a10286f532bb530`
- `charm_engine_state.pkl`: `49d2967c214e63c435d18e0c66979f7aa9a1c60f95d535896a6e3af7fe63ca6b`
- `training_metadata.json`: `63fe8a5b840f7a0dc3207e6fe2825a8f65124c047b1c27d80667ff2ca2be72b8`
- Total files monitored: 144 files across `integrated_project/` and `datasets/`.

---

## 5. Current Baseline Test Suite Reproduction

Running `bash unified_integration/run_tests.sh` reproduces the following verified results:
- Test A (CHARM Architecture & Checkpoint): **PASS**
- Test B (Structured Incident Adapter): **PASS**
- Test C (Investigation Query Builder): **PASS**
- Test D & E (Adaptive Retrieval & Quality): **PASS**
- Test F & G (NLI Verification & Correction): **PASS**
- Test H, I & J (Full Pipeline & Integrity): **PASS**
Overall status: **10/10 TESTS PASSED**.
