# PAADU + Reconstructed Hybrid GraphRAG — Integrated Working Package

This package bundles **two runnable components in one repository**:

1. **Professor's original CHARM network-anomaly detector** and XAI wiring (`charm_original/project`), with original weights and metadata (`charm_original/artifacts`). The original source is preserved, not rewritten.
2. **Reconstructed hybrid GraphRAG retrieval** and evidence-aware NLI verification (`project/`). The final-year student's report contained only selected source listings; this is **not the student's complete original implementation**.

## Quick start (Python 3.10–3.11 recommended)

```bash
python -m pip install -r requirements.txt
bash run_all.sh
```

Expected: retrieval/integration checks pass, original CHARM network-anomaly scoring and XAI smoke test passes. CHARM smoke test trains a **small synthetic-data test model**, not a production evaluation. For real NLI model inference (requires Hugging Face internet access and model download):

```bash
python project/run_demo.py --nli
```

The `project/Phase6_Colab_Reference.ipynb` notebook contains the earlier full TF-IDF/GraphRAG → FLAN-T5 generation → NLI → conservative correction workflow, with HotpotQA evaluation and CHARM smoke test. It requires Colab and internet access.

## Scientific boundaries

- CHARM **has not been adapted to hallucination detection**. It is executed independently on network traffic. There is no scientifically validated CHARM-to-text connection.
- GraphRAG is a reconstruction; complete original student source was unavailable.
- The default quick test does **not** load the NLI model, generate LLM answers, or prove hallucination detection performance. It prints `NOT_EVALUATED` explicitly. Use `--nli` for model-based verification.
- Correction is deliberately non-destructive unless an independently supported replacement can be established. This package does not claim successful general-purpose self-correction.
- The included Phase 6 notebook is a reference workflow, not an offline production application.
