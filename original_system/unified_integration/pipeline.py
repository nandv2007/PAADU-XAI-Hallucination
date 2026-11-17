#!/usr/bin/env python3
"""
Unified PAADU/CHARM + SVA-GRAG-Core Pipeline.
Delivers the complete end-to-end integration:
[Network Input] -> [CHARM MAE + Mahalanobis] -> [XAI Occlusion] ->
[Structured Incident Adapter] -> [Incident Query Builder] ->
[Adaptive GraphRAG Retrieval] -> [Quality Assessment & Bounded Refinement] ->
[Evidence-Grounded Explainer] -> [NLI Hallucination Verification & Correction] ->
[Final Verified Incident Report]
"""
from __future__ import annotations

import os
import sys
import time
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
INTEG_PROJECT = WORKSPACE / 'integrated_project'
CHARM_ORIG = INTEG_PROJECT / 'charm_original'

# Add paths to load unmodified original modules
for p in [str(CHARM_ORIG / 'project'), str(CHARM_ORIG / 'project' / 'original'), str(INTEG_PROJECT / 'project')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from charm_engine import CHARMEngine
from original.paadu_benchmark_v7 import SemanticAnomalyMAE, VolumeAdaptiveRelaxer
from xai_interpreter import XAIInterpreter

from unified_integration.adapters.charm_incident_adapter import CHARMIncidentAdapter, CHARMIncident
from unified_integration.adapters.incident_query_builder import IncidentQueryBuilder, InvestigationQuery
from unified_integration.retrieval.adaptive_controller import AdaptiveRetrievalController, RetrievalResult
from unified_integration.retrieval.quality_assessor import RetrievalQualityAssessor
from unified_integration.retrieval.refinement_controller import RefinementController, RefinementResult
from unified_integration.generation.incident_explainer import IncidentExplainer, IncidentExplanation
from unified_integration.verification.verification_orchestrator import VerificationOrchestrator, VerificationSummary

class UnifiedPAADUPipeline:
    """The master orchestrator connecting PAADU/CHARM and SVA-GRAG-Core."""

    def __init__(
        self,
        engine: CHARMEngine,
        xai: XAIInterpreter,
        checkpoint_path: str,
        enable_nli: bool = True,
        max_refinement_iterations: int = 2
    ):
        self.engine = engine
        self.xai = xai
        self.checkpoint_path = checkpoint_path
        self.enable_nli = enable_nli

        # Initialize integration components
        self.incident_adapter = CHARMIncidentAdapter(checkpoint_path=checkpoint_path)
        self.query_builder = IncidentQueryBuilder(max_features=4)
        self.retrieval_controller = AdaptiveRetrievalController()
        self.quality_assessor = RetrievalQualityAssessor()
        self.refinement_controller = RefinementController(
            retrieval_controller=self.retrieval_controller,
            quality_assessor=self.quality_assessor,
            max_iterations=max_refinement_iterations
        )
        self.explainer = IncidentExplainer()
        self.verifier = VerificationOrchestrator(enable_nli=enable_nli)

    @classmethod
    def from_artifacts(
        cls,
        weights_path: Optional[str] = None,
        state_path: Optional[str] = None,
        enable_nli: bool = True,
        max_refinement_iterations: int = 2
    ) -> UnifiedPAADUPipeline:
        """Instantiates the complete pipeline from existing original model weights and state."""
        w_path = weights_path or str(CHARM_ORIG / 'artifacts' / 'charm_model_weights.pt')
        s_path = state_path or str(CHARM_ORIG / 'artifacts' / 'charm_engine_state.pkl')

        if not os.path.exists(w_path):
            raise FileNotFoundError(f"Model weights not found at: {w_path}")
        if not os.path.exists(s_path):
            raise FileNotFoundError(f"Engine state not found at: {s_path}")

        with open(s_path, 'rb') as f:
            state = pickle.load(f)

        # 1. Instantiate unmodified SemanticAnomalyMAE
        model = SemanticAnomalyMAE(
            num_nodes=state['config'].num_nodes,
            seq_len=state['config'].seq_len,
            feature_dim=len(state['features']),
            embed_dim=state['config'].embed_dim,
            mask_ratio=state['config'].mask_ratio,
            use_gnn=state['config'].use_gnn
        )
        weights = torch.load(w_path, map_location='cpu')
        model.load_state_dict(weights, strict=True)
        model.eval()

        # 2. Reconstruct CHARMEngine state
        engine = CHARMEngine(state['config'], state['features'])
        engine.mean = state['mean']
        engine.std = state['std']
        engine.mse_stats = state['mse_stats']
        engine.dm_stats = state['dm_stats']
        engine.floor = state['floor']
        engine.mahal = state['mahal']
        engine.original_cfg = state['original_cfg']
        engine.model = model
        engine.relaxer = VolumeAdaptiveRelaxer(engine.cfg.hard_z_floor, engine.original_cfg)
        engine.relaxer.set_floor_value(engine.floor, reason='held-out benign calibration')
        engine.fitted = True

        # 3. Instantiate unmodified XAIInterpreter
        xai = XAIInterpreter(engine)

        return cls(
            engine=engine,
            xai=xai,
            checkpoint_path=w_path,
            enable_nli=enable_nli,
            max_refinement_iterations=max_refinement_iterations
        )

    def analyze(
        self,
        network_input: Union[np.ndarray, torch.Tensor],
        metadata: Optional[List[Dict[str, Any]]] = None,
        save_reports: bool = True,
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes complete end-to-end analysis on input telemetry blocks [B, 1, 10, 46].
        Returns detailed report dictionaries for each telemetry block.
        """
        t_pipeline_start = time.time()
        blocks = np.asarray(network_input, dtype=np.float32)
        
        # Validation
        if blocks.ndim != 4 or blocks.shape[1:3] != (self.engine.cfg.num_nodes, self.engine.cfg.seq_len) or blocks.shape[3] != len(self.engine.features):
            raise ValueError(
                f"Input tensor shape mismatch. Expected [B, {self.engine.cfg.num_nodes}, {self.engine.cfg.seq_len}, {len(self.engine.features)}], "
                f"received {list(blocks.shape)}"
            )

        n_blocks = len(blocks)
        meta_list = metadata or [{'block_id': i} for i in range(n_blocks)]

        # STAGE 1 & 2: Original CHARM Inference & Decision
        t0 = time.time()
        raw_scores = self.engine.score(blocks)
        t_inference = round(time.time() - t0, 4)

        # STAGE 3: Original XAI Occlusion
        t0 = time.time()
        explanations = self.xai.explain(blocks, raw_scores, metadata=meta_list)
        t_xai = round(time.time() - t0, 4)

        reports = []
        out_path = Path(output_dir or (WORKSPACE / 'unified_integration' / 'outputs'))
        out_path.mkdir(parents=True, exist_ok=True)

        for i in range(n_blocks):
            exp_record = explanations[i]
            block_meta = meta_list[i] if i < len(meta_list) else {'block_id': i}
            block_ref = block_meta.get('block_identifier', f"telemetry_block_{i}")

            # STAGE 4: Structured Incident Adapter
            t0 = time.time()
            incident = self.incident_adapter.create_incident(
                charm_record=exp_record,
                block_identifier=block_ref,
                custom_metadata=block_meta
            )
            t_adapter = round(time.time() - t0, 4)

            # STAGE 5: Incident-to-Query Generation
            t0 = time.time()
            investigation_query = self.query_builder.build_query(incident)
            t_query = round(time.time() - t0, 4)

            # STAGE 6: Adaptive GraphRAG Retrieval
            t0 = time.time()
            initial_retrieval = self.retrieval_controller.retrieve(investigation_query.query_text)
            t_retrieval = round(time.time() - t0, 4)

            # STAGE 7 & 8: Retrieval Quality Assessment & Bounded Refinement
            t0 = time.time()
            refinement_res = self.refinement_controller.refine_evidence(
                initial_retrieval=initial_retrieval,
                target_facets=investigation_query.investigation_facets
            )
            t_refinement = round(time.time() - t0, 4)

            # STAGE 9: Evidence-Grounded Explanation
            t0 = time.time()
            explanation = self.explainer.generate_explanation(
                incident=incident,
                evidence=refinement_res.final_evidence
            )
            t_explainer = round(time.time() - t0, 4)

            # STAGE 10 & 11: NLI Hallucination Verification & Conservative Correction
            t0 = time.time()
            verification_summary = self.verifier.verify_and_correct(
                explanation=explanation,
                incident=incident,
                evidence=refinement_res.final_evidence
            )
            t_verifier = round(time.time() - t0, 4)

            # Total execution time for this block
            t_block_total = round(t_inference + t_xai + t_adapter + t_query + t_retrieval + t_refinement + t_explainer + t_verifier, 4)

            report = {
                'incident_id': incident.incident_id,
                'input_reference': incident.input_reference,
                'timestamp': incident.timestamp,
                'charm_inference': {
                    'anomaly_detected': incident.anomaly_detected,
                    'hybrid_score': incident.anomaly_score,
                    'decision_threshold': incident.decision_threshold,
                    'trigger_reasons': incident.trigger_reasons,
                    'reconstruction_error': incident.reconstruction_error,
                    'mahalanobis_distance': incident.mahalanobis_distance,
                    'reconstruction_z': incident.reconstruction_z,
                    'latent_z': incident.latent_z,
                    'top_xai_features': [asdict(f) for f in incident.top_xai_features],
                    'xai_summary': incident.xai_summary_text
                },
                'investigation_query': investigation_query.to_dict(),
                'retrieval': {
                    'strategy': initial_retrieval.retrieval_strategy,
                    'query_complexity': initial_retrieval.complexity.to_dict(),
                    'initial_evidence_count': len(initial_retrieval.retrieved_evidence),
                    'refined_evidence_count': len(refinement_res.final_evidence),
                    'refinement_iterations': refinement_res.iterations_executed,
                    'refinement_termination': refinement_res.termination_reason,
                    'evidence_quality': refinement_res.final_quality.to_dict(),
                    'retrieved_evidence': [e.to_dict() for e in refinement_res.final_evidence]
                },
                'explanation': {
                    'explanation_id': explanation.explanation_id,
                    'observed_facts': explanation.observed_facts,
                    'supported_interpretations': explanation.supported_interpretations,
                    'unverified_hypotheses': explanation.unverified_hypotheses,
                    'claims': [c.to_dict() for c in explanation.claims],
                    'citations': explanation.citations
                },
                'verification_and_correction': verification_summary.to_dict(),
                'stage_runtimes_sec': {
                    'charm_inference': t_inference,
                    'charm_xai': t_xai,
                    'incident_adapter': t_adapter,
                    'query_builder': t_query,
                    'retrieval': t_retrieval,
                    'refinement': t_refinement,
                    'explanation_generation': t_explainer,
                    'nli_verification_and_correction': t_verifier,
                    'total_block_sec': t_block_total
                },
                'provenance_trace': {
                    'model_checkpoint_hash': incident.model_checkpoint_hash,
                    'feature_schema': incident.feature_schema,
                    'nli_model': 'cross-encoder/nli-MiniLM2-L6-H768' if self.enable_nli else 'DISABLED',
                    'knowledge_base_sources': [c['citation'] for c in explanation.citations]
                }
            }

            reports.append(report)

            if save_reports:
                # Save JSON
                json_file = out_path / f"incident_report_{incident.incident_id}.json"
                with open(json_file, 'w') as jf:
                    json.dump(report, jf, indent=2)

                # Save Human-readable Markdown
                md_file = out_path / f"incident_report_{incident.incident_id}.md"
                with open(md_file, 'w') as mf:
                    mf.write(self._format_markdown_report(report))

        return reports

    @staticmethod
    def _format_markdown_report(report: Dict[str, Any]) -> str:
        inc = report['charm_inference']
        qry = report['investigation_query']
        ret = report['retrieval']
        ver = report['verification_and_correction']
        runtimes = report['stage_runtimes_sec']

        top_feats = ", ".join([f"{f['feature']} ({f['hybrid_contribution']:+.2f})" for f in inc['top_xai_features'][:4]])
        citations_str = "\n".join([f"- **[{c['title']}]**: {c['citation']}" for c in report['explanation']['citations']]) or "- None"

        md = f"""# Incident Investigation Report: {report['incident_id']}
**Input Reference:** `{report['input_reference']}` | **Timestamp:** `{report['timestamp']}`  
**Model:** PAADU/CHARM + SVA-GRAG-Core Integrated Pipeline  

---

## 1. PAADU/CHARM Anomaly Detection & XAI
- **Anomaly Decision:** {'🚨 **MALICIOUS / ANOMALY DETECTED**' if inc['anomaly_detected'] else '✅ **BENIGN / NORMAL TELEMETRY**'}
- **Hybrid Anomaly Score:** `{inc['hybrid_score']:.4f}` (Calibrated Decision Threshold: `{inc['decision_threshold']:.4f}`)
- **Reconstruction MSE:** `{inc['reconstruction_error']:.4f}` | **Latent Mahalanobis Distance:** `{inc['mahalanobis_distance']:.4f}`
- **Active Trigger Policies:** `{', '.join(inc['trigger_reasons']) or 'None'}`
- **Top Contributing Features (Occlusion):** {top_feats}

---

## 2. Synthesized Cybersecurity Investigation Query
- **Generated Query:** *"{qry['query_text']}"*
- **Query Complexity:** `{ret['query_complexity']['category']}` (Score: `{ret['query_complexity']['complexity_score']:.2f}`)
- **Retrieval Strategy:** `{ret['strategy']}`

---

## 3. Evidence Retrieval & Quality Assessment
- **Retrieved Evidence Documents:** {ret['refined_evidence_count']} documents
- **Quality Score:** `{ret['evidence_quality']['composite_score']:.3f}` (`{ret['evidence_quality']['quality_status']}`)
  - Relevance: `{ret['evidence_quality']['relevance_score']:.2f}` | Coverage: `{ret['evidence_quality']['coverage_score']:.2f}` | Provenance: `{ret['evidence_quality']['provenance_score']:.2f}`
- **Refinement Termination:** {ret['refinement_termination']}

---

## 4. Evidence-Grounded Verified Incident Narrative
{ver['final_verified_narrative']}

---

## 5. Authoritative Document Citations
{citations_str}

---

## 6. Execution Trace & Runtimes
| Stage | Execution Time |
|---|---|
| CHARM Telemetry Inference | {runtimes['charm_inference']:.4f}s |
| XAI Feature Attribution | {runtimes['charm_xai']:.4f}s |
| Incident Adaptation & Query Generation | {runtimes['incident_adapter'] + runtimes['query_builder']:.4f}s |
| Adaptive GraphRAG Retrieval & Refinement | {runtimes['retrieval'] + runtimes['refinement']:.4f}s |
| Evidence-Grounded Explanation & NLI Verification | {runtimes['explanation_generation'] + runtimes['nli_verification_and_correction']:.4f}s |
| **Total Pipeline Latency** | **{runtimes['total_block_sec']:.4f}s** |
"""
        return md
