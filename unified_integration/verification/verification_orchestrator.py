#!/usr/bin/env python3
"""
NLI Verification and Conservative Correction Orchestrator.
Coordinates hallucination verification using the original transformer NLI model
(cross-encoder/nli-MiniLM2-L6-H768).
Distinguishes numerical telemetry claims from conceptual cybersecurity claims.
Applies safe, conservative correction and uncertainty flagging without arbitrary text rewriting.
"""
from __future__ import annotations

import sys
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT_DIR = WORKSPACE / 'integrated_project' / 'project'
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from integrated_pipeline import EvidenceVerifier, Evidence
from unified_integration.adapters.charm_incident_adapter import CHARMIncident
from unified_integration.generation.incident_explainer import FactualClaim, IncidentExplanation
from unified_integration.retrieval.adaptive_controller import EnrichedEvidence
from unified_integration.verification.telemetry_grammar import (
    parse_and_validate_telemetry,
    AtomicAssertionAudit
)

@dataclass
class ClaimVerificationResult:
    claim_id: str
    original_claim_text: str
    claim_type: str
    verification_status: str       # 'VERIFIED_OBSERVED_TELEMETRY', 'SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT_EVIDENCE'
    verification_method: str       # 'TELEMETRY_SCHEMA_CHECK' or 'TRANSFORMER_NLI'
    nli_raw_label: Optional[str]
    nli_confidence: Optional[float]
    evidence_citation: Optional[str]
    evidence_sentence: Optional[str]
    correction_action: str         # 'RETAINED_VERIFIED', 'FLAGGED_WITH_UNCERTAINTY', 'CONTRADICTION_CORRECTED'
    final_presented_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationSummary:
    total_claims: int
    telemetry_facts_verified: int
    evidence_supported_claims: int
    contradicted_claims: int
    insufficient_evidence_claims: int
    verification_rate: float
    claim_results: List[ClaimVerificationResult]
    final_verified_narrative: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_claims': self.total_claims,
            'telemetry_facts_verified': self.telemetry_facts_verified,
            'evidence_supported_claims': self.evidence_supported_claims,
            'contradicted_claims': self.contradicted_claims,
            'insufficient_evidence_claims': self.insufficient_evidence_claims,
            'verification_rate': self.verification_rate,
            'claim_results': [c.to_dict() for c in self.claim_results],
            'final_verified_narrative': self.final_verified_narrative
        }


def validate_telemetry_claim_strict(claim_text: str, incident: CHARMIncident) -> tuple[str, str, str, str]:
    """
    Structured allowlisted grammar-based factual assertion validation of telemetry claims against CHARMIncident ground-truth.
    Individually extracts and audits every asserted numerical measurement, status, feature, and clause.
    
    Fail-closed policy:
    - If ANY assertion contradicts ground-truth telemetry -> CONTRADICTED (CONTRADICTION_RETRACTED).
    - If ANY assertion is unrecognized, ungrounded (e.g. arbitrary attacker identities, unknown features,
      unsupported causal statements, ungrounded IPs) -> UNSUPPORTED (FLAGGED_WITH_UNCERTAINTY).
    - ONLY if ALL asserted facts are supported by authoritative telemetry AND NO unverified or unrecognized
      clause exists -> SUPPORTED (RETAINED_VERIFIED) -> [VERIFIED TELEMETRY].
    """
    status, action, presented_text, reason, _ = parse_and_validate_telemetry(claim_text, incident)
    return status, action, presented_text, reason


class VerificationOrchestrator:
    """Orchestrates NLI verification and conservative correction."""

    def __init__(self, enable_nli: bool = True):
        self.enable_nli = enable_nli
        self.verifier = EvidenceVerifier(load_model=enable_nli)

    def verify_and_correct(
        self,
        explanation: IncidentExplanation,
        incident: CHARMIncident,
        evidence: List[EnrichedEvidence]
    ) -> VerificationSummary:
        # Convert enriched evidence to standard Evidence format expected by EvidenceVerifier
        std_evidence = [Evidence(title=e.title, text=e.text, score=e.similarity_score) for e in evidence]
        
        results: List[ClaimVerificationResult] = []
        verified_lines = []

        telemetry_count = 0
        supported_count = 0
        contradicted_count = 0
        insufficient_count = 0

        for claim in explanation.claims:
            # Type 1: Observed Telemetry Facts -> Strict Field-Level Ground-Truth Check
            if claim.claim_type == "OBSERVED_FACT":
                status, action, pres_text, reason = validate_telemetry_claim_strict(claim.claim_text, incident)
                
                if status == "SUPPORTED":
                    telemetry_count += 1
                elif status == "CONTRADICTED":
                    contradicted_count += 1
                else:
                    insufficient_count += 1

                res = ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    original_claim_text=claim.claim_text,
                    claim_type=claim.claim_type,
                    verification_status=status,
                    verification_method="TELEMETRY_SCHEMA_CHECK",
                    nli_raw_label="FIELD_MATCH" if status == "SUPPORTED" else "CONTRADICTION" if status == "CONTRADICTED" else "UNVERIFIED",
                    nli_confidence=1.0 if status == "SUPPORTED" else 0.0,
                    evidence_citation=f"PAADU/CHARM Execution State ({incident.input_reference})",
                    evidence_sentence=f"Direct model output: score={incident.anomaly_score:.4f}, threshold={incident.decision_threshold:.4f}",
                    correction_action=action,
                    final_presented_text=pres_text
                )
                results.append(res)
                verified_lines.append(res.final_presented_text)
                continue

            # Type 2: Hypothesis Claims -> By definition unverified hypotheses
            if claim.claim_type == "HYPOTHESIS":
                res = ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    original_claim_text=claim.claim_text,
                    claim_type=claim.claim_type,
                    verification_status="HYPOTHESIS_INVESTIGATION_BOUNDARY",
                    verification_method="POLICY_PRESERVATION",
                    nli_raw_label="HYPOTHESIS",
                    nli_confidence=None,
                    evidence_citation="Scientific Methodology (Outside Closed World, Sommer & Paxson)",
                    evidence_sentence="Statistical anomaly metrics require corroboration before attribution.",
                    correction_action="RETAINED_FLAGGED",
                    final_presented_text=f"[OPERATIONAL HYPOTHESIS] {claim.claim_text}"
                )
                results.append(res)
                verified_lines.append(res.final_presented_text)
                continue

            # Type 3: Conceptual Cybersecurity Claims -> Verify using Transformer NLI
            if self.enable_nli and std_evidence:
                verif_out = self.verifier.verify(claim.claim_text, std_evidence)
                label = verif_out.get('label', 'NOT_EVALUATED')
                ev_detail = verif_out.get('evidence')

                if label == 'SUPPORTED':
                    status = 'SUPPORTED'
                    action = 'RETAINED_VERIFIED'
                    pres_text = f"[VERIFIED EVIDENCE] {claim.claim_text}"
                    supported_count += 1
                elif label == 'CONTRADICTED':
                    status = 'CONTRADICTED'
                    action = 'CONTRADICTION_CORRECTED'
                    # Conservative correction: Flag the contradiction and cite correct evidence
                    contra_sent = ev_detail.get('sentence', 'Contradictory evidence found in knowledge base.') if ev_detail else ''
                    pres_text = f"[CONTRADICTED CLAIM RETRACTED] Original: '{claim.claim_text}'. Authoritative evidence states: '{contra_sent}'"
                    contradicted_count += 1
                else:
                    status = 'INSUFFICIENT_EVIDENCE'
                    action = 'FLAGGED_WITH_UNCERTAINTY'
                    # Conservative: Do not delete, append explicit uncertainty flag
                    pres_text = f"[UNVERIFIED - INSUFFICIENT EVIDENCE] {claim.claim_text}"
                    insufficient_count += 1

                res = ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    original_claim_text=claim.claim_text,
                    claim_type=claim.claim_type,
                    verification_status=status,
                    verification_method="TRANSFORMER_NLI",
                    nli_raw_label=ev_detail.get('label') if ev_detail else None,
                    nli_confidence=float(ev_detail.get('confidence')) if ev_detail else None,
                    evidence_citation=f"{ev_detail.get('title')} ({claim.cited_doc_id or 'KB'})" if ev_detail else None,
                    evidence_sentence=ev_detail.get('sentence') if ev_detail else None,
                    correction_action=action,
                    final_presented_text=pres_text
                )
            else:
                # NLI disabled fallback
                res = ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    original_claim_text=claim.claim_text,
                    claim_type=claim.claim_type,
                    verification_status="NOT_EVALUATED",
                    verification_method="NONE",
                    nli_raw_label=None,
                    nli_confidence=None,
                    evidence_citation=claim.cited_doc_title,
                    evidence_sentence=None,
                    correction_action="RETAINED_NO_NLI",
                    final_presented_text=f"[UNVERIFIED (NLI DISABLED)] {claim.claim_text}"
                )
                insufficient_count += 1

            results.append(res)
            verified_lines.append(res.final_presented_text)

        total = len(results)
        verif_rate = round((telemetry_count + supported_count) / total if total > 0 else 0.0, 3)
        final_narrative = "\n\n".join(verified_lines)

        return VerificationSummary(
            total_claims=total,
            telemetry_facts_verified=telemetry_count,
            evidence_supported_claims=supported_count,
            contradicted_claims=contradicted_count,
            insufficient_evidence_claims=insufficient_count,
            verification_rate=verif_rate,
            claim_results=results,
            final_verified_narrative=final_narrative
        )
