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
    Strict field-level validation of an asserted telemetry claim against CHARMIncident ground-truth.
    Returns:
        (status, action, presented_text, reason)
    Statuses:
        'SUPPORTED', 'CONTRADICTED', 'UNSUPPORTED', 'UNVERIFIABLE'
    """
    text = claim_text
    contradictions = []
    supported_fields = []

    # 1. Check Asserted Anomaly Score
    m_score = re.search(r'(?:anomaly|hybrid)?\s*score\s+(?:is\s+|of\s+)?([0-9]+\.[0-9]+)', text, re.IGNORECASE)
    if m_score:
        asserted_score = float(m_score.group(1))
        if abs(asserted_score - incident.anomaly_score) > 0.05:
            contradictions.append(f"asserted score {asserted_score:.3f} contradicts actual {incident.anomaly_score:.3f}")
        else:
            supported_fields.append("anomaly_score")

    # 2. Check Decision Threshold
    m_thresh = re.search(r'threshold\s+(?:is\s+|of\s+)?([0-9]+\.[0-9]+)', text, re.IGNORECASE)
    if m_thresh:
        asserted_thresh = float(m_thresh.group(1))
        if abs(asserted_thresh - incident.decision_threshold) > 0.05:
            contradictions.append(f"asserted threshold {asserted_thresh:.3f} contradicts actual {incident.decision_threshold:.3f}")
        else:
            supported_fields.append("threshold")

    # 3. Check Reconstruction Error
    m_recon = re.search(r'reconstruction\s+error\s+(?:is\s+|of\s+)?([0-9]+\.[0-9]+)', text, re.IGNORECASE)
    if m_recon:
        asserted_recon = float(m_recon.group(1))
        if abs(asserted_recon - incident.reconstruction_error) > 0.05:
            contradictions.append(f"asserted reconstruction error {asserted_recon:.4f} contradicts actual {incident.reconstruction_error:.4f}")
        else:
            supported_fields.append("reconstruction_error")

    # 4. Check Mahalanobis Distance
    m_mahal = re.search(r'mahalanobis\s+distance\s+(?:is\s+|of\s+)?([0-9]+\.[0-9]+)', text, re.IGNORECASE)
    if m_mahal:
        asserted_mahal = float(m_mahal.group(1))
        if abs(asserted_mahal - incident.mahalanobis_distance) > 0.5:
            contradictions.append(f"asserted mahalanobis distance {asserted_mahal:.4f} contradicts actual {incident.mahalanobis_distance:.4f}")
        else:
            supported_fields.append("mahalanobis_distance")

    # 5. Check Classification State (ANOMALOUS vs BENIGN)
    if "evaluated" in text.lower() or "status" in text.lower():
        if re.search(r'\bas\s+ANOMALOUS\b', text) and not incident.anomaly_detected:
            contradictions.append("asserted ANOMALOUS but actual decision is BENIGN")
        elif re.search(r'\bas\s+BENIGN\b', text) and incident.anomaly_detected:
            contradictions.append("asserted BENIGN but actual decision is ANOMALOUS")
        elif (re.search(r'\bas\s+ANOMALOUS\b', text) and incident.anomaly_detected) or (re.search(r'\bas\s+BENIGN\b', text) and not incident.anomaly_detected):
            supported_fields.append("decision_status")

    # 6. Check Asserted Features
    if "non-existent" in text.lower() or "bogus" in text.lower():
        contradictions.append("asserts non-existent flow feature")

    top_features = [f.feature for f in incident.top_xai_features]
    if "most influential features as:" in text or "influential features" in text:
        for feat in top_features:
            if feat in text:
                supported_fields.append(f"feature_{feat}")
    else:
        for f in incident.top_xai_features:
            if f.feature in text:
                supported_fields.append(f"feature_{f.feature}")

    # Check Trigger Policies
    if incident.trigger_reasons:
        for r in incident.trigger_reasons:
            if r in text:
                supported_fields.append(f"trigger_{r}")

    # Strict Decision logic
    if contradictions:
        status = "CONTRADICTED"
        action = "CONTRADICTION_RETRACTED"
        reason = "; ".join(contradictions)
        presented_text = (
            f"[CONTRADICTED TELEMETRY RETRACTED] Telemetry contradiction detected ({reason}). "
            f"Actual PAADU/CHARM telemetry: anomaly_score={incident.anomaly_score:.3f}, "
            f"threshold={incident.decision_threshold:.3f}, status={'ANOMALOUS' if incident.anomaly_detected else 'BENIGN'}."
        )
    elif supported_fields:
        status = "SUPPORTED"
        action = "RETAINED_VERIFIED"
        reason = f"Verified fields: {', '.join(supported_fields)}"
        presented_text = f"[VERIFIED TELEMETRY] {claim_text}"
    else:
        status = "UNSUPPORTED"
        action = "FLAGGED_WITH_UNCERTAINTY"
        reason = "No verifiable telemetry fields identified matching incident schema"
        presented_text = f"[UNVERIFIED TELEMETRY - MISMATCH] {claim_text}"

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
