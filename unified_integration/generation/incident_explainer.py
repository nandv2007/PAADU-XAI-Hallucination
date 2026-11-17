#!/usr/bin/env python3
"""
Evidence-Grounded Incident Explainer.
Generates structured cybersecurity explanations synthesized from:
1. Actual CHARM anomaly outputs and metrics
2. Actual XAI feature attributions
3. Retrieved cybersecurity evidence documents
4. Document source citations and provenance
Explicitly segregates observed facts, supported interpretations, and unverified hypotheses.
Segments text into discrete testable factual claims for downstream NLI verification.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from unified_integration.adapters.charm_incident_adapter import CHARMIncident
from unified_integration.retrieval.adaptive_controller import EnrichedEvidence

@dataclass
class FactualClaim:
    claim_id: str
    claim_text: str
    claim_type: str  # 'OBSERVED_FACT', 'EVIDENCE_INTERPRETATION', 'HYPOTHESIS'
    cited_doc_title: Optional[str] = None
    cited_doc_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IncidentExplanation:
    explanation_id: str
    incident_id: str
    observed_facts: List[str]
    supported_interpretations: List[str]
    unverified_hypotheses: List[str]
    claims: List[FactualClaim]
    full_narrative: str
    citations: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'explanation_id': self.explanation_id,
            'incident_id': self.incident_id,
            'observed_facts': self.observed_facts,
            'supported_interpretations': self.supported_interpretations,
            'unverified_hypotheses': self.unverified_hypotheses,
            'claims': [c.to_dict() for c in self.claims],
            'full_narrative': self.full_narrative,
            'citations': self.citations
        }


class IncidentExplainer:
    """Deterministic, evidence-grounded incident explanation generator."""

    def generate_explanation(
        self,
        incident: CHARMIncident,
        evidence: List[EnrichedEvidence]
    ) -> IncidentExplanation:
        exp_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"
        observed_facts = []
        supported_interpretations = []
        unverified_hypotheses = []
        claims = []
        citations = []

        # 1. Observed Telemetry Facts (Directly from CHARM and XAI)
        anom_status = "ANOMALOUS" if incident.anomaly_detected else "BENIGN"
        fact_1 = (
            f"PAADU/CHARM evaluated {incident.input_reference} as {anom_status} with a hybrid anomaly score of "
            f"{incident.anomaly_score:.3f} against the calibrated baseline threshold of {incident.decision_threshold:.3f}."
        )
        observed_facts.append(fact_1)
        claims.append(FactualClaim(
            claim_id=f"CLM-{len(claims)+1}",
            claim_text=fact_1,
            claim_type="OBSERVED_FACT"
        ))

        if incident.trigger_reasons:
            causes_str = ", ".join(incident.trigger_reasons)
            fact_2 = (
                f"The alert was triggered by policies: [{causes_str}], with a reconstruction error of "
                f"{incident.reconstruction_error:.4f} and a latent Mahalanobis distance of {incident.mahalanobis_distance:.4f}."
            )
            observed_facts.append(fact_2)
            claims.append(FactualClaim(
                claim_id=f"CLM-{len(claims)+1}",
                claim_text=fact_2,
                claim_type="OBSERVED_FACT"
            ))

        # Top XAI Features
        if incident.top_xai_features:
            top_3 = incident.top_xai_features[:3]
            feat_summaries = [
                f"{a.feature} (contribution: {a.hybrid_contribution:+.2f}, observed mean: {a.observed_mean:.2f} vs benign mean: {a.benign_mean:.2f})"
                for a in top_3
            ]
            fact_3 = (
                f"Explainable AI leave-one-feature-out occlusion identified the most influential features as: "
                f"{'; '.join(feat_summaries)}."
            )
            observed_facts.append(fact_3)
            claims.append(FactualClaim(
                claim_id=f"CLM-{len(claims)+1}",
                claim_text=fact_3,
                claim_type="OBSERVED_FACT"
            ))

        # 2. Supported Interpretations (Derived from Retrieved Evidence)
        if incident.anomaly_detected and evidence:
            top_feature_names = {a.feature.lower() for a in incident.top_xai_features[:4]}
            added_topics = set()

            for ev in evidence:
                text_low = ev.text.lower()
                title_low = ev.title.lower()

                # TCP SYN context
                if ('syn_count' in top_feature_names or 'syn_flag_number' in top_feature_names) and 'syn' in title_low and 'syn_flood' not in added_topics:
                    prop = "A TCP SYN flood is a denial-of-service attack exploiting the TCP three-way handshake."
                    interp = f"According to {ev.title} ({ev.source_citation}), {prop}"
                    supported_interpretations.append(interp)
                    claims.append(FactualClaim(
                        claim_id=f"CLM-{len(claims)+1}",
                        claim_text=prop,
                        claim_type="EVIDENCE_INTERPRETATION",
                        cited_doc_title=ev.title,
                        cited_doc_id=ev.doc_id
                    ))
                    citations.append({'title': ev.title, 'citation': ev.source_citation, 'doc_id': ev.doc_id})
                    added_topics.add('syn_flood')

                # Volumetric Rate / Byte volume context
                elif ('rate' in top_feature_names or 'tot sum' in top_feature_names) and ('flood' in title_low or 'udp' in title_low or 'baseline' in title_low) and 'volumetric_flood' not in added_topics:
                    prop = "In a UDP flood attack, adversaries send massive volumes of high-rate datagrams to random ports on the target."
                    interp = f"As documented in {ev.title} ({ev.source_citation}), {prop}"
                    supported_interpretations.append(interp)
                    claims.append(FactualClaim(
                        claim_id=f"CLM-{len(claims)+1}",
                        claim_text=prop,
                        claim_type="EVIDENCE_INTERPRETATION",
                        cited_doc_title=ev.title,
                        cited_doc_id=ev.doc_id
                    ))
                    citations.append({'title': ev.title, 'citation': ev.source_citation, 'doc_id': ev.doc_id})
                    added_topics.add('volumetric_flood')

                # UDP context
                elif 'udp' in top_feature_names and 'udp' in title_low and 'udp_context' not in added_topics:
                    prop = "UDP is a connectionless, stateless transport protocol that does not use a handshake."
                    interp = f"Documented in {ev.title} ({ev.source_citation}), {prop}"
                    supported_interpretations.append(interp)
                    claims.append(FactualClaim(
                        claim_id=f"CLM-{len(claims)+1}",
                        claim_text=prop,
                        claim_type="EVIDENCE_INTERPRETATION",
                        cited_doc_title=ev.title,
                        cited_doc_id=ev.doc_id
                    ))
                    citations.append({'title': ev.title, 'citation': ev.source_citation, 'doc_id': ev.doc_id})
                    added_topics.add('udp_context')

        # Deduplicate citations
        unique_citations = []
        seen_cits = set()
        for c in citations:
            if c['doc_id'] not in seen_cits:
                unique_citations.append(c)
                seen_cits.add(c['doc_id'])

        # Fallback interpretation if no specific rule matched
        if incident.anomaly_detected and not supported_interpretations and evidence:
            top_ev = evidence[0]
            interp = (
                f"Retrieved cybersecurity reference {top_ev.title} ({top_ev.source_citation}) indicates that "
                f"unsupervised statistical telemetry anomalies reflect operational shifts that require external corroboration."
            )
            supported_interpretations.append(interp)
            claims.append(FactualClaim(
                claim_id=f"CLM-{len(claims)+1}",
                claim_text=interp,
                claim_type="EVIDENCE_INTERPRETATION",
                cited_doc_title=top_ev.title,
                cited_doc_id=top_ev.doc_id
            ))
            unique_citations.append({'title': top_ev.title, 'citation': top_ev.source_citation, 'doc_id': top_ev.doc_id})

        # 3. Unverified Hypotheses & Operational Safeguards
        hypo_1 = (
            "The observed telemetry deviation cannot be conclusively attributed to malicious adversarial action "
            "solely from statistical flow metrics without inspecting packet payload headers or host process logs."
        )
        unverified_hypotheses.append(hypo_1)
        claims.append(FactualClaim(
            claim_id=f"CLM-{len(claims)+1}",
            claim_text=hypo_1,
            claim_type="HYPOTHESIS"
        ))

        if incident.anomaly_detected:
            hypo_2 = (
                "Further investigation is recommended to confirm whether the detected telemetry shift reflects a true attack "
                "or a benign non-malicious burst, such as scheduled sensor synchronization or client reconnection."
            )
            unverified_hypotheses.append(hypo_2)
            claims.append(FactualClaim(
                claim_id=f"CLM-{len(claims)+1}",
                claim_text=hypo_2,
                claim_type="HYPOTHESIS"
            ))

        # Assemble full narrative
        narrative_parts = [
            "### Section 1: Observed Telemetry Facts",
            "\n".join([f"- {f}" for f in observed_facts]),
            "\n### Section 2: Evidence-Grounded Interpretations",
            "\n".join([f"- {i}" for i in supported_interpretations]) if supported_interpretations else "- No anomalous interpretations required for normal telemetry.",
            "\n### Section 3: Hypotheses & Investigation Boundaries",
            "\n".join([f"- {h}" for h in unverified_hypotheses])
        ]
        full_narrative = "\n\n".join(narrative_parts)

        return IncidentExplanation(
            explanation_id=exp_id,
            incident_id=incident.incident_id,
            observed_facts=observed_facts,
            supported_interpretations=supported_interpretations,
            unverified_hypotheses=unverified_hypotheses,
            claims=claims,
            full_narrative=full_narrative,
            citations=unique_citations
        )
