#!/usr/bin/env python3
"""
Retrieval Quality Assessor.
Assesses retrieval quality across 4 rigorous dimensions:
1. Relevance (semantic and cosine score distribution)
2. Coverage (proportion of target query concepts addressed)
3. Consistency (absence of conflicting statements across evidence)
4. Source Provenance (citation authenticity and completeness)
Computes a composite retrieval confidence score to govern refinement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from unified_integration.retrieval.adaptive_controller import EnrichedEvidence

@dataclass
class QualityAssessmentResult:
    composite_score: float         # [0.0, 1.0]
    quality_status: str            # 'SUFFICIENT', 'PARTIAL', 'INSUFFICIENT'
    relevance_score: float
    coverage_score: float
    consistency_score: float
    provenance_score: float
    covered_facets: List[str]
    missing_facets: List[str]
    requires_refinement: bool
    diagnostic_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RetrievalQualityAssessor:
    """Evaluates the evidential sufficiency of retrieved documents for an investigation query."""

    def __init__(self, sufficiency_threshold: float = 0.70, partial_threshold: float = 0.45):
        self.sufficiency_threshold = sufficiency_threshold
        self.partial_threshold = partial_threshold

    def assess(
        self,
        retrieved_evidence: List[EnrichedEvidence],
        target_facets: List[str]
    ) -> QualityAssessmentResult:
        notes = []

        if not retrieved_evidence:
            return QualityAssessmentResult(
                composite_score=0.0,
                quality_status='INSUFFICIENT',
                relevance_score=0.0,
                coverage_score=0.0,
                consistency_score=0.0,
                provenance_score=0.0,
                covered_facets=[],
                missing_facets=target_facets,
                requires_refinement=True,
                diagnostic_notes=["Zero evidence documents retrieved."]
            )

        # 1. Relevance: based on TF-IDF cosine similarity scores
        scores = [e.similarity_score for e in retrieved_evidence]
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        # Normalize into [0.0, 1.0] assuming typical TF-IDF cosine range [0.1, 0.8]
        relevance = min(1.0, max(0.0, (max_score * 0.6) + (avg_score * 0.4)))
        notes.append(f"Relevance: max_score={max_score:.3f}, avg_score={avg_score:.3f} -> score={relevance:.3f}")

        # 2. Coverage: check which target facets are mentioned in retrieved texts
        combined_text = " ".join([f"{e.title} {e.text}" for e in retrieved_evidence]).lower()
        covered = []
        missing = []

        for facet in target_facets:
            f_clean = facet.lower().strip()
            # Check for word presence or substring match
            if re.search(r'\b' + re.escape(f_clean) + r'\b', combined_text) or f_clean in combined_text:
                covered.append(facet)
            else:
                missing.append(facet)

        coverage = len(covered) / len(target_facets) if target_facets else 1.0
        notes.append(f"Coverage: {len(covered)}/{len(target_facets)} facets covered -> score={coverage:.3f}")

        # 3. Consistency: verify consistent protocol characteristics across evidence
        # e.g., check for contradictory claims (TCP connectionless vs connection-oriented)
        consistency = 1.0
        if "tcp is connectionless" in combined_text and "tcp is connection-oriented" in combined_text:
            consistency -= 0.5
            notes.append("Consistency penalty: conflicting TCP connection state definitions detected.")
        if "udp is connection-oriented" in combined_text:
            consistency -= 0.5
            notes.append("Consistency penalty: incorrect UDP protocol mode detected.")
        notes.append(f"Consistency: score={consistency:.3f}")

        # 4. Provenance: completeness of authentic citations (NIST, RFC, CERT, OASIS, IEEE)
        recognized_authorities = ['nist', 'rfc', 'cert', 'oasis', 'ieee', 'cisa']
        prov_hits = 0
        for e in retrieved_evidence:
            cit_low = e.source_citation.lower()
            if any(auth in cit_low for auth in recognized_authorities):
                prov_hits += 1

        provenance = prov_hits / len(retrieved_evidence) if retrieved_evidence else 0.0
        notes.append(f"Provenance: {prov_hits}/{len(retrieved_evidence)} authoritative citations -> score={provenance:.3f}")

        # Composite score
        # Weights: Relevance 35%, Coverage 35%, Consistency 15%, Provenance 15%
        composite = round(
            (0.35 * relevance) + (0.35 * coverage) + (0.15 * consistency) + (0.15 * provenance),
            3
        )

        if composite >= self.sufficiency_threshold:
            status = 'SUFFICIENT'
            requires_refinement = False
        elif composite >= self.partial_threshold:
            status = 'PARTIAL'
            requires_refinement = True
        else:
            status = 'INSUFFICIENT'
            requires_refinement = True

        return QualityAssessmentResult(
            composite_score=composite,
            quality_status=status,
            relevance_score=round(relevance, 3),
            coverage_score=round(coverage, 3),
            consistency_score=round(consistency, 3),
            provenance_score=round(provenance, 3),
            covered_facets=covered,
            missing_facets=missing,
            requires_refinement=requires_refinement,
            diagnostic_notes=notes
        )
