#!/usr/bin/env python3
"""
Refinement Controller.
Coordinates bounded iterative retrieval refinement when evidence quality is PARTIAL or INSUFFICIENT.
Formulates targeted supplementary queries for unaddressed concept facets and deduplicates evidence.
Terminates strictly within a bounded iteration limit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from unified_integration.retrieval.adaptive_controller import (
    AdaptiveRetrievalController, RetrievalResult, EnrichedEvidence
)
from unified_integration.retrieval.quality_assessor import (
    RetrievalQualityAssessor, QualityAssessmentResult
)

@dataclass
class RefinementIterationRecord:
    iteration: int
    reformulated_query: str
    target_missing_facets: List[str]
    newly_added_docs: List[str]
    quality_before: float
    quality_after: float
    status_after: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RefinementResult:
    final_evidence: List[EnrichedEvidence]
    final_quality: QualityAssessmentResult
    iterations_executed: int
    max_iterations_allowed: int
    refinement_history: List[RefinementIterationRecord]
    termination_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'final_evidence': [e.to_dict() for e in self.final_evidence],
            'final_quality': self.final_quality.to_dict(),
            'iterations_executed': self.iterations_executed,
            'max_iterations_allowed': self.max_iterations_allowed,
            'refinement_history': [r.to_dict() for r in self.refinement_history],
            'termination_reason': self.termination_reason
        }


class RefinementController:
    """Manages iterative, bounded evidence refinement."""

    def __init__(
        self,
        retrieval_controller: AdaptiveRetrievalController,
        quality_assessor: Optional[RetrievalQualityAssessor] = None,
        max_iterations: int = 2
    ):
        self.retrieval_controller = retrieval_controller
        self.quality_assessor = quality_assessor or RetrievalQualityAssessor()
        self.max_iterations = max(1, min(4, max_iterations))

    def refine_evidence(
        self,
        initial_retrieval: RetrievalResult,
        target_facets: List[str]
    ) -> RefinementResult:
        current_evidence = list(initial_retrieval.retrieved_evidence)
        initial_assessment = self.quality_assessor.assess(current_evidence, target_facets)
        
        history: List[RefinementIterationRecord] = []
        current_quality = initial_assessment

        if not current_quality.requires_refinement:
            return RefinementResult(
                final_evidence=current_evidence,
                final_quality=current_quality,
                iterations_executed=0,
                max_iterations_allowed=self.max_iterations,
                refinement_history=[],
                termination_reason="Initial evidence meets sufficiency threshold (no refinement required)."
            )

        seen_titles = {e.title for e in current_evidence}
        iteration = 0

        while iteration < self.max_iterations and current_quality.requires_refinement:
            iteration += 1
            missing = current_quality.missing_facets
            if not missing:
                break

            # Formulate targeted sub-query targeting the missing facets
            focus_terms = " ".join(missing[:3])
            sub_query = f"Cybersecurity technical protocol details regarding {focus_terms}"
            
            # Retrieve supplementary documents
            sub_res = self.retrieval_controller.retrieve(sub_query, override_k=3)
            newly_added = []

            for new_ev in sub_res.retrieved_evidence:
                if new_ev.title not in seen_titles:
                    current_evidence.append(new_ev)
                    seen_titles.add(new_ev.title)
                    newly_added.append(new_ev.title)

            # Re-assess quality with expanded evidence pool
            new_quality = self.quality_assessor.assess(current_evidence, target_facets)
            
            history.append(RefinementIterationRecord(
                iteration=iteration,
                reformulated_query=sub_query,
                target_missing_facets=missing,
                newly_added_docs=newly_added,
                quality_before=current_quality.composite_score,
                quality_after=new_quality.composite_score,
                status_after=new_quality.quality_status
            ))

            current_quality = new_quality

        if not current_quality.requires_refinement:
            term_reason = f"Sufficiency achieved after {iteration} refinement iteration(s)."
        elif iteration >= self.max_iterations:
            term_reason = f"Bounded limit reached ({self.max_iterations} iterations). Proceeding with best-effort evidence."
        else:
            term_reason = "No further missing facets identified."

        return RefinementResult(
            final_evidence=current_evidence,
            final_quality=current_quality,
            iterations_executed=iteration,
            max_iterations_allowed=self.max_iterations,
            refinement_history=history,
            termination_reason=term_reason
        )
