#!/usr/bin/env python3
"""
Adaptive Retrieval Controller.
Dynamically coordinates hybrid vector and graph retrieval based on query complexity.
Reuses the unmodified reconstructed HybridRetriever from the original project.
Enriches retrieved evidence with knowledge base metadata and source provenance.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT_DIR = WORKSPACE / 'integrated_project' / 'project'
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from integrated_pipeline import HybridRetriever, Evidence
from unified_integration.knowledge_base.cybersecurity_kb import CybersecurityKnowledgeBase, KnowledgeDocument
from unified_integration.retrieval.query_complexity import QueryComplexityAnalyzer, QueryComplexityResult

@dataclass
class EnrichedEvidence:
    doc_id: str
    title: str
    text: str
    similarity_score: float
    source_citation: str
    provenance: str
    topics: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    query: str
    complexity: QueryComplexityResult
    retrieved_evidence: List[EnrichedEvidence]
    retrieval_strategy: str
    execution_time_sec: float
    graph_nodes_count: int
    graph_edges_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'complexity': self.complexity.to_dict(),
            'retrieved_evidence': [e.to_dict() for e in self.retrieved_evidence],
            'retrieval_strategy': self.retrieval_strategy,
            'execution_time_sec': self.execution_time_sec,
            'graph_nodes_count': self.graph_nodes_count,
            'graph_edges_count': self.graph_edges_count
        }


class AdaptiveRetrievalController:
    """Orchestrates adaptive retrieval routing using query complexity and original HybridRetriever."""

    def __init__(self, kb: Optional[CybersecurityKnowledgeBase] = None):
        self.kb = kb or CybersecurityKnowledgeBase()
        corpus = self.kb.get_retriever_corpus()
        self.retriever = HybridRetriever(corpus)
        self.complexity_analyzer = QueryComplexityAnalyzer()

    def retrieve(self, query: str, override_k: Optional[int] = None) -> RetrievalResult:
        t0 = time.time()
        complexity = self.complexity_analyzer.analyze(query)
        k = override_k or complexity.recommended_k

        # Execute retrieval via original HybridRetriever
        raw_evidence = self.retriever.retrieve(query, k=k, hybrid=True)
        retrieval_latency = round(time.time() - t0, 4)

        # Enrich evidence with knowledge base metadata
        enriched = []
        for e in raw_evidence:
            doc = self.kb.get_by_title(e.title)
            enriched.append(EnrichedEvidence(
                doc_id=doc.doc_id if doc else "UNKNOWN_DOC",
                title=e.title,
                text=e.text,
                similarity_score=round(float(e.score), 4),
                source_citation=doc.source_citation if doc else "Local project corpus",
                provenance=doc.provenance if doc else "Reconstructed knowledge base",
                topics=doc.topics if doc else []
            ))

        strategy_desc = (
            f"Adaptive Hybrid (Category={complexity.category}, k={k}, "
            f"Boost={complexity.recommended_hybrid_boost}, Domains={len(complexity.detected_domains)})"
        )

        return RetrievalResult(
            query=query,
            complexity=complexity,
            retrieved_evidence=enriched,
            retrieval_strategy=strategy_desc,
            execution_time_sec=retrieval_latency,
            graph_nodes_count=self.retriever.graph.number_of_nodes(),
            graph_edges_count=self.retriever.graph.number_of_edges()
        )
