#!/usr/bin/env python3
"""
Query Complexity Analyzer.
Analyzes cybersecurity investigation queries to dynamically determine
retrieval strategy, depth, and graph expansion parameters.
Reconstructs the conceptual query complexity component identified as missing in SVA-GRAG-Core.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

TECHNICAL_DOMAINS = {
    'transport': ['tcp', 'udp', 'handshake', 'three-way', 'connectionless', 'connection-oriented', 'stateless'],
    'volumetric': ['rate', 'volume', 'bandwidth', 'flood', 'saturation', 'dos', 'ddos', 'amplification'],
    'flags': ['syn', 'ack', 'rst', 'fin', 'psh', 'urg', 'flag'],
    'routing_layer2': ['arp', 'mac', 'ethernet', 'broadcast', 'cache poisoning', 'spoofing', 'mitm'],
    'application': ['dns', 'mqtt', 'http', 'https', 'port 53', 'port 1883', 'broker', 'publish', 'connect'],
    'temporal': ['iat', 'interarrival', 'duration', 'persistence', 'microburst', 'slowloris']
}

COMPARATIVE_MARKERS = [
    'distinguish', 'differentiate', 'compare', 'contrast', 'versus', 'vs',
    'difference between', 'separate from', 'distinguish between'
]

@dataclass(frozen=True)
class QueryComplexityResult:
    complexity_score: float  # [0.0, 1.0]
    category: str            # 'SIMPLE', 'MODERATE', 'COMPLEX'
    detected_domains: List[str]
    is_comparative: bool
    recommended_k: int
    recommended_hybrid_boost: float
    analysis_details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'complexity_score': self.complexity_score,
            'category': self.category,
            'detected_domains': self.detected_domains,
            'is_comparative': self.is_comparative,
            'recommended_k': self.recommended_k,
            'recommended_hybrid_boost': self.recommended_hybrid_boost,
            'analysis_details': self.analysis_details
        }


class QueryComplexityAnalyzer:
    """Evaluates lexical, semantic, and structural complexity of cybersecurity queries."""

    def analyze(self, query: str) -> QueryComplexityResult:
        q_low = query.lower()
        words = re.findall(r'\b\w+\b', q_low)
        word_count = len(words)

        # 1. Detect technical domains
        detected_domains = []
        domain_matches = {}
        for domain, kws in TECHNICAL_DOMAINS.items():
            matches = [kw for kw in kws if re.search(r'\b' + re.escape(kw) + r'\b', q_low)]
            if matches:
                detected_domains.append(domain)
                domain_matches[domain] = matches

        # 2. Check for comparative/multi-concept markers
        is_comparative = any(marker in q_low for marker in COMPARATIVE_MARKERS)

        # 3. Compute continuous complexity score [0.0, 1.0]
        # Base factor: length normalized up to 30 words (0.2 weight)
        length_factor = min(1.0, word_count / 30.0) * 0.20

        # Domain diversity: number of distinct network domains involved (0.5 weight)
        # 1 domain = 0.1, 2 domains = 0.3, 3+ domains = 0.5
        num_domains = len(detected_domains)
        domain_factor = min(0.50, num_domains * 0.16)

        # Comparative complexity: requiring differentiation of attack vs benign (0.3 weight)
        comparative_factor = 0.30 if is_comparative else 0.0

        raw_score = length_factor + domain_factor + comparative_factor
        complexity_score = round(min(1.0, max(0.1, raw_score)), 3)

        # Categorize
        if complexity_score < 0.35:
            category = 'SIMPLE'
            recommended_k = 2
            boost = 0.15
        elif complexity_score < 0.70:
            category = 'MODERATE'
            recommended_k = 4
            boost = 0.22
        else:
            category = 'COMPLEX'
            recommended_k = 6
            boost = 0.30

        return QueryComplexityResult(
            complexity_score=complexity_score,
            category=category,
            detected_domains=detected_domains,
            is_comparative=is_comparative,
            recommended_k=recommended_k,
            recommended_hybrid_boost=boost,
            analysis_details={
                'word_count': word_count,
                'domains_count': num_domains,
                'domain_matches': domain_matches,
                'length_factor': round(length_factor, 3),
                'domain_factor': round(domain_factor, 3),
                'comparative_factor': round(comparative_factor, 3)
            }
        )
