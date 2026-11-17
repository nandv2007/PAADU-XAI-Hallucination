#!/usr/bin/env python3
"""
Incident-to-Query Builder.
Synthesizes targeted cybersecurity investigation queries from CHARMIncident records.
Maps observed telemetry deviations and XAI attributions to search terms using cautious,
scientifically defensible terminology.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from unified_integration.adapters.charm_incident_adapter import CHARMIncident

FEATURE_CONCEPT_MAP: Dict[str, Dict[str, Any]] = {
    'Rate': {
        'concept': 'transmission packet rate flood',
        'keywords': ['rate flood', 'packet rate spike', 'volumetric attack'],
        'distinguish': 'legitimate network burst traffic'
    },
    'Tot sum': {
        'concept': 'aggregate byte volume surge',
        'keywords': ['high byte volume', 'bandwidth saturation', 'DoS flood'],
        'distinguish': 'large bulk file transfer'
    },
    'AVG': {
        'concept': 'average packet size anomaly',
        'keywords': ['packet size distribution', 'payload size deviation'],
        'distinguish': 'heterogeneous protocol mix'
    },
    'Tot size': {
        'concept': 'total flow frame size expansion',
        'keywords': ['frame size surge', 'flow volume amplification'],
        'distinguish': 'batch IoT firmware update'
    },
    'syn_count': {
        'concept': 'elevated TCP SYN connection requests',
        'keywords': ['TCP SYN flood', 'half-open connection backlog', 'TCP three-way handshake exhaustion'],
        'distinguish': 'rapid client reconnect storms'
    },
    'syn_flag_number': {
        'concept': 'disproportionate SYN flag ratio',
        'keywords': ['SYN ratio anomaly', 'TCP SYN scan', 'handshake initiation flood'],
        'distinguish': 'high-frequency microservice initiation'
    },
    'ack_count': {
        'concept': 'ACK flag flood or reflection',
        'keywords': ['ACK flood', 'TCP state exhaustion', 'reflected ACK traffic'],
        'distinguish': 'asymmetric routing acknowledgement stream'
    },
    'rst_count': {
        'concept': 'abnormal TCP RST connection terminations',
        'keywords': ['RST tear-down attack', 'TCP connection reset attack', 'closed port scanning'],
        'distinguish': 'server restart or network partition'
    },
    'UDP': {
        'concept': 'UDP protocol volume surge',
        'keywords': ['UDP flood attack', 'stateless traffic burst', 'UDP amplification'],
        'distinguish': 'real-time streaming media or video ingest'
    },
    'DNS': {
        'concept': 'DNS query/response amplification',
        'keywords': ['DNS amplification', 'UDP port 53 query flood', 'open resolver abuse'],
        'distinguish': 'cache misses during heavy domain resolution'
    },
    'ARP': {
        'concept': 'ARP resolution broadcast flood',
        'keywords': ['ARP spoofing', 'cache poisoning', 'gratuitous ARP storm', 'man-in-the-middle'],
        'distinguish': 'DHCP address reassignment or subnet scan'
    },
    'IAT': {
        'concept': 'interarrival time compression or irregularity',
        'keywords': ['interarrival timing anomaly', 'microburst timing', 'slowloris low-rate attack'],
        'distinguish': 'jitter from wireless medium contention'
    },
    'flow_duration': {
        'concept': 'persistent connection or flow duration anomaly',
        'keywords': ['long-lived flow persistence', 'session holding', 'slowloris attack'],
        'distinguish': 'persistent MQTT or WebSocket telemetry sessions'
    },
    'HTTP': {
        'concept': 'HTTP request/response flood',
        'keywords': ['HTTP GET flood', 'web application Layer 7 flood', 'web server exhaustion'],
        'distinguish': 'web crawler or spike in legitimate HTTP traffic'
    },
    'HTTPS': {
        'concept': 'encrypted web traffic surge',
        'keywords': ['TLS session exhaustion', 'HTTPS flood', 'SSL renegotiation attack'],
        'distinguish': 'encrypted data sync or backup window'
    }
}

@dataclass
class InvestigationQuery:
    query_id: str
    incident_id: str
    query_text: str
    target_features: List[str]
    identified_concepts: List[str]
    investigation_facets: List[str]
    generated_at: str
    is_anomaly_query: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_id': self.query_id,
            'incident_id': self.incident_id,
            'query_text': self.query_text,
            'target_features': self.target_features,
            'identified_concepts': self.identified_concepts,
            'investigation_facets': self.investigation_facets,
            'generated_at': self.generated_at,
            'is_anomaly_query': self.is_anomaly_query
        }


class IncidentQueryBuilder:
    """Builds investigation queries strictly derived from CHARMIncident data and XAI rankings."""

    def __init__(self, max_features: int = 4):
        self.max_features = max_features

    def build_query(self, incident: CHARMIncident) -> InvestigationQuery:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        q_id = f"QRY-{incident.incident_id[-8:]}"

        if not incident.anomaly_detected:
            # Query for normal operating baseline
            target_feats = [attr.feature for attr in incident.top_xai_features[:self.max_features]] or ['Rate', 'Tot sum']
            feat_list_str = ", ".join(target_feats)
            query_text = (
                f"What are normal IoT network baseline telemetry characteristics for {feat_list_str}, "
                f"and what are the expected operating ranges in benign IoT environments?"
            )
            return InvestigationQuery(
                query_id=q_id,
                incident_id=incident.incident_id,
                query_text=query_text,
                target_features=target_feats,
                identified_concepts=['benign baseline telemetry', 'normal operating profile'],
                investigation_facets=['baseline behavior', 'operating ranges'],
                generated_at=now_iso,
                is_anomaly_query=False
            )

        # Anomaly detected: extract top features by absolute hybrid contribution
        sorted_attrs = sorted(incident.top_xai_features, key=lambda x: abs(x.hybrid_contribution), reverse=True)
        top_attrs = sorted_attrs[:self.max_features]
        target_features = [a.feature for a in top_attrs]

        concepts = []
        distinguish_terms = []
        facets = []

        for feat in target_features:
            mapping = FEATURE_CONCEPT_MAP.get(feat)
            if mapping:
                concepts.append(mapping['concept'])
                distinguish_terms.append(mapping['distinguish'])
                facets.extend(mapping['keywords'][:2])
            else:
                concepts.append(f"telemetry deviation in {feat}")
                facets.append(f"{feat} anomaly")

        feat_str = ", ".join(target_features)
        concept_str = " and ".join(concepts[:2])
        distinguish_str = " or ".join(set(distinguish_terms[:2])) or "benign operational bursts"

        # Cautious scientific framing
        query_text = (
            f"What network behaviors and attack patterns are associated with unusual deviations in {feat_str} "
            f"consistent with possible {concept_str}, and what evidence helps distinguish such anomalies from {distinguish_str}?"
        )

        return InvestigationQuery(
            query_id=q_id,
            incident_id=incident.incident_id,
            query_text=query_text,
            target_features=target_features,
            identified_concepts=concepts,
            investigation_facets=sorted(list(set(facets))),
            generated_at=now_iso,
            is_anomaly_query=True
        )
