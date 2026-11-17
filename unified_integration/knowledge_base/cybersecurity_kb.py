#!/usr/bin/env python3
"""
Authoritative Cybersecurity Knowledge Base.
Provides structured, cited, evidence-grounded cybersecurity documents
for retrieval and fact verification.
Built independently from evaluation labels and model predictions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass(frozen=True)
class KnowledgeDocument:
    doc_id: str
    title: str
    source_citation: str
    content: str
    topics: List[str]
    provenance: str

    def to_retriever_dict(self) -> Dict[str, str]:
        """Returns standard format required by HybridRetriever."""
        return {
            'title': self.title,
            'text': self.content
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CYBERSECURITY_DOCUMENTS: List[KnowledgeDocument] = [
    KnowledgeDocument(
        doc_id="DOC-KB-001",
        title="Normal IoT Baseline Telemetry and Rate Patterns",
        source_citation="NIST SP 800-213: IoT Device Cybersecurity Guidance for the Federal Government (2021)",
        content=(
            "In benign IoT network environments, connected sensor telemetry exhibits periodic heartbeat signals, "
            "deterministic polling intervals, and bounded transmission rates. Occasional burst traffic and rate increases "
            "occur legitimately during firmware updates, scheduled cloud state synchronization, or temporary network reconnects. "
            "Benign telemetry flows generally maintain consistent TCP flag ratios and stable interarrival times, "
            "whereas attack bursts exhibit severe rate anomalies coupled with abnormal flag distributions."
        ),
        topics=["baseline telemetry", "IoT normal behavior", "rate spikes", "benign bursts"],
        provenance="NIST Special Publication 800-213"
    ),
    KnowledgeDocument(
        doc_id="DOC-KB-002",
        title="TCP SYN Flood and Handshake Exhaustion",
        source_citation="CERT Advisory CA-1996-21 / RFC 793: Transmission Control Protocol Specification (1981)",
        content=(
            "A TCP SYN flood is a classic denial-of-service attack exploiting the TCP three-way handshake. "
            "The attacker sends a rapid stream of TCP packets with the SYN flag set, initiating connections without "
            "sending the completing ACK packet. This leaves half-open connection entries in the target system's backlog queue, "
            "eventually exhausting connection table memory and denying access to legitimate clients. "
            "Primary network telemetry indicators include a surging syn_count, disproportionate syn_flag_number, and "
            "drastically reduced ACK completion ratios."
        ),
        topics=["TCP SYN flood", "syn_count", "three-way handshake", "half-open connections", "denial-of-service"],
        provenance="CERT Division & IETF RFC 793"
    ),
    KnowledgeDocument(
        doc_id="DOC-KB-003",
        title="UDP Flood and Reflection Amplification Attacks",
        source_citation="CISA Alert TA14-017A: UDP-Based Amplification Attacks / RFC 768 (1980)",
        content=(
            "UDP is a connectionless, stateless transport protocol that does not use a handshake. "
            "In a UDP flood attack, adversaries send massive volumes of high-rate UDP datagrams to random ports on the target. "
            "Because UDP does not verify sender addresses, attackers routinely spoof source IPs to reflect amplified responses "
            "from public resolvers onto victim networks. In network telemetry, UDP floods present as extreme spikes in UDP packet "
            "counts, massive aggregate byte volume (Tot sum), elevated transmission Rate, and collapsed packet interarrival times (IAT)."
        ),
        topics=["UDP flood", "stateless protocols", "amplification", "rate surges", "Tot sum"],
        provenance="US Cybersecurity and Infrastructure Security Agency (CISA) Alert TA14-017A"
    ),
    KnowledgeDocument(
        doc_id="DOC-KB-004",
        title="DNS Amplification and Port 53 Protocol Floods",
        source_citation="NIST SP 800-81-2: Secure Domain Name System (DNS) Deployment Guide (2013) / RFC 1035",
        content=(
            "DNS amplification is a specific form of distributed denial-of-service attack leveraging open recursive DNS resolvers. "
            "Adversaries transmit small DNS lookup queries (such as ANY requests with EDNS0 extensions) using the victim's spoofed source IP. "
            "The resolver returns responses that can be 50 to 70 times larger than the request payload. Telemetry features typically "
            "show concurrent surges in UDP traffic, elevated DNS protocol flags, high average packet sizes (AVG), and significant byte volumes."
        ),
        topics=["DNS amplification", "UDP port 53", "amplification factor", "volumetric DDoS"],
        provenance="NIST Special Publication 800-81-2"
    ),
    KnowledgeDocument(
        doc_id="DOC-KB-005",
        title="ARP Spoofing and Layer 2 Cache Poisoning",
        source_citation="IETF RFC 826: An Ethernet Address Resolution Protocol (1982) / NIST SP 800-115",
        content=(
            "Address Resolution Protocol (ARP) translates 32-bit IPv4 addresses to 48-bit Ethernet MAC addresses. "
            "Because ARP lacks authentication, any host can send unsolicited, forged gratuitous ARP replies across a local broadcast domain. "
            "In ARP spoofing or ARP cache poisoning attacks, the attacker binds their MAC address to the IP address of an authorized default gateway, "
            "intercepting or modifying traffic in a man-in-the-middle posture. Network telemetry manifests as an anomalous burst of ARP broadcast "
            "frames with conflicting IP-to-MAC associations."
        ),
        topics=["ARP spoofing", "cache poisoning", "gratuitous ARP", "layer 2 attacks"],
        provenance="IETF RFC 826 & NIST SP 800-115"
    ),
    KnowledgeDocument(
        doc_id="DOC-KB-006",
        title="MQTT Protocol Security and IoT Publish/Connect Floods",
        source_citation="OASIS Standard: MQTT Version 5.0 Specification (2019)",
        content=(
            "MQTT (Message Queuing Telemetry Transport) is an asynchronous publish-subscribe messaging protocol designed for constrained IoT sensors, "
            "running over TCP port 1883 or TLS port 8883. Common IoT DoS attacks targeting MQTT brokers include Connect Floods and Publish Floods, "
            "where attackers rapidly open client sessions or broadcast excessive high-frequency payloads across popular topics. "
            "These attacks cause memory exhaustion and queue starvation on the central broker, manifesting as surges in TCP flow rates and payload counts."
        ),
        topics=["MQTT", "IoT security", "publish flood", "connect flood", "TCP port 1883"],
        provenance="OASIS Standards Consortium MQTT v5.0"
    ),
    KnowledgeDocument(
        doc_id="DOC-KB-007",
        title="Network Reconnaissance, Port Scanning, and Host Probing",
        source_citation="NIST SP 800-94: Guide to Intrusion Detection and Prevention Systems (IDPS) (2012)",
        content=(
            "Reconnaissance attacks gather intelligence regarding active IP addresses, open ports, and operating system versions before launching exploits. "
            "Techniques include TCP SYN stealth scans, full connect scans, UDP port probing, and ICMP ping sweeps. "
            "In flow telemetry, reconnaissance produces a large number of very short-lived connections (low flow_duration), "
            "elevated syn_count or rst_count, high ratio of RST packets returned by closed ports, and minimal payload size (low Tot sum and AVG)."
        ),
        topics=["reconnaissance", "port scan", "ping sweep", "RST flags", "flow_duration"],
        provenance="NIST Special Publication 800-94"
    ),
    KnowledgeDocument(
        doc_id="DOC-KB-008",
        title="Limitations of Anomaly-Based Network Intrusion Detection",
        source_citation="Sommer & Paxson: Outside the Closed World: On Using Machine Learning for Network Intrusion Detection (IEEE S&P 2010)",
        content=(
            "Machine learning and statistical anomaly detection models identify statistical deviations from a modeled benign traffic baseline. "
            "However, an anomaly score does not equal an attack classification. Unsupervised models (such as autoencoders and Mahalanobis scorers) "
            "frequently trigger false alarms due to non-malicious shifts in network behavior, including backup syncs, microservice initialization, "
            "or changes in protocol mix. Scientific validation requires that model attributions be treated as hypotheses requiring corroboration "
            "from endpoint telemetry, packet headers, or independent logs before confirming malicious activity."
        ),
        topics=["anomaly detection limitations", "false positives", "drift", "statistical vs causal attribution"],
        provenance="IEEE Symposium on Security and Privacy (S&P 2010)"
    )
]

class CybersecurityKnowledgeBase:
    """Manages cybersecurity knowledge documents for SVA-GRAG retrieval."""

    def __init__(self, documents: Optional[List[KnowledgeDocument]] = None):
        self.documents = documents or CYBERSECURITY_DOCUMENTS
        self._doc_map = {doc.title: doc for doc in self.documents}
        self._id_map = {doc.doc_id: doc for doc in self.documents}

    def get_retriever_corpus(self) -> List[Dict[str, str]]:
        """Returns the document format consumed by HybridRetriever."""
        return [doc.to_retriever_dict() for doc in self.documents]

    def get_by_title(self, title: str) -> Optional[KnowledgeDocument]:
        return self._doc_map.get(title)

    def get_by_id(self, doc_id: str) -> Optional[KnowledgeDocument]:
        return self._id_map.get(doc_id)

    def search_by_keywords(self, keywords: List[str]) -> List[KnowledgeDocument]:
        results = []
        for doc in self.documents:
            match_score = sum(1 for kw in keywords if kw.lower() in doc.content.lower() or kw.lower() in doc.title.lower())
            if match_score > 0:
                results.append((match_score, doc))
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results]
