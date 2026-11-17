#!/usr/bin/env python3
"""
Independent Evaluation of SVA-GRAG-Core and NLI Hallucination Verification (TASK 9).
Tests query processing, vector retrieval, graph retrieval, hybrid fusion,
NLI classification, contradiction detection, insufficient-evidence handling,
and conservative correction.
Outputs:
- external_evaluation/integration_tests/SVA_GRAG_EVALUATION.json
- external_evaluation/metrics/NLI_VERIFICATION_METRICS.csv
"""
import os
import sys
import json
import time
import csv
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

WORKSPACE = Path(__file__).resolve().parents[2]
INTEG_PROJECT = WORKSPACE / 'integrated_project'
OUT_DIR = WORKSPACE / 'external_evaluation' / 'integration_tests'
METRICS_DIR = WORKSPACE / 'external_evaluation' / 'metrics'
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(INTEG_PROJECT / 'project'))
from integrated_pipeline import HybridRetriever, EvidenceVerifier, IntegratedSystem

def main():
    print("=== Testing SVA-GRAG-Core Reconstructed Components & NLI Verification ===")
    
    # 1. Corpus definition (Network protocols & Cybersecurity)
    corpus = [
        {
            'title': 'UDP Protocol',
            'text': 'UDP (User Datagram Protocol) is a connectionless transport protocol. It does not provide reliable delivery, packet reordering, or flow control.'
        },
        {
            'title': 'TCP Protocol',
            'text': 'TCP (Transmission Control Protocol) is connection-oriented and provides reliable, in-order byte stream delivery with congestion control and three-way handshake.'
        },
        {
            'title': 'DNS Protocol',
            'text': 'DNS (Domain Name System) translates human-readable domain names into IP addresses. It primarily uses UDP port 53 for standard queries.'
        },
        {
            'title': 'ARP Protocol',
            'text': 'ARP (Address Resolution Protocol) resolves IPv4 network layer addresses into physical MAC addresses on local Ethernet broadcast domains.'
        },
        {
            'title': 'MQTT Protocol',
            'text': 'MQTT is a lightweight publish-subscribe messaging protocol designed for resource-constrained IoT devices, running over TCP port 1883.'
        },
        {
            'title': 'SYN Flood Attack',
            'text': 'A SYN flood is a denial-of-service attack exploiting the TCP three-way handshake by leaving connections half-open, exhausting server backlog queues.'
        }
    ]
    
    # Check HybridRetriever construction
    t0 = time.time()
    retriever = HybridRetriever(corpus)
    retrieval_init_time = time.time() - t0
    
    graph_nodes = retriever.graph.number_of_nodes()
    graph_edges = retriever.graph.number_of_edges()
    print(f"HybridRetriever initialized in {retrieval_init_time:.4f}s. Graph: {graph_nodes} nodes, {graph_edges} edges.")
    
    # Verify graph links
    edges_found = list(retriever.graph.edges())
    print(f"Extracted document graph edges: {edges_found}")
    
    # Test hybrid boost vs pure vector
    query = "IoT publish subscribe protocol"
    pure_vector = retriever.retrieve(query, k=3, hybrid=False)
    hybrid_vector = retriever.retrieve(query, k=3, hybrid=True)
    print(f"Query: '{query}'")
    print(f"  Pure vector top hit: {pure_vector[0].title} (score={pure_vector[0].score:.4f})")
    print(f"  Hybrid top hit: {hybrid_vector[0].title} (score={hybrid_vector[0].score:.4f})")
    
    # Check SVA-GRAG-Core architectural components status
    component_audit = {
        'query_processing': {'implemented': True, 'details': 'TfidfVectorizer with unigram+bigram and English stopwords'},
        'query_complexity_analysis': {'implemented': False, 'details': 'Not present in reconstructed code; all queries receive uniform retrieval routing'},
        'adaptive_retrieval_control': {'implemented': False, 'details': 'Fixed parameter retrieval (k=3, hybrid=True); dynamic depth control absent'},
        'vector_retrieval': {'implemented': True, 'details': 'Cosine similarity over TF-IDF sparse matrix'},
        'graph_retrieval': {'implemented': True, 'details': 'NetworkX undirected co-mention graph over document titles'},
        'evidence_fusion': {'implemented': True, 'details': 'Additive boost (score + 0.22 * anchor_score) for top-2 anchor neighbors'},
        'retrieval_quality_prediction': {'implemented': False, 'details': 'Not present in reconstructed code; no confidence or calibration estimator'},
        'evidence_selection': {'implemented': True, 'details': 'Score descending sort with positive score filter and top-k slicing'},
        'answer_generation': {'implemented': False, 'details': 'Requires externally supplied candidate_answer; internal LLM generation absent'},
        'claim_segmentation': {'implemented': True, 'details': 'Regex punctuation sentence boundary splitting for evidence candidate search'},
        'nli_verification': {'implemented': True, 'details': 'Hugging Face cross-encoder/nli-MiniLM2-L6-H768 sequence classification'},
        'contradiction_detection': {'implemented': True, 'details': 'Thresholded decision rule (label CONTRADICTION with confidence >= 0.9)'},
        'insufficient_evidence_handling': {'implemented': True, 'details': 'Fallback status INSUFFICIENT_EVIDENCE when confidence thresholds not met'},
        'automatic_correction': {'implemented': True, 'status': 'CONSERVATIVE_IDENTITY', 'details': 'Preserves candidate answer; status UNCHANGED_NO_VERIFIED_REPLACEMENT'},
        'reverification': {'implemented': False, 'details': 'No post-correction feedback loop; output returned after single pass'}
    }
    
    # 2. Test NLI Verification across a rigorous test benchmark
    verifier = EvidenceVerifier(load_model=True)
    
    test_cases = [
        # (Query, Candidate Answer, Ground Truth Label)
        (
            "What delivery guarantees does UDP provide?",
            "UDP is connectionless and does not provide reliable delivery.",
            "SUPPORTED"
        ),
        (
            "Does TCP guarantee ordered delivery?",
            "TCP is connection-oriented and provides reliable in-order byte stream delivery.",
            "SUPPORTED"
        ),
        (
            "Which port does DNS primarily use?",
            "DNS primarily uses UDP port 53 for standard queries.",
            "SUPPORTED"
        ),
        (
            "What transport protocol does MQTT use?",
            "MQTT runs over TCP port 1883 for IoT devices.",
            "SUPPORTED"
        ),
        (
            "Does UDP guarantee ordered delivery?",
            "UDP guarantees strictly reliable and in-order packet delivery.",
            "CONTRADICTED"
        ),
        (
            "Is TCP a connectionless protocol?",
            "TCP is a connectionless protocol that does not use three-way handshakes.",
            "CONTRADICTED"
        ),
        (
            "Does DNS use TCP port 80?",
            "DNS resolves domain names using HTTP over TCP port 80.",
            "CONTRADICTED"
        ),
        (
            "What encryption does quantum computing use?",
            "Post-quantum cryptography uses lattice-based Kyber algorithms for key exchange.",
            "INSUFFICIENT_EVIDENCE"
        ),
        (
            "How does BGP routing function?",
            "BGP uses path-vector protocol with autonomous system numbers.",
            "INSUFFICIENT_EVIDENCE"
        ),
        (
            "What causes ARP spoofing?",
            "ARP spoofing occurs when an attacker broadcasts forged ARP reply messages on the LAN.",
            "INSUFFICIENT_EVIDENCE"
        )
    ]
    
    y_true = []
    y_pred = []
    detailed_bench_results = []
    
    print("\nRunning Ground-Truth NLI Hallucination Verification Benchmark...")
    for idx, (q, claim, true_label) in enumerate(test_cases):
        t_start = time.time()
        retrieved_docs = retriever.retrieve(q, k=3, hybrid=True)
        verif = verifier.verify(claim, retrieved_docs)
        pred_label = verif['label']
        latency = time.time() - t_start
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        match = (true_label == pred_label)
        print(f"[{idx+1}/{len(test_cases)}] True: {true_label:<21} | Pred: {pred_label:<21} | Match: {match} ({latency:.3f}s)")
        if verif['evidence']:
            ev = verif['evidence']
            print(f"    Evidence: [{ev['title']}] '{ev['sentence']}' (NLI: {ev['label']}, conf={ev['confidence']:.4f})")
            
        detailed_bench_results.append({
            'test_id': idx + 1,
            'query': q,
            'claim': claim,
            'true_label': true_label,
            'predicted_label': pred_label,
            'match': match,
            'latency_sec': round(latency, 4),
            'evidence_doc': verif['evidence']['title'] if verif['evidence'] else None,
            'nli_raw_label': verif['evidence']['label'] if verif['evidence'] else None,
            'nli_confidence': verif['evidence']['confidence'] if verif['evidence'] else None
        })
        
    # Calculate metrics
    classes = ['SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT_EVIDENCE']
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    print(f"\nBenchmark Accuracy: {acc * 100:.2f}%")
    for i, cls_name in enumerate(classes):
        print(f"  Class {cls_name:<21}: Precision={prec[i]:.4f}, Recall={rec[i]:.4f}, F1={f1[i]:.4f}, Support={supp[i]}")
        
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print(f"{'':<22} {'SUPPORTED':<12} {'CONTRADICTED':<14} {'INSUFFICIENT':<12}")
    for i, row in enumerate(cm):
        print(f"{classes[i]:<22} {row[0]:<12} {row[1]:<14} {row[2]:<12}")

    # Write NLI metrics CSV
    nli_csv_path = METRICS_DIR / 'NLI_VERIFICATION_METRICS.csv'
    with open(nli_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['verification_class', 'precision', 'recall', 'f1_score', 'support'])
        for i, cls_name in enumerate(classes):
            writer.writerow([cls_name, round(prec[i], 4), round(rec[i], 4), round(f1[i], 4), int(supp[i])])
        macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        writer.writerow(['MACRO_AVERAGE', round(macro_p, 4), round(macro_r, 4), round(macro_f, 4), len(y_true)])
        writer.writerow(['ACCURACY', round(acc, 4), round(acc, 4), round(acc, 4), len(y_true)])
    print(f"Wrote NLI metrics to {nli_csv_path}")
    
    # Save SVA_GRAG_EVALUATION.json
    out_json = {
        'retriever': {
            'corpus_size': len(corpus),
            'graph_nodes': graph_nodes,
            'graph_edges': graph_edges,
            'hybrid_edge_boost': 0.22
        },
        'components_implementation_audit': component_audit,
        'hallucination_verification_benchmark': {
            'samples_tested': len(test_cases),
            'accuracy': round(acc, 4),
            'macro_f1': round(macro_f, 4),
            'confusion_matrix': {
                'labels': classes,
                'matrix': cm.tolist()
            },
            'test_results': detailed_bench_results
        }
    }
    with open(OUT_DIR / 'SVA_GRAG_EVALUATION.json', 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f"Wrote SVA-GRAG evaluation report to {OUT_DIR / 'SVA_GRAG_EVALUATION.json'}")

if __name__ == '__main__':
    main()
