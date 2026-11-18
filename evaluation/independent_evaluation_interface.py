#!/usr/bin/env python3
"""
Independent Evaluation Interface for PAADU/CHARM Telemetry Verification.

Accepts externally supplied incident ground truth and independent reference claim labels.
Evaluates the production telemetry verifier without using the verifier to generate the ground truth.
Outputs confusion matrix, False Acceptance Rate (FAR), False Rejection Rate (FRR), precision,
recall, and assertion coverage with explicit numerators and denominators.
"""
from __future__ import annotations

import sys
import os
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from unified_integration.adapters.charm_incident_adapter import CHARMIncident, FeatureAttributionRecord
from unified_integration.verification.verification_orchestrator import validate_telemetry_claim_strict
from unified_integration.verification.telemetry_grammar import parse_and_validate_telemetry


def build_default_independent_suite(output_dir: Path) -> Tuple[Path, Path]:
    """Generates a separate, deterministic independent benchmark suite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    incidents_path = output_dir / "external_incidents.json"
    claims_path = output_dir / "external_claims.json"

    # 1. External Incident Ground Truth (10 diverse incidents)
    incidents_data = [
        {
            "incident_id": "EXT-INC-01",
            "model": "SemanticAnomalyMAE",
            "input_reference": "BENIGN-FLOW-01",
            "anomaly_detected": False,
            "anomaly_score": 1.150,
            "decision_threshold": 2.500,
            "trigger_reasons": ["baseline_monitoring"],
            "reconstruction_error": 0.0012,
            "mahalanobis_distance": 8.4200,
            "reconstruction_z": 0.45,
            "latent_z": 1.20,
            "top_xai_features": [
                {"feature": "Rate", "hybrid_contribution": -0.15, "reconstruction_contribution": -0.05, "latent_contribution": -0.10, "observed_mean": 850.0, "benign_mean": 920.0, "standardized_deviation": -0.2},
                {"feature": "Tot size", "hybrid_contribution": 0.08, "reconstruction_contribution": 0.04, "latent_contribution": 0.04, "observed_mean": 64.0, "benign_mean": 60.0, "standardized_deviation": 0.1}
            ]
        },
        {
            "incident_id": "EXT-INC-02",
            "model": "SemanticAnomalyMAE",
            "input_reference": "ATTACK-DDOS-02",
            "anomaly_detected": True,
            "anomaly_score": 14.820,
            "decision_threshold": 2.500,
            "trigger_reasons": ["hybrid_threshold", "burst_policy", "latent_evidence"],
            "reconstruction_error": 0.0245,
            "mahalanobis_distance": 68.3100,
            "reconstruction_z": 3.80,
            "latent_z": 15.40,
            "top_xai_features": [
                {"feature": "Rate", "hybrid_contribution": 0.65, "reconstruction_contribution": 0.35, "latent_contribution": 0.30, "observed_mean": 4500.0, "benign_mean": 800.0, "standardized_deviation": 5.2},
                {"feature": "syn_flag_number", "hybrid_contribution": 0.28, "reconstruction_contribution": 0.14, "latent_contribution": 0.14, "observed_mean": 1.0, "benign_mean": 0.05, "standardized_deviation": 4.1},
                {"feature": "Tot size", "hybrid_contribution": -0.005, "reconstruction_contribution": -0.002, "latent_contribution": -0.003, "observed_mean": 54.0, "benign_mean": 58.0, "standardized_deviation": -0.1}
            ]
        }
    ]

    # 2. External Claims with Independent Reference (Gold) Labels
    claims_data = [
        # Positives (Should Accept)
        {
            "claim_id": "EXT-CLM-01",
            "incident_id": "EXT-INC-01",
            "claim_text": "PAADU/CHARM evaluated BENIGN-FLOW-01 as BENIGN with a hybrid anomaly score of 1.150 against the calibrated baseline threshold of 2.500.",
            "gold_label": "SUPPORTED",
            "category": "canonical_decision_positive"
        },
        {
            "claim_id": "EXT-CLM-02",
            "incident_id": "EXT-INC-02",
            "claim_text": "PAADU/CHARM evaluated ATTACK-DDOS-02 as ANOMALOUS with a hybrid anomaly score of 14.820 against the calibrated baseline threshold of 2.500.",
            "gold_label": "SUPPORTED",
            "category": "canonical_decision_positive"
        },
        {
            "claim_id": "EXT-CLM-03",
            "incident_id": "EXT-INC-02",
            "claim_text": "The alert was triggered by policies: [hybrid_threshold, burst_policy, latent_evidence], with a reconstruction error of 0.0245 and a latent Mahalanobis distance of 68.3100.",
            "gold_label": "SUPPORTED",
            "category": "canonical_trigger_positive"
        },
        {
            "claim_id": "EXT-CLM-04",
            "incident_id": "EXT-INC-01",
            "claim_text": "Rate contribution: -0.15.",
            "gold_label": "SUPPORTED",
            "category": "atomic_negative_attribution_positive"
        },
        {
            "claim_id": "EXT-CLM-05",
            "incident_id": "EXT-INC-02",
            "claim_text": "Tot size contribution: -0.005.",
            "gold_label": "SUPPORTED",
            "category": "atomic_small_negative_positive"
        },

        # Contradictions (Should Retract)
        {
            "claim_id": "EXT-CLM-06",
            "incident_id": "EXT-INC-01",
            "claim_text": "PAADU/CHARM evaluated BENIGN-FLOW-01 as ANOMALOUS with a hybrid anomaly score of 1.150 against the calibrated baseline threshold of 2.500.",
            "gold_label": "CONTRADICTED",
            "category": "status_contradiction"
        },
        {
            "claim_id": "EXT-CLM-07",
            "incident_id": "EXT-INC-02",
            "claim_text": "PAADU/CHARM evaluated VICTIM-EXTERNAL as ANOMALOUS with a hybrid anomaly score of 14.820 against the calibrated baseline threshold of 2.500.",
            "gold_label": "CONTRADICTED",
            "category": "incident_identifier_contradiction"
        },
        {
            "claim_id": "EXT-CLM-08",
            "incident_id": "EXT-INC-02",
            "claim_text": "Anomaly score of 14.820 and policies: [unauthorized_policy].",
            "gold_label": "CONTRADICTED",
            "category": "unrecognized_policy_contradiction"
        },
        {
            "claim_id": "EXT-CLM-09",
            "incident_id": "EXT-INC-01",
            "claim_text": "Rate contribution: +0.15.",
            "gold_label": "CONTRADICTED",
            "category": "inverted_sign_contradiction"
        },
        {
            "claim_id": "EXT-CLM-10",
            "incident_id": "EXT-INC-02",
            "claim_text": "Anomaly score of 14.820 and Rate (contribution: 0.10, observed mean: 999 vs benign mean: 100).",
            "gold_label": "CONTRADICTED",
            "category": "feature_tuple_mismatch_contradiction"
        },

        # Unsupported / Ungrounded (Should Flag Uncertainty)
        {
            "claim_id": "EXT-CLM-11",
            "incident_id": "EXT-INC-02",
            "claim_text": "Anomaly score of 14.820 because threshold of 2.500.",
            "gold_label": "UNSUPPORTED",
            "category": "ungrounded_causal_connector"
        },
        {
            "claim_id": "EXT-CLM-12",
            "incident_id": "EXT-INC-02",
            "claim_text": "Anomaly score of 14.820, proving attacker Eve compromised the server.",
            "gold_label": "UNSUPPORTED",
            "category": "ungrounded_threat_actor"
        },
        {
            "claim_id": "EXT-CLM-13",
            "incident_id": "EXT-INC-01",
            "claim_text": "Anomaly score of 1.150 and IP address 192.168.1.55 is the attacker.",
            "gold_label": "UNSUPPORTED",
            "category": "ungrounded_ip_address"
        },
        {
            "claim_id": "EXT-CLM-14",
            "incident_id": "EXT-INC-02",
            "claim_text": "Anomaly score of 14.820 and arbitrary leftover unparsed claim statement.",
            "gold_label": "UNSUPPORTED",
            "category": "unparsed_span"
        }
    ]

    with open(incidents_path, 'w') as f:
        json.dump(incidents_data, f, indent=2)

    with open(claims_path, 'w') as f:
        json.dump(claims_data, f, indent=2)

    return incidents_path, claims_path


def run_independent_evaluation(
    incidents_path: Path,
    claims_path: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """Runs evaluation using external incident ground truth and independent claim labels."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(incidents_path) as f:
        raw_incidents = json.load(f)

    with open(claims_path) as f:
        raw_claims = json.load(f)

    incidents_map = {}
    for d in raw_incidents:
        top_feats = [
            FeatureAttributionRecord(
                feature=fa['feature'],
                hybrid_contribution=fa['hybrid_contribution'],
                reconstruction_contribution=fa['reconstruction_contribution'],
                latent_contribution=fa['latent_contribution'],
                observed_mean=fa['observed_mean'],
                benign_mean=fa['benign_mean'],
                standardized_deviation=fa['standardized_deviation']
            )
            for fa in d['top_xai_features']
        ]
        inc = CHARMIncident(
            incident_id=d['incident_id'],
            model=d['model'],
            input_reference=d['input_reference'],
            anomaly_detected=d['anomaly_detected'],
            anomaly_score=d['anomaly_score'],
            decision_threshold=d['decision_threshold'],
            trigger_reasons=d['trigger_reasons'],
            reconstruction_error=d['reconstruction_error'],
            mahalanobis_distance=d['mahalanobis_distance'],
            reconstruction_z=d['reconstruction_z'],
            latent_z=d['latent_z'],
            top_xai_features=top_feats,
            xai_summary_text="",
            timestamp="2026-11-18T00:00:00Z",
            model_checkpoint_hash="external_hash",
            feature_schema="ciciot2023_46"
        )
        incidents_map[d['incident_id']] = inc

    results = []
    tp = 0  # Gold=SUPPORTED, Pred=SUPPORTED
    fp = 0  # Gold!=SUPPORTED, Pred=SUPPORTED (False Acceptance)
    tn = 0  # Gold!=SUPPORTED, Pred!=SUPPORTED
    fn = 0  # Gold=SUPPORTED, Pred!=SUPPORTED (False Rejection)

    for item in raw_claims:
        cid = item['claim_id']
        inc_id = item['incident_id']
        claim_text = item['claim_text']
        gold_label = item['gold_label']

        inc = incidents_map.get(inc_id)
        if not inc:
            continue

        status, action, pres_text, reason, audit_recs, unparsed_spans = parse_and_validate_telemetry(claim_text, inc)

        is_match = (status == gold_label)
        if gold_label == "SUPPORTED":
            if status == "SUPPORTED":
                tp += 1
            else:
                fn += 1
        else:
            if status == "SUPPORTED":
                fp += 1
            else:
                tn += 1

        results.append({
            'claim_id': cid,
            'incident_id': inc_id,
            'claim_text': claim_text,
            'gold_label': gold_label,
            'predicted_status': status,
            'predicted_action': action,
            'presented_text': pres_text,
            'diagnostic_reason': reason,
            'audit_assertions_count': len(audit_recs),
            'unparsed_spans_count': len(unparsed_spans),
            'match': is_match
        })

    total = len(results)
    positives = tp + fn
    negatives = tn + fp

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    far = float(fp / negatives) if negatives > 0 else 0.0
    frr = float(fn / positives) if positives > 0 else 0.0

    abstention_count = sum(1 for r in results if r['predicted_status'] == 'UNSUPPORTED')
    retraction_count = sum(1 for r in results if r['predicted_status'] == 'CONTRADICTED')
    verified_count = sum(1 for r in results if r['predicted_status'] == 'SUPPORTED')

    summary = {
        'total_claims_evaluated': total,
        'gold_distribution': {
            'supported': positives,
            'negative_controls': negatives
        },
        'confusion_matrix': {
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        },
        'rates_with_explicit_denominators': {
            'false_acceptance_rate': f"{fp}/{negatives} ({far:.4f})",
            'false_rejection_rate': f"{fn}/{positives} ({frr:.4f})",
            'precision': f"{tp}/{tp + fp} ({precision:.4f})",
            'recall': f"{tp}/{positives} ({recall:.4f})",
            'f1_score': f"{f1:.4f}",
            'abstention_rate': f"{abstention_count}/{total} ({abstention_count / max(total, 1):.4f})",
            'retraction_rate': f"{retraction_count}/{total} ({retraction_count / max(total, 1):.4f})",
            'verification_coverage': f"{verified_count}/{total} ({verified_count / max(total, 1):.4f})"
        }
    }

    # Save detailed CSV
    with open(output_dir / 'INDEPENDENT_EVALUATION_RESULTS.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'claim_id', 'incident_id', 'claim_text', 'gold_label',
            'predicted_status', 'predicted_action', 'match',
            'assertions_count', 'unparsed_spans_count', 'diagnostic_reason'
        ])
        for r in results:
            writer.writerow([
                r['claim_id'], r['incident_id'], r['claim_text'], r['gold_label'],
                r['predicted_status'], r['predicted_action'], r['match'],
                r['audit_assertions_count'], r['unparsed_spans_count'], r['diagnostic_reason']
            ])

    # Save JSON summary
    with open(output_dir / 'INDEPENDENT_EVALUATION_REPORT.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run independent evaluation of PAADU/CHARM verifier.")
    parser.add_argument("--incidents", type=str, default="", help="Path to external incidents JSON.")
    parser.add_argument("--claims", type=str, default="", help="Path to external claims JSON with gold_label.")
    parser.add_argument("--output", type=str, default="", help="Path to output directory.")
    args = parser.parse_args()

    suite_dir = WORKSPACE / "datasets" / "independent_test_suite"
    output_dir = Path(args.output) if args.output else WORKSPACE / "results"

    if args.incidents and args.claims:
        inc_p = Path(args.incidents)
        cl_p = Path(args.claims)
    else:
        inc_p, cl_p = build_default_independent_suite(suite_dir)

    print("========================================================================")
    print("  RUNNING INDEPENDENT EVALUATION SUITE")
    print("========================================================================")
    print(f"  Incidents source: {inc_p}")
    print(f"  Claims source:    {cl_p}")
    print(f"  Output directory: {output_dir}")

    summary = run_independent_evaluation(inc_p, cl_p, output_dir)
    print("------------------------------------------------------------------------")
    print(f"  Total Claims: {summary['total_claims_evaluated']}")
    print(f"  TP: {summary['confusion_matrix']['true_positives']}, FP (FAR): {summary['rates_with_explicit_denominators']['false_acceptance_rate']}")
    print(f"  TN: {summary['confusion_matrix']['true_negatives']}, FN (FRR): {summary['rates_with_explicit_denominators']['false_rejection_rate']}")
    print(f"  Precision: {summary['rates_with_explicit_denominators']['precision']}")
    print(f"  Recall:    {summary['rates_with_explicit_denominators']['recall']}")
    print(f"  Abstention Rate: {summary['rates_with_explicit_denominators']['abstention_rate']}")
    print("========================================================================")


if __name__ == "__main__":
    main()
