#!/usr/bin/env python3
"""
Held-Out Adversarial Evaluation Suite for PAADU/CHARM Factual Verifier.
Generates and evaluates a deterministic test set of >=100 distinct compound claims across:
1. Valid telemetry + false causal explanation
2. Valid telemetry + false attacker identity
3. Valid telemetry + contradictory status
4. Valid telemetry + incorrect feature values
5. Multiple repeated numeric assertions
6. Negative attribution values (valid and contradictory)
7. Unknown telemetry fields
8. Unsupported but plausible security explanations
9. Punctuation and syntax variations

Reports:
- False Acceptance Count & Rate with explicit denominators
- False Rejection Count & Rate with explicit denominators
- Precision, Recall, and Abstention Rate with explicit denominators
"""
from __future__ import annotations

import sys
import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from unified_integration.adapters.charm_incident_adapter import CHARMIncident, FeatureAttributionRecord
from unified_integration.verification.verification_orchestrator import validate_telemetry_claim_strict


def build_reference_incident() -> CHARMIncident:
    """Builds a calibrated benchmark incident for adversarial testing."""
    return CHARMIncident(
        incident_id="INC-HELD-OUT-BENCHMARK",
        model="SemanticAnomalyMAE",
        input_reference="BENIGN-BLK-01",
        anomaly_detected=True,
        anomaly_score=10.549,
        decision_threshold=2.500,
        trigger_reasons=["hybrid_threshold", "latent_evidence", "burst"],
        reconstruction_error=0.0071,
        mahalanobis_distance=45.9105,
        reconstruction_z=1.5,
        latent_z=12.0,
        top_xai_features=[
            FeatureAttributionRecord(feature="Rate", hybrid_contribution=-0.39, reconstruction_contribution=0.1, latent_contribution=-0.49, observed_mean=1221.61, benign_mean=1929.59, standardized_deviation=-1.5),
            FeatureAttributionRecord(feature="rst_flag_number", hybrid_contribution=0.22, reconstruction_contribution=0.12, latent_contribution=0.10, observed_mean=0.00, benign_mean=0.00, standardized_deviation=0.0),
            FeatureAttributionRecord(feature="cwr_flag_number", hybrid_contribution=0.16, reconstruction_contribution=0.06, latent_contribution=0.10, observed_mean=0.03, benign_mean=0.00, standardized_deviation=1.2),
            FeatureAttributionRecord(feature="Tot size", hybrid_contribution=-0.05, reconstruction_contribution=-0.02, latent_contribution=-0.03, observed_mean=54.0, benign_mean=60.0, standardized_deviation=-0.5),
            FeatureAttributionRecord(feature="UDP", hybrid_contribution=0.0, reconstruction_contribution=0.0, latent_contribution=0.0, observed_mean=0.0, benign_mean=0.0, standardized_deviation=0.0)
        ],
        xai_summary_text="",
        timestamp="2026-11-17T12:00:00Z",
        model_checkpoint_hash="65cfd22f671c6670e1ca29ebf0ee8fa8f9c968f3db47f523c907b3992b1a82bc",
        feature_schema="ciciot2023_46"
    )


def generate_held_out_dataset(inc: CHARMIncident) -> List[Dict[str, Any]]:
    """Generates >=100 deterministic test cases across all required categories."""
    cases: List[Dict[str, Any]] = []

    s = f"{inc.anomaly_score:.3f}"
    t = f"{inc.decision_threshold:.3f}"
    r = f"{inc.reconstruction_error:.4f}"
    m = f"{inc.mahalanobis_distance:.4f}"

    # Category 1: Valid telemetry + false causal explanation (15 cases, SHOULD_REJECT)
    causal_phrases = [
        "proving Alice compromised the device.",
        "because Alice compromised the device.",
        "causing the IoT camera to reboot.",
        "due to memory corruption in firmware.",
        "proving the botnet infection succeeded.",
        "suggesting kernel panic occurred.",
        "indicating buffer overflow execution.",
        "proving rootkit installation.",
        "because firmware integrity check failed.",
        "resulting in denial of network service.",
        "proving attacker gained root privileges.",
        "attributed to power supply fluctuation.",
        "indicating exfiltration of credentials.",
        "because switch 4 dropped all packets.",
        "proving unauthenticated remote code execution."
    ]
    for i, phrase in enumerate(causal_phrases, 1):
        cases.append({
            'test_id': f'ADV-CAUSAL-{i:03d}',
            'category': 'valid_telemetry_false_causal',
            'claim_text': f"Anomaly score of {s}, {phrase}",
            'ground_truth_label': 'SHOULD_REJECT',
            'expected_verdict': 'UNSUPPORTED'
        })

    # Category 2: Valid telemetry + false attacker identity (15 cases, SHOULD_REJECT)
    identities = [
        "the attack was launched by Alice.",
        "Bob is the adversary.",
        "Charlie orchestrated the scan.",
        "Eve deployed the payload.",
        "threat actor APT28 was identified.",
        "threat actor APT29 is responsible.",
        "the adversary is Lazarus Group.",
        "Fancy Bear executed the probe.",
        "hacker group Anonymous claimed credit.",
        "user Mallory compromised the sensor.",
        "insider threat Dave exfiltrated data.",
        "attacker Frank manipulated the flow.",
        "adversary Grace initiated connection.",
        "threat group Sandworm launched flood.",
        "identity of attacker is root@192.168.1.10."
    ]
    for i, ident in enumerate(identities, 1):
        cases.append({
            'test_id': f'ADV-IDENT-{i:03d}',
            'category': 'valid_telemetry_false_attacker_identity',
            'claim_text': f"Hybrid anomaly score of {s} and {ident}",
            'ground_truth_label': 'SHOULD_REJECT',
            'expected_verdict': 'UNSUPPORTED'
        })

    # Category 3: Valid telemetry + contradictory status (12 cases, SHOULD_REJECT)
    status_contradictions = [
        f"Anomaly score of {s} but status = BENIGN.",
        f"Anomaly score of {s} and status is BENIGN.",
        f"Anomaly score of {s}; status: BENIGN.",
        f"Threshold of {t} and evaluated as BENIGN.",
        f"Anomaly score of {s} and evaluated BENIGN-BLK-01 as BENIGN.",
        f"Score = {s} with status: BENIGN.",
        f"PAADU/CHARM evaluated BENIGN-BLK-01 as BENIGN with anomaly score of {s}.",
        f"Reconstruction error of {r} and classification is BENIGN.",
        f"Latent Mahalanobis distance of {m} but status = BENIGN.",
        f"Policies: [hybrid_threshold] but status: BENIGN.",
        f"Status is BENIGN with threshold of {t}.",
        f"Status = BENIGN and anomaly score of {s}."
    ]
    for i, text in enumerate(status_contradictions, 1):
        cases.append({
            'test_id': f'ADV-STATUS-{i:03d}',
            'category': 'valid_telemetry_contradictory_status',
            'claim_text': text,
            'ground_truth_label': 'SHOULD_REJECT',
            'expected_verdict': 'CONTRADICTED'
        })

    # Category 4: Valid telemetry + incorrect feature values (15 cases, SHOULD_REJECT)
    incorrect_feat_values = [
        f"Anomaly score of {s} and observed mean: 999.99 for Rate.",
        f"Anomaly score of {s} and benign mean is 999.99.",
        f"Rate contribution is +0.39 and threshold of {t}.",
        f"rst_flag_number contribution is -0.99.",
        f"cwr_flag_number observed mean is 50.0.",
        f"Rate observed mean: 0.00 vs benign mean: 1929.59.",
        f"Rate observed mean: 1221.61 vs benign mean: 0.00.",
        f"Rate (contribution: +0.39, observed mean: 1221.61 vs benign mean: 1929.59).",
        f"rst_flag_number (contribution: -0.22, observed mean: 0.00 vs benign mean: 0.00).",
        f"cwr_flag_number (contribution: +0.16, observed mean: 99.99 vs benign mean: 0.00).",
        f"Tot size contribution is +5.50.",
        f"UDP contribution is -1.00.",
        f"Threshold of {t} and Rate observed mean = 555.55.",
        f"Score = {s} and rst_flag_number benign mean = 12.34.",
        f"Score = {s} and contribution of Rate is +0.39."
    ]
    for i, text in enumerate(incorrect_feat_values, 1):
        cases.append({
            'test_id': f'ADV-FEATVAL-{i:03d}',
            'category': 'valid_telemetry_incorrect_feature_values',
            'claim_text': text,
            'ground_truth_label': 'SHOULD_REJECT',
            'expected_verdict': 'CONTRADICTED'
        })

    # Category 5: Multiple repeated numeric assertions (15 cases: 12 contradictory, 3 valid repetitions)
    repeated_cases = [
        (f"Anomaly score of {s} and the anomaly score of 999.000.", "SHOULD_REJECT", "CONTRADICTED"),
        (f"Threshold of {t} and threshold is 99.999.", "SHOULD_REJECT", "CONTRADICTED"),
        (f"Reconstruction error of {r} and reconstruction error of 0.9999.", "SHOULD_REJECT", "CONTRADICTED"),
        (f"Mahalanobis distance of {m} and Mahalanobis distance of 0.0000.", "SHOULD_REJECT", "CONTRADICTED"),
        (f"Anomaly score of {s} and score is 0.000.", "SHOULD_REJECT", "CONTRADICTED"),
        (f"Score: {s}; score: 50.000; score: {s}.", "SHOULD_REJECT", "CONTRADICTED"),
        (f"Threshold = {t} and threshold = 1.000.", "SHOULD_REJECT", "CONTRADICTED"),
        (f"Reconstruction loss = {r} and reconstruction error = 5.0000.", "SHOULD_REJECT", "CONTRADICTED"),
        (f"Score of {s} and score of {s} and score of 999.999.", "SHOULD_REJECT", "CONTRADICTED"),
        (f"Threshold of {t} and threshold of 5.500.", "SHOULD_REJECT", "CONTRADICTED"),
        (f"Anomaly score is {s} and anomaly score is 12.000.", "SHOULD_REJECT", "CONTRADICTED"),
        (f"Reconstruction error: {r} and reconstruction error: 0.1234.", "SHOULD_REJECT", "CONTRADICTED"),
        # Valid repetitions matching ground truth (SHOULD_ACCEPT)
        (f"Anomaly score of {s} and anomaly score of {s}.", "SHOULD_ACCEPT", "SUPPORTED"),
        (f"Threshold of {t} and threshold of {t}.", "SHOULD_ACCEPT", "SUPPORTED"),
        (f"Score = {s} and threshold = {t} and score = {s}.", "SHOULD_ACCEPT", "SUPPORTED")
    ]
    for i, (text, label, exp) in enumerate(repeated_cases, 1):
        cases.append({
            'test_id': f'ADV-REPEAT-{i:03d}',
            'category': 'multiple_repeated_numeric_assertions',
            'claim_text': text,
            'ground_truth_label': label,
            'expected_verdict': exp
        })

    # Category 6: Negative attribution values (15 cases: 10 valid should accept, 5 contradictory should reject)
    neg_attribution_cases = [
        # Valid negative contributions (SHOULD_ACCEPT)
        ("Rate contribution: -0.39.", "SHOULD_ACCEPT", "SUPPORTED"),
        ("Rate contribution = -0.39.", "SHOULD_ACCEPT", "SUPPORTED"),
        ("Rate (contribution: -0.39, observed mean: 1221.61 vs benign mean: 1929.59).", "SHOULD_ACCEPT", "SUPPORTED"),
        ("Tot size contribution: -0.05.", "SHOULD_ACCEPT", "SUPPORTED"),
        ("Tot size (contribution: -0.05, observed mean: 54.00 vs benign mean: 60.00).", "SHOULD_ACCEPT", "SUPPORTED"),
        (f"Rate contribution is -0.39 and anomaly score of {s}.", "SHOULD_ACCEPT", "SUPPORTED"),
        (f"Rate contribution: -0.39; threshold of {t}.", "SHOULD_ACCEPT", "SUPPORTED"),
        ("UDP contribution: 0.00.", "SHOULD_ACCEPT", "SUPPORTED"),
        (f"Rate contribution = -0.39 and reconstruction error of {r}.", "SHOULD_ACCEPT", "SUPPORTED"),
        ("contribution: -0.39.", "SHOULD_ACCEPT", "SUPPORTED"),
        # Contradictory negative contributions (SHOULD_REJECT)
        ("Rate contribution: -5.00.", "SHOULD_REJECT", "CONTRADICTED"),
        ("Rate contribution: +0.39.", "SHOULD_REJECT", "CONTRADICTED"),
        ("Tot size contribution: -10.00.", "SHOULD_REJECT", "CONTRADICTED"),
        ("rst_flag_number contribution: -0.22.", "SHOULD_REJECT", "CONTRADICTED"),
        ("cwr_flag_number contribution: -0.16.", "SHOULD_REJECT", "CONTRADICTED")
    ]
    for i, (text, label, exp) in enumerate(neg_attribution_cases, 1):
        cases.append({
            'test_id': f'ADV-NEGATTR-{i:03d}',
            'category': 'negative_attribution_values',
            'claim_text': text,
            'ground_truth_label': label,
            'expected_verdict': exp
        })

    # Category 7: Unknown telemetry fields (15 cases, SHOULD_REJECT)
    unknown_fields = [
        f"Anomaly score of {s} and QuantumJitter was 42.0.",
        f"Score = {s} and feature NonExistentFlowMetric has observed mean 10.0.",
        f"Threshold of {t} and CPU_Utilization was 98.5%.",
        f"Reconstruction error of {r} and MemoryBufferOverrun occurred.",
        f"Score: {s} and feature FooBar has observed mean 50.0.",
        f"Threshold: {t} and GPU_Core_Temp = 85.0.",
        f"Anomaly score of {s} and telemetry contains non-existent flow feature.",
        f"Score of {s} and PacketDropVariance was elevated.",
        f"Score = {s} and SyntheticJitterIndex = 12.3.",
        f"Threshold = {t} and UnknownProtocolHeader observed.",
        f"Rate contribution: -0.39 and BogusFeature contribution: 1.0.",
        f"Anomaly score of {s} and FanSpeedRPM = 3500.",
        f"Score of {s} and BGP_Flap_Count was 14.",
        f"Reconstruction loss = {r} and OpticalPowerLoss_dBm = -15.4.",
        f"Score = {s} and WirelessSNR was 12dB."
    ]
    for i, text in enumerate(unknown_fields, 1):
        cases.append({
            'test_id': f'ADV-UNKNOWN-{i:03d}',
            'category': 'unknown_telemetry_fields',
            'claim_text': text,
            'ground_truth_label': 'SHOULD_REJECT',
            'expected_verdict': 'CONTRADICTED' if 'feature' in text.lower() or 'non-existent' in text.lower() else 'UNSUPPORTED'
        })

    # Category 8: Unsupported but plausible security explanations (15 cases, SHOULD_REJECT)
    plausible_security = [
        f"Anomaly score of {s}, consistent with zero-day vulnerability in OpenSSL.",
        f"Anomaly score of {s}; traffic matches Mirai botnet command-and-control.",
        f"Score = {s}, indicating exploitation of CVE-2023-44487 HTTP/2 Rapid Reset.",
        f"Threshold of {t}; flow patterns match Log4j JNDI injection attempts.",
        f"Score of {s}, suggesting EternalBlue SMB lateral movement.",
        f"Anomaly score of {s}; behavior matches Reaper IoT botnet propagation.",
        f"Reconstruction error of {r}, consistent with Cobalt Strike beaconing.",
        f"Mahalanobis distance of {m}, indicating SYN flood DDoS weaponization.",
        f"Score = {s}; alert indicates BrickerBot hardware destruction payload.",
        f"Score of {s}, consistent with unauthorized DNS tunneling.",
        f"Threshold of {t}; telemetry reflects Slowloris connection starvation.",
        f"Score: {s}; signatures indicate Shamoon disk-wiping wiper attack.",
        f"Reconstruction error: {r}; flow corresponds to Tor exit node traffic.",
        f"Score of {s}; attack matches WannaCry killswitch domain evasion.",
        f"Threshold = {t}; signatures match DarkSide ransomware staging."
    ]
    for i, text in enumerate(plausible_security, 1):
        cases.append({
            'test_id': f'ADV-PLAUSIBLE-{i:03d}',
            'category': 'unsupported_plausible_security',
            'claim_text': text,
            'ground_truth_label': 'SHOULD_REJECT',
            'expected_verdict': 'UNSUPPORTED'
        })

    # Category 9: Punctuation and syntax variations (15 cases: 10 valid should accept, 5 adversarial should reject)
    syntax_cases = [
        # Valid syntax variations (SHOULD_ACCEPT)
        (f"PAADU/CHARM evaluated BENIGN-BLK-01 as ANOMALOUS with a hybrid anomaly score of {s} against the calibrated baseline threshold of {t}.", "SHOULD_ACCEPT", "SUPPORTED"),
        (f"The alert was triggered by policies: [hybrid_threshold, latent_evidence, burst], with a reconstruction error of {r} and a latent Mahalanobis distance of {m}.", "SHOULD_ACCEPT", "SUPPORTED"),
        (f"Anomaly score of {s} and threshold of {t}.", "SHOULD_ACCEPT", "SUPPORTED"),
        (f"Score: {s}; threshold: {t}.", "SHOULD_ACCEPT", "SUPPORTED"),
        (f"Score = {s}, threshold = {t}, status = ANOMALOUS.", "SHOULD_ACCEPT", "SUPPORTED"),
        (f"Score: {s} - threshold: {t}.", "SHOULD_ACCEPT", "SUPPORTED"),
        (f"Reconstruction error of {r} and latent Mahalanobis distance of {m}.", "SHOULD_ACCEPT", "SUPPORTED"),
        (f"Rate contribution: -0.39; rst_flag_number contribution: +0.22.", "SHOULD_ACCEPT", "SUPPORTED"),
        (f"Rate (contribution: -0.39, observed mean: 1221.61 vs benign mean: 1929.59); rst_flag_number (contribution: +0.22, observed mean: 0.00 vs benign mean: 0.00).", "SHOULD_ACCEPT", "SUPPORTED"),
        (f"Status is ANOMALOUS with score of {s}.", "SHOULD_ACCEPT", "SUPPORTED"),
        # Adversarial syntax variations (SHOULD_REJECT)
        (f"Score: {s}; Rate proves Alice compromised the device.", "SHOULD_REJECT", "UNSUPPORTED"),
        (f"Score = {s} / threshold = {t} / adversary is root.", "SHOULD_REJECT", "UNSUPPORTED"),
        (f"Score: {s} (compromised by APT29).", "SHOULD_REJECT", "UNSUPPORTED"),
        (f"Score: {s} [status: BENIGN].", "SHOULD_REJECT", "CONTRADICTED"),
        (f"Score: {s} -> attack launched from 192.168.1.100.", "SHOULD_REJECT", "UNSUPPORTED")
    ]
    for i, (text, label, exp) in enumerate(syntax_cases, 1):
        cases.append({
            'test_id': f'ADV-SYNTAX-{i:03d}',
            'category': 'punctuation_syntax_variations',
            'claim_text': text,
            'ground_truth_label': label,
            'expected_verdict': exp
        })

    return cases


def run_held_out_adversarial_suite() -> Dict[str, Any]:
    """Executes the held-out adversarial test suite against production verifier."""
    incident = build_reference_incident()
    cases = generate_held_out_dataset(incident)

    total_cases = len(cases)
    should_accept_count = sum(1 for c in cases if c['ground_truth_label'] == 'SHOULD_ACCEPT')
    should_reject_count = sum(1 for c in cases if c['ground_truth_label'] == 'SHOULD_REJECT')

    true_positives = 0
    false_positives = 0  # False Acceptance (CRITICAL MUST BE 0)
    true_negatives = 0
    false_negatives = 0  # False Rejection

    detailed_rows = []

    for c in cases:
        claim_text = c['claim_text']
        gt_label = c['ground_truth_label']

        status, action, presented_text, reason = validate_telemetry_claim_strict(claim_text, incident)
        is_verified = (status == "SUPPORTED" and "[VERIFIED TELEMETRY]" in presented_text)

        if gt_label == 'SHOULD_ACCEPT':
            if is_verified:
                true_positives += 1
                outcome = 'TP_CORRECT_ACCEPTANCE'
            else:
                false_negatives += 1
                outcome = 'FN_FALSE_REJECTION'
        else:  # SHOULD_REJECT
            if is_verified:
                false_positives += 1
                outcome = 'FP_FALSE_ACCEPTANCE_FATAL'
            else:
                true_negatives += 1
                outcome = 'TN_CORRECT_REJECTION'

        detailed_rows.append({
            'test_id': c['test_id'],
            'category': c['category'],
            'ground_truth_label': gt_label,
            'claim_text': claim_text,
            'observed_status': status,
            'observed_action': action,
            'is_verified': is_verified,
            'outcome': outcome,
            'reason': reason,
            'presented_text': presented_text
        })

    # Metrics with explicit denominators
    false_acceptance_rate = false_positives / should_reject_count if should_reject_count > 0 else "N/A"
    false_rejection_rate = false_negatives / should_accept_count if should_accept_count > 0 else "N/A"
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else "N/A"
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else "N/A"
    abstention_count = true_negatives + false_negatives
    abstention_rate = abstention_count / total_cases if total_cases > 0 else "N/A"

    summary = {
        'total_cases_evaluated': total_cases,
        'should_accept_positives': should_accept_count,
        'should_reject_negatives': should_reject_count,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'false_acceptance_rate_str': f"{false_positives}/{should_reject_count} ({false_acceptance_rate:.4f})" if isinstance(false_acceptance_rate, float) else "N/A",
        'false_rejection_rate_str': f"{false_negatives}/{should_accept_count} ({false_rejection_rate:.4f})" if isinstance(false_rejection_rate, float) else "N/A",
        'precision_str': f"{true_positives}/{(true_positives + false_positives)} ({precision:.4f})" if isinstance(precision, float) else "N/A",
        'recall_str': f"{true_positives}/{(true_positives + false_negatives)} ({recall:.4f})" if isinstance(recall, float) else "N/A",
        'abstention_rate_str': f"{abstention_count}/{total_cases} ({abstention_rate:.4f})" if isinstance(abstention_rate, float) else "N/A",
        'raw_metrics': {
            'false_acceptance_rate': false_acceptance_rate,
            'false_rejection_rate': false_rejection_rate,
            'precision': precision,
            'recall': recall,
            'abstention_rate': abstention_rate
        }
    }

    # Save detailed CSV
    csv_path = WORKSPACE / 'HELD_OUT_ADVERSARIAL_RESULTS.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_rows)

    # Save JSON summary
    json_path = WORKSPACE / 'results' / 'HELD_OUT_ADVERSARIAL_RESULTS.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == '__main__':
    res = run_held_out_adversarial_suite()
    print("=" * 72)
    print("  HELD-OUT ADVERSARIAL EVALUATION SUITE RESULTS")
    print("=" * 72)
    print(f"Total Cases Evaluated: {res['total_cases_evaluated']}")
    print(f"Ground Truth Positives (Should Accept): {res['should_accept_positives']}")
    print(f"Ground Truth Negatives (Should Reject): {res['should_reject_negatives']}")
    print(f"True Positives: {res['true_positives']}")
    print(f"False Positives (False Acceptance): {res['false_positives']} [{res['false_acceptance_rate_str']}]")
    print(f"True Negatives: {res['true_negatives']}")
    print(f"False Negatives (False Rejection): {res['false_negatives']} [{res['false_rejection_rate_str']}]")
    print(f"Precision: {res['precision_str']}")
    print(f"Recall: {res['recall_str']}")
    print(f"Abstention / Rejection Rate: {res['abstention_rate_str']}")
    print("=" * 72)
