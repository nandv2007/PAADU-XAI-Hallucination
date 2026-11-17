#!/usr/bin/env python3
"""
Standardized Evaluation Dataset Generator.
Generates controlled, reproducible network telemetry blocks matching the 46-feature
CICIoT2023 schema from charm_engine_state.pkl.
Includes 4 traffic profiles:
1. Benign IoT Sensor Telemetry (N=15)
2. TCP SYN Flood Burst (N=15)
3. Volumetric Rate / Byte Flood (N=15)
4. Reconnaissance & Port Scan (N=15)
Total N = 60 evaluation blocks.
"""
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
INTEG_PROJECT = WORKSPACE / 'integrated_project'
CHARM_ORIG = INTEG_PROJECT / 'charm_original'

for p in [str(CHARM_ORIG / 'project'), str(CHARM_ORIG / 'project' / 'original')]:
    if p not in sys.path:
        sys.path.insert(0, p)

def load_engine_state():
    state_path = CHARM_ORIG / 'artifacts' / 'charm_engine_state.pkl'
    with open(state_path, 'rb') as f:
        state = pickle.load(f)
    return state

def generate_evaluation_dataset(seed: int = 42) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generates 60 evaluation blocks (4 profiles x 15 blocks) with full ground-truth metadata."""
    state = load_engine_state()
    features = state['features']
    mean = state['mean']  # shape (1, 1, 1, 46)
    std = state['std']    # shape (1, 1, 1, 46)

    rng = np.random.default_rng(seed)
    blocks = []
    metadata = []

    # Indices of key features
    syn_cnt_idx = features.index('syn_count')
    syn_flag_idx = features.index('syn_flag_number')
    tcp_idx = features.index('TCP')
    rate_idx = features.index('Rate')
    tot_sum_idx = features.index('Tot sum')
    tot_size_idx = features.index('Tot size')
    rst_flag_idx = features.index('rst_flag_number')
    fin_flag_idx = features.index('fin_flag_number')
    ack_cnt_idx = features.index('ack_count')
    udp_idx = features.index('UDP')

    # 1. Benign IoT Sensor Telemetry (N=15)
    for i in range(15):
        # Sample directly within 0.05 std of benign mean (strict normal operating behavior)
        b = rng.normal(loc=mean, scale=std * 0.05, size=(1, 1, 10, 46)).astype(np.float32)
        # Ensure non-negative physical features
        b = np.maximum(b, 0.0)
        blocks.append(b)
        metadata.append({
            'block_id': f'BENIGN-BLK-{i+1:02d}',
            'ground_truth_label': 'BENIGN',
            'traffic_profile': 'Benign IoT Sensor Telemetry',
            'target_features': [],
            'simulated_attack': None,
            'source_ip': f'192.168.1.{10 + i}',
            'dest_ip': '192.168.1.1',
            'sensor_id': f'sensor-node-{i+1:03d}',
            'timestamp': f'2026-11-14T10:{i:02d}:00Z',
            'interface': 'eth0'
        })

    # 2. TCP SYN Flood Attack Simulation (N=15)
    for i in range(15):
        b = rng.normal(loc=mean, scale=std * 0.05, size=(1, 1, 10, 46)).astype(np.float32)
        # Inject realistic SYN flood perturbations
        perturb_mult = 15.0 + (i * 1.5)
        b[0, 0, :, syn_cnt_idx] += std[0, 0, 0, syn_cnt_idx] * perturb_mult
        b[0, 0, :, syn_flag_idx] += std[0, 0, 0, syn_flag_idx] * (8.0 + (i * 0.5))
        b[0, 0, :, tcp_idx] += std[0, 0, 0, tcp_idx] * (4.0 + (i * 0.3))
        b = np.maximum(b, 0.0)
        blocks.append(b)
        metadata.append({
            'block_id': f'SYN-FLOOD-BLK-{i+1:02d}',
            'ground_truth_label': 'ATTACK',
            'traffic_profile': 'TCP SYN Flood Attack Simulation',
            'target_features': ['syn_count', 'syn_flag_number', 'TCP'],
            'simulated_attack': 'TCP SYN Flood',
            'source_ip': f'10.0.0.{100 + i}',
            'dest_ip': '192.168.1.100',
            'sensor_id': f'gateway-gw-{i+1:03d}',
            'timestamp': f'2026-11-14T11:{i:02d}:00Z',
            'interface': 'eth0'
        })

    # 3. Volumetric Rate / Byte Flood (N=15)
    for i in range(15):
        b = rng.normal(loc=mean, scale=std * 0.05, size=(1, 1, 10, 46)).astype(np.float32)
        vol_mult = 20.0 + (i * 2.0)
        b[0, 0, :, rate_idx] += std[0, 0, 0, rate_idx] * vol_mult
        b[0, 0, :, tot_sum_idx] += std[0, 0, 0, tot_sum_idx] * vol_mult
        b[0, 0, :, tot_size_idx] += std[0, 0, 0, tot_size_idx] * (vol_mult * 0.8)
        b[0, 0, :, udp_idx] += std[0, 0, 0, udp_idx] * 5.0
        b = np.maximum(b, 0.0)
        blocks.append(b)
        metadata.append({
            'block_id': f'VOLUMETRIC-BLK-{i+1:02d}',
            'ground_truth_label': 'ATTACK',
            'traffic_profile': 'Volumetric Rate / Byte Flood',
            'target_features': ['Rate', 'Tot sum', 'Tot size', 'UDP'],
            'simulated_attack': 'UDP Volumetric Flood',
            'source_ip': f'172.16.0.{50 + i}',
            'dest_ip': '192.168.1.200',
            'sensor_id': f'edge-router-{i+1:03d}',
            'timestamp': f'2026-11-14T12:{i:02d}:00Z',
            'interface': 'wan0'
        })

    # 4. Reconnaissance & Port Scan (N=15)
    for i in range(15):
        b = rng.normal(loc=mean, scale=std * 0.05, size=(1, 1, 10, 46)).astype(np.float32)
        scan_mult = 12.0 + (i * 1.0)
        b[0, 0, :, rst_flag_idx] += std[0, 0, 0, rst_flag_idx] * scan_mult
        b[0, 0, :, fin_flag_idx] += std[0, 0, 0, fin_flag_idx] * (scan_mult * 0.7)
        b[0, 0, :, ack_cnt_idx] += std[0, 0, 0, ack_cnt_idx] * (scan_mult * 0.5)
        b = np.maximum(b, 0.0)
        blocks.append(b)
        metadata.append({
            'block_id': f'PORTSCAN-BLK-{i+1:02d}',
            'ground_truth_label': 'ATTACK',
            'traffic_profile': 'Reconnaissance & Port Scan',
            'target_features': ['rst_flag_number', 'fin_flag_number', 'ack_count'],
            'simulated_attack': 'Port Scan / Reconnaissance',
            'source_ip': f'192.168.1.{200 + i}',
            'dest_ip': '192.168.1.1',
            'sensor_id': f'ids-probe-{i+1:03d}',
            'timestamp': f'2026-11-14T13:{i:02d}:00Z',
            'interface': 'eth1'
        })

    dataset_tensor = np.concatenate(blocks, axis=0)  # Shape (60, 1, 10, 46)
    return dataset_tensor, metadata

if __name__ == '__main__':
    tensor, meta = generate_evaluation_dataset()
    print(f"Generated dataset shape: {tensor.shape}")
    print(f"Total metadata records: {len(meta)}")
    print(f"Profiles: {[m['traffic_profile'] for m in meta[::15]]}")
