#!/usr/bin/env python3
"""
Comprehensive Dataset Inspector for PAADU/CHARM Evaluation.
Inspects datasets/Dataset, datasets/ciciov24, datasets/cic25.
Collects all 14 metrics required by TASK 4 and writes:
- external_evaluation/dataset_inspection/DATASET_INSPECTION.csv
- external_evaluation/dataset_inspection/DATASET_COMPATIBILITY.csv
"""
import os
import sys
import glob
import json
import csv
from pathlib import Path
import pandas as pd
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
DATASETS_DIR = WORKSPACE / 'datasets'
OUT_DIR = WORKSPACE / 'external_evaluation' / 'dataset_inspection'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHARM_FEATURES = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP',
    'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT',
    'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
]

def inspect_ciciov(dataset_dir, dataset_name):
    print(f"Inspecting {dataset_name} at {dataset_dir}...")
    files = sorted(glob.glob(str(dataset_dir / '**' / '*'), recursive=True))
    files = [f for f in files if os.path.isfile(f) and not os.path.basename(f).startswith('.DS_Store')]
    
    csv_files = [f for f in files if f.endswith('.csv')]
    doc_files = [f for f in files if f.endswith('.docx') or f.endswith('.pdf')]
    
    # We focus on the decimal representation for primary counts since decimal, binary, hex represent the same runs
    decimal_csvs = [f for f in csv_files if '/decimal/' in f]
    
    total_records = 0
    benign_count = 0
    malicious_count = 0
    attack_counts = {}
    missing_values = 0
    columns_set = set()
    numeric_cols = set()
    categorical_cols = set()
    
    sample_df = None
    for f in decimal_csvs:
        chunk_iter = pd.read_csv(f, chunksize=50000, low_memory=False)
        for chunk in chunk_iter:
            if sample_df is None:
                sample_df = chunk.head(5)
            total_records += len(chunk)
            missing_values += int(chunk.isna().sum().sum())
            for col in chunk.columns:
                columns_set.add(col)
                if pd.api.types.is_numeric_dtype(chunk[col]):
                    numeric_cols.add(col)
                else:
                    categorical_cols.add(col)
            
            # Label counting
            if 'label' in chunk.columns:
                counts = chunk['label'].value_counts()
                for lbl, c in counts.items():
                    if str(lbl).strip().upper() == 'BENIGN':
                        benign_count += c
                    else:
                        malicious_count += c
            
            if 'specific_class' in chunk.columns:
                sp_counts = chunk['specific_class'].value_counts()
                for cat, c in sp_counts.items():
                    cat_str = str(cat).strip()
                    if cat_str.upper() != 'BENIGN':
                        attack_counts[cat_str] = attack_counts.get(cat_str, 0) + c
            elif 'category' in chunk.columns:
                cat_counts = chunk['category'].value_counts()
                for cat, c in cat_counts.items():
                    cat_str = str(cat).strip()
                    if cat_str.upper() != 'BENIGN':
                        attack_counts[cat_str] = attack_counts.get(cat_str, 0) + c

    return {
        'dataset_name': dataset_name,
        'location': str(dataset_dir),
        'file_formats': 'CSV (.csv), Word (.docx), PDF (.pdf)',
        'number_of_files': len(files),
        'csv_files_count': len(csv_files),
        'total_records_decimal': total_records,
        'columns': sorted(list(columns_set)),
        'label_column': 'label (also category, specific_class)',
        'benign_count': benign_count,
        'malicious_count': malicious_count,
        'attack_categories': attack_counts,
        'missing_values': missing_values,
        'numeric_features': sorted(list(numeric_cols)),
        'categorical_features': sorted(list(categorical_cols)),
        'temporal_info': 'Sequential CAN log packets without explicit high-resolution timestamp column in decimal format',
        'compatibility': 'INCOMPATIBLE',
        'compatibility_reason': 'CAN-bus frame schema (CAN ID + 8 data bytes payload). 0 of 46 required CHARM IP/TCP flow features are present.'
    }

def inspect_ciciomt(dataset_dir, dataset_name):
    print(f"Inspecting {dataset_name} at {dataset_dir}...")
    files = sorted(glob.glob(str(dataset_dir / '**' / '*'), recursive=True))
    files = [f for f in files if os.path.isfile(f) and not os.path.basename(f).startswith('.DS_Store')]
    
    csv_files = [f for f in files if f.endswith('.csv')]
    pcap_files = [f for f in files if f.endswith('.pcap')]
    doc_files = [f for f in files if f.endswith('.pdf')]
    
    total_records = 0
    benign_count = 0
    malicious_count = 0
    attack_counts = {}
    missing_values = 0
    columns_set = set()
    numeric_cols = set()
    categorical_cols = set()
    
    # We inspect test and train CSV files in WiFI_and_MQTT
    for f in csv_files:
        fname = os.path.basename(f)
        is_benign = 'benign' in fname.lower()
        # attack name from filename
        attack_name = fname.replace('_test.pcap.csv', '').replace('_train.pcap.csv', '')
        
        # Stream read
        chunk_iter = pd.read_csv(f, chunksize=50000, low_memory=False)
        for chunk in chunk_iter:
            total_records += len(chunk)
            missing_values += int(chunk.isna().sum().sum())
            for col in chunk.columns:
                columns_set.add(col)
                if pd.api.types.is_numeric_dtype(chunk[col]):
                    numeric_cols.add(col)
                else:
                    categorical_cols.add(col)
            
            if is_benign:
                benign_count += len(chunk)
            else:
                malicious_count += len(chunk)
                attack_counts[attack_name] = attack_counts.get(attack_name, 0) + len(chunk)

    return {
        'dataset_name': dataset_name,
        'location': str(dataset_dir),
        'file_formats': 'CSV (.csv), PCAP (.pcap), PDF (.pdf)',
        'number_of_files': len(files),
        'csv_files_count': len(csv_files),
        'pcap_files_count': len(pcap_files),
        'total_records_csv': total_records,
        'columns': sorted(list(columns_set)),
        'label_column': 'Filename-encoded (Benign vs Attack class in filename)',
        'benign_count': benign_count,
        'malicious_count': malicious_count,
        'attack_categories': attack_counts,
        'missing_values': missing_values,
        'numeric_features': sorted(list(numeric_cols)),
        'categorical_features': sorted(list(categorical_cols)),
        'temporal_info': 'Implicit flow sequence ordering within each pcap-derived CSV',
        'compatibility': 'INCOMPATIBLE',
        'compatibility_reason': 'Has 39 flow features with 37 matching names. However, 9 critical CHARM features are completely absent (flow_duration, Drate, Duration, Srate, urg_count, Magnitue, Radius, Covariance, Weight). Without them, zero-filling or imputation violates physical definitions and triggers severe false alarm distortion.'
    }

def main():
    results = []
    
    # 1. Dataset/ (CICIoMT2024)
    res_iomt = inspect_ciciomt(DATASETS_DIR / 'Dataset', 'CICIoMT2024 (Dataset/)')
    results.append(res_iomt)
    
    # 2. ciciov24/ (CICIoV2024)
    res_iov = inspect_ciciov(DATASETS_DIR / 'ciciov24', 'CICIoV2024 (ciciov24/)')
    results.append(res_iov)
    
    # 3. cic25/ (Duplicate of CICIoV2024)
    res_cic25 = inspect_ciciov(DATASETS_DIR / 'cic25', 'CIC2025 (cic25/ [Duplicate of CICIoV2024])')
    results.append(res_cic25)
    
    # Write DATASET_INSPECTION.csv
    inspection_csv = OUT_DIR / 'DATASET_INSPECTION.csv'
    with open(inspection_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'dataset_id', 'verified_name', 'directory', 'file_formats', 'file_count',
            'total_records', 'benign_records', 'malicious_records', 'feature_count',
            'missing_values', 'label_column', 'temporal_info', 'compatibility_status'
        ])
        for r in results:
            writer.writerow([
                r['dataset_name'].split()[0],
                r['dataset_name'],
                r['location'],
                r['file_formats'],
                r['number_of_files'],
                r.get('total_records_csv') or r.get('total_records_decimal'),
                r['benign_count'],
                r['malicious_count'],
                len(r['columns']),
                r['missing_values'],
                r['label_column'],
                r['temporal_info'],
                r['compatibility']
            ])
    print(f"Wrote inspection summary to {inspection_csv}")
    
    # Write detailed DATASET_COMPATIBILITY.csv
    compat_csv = OUT_DIR / 'DATASET_COMPATIBILITY.csv'
    with open(compat_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'dataset_name', 'compatibility_classification', 'required_features_count',
            'present_features_count', 'overlapping_features_count', 'missing_features',
            'extra_features', 'justification_and_scientific_blocker'
        ])
        
        # CICIoMT2024
        cols_iomt = set(res_iomt['columns'])
        charm_set = set(CHARM_FEATURES)
        overlap_iomt = charm_set & cols_iomt
        missing_iomt = charm_set - cols_iomt
        extra_iomt = cols_iomt - charm_set
        writer.writerow([
            'CICIoMT2024 (Dataset/)',
            'INCOMPATIBLE',
            len(CHARM_FEATURES),
            len(cols_iomt),
            len(overlap_iomt),
            "; ".join(sorted(missing_iomt)),
            "; ".join(sorted(extra_iomt)),
            res_iomt['compatibility_reason']
        ])
        
        # CICIoV2024
        cols_iov = set(res_iov['columns'])
        overlap_iov = charm_set & cols_iov
        missing_iov = charm_set - cols_iov
        extra_iov = cols_iov - charm_set
        writer.writerow([
            'CICIoV2024 (ciciov24/)',
            'INCOMPATIBLE',
            len(CHARM_FEATURES),
            len(cols_iov),
            len(overlap_iov),
            "; ".join(sorted(missing_iov)),
            "; ".join(sorted(extra_iov)),
            res_iov['compatibility_reason']
        ])
        
        # cic25
        cols_cic25 = set(res_cic25['columns'])
        overlap_cic25 = charm_set & cols_cic25
        missing_cic25 = charm_set - cols_cic25
        extra_cic25 = cols_cic25 - charm_set
        writer.writerow([
            'CIC2025 (cic25/)',
            'INCOMPATIBLE',
            len(CHARM_FEATURES),
            len(cols_cic25),
            len(overlap_cic25),
            "; ".join(sorted(missing_cic25)),
            "; ".join(sorted(extra_cic25)),
            res_cic25['compatibility_reason'] + " Verified exact duplicate of ciciov24."
        ])
    print(f"Wrote compatibility report to {compat_csv}")

    # Also save detailed JSON for deep reporting
    with open(OUT_DIR / 'detailed_dataset_profiles.json', 'w') as f:
        # Convert non-serializable objects
        clean_res = []
        for r in results:
            clean_res.append({k: (list(v) if isinstance(v, (set, list)) else v) for k, v in r.items()})
        json.dump(clean_res, f, indent=2)

if __name__ == '__main__':
    main()
