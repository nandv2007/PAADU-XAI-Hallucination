#!/usr/bin/env python3
"""
Integrity verification script:
Calculates SHA-256 hashes of all files in integrated_project and datasets.
Can be run in 'record' mode (initial snapshot) or 'verify' mode (comparing before & after).
"""
import os
import sys
import json
import hashlib
import csv
from pathlib import Path

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def scan_directory(base_dir):
    records = {}
    base_path = Path(base_dir).resolve()
    for root, _, files in sorted(os.walk(base_path)):
        for fname in sorted(files):
            if fname.startswith('.DS_Store') or fname.endswith('.pyc'):
                continue
            if '__pycache__' in root:
                continue
            fpath = Path(root) / fname
            rel_path = fpath.relative_to(base_path.parent)
            sha = compute_sha256(fpath)
            size = fpath.stat().st_size
            records[str(rel_path)] = {
                'sha256': sha,
                'size_bytes': size
            }
    return records

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'record'
    workspace = Path(__file__).resolve().parents[2]
    integrity_dir = Path(__file__).resolve().parents[1] / 'integrity'
    integrity_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = integrity_dir / 'initial_hashes.json'
    csv_path = integrity_dir / 'SOURCE_HASHES_BEFORE_AFTER.csv'

    print(f"Running integrity scan in mode: {mode}")
    targets = ['integrated_project', 'datasets']
    current_hashes = {}
    for target in targets:
        target_path = workspace / target
        if target_path.exists():
            print(f"Scanning {target_path}...")
            current_hashes.update(scan_directory(target_path))

    if mode == 'record':
        with open(snapshot_path, 'w') as f:
            json.dump(current_hashes, f, indent=2)
        print(f"Recorded {len(current_hashes)} file hashes to {snapshot_path}")
        
        # Write initial CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['relative_path', 'initial_sha256', 'final_sha256', 'status', 'size_bytes'])
            for path, info in sorted(current_hashes.items()):
                writer.writerow([path, info['sha256'], 'PENDING', 'RECORDED', info['size_bytes']])
        print(f"Created initial template {csv_path}")

    elif mode == 'verify':
        if not snapshot_path.exists():
            print(f"Error: snapshot {snapshot_path} does not exist.")
            sys.exit(1)
        with open(snapshot_path, 'r') as f:
            initial_hashes = json.load(f)

        all_matched = True
        matched_count = 0
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['relative_path', 'initial_sha256', 'final_sha256', 'status', 'size_bytes'])
            all_keys = sorted(set(initial_hashes.keys()) & set(current_hashes.keys()))
            for key in all_keys:
                init_info = initial_hashes.get(key)
                curr_info = current_hashes.get(key)
                init_sha = init_info['sha256'] if init_info else 'MISSING'
                curr_sha = curr_info['sha256'] if curr_info else 'MISSING'
                size = curr_info['size_bytes'] if curr_info else (init_info['size_bytes'] if init_info else 0)
                if init_sha == curr_sha:
                    status = 'UNCHANGED'
                    matched_count += 1
                else:
                    status = 'MODIFIED'
                    all_matched = False
                writer.writerow([key, init_sha, curr_sha, status, size])

        print(f"Verification complete: {matched_count} files checked.")
        if all_matched and matched_count > 0:
            print("SUCCESS: ALL ORIGINAL FILES ARE 100% UNCHANGED!")
            sys.exit(0)
        else:
            print("WARNING: INTEGRITY FAILURE! MODIFICATIONS DETECTED!")
            sys.exit(2)

if __name__ == '__main__':
    main()
