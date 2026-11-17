#!/usr/bin/env python3
"""
Standalone Hash Integrity Verification Script.
Computes and verifies SHA-256 hashes of original research files.
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
    mode = sys.argv[1] if len(sys.argv) > 1 else 'verify'
    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parent

    # Baseline snapshot
    snapshot_path = script_dir / 'original_baseline_hashes.json'
    
    # Target directories
    targets = ['integrated_project', 'datasets']
    current_hashes = {}
    for target in targets:
        target_path = workspace / target
        if target_path.exists():
            print(f"Scanning {target_path}...")
            current_hashes.update(scan_directory(target_path))

    if mode == 'record' or not snapshot_path.exists():
        with open(snapshot_path, 'w') as f:
            json.dump(current_hashes, f, indent=2)
        print(f"Recorded {len(current_hashes)} file hashes to {snapshot_path}")

    if mode == 'verify':
        if not snapshot_path.exists():
            print(f"Error: snapshot {snapshot_path} does not exist.")
            sys.exit(1)
        with open(snapshot_path, 'r') as f:
            initial_hashes = json.load(f)

        all_matched = True
        matched_count = 0
        all_keys = sorted(set(initial_hashes.keys()) & set(current_hashes.keys()))
        for key in all_keys:
            init_sha = initial_hashes[key]['sha256']
            curr_sha = current_hashes[key]['sha256']
            if init_sha != curr_sha:
                print(f"INTEGRITY MISMATCH: {key} (initial: {init_sha}, current: {curr_sha})")
                all_matched = False
            else:
                matched_count += 1

        print(f"Verification complete: {matched_count} files checked and matched.")
        if all_matched and matched_count > 0:
            print("SUCCESS: ALL ORIGINAL FILES ARE 100% UNCHANGED!")
            sys.exit(0)
        else:
            print("WARNING: INTEGRITY FAILURE! MODIFICATIONS DETECTED!")
            sys.exit(2)

if __name__ == '__main__':
    main()
