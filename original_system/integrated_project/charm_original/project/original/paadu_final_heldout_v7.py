#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_framework(path: str):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("paadu_framework_v7", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load framework: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_manifest(path: str) -> list[str]:
    """Read a path manifest and return absolute CSV paths."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")

    raw = []
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text())
        if isinstance(obj, dict):
            for key in ("files", "paths", "file_paths"):
                if key in obj:
                    obj = obj[key]
                    break
        if not isinstance(obj, list):
            raise ValueError(f"JSON manifest must be a list or contain files[]: {p}")
        raw = obj
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        col = next((c for c in ("path", "file", "filepath", "csv_path") if c in df.columns), None)
        if col is None:
            raise ValueError(f"CSV manifest needs one of path/file/filepath/csv_path columns: {p}")
        raw = df[col].dropna().astype(str).tolist()
    else:
        raw = [line.strip() for line in p.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]

    out = []
    for item in raw:
        value = str(item).strip()
        q = Path(value).expanduser()
        if not q.is_absolute():
            q = (p.parent / q).resolve()
        else:
            q = q.resolve()
        if q.suffix.lower() != ".csv":
            raise ValueError(f"Manifest entry is not a CSV file: {q}")
        if not q.exists():
            raise FileNotFoundError(f"Manifest entry not found: {q}")
        out.append(str(q))
    if not out:
        raise ValueError(f"Manifest is empty: {p}")
    return out


def unique_realpaths(paths: list[str]) -> list[str]:
    seen = set()
    out = []
    for p in paths:
        q = os.path.realpath(p)
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def stage_profile_files(train_files: list[str], root: str):
    """Create a temporary directory containing only training files for profiling."""
    stage = Path(root) / "_profile_training_files"
    stage.mkdir(parents=True, exist_ok=True)
    used = set()
    staged = []
    for i, src in enumerate(train_files):
        name = Path(src).name
        if name in used:
            name = f"{i:04d}_{name}"
        used.add(name)
        dst = stage / name
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        staged.append(str(dst))
    return str(stage), staged


def audit_benign_manifest(mod, files: list[str], role: str, allow_mixed: bool = False):
    """Fail closed if a training/calibration manifest contains attack rows."""
    issues = []
    for fp in files:
        try:
            df = pd.read_csv(fp, nrows=100_000, low_memory=False)
            schema = mod.detect_schema(df, os.path.basename(fp))
            label_col = schema.get("label_col")
            if label_col and label_col in df.columns:
                vals = df[label_col].astype(str).str.lower()
                attack_like = vals.map(mod._row_is_attack).sum()
                if attack_like > 0:
                    issues.append({"file": fp, "attack_like_rows_in_sample": int(attack_like),
                                   "label_col": str(label_col)})
            elif not mod.is_benign_file(os.path.basename(fp), ["benign", "normal"]):
                issues.append({"file": fp, "reason": "filename is not clearly benign and no label column was found"})
        except Exception as e:
            issues.append({"file": fp, "reason": f"audit error: {e}"})
    if issues and not allow_mixed:
        raise RuntimeError(
            f"{role} manifest failed benign-only audit. Use only benign files or pass "
            f"--allow-mixed-calibration after explicitly documenting the row-level split.\n"
            + json.dumps(issues[:20], indent=2)
        )
    return issues


def is_attack_label(mod, label) -> bool:
    return bool(mod._row_is_attack(str(label)))


def bootstrap_rate(rows, numerator_key: str, denominator_key: str,
                   group_predicate, n_boot: int, seed: int):
    groups = [r for r in rows if group_predicate(r)]
    if not groups:
        return {"n_groups": 0, "estimate": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    estimate = sum(r[numerator_key] for r in groups) / max(1, sum(r[denominator_key] for r in groups))
    vals = []
    for _ in range(n_boot):
        sample = [groups[i] for i in rng.integers(0, len(groups), size=len(groups))]
        vals.append(sum(r[numerator_key] for r in sample) /
                    max(1, sum(r[denominator_key] for r in sample)))
    return {"n_groups": len(groups), "estimate": float(estimate),
            "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def bootstrap_file_f1(rows, n_boot: int, seed: int):
    if not rows:
        return {"n_groups": 0, "estimate": None, "ci95": [None, None]}
    def f1(sample):
        tp = sum(r["file_tp"] for r in sample)
        fp = sum(r["file_fp"] for r in sample)
        fn = sum(r["file_fn"] for r in sample)
        return 2.0 * tp / max(1e-12, 2.0 * tp + fp + fn)
    rng = np.random.default_rng(seed)
    estimate = f1(rows)
    vals = [f1([rows[i] for i in rng.integers(0, len(rows), size=len(rows))])
            for _ in range(n_boot)]
    return {"n_groups": len(rows), "estimate": float(estimate),
            "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def make_final_evaluator(mod, model, profile, train_ds, device, threshold,
                         mahal, mse_stats, dm_stats, relaxer, cfg):
    """Subclass RichEvaluator to fix graph-off attention handling and add final-safe output."""
    import torch
    from torch.utils.data import DataLoader

    class FinalEvaluator(mod.RichEvaluator):
        def evaluate_file(self, csv_path, nrows=None, row_fraction=None, benign_only=False):
            fname = os.path.basename(csv_path)
            ds = mod.PADataset(
                [csv_path], profile, scaler=train_ds.scaler,
                train_columns=train_ds.train_columns, nrows=nrows,
                benign_only=benign_only, row_fraction=row_fraction,
                window_seconds=cfg.window_seconds, verbose=False,
            )
            if len(ds) == 0:
                return None
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            n_rows = len(ds.features)
            n_blocks = len(ds)
            z_threshold = (self.relaxer.get_z_threshold(n_rows, n_blocks=n_blocks)
                           if cfg.volume_adaptive else self.relaxer.z_floor)
            tracker = (mod.PerceptualDriftTracker(
                base_leak=profile.base_leak, limit=profile.limit,
                ema_alpha=cfg.ema_alpha, ema_multiplier=cfg.ema_multiplier,
                min_alert_severity=cfg.min_alert_severity,
                drift_requires_current_severity=cfg.drift_requires_current_severity)
                       if cfg.use_drift else None)
            records, attn_acc = [], []
            start = time.perf_counter()
            with torch.no_grad():
                for batch, adj, lbls in loader:
                    batch = batch.to(device)
                    pred, mask, p_score, attn, lat = self.model(
                        batch, adj.to(device), return_attn=True, return_latent=True)
                    target = batch.reshape(pred.shape)
                    raw_err = float(((pred - target) ** 2).mean().item())
                    dm = float(self.mahal.score(lat)[0])
                    hybrid, z_mse, z_mahal = self._hybrid_score(raw_err, dm)
                    double_soft = z_mse > cfg.double_soft_z and z_mahal > cfg.double_soft_z
                    latent_only = z_mahal > cfg.latent_only_z and z_mse <= cfg.double_soft_z
                    if cfg.soft_triggers and double_soft:
                        eff = z_threshold * cfg.double_soft_relax
                    elif cfg.soft_triggers and latent_only:
                        eff = z_threshold * cfg.latent_only_relax
                    else:
                        eff = z_threshold
                    fires = bool(hybrid > eff)
                    severity = max(0.0, hybrid - eff) if fires else 0.0
                    if severity < cfg.min_alert_severity:
                        severity = 0.0
                    if tracker is not None:
                        is_alert, cause = tracker.update(severity, raw_err)
                    else:
                        is_alert, cause = int(fires and severity > 0), ("fires" if fires else "none")
                    flat = mod._flatten_labels(lbls)
                    dom = pd.Series(flat).astype(str).mode()[0].lower() if flat else "unknown"
                    records.append({
                        "label": dom, "error": raw_err, "dm": dm, "hybrid": float(hybrid),
                        "z_mse": float(z_mse), "z_mahal": float(z_mahal),
                        "effective_threshold": float(eff), "fires": int(fires),
                        "severity": float(severity), "alert": int(is_alert), "cause": cause,
                        "bucket": float(tracker.value) if tracker is not None else 0.0,
                        "ema": float(tracker.ema) if tracker is not None and tracker.ema is not None else 0.0,
                    })
                    if attn is not None:
                        attn_acc.append(attn.detach().cpu().numpy())
            if not records:
                return None
            out = pd.DataFrame(records)
            alert_pct = float(out.alert.mean() * 100.0)
            fires_pct = float(out.fires.mean() * 100.0)
            cand = ((out.fires.astype(bool)) |
                    (out.hybrid > out.effective_threshold * cfg.file_candidate_relax) |
                    ((out.z_mse > cfg.file_candidate_soft_z) &
                     (out.z_mahal > cfg.file_candidate_soft_z)))
            cand_pct = float(cand.mean() * 100.0)
            if cfg.use_file_evidence_rule:
                file_score = max(alert_pct, fires_pct, cand_pct)
                basis = ("candidate" if file_score == cand_pct else
                          ("fires" if file_score == fires_pct else "alert"))
            else:
                file_score, basis = alert_pct, "alert"
            return {
                "file": fname, "path": os.path.realpath(csv_path), "df": out,
                "elapsed": time.perf_counter() - start, "blocks": len(out),
                "alerts": int(out.alert.sum()), "alert_pct": alert_pct,
                "fires_pct": fires_pct, "candidate_pct": cand_pct,
                "file_score_pct": file_score, "file_basis": basis,
                "avg_attn": np.mean(attn_acc, axis=0) if attn_acc else None,
                "z_thresh": float(z_threshold),
            }

    return FinalEvaluator(model, profile, train_ds.scaler, train_ds.train_columns,
                          device, threshold, mahal, mse_stats, dm_stats, relaxer, cfg)


def main():
    ap = argparse.ArgumentParser(description="Leakage-safe final held-out PAADU evaluation")
    ap.add_argument("--framework", required=True, help="path to paadu_benchmark_v7.py")
    ap.add_argument("--train-files", required=True, help="benign training-file manifest")
    ap.add_argument("--calib-files", required=True, help="benign calibration-file manifest")
    ap.add_argument("--test-files", required=True, help="final held-out benign+attack manifest")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mask-ratio", required=True, type=float)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--no-gnn", action="store_true")
    ap.add_argument("--window-seconds", type=float, default=0.0)
    ap.add_argument("--train-nrows", type=int, default=None)
    ap.add_argument("--calib-nrows", type=int, default=None)
    ap.add_argument("--test-nrows", type=int, default=None)
    ap.add_argument("--w-recon", type=float, default=0.20)
    ap.add_argument("--w-mahal", type=float, default=0.80)
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--hybrid-floor-percentile", type=float, default=99.0)
    ap.add_argument("--hard-z-floor", type=float, default=2.5)
    ap.add_argument("--double-soft-z", type=float, default=2.5)
    ap.add_argument("--double-soft-relax", type=float, default=0.60)
    ap.add_argument("--latent-only-z", type=float, default=3.0)
    ap.add_argument("--latent-only-relax", type=float, default=0.70)
    ap.add_argument("--ema-alpha", type=float, default=0.05)
    ap.add_argument("--ema-multiplier", type=float, default=6.5)
    ap.add_argument("--min-alert-severity", type=float, default=0.05)
    ap.add_argument("--bucket-limit", type=float, default=3.2)
    ap.add_argument("--vol-high-mult", type=float, default=1.20)
    ap.add_argument("--vol-max-mult", type=float, default=1.25)
    ap.add_argument("--file-threshold", type=float, default=20.0)
    ap.add_argument("--no-drift", action="store_true")
    ap.add_argument("--no-soft-triggers", action="store_true")
    ap.add_argument("--no-volume", action="store_true")
    ap.add_argument("--no-per-file-floor", action="store_true")
    ap.add_argument("--no-small-file-evidence", action="store_true")
    ap.add_argument("--small-file-blocks", type=int, default=150)
    ap.add_argument("--allow-mixed-calibration", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    if not (0.0 < args.mask_ratio < 1.0):
        ap.error("--mask-ratio must be between 0 and 1")
    if abs(args.w_recon + args.w_mahal - 1.0) > 1e-8:
        ap.error("--w-recon + --w-mahal must equal 1.0")
    if args.vol_max_mult < args.vol_high_mult:
        ap.error("--vol-max-mult must be >= --vol-high-mult")

    mod = load_framework(args.framework)
    train_files = unique_realpaths(read_manifest(args.train_files))
    calib_files = unique_realpaths(read_manifest(args.calib_files))
    test_files = unique_realpaths(read_manifest(args.test_files))
    role_sets = {"train": set(train_files), "calibration": set(calib_files), "test": set(test_files)}
    for a, b in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        overlap = role_sets[a] & role_sets[b]
        if overlap:
            raise RuntimeError(f"Manifest leakage: {a} and {b} share files: {sorted(overlap)}")

    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    import torch
    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device("cpu") if args.device != "cuda" else torch.device("cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    cfg = mod.PAADUConfigV7(
        epochs=args.epochs, batch_size=args.batch_size, embed_dim=args.embed_dim,
        mask_ratio=args.mask_ratio, use_gnn=not args.no_gnn,
        window_seconds=args.window_seconds, train_nrows_per_file=args.train_nrows,
        nrows_test=args.test_nrows, w_recon=args.w_recon, w_mahal=args.w_mahal,
        percentile=args.percentile, hybrid_floor_percentile=args.hybrid_floor_percentile,
        hard_z_floor=args.hard_z_floor,
        double_soft_z=args.double_soft_z,
        double_soft_relax=args.double_soft_relax,
        latent_only_z=args.latent_only_z,
        latent_only_relax=args.latent_only_relax,
        ema_alpha=args.ema_alpha,
        ema_multiplier=args.ema_multiplier,
        min_alert_severity=args.min_alert_severity,
        bucket_limit=args.bucket_limit,
        vol_high_mult=args.vol_high_mult, vol_max_mult=args.vol_max_mult,
        file_attack_threshold_pct=args.file_threshold,
        use_drift=not args.no_drift, soft_triggers=not args.no_soft_triggers,
        volume_adaptive=not args.no_volume,
        use_per_file_max_floor=not args.no_per_file_floor,
        small_file_evidence=not args.no_small_file_evidence,
        small_file_blocks=args.small_file_blocks, seed=args.seed,
        seeds=(args.seed,), output_dir=args.output_dir,
    )

    train_audit = audit_benign_manifest(mod, train_files, "training", args.allow_mixed_calibration)
    calib_audit = audit_benign_manifest(mod, calib_files, "calibration", args.allow_mixed_calibration)

    with tempfile.TemporaryDirectory(prefix="paadu_profile_") as td:
        profile_dir, _ = stage_profile_files(train_files, td)
        profile = mod.generate_profile(profile_dir, cfg)
    if profile is None:
        raise RuntimeError("Could not generate a profile from training files")
    profile.percentile = cfg.percentile
    profile.limit = cfg.bucket_limit

    print(mod.line("FINAL HELD-OUT EVALUATION"))
    print(f"  device={device} seed={args.seed} mask_ratio={cfg.mask_ratio}")
    print(f"  train files={len(train_files)} calibration files={len(calib_files)} test files={len(test_files)}")
    print(f"  use_gnn={cfg.use_gnn} drift={cfg.use_drift} soft={cfg.soft_triggers}")
    print(f"  resolved bucket_limit/profile.limit={cfg.bucket_limit}/{profile.limit}")
    print(f"  resolved vol_high_mult/vol_max_mult={cfg.vol_high_mult}/{cfg.vol_max_mult}")

    train_ds = mod.PADataset(
        train_files, profile, scaler=None, train_columns=None,
        nrows=args.train_nrows, benign_only=True,
        window_seconds=cfg.window_seconds, verbose=True,
    )
    calib_ds = mod.PADataset(
        calib_files, profile, scaler=train_ds.scaler,
        train_columns=train_ds.train_columns, nrows=args.calib_nrows,
        benign_only=True, window_seconds=cfg.window_seconds, verbose=True,
    )
    if len(train_ds) < 2 or len(calib_ds) < 1:
        raise RuntimeError(f"Insufficient blocks: train={len(train_ds)} calibration={len(calib_ds)}")

    n_val = max(1, int(len(train_ds) * 0.20))
    n_tr = max(1, len(train_ds) - n_val)
    train_sub, val_sub = mod.random_split(
        train_ds, [n_tr, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = mod.DataLoader(train_sub, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = mod.DataLoader(val_sub, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = mod.SemanticAnomalyMAE(
        profile.num_nodes, profile.seq_len, train_ds.features.shape[1],
        embed_dim=cfg.embed_dim, mask_ratio=cfg.mask_ratio, use_gnn=cfg.use_gnn,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    train_log, val_log, stopped = mod.train_with_early_stopping(
        model, optimizer, scheduler, train_loader, val_loader, device,
        cfg.epochs, cfg.early_stop_patience, args.output_dir,
    )

    # Calibrate ONLY on the explicit calibration manifest.
    old_train_nrows = cfg.train_nrows_per_file
    cfg.train_nrows_per_file = args.calib_nrows
    threshold, _, mahal, mse_stats, dm_stats, relaxer = mod.calibrate_threshold_v7(
        model, calib_ds, calib_files, profile, device, cfg, args.output_dir
    )
    cfg.train_nrows_per_file = old_train_nrows

    evaluator = make_final_evaluator(
        mod, model, profile, train_ds, device, threshold,
        mahal, mse_stats, dm_stats, relaxer, cfg,
    )

    file_rows, block_rows = [], []
    prediction_dir = Path(args.output_dir) / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for fp in test_files:
        res = evaluator.evaluate_file(fp, nrows=args.test_nrows, benign_only=False)
        if res is None:
            print(f"[SKIP] {fp}")
            continue
        df = res["df"].copy()
        df.insert(0, "file", res["file"])
        df.insert(1, "source_path", res["path"])
        df.to_csv(prediction_dir / f"predictions_{mod._safe_name(res['file'])}.csv", index=False)
        attack = int(any(is_attack_label(mod, x) for x in df["label"]))
        pred = int(res["file_score_pct"] >= cfg.file_attack_threshold_pct)
        file_rows.append({
            "file": res["file"], "path": res["path"], "is_attack": attack,
            "predicted_attack": pred, "blocks": int(res["blocks"]),
            "alerts": int(res["alerts"]), "alert_pct": float(res["alert_pct"]),
            "fires_pct": float(res["fires_pct"]), "candidate_pct": float(res["candidate_pct"]),
            "file_score_pct": float(res["file_score_pct"]), "file_basis": res["file_basis"],
            "z_threshold": float(res["z_thresh"]), "elapsed_sec": float(res["elapsed"]),
            "file_tp": int(attack == 1 and pred == 1),
            "file_fp": int(attack == 0 and pred == 1),
            "file_tn": int(attack == 0 and pred == 0),
            "file_fn": int(attack == 1 and pred == 0),
        })
        for _, row in df.iterrows():
            atk = int(is_attack_label(mod, row["label"]))
            block_rows.append({
                "file": res["file"], "path": res["path"], "is_attack": atk,
                "label": row["label"], "alert": int(row["alert"]),
                "hybrid": float(row["hybrid"]), "effective_threshold": float(row["effective_threshold"]),
            })
        print(f"{res['file']:<45} score={res['file_score_pct']:7.2f}% blocks={res['blocks']:6d} "
              f"attack={attack} pred={pred}")

    if not file_rows:
        raise RuntimeError("No final test files produced predictions")
    file_df = pd.DataFrame(file_rows)
    block_df = pd.DataFrame(block_rows)
    file_df.to_csv(Path(args.output_dir) / "final_file_metrics.csv", index=False)
    block_df.to_csv(Path(args.output_dir) / "final_block_predictions.csv", index=False)

    benign = block_df[block_df.is_attack == 0]
    attacks = block_df[block_df.is_attack == 1]
    tp = int(file_df.file_tp.sum()); fp = int(file_df.file_fp.sum())
    tn = int(file_df.file_tn.sum()); fn = int(file_df.file_fn.sum())
    b_alert = int(benign.alert.sum()); b_n = int(len(benign))
    a_alert = int(attacks.alert.sum()); a_n = int(len(attacks))
    metrics = {
        "protocol": "explicit train/calibration/final-test file manifests",
        "mask_ratio": cfg.mask_ratio,
        "seed": args.seed,
        "device": str(device),
        "train_files": len(train_files), "calibration_files": len(calib_files),
        "final_test_files": len(test_files), "evaluated_files": len(file_df),
        "train_blocks": int(len(train_ds)), "calibration_blocks": int(len(calib_ds)),
        "final_benign_blocks": b_n, "final_attack_blocks": a_n,
        "block_benign_alerts": b_alert, "block_attack_alerts": a_alert,
        "block_fpr_pct": 100.0 * b_alert / max(1, b_n),
        "block_tpr_pct": 100.0 * a_alert / max(1, a_n),
        "file_tp": tp, "file_fp": fp, "file_tn": tn, "file_fn": fn,
        "file_f1": 2.0 * tp / max(1e-12, 2.0 * tp + fp + fn),
        "file_fpr_pct": 100.0 * fp / max(1, fp + tn),
        "file_tpr_pct": 100.0 * tp / max(1, tp + fn),
        "bootstrap_cluster_unit": "file",
        "block_fpr_bootstrap": bootstrap_rate(file_rows, "alerts", "blocks",
                                               lambda r: r["is_attack"] == 0,
                                               args.bootstrap, args.seed + 101),
        "block_tpr_bootstrap": bootstrap_rate(file_rows, "alerts", "blocks",
                                               lambda r: r["is_attack"] == 1,
                                               args.bootstrap, args.seed + 102),
        "file_f1_bootstrap": bootstrap_file_f1(file_rows, args.bootstrap, args.seed + 103),
        "calibration_data_attack_rows": int(sum(x.get("attack_like_rows_in_sample", 0)
                                                 for x in calib_audit)),
    }
    json.dump(jsonable(metrics), open(Path(args.output_dir) / "final_metrics.json", "w"), indent=2)

    manifests = {
        "train": [{"path": p, "sha256": file_sha256(p)} for p in train_files],
        "calibration": [{"path": p, "sha256": file_sha256(p)} for p in calib_files],
        "final_test": [{"path": p, "sha256": file_sha256(p)} for p in test_files],
    }
    config = {
        "framework": os.path.realpath(args.framework),
        "framework_sha256": file_sha256(args.framework),
        "args": vars(args),
        "resolved_config": jsonable(cfg.__dict__),
        "resolved_profile": jsonable({
            "name": profile.name, "num_nodes": profile.num_nodes, "seq_len": profile.seq_len,
            "feature_dim": train_ds.features.shape[1], "transport": profile.transport,
            "schema": profile.schema, "base_leak": profile.base_leak, "limit": profile.limit,
        }),
        "runtime": {
            "device": str(device), "resolved_bucket_limit": cfg.bucket_limit,
            "resolved_profile_limit": profile.limit,
            "resolved_vol_high_mult": cfg.vol_high_mult,
            "resolved_vol_max_mult": cfg.vol_max_mult,
            "threshold_recon": threshold, "z_floor": relaxer.z_floor,
            "mse_stats": mse_stats, "dm_stats": dm_stats,
        },
        "manifests": manifests,
        "audit": {"train": train_audit, "calibration": calib_audit},
        "training": {"epochs_completed": stopped, "train_loss": train_log, "val_loss": val_log},
    }
    json.dump(jsonable(config), open(Path(args.output_dir) / "final_run_manifest.json", "w"), indent=2)
    torch.save(model.state_dict(), Path(args.output_dir) / "final_model_state.pt")

    print(mod.line("FINAL HELD-OUT RESULTS"))
    print(json.dumps(jsonable(metrics), indent=2))
    print(f"\nOutputs written to: {args.output_dir}")
    print("  final_metrics.json")
    print("  final_run_manifest.json")
    print("  final_file_metrics.csv")
    print("  final_block_predictions.csv")
    print("  predictions/")


if __name__ == "__main__":
    main()

