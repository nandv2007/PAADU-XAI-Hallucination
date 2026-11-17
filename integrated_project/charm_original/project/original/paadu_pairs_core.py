
#  paadu_pairs_core.py — shared machinery for per-source 3-pair

import os, sys, subprocess, json, ast, time, glob
import numpy as np
import pandas as pd

# ---------------- overridable globals (runners set these) ---------------
SCRIPT      = "/kaggle/working/paadu_benchmark_v7.py"
ROOT        = "/kaggle/working"
EPOCHS      = 15
W_RECON     = 0.8          # cross-domain weights (transfer-robust)
W_MAHAL     = 0.2
TEST_NROWS  = 30000
CICIOV_NROWS = 15000       # CAN files are wide; cap harder
MIN_OVERLAP = 3            # below this -> statistical bridge
WIN         = 64           # bridge window = 64 rows

BRIDGE_FEATS = ["n_rows", "n_cols", "mean_of_means", "std_of_means",
                "mean_of_stds", "mean_of_mins", "mean_of_maxs",
                "mean_zero_frac", "p25", "p50", "p75", "p90", "p99",
                "gmax", "gmean", "gstd", "frac_gt10", "frac_gt100",
                "rate_proxy", "cv_proxy"]

PATHS = {
    "cic2023":  "/kaggle/input/datasets/nirmalrajjayapal/cic-iot-2023",
    "cic2025":  "/kaggle/input/datasets/nirmalrajjayapal/cic-2025",
    "ciciotmt": "/kaggle/input/datasets/outstandard09/ciciomt2024",
    "ciciov":   "/kaggle/input/datasets/tusharchauhan1898/ciciov2024",
}

SRC     = None            # source dataset key, e.g. "cic2023"
TARGETS = None            # list of target keys, e.g. ["cic2025", "ciciotmt", "ciciov"]
OUT     = None            # output dir (default: ROOT/paadu_crossdomain_12/src_<SRC>)

# ---------------- overlap preflight --------------------------------------
META_KW = ("label", "src ip", "dst ip", "src port", "dst port", "timestamp",
           "time", "protocol", "can id", "flow id", "header")

def _numeric_cols(path, n=2000):
    first = sorted(glob.glob(os.path.join(path, "*.csv")))[:1]
    if not first:
        return set()
    df = pd.read_csv(first[0], nrows=n, low_memory=False)
    drop = [c for c in df.columns if any(k in str(c).lower() for k in META_KW)]
    return set(df.drop(columns=drop, errors="ignore")
                 .select_dtypes(include=[np.number]).columns)

# ---------------- statistical bridge -------------------------------------
def _num_series(s, name):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s2 = s.astype(str).str.strip()
    out = pd.to_numeric(s2, errors="coerce")
    is_payload = any(k in str(name).lower()
                     for k in ("data", "byte", "payload", "d0", "d1", "d2", "d3",
                               "d4", "d5", "d6", "d7"))
    if is_payload:  # hex CAN payloads (DATA_0..7), only when letters a-f present
        hexm = s2.str.contains(r"[a-fA-F]", na=False) & \
               s2.str.fullmatch(r"(0x)?[0-9a-fA-F]{1,8}")
        hx = s2[hexm].str.replace("0x", "", regex=False) \
                      .map(lambda h: float(int(h, 16)) if h else np.nan)
        out[hexm] = hx
    return out

def _label_col(df):
    for c in df.columns:
        if "label" in str(c).lower():
            return c
    return None

def _window_stats(X):
    flat = X.ravel()
    colm = X.mean(axis=0); colsd = X.std(axis=0)
    with np.errstate(all="ignore"):
        return {
            "n_rows": len(X), "n_cols": X.shape[1],
            "mean_of_means": float(colm.mean()), "std_of_means": float(colm.std()),
            "mean_of_stds": float(colsd.mean()),
            "mean_of_mins": float(X.min(axis=0).mean()),
            "mean_of_maxs": float(X.max(axis=0).mean()),
            "mean_zero_frac": float((X == 0).mean()),
            "p25": float(np.percentile(flat, 25)), "p50": float(np.percentile(flat, 50)),
            "p75": float(np.percentile(flat, 75)), "p90": float(np.percentile(flat, 90)),
            "p99": float(np.percentile(flat, 99)),
            "gmax": float(flat.max()), "gmean": float(flat.mean()), "gstd": float(flat.std()),
            "frac_gt10": float((flat > 10).mean()), "frac_gt100": float((flat > 100).mean()),
            "rate_proxy": float(np.abs(np.diff(flat[:1000])).mean()) if len(flat) > 1 else 0.0,
            "cv_proxy": float((colm / (colsd + 1e-9)).mean()),
        }

def _bridge_file(fp, outd, nrows):
    """Convert one raw file -> CSV of per-window bridge features + label."""
    df = pd.read_csv(fp, nrows=nrows, low_memory=False)
    lc = _label_col(df)
    labels = df[lc] if lc is not None else None
    num = pd.DataFrame({c: _num_series(df[c], c) for c in df.columns
                        if lc is None or c != lc})
    num = num.fillna(0.0)
    # numeric binary labels (0=benign, 1=attack) -> map to strings first
    numeric_label = False
    if labels is not None:
        num_lbl = pd.to_numeric(labels, errors="coerce")
        numeric_label = num_lbl.notna().mean() > 0.5 and \
                        set(num_lbl.dropna().unique()) <= {0.0, 1.0}
    rows = []
    n = len(num)
    for i in range(0, n, WIN):
        w = num.iloc[i:i + WIN]
        if len(w) < 8:
            continue
        r = _window_stats(w.values.astype(float))
        if labels is not None:
            wl = labels.iloc[i:i + WIN]
            if numeric_label:
                mode = pd.to_numeric(wl, errors="coerce").mode()
                lab = "benign" if len(mode) and mode.iloc[0] == 0 else "attack"
            else:
                mode = wl.astype(str).mode()
                lab = mode.iloc[0] if len(mode) else "attack"
                lab = ("benign" if any(k in lab.lower()
                       for k in ("benign", "normal", "legit")) else "attack")
        else:
            lab = "benign" if "benign" in os.path.basename(fp).lower() else "attack"
        r["label"] = lab
        rows.append(r)
    if not rows:
        return None
    # benign_ prefix so the benchmark's filename-based benign detection
    # accepts the file even when it is small (its label-column fallback
    # requires >=100 benign rows, which tiny bridge files cannot reach)
    if labels is not None:
        if numeric_label:
            file_benign = pd.to_numeric(labels, errors="coerce").mode().iloc[0] == 0
        else:
            mode = labels.astype(str).mode()
            file_benign = (len(mode) and any(k in mode.iloc[0].lower()
                           for k in ("benign", "normal", "legit")))
    else:
        file_benign = "benign" in os.path.basename(fp).lower()
    prefix = "benign_" if file_benign else ""
    out = pd.DataFrame(rows, columns=BRIDGE_FEATS + ["label"])
    outp = os.path.join(outd, prefix + os.path.splitext(os.path.basename(fp))[0] + "_bridge.csv")
    out.to_csv(outp, index=False)
    return len(out)

def build_bridge(a, b, nrows=30000):
    """Build shared-representation copies of both sides; return (src, tgt) dirs."""
    bd = os.path.join(ROOT, "paadu_bridge", f"{a}_to_{b}")
    src_out, tgt_out = os.path.join(bd, "source"), os.path.join(bd, "target")
    if os.path.isdir(src_out) and os.path.isdir(tgt_out) and \
       os.listdir(src_out) and os.listdir(tgt_out):
        print(f"  [bridge] cached at {bd}")
        return src_out, tgt_out
    os.makedirs(src_out, exist_ok=True); os.makedirs(tgt_out, exist_ok=True)
    for side, d, outd in (("src", PATHS[a], src_out), ("tgt", PATHS[b], tgt_out)):
        files = sorted(glob.glob(os.path.join(d, "*.csv")))
        for fp in files:
            nw = _bridge_file(fp, outd, nrows)
            print(f"  [bridge:{side}] {os.path.basename(fp)[:40]:<42} -> {nw} windows")
    return src_out, tgt_out

# ---------------- metrics (in-file labels first) --------------------------
def _parse_ls(v):
    if v is None or (isinstance(v, float) and v != v):
        return None
    try:
        return ast.literal_eval(v) if isinstance(v, str) else v
    except Exception:
        return None

def _file_is_attack(row):
    ls = _parse_ls(row.get("label_stats"))
    if ls:
        keys = [str(k).lower() for k in ls]
        nonben = [k for k in keys if not any(w in k for w in ("benign", "normal", "legit"))]
        if nonben and len(nonben) == len(keys):
            return True
        if not nonben and keys:
            return False
    return "benign" not in str(row["file"]).lower()

def summary_metrics(sp, thr=20.0):
    s = pd.read_csv(sp)
    y = s.apply(_file_is_attack, axis=1).astype(int).values
    pr = (s["file_score_pct"] >= thr).astype(int).values
    tp = int(((y == 1) & (pr == 1)).sum()); fp = int(((y == 0) & (pr == 1)).sum())
    tn = int(((y == 0) & (pr == 0)).sum()); fn = int(((y == 1) & (pr == 0)).sum())
    f1 = 2 * tp / max(1e-9, 2 * tp + fp + fn)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "file_f1": round(f1, 4),
            "n_files": len(s), "n_benign_gt": int((y == 0).sum())}

# ---------------- single pair run -----------------------------------------
def run_one(a, b, proto, extra, train_dir, test_dir, ov):
    out_dir = os.path.join(OUT, f"{a}_to_{b}__{proto}")
    done = os.path.join(out_dir, "DONE.json")
    logp = os.path.join(out_dir, "run.log")
    if os.path.exists(done):
        return json.load(open(done)), True
    os.makedirs(out_dir, exist_ok=True)
    tn = CICIOV_NROWS if (a == "ciciov" or b == "ciciov") else TEST_NROWS
    cmd = [sys.executable, SCRIPT, "--train-dir", train_dir, "--test-dir", test_dir,
           "--output-dir", out_dir, "--epochs", str(EPOCHS),
           "--w-recon", str(W_RECON), "--w-mahal", str(W_MAHAL),
           "--test-nrows", str(tn)] + extra
    for attempt in (1, 2):
        t0 = time.time()
        with open(logp, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            proc.wait()
        sp = os.path.join(out_dir, "paadu_summary.csv")
        if proc.returncode == 0 and os.path.exists(sp):
            m = summary_metrics(sp)
            m.update({"src": a, "tgt": b, "status": "ok", "attempt": attempt,
                      "elapsed_s": round(time.time() - t0, 1), "overlap": ov})
            json.dump(m, open(done, "w"))
            return m, True
        if attempt == 1:
            print(f"  [retry] {a}->{b} [{proto}] attempt 1 failed rc={proc.returncode}")
    tail = [l for l in open(logp).read().splitlines() if l.strip()][-15:]
    return {"src": a, "tgt": b, "proto": proto, "status": "FAILED",
            "overlap": ov, "error": " | ".join(tail)[:400]}, False

# ---------------- main entry -----------------------------------------------
def run():
    global OUT
    assert SRC is not None and TARGETS, "set core.SRC and core.TARGETS first"
    OUT = OUT or os.path.join(ROOT, "paadu_crossdomain_12", f"src_{SRC}")

    missing = [k for k, p in PATHS.items() if not os.path.isdir(p)]
    if missing:
        print("!! MISSING DATASETS — add via 'Add Input':", {k: PATHS[k] for k in missing})
        raise SystemExit(1)
    assert "v7.9" in open(SCRIPT).read(), "STALE benchmark script (need v7.9)"
    os.makedirs(OUT, exist_ok=True)

    print(f"=== cross-domain runner | SRC={SRC} | TARGETS={TARGETS} ===")
    feat = {k: _numeric_cols(p) for k, p in PATHS.items()}
    for b in TARGETS:
        ov = len(feat[SRC] & feat[b])
        print(f"  {SRC:8s} -> {b:8s}  overlap={ov:3d}  -> "
              f"{'STAT-BRIDGE' if ov < MIN_OVERLAP else 'standard cross-domain'}")

    rows = []
    for b in TARGETS:
        a = SRC
        ov = len(feat[a] & feat[b])
        if ov < MIN_OVERLAP:
            src_d, tgt_d = build_bridge(a, b, nrows=TEST_NROWS)
            protos = [("stat_bridge", []),
                      ("stat_bridge_target_calib", ["--target-calib"])]
        else:
            src_d, tgt_d = PATHS[a], PATHS[b]
            protos = [("source_only", []), ("target_calib", ["--target-calib"])]
        for proto, extra in protos:
            m, ok = run_one(a, b, proto, extra, src_d, tgt_d, ov)
            rows.append({**m, "proto": proto})
            print(f"[{'OK ' if ok else 'FAIL'}] {a}->{b} [{proto:26s}] "
                  f"F1={m.get('file_f1', '—')} TP={m.get('tp', '—')} "
                  f"FP={m.get('fp', '—')} {m.get('error', '')[:120]}")

    df = pd.DataFrame(rows)
    outp = os.path.join(OUT, f"cross_domain_{SRC}_src.csv")
    df.to_csv(outp, index=False)
    print("\n=== report ===")
    print(df.reindex(columns=["src", "tgt", "proto", "status", "overlap", "tp", "fp",
                              "tn", "fn", "file_f1"]).to_string())
    print("\nSaved:", outp)
    print("\nNOTE: stat_bridge rows used a shared per-window statistical descriptor;")
    print("report them as a separate protocol in the paper, not as raw transfer.")