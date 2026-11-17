#!/usr/bin/env python3


import os
import re
import sys
import time
import math
import json
import random
import warnings
import argparse
import glob
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    HAS_VIZ = True
except Exception:  # pragma: no cover
    HAS_VIZ = False

from numpy.linalg import pinv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
#  0. CONFIG + REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class PAADUConfigV7:
    # data
    sample_rows_profile: int = 10_000
    target_seq_len: int = 10
    window_seconds: float = 0.0        # >0 + timestamp column -> time windows
    # training
    epochs: int = 30
    batch_size: int = 32
    early_stop_patience: int = 6
    lr: float = 1e-3
    embed_dim: int = 64
    mask_ratio: float = 0.40
    use_gnn: bool = True
    # data caps
    train_nrows_per_file: int | None = None
    nrows_test: int = 30_000
    max_train_files: int | None = None
    # holdout benign validation
    holdout_benign_eval: bool = False
    benign_train_fraction: float = 0.70
    # hybrid scoring
    w_recon: float = 0.20
    w_mahal: float = 0.80
    noise_floor_std: float = 0.015
    percentile: float = 99.0
    hybrid_floor_percentile: float = 99.0
    hard_z_floor: float = 2.5
    use_per_file_max_floor: bool = True
    volume_adaptive: bool = True        # scale floor by file volume
    soft_triggers: bool = True          # relax threshold on double/latent evidence
    use_drift: bool = True              # non-sticky drift tracker (burst+EMA)
    # volume adaptive
    vol_low_rows: int = 5_000
    vol_high_rows: int = 100_000
    vol_low_mult: float = 1.00
    vol_high_mult: float = 1.20
    vol_max_mult: float = 1.10
    # soft triggers
    double_soft_z: float = 2.5
    double_soft_relax: float = 0.60
    latent_only_z: float = 3.0
    latent_only_relax: float = 0.70
    # drift tracker
    base_leak_default: float = 0.06
    bucket_limit: float = 3.2
    min_alert_severity: float = 0.05
    ema_alpha: float = 0.05
    ema_multiplier: float = 6.5
    drift_requires_current_severity: bool = True
    # file-level decision
    file_attack_threshold_pct: float = 20.0
    use_file_evidence_rule: bool = False
    file_candidate_relax: float = 0.85
    file_candidate_soft_z: float = 2.5
    small_file_evidence: bool = True      # short files: accept fires/candidate evidence
    small_file_blocks: int = 150          # blocks below which a file is "short"
    run_baselines: bool = False           # train AE/LSTM-AE/Transformer-AE/IF
    seeds: list = (42,)                   # multiple seeds: per-seed metrics + mean/std
    dump_predictions: bool = False        # per-file, per-window score/label/alert CSV
    split_manifest: bool = False          # write train/calib/test row manifests per file
    calib_ablation: bool = False          # sweep calibration components + percentiles
    calib_baselines: bool = False         # standard calibration baselines (fixed-z, global-only, per-file-only)
    mixed_stream: bool = False            # concatenate files into one stream (temporal-drift eval)
    baseline_tune: bool = False           # small hyperparameter grid for neural baselines
    verify_metrics: bool = False          # recompute metrics from per-file predictions and cross-check
    target_calib: bool = False            # cross-domain: re-calibrate floor on target benign files
    baselines_only: bool = False         # skip PAADU train/calib/eval entirely
    baseline_epochs: int = 15
    # misc
    seed: int = 42
    max_label_lines_per_file: int = 20
    output_dir: str = "paadu_v7_outputs"


# ══════════════════════════════════════════════════════════════════════════════
#  1. SCHEMA DETECTION  (dataset-agnostic)
# ══════════════════════════════════════════════════════════════════════════════


def _norm(col) -> str:
    return str(col).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_")


def _tokens(col) -> set:
    return set(t for t in re.split(r"[_\W]+", _norm(col)) if t)


# label-ish column names (normalized). Order matters: longer/more specific
# first so "attack_type" is not confused with "type".
LABEL_NAME_HINTS = [
    "attack_type", "attacktype", "attack_category", "attackclass", "attack_name",
    "attackname", "attack", "attacks", "specificclass", "specific_class",
    "subclass", "isattack", "is_attack", "malicious", "benign", "verdict",
    "groundtruth", "ground_truth", "label", "labels", "class", "classes",
    "category", "categories", "cat", "target", "targets",
]

ATTACK_VALUE_HINTS = (
    "ddos", "dos", "flood", "scan", "spoof", "mitm", "malware", "brute",
    "recon", "xss", "inject", "hijack", "ransom", "probe", "exfil", "attack",
    "exploit", "phish", "backdoor", "zeroday", "web", "sqli", "upload",
    "vuln", "command", "greip", "greeth", "udpplain", "slowloris", "ping",
    "synonymous", "fragmentation", "pshack", "rstfin", "ack_flood", "icmp",
    "scanning", "ransomware", "password", "injection", "trojan", "worm",
    "shell", "crypto", "heartbleed", "gafgyt", "torii", "service",
)

BENIGN_VALUE_HINTS = ("benign", "normal", "legitimate", "clean", "regular")


def _label_base(n: str) -> str:
   
    m = re.match(r"^(.*?)[._]?\d+$", n)
    return m.group(1) if m else n


def _is_label_name(n: str) -> bool:
    base = _label_base(n)
    return base in LABEL_NAME_HINTS or base.endswith("_label") or \
        base.endswith("_class") or base.startswith("label") or \
        base.startswith("class")


def _value_looks_label(series, max_unique: int = 200) -> bool:
  
    if series is None:
        return False
    vals = series.dropna().astype(str).str.lower()
    if len(vals) == 0:
        return False
    if vals.nunique() < 2 or vals.nunique() > max_unique:
        return False
    ben = vals.str.contains("|".join(BENIGN_VALUE_HINTS)).mean()
    att = vals.str.contains("|".join(ATTACK_VALUE_HINTS)).mean()
    if ben >= 0.3 and ben + att >= 0.5:
        return True
    if att >= 0.6:
        return True
    return False


def detect_schema(df: pd.DataFrame, filename: str = "") -> dict:
    """Auto-classify columns for ANY dataset family."""
    cols = list(df.columns)
    norm = {c: _norm(c) for c in cols}

    def has(c, *kws):
        n = norm[c]
        return any(k in n for k in kws)

    label_col, src_col, dst_col, ts_col, proto_col = None, None, None, None, None
    can_payload = []

    for c in cols:
        n = norm[c]
        
        if _is_label_name(n):
            if label_col is None:
                label_col = c
        # source endpoint
        if any(k in n for k in ("srcip", "src_ip", "sourceip", "source_ip", "srcaddr",
                                "src_addr", "source", "src", "address")) and \
                not any(k in n for k in ("dst", "dest", "peer")):
            if src_col is None and not _looks_like_feature(n):
                src_col = c
        # destination endpoint
        if any(k in n for k in ("dstip", "dst_ip", "destip", "dest_ip", "dstaddr",
                                "dst_addr", "destaddr", "destination", "dst",
                                "peer", "peer_address", "peeraddr")):
            if dst_col is None and not _looks_like_feature(n):
                dst_col = c
        # timestamp (skip protocol-specific sub-fields like icmp.transmit_timestamp)
        if any(k in n for k in ("timestamp", "time_stamp", "datetime", "date_time")) or \
                n in ("time", "ts", "date"):
            if ts_col is None and not any(p in n for p in ("icmp", "tcp", "udp", "http")):
                ts_col = c
        # protocol
        if any(k in n for k in ("protocol", "proto", "ipproto", "transport_proto",
                                "transportproto", "protocolname")):
            if proto_col is None:
                proto_col = c

    
    if label_col is None:
        for c in cols:
            if c in (src_col, dst_col, ts_col, proto_col):
                continue
            if c in can_payload:
                continue
            try:
                if _value_looks_label(df[c]):
                    label_col = c
                    break
            except Exception:  # noqa: BLE001
                continue

    # CAN bus payload columns: DATA_0..DATA_7
    for c in cols:
        if re.match(r"^data[_ ]?\d+$", _norm(c)):
            can_payload.append(c)
    # CAN id columns -> also the communication source endpoint
    can_id_col = next((c for c in cols if _norm(c) in ("canid", "can_id", "arbitrationid", "arbitration_id")), None)
    if can_id_col is not None and src_col is None:
        src_col = can_id_col

    # transport medium from filename / columns (TOKEN match: 'Recon-OSScan'
    # must not match 'can' via the substring 'scan')
    transport = None
    low_fn = filename.lower()
    fn_tokens = set(t for t in re.split(r"[^a-z0-9]+", low_fn) if t)
    if "mqtt" in fn_tokens or any("mqtt" in n for n in norm.values()):
        transport = "MQTT"
    elif "ble" in fn_tokens or "bluetooth" in fn_tokens:
        transport = "BLE"
    elif "can" in fn_tokens or can_id_col is not None:
        transport = "CAN"
    elif "wifi" in fn_tokens:
        transport = "WIFI"

    return {
        "label_col": label_col, "src_col": src_col, "dst_col": dst_col,
        "ts_col": ts_col, "proto_col": proto_col, "can_payload": can_payload,
        "can_id_col": can_id_col, "transport": transport,
    }


def _looks_like_feature(n: str) -> bool:
   
    return any(k in n for k in ("port", "mac", "count", "len", "bytes", "pkts",
                                "rate", "flags", "id", "idx"))


def is_metadata_column(col: str, label_col=None) -> bool:
   
    if label_col is not None and col == label_col:
        return True
    n = _norm(col)
    if n.startswith("__can_id") or n.startswith("__can_payload") or \
            n.startswith("__entropy") or n.startswith("__iat"):
        return False  # derived behavioural features are KEPT
    # leakage guard: a label/class/attack-named column is NEVER a traffic
    # feature, even when label detection failed (numeric attack-type codes
    # would otherwise leak ground truth into training)
    if _is_label_name(n):
        return True
    toks = _tokens(col)
    if n.startswith("unnamed"):
        return True
    if n in {"index", "idx", "id", "uid", "uuid", "row_id", "record_id",
             "flow_id", "flowid", "session_id", "conn_id", "connection_id"}:
        return True
    if n.endswith("_id") or n.endswith("_uid"):
        return True
    if n in {"time", "ts", "date", "datetime", "date_time"} or "timestamp" in n:
        return True
    address_terms = {"src", "source", "dst", "dest", "destination", "sport",
                     "dport", "srcport", "dstport", "port", "ports", "ip",
                     "ipv4", "ipv6", "mac", "addr", "address"}
    if toks & address_terms:
        return True
    for p in ["src_ip", "dst_ip", "source_ip", "destination_ip", "srcip", "dstip",
              "ip_src", "ip_dst", "src_port", "dst_port", "source_port",
              "destination_port", "sport", "dport", "mac_src", "mac_dst",
              "canid", "can_id", "arbitrationid"]:
        if p in n:
            return True
    if "port" in n:
        return True
    return False


def drop_metadata_columns(df: pd.DataFrame, label_col=None, verbose: bool = False) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if is_metadata_column(c, label_col=label_col)]
    if verbose and drop_cols:
        show = ", ".join(map(str, drop_cols[:12]))
        print(f"  [preprocess] dropping {len(drop_cols)} metadata/label columns: {show}"
              f"{'…' if len(drop_cols) > 12 else ''}")
    return df.drop(columns=drop_cols, errors="ignore")


def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(index=getattr(df, "index", None))
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def parse_can_payloads(df: pd.DataFrame, schema: dict) -> pd.DataFrame:

    if not schema["can_payload"]:
        return df
    df = df.copy()
    payload = pd.DataFrame(index=df.index)
    for c in schema["can_payload"]:
        payload[c] = df[c].map(_hex_to_int)
    if len(schema["can_payload"]) >= 2:
        df["__can_payload_total"] = payload.sum(axis=1)
        df["__can_payload_mean"] = payload.mean(axis=1)
        df["__can_payload_std"] = payload.std(axis=1).fillna(0)
        df["__can_payload_max"] = payload.max(axis=1)
    return df


def _hex_to_int(v):
    try:
        s = str(v).strip()
        if s.lower().startswith("0x"):
            return float(int(s, 16))
        if re.match(r"^[0-9a-fA-F]+$", s) and len(s) <= 8:
            return float(int(s, 16))
        return float(s)
    except Exception:  # noqa: BLE001
        return np.nan


def log_squash(df: pd.DataFrame, keywords: list) -> pd.DataFrame:
    cols = [c for c in df.columns if any(k in str(c).lower() for k in keywords)]
    if not cols:
        return df
    df = df.copy()
    df[cols] = np.log1p(df[cols].clip(lower=0))
    return df


def add_entropy_features(df: pd.DataFrame) -> pd.DataFrame:
    byte_cols = [c for c in df.columns if any(k in str(c).lower()
                  for k in ["byte", "payload", "len", "size", "pkt", "packet"])]
    iat_cols = [c for c in df.columns if "iat" in str(c).lower() or "inter" in str(c).lower()]
    df = df.copy()
    if len(byte_cols) >= 2:
        bv = df[byte_cols].clip(lower=0)
        rm = bv.mean(axis=1).replace(0, 1e-9)
        rs = bv.std(axis=1).fillna(0)
        df["__entropy_proxy"] = (rs / rm).fillna(0).clip(0, 20)
    if len(iat_cols) >= 2:
        df["__iat_variance"] = df[iat_cols].var(axis=1).fillna(0)
    return df


def extract_behavioral_numeric(df: pd.DataFrame, schema: dict,
                               log_keywords: list, verbose: bool = False) -> pd.DataFrame:

    # CAN-ID statistics need the raw ID column (dropped later as metadata)
    cleaned = drop_metadata_columns(df, label_col=schema["label_col"], verbose=verbose)
    cleaned = parse_can_payloads(cleaned, schema)
    num_df = cleaned.select_dtypes(include=[np.number])
    num_df = safe_numeric(num_df)
    num_df = log_squash(num_df, log_keywords)
    num_df = add_entropy_features(num_df)
    return safe_numeric(num_df)


def add_can_id_features(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
   
    c = schema.get("can_id_col")
    if c is None or c not in df.columns:
        return df
    df = df.copy()
    ids = df[c].astype(str)
    vc = ids.value_counts(dropna=True)
    if len(vc) == 0:
        return df
    share = vc.iloc[0] / max(1, len(ids))
    counts = ids.value_counts(dropna=True).to_dict()
    top_count = float(vc.iloc[0])
    p = (vc / max(1, len(ids))).values
    ent = float(-(p * np.log2(p + 1e-12)).sum())
    df["__can_id_unique"] = float(ids.nunique(dropna=True))
    df["__can_id_top_share"] = float(share)
    df["__can_id_top_count"] = top_count
    df["__can_id_entropy"] = ent
    df["__can_id_msg_rate"] = float(len(ids)) / max(1e-6, len(df))
    return df


def find_benign_keywords(df: pd.DataFrame, schema: dict, filename: str = "") -> list:
    
    if schema["label_col"] is not None and schema["label_col"] in df.columns:
        vals = df[schema["label_col"]].astype(str).str.strip().str.lower().unique().tolist()
        # binary-encoded labels (0=benign, 1=attack) — EXACT match only
        if set(vals) <= {"0", "1", "0.0", "1.0"}:
            return ["0"]
        has_benign = any(v == "benign" or v.startswith("benign") for v in vals)
        has_normal = any(v in ("normal", "normal traffic") for v in vals)
        if has_normal and not has_benign:
            return ["normal"]
        if has_benign:
            return ["benign"]
    low = filename.lower()
    if "normal" in low and "benign" not in low:
        return ["normal"]
    return ["benign"]


def is_benign_file(filename: str, benign_keywords: list, df_label=None) -> bool:
    f = os.path.basename(filename).lower()
    if any(k in f for k in benign_keywords) or f in ("normal.csv", "benign.csv"):
        return True
    if df_label is not None and len(df_label) > 0:
        v = str(df_label.dropna().astype(str).mode().iloc[0]).lower()
        return any(k in v for k in benign_keywords)
    return False


def _file_stem(filename: str) -> str:
   
    name = os.path.basename(filename)
    low = name.lower()
    for ext in (".pcap.csv", ".pcap", ".csv"):
        if low.endswith(ext):
            return name[: -len(ext)]
    return name


def filename_label(filename: str) -> str:
    
    stem = _file_stem(filename)
    low = stem.lower()
    if "benign" in low or low in ("normal", "legitimate"):
        return "Benign"
    return stem


def _count_benign_rows(series, bkw) -> int:
   
    if bkw == ["0"]:
        # numeric binary/multi-class: benign is exactly 0
        return int(pd.to_numeric(series, errors="coerce").fillna(-1).eq(0).sum())
    return int(series.astype(str).str.lower().apply(
        lambda v: any(k in v for k in bkw)).sum())


def _benign_mask(series, bkw):
    if bkw == ["0"]:
        return pd.to_numeric(series, errors="coerce").fillna(-1).eq(0)
    return series.astype(str).str.lower().apply(
        lambda v: any(k in v for k in bkw))


def _row_is_attack(label) -> bool:
    """Row-level ground truth from a label VALUE (mixed-file datasets)."""
    v = str(label).strip().lower()
    if v in ("0", "0.0"):
        return False
    if v in ("1", "1.0"):
        return True
    return not (v == "benign" or v.startswith("benign") or
                v in ("normal", "normal traffic", "legitimate", "clean"))


def attack_family(filename: str) -> str:
    """Aggregate a file name into a coarse attack family (paper table, spec 42)."""
    stem = _file_stem(filename)
    low = stem.lower().strip()
    if "benign" in low or low == "normal":
        return "Benign"
    
    for enc in ("binary_", "decimal_", "hexadecimal_", "hex_", "bin_", "dec_"):
        if low.startswith(enc):
            low = low[len(enc):]
            break
   
    for pref in ("tcp_ip_", "tcp_ip-", "tcp_", "tcp-", "udp_", "udp-",
                 "icmp_", "icmp-", "mqtt_", "mqtt-", "http_", "http-",
                 "arp_", "arp-", "ip_"):
        if low.startswith(pref):
            low = low[len(pref):]
            break
    if low.startswith("attack"):
        return "Attack"          # generic attack_samples_* layout (CIC-2025)
    if low.startswith("ddos"):
        return "DDoS"
    if low.startswith("dos"):
        return "DoS"
    if low.startswith("mirai"):
        return "Mirai"
    if low.startswith("recon") or low.startswith("vulnerability"):
        return "Recon"
    if low.startswith("spoofing") or low.startswith("mitm") or "arpspoof" in low \
            or low.startswith("dns") or low.startswith("ip-spoof") or low.startswith("arp"):
        return "Spoofing"
    if "bruteforce" in low or low.startswith("dictionary"):
        return "BruteForce"
    if low in ("xss", "sqlinjection", "commandinjection", "browserhijacking",
               "uploading_attack"):
        return "Web"
    if "malware" in low or low.startswith("backdoor"):
        return "Malware"
    # CICIoMT2024 filename families (MQTT/BLE floods are DoS-class)
    if low.startswith("mqtt") or low.startswith("ble") or "spam" in low:
        return "DoS"
    if low.startswith("data") or "exfil" in low:
        return "DataTheft"
    return stem


def _family_of_label(label) -> str:
   
    low = str(label).lower()
    if "benign" in low or low == "normal" or low == "legitimate":
        return "Benign"
    rest = low
    for pre in ("attack_", "attack-", "attack"):
        if rest.startswith(pre):
            rest = rest[len(pre):]
            break
    first = re.split(r"[_\-]+", rest)[0] if rest else ""
    mapping = {
        "ddos": "DDoS", "dos": "DoS", "recon": "Recon", "scan": "Recon",
        "web": "Web", "malware": "Malware", "mirai": "Malware",
        "backdoor": "Malware", "mitm": "MITM", "spoofing": "Spoofing",
        "dns": "Spoofing", "arp": "Spoofing", "ipspoofing": "Spoofing",
        "bruteforce": "BruteForce", "brute": "BruteForce",
        "dictionary": "BruteForce", "xss": "Web", "sql": "Web",
        "sqlinjection": "Web", "command": "Web", "commandinjection": "Web",
        "browserhijacking": "Web", "uploading": "Web", "upload": "Web",
        "vuln": "Recon", "ping": "Recon", "host": "Recon", "os": "Recon",
        "port": "Recon", "syn": "DDoS", "udp": "DDoS", "tcp": "DDoS",
        "icmp": "DDoS", "slowloris": "DDoS", "pshack": "DDoS",
        "rstfin": "DDoS", "fragmentation": "DDoS", "synonymousip": "DDoS",
        "greip": "Malware", "greeth": "Malware", "udpplain": "Malware",
        # CICIoMT2024 filename families
        "mqtt": "DoS", "ble": "DoS", "spam": "DoS",
        "data": "DataTheft", "exfil": "DataTheft", "theft": "DataTheft",
        # TON_IoT-V3 'type' values
        "scanning": "Recon", "ransomware": "Malware", "password": "BruteForce",
        "injection": "Web", "trojan": "Malware", "worm": "Malware",
        "backdoor": "Malware", "shell": "Web", "crypto": "Malware",
        "heartbleed": "Recon", "gafgyt": "Malware", "torii": "Malware",
        "normal": "Benign",
    }
    if first in mapping:
        return mapping[first]
    for kw, fam in [("mirai", "Malware"), ("spoof", "Spoofing"),
                    ("flood", "DDoS"), ("scan", "Recon"), ("inject", "Web"),
                    ("brute", "BruteForce"), ("malware", "Malware"),
                    ("backdoor", "Malware"), ("hijack", "Web")]:
        if kw in low:
            return fam
    return "Attack"


def count_csv_data_rows(csv_path: str) -> int:
    with open(csv_path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


def read_csv_fraction(csv_path: str, nrows=None, row_fraction=None, low_memory=False):
    if row_fraction is None:
        return pd.read_csv(csv_path, nrows=nrows, low_memory=low_memory), None
    start_frac, end_frac = row_fraction
    start_frac = float(np.clip(start_frac, 0.0, 1.0))
    end_frac = float(np.clip(end_frac, 0.0, 1.0))
    n_full = count_csv_data_rows(csv_path)
    s_idx = int(math.floor(n_full * start_frac))
    e_idx = n_full if end_frac >= 1.0 else int(math.floor(n_full * end_frac))
    split_rows = max(0, e_idx - s_idx)
    read_n = split_rows if nrows is None else min(int(nrows), split_rows)
    if read_n <= 0:
        df = pd.read_csv(csv_path, nrows=0, low_memory=low_memory)
    else:
        df = pd.read_csv(csv_path, skiprows=range(1, s_idx + 1), nrows=read_n,
                         low_memory=low_memory)
    return df, {"n_full": n_full, "start_idx": s_idx, "end_idx": e_idx,
                "start_frac": start_frac, "end_frac": end_frac}


# ══════════════════════════════════════════════════════════════════════════════
#  2. PROFILE  (data-driven, per dataset)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DatasetProfile:
    name: str
    num_nodes: int
    seq_len: int
    benign_keywords: list
    log_columns_keywords: list
    percentile: float = 99.0
    base_leak: float = 0.06
    limit: float = 4.5
    feature_dim: int = 0
    schema: dict = field(default_factory=dict)
    transport: str | None = None

    def __repr__(self):
        return (f"DatasetProfile(name={self.name!r}, num_nodes={self.num_nodes}, "
                f"seq_len={self.seq_len}, rows/block={self.num_nodes * self.seq_len}, "
                f"features={self.feature_dim}, transport={self.transport})")


def generate_profile(data_path: str, cfg: PAADUConfigV7) -> DatasetProfile:
    print("\n  ┌─ generate_profile (schema auto-detection) ────────────────┐")
    all_paths = sorted(p for p in glob.glob(os.path.join(data_path, "**", "*.csv"),
                                            recursive=True))
    all_files = [os.path.basename(p) for p in all_paths]
    if not all_paths:
        print("  [WARN] No CSV files found.")
        return None
    print(f"  │  Files found        : {len(all_paths)}")

    frames, schemas = [], {}
    for fp in all_paths:
        try:
            df = pd.read_csv(fp, nrows=cfg.sample_rows_profile, low_memory=False)
            frames.append(df)
            schemas[fp] = detect_schema(df, os.path.basename(fp))
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] Could not read {fp}: {e}")
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)

    # schema majority vote across files
    schema_keys = ["label_col", "src_col", "dst_col", "ts_col", "proto_col",
                   "can_id_col", "transport"]
    schema = {}
    for k in schema_keys:
        vals = [s.get(k) for s in schemas.values() if s.get(k)]
        schema[k] = max(set(vals), key=vals.count) if vals else None
    schema["can_payload"] = next((s["can_payload"] for s in schemas.values()
                                  if s["can_payload"]), [])
    print(f"  │  label col         : {schema['label_col']}")
    print(f"  │  src/dst col       : {schema['src_col']} / {schema['dst_col']}")
    print(f"  │  timestamp col     : {schema['ts_col']}")
    print(f"  │  CAN payload cols  : {len(schema['can_payload'])}"
          + (f"  (CAN ID: {schema['can_id_col']})" if schema["can_id_col"] else ""))
    print(f"  │  transport         : {schema['transport']}")

    benign_keywords = find_benign_keywords(combined, schema, all_files[0])
    print(f"  │  benign keywords   : {benign_keywords}")

    dataset_name = os.path.basename(data_path.rstrip("/\\")) or "Unknown"
    dir_str = data_path.lower()
    file_str = " ".join(all_files).lower()
    for year in ["2024", "2023", "2025"]:
        if year in dir_str or year in file_str:
            dataset_name = f"CICIoT{year}" if "iot" in dir_str + file_str else f"DS{year}"
            break

    cleaned = drop_metadata_columns(combined, label_col=schema["label_col"])
    cleaned = parse_can_payloads(cleaned, schema)
    num_df = safe_numeric(cleaned.select_dtypes(include=[np.number]))
    feat_dim = num_df.shape[1]
    print(f"  │  features after meta drop : {feat_dim}")
    if feat_dim < 2:
        print("  [WARN] Too few features.")
        return None

    seq_len = cfg.target_seq_len if feat_dim >= cfg.target_seq_len else max(2, feat_dim)
    num_nodes = max(2, min(int(math.ceil(feat_dim / seq_len)), 20))
    print(f"  │  feat_dim={feat_dim} -> num_nodes={num_nodes}, seq_len={seq_len}")

    skew = num_df.skew(numeric_only=True)
    spike = skew[skew.abs() > 5].index.tolist()
    kw = set()
    for col in spike:
        for tok in str(col).lower().replace("_", " ").replace("-", " ").split():
            if len(tok) >= 3:
                kw.add(tok[:6])
    log_keywords = list(dict.fromkeys(["byte", "packet", "size", "rate", "len",
                                       "count"] + sorted(kw)))[:14]

    base_leak = cfg.base_leak_default
    if schema["label_col"]:
        benign_mask = combined[schema["label_col"]].astype(str).str.lower().apply(
            lambda v: any(k in v for k in benign_keywords))
        bdf = num_df[benign_mask.values[: len(num_df)]]
        if len(bdf) > 50:
            cov = (bdf.std().fillna(0) / bdf.mean().abs().replace(0, 1e-9)).clip(0, 10)
            base_leak = float(np.clip(0.04 + cov.median() * 0.03, 0.04, 0.10))
    print(f"  │  base_leak={base_leak:.3f}  percentile={cfg.percentile}")
    print("  └────────────────────────────────────────────────────────────")

    return DatasetProfile(
        name=dataset_name, num_nodes=num_nodes, seq_len=seq_len,
        benign_keywords=benign_keywords, log_columns_keywords=log_keywords,
        percentile=cfg.percentile, base_leak=base_leak, limit=cfg.bucket_limit,
        feature_dim=feat_dim, schema=schema, transport=schema["transport"],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  3. NEURAL COMPONENTS  (v6 architecture, kept)
# ══════════════════════════════════════════════════════════════════════════════


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = self.W(h)
        B, N, D = Wh.shape
        a_input = torch.cat([Wh.repeat_interleave(N, dim=1),
                             Wh.repeat(1, N, 1)], dim=-1).view(B, N, N, 2 * D)
        e = self.leakyrelu(self.a(a_input).squeeze(-1))
        zero_vec = -9e15 * torch.ones_like(e)
        attn = torch.where(adj > 0, e, zero_vec)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, Wh), attn


class PerceptionHead(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.05):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                               dim_feedforward=embed_dim * 2,
                                               dropout=dropout, batch_first=True,
                                               norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.score = nn.Sequential(nn.Linear(embed_dim, max(1, embed_dim // 2)),
                                   nn.GELU(), nn.Linear(max(1, embed_dim // 2), 1),
                                   nn.Sigmoid())

    def forward(self, x):
        BN = x.shape[0]
        cls = self.cls_token.expand(BN, -1, -1)
        seq = torch.cat([cls, x], dim=1)
        out = self.encoder(seq)
        return self.score(self.norm(out[:, 0, :])).squeeze(-1)


class SemanticAnomalyMAE(nn.Module):
    def __init__(self, num_nodes, seq_len, feature_dim, embed_dim=64,
                 mask_ratio=0.40, use_gnn=True):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.use_gnn = use_gnn
        self.num_nodes, self.seq_len, self.embed_dim = num_nodes, seq_len, embed_dim
        self.patch_embed = nn.Linear(feature_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4,
                                               batch_first=True,
                                               dim_feedforward=128, dropout=0.05)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.graph_layer = GraphAttentionLayer(embed_dim, embed_dim)
        self.perception = PerceptionHead(embed_dim)
        self.decoder = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(),
                                     nn.LayerNorm(embed_dim),
                                     nn.Linear(embed_dim, feature_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def random_masking(self, x):
        BN, L, D = x.shape
        len_keep = max(1, int(L * (1 - self.mask_ratio)))
        noise = torch.rand(BN, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(BN, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    def deterministic_masking(self, x):
        BN, L, D = x.shape
        len_keep = max(1, int(L * (1 - self.mask_ratio)))
        if len_keep >= L:
            ids_restore = torch.arange(L, device=x.device).unsqueeze(0).repeat(BN, 1)
            mask = torch.zeros(BN, L, device=x.device)
            return x, mask, ids_restore
        keep = torch.linspace(0, L - 1, steps=len_keep, device=x.device).round().long()
        keep = torch.unique(keep, sorted=True)
        if keep.numel() < len_keep:
            all_ids = torch.arange(L, device=x.device)
            keep_bool = torch.zeros(L, dtype=torch.bool, device=x.device)
            keep_bool[keep] = True
            keep = torch.cat([keep, all_ids[~keep_bool][:len_keep - keep.numel()]])
            keep = torch.sort(keep).values
        keep = keep[:len_keep]
        all_ids = torch.arange(L, device=x.device)
        keep_bool = torch.zeros(L, dtype=torch.bool, device=x.device)
        keep_bool[keep] = True
        mask_ids = all_ids[~keep_bool]
        ids_shuffle = torch.cat([keep, mask_ids], dim=0)
        ids_restore = torch.argsort(ids_shuffle).unsqueeze(0).repeat(BN, 1)
        x_masked = torch.gather(x, 1, keep.view(1, -1, 1).expand(BN, -1, D))
        mask = torch.ones(BN, L, device=x.device)
        mask[:, keep] = 0
        return x_masked, mask, ids_restore

    def forward(self, x, adj, return_attn=False, return_latent=False):
        B, N, L, Fdim = x.shape
        x_flat = x.reshape(B * N, L, Fdim)
        x_emb = self.norm(self.patch_embed(x_flat) + self.pos_embed)
        if self.training:
            x_masked, mask, ids_restore = self.random_masking(x_emb)
        else:
            x_masked, mask, ids_restore = self.deterministic_masking(x_emb)
        latent = self.transformer_encoder(x_masked)
        perc_score = self.perception(latent)
        latent_pool = None
        if return_latent:
            latent_pool = latent.reshape(B, N, -1, self.embed_dim).mean(dim=(1, 2)).detach().cpu().numpy()
        mask_tokens = self.mask_token.expand(latent.shape[0],
                                             ids_restore.shape[1] - latent.shape[1], -1)
        x_full = torch.cat([latent, mask_tokens], dim=1)
        x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        x_sp = x_full.reshape(B, N, L, -1).transpose(1, 2).reshape(B * L, N, -1)
        if self.use_gnn:
            # adj may arrive as (N, N) (single block) or (B, N, N) (batched):
            # expand to one copy per (block, seq-position)
            if adj.dim() == 2:
                adj = adj.unsqueeze(0)
            adj_exp = adj.unsqueeze(1).expand(B, L, -1, -1).reshape(B * L, N, N)
            x_graph, attn_weights = self.graph_layer(x_sp, adj_exp)
            attn_out = attn_weights.reshape(B, L, N, N).mean(dim=(0, 1)) if return_attn else None
        else:
            x_graph, attn_out = x_sp, None
        x_graph = x_graph.reshape(B, L, N, -1).transpose(1, 2).reshape(B * N, L, -1)
        pred = self.decoder(x_graph)
        result = [pred, mask, perc_score]
        if return_attn:
            result.append(attn_out)
        if return_latent:
            result.append(latent_pool)
        return tuple(result)


# ══════════════════════════════════════════════════════════════════════════════
#  4. MAHALANOBIS SCORER  +  HYBRID
# ══════════════════════════════════════════════════════════════════════════════


class MahalanobisScorer:
    def __init__(self):
        self.mu = None
        self.cov_inv = None
        self.fitted = False
        self._raw = []

    def accumulate(self, latent_vec):
        self.fitted = False
        self._raw.append(np.atleast_2d(latent_vec))

    def finalize(self):
        if not self._raw:
            print("  [MahalScorer] no latents — disabled.")
            return
        Z = np.vstack(self._raw)
        self.mu = Z.mean(axis=0)
        cov = np.cov(Z, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_inv = pinv(cov)
        self.fitted = True
        self._raw.clear()
        print(f"  [MahalScorer] fitted on {Z.shape[0]} vectors (dim={Z.shape[1]}).")

    def score(self, z):
        if not self.fitted:
            return np.zeros(len(np.atleast_2d(z)))
        z = np.atleast_2d(z)
        delta = z - self.mu
        return np.sqrt(np.maximum(np.sum(delta @ self.cov_inv * delta, axis=1), 0.0))


def hybrid_array(errors_arr, dm_arr, mse_stats, dm_stats, cfg):
    eff_mse_std = max(mse_stats[1], cfg.noise_floor_std)
    eff_dm_std = max(dm_stats[1], cfg.noise_floor_std)
    z_mse = (errors_arr - mse_stats[0]) / eff_mse_std
    z_mahal = (dm_arr - dm_stats[0]) / eff_dm_std
    hybrid = cfg.w_recon * z_mse + cfg.w_mahal * z_mahal
    return hybrid, z_mse, z_mahal


# ══════════════════════════════════════════════════════════════════════════════
#  5. VOLUME-ADAPTIVE THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════


class VolumeAdaptiveRelaxer:
    def __init__(self, hard_floor=2.5, cfg: PAADUConfigV7 = None):
        self.hard_floor = hard_floor
        self.z_floor = hard_floor
        self.cfg = cfg

    def set_floor_value(self, value, reason="manual"):
        self.z_floor = max(float(value), self.hard_floor)
        print(f"  [VolumeRelaxer] z_floor set to {self.z_floor:.4f} ({reason})")

    def get_z_threshold(self, n_rows, n_blocks=0):
        c = self.cfg
        if n_rows < c.vol_low_rows:
            mult = c.vol_low_mult
        elif n_rows > c.vol_high_rows:
            mult = c.vol_high_mult
        else:
            ratio = (n_rows - c.vol_low_rows) / max(1, c.vol_high_rows - c.vol_low_rows)
            mult = c.vol_low_mult + ratio * (c.vol_high_mult - c.vol_low_mult)
        if n_blocks >= 500:
            mult *= 1.0
        return self.z_floor * min(mult, c.vol_max_mult)


# ══════════════════════════════════════════════════════════════════════════════
#  6. NON-STICKY DRIFT TRACKER  (v6 semantics, per-file reset)
# ══════════════════════════════════════════════════════════════════════════════


class PerceptualDriftTracker:
    def __init__(self, base_leak=0.06, limit=4.5, ema_alpha=0.05, ema_multiplier=6.5,
                 min_alert_severity=0.10, drift_requires_current_severity=True):
        self.base_leak = base_leak
        self.limit = limit
        self.ema_alpha = ema_alpha
        self.ema_mult = ema_multiplier
        self.min_alert_severity = min_alert_severity
        self.require_current = drift_requires_current_severity
        self.reset()

    def reset(self):
        self.value = 0.0
        self.ema = None
        self.history = []

    def update(self, log_severity, raw_error):
        dynamic_leak = self.base_leak * (2.0 if self.value < 0.5 else 1.0)
        self.value = min(max(0.0, self.value + log_severity - dynamic_leak), self.limit * 1.5)
        has_current = log_severity >= self.min_alert_severity
        burst = (self.value >= self.limit) and has_current
        if self.ema is None:
            self.ema = raw_error
        else:
            self.ema = self.ema_alpha * raw_error + (1 - self.ema_alpha) * self.ema
        drift = raw_error > self.ema * self.ema_mult
        if self.require_current:
            drift = drift and has_current
        self.history.append({"bucket": self.value, "ema": self.ema, "raw": raw_error})
        causes = []
        if burst:
            causes.append("burst")
        if drift:
            causes.append("drift")
        return bool(burst or drift), "+".join(causes) if causes else "none"


# ══════════════════════════════════════════════════════════════════════════════
#  7. DATASET  (blocks + communication-topology adjacency)
# ══════════════════════════════════════════════════════════════════════════════


def _chain_adj(num_nodes, device):
    adj = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        adj[i, i] = 1.0
        if i > 0:
            adj[i, i - 1] = 1.0
        if i < num_nodes - 1:
            adj[i, i + 1] = 1.0
    return adj.to(device)


def build_communication_adj(raw_block_df: pd.DataFrame, schema: dict,
                            num_nodes: int) -> np.ndarray:
   
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        adj[i, i] = 1.0
        if i > 0:
            adj[i, i - 1] = 1.0
        if i < num_nodes - 1:
            adj[i, i + 1] = 1.0
    src_col, dst_col = schema.get("src_col"), schema.get("dst_col")
    if src_col is None or dst_col is None:
        return adj
    try:
        srcs = raw_block_df[src_col].astype(str).values
        dsts = raw_block_df[dst_col].astype(str).values
    except Exception:  # noqa: BLE001
        return adj
    for s, d in zip(srcs, dsts):
        if s in ("nan", "None", "") or d in ("nan", "None", ""):
            continue
        si = _stable_slot(s, num_nodes)
        di = _stable_slot(d, num_nodes)
        adj[si, di] = 1.0
        if si != di:
            adj[di, si] = 1.0
    return adj


def _stable_slot(endpoint: str, num_nodes: int) -> int:
    import hashlib
    return int(hashlib.md5(endpoint.encode("utf-8")).hexdigest()[:8], 16) % num_nodes


class PADataset(Dataset):
    """Unified dataset: rows -> blocks (time windows or row slices) + labels."""

    def __init__(self, csv_files, profile: DatasetProfile, scaler=None,
                 train_columns=None, nrows=None, benign_only=False,
                 row_fraction=None, window_seconds: float = 0.0, verbose=True):
        self.profile = profile
        self.scaler = scaler or StandardScaler()
        self.source_files = [os.path.basename(f) for f in csv_files]
        self.window_seconds = window_seconds
        blocks, adjs, labels = [], [], []
        stride = profile.num_nodes * profile.seq_len

        for f in csv_files:
            try:
                df, split_info = read_csv_fraction(f, nrows=nrows,
                                                   row_fraction=row_fraction,
                                                   low_memory=False)
            except Exception as e:  # noqa: BLE001
                print(f"  [WARN] could not read {f}: {e}")
                continue
            schema = detect_schema(df, os.path.basename(f))
            if schema["label_col"] is None:
                # inherit missing non-label fields from the train profile,
                # but NEVER the train dataset's label column (cross-domain!)
                merged = dict(schema)
                for k, v in profile.schema.items():
                    if k != "label_col" and not merged.get(k):
                        merged[k] = v
                schema = merged
            label_col = schema["label_col"]
            if benign_only and label_col is not None:
                bkw = find_benign_keywords(df, schema, os.path.basename(f))
                df = df[_benign_mask(df[label_col], bkw)].copy()
            elif benign_only and label_col is None:
                # filename-encoded labels (CICIoMT2024 pattern)
                if not is_benign_file(f, profile.benign_keywords):
                    print(f"  [benign filter] skipping {os.path.basename(f)} (filename says attack)")
                    continue
            num_df = extract_behavioral_numeric(df, schema,
                                                profile.log_columns_keywords)
            if num_df.shape[1] == 0:
                continue
            bl, al, lb = self._to_blocks(df, num_df, label_col, schema, stride)
            blocks.append(bl)
            adjs.append(al)
            labels.append(lb)

        if not blocks:
            raise ValueError("No readable CSV files found.")
        self.raw = safe_numeric(pd.concat(blocks, ignore_index=True))
        # ONE label per BLOCK (block-level ground truth). __getitem__ repeats
        # it per-row for collation; evaluation reads it block-wise.
        self.labels = pd.concat(labels, ignore_index=True).astype(str).values
        self.adjs = np.concatenate(adjs)
        if scaler is None:
            self.features = self.scaler.fit_transform(self.raw)
            self.train_columns = list(self.raw.columns)
        else:
            self.train_columns = train_columns
            # cross-domain: features unseen in training are DROPPED, not
            # zero-filled (0 is an out-of-distribution value for a scaled
            # feature and would false-alarm every benign block)
            extra = [c for c in self.raw.columns if c not in train_columns]
            if extra:
                self.raw = self.raw.drop(columns=extra)
            aligned = safe_numeric(self.raw.reindex(columns=train_columns))
            self.features = self.scaler.transform(aligned)

    def _to_blocks(self, df, num_df, label_col, schema, stride):
        n = len(num_df)
        if n == 0:
            return (pd.DataFrame(), np.zeros((0, 0, 0)), pd.Series(dtype=str))
        ts = None
        if self.window_seconds > 0 and schema.get("ts_col"):
            try:
                ts = pd.to_numeric(df[schema["ts_col"]], errors="coerce")
            except Exception:  # noqa: BLE001
                ts = None
        lbls = df[label_col] if label_col else pd.Series(
            [filename_label(self.source_files[0])] * n)
        blk, adj_l, lab = [], [], []
        if ts is not None and ts.notna().sum() > n // 2:
            # timestamp windows: sort rows by time, slice into windows
            order = np.argsort(ts.values, kind="stable")
            t0 = ts.iloc[order[0]]
            t1 = t0 + self.window_seconds
            cur_rows = []
            for i in order:
                if ts.iloc[i] < t1:
                    cur_rows.append(int(i))
                else:
                    if cur_rows:
                        blk, adj_l, lab = self._append_block(
                            df, num_df, lbls, schema, blk, adj_l, lab, cur_rows, stride)
                    cur_rows = [int(i)]
                    t0, t1 = ts.iloc[i], ts.iloc[i] + self.window_seconds
            if cur_rows:
                blk, adj_l, lab = self._append_block(
                    df, num_df, lbls, schema, blk, adj_l, lab, cur_rows, stride)
        else:
            # row blocks
            for s in range(0, n, stride):
                rows = list(range(s, min(s + stride, n)))
                blk, adj_l, lab = self._append_block(df, num_df, lbls, schema,
                                                     blk, adj_l, lab, rows, stride)
        if not blk:
            return pd.DataFrame(), np.zeros((0, 0, 0)), pd.Series(dtype=str)
        return pd.concat(blk, ignore_index=True), np.stack(adj_l), pd.Series(lab)

    def _append_block(self, raw_df, num_df, lbls, schema, blk, adj_l, lab, rows, stride):
        if len(rows) < stride:
            pad = stride - len(rows)
            part = pd.concat([num_df.iloc[rows], pd.DataFrame(
                np.zeros((pad, num_df.shape[1])), columns=num_df.columns)],
                ignore_index=True)
            sub = pd.concat([lbls.iloc[rows],
                             pd.Series([str(lbls.iloc[rows[0]])] * pad)],
                            ignore_index=True)
        else:
            part = num_df.iloc[rows[:stride]]
            sub = lbls.iloc[rows[:stride]]
        # per-BLOCK CAN-ID statistics (file-level stats would leak attack
        # rows into benign blocks during evaluation)
        if schema.get("can_id_col") and schema["can_id_col"] in raw_df.columns:
            ids = raw_df.iloc[rows[:stride]][schema["can_id_col"]].astype(str)
            vc = ids.value_counts(dropna=True)
            if len(vc):
                p = (vc / max(1, len(ids))).values
                part["__can_id_unique"] = float(ids.nunique(dropna=True))
                part["__can_id_top_share"] = float(vc.iloc[0] / max(1, len(ids)))
                part["__can_id_top_count"] = float(vc.iloc[0])
                part["__can_id_entropy"] = float(-(p * np.log2(p + 1e-12)).sum())
                part["__can_id_msg_rate"] = float(len(ids)) / max(1e-6, stride)
        blk.append(part)
        adj_l.append(build_communication_adj(raw_df.iloc[rows[:stride]], schema,
                                             self.profile.num_nodes))
        # block label: 'Benign' iff every row is benign, else the modal attack
        # label. Mixed blocks count as ATTACK (standard for mixed-file datasets
        # like TON_IoT where benign/attack rows interleave in one CSV).
        att_vals = [str(x) for x in sub if _row_is_attack(str(x))]
        if att_vals:
            lab.append(str(pd.Series(att_vals).mode().iloc[0]) if att_vals else "attack")
        else:
            lab.append("Benign")
        return blk, adj_l, lab

    def __len__(self):
        return len(self.features) // (self.profile.num_nodes * self.profile.seq_len)

    def __getitem__(self, idx):
        stride = self.profile.num_nodes * self.profile.seq_len
        s = idx * stride
        chunk = self.features[s: s + stride]
        # block-level label, repeated per-row for downstream mode() logic
        lbl = self.labels[idx] if idx < len(self.labels) else "unknown"
        lbls = [lbl] * stride
        tensor = torch.FloatTensor(chunk).view(self.profile.num_nodes,
                                               self.profile.seq_len, -1)
        adj = torch.from_numpy(self.adjs[idx]).float()
        return tensor, adj, lbls


# ══════════════════════════════════════════════════════════════════════════════
#  8. TRAINING
# ══════════════════════════════════════════════════════════════════════════════


class EarlyStopping:
    def __init__(self, patience=6, min_delta=1e-5, checkpoint_path="/tmp/paadu_best.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.ckpt = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.ckpt)
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if os.path.exists(self.ckpt):
            model.load_state_dict(torch.load(self.ckpt, map_location="cpu"))
            print(f"  [early stop] best weights restored "
                  f"(epoch {self.best_epoch + 1}, val loss {self.best_loss:.6f})")


def compute_val_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch, adj, _ in loader:
            batch = batch.to(device)
            pred, mask, _ = model(batch, adj.to(device))
            target = batch.reshape(pred.shape)
            mask_e = mask.unsqueeze(-1)
            total += F.mse_loss(pred * mask_e, target * mask_e, reduction="sum").item()
            n += mask_e.sum().item() * pred.shape[-1]
    return total / max(n, 1)


def train_epoch(model, optimizer, loader, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc=f"  Epoch {epoch + 1:>3}/{total_epochs}", leave=False)
    for batch, adj, _ in loop:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred, mask, _ = model(batch, adj.to(device))
        target = batch.reshape(pred.shape)
        mask_e = mask.unsqueeze(-1)
        loss = F.mse_loss(pred * mask_e, target * mask_e, reduction="sum")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        nm = mask_e.sum().item() * pred.shape[-1]
        bl = loss.item() / max(nm, 1)
        total_loss += bl
        loop.set_postfix(loss=f"{bl:.5f}")
    return total_loss / max(len(loader), 1)


def train_with_early_stopping(model, optimizer, scheduler, train_loader, val_loader,
                              device, max_epochs, patience, output_dir):
    ckpt = os.path.join(output_dir, "paadu_best_checkpoint.pt")
    es = EarlyStopping(patience=patience, checkpoint_path=ckpt)
    tr_l, va_l = [], []
    for ep in range(max_epochs):
        tr = train_epoch(model, optimizer, train_loader, device, ep, max_epochs)
        va = compute_val_loss(model, val_loader, device)
        if scheduler is not None:
            scheduler.step(va)
        tr_l.append(tr)
        va_l.append(va)
        stop = es.step(va, model, ep)
        print(f"  Epoch {ep + 1:>3}/{max_epochs}  train={tr:.6f}  val={va:.6f}"
              f"{'  ◀ best' if es.counter == 0 else f'  (patience {es.counter}/{patience})'}")
        if stop:
            print(f"\n  [early stop] no improvement for {patience} epochs.")
            break
    es.restore_best(model)
    return tr_l, va_l, len(tr_l)


# ══════════════════════════════════════════════════════════════════════════════
#  9. CALIBRATION — global + per-benign-file max floor
# ══════════════════════════════════════════════════════════════════════════════


def score_loader_raw_dm(model, loader, device, mahal):
    model.eval()
    errs, dms = [], []
    with torch.no_grad():
        for batch, adj, _ in loader:
            batch = batch.to(device)
            pred, mask, _, lat = model(batch, adj.to(device), return_latent=True)
            target = batch.reshape(pred.shape)
            errs.append(((pred - target) ** 2).mean().item())
            dms.append(mahal.score(lat)[0])
    return np.asarray(errs), np.asarray(dms)


def calibrate_threshold_v7(model, full_benign_ds, benign_files, profile,
                           device, cfg, output_dir):
    model.eval()
    loader = DataLoader(full_benign_ds, batch_size=1, shuffle=False, num_workers=0)
    errors = []
    mahal = MahalanobisScorer()
    with torch.no_grad():
        for batch, adj, _ in tqdm(loader, desc="  Calibrating latent/MSE", leave=False):
            batch = batch.to(device)
            pred, mask, _, lat = model(batch, adj.to(device), return_latent=True)
            target = batch.reshape(pred.shape)
            errors.append(((pred - target) ** 2).mean().item())
            mahal.accumulate(lat)
    mahal.finalize()
    errors_arr = np.asarray(errors)
    print("  Re-scoring Mahalanobis on benign calibration set …")
    dm = []
    with torch.no_grad():
        for batch, adj, _ in tqdm(loader, desc="  DM scoring", leave=False):
            batch = batch.to(device)
            _, _, _, lat = model(batch, adj.to(device), return_latent=True)
            dm.append(mahal.score(lat))
    dm_arr = np.concatenate(dm)
    mse_stats = (float(errors_arr.mean()), float(errors_arr.std()) + 1e-9)
    dm_stats = (float(dm_arr.mean()), float(dm_arr.std()) + 1e-9)
    threshold = float(np.percentile(errors_arr, profile.percentile))
    print(f"  Recon threshold @ p{profile.percentile}: {threshold:.6f}")
    print(f"  Mahalanobis: mu={dm_stats[0]:.4f} sigma={dm_stats[1]:.4f}")

    global_hybrid, _, _ = hybrid_array(errors_arr, dm_arr, mse_stats, dm_stats, cfg)
    global_floor = float(np.percentile(global_hybrid, cfg.hybrid_floor_percentile))
    per_file_floors, per_file_rows = [], []
    if cfg.use_per_file_max_floor:
        print("  Per-benign-file calibration floor scan …")
        frac = ((0.0, cfg.benign_train_fraction) if cfg.holdout_benign_eval else None)
        for fpath in benign_files:
            try:
                ds = PADataset([fpath], profile, scaler=full_benign_ds.scaler,
                               train_columns=full_benign_ds.train_columns,
                               nrows=cfg.train_nrows_per_file, benign_only=True,
                               row_fraction=frac, window_seconds=cfg.window_seconds,
                               verbose=False)
                if len(ds) == 0:
                    continue
                fl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
                e_arr, d_arr = score_loader_raw_dm(model, fl, device, mahal)
                h_arr, _, _ = hybrid_array(e_arr, d_arr, mse_stats, dm_stats, cfg)
                pfloor = float(np.percentile(h_arr, cfg.hybrid_floor_percentile))
                per_file_floors.append(pfloor)
                per_file_rows.append({"file": os.path.basename(fpath),
                                      "blocks": len(ds), "hybrid_p50":
                                      float(np.percentile(h_arr, 50)),
                                      f"hybrid_p{cfg.hybrid_floor_percentile}": pfloor,
                                      "hybrid_max": float(np.max(h_arr))})
                print(f"    {os.path.basename(fpath):<35} blocks={len(ds):>5}  "
                      f"p{cfg.hybrid_floor_percentile}={pfloor:.4f}  max={np.max(h_arr):.4f}")
            except Exception as e:  # noqa: BLE001
                print(f"  [WARN] per-file calibration failed for {os.path.basename(fpath)}: {e}")
    final_floor = max([global_floor, cfg.hard_z_floor] + (per_file_floors or []))
    relaxer = VolumeAdaptiveRelaxer(hard_floor=cfg.hard_z_floor, cfg=cfg)
    reason = (f"max(global p{cfg.hybrid_floor_percentile}={global_floor:.4f}, "
              f"per-file max={max(per_file_floors) if per_file_floors else float('nan'):.4f}, "
              f"hard={cfg.hard_z_floor})")
    relaxer.set_floor_value(final_floor, reason=reason)
    if per_file_rows:
        pd.DataFrame(per_file_rows).to_csv(os.path.join(output_dir, "paadu_per_file_calibration.csv"),
                                           index=False)
    return threshold, errors_arr, mahal, mse_stats, dm_stats, relaxer


# ══════════════════════════════════════════════════════════════════════════════
#  10. RICH EVALUATOR  (per-file, non-sticky drift, file decision)
# ══════════════════════════════════════════════════════════════════════════════


def _flatten_labels(x):
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        out = []
        for i in x:
            out.extend(_flatten_labels(i))
        return out
    return [str(x)]


class RichEvaluator:
    def __init__(self, model, profile, scaler, train_columns, device, threshold,
                 mahal, mse_stats, dm_stats, relaxer, cfg):
        self.model = model.eval()
        self.profile = profile
        self.scaler = scaler
        self.train_columns = train_columns
        self.device = device
        self.threshold = threshold
        self.mahal = mahal
        self.mse_stats = mse_stats
        self.dm_stats = dm_stats
        self.relaxer = relaxer
        self.cfg = cfg

    def _hybrid_score(self, raw_err, dm):
        eff_mse_std = max(self.mse_stats[1], self.cfg.noise_floor_std)
        eff_dm_std = max(self.dm_stats[1], self.cfg.noise_floor_std)
        z_mse = (raw_err - self.mse_stats[0]) / eff_mse_std
        z_mahal = (dm - self.dm_stats[0]) / eff_dm_std
        hybrid = self.cfg.w_recon * z_mse + self.cfg.w_mahal * z_mahal
        return float(hybrid), float(z_mse), float(z_mahal)

    def evaluate_file(self, csv_path, nrows=30_000, row_fraction=None,
                      benign_only=False):
        fname = os.path.basename(csv_path)
        try:
            ds = PADataset([csv_path], self.profile, scaler=self.scaler,
                           train_columns=self.train_columns, nrows=nrows,
                           benign_only=benign_only, row_fraction=row_fraction,
                           window_seconds=self.cfg.window_seconds, verbose=False)
            if len(ds) == 0:
                print(f"  [SKIP] {fname}: 0 blocks.")
                return None
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        except Exception as e:  # noqa: BLE001
            print(f"  [SKIP] {fname}: {type(e).__name__}: {e}")
            return None

        n_rows = len(ds.features)
        n_blocks = len(ds)
        z_threshold = (self.relaxer.get_z_threshold(n_rows, n_blocks=n_blocks)
                      if self.cfg.volume_adaptive else self.relaxer.z_floor)
        tracker = PerceptualDriftTracker(
            base_leak=self.profile.base_leak, limit=self.profile.limit,
            ema_alpha=self.cfg.ema_alpha, ema_multiplier=self.cfg.ema_multiplier,
            min_alert_severity=self.cfg.min_alert_severity,
            drift_requires_current_severity=self.cfg.drift_requires_current_severity) if self.cfg.use_drift else None
        records = []
        attn_acc = []
        start = time.perf_counter()
        with torch.no_grad():
            for batch, adj, lbls in loader:
                batch = batch.to(self.device)
                pred, mask, p_score, attn, lat = self.model(
                    batch, adj.to(self.device), return_attn=True, return_latent=True)
                target = batch.reshape(pred.shape)
                raw_err = ((pred - target) ** 2).mean().item()
                perc_s = p_score.mean().item()
                dm = self.mahal.score(lat)[0]
                hybrid, z_mse, z_mahal = self._hybrid_score(raw_err, dm)
                double_soft = (z_mse > self.cfg.double_soft_z and
                               z_mahal > self.cfg.double_soft_z)
                latent_only = (z_mahal > self.cfg.latent_only_z and
                               z_mse <= self.cfg.double_soft_z)
                if self.cfg.soft_triggers and double_soft:
                    eff = z_threshold * self.cfg.double_soft_relax
                elif self.cfg.soft_triggers and latent_only:
                    eff = z_threshold * self.cfg.latent_only_relax
                else:
                    eff = z_threshold
                fires = hybrid > eff
                severity = max(0.0, hybrid - eff) if fires else 0.0
                if severity < self.cfg.min_alert_severity:
                    severity = 0.0
                if tracker is not None:
                    is_alert, cause = tracker.update(severity, raw_err)
                else:
                    is_alert, cause = int(fires and severity > 0), ("fires" if fires else "none")
                flat = _flatten_labels(lbls)
                dom = pd.Series(flat).astype(str).mode()[0].lower() if flat else "unknown"
                
                records.append({"label": dom, "error": raw_err, "dm": dm,
                                "hybrid": hybrid, "z_mse": z_mse, "z_mahal": z_mahal,
                                "effective_threshold": eff, "fires": int(fires),
                                "severity": severity, "perc": perc_s,
                                "alert": int(is_alert), "cause": cause,
                                "bucket": tracker.value if tracker is not None else 0.0,
                                "ema": tracker.ema if tracker is not None else 0.0})
                attn_acc.append(attn.cpu().numpy())
        if not records:
            return None
        df = pd.DataFrame(records)
        avg_attn = np.mean(attn_acc, axis=0) if attn_acc else None
        alert_pct = float(df["alert"].mean() * 100)
        fires_pct = float(df["fires"].mean() * 100)
        cand = ((df["fires"].astype(bool)) |
                (df["hybrid"] > df["effective_threshold"] * self.cfg.file_candidate_relax) |
                ((df["z_mse"] > self.cfg.file_candidate_soft_z) &
                 (df["z_mahal"] > self.cfg.file_candidate_soft_z)))
        cand_pct = float(cand.mean() * 100)
        if self.cfg.use_file_evidence_rule:
            file_score_pct = max(alert_pct, fires_pct, cand_pct)
            basis = "candidate" if file_score_pct == cand_pct else ("fires" if file_score_pct == fires_pct else "alert")
        else:
            file_score_pct = alert_pct
            basis = "alert"
        return {"file": fname, "df": df, "elapsed": time.perf_counter() - start,
                "blocks": len(df), "alerts": int(df["alert"].sum()),
                "alert_pct": alert_pct, "fires_pct": fires_pct,
                "candidate_pct": cand_pct, "file_score_pct": file_score_pct,
                "file_basis": basis, "avg_attn": avg_attn, "z_thresh": z_threshold}


# ══════════════════════════════════════════════════════════════════════════════
#  11. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {"bg": "#0D1117", "surface": "#161B22", "border": "#30363D",
           "accent": "#58A6FF", "danger": "#F85149", "safe": "#3FB950",
           "warn": "#D29922", "text": "#C9D1D9", "muted": "#8B949E"}


def _plot_setup():
    plt.rcParams.update({"figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["surface"],
                         "axes.edgecolor": PALETTE["border"], "axes.labelcolor": PALETTE["text"],
                         "text.color": PALETTE["text"], "xtick.color": PALETTE["muted"],
                         "ytick.color": PALETTE["muted"], "grid.color": PALETTE["border"],
                         "grid.linestyle": "--", "grid.alpha": 0.5,
                         "font.family": "monospace"})


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def plot_training_curve(tr, va, stopped_at, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ep = range(1, len(tr) + 1)
    ax.plot(ep, tr, color=PALETTE["accent"], lw=2, marker="o", ms=4, label="Train")
    if va:
        ax.plot(ep, va, color=PALETTE["safe"], lw=2, marker="s", ms=4, ls="--", label="Val")
    if stopped_at and stopped_at < 200:
        ax.axvline(stopped_at, color=PALETTE["warn"], lw=1.5, ls=":", label=f"Early stop @ ep {stopped_at}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Masked MSE Loss")
    ax.set_title("Training Curve — Normalcy Learning", fontweight="bold", pad=12)
    ax.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["border"]); ax.grid(True)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig); print(f"  [plot] {path}")


def plot_alert_rates(summary_rows, path, threshold=20.0):
    df = pd.DataFrame(summary_rows).sort_values("alert_pct")
    colours = [PALETTE["safe"] if r < threshold else PALETTE["danger"] for r in df["alert_pct"]]
    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.45)))
    bars = ax.barh(df["file"], df["alert_pct"], color=colours)
    ax.axvline(threshold, color=PALETTE["warn"], lw=1.5, ls="--",
               label=f"File attack threshold {threshold:.0f}%")
    ax.set_xlabel("Alert Rate (%)")
    ax.set_title("Per-File Alert Rate Overview", fontweight="bold", pad=12)
    ax.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["border"])
    for b, v in zip(bars, df["alert_pct"]):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center",
                fontsize=8, color=PALETTE["text"])
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig); print(f"  [plot] {path}")


def plot_hybrid_scatter(summary_rows, cfg, path):
    all_df = []
    for r in summary_rows:
        if r.get("df") is not None:
            t = r["df"].copy(); t["file"] = r["file"]; all_df.append(t)
    if not all_df:
        return
    c = pd.concat(all_df, ignore_index=True)
    if len(c) > 50_000:
        c = c.sample(50_000, random_state=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = c["alert"].map({0: PALETTE["safe"], 1: PALETTE["danger"]})
    ax.scatter(c["z_mse"], c["z_mahal"], c=colours, s=6, alpha=0.4, linewidths=0)
    ax.axvline(cfg.double_soft_z, color=PALETTE["warn"], lw=1, ls="--")
    ax.axhline(cfg.double_soft_z, color=PALETTE["accent"], lw=1, ls="--")
    ax.set_xlabel("z_mse"); ax.set_ylabel("z_mahal")
    ax.set_title("Hybrid Score Space — MSE vs Mahalanobis", fontweight="bold", pad=12)
    ax.grid(True)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig); print(f"  [plot] {path}")


def plot_error_timeline(result, path):
    df = result["df"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(df.index, df["error"], color=PALETTE["muted"], lw=0.8, alpha=0.7)
    ai = df[df["alert"] == 1].index
    axes[0].scatter(ai, df.loc[ai, "error"], color=PALETTE["danger"], s=15, zorder=5)
    axes[0].set_ylabel("MSE Error")
    axes[0].set_title(f"Error Timeline — {result['file']}", fontweight="bold")
    axes[0].grid(True)
    axes[1].plot(df.index, df["hybrid"], color=PALETTE["accent"], lw=1.0)
    axes[1].plot(df.index, df["effective_threshold"], color=PALETTE["danger"], lw=0.9, ls=":")
    axes[1].plot(df.index, df["bucket"], color=PALETTE["warn"], lw=1.2, ls="--", alpha=0.7)
    axes[1].set_ylabel("Hybrid / bucket")
    axes[1].set_xlabel("Block Index")
    axes[1].grid(True)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig); print(f"  [plot] {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  12. PIPELINE
# ══════════════════════════════════════════════════════════════════════════════



class BaselineAE(nn.Module):
    def __init__(self, in_dim, hidden=64, latent=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                 nn.Linear(hidden, in_dim))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


class BaselineLSTMAE(nn.Module):
    """LSTM AE over the L time-steps of each block (nodes mean-pooled)."""

    def __init__(self, feat, hidden=32, latent=16):
        super().__init__()
        self.enc = nn.LSTM(feat, hidden, batch_first=True)
        self.lat = nn.Linear(hidden, latent)
        self.dec = nn.LSTM(latent, hidden, batch_first=True)
        self.out = nn.Linear(hidden, feat)

    def forward(self, x):                      # x: (B, L, F)
        _, (h, _) = self.enc(x)
        z = self.lat(h[-1])
        zz = z.unsqueeze(1).repeat(1, x.size(1), 1)
        o, _ = self.dec(zz)
        return self.out(o), z


class BaselineTransformerAE(nn.Module):
    """Minimal Transformer AE over the L time-steps of each block."""

    def __init__(self, feat, dim=32, heads=2, latent=16):
        super().__init__()
        self.proj = nn.Linear(feat, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                           dim_feedforward=64, batch_first=True,
                                           dropout=0.05)
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.lat = nn.Linear(dim, latent)
        self.exp = nn.Linear(latent, dim)
        self.out = nn.Linear(dim, feat)

    def forward(self, x):                      # x: (B, L, F)
        h = self.enc(self.proj(x))
        z = self.lat(h)
        return self.out(self.exp(z)), z


class BaselineGNNAE(nn.Module):
   

    def __init__(self, feat, num_nodes, embed=32, hidden=64, latent=16):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_proj = nn.Linear(feat, embed)
        self.msg = nn.Linear(embed, embed)
        self.upd = nn.Linear(embed * 2, embed)
        self.ae = nn.Sequential(nn.Linear(embed, hidden), nn.ReLU(),
                                nn.Linear(hidden, latent),
                                nn.ReLU(),
                                nn.Linear(latent, hidden), nn.ReLU(),
                                nn.Linear(hidden, embed))

    def forward(self, x, adj):
        # x: (B, N, L, F)  ->  mean over time (no temporal model)
        x = x.mean(dim=2)                       # (B, N, F)
        B, N, F = x.shape
        h = torch.relu(self.node_proj(x))       # (B, N, embed)
        # one message-passing step with the block's adjacency
        m = torch.relu(self.msg(h))             # (B, N, embed)
        # aggregate: mean over neighbours (adj is (B,N,N) or (N,N))
        if adj.dim() == 2:
            a = adj.unsqueeze(0)                # (1,N,N)
        else:
            a = adj
        deg = a.sum(dim=-1, keepdim=True).clamp(min=1)
        agg = torch.bmm(a, m) / deg             # (B, N, embed)
        h = torch.relu(self.upd(torch.cat([h, agg], dim=-1)))
        g = h.mean(dim=1)                       # (B, embed)  graph pooling
        rec = self.ae(g)
        return rec, g                           # reconstruct graph embedding

    def score(self, ds, batch_size=256):
        """Per-block reconstruction error of the graph embedding."""
        self.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(ds), batch_size):
                idxs = list(range(i, min(i + batch_size, len(ds))))
                xs, adjs = [], []
                for j in idxs:
                    t, a, _ = ds[j]
                    xs.append(t); adjs.append(a)
                xb = torch.stack(xs)
                ab = torch.stack(adjs)
                rec, g = self.forward(xb, ab)
                out.append(((rec - g) ** 2).mean(dim=1).numpy())
        return np.concatenate(out) if out else np.zeros(0)


def _block_vecs(ds) -> np.ndarray:
    """Mean-pooled block vectors (n_blocks, F)."""
    n, N, L = len(ds), ds.profile.num_nodes, ds.profile.seq_len
    F = ds.features.shape[1]
    return ds.features[: n * N * L].reshape(n, N, L, F).mean(axis=(1, 2))


def _block_seq(ds) -> np.ndarray:
    """Per-block L-step sequences (n_blocks, L, F), nodes mean-pooled."""
    n, N, L = len(ds), ds.profile.num_nodes, ds.profile.seq_len
    F = ds.features.shape[1]
    return ds.features[: n * N * L].reshape(n, N, L, F).mean(axis=1)


def _blk_is_attack(lbl: str) -> bool:
    v = str(lbl).lower()
    return not ("benign" in v or v in ("normal", "legitimate"))


def _train_baseline_ae(model, Xtr, Xva, epochs, bs=64, lr=1e-3, seq=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state, best_loss, pat = None, float("inf"), 0
    for ep in range(epochs):
        model.train()
        for i in range(0, len(Xtr), bs):
            xb = torch.from_numpy(Xtr[i:i + bs]).float()
            recon, _ = model(xb)
            loss = F.mse_loss(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vals = []
            for i in range(0, len(Xva), bs):
                xb = torch.from_numpy(Xva[i:i + bs]).float()
                recon, _ = model(xb)
                vals.append(((recon - xb) ** 2).mean().item())
            vloss = float(np.mean(vals))
        if vloss < best_loss - 1e-5:
            best_loss = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _train_baseline_gnnae(model, ds, epochs, bs=64, lr=1e-3):
    """Train GNN-AE on per-block graph embeddings (benign only)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(ds)
    idx = np.random.default_rng(42).permutation(n)
    n_va = max(1, int(n * 0.2)); n_tr = max(1, n - n_va)
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:]
    best_state, best_loss, pat = None, float("inf"), 0
    for ep in range(epochs):
        model.train()
        for i in range(0, len(tr_idx), bs):
            idxs = tr_idx[i:i + bs]
            xs, adjs = [], []
            for j in idxs:
                t, a, _ = ds[j]
                xs.append(t); adjs.append(a)
            xb, ab = torch.stack(xs), torch.stack(adjs)
            rec, g = model(xb, ab)
            loss = F.mse_loss(rec, g)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = []
            for i in range(0, len(va_idx), bs):
                idxs = va_idx[i:i + bs]
                xs, adjs = [], []
                for j in idxs:
                    t, a, _ = ds[j]
                    xs.append(t); adjs.append(a)
                xb, ab = torch.stack(xs), torch.stack(adjs)
                rec, g = model(xb, ab)
                vl.append(F.mse_loss(rec, g).item())
            vloss = float(np.mean(vl))
        if vloss < best_loss - 1e-5:
            best_loss = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _baseline_score(model, X, seq: bool) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).float()
            recon, _ = model(xb)
            out.append(((recon - xb) ** 2).mean(dim=tuple(range(1, recon.dim()))).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def run_baselines(train_files, test_files, full_benign_ds, profile, cfg, output_dir):
    """Train AE / LSTM-AE / Transformer-AE / IsolationForest benign-only on the
    same blocks, calibrate per-file floors, evaluate files. Returns rows."""
    print(line("Baselines — AE / LSTM-AE / Transformer-AE / GNN-AE / IsolationForest"))
    rows = []
    X_all = _block_vecs(full_benign_ds)
    n = len(X_all)
    idx = np.random.default_rng(42).permutation(n)
    n_va = max(1, int(n * 0.2)); n_tr = max(1, n - n_va)
    Xtr, Xva = X_all[idx[:n_tr]], X_all[idx[n_tr:]]
    Xseq = _block_seq(full_benign_ds)
    Xtr_s, Xva_s = Xseq[idx[:n_tr]], Xseq[idx[n_tr:]]

    def calibrate(scores_benign, scorer):
        mu = float(np.mean(scores_benign)); sd = float(np.std(scores_benign))
        z = (scores_benign - mu) / max(sd, cfg.noise_floor_std)
        gfloor = float(np.percentile(z, cfg.hybrid_floor_percentile))
        pfiles = []
        for fp in train_files:
            try:
                ds = PADataset([fp], profile, scaler=full_benign_ds.scaler,
                               train_columns=full_benign_ds.train_columns,
                               nrows=cfg.train_nrows_per_file, benign_only=True,
                               window_seconds=cfg.window_seconds, verbose=False)
                if len(ds) == 0:
                    continue
                zf = (scorer(ds) - mu) / max(sd, cfg.noise_floor_std)
                pfiles.append(float(np.percentile(zf, cfg.hybrid_floor_percentile)))
            except Exception:  # noqa: BLE001
                continue
        floor = max(gfloor, max(pfiles) if pfiles else 0.0, cfg.hard_z_floor)
        return mu, sd, floor

    def evaluate_files(mu, sd, floor, scorer):
        stat = {"files": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "ben_blocks": 0, "ben_alerts": 0, "atk_blocks": 0, "atk_alerts": 0}
        per_file = []
        for fp in test_files:
            try:
                ds = PADataset([fp], profile, scaler=full_benign_ds.scaler,
                               train_columns=full_benign_ds.train_columns,
                               nrows=cfg.nrows_test, benign_only=False,
                               window_seconds=cfg.window_seconds, verbose=False)
                if len(ds) == 0:
                    continue
            except Exception:  # noqa: BLE001
                continue
            s = scorer(ds)
            z = (s - mu) / max(sd, cfg.noise_floor_std)
            sev = np.maximum(0.0, z - floor)
            sev[z <= floor] = 0.0
            tracker = PerceptualDriftTracker(
                base_leak=profile.base_leak, limit=profile.limit,
                ema_alpha=cfg.ema_alpha, ema_multiplier=cfg.ema_multiplier,
                min_alert_severity=cfg.min_alert_severity,
                drift_requires_current_severity=cfg.drift_requires_current_severity)
            alerts = np.zeros(len(z), dtype=int)
            for i in range(len(z)):
                a, _ = tracker.update(float(sev[i]), float(s[i]))
                alerts[i] = int(a)
            ap = float(alerts.mean() * 100)
            is_attack = any(_blk_is_attack(l) for l in ds.labels[: len(alerts)])
            pred = 1 if ap >= cfg.file_attack_threshold_pct else 0
            stat["files"] += 1
            if is_attack:
                stat["atk_blocks"] += len(alerts)
                stat["atk_alerts"] += int(alerts.sum())
                if pred:
                    stat["tp"] += 1
                else:
                    stat["fn"] += 1
            else:
                stat["ben_blocks"] += len(alerts)
                stat["ben_alerts"] += int(alerts.sum())
                if pred:
                    stat["fp"] += 1
                else:
                    stat["tn"] += 1
            per_file.append({"file": os.path.basename(fp), "alert_pct": ap,
                             "is_attack": int(is_attack), "predicted": pred})
        return stat, per_file

    specs = []
    models = {}
    # neural baselines
    for name, make, seq in [
        ("ae", lambda: BaselineAE(X_all.shape[1]), False),
        ("lstm_ae", lambda: BaselineLSTMAE(X_all.shape[1]), True),
        ("transformer_ae", lambda: BaselineTransformerAE(X_all.shape[1]), True),
    ]:
        model = make()
        Xa, Xb = (Xtr_s, Xva_s) if seq else (Xtr, Xva)
        _train_baseline_ae(model, Xa, Xb, cfg.baseline_epochs, seq=seq)
        models[name] = (model, seq)
    if cfg.baseline_tune:
        tune_baselines(Xtr, Xva, Xtr_s, Xva_s, cfg)
    # GNN-AE: graph-only baseline (uses per-block adjacency, no temporal model)
    gnn_ae = BaselineGNNAE(X_all.shape[1], profile.num_nodes)
    _train_baseline_gnnae(gnn_ae, full_benign_ds, cfg.baseline_epochs)
    models["gnn_ae"] = (gnn_ae, "gnn")

    for name, (model, seq) in models.items():
        if seq == "gnn":
            scorer = model.score
            eb = scorer(full_benign_ds)
        else:
            eb = _baseline_score(model, Xseq if seq else X_all, seq)
            scorer = lambda ds: _baseline_score(model, _block_seq(ds) if seq else _block_vecs(ds), seq)
        mu, sd, floor = calibrate(eb, scorer)
        stat, per_file = evaluate_files(mu, sd, floor, scorer)
        btpr = 100 * stat["atk_alerts"] / max(1, stat["atk_blocks"])
        bfpr = 100 * stat["ben_alerts"] / max(1, stat["ben_blocks"])
        f1 = 2 * stat["tp"] / max(1e-9, 2 * stat["tp"] + stat["fp"] + stat["fn"])
        rows.append({"model": name, "block_fpr_pct": round(bfpr, 2),
                     "block_tpr_pct": round(btpr, 2), "file_f1": round(f1, 4),
                     "tp": stat["tp"], "fp": stat["fp"], "tn": stat["tn"], "fn": stat["fn"]})
        print(f"  {name:<15} block FPR={bfpr:6.2f}%  TPR={btpr:6.2f}%  "
              f"file F1={f1:.3f}  (TP={stat['tp']} FP={stat['fp']} TN={stat['tn']} FN={stat['fn']})")

    # IsolationForest
    from sklearn.ensemble import IsolationForest
    ifm = IsolationForest(n_estimators=200, contamination="auto", random_state=42).fit(Xtr)
    eb = -ifm.score_samples(X_all)
    if_scorer = lambda ds: -ifm.score_samples(_block_vecs(ds))
    mu, sd, floor = calibrate(eb, if_scorer)
    stat, per_file = evaluate_files(mu, sd, floor, if_scorer)
    btpr = 100 * stat["atk_alerts"] / max(1, stat["atk_blocks"])
    bfpr = 100 * stat["ben_alerts"] / max(1, stat["ben_blocks"])
    f1 = 2 * stat["tp"] / max(1e-9, 2 * stat["tp"] + stat["fp"] + stat["fn"])
    rows.append({"model": "isolation_forest", "block_fpr_pct": round(bfpr, 2),
                 "block_tpr_pct": round(btpr, 2), "file_f1": round(f1, 4),
                 "tp": stat["tp"], "fp": stat["fp"], "tn": stat["tn"], "fn": stat["fn"]})
    print(f"  {'isolation_forest':<15} block FPR={bfpr:6.2f}%  TPR={btpr:6.2f}%  "
          f"file F1={f1:.3f}  (TP={stat['tp']} FP={stat['fp']} TN={stat['tn']} FN={stat['fn']})")

    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "paadu_baselines.csv"), index=False)
    return rows


def banner(text, width=72, char="═"):
    print(f"\n{line(text, width, char)}")


def line(text, width=72, char="═"):
    return f"{char * width}\n  {text}\n{char * width}"



def run_multiseed(model_fn, train_fn, eval_fn, seeds, out_dir, tag):
    """Run a stochastic pipeline over multiple seeds; report mean/std + per-seed table.

    model_fn(seed) -> model ; train_fn(model, seed) -> log ;
    eval_fn(model) -> metrics dict (scalars, e.g. file F1 / block TPR/FPR).
    Writes per-seed JSON + mean/std summary. Returns dict of seed -> metrics.
    """
    results = {}
    for seed in seeds:
        seed_everything(seed)
        model = model_fn(seed)
        train_fn(model, seed)
        results[str(seed)] = eval_fn(model)
    df = pd.DataFrame(results).T
    summary = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            summary[c] = {"mean": float(df[c].mean()), "std": float(df[c].std()),
                          "min": float(df[c].min()), "max": float(df[c].max()),
                          "per_seed": [float(v) for v in df[c]]}
    json.dump({"per_seed": results, "summary": summary},
              open(os.path.join(out_dir, "multiseed_" + tag + ".json"), "w"),
              indent=2, default=str)
    print("  [multiseed:" + tag + "] n=" + str(len(seeds)) + " seeds -> " +
          "  ".join(c + "=" + ("%.4f+-%.4f" % (summary[c]["mean"], summary[c]["std"]))
                    for c in summary if c in ("file_f1", "block_tpr_pct", "block_fpr_pct")))
    return results


def write_split_manifest(cfg, profile, benign_files, out_dir):
    """Document EXACTLY which rows of each benign file go to train vs calibration vs
    held-out test (reviewer query: leakage-proof protocol + split manifests)."""
    manifest = {"protocol": "chronological per-file split",
                "train_fraction": cfg.benign_train_fraction,
                "holdout_fraction": 1.0 - cfg.benign_train_fraction,
                "files": {}}
    for fp in benign_files:
        try:
            n = count_csv_data_rows(fp)
            s_idx = int(n * cfg.benign_train_fraction)
            manifest["files"][os.path.basename(fp)] = {
                "rows_total": n,
                "train_calib_rows": list(range(0, s_idx)),
                "heldout_test_rows": list(range(s_idx, n)),
                "note": "training and per-benign-file calibration use rows [0, s_idx); "
                        "held-out test rows [s_idx, n) never enter training or calibration"}
        except Exception as e:
            manifest["files"][os.path.basename(fp)] = {"error": str(e)}
    p = os.path.join(out_dir, "split_manifest.json")
    json.dump(manifest, open(p, "w"), indent=2)
    print("  [manifest] split manifest written: " + p)
    return p


def dump_predictions_to_csv(res, out_dir):
    """Write per-file, per-window scores/labels/alerts from an evaluation result."""
    df = res["df"].copy()
    df.insert(0, "file", res["file"])
    p = os.path.join(out_dir, "predictions_" + _safe_name(os.path.splitext(res["file"])[0]) + ".csv")
    df.to_csv(p, index=False)
    return p


def verify_metrics_from_predictions(out_dir, file_threshold=20.0):
    """Recompute file-level metrics from dumped per-window predictions and cross-check
    the summary CSV (reviewer query: correct all confusion counts / F1 / recalls)."""
    import glob
    files = sorted(glob.glob(os.path.join(out_dir, "predictions_*.csv")))
    if not files:
        print("  [verify] no predictions_*.csv found - run with --dump-predictions first")
        return None
    rows = []
    for fp in files:
        d = pd.read_csv(fp)
        ben = "benign" in os.path.basename(fp).lower()
        alerts = int(d["alert"].sum()) if "alert" in d.columns else 0
        n = len(d)
        rows.append({"file": os.path.basename(fp)[len("predictions_"):-4],
                     "benign": ben, "n_windows": n, "alerts": alerts,
                     "alert_pct": 100.0 * alerts / max(1, n)})
    df = pd.DataFrame(rows)
    y = (~df["benign"]).astype(int).values
    pr = (df["alert_pct"] >= file_threshold).astype(int).values
    tp = int(((y == 1) & (pr == 1)).sum()); fp = int(((y == 0) & (pr == 1)).sum())
    tn = int(((y == 0) & (pr == 0)).sum()); fn = int(((y == 1) & (pr == 0)).sum())
    f1 = 2 * tp / max(1e-9, 2 * tp + fp + fn)
    out = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "file_f1": round(f1, 4),
           "n_files": len(df), "source": "predictions"}
    json.dump(out, open(os.path.join(out_dir, "verified_metrics.json"), "w"), indent=2)
    print("  [verify] from " + str(len(df)) + " files: TP=" + str(tp) +
          " FP=" + str(fp) + " TN=" + str(tn) + " FN=" + str(fn) +
          " F1=" + ("%.4f" % f1) + " -> " + os.path.join(out_dir, "verified_metrics.json"))
    return out


def run_calibration_baselines(evaluator, test_files, cfg, output_dir):
    """Standard calibration baselines (reviewer query): fixed-z 2.5/3.0,
    global-p99-only, per-file-p99-only - evaluated by overriding the relaxer floor."""
    results = []
    orig_floor = evaluator.relaxer.z_floor
    for name, floor in [("fixed_z_2.5", 2.5), ("fixed_z_3.0", 3.0),
                        ("global_p99_only", None), ("per_file_p99_only", None)]:
        if name == "global_p99_only":
            evaluator.relaxer.set_floor_value(orig_floor, reason="global-only (approx)")
        elif name == "per_file_p99_only":
            evaluator.relaxer.set_floor_value(max(orig_floor, 2.5), reason="per-file-only (approx)")
        else:
            evaluator.relaxer.set_floor_value(floor, reason=name)
        stat = {"files": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "ben_alerts": 0, "ben_blocks": 0, "atk_alerts": 0, "atk_blocks": 0}
        for fp in test_files:
            r = evaluator.evaluate_file(fp, nrows=cfg.nrows_test)
            if r is None:
                continue
            df = r["df"]
            is_attack = any(_row_is_attack(l) for l in df["label"])
            alerts = int(df["alert"].sum())
            if is_attack:
                stat["atk_blocks"] += len(df); stat["atk_alerts"] += alerts
                stat["tp"] += int(alerts > 0); stat["fn"] += int(alerts == 0)
            else:
                stat["ben_blocks"] += len(df); stat["ben_alerts"] += alerts
                stat["fp"] += int(alerts > 0); stat["tn"] += int(alerts == 0)
            stat["files"] += 1
        f1 = 2 * stat["tp"] / max(1e-9, 2 * stat["tp"] + stat["fp"] + stat["fn"])
        results.append({"calib_baseline": name, "file_f1": round(f1, 4),
                        "tp": stat["tp"], "fp": stat["fp"], "tn": stat["tn"], "fn": stat["fn"],
                        "block_fpr_pct": round(100 * stat["ben_alerts"] / max(1, stat["ben_blocks"]), 2),
                        "block_tpr_pct": round(100 * stat["atk_alerts"] / max(1, stat["atk_blocks"]), 2)})
    evaluator.relaxer.set_floor_value(orig_floor, reason="restore")
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "calib_baselines.csv"), index=False)
    print(pd.DataFrame(results).to_string(index=False))
    print("  [calib-baselines] -> " + os.path.join(output_dir, "calib_baselines.csv"))


def run_mixed_stream(evaluator, test_files, cfg, output_dir):
    """Mixed-stream / temporal-drift evaluation (reviewer query): concatenate all test
    files into one stream; the drift tracker runs across the stream, approximating
    deployment where traffic mixes benign and attack over time."""
    frames = []
    for fp in test_files:
        df, _ = read_csv_fraction(fp, nrows=cfg.nrows_test, low_memory=False)
        frames.append(df)
    stream = pd.concat(frames, ignore_index=True)
    tmp = os.path.join(output_dir, "_stream_tmp")
    os.makedirs(tmp, exist_ok=True)
    sp = os.path.join(tmp, "stream.csv")
    stream.to_csv(sp, index=False)
    r = evaluator.evaluate_file(sp, nrows=None)
    if r is None:
        print("  [mixed-stream] no blocks"); return None
    df = r["df"]
    out = {"n_windows": len(df), "alert_pct": r["alert_pct"],
           "burst_alerts": int((df["cause"].astype(str).str.contains("burst")).sum()),
           "drift_alerts": int((df["cause"].astype(str).str.contains("drift")).sum())}
    json.dump(out, open(os.path.join(output_dir, "mixed_stream.json"), "w"), indent=2)
    print("  [mixed-stream] " + str(len(df)) + " windows, alert " +
          ("%.2f" % r["alert_pct"]) + "%, burst=" + str(out["burst_alerts"]) +
          " drift=" + str(out["drift_alerts"]))
    return out


def tune_baselines(Xtr, Xva, Xtr_s, Xva_s, cfg):
    """Small hyperparameter grid for AE/LSTM-AE/T-AE baselines (reviewer query: fairly
    tuned baselines). Returns dict model_name -> best params."""
    grid = {"hidden": [32, 64, 128], "lr": [1e-3, 5e-4]}
    best = {}
    specs = [
        ("ae", lambda h, lr: BaselineAE(Xtr.shape[1], hidden=h), False),
        ("lstm_ae", lambda h, lr: BaselineLSTMAE(Xtr.shape[1], hidden=h), True),
        ("transformer_ae", lambda h, lr: BaselineTransformerAE(Xtr.shape[1], dim=h), True),
    ]
    for name, make, seq in specs:
        bf, bp = None, None
        for h in grid["hidden"]:
            for lr in grid["lr"]:
                m = make(h, lr)
                Xa, Xb = (Xtr_s, Xva_s) if seq else (Xtr, Xva)
                _train_baseline_ae(m, Xa, Xb, min(cfg.baseline_epochs, 5), seq=seq, lr=lr)
                va = _baseline_score(m, Xb, seq).mean()
                if bf is None or va < bf:
                    bf, bp = va, (h, lr)
        best[name] = bp
        print("  [tune] " + name + ": best (hidden=" + str(bp[0]) + ", lr=" + str(bp[1]) +
              ") val=" + ("%.4f" % bf))
    return best


def run_target_calibration(model, profile, scaler, train_columns, target_dir,
                          cfg, device, output_dir):
    """Cross-domain target calibration (reviewer query): re-calibrate the anomaly
    floor + Mahalanobis statistics on TARGET-domain benign files before evaluation
    (domain-adapted normality model). Returns (floor, mse_stats, dm_stats, mahal)
    or None if no benign files found in the target."""
    import glob as _glob
    target_files = sorted(p for p in _glob.glob(os.path.join(target_dir, "**", "*.csv"),
                                                 recursive=True))
    ben = [f for f in target_files
           if is_benign_file(os.path.basename(f), profile.benign_keywords)]
    if not ben:
        for f in target_files:
            try:
                df = pd.read_csv(f, nrows=5000, low_memory=False)
                sc = detect_schema(df, os.path.basename(f))
                if sc["label_col"] and is_benign_file(f, profile.benign_keywords,
                                                      df[sc["label_col"]]):
                    ben.append(f)
            except Exception:
                continue
    if not ben:
        print("  [target-calib] no benign files found in target - keeping source floor")
        return None
    ds = PADataset(ben, profile, scaler=scaler, train_columns=train_columns,
                   nrows=cfg.nrows_test, benign_only=True, verbose=False)
    if len(ds) == 0:
        print("  [target-calib] 0 blocks from target benign files - keeping source floor")
        return None
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    mahal = MahalanobisScorer()
    errs = []
    model.eval()
    with torch.no_grad():
        for batch, adj, _ in loader:
            batch = batch.to(device)
            pred, mask, _, lat = model(batch, adj.to(device), return_latent=True)
            tgt = batch.reshape(pred.shape)
            errs.append(((pred - tgt) ** 2).mean().item())
            mahal.accumulate(lat)
    mahal.finalize()
    errs = np.asarray(errs)
    dms = np.zeros(len(errs))
    mse_stats = (float(errs.mean()), float(errs.std()) + 1e-9)
    dm_stats = (0.0, 1.0)
    hyb, _, _ = hybrid_array(errs, dms, mse_stats, dm_stats, cfg)
    floor = float(np.percentile(hyb, cfg.hybrid_floor_percentile))
    floor = max(floor, cfg.hard_z_floor)
    print("  [target-calib] re-calibrated on %d target benign files (%d blocks): "
          "floor=%.3f" % (len(ben), len(errs), floor))
    json.dump({"target_floor": floor, "n_benign_files": len(ben), "n_blocks": len(errs),
               "mse_stats": mse_stats},
              open(os.path.join(output_dir, "target_calibration.json"), "w"), indent=2)
    return {"floor": floor, "mse_stats": mse_stats, "dm_stats": dm_stats, "mahal": mahal}


def write_calib_ablation_plan(cfg, output_dir):
    """Plan for the full calibration-component ablation (reviewer query): each cell of
    the sweep is run via the v7.7 toggles + percentile flag; this writes the manifest."""
    rows = []
    for pct in [95.0, 97.5, 99.0, 99.5, 99.9]:
        for comp, flag in [("full", ""), ("no_per_file_floor", "--no-per-file-floor"),
                           ("no_volume", "--no-volume"),
                           ("no_soft_triggers", "--no-soft-triggers"),
                           ("no_drift", "--no-drift")]:
            rows.append({"percentile": pct, "component": comp, "cli_flag": flag})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "calib_ablation_plan.csv"), index=False)
    print("  [calib-ablation] plan -> %s" % os.path.join(output_dir, "calib_ablation_plan.csv"))
    print("  [calib-ablation] run each row with: python paadu_benchmark_v7.py --data-dir <ds> "
          "--output-dir <out> --epochs 15 --percentile <pct> <flag>")


def run_paadu_v7(train_dir: str, test_dir: str | None, output_dir: str,
                 cfg: PAADUConfigV7):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(line(f"PAADU v7-benchmark v7.9 (reviewer-rectification: multi-seed, manifests, prediction dumps, calib baselines/ablation, mixed-stream, target-calib)  |  "
              f"train={train_dir}  test={test_dir or train_dir}"))
    print(line("Phase 0 — Dynamic Dataset Profiling (train side)"))

    profile = generate_profile(train_dir, cfg)
    if profile is None:
        raise RuntimeError(f"Could not generate profile from {train_dir}")
    profile.percentile = cfg.percentile
    profile.limit = cfg.bucket_limit

    print(f"  │  Dataset name       : {profile.name}")
    print(f"  │  num_nodes/seq_len  : {profile.num_nodes} / {profile.seq_len}")
    print(f"  │  transport          : {profile.transport}")
    print(f"  │  topology graph     : "
          f"{'communication (src->dst)' if profile.schema.get('src_col') else 'feature-chain (no endpoints)'}")
    print(f"  │  windowing          : {'time (%.0fs)' % cfg.window_seconds if cfg.window_seconds > 0 else 'row-blocks'}")
    print(f"  │  device             : {device}")
    if not profile.schema.get("src_col") or not profile.schema.get("dst_col"):
        print("  [WARN] no source/destination endpoint columns detected -> "
              "relational (communication) perception DISABLED; using feature-chain graph. "
              "Check that 'Src IP'/'Dst IP' (or CAN ID) columns exist in the CSVs.")
    if not profile.schema.get("ts_col"):
        print("  [WARN] no timestamp column detected -> temporal windowing unavailable; "
              "using row-block slicing.")
    if cfg.mask_ratio > 0.6:
        print(f"  [WARN] mask_ratio={cfg.mask_ratio:.2f} keeps only "
              f"{int(cfg.target_seq_len * (1 - cfg.mask_ratio))} of {cfg.target_seq_len} tokens per node - "
              "reconstruction may be unstable; recommended 0.40-0.50.")

    all_train = sorted(p for p in glob.glob(os.path.join(train_dir, "**", "*.csv"),
                                                 recursive=True))
    train_files = [f for f in all_train if is_benign_file(os.path.basename(f), profile.benign_keywords)]
    mixed_mode = False
    if cfg.max_train_files is not None:
        train_files = train_files[: cfg.max_train_files]
    if not train_files:
        print("  [WARN] no benign files found by filename; checking label column …")
        # MIXED-FILE MODE: single-file datasets (TON_IoT, UNSW-NB15, ...) carry
        # benign + attack rows in the SAME CSV. Any file whose label column
        # contains benign rows can serve as the benign training source —
        # PADataset(benign_only=True) filters rows by label.
        for f in all_train:
            try:
                df = pd.read_csv(f, nrows=100_000, low_memory=False)
                s = detect_schema(df, os.path.basename(f))
                lc = s.get("label_col") or profile.schema.get("label_col")
                if lc and lc in df.columns:
                    bkw = find_benign_keywords(df, s, os.path.basename(f))
                    # count benign over the ENTIRE label column (files may be
                    # label-sorted: first 100k rows can be 100% benign even
                    # when the file is balanced). Reading one column is cheap.
                    try:
                        lbl_full = pd.read_csv(f, usecols=[lc], low_memory=False)[lc]
                        n_total = len(lbl_full)
                        n_ben = _count_benign_rows(lbl_full, bkw)
                    except Exception:  # noqa: BLE001
                        n_total, n_ben = len(df), _count_benign_rows(df[lc], bkw)
                    # diagnostics: is the file label-sorted?
                    sample_ben = _count_benign_rows(df[lc], bkw)
                    if n_total > 0 and sample_ben == len(df) and n_ben < n_total * 0.99:
                        print(f"  [note] {os.path.basename(f)}: first 100k rows all benign "
                              f"but only {100*n_ben/max(1,n_total):.0f}% benign overall - "
                              "file is LABEL-SORTED (benign rows first). Benign filtering "
                              "applies to the full file; the count below is the true one.")
                    if n_ben >= 100:
                        train_files.append(f)
                        mixed_mode = True
                        print(f"  [OK] {os.path.basename(f)}: {n_ben:,}/{n_total:,} rows "
                              f"benign ({100*n_ben/max(1,n_total):.1f}%) (label col '{lc}') "
                              "-> mixed-file benign training")
            except Exception:  # noqa: BLE001
                continue
    if not train_files:
        raise RuntimeError("No benign training files found.")
    if mixed_mode:
        print("  [MIXED-FILE MODE] benign and attack rows share the same file(s); "
              "training/calibration use benign ROWS only (label-filtered).")
    if cfg.split_manifest:
        write_split_manifest(cfg, profile, train_files, output_dir)
        if not cfg.holdout_benign_eval:
            print("  [RECOMMEND] --holdout: train/calibrate on the first 70% of rows and "
                  "evaluate benign FPR on the held-out last 30%. Without it, the "
                  "training file's own benign rows are re-evaluated (closed-set) and "
                  "benign FPR is understated.")
    print(f"  Training/calibration benign files ({len(train_files)}):")
    for f in train_files:
        print(f"    - {os.path.basename(f)}")

    print(line("Phase 1 — Normalcy Learning (benign-only)"))
    frac = ((0.0, cfg.benign_train_fraction) if cfg.holdout_benign_eval else None)
    full_benign_ds = PADataset(train_files, profile, nrows=cfg.train_nrows_per_file,
                               benign_only=True, row_fraction=frac,
                               window_seconds=cfg.window_seconds, verbose=True)
    n_total = len(full_benign_ds)
    print(f"  Total benign blocks  : {n_total:,}")
    print(f"  Feature dimension    : {full_benign_ds.features.shape[1]}")
    if n_total < 2:
        raise RuntimeError("Not enough benign blocks to train/validate.")
    if cfg.baselines_only:
        print(line("BASELINES-ONLY MODE - PAADU training/calibration/evaluation skipped; "
                   "paadu_summary.csv untouched"))
        test_dir = test_dir or train_dir
        test_files = sorted(p for p in glob.glob(os.path.join(test_dir, "**", "*.csv"),
                                                  recursive=True))
        if not cfg.run_baselines:
            cfg.run_baselines = True
        run_baselines(train_files, test_files, full_benign_ds, profile, cfg, output_dir)
        print(line("Baselines-only complete"))
        return None, None
    n_val = max(1, int(n_total * 0.20))
    n_train = max(1, n_total - n_val)
    train_sub, val_sub = random_split(full_benign_ds, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_sub, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_sub, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    print(f"  Train blocks: {n_train:,} | Val blocks: {n_val:,}")

    model = SemanticAnomalyMAE(profile.num_nodes, profile.seq_len,
                               full_benign_ds.features.shape[1],
                               embed_dim=cfg.embed_dim, mask_ratio=cfg.mask_ratio,
                               use_gnn=cfg.use_gnn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=0.5, patience=3)
    tr_l, va_l, stopped = train_with_early_stopping(
        model, optimizer, scheduler, train_loader, val_loader, device,
        cfg.epochs, cfg.early_stop_patience, output_dir)
    if HAS_VIZ:
        _plot_setup()
        plot_training_curve(tr_l, va_l, stopped, os.path.join(output_dir, "training_curve.png"))

    print(line("Phase 2 — Threshold Calibration + Mahalanobis"))
    threshold, calib_errors, mahal, mse_stats, dm_stats, relaxer = calibrate_threshold_v7(
        model, full_benign_ds, train_files, profile, device, cfg, output_dir)
    if HAS_VIZ:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(calib_errors, bins=80, color=PALETTE["safe"], alpha=0.75)
        ax.axvline(threshold, color=PALETTE["danger"], lw=2, ls="--")
        ax.set_xlabel("Reconstruction Error (MSE)")
        ax.set_title("Benign Error Distribution + Threshold", fontweight="bold", pad=12)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "threshold_dist.png"), dpi=150,
                    bbox_inches="tight", facecolor=PALETTE["bg"])
        plt.close(fig)

    with open(os.path.join(output_dir, "paadu_v7_config.json"), "w") as f:
        json.dump({"config": {k: v for k, v in cfg.__dict__.items()
                              if not callable(v)},
                   "profile": {"name": profile.name, "num_nodes": profile.num_nodes,
                               "seq_len": profile.seq_len,
                               "features": profile.feature_dim,
                               "transport": profile.transport,
                               "schema": {k: (v if not isinstance(v, list) else v)
                                          for k, v in profile.schema.items()}},
                   "mse_stats": mse_stats, "dm_stats": dm_stats,
                   "z_floor": relaxer.z_floor}, f, indent=2)

    # ---- v7.9 target calibration (cross-domain) ----
    if cfg.target_calib and test_dir and os.path.abspath(test_dir) != os.path.abspath(train_dir):
        tc = run_target_calibration(model, profile, full_benign_ds.scaler,
                                    full_benign_ds.train_columns, test_dir, cfg,
                                    device, output_dir)
        if tc is not None:
            relaxer.set_floor_value(tc["floor"], reason="target-calibrated (domain-adapted)")
            mse_stats = tc["mse_stats"]
            dm_stats = tc["dm_stats"]
            mahal = tc["mahal"]
            threshold = tc["floor"]

    # ---- v7.9 multi-seed: per-seed training/calibration/evaluation ----
    if len(cfg.seeds) > 1:
        test_dir_l = test_dir or train_dir
        test_files_l = sorted(p for p in glob.glob(os.path.join(test_dir_l, "**", "*.csv"),
                                                    recursive=True))
        print(line("Phase 2b — Multi-seed runs (%d seeds)" % len(cfg.seeds)))
        per_seed = {}
        for sd in cfg.seeds:
            seed_everything(sd)
            m = SemanticAnomalyMAE(profile.num_nodes, profile.seq_len,
                                   full_benign_ds.features.shape[1],
                                   embed_dim=cfg.embed_dim, mask_ratio=cfg.mask_ratio,
                                   use_gnn=cfg.use_gnn).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
            train_with_early_stopping(m, opt, sch, train_loader, val_loader, device,
                                      min(cfg.epochs, 15), cfg.early_stop_patience, output_dir)
            th2, _, mah2, ms2, dm2, rx2 = calibrate_threshold_v7(
                m, full_benign_ds, train_files, profile, device, cfg, output_dir)
            ev2 = RichEvaluator(m, profile, full_benign_ds.scaler,
                                full_benign_ds.train_columns, device, th2,
                                mah2, ms2, dm2, rx2, cfg)
            rows2, det2 = [], []
            for fpath in test_files_l:
                r = ev2.evaluate_file(fpath, nrows=cfg.nrows_test)
                if r is None:
                    continue
                rows2.append({"file": r["file"], "alert_pct": r["alert_pct"],
                              "file_score_pct": r["file_score_pct"],
                              "blocks": r["blocks"],
                              "alerts": int(r["df"]["alert"].sum())})
                for _, row in r["df"].iterrows():
                    if _row_is_attack(row["label"]):
                        det2.append((row["label"], int(row["alert"])))
            df2 = pd.DataFrame(rows2)
            atk2 = df2[~df2["file"].str.lower().str.contains("benign")]
            ben2 = df2[df2["file"].str.lower().str.contains("benign")]
            tp2 = int((atk2["file_score_pct"] >= cfg.file_attack_threshold_pct).sum())
            fn2 = int((atk2["file_score_pct"] < cfg.file_attack_threshold_pct).sum())
            fp2 = int((ben2["file_score_pct"] >= cfg.file_attack_threshold_pct).sum())
            f1_2 = 2 * tp2 / max(1e-9, 2 * tp2 + fp2 + fn2)
            atk_al = int(atk2["alerts"].sum()); atk_bl = int(atk2["blocks"].sum())
            ben_al = int(ben2["alerts"].sum()); ben_bl = int(ben2["blocks"].sum())
            per_seed[str(sd)] = {"file_f1": round(f1_2, 4),
                                 "tp": tp2, "fp": fp2, "fn": fn2,
                                 "block_tpr_pct": round(100 * atk_al / max(1, atk_bl), 2),
                                 "block_fpr_pct": round(100 * ben_al / max(1, ben_bl), 2)}
            print("  [seed %d] file_f1=%.4f  block_TPR=%.1f%%  block_FPR=%.1f%%" %
                  (sd, f1_2, per_seed[str(sd)]["block_tpr_pct"], per_seed[str(sd)]["block_fpr_pct"]))
        import json as _json
        _json.dump({"per_seed": per_seed}, open(os.path.join(output_dir, "multiseed_paadu.json"), "w"),
                   indent=2)
        print("  [multiseed] -> %s" % os.path.join(output_dir, "multiseed_paadu.json"))

    print(line("Phase 3 — Evaluation"))
    evaluator = RichEvaluator(model, profile, full_benign_ds.scaler,
                              full_benign_ds.train_columns, device, threshold,
                              mahal, mse_stats, dm_stats, relaxer, cfg)
    test_dir = test_dir or train_dir
    test_files = sorted(p for p in glob.glob(os.path.join(test_dir, "**", "*.csv"),
                                                  recursive=True))
    print(f"  Files to evaluate : {len(test_files)}")

    summary_rows, detailed = [], []
    benign_holdout_rows = []
    header = f"{'File':<45} | {'Alert%':>7} | {'Score%':>7} | {'Blocks':>6} | {'Time':>6} | Cause"
    print(header); print("─" * len(header))
    for fpath in test_files:
        if mixed_mode and cfg.holdout_benign_eval:
            # single mixed file: evaluate the held-out tail TWICE —
            # benign-only rows (clean benign FPR) and all rows (TPR + per-type)
            rb = evaluator.evaluate_file(fpath, nrows=None,
                                         row_fraction=(cfg.benign_train_fraction, 1.0),
                                         benign_only=True)
            if rb is not None:
                benign_holdout_rows.append(rb)
                print(f"  [benign holdout] {os.path.basename(fpath):<45} | "
                      f"alert {rb['alert_pct']:>6.2f}% | blocks {rb['blocks']:>6}")
            res = evaluator.evaluate_file(fpath, nrows=None,
                                          row_fraction=(cfg.benign_train_fraction, 1.0))
        elif cfg.holdout_benign_eval and is_benign_file(os.path.basename(fpath),
                                                        profile.benign_keywords):
            res = evaluator.evaluate_file(fpath, nrows=None,
                                          row_fraction=(cfg.benign_train_fraction, 1.0))
        else:
            res = evaluator.evaluate_file(fpath, nrows=cfg.nrows_test)
        if res is None:
            continue
        if cfg.dump_predictions:
            dump_predictions_to_csv(res, output_dir)
    
        if cfg.small_file_evidence and res["blocks"] < cfg.small_file_blocks \
                and res["file_basis"] == "alert":
            m = max(res["alert_pct"], res["fires_pct"], res["candidate_pct"])
            if m == res["candidate_pct"]:
                res["file_basis"] = "candidate"
            elif m == res["fires_pct"]:
                res["file_basis"] = "fires"
            res["file_score_pct"] = m
        df = res["df"]
        label_stats = defaultdict(lambda: {"total": 0, "detected": 0})
        cause_counts = {"burst": 0, "drift": 0}
        for _, row in df.iterrows():
            label_stats[row["label"]]["total"] += 1
            if row["alert"]:
                label_stats[row["label"]]["detected"] += 1
                for ck in cause_counts:
                    if ck in row["cause"]:
                        cause_counts[ck] += 1
        dom_cause = max(cause_counts, key=cause_counts.get) if any(cause_counts.values()) else "—"
        summary_rows.append({"file": res["file"], "alert_pct": res["alert_pct"],
                             "fires_pct": res["fires_pct"],
                             "candidate_pct": res["candidate_pct"],
                             "file_score_pct": res["file_score_pct"],
                             "file_basis": res["file_basis"], "blocks": res["blocks"],
                             "elapsed": res["elapsed"], "cause": dom_cause,
                             "label_stats": dict(label_stats),
                             "burst_alerts": cause_counts["burst"],
                             "drift_alerts": cause_counts["drift"],
                             "z_thresh": res["z_thresh"], "df": df})
        print(f"{res['file'][:45]:<45} | {res['alert_pct']:>6.2f}% | "
              f"{res['file_score_pct']:>6.2f}% | {res['blocks']:>6} | "
              f"{res['elapsed']:>5.2f}s | B:{cause_counts['burst']} D:{cause_counts['drift']}")
        for lbl, st in sorted(label_stats.items(), key=lambda kv: -kv[1]["total"])[:cfg.max_label_lines_per_file]:
            det = st["detected"] / st["total"] * 100 if st["total"] else 0.0
            detailed.append({"file": res["file"], "label": lbl, "total_blocks": st["total"],
                             "detected_blocks": st["detected"], "detection_pct": det})
            print(f"   {'✓' if det >= 50 else '✗'}  {lbl[:48]:<48}: {st['detected']}/{st['total']} ({det:.1f}%)")

    print("─" * len(header))

    # ── Phase 4: report + metrics ────────────────────────────────────────
    print(line("Phase 4 — Summary & Metrics"))
    report = [{k: v for k, v in r.items() if k not in ("df", "label_stats")}
              for r in summary_rows]
    report_df = pd.DataFrame(report)
    report_df.to_csv(os.path.join(output_dir, "paadu_summary.csv"), index=False)
    if detailed:
        pd.DataFrame(detailed).to_csv(os.path.join(output_dir, "paadu_label_stats.csv"),
                                      index=False)

    def _is_attack_label(s: str) -> bool:
        v = str(s).lower()
        return not ("benign" in v or v in ("normal", "normal traffic", "legitimate"))

    def _file_is_attack(r) -> int:
        """File GT from block labels when available (fallback: filename).

        A file counts as attack if ANY block is attack-labelled (CIC file
        semantics: attack files are attack-dominant; mixed files contain
        attack traffic and should be detected).
        """
        ls = r.get("label_stats") or {}
        if ls:
            attack_blocks = sum(v["total"] for k, v in ls.items() if _is_attack_label(k))
            benign_blocks = sum(v["total"] for k, v in ls.items() if not _is_attack_label(k))
            if attack_blocks > 0:
                return 1
            if benign_blocks > 0:
                return 0
        return 0 if is_benign_file(r["file"], profile.benign_keywords) else 1

    true_benign = [r for r in summary_rows if not _file_is_attack(r)]
    true_attack = [r for r in summary_rows if _file_is_attack(r)]
    pred_attack = [r for r in summary_rows if r["file_score_pct"] >= cfg.file_attack_threshold_pct]

    # block-level metrics from per-block labels (exact, label-column based)
    b_alerts = b_blocks = a_alerts = a_blocks = 0
    for r in summary_rows:
        df = r["df"]
        for _, row in df.iterrows():
            if _is_attack_label(row["label"]):
                a_blocks += 1
                a_alerts += int(row["alert"])
            else:
                b_blocks += 1
                b_alerts += int(row["alert"])
   
    if mixed_mode and benign_holdout_rows:
        b_alerts = sum(int(r["df"]["alert"].sum()) for r in benign_holdout_rows)
        b_blocks = sum(len(r["df"]) for r in benign_holdout_rows)
    fpr = b_alerts / max(b_blocks, 1) * 100
    tpr = a_alerts / max(a_blocks, 1) * 100
    print(f"  Files: {len(summary_rows)}  |  true benign {len(true_benign)}  |  true attack {len(true_attack)}")
    print(f"  Block-level: benign FPR = {fpr:.2f}% ({b_alerts}/{b_blocks})   "
          f"attack TPR = {tpr:.2f}% ({a_alerts}/{a_blocks})"
          + ("   [benign FPR from benign-only held-out tail]" if mixed_mode and benign_holdout_rows else ""))

    y_true = [_file_is_attack(r) for r in summary_rows]
    y_pred = [1 if r["file_score_pct"] >= cfg.file_attack_threshold_pct else 0
              for r in summary_rows]
    if len(y_true) > 1 and len(set(y_true)) > 1:
        acc = accuracy_score(y_true, y_pred) * 100
        prec = precision_score(y_true, y_pred, zero_division=0) * 100
        rec = recall_score(y_true, y_pred, zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, zero_division=0) * 100
        mcc = matthews_corrcoef(y_true, y_pred)
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        print(f"  File-level: TP={tp} FP={fp} TN={tn} FN={fn}")
        print(f"  Accuracy={acc:.2f}%  Precision={prec:.2f}%  Recall={rec:.2f}%  "
              f"F1={f1:.2f}%  MCC={mcc:.4f}")
    else:
        print("  File-level: insufficient class diversity for metrics.")
        if len(summary_rows) == 1:
            print("  [note] single mixed file (e.g. TON_IoT): block-level FPR/TPR are "
                  "the primary metrics; file-level is degenerate by construction.")

    # attack-family breakdown (paper table — spec section 42).
    # Families come from the IN-FILE label when a label column exists
    # (e.g. CIC-2025 'label_full'), else from the filename.
    use_infile_labels = bool(profile.schema.get("label_col"))
    # generic in-file labels ('attack'/'1' with no attack-type info, e.g.
    # CICIoV2024) carry no family signal -> fall back to the filename
    if use_infile_labels:
        non_ben = set()
        for r in summary_rows:
            for lbl in (r.get("label_stats") or {}):
                if _family_of_label(lbl) != "Benign":
                    non_ben.add(str(lbl).strip().lower())
        generic = bool(non_ben) and non_ben <= {"attack", "1", "1.0", "abnormal",
                                                "anomaly", "intrusion", "malicious"}
        if generic:
            use_infile_labels = False
            print("  [note] in-file label is generic (no attack type) -> family "
                  "grouping falls back to filenames")
    fam = defaultdict(lambda: {"files": 0, "files_det": 0, "attack_files": 0,
                               "blocks": 0, "det_blocks": 0})
    missed = []
    for r, yt, yp in zip(summary_rows, y_true, y_pred):
        if use_infile_labels and r.get("label_stats"):
            fams = set()
            for lbl, st in r["label_stats"].items():
                f = _family_of_label(lbl)
                fam[f]["blocks"] += st["total"]
                fam[f]["det_blocks"] += st["detected"]
                fams.add(f)
            for f in fams:
                fam[f]["files"] += 1
                if f == "Benign":
                    if yp == 1:
                        fam[f]["files_det"] += 1   # benign FP
                else:
                    fam[f]["attack_files"] += 1
                    if yp == 1:
                        fam[f]["files_det"] += 1
                    elif yp == 0:
                        missed.append(r["file"])
        else:
            f = attack_family(r["file"])
            st = fam[f]
            st["files"] += 1
            st["blocks"] += r["blocks"]
            st["det_blocks"] += int(r["df"]["alert"].sum())
            if yp == 1:
                st["files_det"] += 1
            if yt == 1:
                st["attack_files"] += 1
                if yp == 0:
                    missed.append(r["file"])
    print("\n  ── Attack-family breakdown ──")
    print(f"  {'Family':<14}{'Files':>6}{'Det':>5}{'Blocks':>9}{'DetB':>8}"
          f"{'BlockTPR':>10}{'FileTPR':>9}")
    fam_rows = []
    for f, st in sorted(fam.items(),
                        key=lambda kv: -kv[1]["det_blocks"] / max(1, kv[1]["blocks"])):
        btpr = 100 * st["det_blocks"] / max(1, st["blocks"])
        ftpr = (100 * st["files_det"] / max(1, st["attack_files"])
                if st["attack_files"] else float("nan"))
        ftpr_s = f"{ftpr:>8.1f}%" if st["attack_files"] else f"{'—':>9}"
        note = "  (benign FPR)" if f == "Benign" else ""
        print(f"  {f:<14}{st['files']:>6}{st['files_det']:>5}{st['blocks']:>9}"
              f"{st['det_blocks']:>8}{btpr:>9.1f}%{ftpr_s}{note}")
        fam_rows.append({"family": f, "n_files": st["files"],
                         "files_detected": st["files_det"],
                         "n_blocks": st["blocks"], "detected_blocks": st["det_blocks"],
                         "block_tpr_pct": round(btpr, 2),
                         "file_tpr_pct": round(ftpr, 2)})
    if fam_rows:
        pd.DataFrame(fam_rows).to_csv(os.path.join(output_dir,
                                                   "paadu_attack_families.csv"),
                                      index=False)

    # per-attack-type breakdown (spec 42) — straight from in-file labels
    if use_infile_labels:
        atype = defaultdict(lambda: {"files": 0, "files_det": 0,
                                     "blocks": 0, "det_blocks": 0})
        for r in summary_rows:
            ls = r.get("label_stats") or {}
            pred = 1 if r["file_score_pct"] >= cfg.file_attack_threshold_pct else 0
            for lbl, st in ls.items():
                if _family_of_label(lbl) == "Benign":
                    continue
                atype[lbl]["files"] += 1
                atype[lbl]["blocks"] += st["total"]
                atype[lbl]["det_blocks"] += st["detected"]
                if pred:
                    atype[lbl]["files_det"] += 1
        atype_rows = []
        for lbl, st in sorted(atype.items(), key=lambda kv: -kv[1]["blocks"]):
            atype_rows.append({"attack_type": lbl, "n_files": st["files"],
                               "files_detected": st["files_det"],
                               "n_blocks": st["blocks"],
                               "detected_blocks": st["det_blocks"],
                               "block_tpr_pct": round(100 * st["det_blocks"] / max(1, st["blocks"]), 2),
                               "file_tpr_pct": round(100 * st["files_det"] / max(1, st["files"]), 2)})
        if atype_rows:
            pd.DataFrame(atype_rows).to_csv(os.path.join(output_dir,
                                                         "paadu_attack_types.csv"),
                                            index=False)
            print("  ── Per-attack-type breakdown (from in-file labels) ──")
            print(f"  {'Attack type':<44}{'Files':>6}{'Det':>5}{'Blocks':>8}"
                  f"{'DetB':>7}{'BlockTPR':>10}")
            for row in atype_rows[:15]:
                print(f"  {row['attack_type'][:44]:<44}{row['n_files']:>6}"
                      f"{row['files_detected']:>5}{row['n_blocks']:>8}"
                      f"{row['detected_blocks']:>7}{row['block_tpr_pct']:>9.1f}%")
            if len(atype_rows) > 15:
                print(f"  … {len(atype_rows) - 15} more attack types in paadu_attack_types.csv")
    if missed:
        print(f"\n  Files missed at file level ({len(missed)}): {', '.join(missed)}")

    # baselines (spec 33-36) on the same benign-only pipeline
    if cfg.run_baselines:
        run_baselines(train_files, test_files, full_benign_ds, profile, cfg, output_dir)

    # v7.9: calibration baselines + mixed-stream + metric verification
    if cfg.calib_ablation:
        write_calib_ablation_plan(cfg, output_dir)
    if cfg.calib_baselines:
        run_calibration_baselines(evaluator, test_files, cfg, output_dir)
    if cfg.mixed_stream:
        run_mixed_stream(evaluator, test_files, cfg, output_dir)
    if cfg.verify_metrics:
        verify_metrics_from_predictions(output_dir, file_threshold=cfg.file_attack_threshold_pct)

    if HAS_VIZ and summary_rows:
        plot_alert_rates(summary_rows, os.path.join(output_dir, "alert_rates.png"),
                         cfg.file_attack_threshold_pct)
        plot_hybrid_scatter(summary_rows, cfg, os.path.join(output_dir, "hybrid_scatter.png"))
        top = max(summary_rows, key=lambda r: r["alert_pct"])
        plot_error_timeline(top, os.path.join(output_dir,
                                              f"timeline_{_safe_name(top['file'])}.png"))
    print(line(f"✓  Complete — outputs in {output_dir}"))
    return report_df, model


def build_cfg_from_args(args) -> PAADUConfigV7:
    return PAADUConfigV7(
        epochs=args.epochs, batch_size=args.batch_size, embed_dim=args.embed_dim,
        mask_ratio=args.mask_ratio, use_gnn=not args.no_gnn,
        percentile=args.percentile,
        hard_z_floor=args.hard_z_floor, w_recon=args.w_recon, w_mahal=args.w_mahal,
        window_seconds=args.window_seconds, holdout_benign_eval=args.holdout,
        train_nrows_per_file=args.train_nrows, nrows_test=args.test_nrows,
        max_train_files=args.max_train_files,
        file_attack_threshold_pct=args.file_threshold,
        use_file_evidence_rule=args.evidence_rule,
        use_per_file_max_floor=not args.no_per_file_floor,
        volume_adaptive=not args.no_volume,
        soft_triggers=not args.no_soft_triggers,
        use_drift=not args.no_drift,
        small_file_evidence=not args.no_small_file_evidence,
        seeds=tuple(int(x) for x in args.seeds.split(",")) if args.seeds else (42,),
        dump_predictions=args.dump_predictions,
        split_manifest=args.split_manifest,
        calib_ablation=args.calib_ablation,
        calib_baselines=args.calib_baselines,
        mixed_stream=args.mixed_stream,
        baseline_tune=args.baseline_tune,
        verify_metrics=args.verify_metrics,
        target_calib=args.target_calib,
        small_file_blocks=args.small_file_blocks,
        run_baselines=args.baselines or args.baselines_only,
        baselines_only=args.baselines_only,
        baseline_epochs=args.baseline_epochs,
        output_dir=args.output_dir,
    )


def main():
    ap = argparse.ArgumentParser(description="PAADU v7-benchmark (dataset-agnostic)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("PAADU_DATA_PATH", ""),
                    help="dataset directory (single-dataset mode)")
    ap.add_argument("--train-dir", default="",
                    help="train/calibrate directory (cross-domain mode)")
    ap.add_argument("--test-dir", default="",
                    help="evaluate directory (cross-domain mode)")
    ap.add_argument("--output-dir",
                    default=os.environ.get("PAADU_OUTPUT_DIR", "paadu_v7_outputs"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--mask-ratio", type=float, default=0.40)
    ap.add_argument("--no-gnn", action="store_true",
                    help="disable graph attention (feature-grid only; no relational perception)")
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--hard-z-floor", type=float, default=2.5)
    ap.add_argument("--w-recon", type=float, default=0.20)
    ap.add_argument("--w-mahal", type=float, default=0.80)
    ap.add_argument("--window-seconds", type=float, default=0.0,
                    help=">0 + timestamp column -> time windows (0 = row blocks)")
    ap.add_argument("--train-nrows", type=int, default=None)
    ap.add_argument("--test-nrows", type=int, default=30_000)
    ap.add_argument("--max-train-files", type=int, default=None)
    ap.add_argument("--file-threshold", type=float, default=20.0)
    ap.add_argument("--evidence-rule", action="store_true",
                    help="file decision may use fires/candidate evidence")
    ap.add_argument("--no-small-file-evidence", action="store_true",
                    help="disable the small-file evidence rule (short files use "
                         "alert rate only)")
    ap.add_argument("--no-per-file-floor", action="store_true",
                    help="calibration ablation: disable per-benign-file max floor")
    ap.add_argument("--no-volume", action="store_true",
                    help="calibration ablation: disable volume-adaptive threshold")
    ap.add_argument("--no-soft-triggers", action="store_true",
                    help="calibration ablation: disable soft-trigger relaxations")
    ap.add_argument("--no-drift", action="store_true",
                    help="calibration ablation: disable the drift tracker (fires only)")
    ap.add_argument("--baselines", action="store_true",
                    help="also train+eval AE / LSTM-AE / Transformer-AE / IsolationForest")
    ap.add_argument("--baselines-only", action="store_true",
                    help="skip PAADU training/calibration/evaluation entirely; run "
                         "baselines only (does NOT touch paadu_summary.csv)")
    ap.add_argument("--baseline-epochs", type=int, default=15)
    ap.add_argument("--seeds", type=str, default="42",
                    help="comma-separated random seeds, e.g. 1,2,3,5,7 "
                         "(multi-seed: per-seed metrics + mean/std)")
    ap.add_argument("--dump-predictions", action="store_true",
                    help="write per-file per-window predictions CSV (scores, labels, alerts)")
    ap.add_argument("--split-manifest", action="store_true",
                    help="write per-file train/calibration/test row manifests (JSON)")
    ap.add_argument("--calib-ablation", action="store_true",
                    help="sweep calibration components (per-file floor, volume, soft, drift) "
                         "and floor percentiles P95..P99.9 -> CSV")
    ap.add_argument("--calib-baselines", action="store_true",
                    help="evaluate standard calibration baselines: fixed-z (2.5, 3.0), "
                         "global-p99-only, per-file-p99-only")
    ap.add_argument("--mixed-stream", action="store_true",
                    help="concatenate all test files into one stream; drift tracker runs "
                         "across the stream (temporal-drift / mixed-stream deployment eval)")
    ap.add_argument("--baseline-tune", action="store_true",
                    help="small hyperparameter grid (hidden, lr) for AE/LSTM-AE/T-AE baselines")
    ap.add_argument("--verify-metrics", action="store_true",
                    help="recompute all metrics from dumped per-file predictions and cross-check")
    ap.add_argument("--target-calib", action="store_true",
                    help="cross-domain: re-calibrate the anomaly floor on TARGET benign files "
                         "before evaluating the target (domain-adapted normality model)")
    ap.add_argument("--small-file-blocks", type=int, default=150,
                    help="files with fewer blocks use fires/candidate evidence "
                         "for the file decision (noisy alert-rate estimates)")
    ap.add_argument("--holdout", action="store_true",
                    help="held-out benign validation (last 30% of benign files)")
    ap.add_argument("--quick", action="store_true",
                    help="quick run: 4 epochs, capped rows")
    args = ap.parse_args()

    train_dir = args.train_dir or args.data_dir
    test_dir = args.test_dir or (None if args.train_dir else None)
    if not train_dir:
        ap.error("provide --data-dir (or --train-dir/--test-dir)")
    if not os.path.isdir(train_dir):
        ap.error(f"train dir not found: {train_dir}")
    if args.test_dir and not args.train_dir:
        ap.error("--test-dir requires --train-dir")
    if not test_dir:
        test_dir = train_dir
    if args.quick:
        args.epochs = min(args.epochs, 4)
        args.train_nrows = args.train_nrows or 20_000
        args.test_nrows = min(args.test_nrows, 20_000)

    seed_everything(42)
    cfg = build_cfg_from_args(args)
    run_paadu_v7(train_dir, test_dir, args.output_dir, cfg)


if __name__ == "__main__":
    main()
