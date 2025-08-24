# -*- coding: utf-8 -*-
"""
Utility di supporto per preprocessing: timestamp, sessioni, hashing cache,
aggregazione features.
"""

import os
import glob
import json
import hashlib
from datetime import timedelta
from typing import List, Dict, Any

import numpy as np
import pandas as pd


def is_malicious(label: str, benign_label: str = "BENIGN") -> bool:
    return pd.notna(label) and str(label) != str(benign_label)


def ensure_timestamp(df: pd.DataFrame, ts_col: str, ts_fmt: str | None = None) -> pd.DataFrame:
    if ts_col not in df.columns:
        return df
    if not np.issubdtype(df[ts_col].dtype, np.datetime64):
        if ts_fmt:
            df[ts_col] = pd.to_datetime(df[ts_col], format=ts_fmt, errors='coerce')
        else:
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    return df


def sessionize_flows(
    df: pd.DataFrame,
    flow_col: str,
    ts_col: str,
    timeout_seconds: int = 60
) -> pd.DataFrame:
    if flow_col not in df.columns or ts_col not in df.columns:
        return df

    df = ensure_timestamp(df, ts_col)
    df = df.sort_values([flow_col, ts_col]).reset_index(drop=True)

    session_indices: List[int] = []
    current_idx = -1
    last_flow = None
    last_ts = None
    timeout = timedelta(seconds=timeout_seconds)

    for flow_id, ts in df[[flow_col, ts_col]].itertuples(index=False, name=None):
        if (flow_id != last_flow) or (last_ts is None) or (pd.isna(ts)) or (pd.isna(last_ts)) or (ts - last_ts > timeout):
            current_idx += 1
        session_indices.append(current_idx)
        last_flow = flow_id
        last_ts = ts

    df["Session_Index"] = session_indices
    df["Session_ID"] = df[flow_col].astype(str) + "#" + df["Session_Index"].astype(str)
    return df


def json_hash(data: Dict[str, Any]) -> str:
    try:
        payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        payload = str(data).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:10]


def compute_base_cache_dir(
    data_path: str,
    sample_size: int | None,
    balance_strategy: str,
    benign_ratio: float,
    feature_columns: List[str],
    convert_ip_to_octets: bool,
    cache_root: str = "preprocessed_cache",
    target_type: str | None = None,
) -> str:
    os.makedirs(cache_root, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    files_meta = []
    for fp in csv_files:
        try:
            files_meta.append({
                "name": os.path.basename(fp),
                "size": os.path.getsize(fp),
                "mtime": int(os.path.getmtime(fp))
            })
        except Exception:
            files_meta.append({"name": os.path.basename(fp)})
    signature = {
        "files": files_meta,
        "sample_size": int(sample_size) if sample_size is not None else None,
        "balance_strategy": balance_strategy,
        "benign_ratio": float(benign_ratio),
        "feature_columns": feature_columns,
        "convert_ip_to_octets": bool(convert_ip_to_octets),
        "target_type": target_type,
    }
    key = json_hash(signature)
    return os.path.join(cache_root, f"base_{key}")


def compute_windows_cache_dir(
    base_dir: str,
    params: Dict[str, Any]
) -> str:
    key = json_hash(params)
    return os.path.join(base_dir, "windows", key)


def aggregate_features(df_window: pd.DataFrame, feature_cols: List[str], stats: List[str]) -> np.ndarray:
    agg: Dict[tuple, float] = {}
    for col in feature_cols:
        series = pd.to_numeric(df_window[col], errors='coerce').fillna(0)
        for st in stats:
            if st == "sum":
                agg[(col, "sum")] = float(series.sum())
            elif st == "mean":
                agg[(col, "mean")] = float(series.mean())
            elif st == "std":
                agg[(col, "std")] = float(series.std(ddof=0))
            elif st == "min":
                agg[(col, "min")] = float(series.min())
            elif st == "max":
                agg[(col, "max")] = float(series.max())
    ordered: List[float] = []
    for col in feature_cols:
        for st in stats:
            ordered.append(agg[(col, st)])
    return np.array(ordered, dtype=float)


