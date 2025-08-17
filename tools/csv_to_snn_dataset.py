#!/usr/bin/env python3
"""
Utility: converti un CSV generico in dataset SNN conforme alla repo
- Aggiunge colonna 'timestamp' se assente
- Normalizza colonne numeriche in [0,1]
- One-hot sulle categoriali fino a max_categories (oltre: hashing numerico)
- Prefissa le colonne con 'feat_'

Uso CLI:
python -m tools.csv_to_snn_dataset --input /path/raw.csv --output /path/snn_dataset.csv \
    [--label-col Label] [--max-categories 20]
"""

from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def _hash_to_bucket(values: pd.Series, buckets: int) -> np.ndarray:
    def h(x: str) -> int:
        return int(hashlib.sha1(x.encode('utf-8', errors='ignore')).hexdigest(), 16) % buckets
    return values.fillna('NA').astype(str).map(h).values.astype(np.int64)


def convert_csv_to_snn_dataset(
    input_csv: str,
    output_csv: str,
    label_col: Optional[str] = None,
    max_categories: int = 20,
) -> str:
    df = pd.read_csv(input_csv)

    # Rimuovi colonna label se presente
    if label_col and label_col in df.columns:
        df = df.drop(columns=[label_col])

    # Timestamp: usa se presente, altrimenti genera indice temporale fittizio
    timestamp_col = None
    for cand in [
        'timestamp', 'time', 'eventtime', 'rt', 'Event Time', 'date']:
        if cand in df.columns:
            timestamp_col = cand
            break
    if timestamp_col is None:
        df['timestamp'] = np.arange(len(df), dtype=np.int64)
        timestamp_col = 'timestamp'

    # Seleziona numeriche e categoriali
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if timestamp_col in numeric_cols:
        numeric_cols.remove(timestamp_col)
    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols and c != timestamp_col
    ]

    # Normalizza numeriche
    if numeric_cols:
        scaler = MinMaxScaler()
        X_num = scaler.fit_transform(df[numeric_cols].fillna(0).values.astype(float))
    else:
        X_num = np.zeros((len(df), 0), dtype=np.float32)

    # One-hot limitata + hashing per alta cardinalità
    one_hot_frames = []
    for col in categorical_cols:
        vc = df[col].astype(str).value_counts()
        if len(vc) <= max_categories:
            oh = pd.get_dummies(df[col].fillna('NA').astype(str), prefix=f"{col}")
            one_hot_frames.append(oh)
        else:
            buckets = max_categories
            hashed = _hash_to_bucket(df[col], buckets)
            oh = pd.get_dummies(pd.Series(hashed, name=col), prefix=f"{col}_h", dtype=int)
            # garantisci numero fisso di bucket
            for i in range(buckets):
                name = f"{col}_h_{i}"
                if name not in oh.columns:
                    oh[name] = 0
            one_hot_frames.append(oh.sort_index(axis=1))

    if one_hot_frames:
        X_cat_df = pd.concat(one_hot_frames, axis=1)
        X_cat = X_cat_df.values.astype(np.float32)
    else:
        X_cat = np.zeros((len(df), 0), dtype=np.float32)

    # Compose features
    X = np.hstack([X_num, X_cat]).astype(np.float32)
    feat_cols = [f"feat_{i}" for i in range(X.shape[1])]
    out_df = pd.DataFrame(X, columns=feat_cols)
    out_df.insert(0, 'timestamp', df[timestamp_col].values)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return output_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--label-col', default=None)
    ap.add_argument('--max-categories', type=int, default=20)
    args = ap.parse_args()

    path = convert_csv_to_snn_dataset(
        args.input,
        args.output,
        label_col=args.label_col,
        max_categories=args.max_categories,
    )
    print(f"SNN dataset salvato in: {path}")


if __name__ == '__main__':
    main()


