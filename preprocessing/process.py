# -*- coding: utf-8 -*-

"""
Modulo per il Preprocessing Avanzato dei Dati.
Refattorizzato per accettare override della configurazione.
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler
from copy import deepcopy

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_CONFIG as DC, PREPROCESSING_CONFIG as PC, PREDICTION_CONFIG as PredC

def save_json_map(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Mappa salvata in: {path}")

def load_data_from_directory(path, sample_size=None):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    if not all_files:
        print(f"Attenzione: Nessun file CSV trovato in '{path}'.")
        return pd.DataFrame()

    df_list = [pd.read_csv(f, low_memory=False) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    if sample_size:
        df = df.head(sample_size)
    return df

def preprocess_data(config_override=None):
    """
    Esegue il preprocessing completo. Accetta un dizionario per sovrascrivere
    le configurazioni di default al volo.
    """
    # Unisci le configurazioni di default con l'override
    data_config = deepcopy(DC)
    proc_config = deepcopy(PC)
    pred_config = deepcopy(PredC)
    if config_override:
        data_config.update(config_override.get("DATA_CONFIG", {}))
        proc_config.update(config_override.get("PREPROCESSING_CONFIG", {}))

    sample_size = proc_config.get("sample_size")
    print(f"--- Inizio Preprocessing (Sample Size: {sample_size or 'Completo'}) ---")

    df = load_data_from_directory(data_config["dataset_path"], sample_size)
    if df.empty:
        return None, None

    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    df[data_config["timestamp_column"]] = pd.to_datetime(df[data_config["timestamp_column"]])
    df = df.sort_values(by=data_config["timestamp_column"]).reset_index(drop=True)

    target_encoder = LabelEncoder()
    df[data_config["target_column"]] = target_encoder.fit_transform(df[data_config["target_column"]])
    target_map = {
        "map": {label: int(code) for label, code in zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))},
        "inverse_map": {str(code): label for code, label in enumerate(target_encoder.classes_)}
    }
    save_json_map(target_map, pred_config["target_anonymization_map_path"])

    all_ips = pd.concat([df[col] for col in data_config["ip_columns_to_anonymize"]]).unique()
    ip_encoder = LabelEncoder().fit(all_ips)
    for col in data_config["ip_columns_to_anonymize"]:
        df[col] = ip_encoder.transform(df[col])
    ip_map = {
        "map": {ip: int(code) for ip, code in zip(ip_encoder.classes_, ip_encoder.transform(ip_encoder.classes_))},
        "inverse_map": {str(code): ip for code, ip in enumerate(ip_encoder.classes_)}
    }
    save_json_map(ip_map, pred_config["ip_anonymization_map_path"])

    categorical_features = [col for col in data_config["feature_columns"] if col in df.columns and df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_features)

    numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col != data_config["target_column"]]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    final_feature_columns = [col for col in df.columns if col not in [data_config["target_column"], data_config["timestamp_column"]]]
    save_json_map(final_feature_columns, pred_config["column_order_path"])

    features_df = df[final_feature_columns].astype(np.float32)
    target_series = df[data_config["target_column"]]

    if not proc_config["use_time_windows"]:
        return features_df.values, target_series.values

    X, y = [], []
    window_size = proc_config['window_size']
    step = proc_config['step']
    for i in range(0, len(features_df) - window_size + 1, step):
        window = features_df.iloc[i : i + window_size].values
        label = target_series.iloc[i + window_size - 1]
        X.append(window)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"Preprocessing completato. Shape di X: {X.shape}, Shape di y: {y.shape}")
    return X, y
