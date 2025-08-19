# -*- coding: utf-8 -*-

"""
Modulo per il Preprocessing Avanzato dei Dati.

Include logiche per:
- Anonimizzazione e One-Hot Encoding.
- Creazione di finestre temporali (time windows) per modelli sequenziali.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Importa le configurazioni
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_CONFIG, PREPROCESSING_CONFIG, PREDICTION_CONFIG

def save_json_map(data, path):
    """Salva un dizionario in un file JSON, creando la directory se non esiste."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Mappa salvata in: {path}")

def preprocess_data():
    """
    Esegue il preprocessing completo, inclusa la creazione di finestre temporali.
    """
    print("Inizio del preprocessing avanzato dei dati...")

    # 1. Caricamento e ordinamento dei dati
    df = pd.read_csv(DATA_CONFIG["dataset_path"])
    df[DATA_CONFIG["timestamp_column"]] = pd.to_datetime(df[DATA_CONFIG["timestamp_column"]])
    df = df.sort_values(by=DATA_CONFIG["timestamp_column"]).reset_index(drop=True)

    # 2. Label Encoding del target
    target_encoder = LabelEncoder()
    df[DATA_CONFIG["target_column"]] = target_encoder.fit_transform(df[DATA_CONFIG["target_column"]])
    target_map = {
        "map": {label: int(code) for label, code in zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))},
        "inverse_map": {str(code): label for code, label in enumerate(target_encoder.classes_)}
    }
    save_json_map(target_map, PREDICTION_CONFIG["target_anonymization_map_path"])

    # 3. Anonimizzazione IP
    all_ips = pd.concat([df[col] for col in DATA_CONFIG["ip_columns_to_anonymize"]]).unique()
    ip_encoder = LabelEncoder().fit(all_ips)
    for col in DATA_CONFIG["ip_columns_to_anonymize"]:
        df[col] = ip_encoder.transform(df[col])
    ip_map = {
        "map": {ip: int(code) for ip, code in zip(ip_encoder.classes_, ip_encoder.transform(ip_encoder.classes_))},
        "inverse_map": {str(code): ip for code, ip in enumerate(ip_encoder.classes_)}
    }
    save_json_map(ip_map, PREDICTION_CONFIG["ip_anonymization_map_path"])

    # 4. One-Hot Encoding per le feature categoriche
    df = pd.get_dummies(df, columns=[col for col in DATA_CONFIG["feature_columns"] if df[col].dtype == 'object'])

    # 5. Normalizzazione delle feature numeriche
    numeric_cols = df.select_dtypes(include=np.number).columns.drop(DATA_CONFIG["target_column"])
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 6. Salva l'ordine finale delle colonne
    final_feature_columns = list(df.columns.drop([DATA_CONFIG["target_column"], DATA_CONFIG["timestamp_column"]]))
    save_json_map(final_feature_columns, PREDICTION_CONFIG["column_order_path"])

    features_df = df[final_feature_columns].astype(np.float32)
    target_series = df[DATA_CONFIG["target_column"]]

    # 7. Creazione delle finestre temporali
    if not PREPROCESSING_CONFIG["use_time_windows"]:
        print("Creazione finestre temporali disabilitata. Ritorno i dati come sequenza piatta.")
        return features_df.values, target_series.values

    print(f"Creazione finestre temporali (size={PREPROCESSING_CONFIG['window_size']}, step={PREPROCESSING_CONFIG['step']})...")

    X, y = [], []
    window_size = PREPROCESSING_CONFIG['window_size']
    step = PREPROCESSING_CONFIG['step']

    for i in range(0, len(features_df) - window_size + 1, step):
        window = features_df.iloc[i : i + window_size].values
        # L'etichetta della finestra è quella dell'ultimo elemento
        label = target_series.iloc[i + window_size - 1]
        X.append(window)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"Preprocessing completato. Shape di X: {X.shape}, Shape di y: {y.shape}")
    return X, y

if __name__ == '__main__':
    # Esempio di utilizzo del modulo
    X_processed, y_processed = preprocess_data()
    if X_processed is not None:
        print("\n--- Esempio di dati processati ---")
        print("Forma di X:", X_processed.shape)
        print("Forma di y:", y_processed.shape)
        print("Primo campione di X:\n", X_processed[0])
        print("Prima etichetta di y:", y_processed[0])
