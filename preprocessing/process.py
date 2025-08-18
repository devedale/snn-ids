# -*- coding: utf-8 -*-

"""
Modulo per il Preprocessing dei Dati di Cybersecurity.

Contiene funzioni per caricare, trasformare e preparare i dati di rete
per il training, seguendo le specifiche in `config.py`.
Include l'anonimizzazione di IP, one-hot encoding e la gestione del target.
"""

import pandas as pd
import json
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Importa le configurazioni
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_CONFIG, PREDICTION_CONFIG

def save_json_map(data, path):
    """Salva un dizionario in un file JSON, creando la directory se non esiste."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Mappa salvata in: {path}")

def preprocess_data():
    """
    Esegue il preprocessing completo dei dati di cybersecurity.

    Returns:
        tuple: (X, y, ip_map, target_map)
    """
    print("Inizio del preprocessing dei dati di cybersecurity...")

    # 1. Caricamento del dataset
    try:
        df = pd.read_csv(DATA_CONFIG["dataset_path"])
        print(f"Dataset caricato da: {DATA_CONFIG['dataset_path']}")
    except FileNotFoundError:
        print(f"Errore: Il file {DATA_CONFIG['dataset_path']} non è stato trovato.")
        return None, None, None, None

    # 2. Anonimizzazione degli indirizzi IP
    ip_map = {}
    ip_data = pd.DataFrame()
    if DATA_CONFIG.get("ip_columns_to_anonymize"):
        all_ips = pd.concat([df[col] for col in DATA_CONFIG["ip_columns_to_anonymize"]]).unique()
        ip_encoder = LabelEncoder().fit(all_ips)

        for col in DATA_CONFIG["ip_columns_to_anonymize"]:
            ip_data[f"{col}_id"] = ip_encoder.transform(df[col])

        ip_map = {
            "map": {ip: int(code) for ip, code in zip(ip_encoder.classes_, ip_encoder.transform(ip_encoder.classes_))},
            "inverse_map": {int(code): ip for ip, code in zip(ip_encoder.classes_, ip_encoder.transform(ip_encoder.classes_))}
        }
        save_json_map(ip_map, PREDICTION_CONFIG["ip_anonymization_map_path"])

    # 3. One-hot encoding
    categorical_data = pd.get_dummies(df[DATA_CONFIG["one_hot_encode_columns"]], drop_first=True)
    print(f"Eseguito one-hot encoding per: {DATA_CONFIG['one_hot_encode_columns']}")

    # 4. Selezione delle feature numeriche
    numeric_data = df[DATA_CONFIG["numeric_feature_columns"]]

    # 5. Combinazione di tutte le feature processate in X
    X = pd.concat([numeric_data, ip_data, categorical_data], axis=1)

    # Converte l'intero DataFrame in un tipo numerico per la compatibilità con TensorFlow
    X = X.astype(np.float32)

    print("Feature finali per il training:", list(X.columns))

    # 6. Gestione della colonna Target
    y = df[DATA_CONFIG["target_column"]]
    target_map = None
    if DATA_CONFIG["anonymize_target"]:
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)

        target_map = {
            "map": {label: int(code) for label, code in zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))},
            "inverse_map": {int(code): label for label, code in zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))}
        }
        y = pd.Series(y_encoded, name=DATA_CONFIG["target_column"])
        save_json_map(target_map, PREDICTION_CONFIG["target_anonymization_map_path"])

    print("Preprocessing completato.")
    return X, y, ip_map, target_map

if __name__ == '__main__':
    # Esempio di utilizzo del modulo
    X_processed, y_processed, ip_map, target_map = preprocess_data()

    if X_processed is not None:
        print("\n--- Anteprima dei dati processati ---")
        print("Prime 5 righe delle feature (X):")
        print(X_processed.head())
        print("\nPrime 5 righe del target (y):")
        print(y_processed.head())

        if ip_map:
            print("\n--- Mappa di Anonimizzazione IP (primi 5) ---")
            print({k: v for i, (k, v) in enumerate(ip_map['map'].items()) if i < 5})

        if target_map:
            print("\n--- Mappa di Anonimizzazione Target ---")
            print(json.dumps(target_map, indent=2))
