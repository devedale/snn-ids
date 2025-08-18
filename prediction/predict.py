# -*- coding: utf-8 -*-

"""
Modulo per la Predizione.

Questo modulo contiene le funzioni per caricare un modello addestrato
e utilizzarlo per fare predizioni su nuovi dati.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

# Assicura che i messaggi di log di TensorFlow siano meno verbosi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Importa le configurazioni
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_CONFIG, TRAINING_CONFIG, PREDICTION_CONFIG

def load_json_map(path):
    """Carica una mappa da un file JSON."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Errore: Il file di mappa {path} non è stato trovato.")
        return None

def predict(new_data_sample):
    """
    Esegue una predizione su un nuovo campione di dati.

    Args:
        new_data_sample (dict or pd.DataFrame):
            Un campione di dati con le stesse colonne del dataset originale.

    Returns:
        str: L'etichetta di predizione (es. "normal", "dos").
    """
    print("Avvio del processo di predizione...")

    # 1. Caricamento del modello e delle mappe
    model_path = PREDICTION_CONFIG['model_path'] or os.path.join(TRAINING_CONFIG['output_path'], 'best_model.keras')
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Modello caricato da: {model_path}")
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        return None

    ip_map = load_json_map(PREDICTION_CONFIG["ip_anonymization_map_path"])
    target_map = load_json_map(PREDICTION_CONFIG["target_anonymization_map_path"])

    if not ip_map or not target_map:
        print("Predizione interrotta: mappe di anonimizzazione non trovate.")
        return None

    # 2. Preprocessing del nuovo campione di dati
    if isinstance(new_data_sample, dict):
        df_sample = pd.DataFrame([new_data_sample])
    else:
        df_sample = new_data_sample

    # Applica lo stesso preprocessing del training
    # a. Anonimizza IP
    ip_data = pd.DataFrame()
    for col in DATA_CONFIG["ip_columns_to_anonymize"]:
        # Usa la mappa esistente. Se un IP è nuovo, assegna un ID speciale (es. -1)
        ip_data[f"{col}_id"] = [ip_map['map'].get(ip, -1) for ip in df_sample[col]]

    # b. One-hot encode
    categorical_data = pd.get_dummies(df_sample[DATA_CONFIG["one_hot_encode_columns"]], drop_first=True)

    # c. Dati numerici
    numeric_data = df_sample[DATA_CONFIG["numeric_feature_columns"]]

    # d. Combina e riordina le colonne per matchare l'input del modello
    X_sample = pd.concat([numeric_data, ip_data, categorical_data], axis=1)

    # Assicura che le colonne siano nello stesso ordine del training caricando l'ordine salvato
    column_order_path = os.path.join(TRAINING_CONFIG['output_path'], 'column_order.json')
    trained_model_features = load_json_map(column_order_path)

    if not trained_model_features:
        print("Predizione interrotta: file con ordine delle colonne non trovato.")
        return None

    # Aggiungi colonne mancanti nel campione (es. protocollo_UDP se il campione era solo TCP)
    for col in trained_model_features:
        if col not in X_sample.columns:
            X_sample[col] = 0

    # Riordina le colonne del campione per matchare esattamente l'ordine del training
    X_sample = X_sample[trained_model_features]

    # e. Conversione tipo
    X_sample = X_sample.astype(np.float32)

    # 3. Esecuzione della predizione
    prediction_probs = model.predict(X_sample)
    predicted_class_index = np.argmax(prediction_probs, axis=1)[0]

    # 4. De-anonimizzazione del risultato
    predicted_label = target_map['inverse_map'][str(predicted_class_index)]

    print(f"Predizione completata. Risultato: {predicted_label}")
    return predicted_label

if __name__ == '__main__':
    # Esempio di utilizzo: crea un campione di dati finto e fai una predizione
    sample = {
        "ip_sorgente": "1.119.61.88", # Un IP presente nel training set
        "ip_destinazione": "10.0.0.5", # Un IP probabilmente nuovo
        "porta_sorgente": 54321,
        "porta_destinazione": 80,
        "protocollo": "TCP",
        "byte_inviati": 100,
        "byte_ricevuti": 1200,
        "pacchetti_inviati": 5,
        "pacchetti_ricevuti": 8,
    }

    prediction = predict(sample)
    if prediction:
        print(f"\nIl campione di dati è stato classificato come: '{prediction}'")
