# -*- coding: utf-8 -*-

"""
Modulo per la Predizione su Dati Sequenziali.
Refattorizzato per accettare un percorso del modello e usare le nuove feature.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_CONFIG, TRAINING_CONFIG, PREDICTION_CONFIG, PREPROCESSING_CONFIG

def load_json_map(path):
    """Carica una mappa da un file JSON."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Errore: Il file di mappa {path} non è stato trovato.")
        return None

def predict_on_window(data_window, model_path=None):
    """
    Esegue una predizione su una singola finestra di dati.

    Args:
        data_window (list of dict): Una lista di N dizionari, dove N è la window_size.
        model_path (str, optional): Percorso al modello da usare. Se None, usa
                                    il percorso di default dalla configurazione.

    Returns:
        str: L'etichetta di predizione.
    """
    print("Avvio del processo di predizione su una finestra di dati...")

    # 1. Validazione dell'input
    if len(data_window) != PREPROCESSING_CONFIG['window_size']:
        print(f"Errore: La finestra di input deve avere una dimensione di {PREPROCESSING_CONFIG['window_size']}.")
        return None

    # 2. Caricamento del modello e delle mappe
    if model_path is None:
        model_path = PREDICTION_CONFIG['model_path'] or os.path.join(TRAINING_CONFIG['output_path'], 'best_model.keras')

    if not os.path.exists(model_path):
        print(f"Errore: Il file del modello {model_path} non è stato trovato.")
        return None
    model = tf.keras.models.load_model(model_path)

    ip_map = load_json_map(PREDICTION_CONFIG["ip_anonymization_map_path"])
    target_map = load_json_map(PREDICTION_CONFIG["target_anonymization_map_path"])
    column_order = load_json_map(PREDICTION_CONFIG["column_order_path"])

    if not all([ip_map, target_map, column_order]):
        print("Predizione interrotta: file di configurazione o mappe mancanti.")
        return None

    # 3. Preprocessing della finestra di input
    df = pd.DataFrame(data_window)

    # a. Assicura che tutte le feature necessarie siano presenti
    for col in DATA_CONFIG['feature_columns']:
        if col not in df.columns:
            df[col] = 0 # Aggiungi colonna mancante con valore di default

    # b. Anonimizzazione IP
    for col in DATA_CONFIG["ip_columns_to_anonymize"]:
        df[col] = df[col].map(ip_map['map']).fillna(-1).astype(int)

    # c. One-Hot Encoding
    categorical_features = [col for col in DATA_CONFIG["feature_columns"] if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_features)

    # d. Allinea le colonne con quelle del training
    for col in column_order:
        if col not in df.columns:
            df[col] = 0
    df = df[column_order]

    # e. Normalizzazione
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)

    # 4. Predizione
    X_sample = df.astype(np.float32).values
    X_sample = np.expand_dims(X_sample, axis=0)

    prediction_probs = model.predict(X_sample)
    predicted_class_index = np.argmax(prediction_probs, axis=1)[0]

    # 5. De-anonimizzazione del risultato
    predicted_label = target_map['inverse_map'][str(predicted_class_index)]

    print(f"Predizione completata. Risultato: {predicted_label}")
    return predicted_label

if __name__ == '__main__':
    # Creiamo una finestra di dati fittizia per il test con le nuove feature
    window_size = PREPROCESSING_CONFIG.get('window_size', 10)
    #sample_window = [
    #    {
    #        'Src IP': '10.42.0.1', 'Dst IP': '10.42.0.2', 'Src Port': 12345, 'Dst Port': 80, 'Protocol': 6,
    #        'Flow Duration': 100, 'Total Fwd Packet': 2, 'Total Bwd packets': 2, 'Total Length of Fwd Packet': 100,
    #        'Total Length of Bwd Packet': 100, 'Flow Bytes/s': 2000, 'Flow Packets/s': 40, 'Flow IAT Mean': 25.0,
    #        'Flow IAT Std': 10.0, 'Flow IAT Max': 50, 'Flow IAT Min': 10, 'Fwd IAT Mean': 50.0, 'Bwd IAT Mean': 50.0,
    #        'Fwd Header Length': 40, 'Bwd Header Length': 40, 'Average Packet Size': 50.0, 'Fwd Segment Size Avg': 50.0,
    #        'Bwd Segment Size Avg': 50.0
    #    } for _ in range(window_size)
    #]

    sample_window = [{'Src IP': '10.42.0.1', 'Dst IP': '10.42.0.2', 'Src Port': 12345, 'Dst Port': 80, 'Protocol': 6, 'Total Fwd Packet': 2, 'Total Bwd packets': 2}for _ in range(window_size)]

    prediction = predict_on_window(sample_window)
    if prediction:
        print(f"\nLa finestra di dati è stata classificata come: '{prediction}'")
