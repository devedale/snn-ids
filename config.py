import os
# -*- coding: utf-8 -*-

"""
File di Configurazione Globale per la Pipeline di Cybersecurity.

Questo file agisce come un pannello di controllo per l'intera pipeline,
permettendo di modificare parametri, percorsi e strategie senza cambiare il codice.
"""

# ==============================================================================
# CONFIGURAZIONE GENERALE E DEI DATI
# ==============================================================================
DATA_CONFIG = {
    # Percorso del file CSV del dataset (verrà generato sinteticamente).
    "dataset_path": "data/cybersecurity_data.csv",

    # Nome della colonna usata come timestamp. Configurabile dall'utente.
    "timestamp_column": "timestamp",

    # Colonne da usare come feature. Le colonne IP vengono gestite a parte.
    "feature_columns": [
        "porta_sorgente", "porta_destinazione", "protocollo", "byte_inviati", "byte_ricevuti"
    ],

    # Colonne contenenti indirizzi IP da anonimizzare.
    "ip_columns_to_anonymize": ["ip_sorgente", "ip_destinazione"],

    # Nome della colonna target (l'etichetta da predire).
    "target_column": "tipo_attacco",
}


# ==============================================================================
# CONFIGURAZIONE DEL PREPROCESSING
# ==============================================================================
PREPROCESSING_CONFIG = {
    # Abilita la trasformazione dei dati in finestre temporali.
    # Se False, i dati vengono trattati come campioni indipendenti.
    "use_time_windows": True,

    # Dimensione della finestra temporale (numero di eventi/righe in una sequenza).
    "window_size": 10,

    # Passo (step) con cui la finestra si sposta sui dati.
    # Se step < window_size, le finestre saranno sovrapposte (overlapping).
    # Esempio: window_size=10, step=5 -> le finestre si sovrappongono di 5 elementi.
    "step": 5,
}


# ==============================================================================
# CONFIGURAZIONE DEL TRAINING
# ==============================================================================
TRAINING_CONFIG = {
    # Percorso dove salvare modelli, log e mappe di anonimizzazione.
    "output_path": "models/",

    # Strategia di validazione da utilizzare.
    # Opzioni: 'train_test_split', 'k_fold'
    "validation_strategy": "k_fold",

    # Numero di "fold" (divisioni) da usare se la strategia è 'k_fold'.
    "k_fold_splits": 5,

    # Tipo di architettura del modello da addestrare.
    # Opzioni: 'dense' (per dati non sequenziali), 'lstm' (per finestre temporali)
    "model_type": "lstm",

    # Dizionario di iperparametri per la Grid Search.
    # Per un test veloce, si può lasciare un solo valore per parametro.
    "hyperparameters": {
        "activation": ["relu"],
        "batch_size": [32],
        "epochs": [10], # Ridotto per test veloci
        "learning_rate": [0.001],
        "lstm_units": [50] # Parametro specifico per il modello LSTM
    }
}


# ==============================================================================
# CONFIGURAZIONE DELLA PREDIZIONE
# ==============================================================================
PREDICTION_CONFIG = {
    # Percorso del modello da caricare. Se None, usa il migliore del training.
    "model_path": None,

    # Percorsi per le mappe di anonimizzazione.
    "target_anonymization_map_path": os.path.join(TRAINING_CONFIG["output_path"], "target_anonymization_map.json"),
    "ip_anonymization_map_path": os.path.join(TRAINING_CONFIG["output_path"], "ip_anonymization_map.json"),
    "column_order_path": os.path.join(TRAINING_CONFIG["output_path"], "column_order.json")
}
