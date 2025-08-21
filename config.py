# -*- coding: utf-8 -*-

"""
File di Configurazione Globale per la Pipeline di Cybersecurity.

Questo file agisce come un pannello di controllo per l'intera pipeline,
permettendo di modificare parametri, percorsi e strategie senza cambiare il codice.
Questo file è stato adattato per il dataset CIC-IDS-2017/2018.
"""

import os

# ==============================================================================
# CONFIGURAZIONE GENERALE E DEI DATI
# ==============================================================================
DATA_CONFIG = {
    # Percorso del file CSV del dataset.
    # NOTA: Il dataset è composto da più file. La logica di preprocessing
    # dovrà essere adattata per caricarli e concatenarli.
    # Qui indichiamo la directory che li contiene.
    "dataset_path": "data/CSECICIDS2018_improved/", # Esempio, punta a una delle cartelle

    # Nome della colonna usata come timestamp.
    "timestamp_column": "Timestamp",

    # Colonne da usare come feature.
    # Abbiamo selezionato un set di feature numeriche rilevanti dal dataset originale.
    "feature_columns": [
        "Src Port", "Dst Port", "Protocol", "Flow Duration", "Total Fwd Packet",
        "Total Bwd packets", "Total Length of Fwd Packet", "Total Length of Bwd Packet",
        "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
        "Flow IAT Max", "Flow IAT Min", "Fwd IAT Mean", "Bwd IAT Mean",
        "Fwd Header Length", "Bwd Header Length", "Average Packet Size",
        "Fwd Segment Size Avg", "Bwd Segment Size Avg"
    ],

    # Colonne contenenti indirizzi IP da anonimizzare.
    "ip_columns_to_anonymize": ["Src IP", "Dst IP"],

    # Nome della colonna target (l'etichetta da predire).
    "target_column": "Label",
}


# ==============================================================================
# CONFIGURAZIONE DEL PREPROCESSING
# ==============================================================================
PREPROCESSING_CONFIG = {
    # Abilita la trasformazione dei dati in finestre temporali.
    "use_time_windows": True,

    # Dimensione della finestra temporale (numero di eventi/righe in una sequenza).
    "window_size": 10,

    # Passo (step) con cui la finestra si sposta sui dati.
    "step": 5,
}


# ==============================================================================
# CONFIGURAZIONE DEL TRAINING
# ==============================================================================
TRAINING_CONFIG = {
    # Percorso dove salvare modelli, log e mappe di anonimizzazione.
    "output_path": "models/",

    # Strategia di validazione da utilizzare.
    "validation_strategy": "train_test_split", # Cambiato per un training più veloce su un dataset grande

    # Rapporto di divisione per 'train_test_split'.
    "test_size": 0.2,

    # Numero di "fold" (divisioni) da usare se la strategia è 'k_fold'.
    "k_fold_splits": 5,

    # Tipo di architettura del modello da addestrare.
    "model_type": "lstm",

    # Dizionario di iperparametri per la Grid Search.
    "hyperparameters": {
        "activation": ["relu"],
        "batch_size": [64], # Aumentato per dataset più grandi
        "epochs": [20],
        "learning_rate": [0.001],
        "lstm_units": [64]
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
