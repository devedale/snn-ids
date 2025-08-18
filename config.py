# -*- coding: utf-8 -*-

"""
File di Configurazione Globale per il Progetto di Cybersecurity.

Questo file contiene tutte le configurazioni per il progetto, adattate
per l'analisi di dati di rete simulati.
"""

# ==============================================================================
# CONFIGURAZIONE DEL DATASET E PREPROCESSING
# ==============================================================================
DATA_CONFIG = {
    # Percorso del file CSV del dataset sintetico.
    "dataset_path": "data/cybersecurity_data.csv",

    # Elenco delle colonne da usare come feature numeriche dirette.
    "numeric_feature_columns": [
        "porta_sorgente",
        "porta_destinazione",
        "byte_inviati",
        "byte_ricevuti",
        "pacchetti_inviati",
        "pacchetti_ricevuti"
    ],

    # Colonne categoriche da trasformare con one-hot encoding.
    "one_hot_encode_columns": ["protocollo"],

    # Colonne contenenti indirizzi IP da anonimizzare.
    # Questi verranno trasformati in feature numeriche (ID).
    "ip_columns_to_anonymize": ["ip_sorgente", "ip_destinazione"],

    # Nome della colonna target (l'etichetta da predire).
    "target_column": "tipo_attacco",

    # Abilita l'anonimizzazione della colonna target (es. 'normal' -> 0, 'dos' -> 1).
    "anonymize_target": True,
}


# ==============================================================================
# CONFIGURAZIONE DEL TRAINING
# ==============================================================================
TRAINING_CONFIG = {
    # Percorso dove salvare modelli e log.
    "output_path": "models/",

    # Iperparametri da testare con la grid search.
    "hyperparameters": {
        "activation": ["relu", "tanh"],
        "batch_size": [16, 32],
        "epochs": [50],
        "learning_rate": [0.001, 0.01],
        "test_size": [0.2],
        "hidden_layer_size": [32, 64]
    }
}


# ==============================================================================
# CONFIGURAZIONE DELLA PREDIZIONE E MAPPE
# ==============================================================================
PREDICTION_CONFIG = {
    # Percorso del modello da caricare per la predizione.
    # Se None, verrà usato il modello migliore trovato durante il training.
    "model_path": None,

    # Percorso della mappa di anonimizzazione per il target.
    "target_anonymization_map_path": "models/target_anonymization_map.json",

    # Percorso della mappa di anonimizzazione per gli IP.
    "ip_anonymization_map_path": "models/ip_anonymization_map.json"
}
