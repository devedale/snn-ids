# -*- coding: utf-8 -*-
"""
Configurazione Centrale SNN-IDS
Sistema unificato per tutte le configurazioni del progetto.
"""

import os

# ==============================================================================
# CONFIGURAZIONE DATI
# ==============================================================================

DATA_CONFIG = {
    # Dataset
    "dataset_path": "data",
    "timestamp_column": "Timestamp",
    "target_column": "Label",
    # Identificatore del flusso (necessario per reassembly)
    "flow_id_column": "Flow_ID",
    # Valore che identifica il traffico benigno
    "benign_label": "BENIGN",
    # Formato timestamp opzionale (se non parsabile automaticamente)
    "timestamp_format": None,
    
    # Colonne IP da anonimizzare e trasformare in ottetti
    "ip_columns": ["Src IP", "Dst IP"],
    
    # Features da utilizzare per il training
    "feature_columns": [
        # IP (verranno automaticamente convertite in ottetti)
        "Src IP", "Dst IP",
        
        # Features di rete
        "Src Port", "Dst Port", "Protocol", "Flow Duration",
        "Total Fwd Packet", "Total Bwd packets",
        "Total Length of Fwd Packet", "Total Length of Bwd Packet",
        "Flow Bytes/s", "Flow Packets/s",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Mean", "Bwd IAT Mean",
        "Fwd Header Length", "Bwd Header Length",
        "Average Packet Size", "Fwd Segment Size Avg", "Bwd Segment Size Avg"
    ]
}

# ==============================================================================
# CONFIGURAZIONE PREPROCESSING
# ==============================================================================

PREPROCESSING_CONFIG = {
    # Campionamento e bilanciamento
    "sample_size": 100000,
    "balance_strategy": "security",  # legacy bilanciamento record-level
    "benign_ratio": 0.5,
    "max_samples_per_class": 100000,
    "min_samples_per_class": 1000,
    # Bilanciamento a livello di flusso/sessione
    "flow_balance": {
        "enabled": True,
        "method": "undersample",  # "undersample" | "smote" | "none"
        "ratio": 1.0  # BENIGN:MALEVOLI = 1.0 -> parità
    },
    
    # Finestre temporali per modelli sequenziali
    "use_time_windows": True,
    "window_size": 10,
    "step": 5,

    # Strategia finestre per contesto di attacco (N prima, T dopo)
    "flow_window_strategy": "first_malicious_context",  # "first_malicious_context" | "fixed"
    "window_before_first_malicious_s": 180,
    "window_after_first_malicious_s": 60,
    "time_bin_seconds": 5,  # granularità per sequenze dentro il contesto N/T
    "session_timeout_seconds": 60,
    # Solo intervalli malevoli nelle finestre e filtro sequenze troppo corte
    "malicious_only_windows": True,
    "min_sequence_bins": 2,

    # Propagazione/gestione etichette per finestra
    "label_propagation": {
        "mode": "majority",  # "any" | "majority" | "probabilistic" | "smoothing"
        "prob_threshold": 0.5,
        "smoothing_alpha": 0.6,
        "noise_filter": {
            "enabled": True,
            "method": "temporal_smoothing",  # "majority" | "temporal_smoothing"
            "window": 3,
            "threshold": 0.5
        }
    },

    # Output per modello
    "output_mode": "sequence",  # "sequence" | "mlp_aggregated"
    "aggregation_stats": ["sum", "mean", "std", "min", "max"],
    
    # Trasformazione IP in ottetti
    "convert_ip_to_octets": True,
    
    # Caching preprocessing
    "cache_enabled": True,
    "cache_dir": "preprocessed_cache"
}

# ==============================================================================
# CONFIGURAZIONE TRAINING
# ==============================================================================

TRAINING_CONFIG = {
    # Output
    "output_path": "models/",
    
    # Strategia di validazione
    "validation_strategy": "k_fold",  # o "train_test_split"
    "k_fold_splits": 5,
    "test_size": 0.2,
    
    # Modello (focus GRU per cybersecurity)
    "model_type": "gru",  # "gru", "lstm", "dense"
    
    # Hyperparameters
    "hyperparameters": {
        "epochs": [10, 20, 30],
        "batch_size": [64],
        "learning_rate": [0.001],
        "activation": ["relu"],
        "lstm_units": [64],
        "gru_units": [64]
    },
    # Config MLP 4 hidden layers
    "mlp_hidden_layers": [256, 128, 64, 32],
    "dropout_rate": 0.2,
    # Logging
    "log_training_history": True,
    "max_epochs": 30,
    # Esecuzione per-epoca (preprocessing a flussi per epoca)
    "preprocess_per_epoch": False,
    "flows_per_epoch": 5000,
    "epoch_selection_mode": "sequential",  # "sequential" | "parallel"
    # Target type preferito (non vincolante; i modelli scelgono l'head in base a y)
    "target_type": "multiclass"  # "binary" | "multiclass"
}

# ==============================================================================
# CONFIGURAZIONE EVALUATION
# ==============================================================================

EVALUATION_CONFIG = {
    # Metriche cybersecurity
    "cybersecurity_focus": True,
    "generate_confusion_matrix": True,
    "generate_roc_curves": True,
    "detailed_attack_analysis": True,
    
    # Output
    "save_plots": True,
    "save_detailed_reports": True
}

# ==============================================================================
# CONFIGURAZIONE BENCHMARK
# ==============================================================================

BENCHMARK_CONFIG = {
    # Test configurations
    "test_configs": [
        {
            "name": "baseline",
            "description": "Baseline senza anonimizzazione",
            "use_cryptopan": False
        },
        {
            "name": "cryptopan", 
            "description": "Con anonimizzazione Crypto-PAn",
            "use_cryptopan": True
        }
    ],
    
    # Risoluzioni temporali da testare (focus cybersecurity)
    "time_resolutions": ["5s", "1m", "5m"],
    
    # Output
    "output_dir": "benchmark_results",
    "generate_visualizations": True,
    "save_intermediate_results": True,
    
    # Configurazioni per smoke test
    "smoke_test": {
        "sample_size": 5000,
        "time_resolutions": ["5s"],
        "hyperparameters": {
            "epochs": [2],
            "batch_size": [32]
        }
    }
}

# ==============================================================================
# REGISTRI / PROFILI (per configurazioni modulari e generalizzate)
# ==============================================================================

# Registri semplici (nomi simbolici -> identificatori logici)
# NOTA: i nomi qui definiti sono usati come chiavi nelle config/preset;
# l'effettiva implementazione è demandata ai moduli (preprocessing/training).

MODEL_REGISTRY = {
    "dense": "dense",
    "gru": "gru",
    "lstm": "lstm"
}

VALIDATION_REGISTRY = {
    "k_fold": "k_fold",
    "train_test_split": "train_test_split"
}

WINDOWING_STRATEGIES = {
    "FirstMaliciousContext": {
        "description": "Finestra dal primo all'ultimo evento malevolo con margini",
        "params": ["before_s", "after_s", "malicious_only"]
    },
    "FixedWindow": {
        "description": "Finestra fissa scorrevole",
        "params": ["window_size", "step"]
    },
    "FullFlow": {
        "description": "Intero flusso",
        "params": []
    }
}

LABELING_STRATEGIES = {
    "BinaryAny": "Binario: qualsiasi evento malevolo -> 1",
    "BinaryDominant": "Binario: maggioranza su finestra",
    "MultiClassDominant": "Multiclasse: classe non-BENIGN dominante"
}

FEATURE_STRATEGIES = {
    "SequenceBinner": {
        "description": "Binning temporale e aggregazioni per sequenze",
        "params": ["bin_seconds", "stats"]
    },
    "Aggregator": {
        "description": "Aggregazione finestra per MLP",
        "params": ["stats"]
    }
}

BALANCING_STRATEGIES = {
    "FlowLevel": {
        "description": "Bilanciamento a livello di flusso",
        "params": ["method", "ratio"]
    },
    "None": {
        "description": "Nessun bilanciamento",
        "params": []
    }
}

# Profili d'esempio (preset) per esecuzioni rapide e ripetibili
PROFILES = {
    "gru_sequence": {
        "description": "GRU su sequenze con finestra malevola (before/after)",
        "preprocessing": {
            "use_time_windows": True,
            "flow_window_strategy": "FirstMaliciousContext",
            "window_before_first_malicious_s": 180,
            "window_after_first_malicious_s": 60,
            "malicious_only_windows": True,
            "time_bin_seconds": 5,
            "min_sequence_bins": 3,
            "output_mode": "sequence",
            "aggregation_stats": ["sum", "mean", "std", "max"],
            "cache_enabled": True
        },
        "training": {
            "model_type": "gru",
            "validation_strategy": "k_fold",
            "hyperparameters": {
                "epochs": [3, 6],
                "batch_size": [32],
                "learning_rate": [0.001, 0.0005],
                "activation": ["relu"],
                "gru_units": [64]
            },
            "target_type": "multiclass"
        }
    },
    "dense_aggregate": {
        "description": "DENSE aggregato finestra (baseline veloce)",
        "preprocessing": {
            "use_time_windows": True,
            "flow_window_strategy": "FirstMaliciousContext",
            "window_before_first_malicious_s": 60,
            "window_after_first_malicious_s": 30,
            "malicious_only_windows": True,
            "output_mode": "mlp_aggregated",
            "aggregation_stats": ["sum", "mean", "max"],
            "cache_enabled": True
        },
        "training": {
            "model_type": "dense",
            "validation_strategy": "train_test_split",
            "hyperparameters": {
                "epochs": [3],
                "batch_size": [64],
                "learning_rate": [0.001],
                "activation": ["relu"]
            },
            "target_type": "binary"
        }
    },
    "ablation_non_mal_only": {
        "description": "Sequenze senza filtro 'malicious_only' per confronto",
        "preprocessing": {
            "use_time_windows": True,
            "flow_window_strategy": "FirstMaliciousContext",
            "window_before_first_malicious_s": 180,
            "window_after_first_malicious_s": 60,
            "malicious_only_windows": False,
            "time_bin_seconds": 10,
            "min_sequence_bins": 2,
            "output_mode": "sequence",
            "aggregation_stats": ["sum", "mean"],
            "cache_enabled": True
        },
        "training": {
            "model_type": "gru",
            "validation_strategy": "k_fold",
            "hyperparameters": {
                "epochs": [3],
                "batch_size": [32],
                "learning_rate": [0.001],
                "activation": ["relu"],
                "gru_units": [64]
            },
            "target_type": "multiclass"
        }
    }
}

# ==============================================================================
# CONFIGURAZIONE PREDIZIONE
# ==============================================================================

PREDICTION_CONFIG = {
    # Mappe di anonimizzazione
    "target_map_path": os.path.join(TRAINING_CONFIG["output_path"], "target_map.json"),
    "ip_map_path": os.path.join(TRAINING_CONFIG["output_path"], "ip_map.json"),
    "column_order_path": os.path.join(TRAINING_CONFIG["output_path"], "column_order.json")
}
