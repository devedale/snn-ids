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
    
    # Colonne IP da anonimizzare e trasformare in ottetti
    "ip_columns": ["Src IP", "Dst IP"],
    
    # Features ottimizzate basate su feature_analysis/feature_selector_analyzer.py
    "feature_columns": [
        # === PRIVACY SENSITIVE (da cifrare con HE) ===
        "Src IP", "Dst IP",        # Convertiti in 8 ottetti (Src_Octet_1-4, Dst_Octet_1-4)
        "Src Port", "Dst Port",    # Dst Port molto discriminante (score: 0.945, ratio attack/benign: 9.96x)
        
        # === TOP DISCRIMINATIVE FEATURES (Score > 1.0) ===
        "FIN Flag Count",           # Score: 1.281 | attack_mean: 1.53 vs benign_mean: 0.15 (ratio: 10.2x)
                                    # TCP connection termination - attacchi spesso terminano connessioni bruscamente
        
        "Flow IAT Max",             # Score: 1.069 | attack_mean: 189k vs benign_mean: 15.7M (ratio: 0.012x)  
                                    # Max tempo tra pacchetti - attacchi più rapidi, meno pause
        
        "Flow IAT Std",             # Score: 1.065 | attack_mean: 9k vs benign_mean: 4.8M (ratio: 0.002x)
                                    # Variabilità timing - attacchi più regolari/automatizzati
        
        "Fwd IAT Std",              # Score: 1.053 | attack_mean: 10k vs benign_mean: 6.3M (ratio: 0.002x)
                                    # Variabilità forward timing - bot/script vs comportamento umano
        
        "Fwd IAT Max",              # Score: 1.032 | attack_mean: 190k vs benign_mean: 15.6M (ratio: 0.012x)
                                    # Max tempo forward - attacchi più continui
        
        "Flow IAT Mean",            # Score: 1.029 | attack_mean: 23k vs benign_mean: 1.8M (ratio: 0.013x)
                                    # Tempo medio tra pacchetti - signature di automazione
        
        "Bwd Packet Length Min",    # Score: 1.029 | attack_mean: 0.0 vs benign_mean: 58.3 (ratio: 0.0x)
                                    # Dimensione minima backward - molti attacchi unidirezionali
        
        # === HIGHLY DISCRIMINATIVE FEATURES (Score 0.8-1.0) ===
        "Bwd IAT Total", "Fwd IAT Mean", "Bwd IAT Max", "Bwd IAT Std",
        "Total Length of Fwd Packet",  # Score: 0.882 | ratio: 374x - attacchi payload grandi
        "Total Fwd Packet",            # Score: 0.865 | ratio: 3590x - flood/DDoS patterns  
        "RST Flag Count",              # Score: 0.764 | ratio: 0.002x - attacchi evitano RST
        
        # === ADDITIONAL USEFUL FEATURES (non privacy-sensitive) ===
        "Protocol", "Flow Duration", "Total Bwd packets",
        "Flow Bytes/s", "Flow Packets/s", "Flow IAT Min", 
        "Fwd Header Length", "Bwd Header Length",
        "Average Packet Size", "Fwd Segment Size Avg", "Bwd Segment Size Avg"
    ]
}

# ==============================================================================
# CONFIGURAZIONE PREPROCESSING
# ==============================================================================

PREPROCESSING_CONFIG = {
    # Campionamento e bilanciamento
    "sample_size": 10000,
    "balance_strategy": "security",  # 50% BENIGN, 50% ATTACCHI
    "benign_ratio": 0.5,
    "max_samples_per_class": 100000,
    "min_samples_per_class": 1000,

    # Caching per ottimizzazione
    "cache_enabled": True,
    "cache_dir": "preprocessed_cache",
    "delete_processed_files": False,  # Se True, elimina i file sorgente dopo la cache
    "parallel_processes": 2,  # Numero di processi per parallelizzare il caching (ridotto per evitare OOM)
    
    # Finestre temporali per modelli sequenziali
    "use_time_windows": True,
    "window_size": 10,
    "step": 5,
    
    # Trasformazione IP in ottetti
    "convert_ip_to_octets": True,

    # Cache model-ready (X,y,label) per training rapido
    "model_cache_enabled": True,
    "model_cache_dir": "model_cache"
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
        "epochs": [5],
        "batch_size": [64],
        "learning_rate": [0.001],
        "activation": ["relu"],
        "lstm_units": [64],
        "gru_units": [64]
    }
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
        "sample_size": 10000,
        "time_resolutions": ["5s"],
        "hyperparameters": {
            "epochs": [2],
            "batch_size": [32]
        }
    }
}

# ==============================================================================
# CONFIGURAZIONE HOMOMORPHIC ENCRYPTION
# ==============================================================================

HOMOMORPHIC_CONFIG = {
    # Abilitazione HE
    "enabled": False,  # Set True per abilitare encryption
    
    # Schema di encryption
    "scheme": "CKKS",  # CKKS per dati real-valued
    "poly_modulus_degree": 8192,  # Grado polinomio (sicurezza vs performance)
    "coeff_mod_bit_sizes": [60, 40, 40, 60],  # Bit sizes moduli
    "scale": 2**40,  # Scala per encoding CKKS
    
    # Features da cifrare (SOLO quelle privacy-sensitive)
    "features_to_encrypt": [
        # IP ottetti (automaticamente generati da Src IP, Dst IP)
        "Src IP_Octet_1", "Src IP_Octet_2", "Src IP_Octet_3", "Src IP_Octet_4",
        "Dst IP_Octet_1", "Dst IP_Octet_2", "Dst IP_Octet_3", "Dst IP_Octet_4",
        # Porte
        "Src Port", "Dst Port"
    ],
    
    # Features NON cifrate (performance-critical ma non privacy-sensitive)
    "plaintext_features": [
        # Timing features (discriminanti ma non sensibili)
        "Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Mean", "Bwd IAT Mean",
        # Size/packet features
        "Total Fwd Packet", "Total Bwd packets", 
        "Total Length of Fwd Packet", "Total Length of Bwd Packet",
        "Flow Bytes/s", "Flow Packets/s",
        "Average Packet Size", "Fwd Segment Size Avg", "Bwd Segment Size Avg",
        # Header features
        "Fwd Header Length", "Bwd Header Length",
        # Protocol (non sensibile, solo TCP/UDP/ICMP)
        "Protocol"
    ],
    
    # Configurazioni performance
    "batch_size_encrypted": 32,  # Ridotto per overhead HE
    "batch_size_plaintext": 64,  # Normale per features non cifrate
    "enable_key_rotation": True,  # Rotazione chiavi per sicurezza
    "max_encrypted_depth": 5,    # Profondità massima operazioni cifrate
}

# ==============================================================================
# CONFIGURAZIONE FEDERATED LEARNING
# ==============================================================================

FEDERATED_CONFIG = {
    # Abilitazione FL
    "enabled": False,  # Set True per modalità federata
    
    # Configurazione client
    "num_clients": 5,  # Numero client federati
    "client_data_split": "iid",  # "iid" o "non_iid"
    "min_clients_per_round": 3,  # Minimo client per round
    
    # Configurazione rounds
    "num_rounds": 10,  # Numero round federati
    "local_epochs": 2,  # Epoche locali per client
    "client_lr": 0.001,  # Learning rate client
    
    # Aggregazione
    "aggregation_method": "fedavg",  # "fedavg", "fedprox", "scaffold"
    "differential_privacy": False,  # DP sui gradienti
    "dp_noise_scale": 0.1,  # Scala rumore DP
    
    # Homomorphic encryption in FL
    "use_homomorphic_aggregation": False,  # HE per aggregazione modelli
    "encrypt_gradients": True,  # Cifra solo gradienti, non tutto il modello
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
