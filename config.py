# -*- coding: utf-8 -*-
"""
Central Configuration for SNN-IDS
A unified system for all project configurations.
"""

import os

# ==============================================================================
# RANDOM/SEED CONFIGURATION
# ==============================================================================
RANDOM_CONFIG = {
    # Seed globale per tutte le librerie
    "seed": 79
}

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================

DATA_CONFIG = {
    # Dataset settings
    "dataset_path": "data",
    "timestamp_column": "Timestamp",
    "target_column": "Label",
    
    # IP columns to be anonymized and converted to octets
    "ip_columns": ["Src IP", "Dst IP"],
    
    # Optimized features based on feature_analysis/feature_selector_analyzer.py.
    # The comments explain the rationale for selecting each feature.
    "feature_columns": [
        # === PRIVACY SENSITIVE (candidates for Homomorphic Encryption) ===
        "Src IP", "Dst IP",        # Converted to 8 octets (Src_Octet_1-4, Dst_Octet_1-4)
        "Src Port", "Dst Port",    # Dst Port is highly discriminative (score: 0.945, attack/benign ratio: 9.96x)
        
        # === TOP DISCRIMINATIVE FEATURES (Selection Score > 1.0) ===
        "FIN Flag Count",           # Score: 1.281 | attack_mean: 1.53 vs benign_mean: 0.15 (ratio: 10.2x)
                                    # Reason: TCP connection termination. Attacks often end connections abruptly.
        
        "Flow IAT Max",             # Score: 1.069 | attack_mean: 189k vs benign_mean: 15.7M (ratio: 0.012x)  
                                    # Reason: Max time between packets. Attacks are faster, with fewer pauses.
        
        "Flow IAT Std",             # Score: 1.065 | attack_mean: 9k vs benign_mean: 4.8M (ratio: 0.002x)
                                    # Reason: Timing variability. Attacks are more regular and automated.
        
        "Fwd IAT Std",              # Score: 1.053 | attack_mean: 10k vs benign_mean: 6.3M (ratio: 0.002x)
                                    # Reason: Forward timing variability. Bot/script behavior vs. human.
        
        "Fwd IAT Max",              # Score: 1.032 | attack_mean: 190k vs benign_mean: 15.6M (ratio: 0.012x)
                                    # Reason: Max forward time. Attacks are more continuous.
        
        "Flow IAT Mean",            # Score: 1.029 | attack_mean: 23k vs benign_mean: 1.8M (ratio: 0.013x)
                                    # Reason: Mean time between packets. A signature of automation.
        
        "Bwd Packet Length Min",    # Score: 1.029 | attack_mean: 0.0 vs benign_mean: 58.3 (ratio: 0.0x)
                                    # Reason: Min backward packet size. Many attacks are unidirectional.
        
        # === HIGHLY DISCRIMINATIVE FEATURES (Selection Score 0.8-1.0) ===
        "Bwd IAT Total", "Fwd IAT Mean", "Bwd IAT Max", "Bwd IAT Std",
        "Total Length of Fwd Packet",  # Score: 0.882 | ratio: 374x - Attacks with large payloads.
        "Total Fwd Packet",            # Score: 0.865 | ratio: 3590x - Flood/DDoS patterns.
        "RST Flag Count",              # Score: 0.764 | ratio: 0.002x - Attacks often avoid RST flags.
        
        # === ADDITIONAL USEFUL FEATURES (non-privacy-sensitive) ===
        "Protocol", "Flow Duration", "Total Bwd packets",
        "Flow Bytes/s", "Flow Packets/s", "Flow IAT Min", 
        "Fwd Header Length", "Bwd Header Length",
        "Average Packet Size", "Fwd Segment Size Avg", "Bwd Segment Size Avg"
    ]
}

# ==============================================================================
# PREPROCESSING CONFIGURATION
# ==============================================================================

PREPROCESSING_CONFIG = {
    # Sampling and balancing
    "sample_size": 10000,
    "balance_strategy": "security",  # Strategy: 50% BENIGN, 50% ATTACKS
    "benign_ratio": 0.5,
    "max_samples_per_class": 100000,
    "min_samples_per_class": 1000,

    # Caching for optimization
    "cache_enabled": True,
    "cache_dir": "preprocessed_cache",
    "delete_processed_files": False,  # If True, deletes source CSVs after caching.
    "parallel_processes": 2,  # Number of processes for parallel caching (can be increased on high-mem machines).
    
    # Time windows for sequential models (GRU, LSTM)
    "use_time_windows": True,
    "window_size": 10,
    "step": 5,
    
    # IP to octet conversion
    "convert_ip_to_octets": True,

    # Cache model-ready data (X,y,label) for faster training iterations
    "model_cache_enabled": True,
    "model_cache_dir": "model_cache"
}

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

TRAINING_CONFIG = {
    # Output directory for models and logs
    "output_path": "models/",
    
    # Validation strategy: "k_fold" or "train_test_split"
    "validation_strategy": "k_fold",
    "k_fold_splits": 5,
    "test_size": 0.2,
    
    # Default model type if not specified in the benchmark command
    "default_model_type": "gru",
    
    # --- Hyperparameters ---
    # This structure allows for both common and model-specific hyperparameters.
    # The training script will merge the common ones with the model-specific ones.
    "hyperparameters": {
        # Common parameters applicable to all models. Can be overridden by command-line args.
        "common": {
            "batch_size": [64],
            "learning_rate": [0.001] # Default for non-RNNs
        },
        # Model-specific parameters, which override common ones if there's a conflict.
        # This is where the bug fix for GRU/LSTM is implemented.
        "model_specific": {
            "gru": {
                "units": [64],
                "activation": ["tanh"],
                "learning_rate": [0.01], # Unconventionally high, but proven to work for this model
                "epochs": [15],
                "batch_size": [64] # Optimal batch size for GRU
            },
            "lstm": {
                "units": [64],
                "activation": ["tanh"],
                "learning_rate": [0.01], # Unconventionally high, but proven to work for this model
                "epochs": [15]
            },
            "dense": {
                "activation": ["relu"]
            },
            "mlp_4_layer": {
                "activation": ["relu"],
                "hidden_layer_units": [[256, 128, 64, 32]] # Default layer sizes for non-hyperband runs.
            }
        }
    }
}

# ==============================================================================
# EVALUATION CONFIGURATION
# ==============================================================================

EVALUATION_CONFIG = {
    # Cybersecurity-focused metrics
    "cybersecurity_focus": True,
    "generate_confusion_matrix": True,
    "generate_roc_curves": True,
    "detailed_attack_analysis": True,
    
    # Output settings
    "save_plots": True,
    "save_detailed_reports": True
}

# ==============================================================================
# BENCHMARK CONFIGURATION
# ==============================================================================

BENCHMARK_CONFIG = {
    # Pre-defined test configurations
    "test_configs": [
        {
            "name": "baseline",
            "description": "Baseline without IP anonymization",
            "use_cryptopan": False
        },
        {
            "name": "cryptopan", 
            "description": "With Crypto-PAn IP anonymization",
            "use_cryptopan": True
        }
    ],
    
    # Time resolutions to test
    "time_resolutions": ["5s", "1m", "5m"],
    
    # Output settings
    "output_dir": "benchmark_results",
    "generate_visualizations": True,
    "save_intermediate_results": True,
    
    # Configuration for quick smoke tests
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
# HOMOMORPHIC ENCRYPTION (HE) CONFIGURATION
# ==============================================================================

HOMOMORPHIC_CONFIG = {
    # Enable HE
    "enabled": False,  # Set to True to enable encryption during training/inference
    
    # Encryption scheme
    "scheme": "CKKS",  # CKKS is suitable for real-valued data
    "poly_modulus_degree": 8192,  # Polynomial degree (impacts security vs. performance)
    "coeff_mod_bit_sizes": [60, 40, 40, 60],  # Bit sizes of the moduli in the chain
    "scale": 2**40,  # Scale for CKKS encoding (precision)
    
    # Features to encrypt (ONLY privacy-sensitive ones)
    "features_to_encrypt": [
        # IP octets (automatically generated from Src IP, Dst IP)
        "Src IP_Octet_1", "Src IP_Octet_2", "Src IP_Octet_3", "Src IP_Octet_4",
        "Dst IP_Octet_1", "Dst IP_Octet_2", "Dst IP_Octet_3", "Dst IP_Octet_4",
        # Port numbers
        "Src Port", "Dst Port"
    ],
    
    # Plaintext features (performance-critical but not privacy-sensitive)
    "plaintext_features": [
        # Timing features
        "Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Mean", "Bwd IAT Mean",
        # Size/packet features
        "Total Fwd Packet", "Total Bwd packets", 
        "Total Length of Fwd Packet", "Total Length of Bwd Packet",
        "Flow Bytes/s", "Flow Packets/s",
        "Average Packet Size", "Fwd Segment Size Avg", "Bwd Segment Size Avg",
        # Header features
        "Fwd Header Length", "Bwd Header Length",
        # Protocol
        "Protocol"
    ],
    
    # Performance settings for HE
    "batch_size_encrypted": 32,  # Often reduced due to HE computational overhead
    "batch_size_plaintext": 64,  # Normal batch size for non-encrypted features
    "enable_key_rotation": True,  # Key rotation for enhanced security
    "max_encrypted_depth": 5,    # Maximum depth of encrypted operations
}

# ==============================================================================
# FEDERATED LEARNING (FL) CONFIGURATION
# ==============================================================================

FEDERATED_CONFIG = {
    # Enable FL
    "enabled": False,  # Set to True to run in federated mode
    
    # Client configuration
    "num_clients": 5,  # Number of federated clients
    "client_data_split": "iid",  # Data distribution: "iid" or "non_iid"
    "min_clients_per_round": 3,  # Minimum number of clients for an aggregation round
    
    # Round configuration
    "num_rounds": 10,  # Number of federated rounds
    "local_epochs": 2,  # Number of local epochs per client per round
    "client_lr": 0.001,  # Client-side learning rate
    
    # Aggregation strategy
    "aggregation_method": "fedavg",  # e.g., "fedavg", "fedprox", "scaffold"
    "differential_privacy": False,  # Enable/disable DP on gradients
    "dp_noise_scale": 0.1,  # Noise scale for DP
    
    # Homomorphic encryption in FL for secure aggregation
    "use_homomorphic_aggregation": False,  # Use HE for aggregating model updates
    "encrypt_gradients": True,  # Encrypt only gradients, not the entire model
}

# ==============================================================================
# PREDICTION CONFIGURATION
# ==============================================================================

PREDICTION_CONFIG = {
    # Paths to mapping files generated during training, needed for inference
    "target_map_path": os.path.join(TRAINING_CONFIG["output_path"], "target_map.json"),
    "ip_map_path": os.path.join(TRAINING_CONFIG["output_path"], "ip_map.json"),
    "column_order_path": os.path.join(TRAINING_CONFIG["output_path"], "column_order.json")
}
