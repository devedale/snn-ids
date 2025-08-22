# -*- coding: utf-8 -*-
"""
Configurazione per il Benchmark Completo.
Configura tutte le opzioni per il benchmark con finestre temporali e confronto Crypto-PAn.
"""

BENCHMARK_CONFIG = {
    # Configurazione finestre temporali
    "time_resolutions": {
        "quick_test": ['5s', '1m'],  # Solo 2 risoluzioni per test veloci
        "standard": ['1s', '5s', '10s', '1m', '5m', '10m'],  # Configurazione standard
        "extended": ['1s', '5s', '10s', '30s', '1m', '5m', '10m', '30m', '1h'],  # Configurazione estesa
        "custom": ['2s', '7s', '15s', '45s', '2m', '7m', '15m']  # Configurazione personalizzata
    },
    
    # Configurazione dataset
    "dataset": {
        "sample_sizes": {
            "tiny": 1000,      # Test molto veloci
            "small": 5000,     # Test veloci
            "medium": 25000,   # Test standard
            "large": 100000,   # Test completi
            "full": None       # Dataset completo
        },
        "min_window_size": 5,  # Dimensione minima finestra per essere considerata valida
        "max_window_size": 1000,  # Dimensione massima finestra
        "timestamp_column": "Timestamp"
    },
    
    # Configurazione test
    "test_configurations": {
        "baseline": {
            "name": "baseline",
            "use_cryptopan": False,
            "description": "Baseline senza anonimizzazione",
            "enabled": True
        },
        "cryptopan": {
            "name": "cryptopan", 
            "use_cryptopan": True,
            "description": "Con anonimizzazione Crypto-PAn",
            "enabled": True
        },
        "cryptopan_enhanced": {
            "name": "cryptopan_enhanced",
            "use_cryptopan": True,
            "description": "Crypto-PAn con chiave migliorata",
            "enabled": False  # Disabilitato di default
        }
    },
    
    # Configurazione output
    "output": {
        "base_directory": "benchmark_results",
        "save_intermediate_results": True,
        "save_detailed_logs": True,
        "generate_comparison_plots": True,
        "save_individual_reports": True,
        "compression": False  # Compressione dei risultati
    },
    
    # Configurazione training
    "training": {
        "model_types": ["lstm", "gru", "dense"],
        "validation_strategy": "k_fold",  # "k_fold" o "train_test_split"
        "k_fold_splits": 3,  # Per velocizzare i test
        "max_training_time": 300,  # Secondi massimi per training
        "early_stopping": True,
        "patience": 10
    },
    
    # Configurazione statistiche
    "statistics": {
        "generate_confusion_matrix": True,
        "save_confusion_matrix_plot": True,
        "calculate_per_class_metrics": True,
        "include_feature_statistics": True,
        "custom_metrics": [
            "balanced_accuracy",
            "cohen_kappa", 
            "matthews_corrcoef"
        ]
    },
    
    # Configurazione visualizzazioni
    "visualization": {
        "plot_style": "seaborn",
        "color_scheme": "RdYlGn",
        "plot_dpi": 300,
        "plot_format": "png",
        "generate_radar_chart": True,
        "generate_heatmap": True,
        "generate_time_series": True
    },
    
    # Configurazione performance
    "performance": {
        "parallel_processing": False,  # Elaborazione parallela (richiede più RAM)
        "memory_limit_gb": 8,  # Limite memoria per test
        "cpu_cores": 4,  # Core CPU da utilizzare
        "batch_size_optimization": True
    },
    
    # Configurazione sicurezza
    "security": {
        "cryptopan_key_size": 32,  # Byte per la chiave Crypto-PAn
        "save_encryption_keys": False,  # Non salvare le chiavi in produzione
        "key_rotation": False,  # Rotazione automatica delle chiavi
        "audit_log": True  # Log delle operazioni di anonimizzazione
    },
    
    # Configurazione report
    "reporting": {
        "generate_executive_summary": True,
        "include_performance_charts": True,
        "include_statistical_analysis": True,
        "export_formats": ["json", "csv", "html"],
        "auto_email_report": False
    },
    
    # Configurazione validazione
    "validation": {
        "cross_validation_folds": 3,
        "test_size": 0.2,
        "random_state": 42,
        "stratify_sampling": True
    }
}

def get_benchmark_config(config_name: str = "standard") -> dict:
    """
    Ottiene una configurazione predefinita per il benchmark.
    
    Args:
        config_name: Nome della configurazione
    
    Returns:
        Dizionario di configurazione
    """
    configs = {
        "quick": {
            "time_resolutions": BENCHMARK_CONFIG["time_resolutions"]["quick_test"],
            "sample_size": BENCHMARK_CONFIG["dataset"]["sample_sizes"]["small"],
            "output_dir": "quick_benchmark_results",
            "save_intermediate_results": False,
            "generate_comparison_plots": True,
            "k_fold_splits": 2,
            "max_training_time": 120
        },
        "standard": {
            "time_resolutions": BENCHMARK_CONFIG["time_resolutions"]["standard"],
            "sample_size": BENCHMARK_CONFIG["dataset"]["sample_sizes"]["medium"],
            "output_dir": "standard_benchmark_results",
            "save_intermediate_results": True,
            "generate_comparison_plots": True,
            "k_fold_splits": 3,
            "max_training_time": 300
        },
        "extended": {
            "time_resolutions": BENCHMARK_CONFIG["time_resolutions"]["extended"],
            "sample_size": BENCHMARK_CONFIG["dataset"]["sample_sizes"]["large"],
            "output_dir": "extended_benchmark_results",
            "save_intermediate_results": True,
            "generate_comparison_plots": True,
            "k_fold_splits": 5,
            "max_training_time": 600
        },
        "production": {
            "time_resolutions": BENCHMARK_CONFIG["time_resolutions"]["standard"],
            "sample_size": BENCHMARK_CONFIG["dataset"]["sample_sizes"]["full"],
            "output_dir": "production_benchmark_results",
            "save_intermediate_results": True,
            "generate_comparison_plots": True,
            "k_fold_splits": 5,
            "max_training_time": 1800,
            "parallel_processing": True,
            "audit_log": True
        }
    }
    
    return configs.get(config_name, configs["standard"])

def create_custom_benchmark_config(**kwargs) -> dict:
    """
    Crea una configurazione personalizzata per il benchmark.
    
    Args:
        **kwargs: Parametri personalizzati
    
    Returns:
        Dizionario di configurazione personalizzata
    """
    # Configurazione di base
    base_config = get_benchmark_config("standard")
    
    # Applica personalizzazioni
    for key, value in kwargs.items():
        if key in base_config:
            base_config[key] = value
    
    return base_config

# Configurazioni predefinite per diversi scenari
QUICK_BENCHMARK_CONFIG = get_benchmark_config("quick")
STANDARD_BENCHMARK_CONFIG = get_benchmark_config("standard")
EXTENDED_BENCHMARK_CONFIG = get_benchmark_config("extended")
PRODUCTION_BENCHMARK_CONFIG = get_benchmark_config("production")
