# -*- coding: utf-8 -*-
"""
Configurazione per le Statistiche e la Valutazione del Modello.
"""

STATS_CONFIG = {
    # Configurazione per la generazione delle statistiche
    "generate_confusion_matrix": True,
    "save_confusion_matrix_plot": True,
    "plot_dpi": 300,
    "plot_format": "png",
    
    # Configurazione per le metriche
    "calculate_per_class_metrics": True,
    "calculate_overall_metrics": True,
    "include_feature_statistics": True,
    
    # Configurazione per il report
    "generate_training_summary": True,
    "generate_dataset_statistics": True,
    "save_individual_files": True,
    "save_comprehensive_report": True,
    
    # Configurazione per la console
    "print_console_summary": True,
    "console_summary_format": "detailed",  # "detailed" o "compact"
    
    # Configurazione per i percorsi di output
    "statistics_output_dir": "statistics",
    "plots_output_dir": "plots",
    "reports_output_dir": "reports",
    
    # Configurazione per la validazione
    "validation_split": 0.2,
    "use_cross_validation": False,
    "cross_validation_folds": 5,
    
    # Configurazione per le metriche avanzate
    "calculate_roc_auc": True,
    "calculate_precision_recall_curves": True,
    "include_classification_report": True,
    
    # Configurazione per la personalizzazione
    "custom_metrics": [],  # Lista di metriche personalizzate
    "plot_style": "seaborn",  # "seaborn", "matplotlib", "plotly"
    "color_scheme": "Blues",  # Schema colori per le heatmap
}
