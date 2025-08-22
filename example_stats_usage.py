#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esempio di utilizzo delle funzionalità di statistiche e valutazione.
Questo script mostra come:
1. Generare statistiche complete del modello
2. Utilizzare configurazioni personalizzate
3. Generare report dettagliati
"""

import numpy as np
import pandas as pd
from evaluation.stats import generate_comprehensive_report, print_summary_to_console
from evaluation.quick_stats import quick_model_evaluation, create_evaluation_config
from config.stats_config import STATS_CONFIG

def example_comprehensive_stats():
    """
    Esempio di generazione di statistiche complete
    """
    print("=== Esempio Statistiche Complete ===")
    
    # Simula dati di training e predizioni
    np.random.seed(42)
    n_samples = 1000
    n_classes = 4
    
    # Genera etichette vere e predizioni
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    
    # Simula alcune predizioni corrette per rendere l'esempio più realistico
    correct_indices = np.random.choice(n_samples, size=int(n_samples * 0.8), replace=False)
    y_pred[correct_indices] = y_true[correct_indices]
    
    # Genera features simulate
    X = np.random.randn(n_samples, 20)
    
    # Nomi delle classi
    class_names = ["Normal", "DoS", "PortScan", "BruteForce"]
    
    # Log di training simulato
    training_log = [
        {"params": {"lr": 0.001, "epochs": 10}, "accuracy": 0.75},
        {"params": {"lr": 0.0001, "epochs": 20}, "accuracy": 0.82},
        {"params": {"lr": 0.0005, "epochs": 15}, "accuracy": 0.79}
    ]
    
    best_model_path = "models/best_model.keras"
    
    # Genera report completo
    output_path = "example_output/statistics"
    report_data = generate_comprehensive_report(
        X=X, y=y_true, y_pred=y_pred, class_names=class_names,
        training_log=training_log, best_model_path=best_model_path,
        output_path=output_path, config=STATS_CONFIG
    )
    
    print(f"\nReport completo generato in: {output_path}")
    return report_data

def example_quick_stats():
    """
    Esempio di statistiche rapide e configurabili
    """
    print("\n=== Esempio Statistiche Rapide ===")
    
    # Simula dati
    np.random.seed(123)
    n_samples = 500
    n_classes = 3
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    
    # Simula predizioni corrette
    correct_indices = np.random.choice(n_samples, size=int(n_samples * 0.7), replace=False)
    y_pred[correct_indices] = y_true[correct_indices]
    
    class_names = ["Class_A", "Class_B", "Class_C"]
    
    # Configurazione personalizzata
    custom_config = {
        "save_confusion_matrix": True,
        "save_classification_report": True,
        "save_performance_summary": True,
        "print_to_console": True,
        "include_per_class_stats": True
    }
    
    # Genera statistiche rapide
    output_path = "example_output/quick_stats"
    performance_summary = quick_model_evaluation(
        y_true=y_true, y_pred=y_pred, class_names=class_names,
        output_path=output_path, config=custom_config
    )
    
    print(f"\nStatistiche rapide generate in: {output_path}")
    return performance_summary

def example_configuration_templates():
    """
    Esempio di utilizzo dei template di configurazione
    """
    print("\n=== Esempio Template di Configurazione ===")
    
    # Mostra i template disponibili
    templates = ["default", "minimal", "comprehensive"]
    
    for template_name in templates:
        config = create_evaluation_config(template_name)
        print(f"\nTemplate '{template_name}':")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Simula dati per testare un template
    np.random.seed(456)
    y_true = np.random.randint(0, 3, 200)
    y_pred = np.random.randint(0, 3, 200)
    correct_indices = np.random.choice(200, size=int(200 * 0.6), replace=False)
    y_pred[correct_indices] = y_true[correct_indices]
    
    class_names = ["Type_1", "Type_2", "Type_3"]
    
    # Usa template minimal
    minimal_config = create_evaluation_config("minimal")
    output_path = "example_output/minimal_stats"
    
    performance_summary = quick_model_evaluation(
        y_true=y_true, y_pred=y_pred, class_names=class_names,
        output_path=output_path, config=minimal_config
    )
    
    print(f"\nStatistiche minimal generate in: {output_path}")
    return performance_summary

def example_custom_analysis():
    """
    Esempio di analisi personalizzata
    """
    print("\n=== Esempio Analisi Personalizzata ===")
    
    # Simula dati
    np.random.seed(789)
    y_true = np.random.randint(0, 2, 300)  # Classificazione binaria
    y_pred = np.random.randint(0, 2, 300)
    correct_indices = np.random.choice(300, size=int(300 * 0.85), replace=False)
    y_pred[correct_indices] = y_true[correct_indices]
    
    class_names = ["Benign", "Malicious"]
    
    # Configurazione per metriche personalizzate
    custom_metrics_config = {
        "balanced_accuracy": {
            "enabled": True,
            "description": "Accuratezza bilanciata per dataset sbilanciati"
        },
        "cohen_kappa": {
            "enabled": True,
            "description": "Coefficiente Kappa di Cohen"
        },
        "matthews_corrcoef": {
            "enabled": True,
            "description": "Coefficiente di correlazione di Matthews"
        }
    }
    
    # Genera statistiche con metriche personalizzate
    output_path = "example_output/custom_analysis"
    os.makedirs(output_path, exist_ok=True)
    
    from evaluation.quick_stats import generate_custom_metrics
    custom_results = generate_custom_metrics(
        y_true, y_pred, custom_metrics_config, output_path
    )
    
    print(f"\nAnalisi personalizzata generata in: {output_path}")
    print("Metriche personalizzate calcolate:")
    for metric_name, result in custom_results.items():
        if result["enabled"]:
            print(f"  {metric_name}: {result['value']:.4f}")
    
    return custom_results

if __name__ == "__main__":
    import os
    
    # Crea directory di output
    os.makedirs("example_output", exist_ok=True)
    
    # Esegui tutti gli esempi
    try:
        # 1. Statistiche complete
        comprehensive_report = example_comprehensive_stats()
        
        # 2. Statistiche rapide
        quick_stats = example_quick_stats()
        
        # 3. Template di configurazione
        minimal_stats = example_configuration_templates()
        
        # 4. Analisi personalizzata
        custom_analysis = example_custom_analysis()
        
        print("\n" + "="*60)
        print("TUTTI GLI ESEMPI COMPLETATI CON SUCCESSO!")
        print("="*60)
        print("I file sono stati salvati nelle seguenti directory:")
        print("- example_output/statistics/ (statistiche complete)")
        print("- example_output/quick_stats/ (statistiche rapide)")
        print("- example_output/minimal_stats/ (template minimal)")
        print("- example_output/custom_analysis/ (analisi personalizzata)")
        
    except Exception as e:
        print(f"Errore durante l'esecuzione degli esempi: {e}")
        import traceback
        traceback.print_exc()
