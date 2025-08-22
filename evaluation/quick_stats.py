# -*- coding: utf-8 -*-
"""
Modulo per Statistiche Rapide e Configurabili del Modello.
Permette di generare statistiche specifiche in base alla configurazione.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

def quick_model_evaluation(y_true, y_pred, class_names, output_path, config=None):
    """
    Valutazione rapida del modello con configurazione personalizzabile.
    
    Args:
        y_true: Etichette vere
        y_pred: Predizioni del modello
        class_names: Nomi delle classi
        output_path: Percorso per salvare i risultati
        config: Configurazione per le statistiche
    """
    if config is None:
        config = {
            "save_confusion_matrix": True,
            "save_classification_report": True,
            "save_performance_summary": True,
            "print_to_console": True
        }
    
    os.makedirs(output_path, exist_ok=True)
    
    # Calcola accuratezza
    accuracy = accuracy_score(y_true, y_pred)
    
    # Genera report di classificazione
    if config.get("save_classification_report", True):
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        report_path = os.path.join(output_path, "classification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Report di classificazione salvato in: {report_path}")
    
    # Genera riepilogo performance
    if config.get("save_performance_summary", True):
        performance_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(y_true),
            "accuracy": float(accuracy),
            "class_names": class_names,
            "predictions_summary": {
                "correct_predictions": int(np.sum(y_true == y_pred)),
                "incorrect_predictions": int(np.sum(y_true != y_pred))
            }
        }
        
        # Aggiungi statistiche per classe se richiesto
        if config.get("include_per_class_stats", True):
            per_class_stats = {}
            for i, class_name in enumerate(class_names):
                class_mask = (y_true == i)
                if np.sum(class_mask) > 0:
                    class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
                    per_class_stats[class_name] = {
                        "samples": int(np.sum(class_mask)),
                        "accuracy": float(class_accuracy)
                    }
            performance_summary["per_class_statistics"] = per_class_stats
        
        summary_path = os.path.join(output_path, "performance_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(performance_summary, f, indent=4)
        print(f"Riepilogo performance salvato in: {summary_path}")
    
    # Stampa nella console se richiesto
    if config.get("print_to_console", True):
        print("\n" + "="*50)
        print("VALUTAZIONE RAPIDA MODELLO")
        print("="*50)
        print(f"Accuratezza complessiva: {accuracy:.4f}")
        print(f"Campioni totali: {len(y_true)}")
        print(f"Predizioni corrette: {np.sum(y_true == y_pred)}")
        print(f"Predizioni errate: {np.sum(y_true != y_pred)}")
        
        if config.get("include_per_class_stats", True):
            print("\nStatistiche per Classe:")
            for i, class_name in enumerate(class_names):
                class_mask = (y_true == i)
                if np.sum(class_mask) > 0:
                    class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
                    print(f"  {class_name}: {class_accuracy:.4f} ({np.sum(class_mask)} campioni)")
        print("="*50)
    
    return performance_summary

def generate_custom_metrics(y_true, y_pred, custom_metrics_config, output_path):
    """
    Genera metriche personalizzate in base alla configurazione.
    
    Args:
        y_true: Etichette vere
        y_pred: Predizioni del modello
        custom_metrics_config: Configurazione per le metriche personalizzate
        output_path: Percorso per salvare i risultati
    """
    custom_results = {}
    
    for metric_name, metric_config in custom_metrics_config.items():
        if metric_config.get("enabled", True):
            try:
                if metric_name == "balanced_accuracy":
                    from sklearn.metrics import balanced_accuracy_score
                    value = balanced_accuracy_score(y_true, y_pred)
                elif metric_name == "cohen_kappa":
                    from sklearn.metrics import cohen_kappa_score
                    value = cohen_kappa_score(y_true, y_pred)
                elif metric_name == "matthews_corrcoef":
                    from sklearn.metrics import matthews_corrcoef
                    value = matthews_corrcoef(y_true, y_pred)
                else:
                    print(f"Metrica personalizzata '{metric_name}' non riconosciuta")
                    continue
                
                custom_results[metric_name] = {
                    "value": float(value),
                    "description": metric_config.get("description", ""),
                    "enabled": True
                }
                
            except Exception as e:
                print(f"Errore nel calcolo della metrica '{metric_name}': {e}")
                custom_results[metric_name] = {
                    "value": None,
                    "error": str(e),
                    "enabled": False
                }
    
    # Salva metriche personalizzate
    if custom_results:
        custom_metrics_path = os.path.join(output_path, "custom_metrics.json")
        with open(custom_metrics_path, 'w') as f:
            json.dump(custom_results, f, indent=4)
        print(f"Metriche personalizzate salvate in: {custom_metrics_path}")
    
    return custom_results

def create_evaluation_config(template_name="default"):
    """
    Crea una configurazione di valutazione predefinita.
    
    Args:
        template_name: Nome del template di configurazione
    
    Returns:
        Dizionario di configurazione
    """
    templates = {
        "default": {
            "save_confusion_matrix": True,
            "save_classification_report": True,
            "save_performance_summary": True,
            "print_to_console": True,
            "include_per_class_stats": True,
            "save_plots": False,
            "custom_metrics": []
        },
        "minimal": {
            "save_confusion_matrix": False,
            "save_classification_report": False,
            "save_performance_summary": True,
            "print_to_console": True,
            "include_per_class_stats": False,
            "save_plots": False,
            "custom_metrics": []
        },
        "comprehensive": {
            "save_confusion_matrix": True,
            "save_classification_report": True,
            "save_performance_summary": True,
            "print_to_console": True,
            "include_per_class_stats": True,
            "save_plots": True,
            "custom_metrics": ["balanced_accuracy", "cohen_kappa", "matthews_corrcoef"]
        }
    }
    
    return templates.get(template_name, templates["default"])
