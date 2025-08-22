# -*- coding: utf-8 -*-
"""
Modulo per la Valutazione e Statistiche del Modello.
Genera statistiche complete, matrice di confusione e report dettagliati.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from datetime import datetime

def generate_confusion_matrix(y_true, y_pred, class_names, output_path, save_plot=True):
    """
    Genera e salva la matrice di confusione.
    
    Args:
        y_true: Etichette vere
        y_pred: Predizioni del modello
        class_names: Nomi delle classi
        output_path: Percorso per salvare i risultati
        save_plot: Se salvare anche il plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Salva matrice di confusione numerica
    cm_data = {
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "total_samples": len(y_true),
        "correct_predictions": np.sum(np.diag(cm)),
        "incorrect_predictions": len(y_true) - np.sum(np.diag(cm))
    }
    
    cm_path = os.path.join(output_path, "confusion_matrix.json")
    with open(cm_path, 'w') as f:
        json.dump(cm_data, f, indent=4)
    
    print(f"Matrice di confusione salvata in: {cm_path}")
    
    # Genera plot se richiesto
    if save_plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Matrice di Confusione')
        plt.ylabel('Etichetta Vera')
        plt.xlabel('Etichetta Predetta')
        plt.tight_layout()
        
        plot_path = os.path.join(output_path, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot matrice di confusione salvato in: {plot_path}")
    
    return cm_data

def generate_classification_metrics(y_true, y_pred, class_names, output_path):
    """
    Genera metriche dettagliate per ogni classe.
    
    Args:
        y_true: Etichette vere
        y_pred: Predizioni del modello
        class_names: Nomi delle classi
        output_path: Percorso per salvare i risultati
    """
    # Calcola metriche per classe
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    # Calcola accuratezza per classe
    class_accuracy = []
    for i in range(len(class_names)):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
        else:
            class_acc = 0.0
        class_accuracy.append(class_acc)
    
    # Crea dizionario con tutte le metriche
    metrics_data = {
        "overall_metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "weighted_precision": float(np.average(precision, weights=support)),
            "weighted_recall": float(np.average(recall, weights=support)),
            "weighted_f1": float(np.average(f1, weights=support))
        },
        "per_class_metrics": {}
    }
    
    # Aggiungi metriche per ogni classe
    for i, class_name in enumerate(class_names):
        metrics_data["per_class_metrics"][class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
            "accuracy": float(class_accuracy[i])
        }
    
    # Salva metriche
    metrics_path = os.path.join(output_path, "classification_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    print(f"Metriche di classificazione salvate in: {metrics_path}")
    return metrics_data

def generate_dataset_statistics(X, y, class_names, output_path):
    """
    Genera statistiche complete del dataset.
    
    Args:
        X: Features del dataset
        y: Etichette del dataset
        class_names: Nomi delle classi
        output_path: Percorso per salvare i risultati
    """
    # Statistiche generali
    total_samples = len(X)
    unique_classes = np.unique(y)
    
    # Distribuzione delle classi
    class_distribution = {}
    for i, class_name in enumerate(class_names):
        if i in unique_classes:
            count = np.sum(y == i)
            percentage = (count / total_samples) * 100
            class_distribution[class_name] = {
                "count": int(count),
                "percentage": float(percentage)
            }
        else:
            class_distribution[class_name] = {
                "count": 0,
                "percentage": 0.0
            }
    
    # Statistiche delle features
    feature_stats = {}
    if len(X.shape) > 1:
        for i in range(X.shape[1]):
            feature_stats[f"feature_{i}"] = {
                "mean": float(np.mean(X[:, i])),
                "std": float(np.std(X[:, i])),
                "min": float(np.min(X[:, i])),
                "max": float(np.max(X[:, i])),
                "median": float(np.median(X[:, i]))
            }
    
    # Crea report completo
    dataset_stats = {
        "dataset_overview": {
            "total_samples": total_samples,
            "total_features": X.shape[1] if len(X.shape) > 1 else 1,
            "number_of_classes": len(unique_classes),
            "classes_present": [int(c) for c in unique_classes],
            "timestamp": datetime.now().isoformat()
        },
        "class_distribution": class_distribution,
        "feature_statistics": feature_stats
    }
    
    # Salva statistiche
    stats_path = os.path.join(output_path, "dataset_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(dataset_stats, f, indent=4)
    
    print(f"Statistiche del dataset salvate in: {stats_path}")
    return dataset_stats

def generate_training_summary(training_log, best_model_path, output_path):
    """
    Genera un riepilogo del training.
    
    Args:
        training_log: Log del training
        best_model_path: Percorso del modello migliore
        output_path: Percorso per salvare i risultati
    """
    if not training_log:
        print("Nessun log di training disponibile")
        return None
    
    # Trova la migliore configurazione
    best_run = max(training_log, key=lambda x: x['accuracy'])
    
    # Statistiche del training
    accuracies = [run['accuracy'] for run in training_log]
    training_summary = {
        "training_overview": {
            "total_runs": len(training_log),
            "best_accuracy": float(best_run['accuracy']),
            "worst_accuracy": float(min(accuracies)),
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "best_model_path": best_model_path,
            "timestamp": datetime.now().isoformat()
        },
        "best_run": {
            "accuracy": float(best_run['accuracy']),
            "parameters": best_run['params']
        },
        "all_runs": training_log
    }
    
    # Salva riepilogo
    summary_path = os.path.join(output_path, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=4)
    
    print(f"Riepilogo del training salvato in: {summary_path}")
    return training_summary

def generate_comprehensive_report(X, y, y_pred, class_names, training_log, 
                                best_model_path, output_path, config=None):
    """
    Genera un report completo con tutte le statistiche.
    
    Args:
        X: Features del dataset
        y: Etichette vere
        y_pred: Predizioni del modello
        class_names: Nomi delle classi
        training_log: Log del training
        best_model_path: Percorso del modello migliore
        output_path: Percorso per salvare i risultati
        config: Configurazione utilizzata (opzionale)
    """
    os.makedirs(output_path, exist_ok=True)
    
    print("=== Generazione Report Completo ===")
    
    # Genera tutte le statistiche
    cm_data = generate_confusion_matrix(y, y_pred, class_names, output_path)
    metrics_data = generate_classification_metrics(y, y_pred, class_names, output_path)
    dataset_stats = generate_dataset_statistics(X, y, class_names, output_path)
    training_summary = generate_training_summary(training_log, best_model_path, output_path)
    
    # Crea report principale
    main_report = {
        "report_metadata": {
            "generation_timestamp": datetime.now().isoformat(),
            "dataset_samples": len(X),
            "model_accuracy": float(accuracy_score(y, y_pred))
        },
        "confusion_matrix": cm_data,
        "classification_metrics": metrics_data,
        "dataset_statistics": dataset_stats,
        "training_summary": training_summary,
        "configuration": config or {}
    }
    
    # Salva report principale
    main_report_path = os.path.join(output_path, "comprehensive_report.json")
    with open(main_report_path, 'w') as f:
        json.dump(main_report, f, indent=4)
    
    print(f"\nReport completo salvato in: {main_report_path}")
    print("=== Generazione Report Completata ===")
    
    return main_report

def print_summary_to_console(report_data):
    """
    Stampa un riepilogo delle statistiche principali nella console.
    
    Args:
        report_data: Dati del report generato
    """
    print("\n" + "="*60)
    print("RIEPILOGO STATISTICHE MODELLO")
    print("="*60)
    
    # Statistiche generali
    print(f"Dataset: {report_data['report_metadata']['dataset_samples']} campioni")
    print(f"Accuratezza complessiva: {report_data['report_metadata']['model_accuracy']:.4f}")
    
    # Matrice di confusione
    cm = report_data['confusion_matrix']
    print(f"\nMatrice di Confusione:")
    print(f"Predizioni corrette: {cm['correct_predictions']}")
    print(f"Predizioni errate: {cm['incorrect_predictions']}")
    
    # Metriche per classe
    metrics = report_data['classification_metrics']
    print(f"\nMetriche per Classe:")
    for class_name, class_metrics in metrics['per_class_metrics'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall: {class_metrics['recall']:.4f}")
        print(f"    F1-Score: {class_metrics['f1_score']:.4f}")
        print(f"    Support: {class_metrics['support']}")
    
    print("="*60)
