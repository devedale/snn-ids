# -*- coding: utf-8 -*-
"""
Metriche di Valutazione SNN-IDS
Sistema completo per valutazione modelli con focus cybersecurity.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo per server
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')  # Stile pulito
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import json
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
import tensorflow as tf

def evaluate_model_comprehensive(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    output_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """
    Valutazione completa di un modello con metriche cybersecurity.
    
    Args:
        model: Modello addestrato
        X_test: Features di test
        y_test: Labels di test
        class_names: Nomi delle classi
        output_dir: Directory output
        
    Returns:
        Dizionario con tutte le metriche
    """
    print("📊 Valutazione completa del modello")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Predizioni
    print(f"  🔮 Generazione predizioni su {len(X_test)} campioni...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print(f"  📊 Predizioni completate, classi uniche: {len(np.unique(y_pred))}")
    
    # Metriche base
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    print(f"  ✅ Accuratezza: {accuracy:.4f}")
    
    # Matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    
    # Metriche cybersecurity
    cyber_metrics = _calculate_cybersecurity_metrics(y_test, y_pred, class_names)
    
    # Visualizzazioni
    viz_paths = _create_visualizations(
        y_test, y_pred, y_pred_proba, cm, class_names, output_dir
    )
    
    # Report completo
    report = {
        'basic_metrics': {
            'accuracy': float(accuracy),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist()
        },
        'cybersecurity_metrics': cyber_metrics,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'visualizations': viz_paths
    }
    
    # Salva report nella directory parent (accanto a visualizations/)
    parent_dir = os.path.dirname(output_dir)
    report_path = os.path.join(parent_dir, "evaluation_report.json")
    
    os.makedirs(parent_dir, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"  💾 Report JSON: {report_path}")
    print(f"  🖼️ Visualizzazioni PNG: {len(viz_paths)} file in {output_dir}")
    
    # Salva anche summary delle visualizzazioni
    viz_summary = {
        'timestamp': str(datetime.now()),
        'total_images': len(viz_paths),
        'generated_files': [os.path.basename(path) for path in viz_paths],
        'output_directory': output_dir
    }
    
    summary_path = os.path.join(parent_dir, "visualization_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(viz_summary, f, indent=2)
    
    print(f"  📋 Summary: {summary_path}")
    
    return report

def _calculate_cybersecurity_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str]
) -> Dict[str, Any]:
    """Calcola metriche specifiche per cybersecurity."""
    
    # Identifica classe BENIGN (indice 0 di solito)
    benign_idx = 0
    if 'BENIGN' in class_names:
        benign_idx = class_names.index('BENIGN')
    
    # Converti in binario: BENIGN vs ATTACCHI
    y_true_binary = (y_true != benign_idx).astype(int)  # 1 = attacco, 0 = benign
    y_pred_binary = (y_pred != benign_idx).astype(int)
    
    # Matrice di confusione binaria
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))  # True Negative
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))  # False Positive
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))  # False Negative
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))  # True Positive
    
    # Metriche cybersecurity
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall per attacchi
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # FPR
    precision_attacks = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision per attacchi
    
    # Analisi per tipo di attacco
    attack_analysis = {}
    for i, class_name in enumerate(class_names):
        if i != benign_idx:  # Solo attacchi
            class_mask = (y_true == i)
            if np.any(class_mask):
                class_tp = np.sum((y_true == i) & (y_pred == i))
                class_total = np.sum(class_mask)
                detection_rate_class = class_tp / class_total if class_total > 0 else 0.0
                
                attack_analysis[class_name] = {
                    'samples': int(class_total),
                    'detected': int(class_tp),
                    'detection_rate': float(detection_rate_class)
                }
    
    return {
        'detection_rate': float(detection_rate),
        'false_alarm_rate': float(false_alarm_rate), 
        'precision_for_attacks': float(precision_attacks),
        'binary_confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        },
        'attack_types_analysis': attack_analysis
    }

def _create_visualizations(
    y_true: np.ndarray,
    y_pred: np.ndarray, 
    y_pred_proba: np.ndarray,
    cm: np.ndarray,
    class_names: List[str],
    output_dir: str
) -> List[str]:
    """Crea visualizzazioni PNG."""
    
    # Assicurati che la directory esista
    os.makedirs(output_dir, exist_ok=True)
    
    viz_paths = []
    
    # 1. Matrice di Confusione Dettagliata
    num_classes = len(class_names)
    
    # Dimensioni fisse e leggibili
    plt.figure(figsize=(14, 12))
    
    # Heatmap semplice e chiara
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 10, 'weight': 'bold'},
                cbar_kws={'label': 'Numero di Campioni'})
    
    plt.title('Matrice di Confusione - Dettagliata\n(Righe=Reale, Colonne=Predizione)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predizione', fontsize=12, fontweight='bold')
    plt.ylabel('Classe Reale', fontsize=12, fontweight='bold')
    
    # Etichette sempre a 45° per leggibilità
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "confusion_matrix_detailed.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    viz_paths.append(cm_path)
    print(f"  🖼️ Matrice confusione dettagliata: {cm_path}")
    
    # 2. Matrice di Confusione Binaria (BENIGN vs ATTACCHI)
    benign_idx = 0
    if 'BENIGN' in class_names:
        benign_idx = class_names.index('BENIGN')
    
    y_true_binary = (y_true != benign_idx).astype(int)
    y_pred_binary = (y_pred != benign_idx).astype(int)
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Reds',
                xticklabels=['BENIGN', 'ATTACCO'], yticklabels=['BENIGN', 'ATTACCO'])
    plt.title('Matrice di Confusione - Cybersecurity Focus', fontsize=16, fontweight='bold')
    plt.xlabel('Predizione', fontsize=12)
    plt.ylabel('Reale', fontsize=12)
    plt.tight_layout()
    
    cm_binary_path = os.path.join(output_dir, "confusion_matrix_cybersecurity.png")
    plt.savefig(cm_binary_path, dpi=300, bbox_inches='tight')
    plt.close()
    viz_paths.append(cm_binary_path)
    print(f"  🖼️ Matrice confusione cybersecurity: {cm_binary_path}")
    
    # 3. Distribuzione Accuracy per Classe
    class_accuracies = []
    for i in range(len(class_names)):
        class_mask = (y_true == i)
        if np.any(class_mask):
            class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), class_accuracies, 
                   color=['green' if i == benign_idx else 'red' for i in range(len(class_names))])
    plt.title('Accuracy per Classe di Traffico', fontsize=16, fontweight='bold')
    plt.xlabel('Classe', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Aggiungi valori sopra le barre
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    acc_path = os.path.join(output_dir, "accuracy_per_class.png")
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close()
    viz_paths.append(acc_path)
    print(f"  🖼️ Accuracy per classe: {acc_path}")
    
    # 4. ROC Curve (solo se multiclasse e dataset sufficiente)
    if len(class_names) > 2 and y_pred_proba.shape[1] > 2 and len(y_true) > 10:
        try:
            plt.figure(figsize=(10, 8))
            
            # Binarizza le etichette
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
            if y_true_bin.shape[1] == 1:  # Caso binario
                y_true_bin = np.hstack([1-y_true_bin, y_true_bin])
            
            # Calcola ROC per ogni classe
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(min(len(class_names), y_true_bin.shape[1])):
                if np.any(y_true == i) and len(np.unique(y_true_bin[:, i])) > 1:  # Solo se la classe esiste e ha variabilità
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    plt.plot(fpr[i], tpr[i], 
                            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
            
            if len(fpr) > 0:  # Solo se abbiamo almeno una curva
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Random Classifier')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Tasso Falsi Positivi (FPR)', fontsize=14, fontweight='bold')
                plt.ylabel('Tasso Veri Positivi (TPR)', fontsize=14, fontweight='bold')
                plt.title('Curve ROC - Analisi Performance per Classe', fontsize=16, fontweight='bold', pad=20)
                
                # Leggenda intelligente
                if len(fpr) > 8:
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                else:
                    plt.legend(loc='lower right', fontsize=12)
                
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                roc_path = os.path.join(output_dir, "roc_curves.png")
                plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths.append(roc_path)
                print(f"  📈 ROC curves: {roc_path}")
            else:
                print(f"  ⚠️ Skipping ROC: dataset troppo piccolo o senza variabilità")
            
        except Exception as e:
            print(f"  ⚠️ Errore ROC curve: {e}")
    else:
        print(f"  ⚠️ Skipping ROC: dataset piccolo ({len(y_true)} campioni) o binario")
    
    return viz_paths

def create_benchmark_comparison(
    results_list: List[Dict],
    output_dir: str = "benchmark_comparison"
) -> str:
    """
    Crea visualizzazione comparativa di più modelli.
    
    Args:
        results_list: Lista di risultati di valutazione
        output_dir: Directory output
        
    Returns:
        Path del grafico di confronto
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Estrai metriche
    model_names = []
    accuracies = []
    detection_rates = []
    false_alarms = []
    
    for result in results_list:
        model_names.append(result.get('model_name', 'Modello'))
        accuracies.append(result.get('basic_metrics', {}).get('accuracy', 0))
        
        cyber_metrics = result.get('cybersecurity_metrics', {})
        detection_rates.append(cyber_metrics.get('detection_rate', 0))
        false_alarms.append(cyber_metrics.get('false_alarm_rate', 0))
    
    # Grafico comparativo
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    bars1 = ax1.bar(model_names, accuracies, color='skyblue')
    ax1.set_title('Accuratezza per Modello', fontweight='bold')
    ax1.set_ylabel('Accuratezza')
    ax1.set_ylim(0, 1)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Detection Rate
    bars2 = ax2.bar(model_names, detection_rates, color='lightgreen')
    ax2.set_title('Tasso di Rilevamento Attacchi', fontweight='bold')
    ax2.set_ylabel('Detection Rate')
    ax2.set_ylim(0, 1)
    for bar, dr in zip(bars2, detection_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{dr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # False Alarm Rate
    bars3 = ax3.bar(model_names, false_alarms, color='lightcoral')
    ax3.set_title('Tasso Falsi Allarmi', fontweight='bold')
    ax3.set_ylabel('False Alarm Rate')
    ax3.set_ylim(0, max(false_alarms) * 1.1 if false_alarms else 1)
    for bar, fa in zip(bars3, false_alarms):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{fa:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Scatter Detection vs False Alarm
    ax4.scatter(false_alarms, detection_rates, s=100, alpha=0.7)
    for i, name in enumerate(model_names):
        ax4.annotate(name, (false_alarms[i], detection_rates[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax4.set_xlabel('Tasso Falsi Allarmi')
    ax4.set_ylabel('Tasso Rilevamento')
    ax4.set_title('Performance Trade-off', fontweight='bold')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    comparison_path = os.path.join(output_dir, "models_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Confronto modelli salvato: {comparison_path}")
    return comparison_path
