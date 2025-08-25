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
import pandas as pd

def evaluate_model_comprehensive(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    output_dir: str = "evaluation_results",
    model_config: Dict = None
) -> Dict[str, Any]:
    """
    Valutazione completa di un modello con metriche cybersecurity.
    
    Args:
        model: Modello addestrato
        X_test: Features di test
        y_test: Labels di test
        class_names: Nomi delle classi
        output_dir: Directory output
        model_config: Configurazione del modello (iperparametri, ecc.)
        
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
    # Usa classification_report per metriche dettagliate
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    print(f"  ✅ Accuratezza: {accuracy:.4f}")
    
    # Matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    
    # Metriche cybersecurity
    cyber_metrics = _calculate_cybersecurity_metrics(y_test, y_pred, class_names)
    
    # Visualizzazioni
    viz_paths = _create_visualizations(
        y_test, y_pred, y_pred_proba, cm, class_names, report_dict, output_dir, model_config
    )
    
    # Report completo
    report = {
        'model_config': model_config,
        'classification_report': report_dict,
        'cybersecurity_metrics': cyber_metrics,
        'confusion_matrix': cm.tolist(),
        'binary_confusion_matrix': cyber_metrics.get('binary_confusion_matrix'),
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
    report_dict: Dict,
    output_dir: str,
    model_config: Dict = None
) -> List[str]:
    """Crea visualizzazioni PNG."""
    
    # Assicurati che la directory esista
    os.makedirs(output_dir, exist_ok=True)
    
    # Genera stringa con iperparametri per i titoli
    hyperparams_str = _format_hyperparameters_for_title(model_config)
    
    viz_paths = []
    
    # 1. Matrice di Confusione Dettagliata
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 10, 'weight': 'bold'},
                cbar_kws={'label': 'Numero di Campioni'})
    title = 'Matrice di Confusione - Dettagliata\n(Righe=Reale, Colonne=Predizione)'
    if hyperparams_str:
        title += f'\n{hyperparams_str}'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predizione', fontsize=12, fontweight='bold')
    plt.ylabel('Classe Reale', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix_detailed.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    viz_paths.append(cm_path)
    print(f"  🖼️ Matrice confusione dettagliata: {cm_path}")
    
    # 2. Matrice di Confusione Binaria (BENIGN vs ATTACCHI)
    benign_idx = class_names.index('BENIGN') if 'BENIGN' in class_names else -1
    if benign_idx != -1:
        y_true_binary = (y_true != benign_idx).astype(int)
        y_pred_binary = (y_pred != benign_idx).astype(int)
        cm_binary = confusion_matrix(y_true_binary, y_pred_binary)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['BENIGN', 'ATTACCO'], yticklabels=['BENIGN', 'ATTACCO'])
        title_binary = 'Matrice di Confusione - Cybersecurity Focus\n(BENIGN vs ATTACCHI)'
        if hyperparams_str:
            title_binary += f'\n{hyperparams_str}'
        plt.title(title_binary, fontsize=12, fontweight='bold')
        plt.xlabel('Predizione', fontsize=12)
        plt.ylabel('Reale', fontsize=12)
        plt.tight_layout()

        cm_binary_path = os.path.join(output_dir, "confusion_matrix_cybersecurity.png")
        plt.savefig(cm_binary_path, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths.append(cm_binary_path)
        print(f"  🖼️ Matrice confusione cybersecurity: {cm_binary_path}")

    # 3. Grafici F1-Score e Recall per classe
    metric_plot_paths = _create_per_class_metric_plots(report_dict, class_names, output_dir)
    viz_paths.extend(metric_plot_paths)

    # 4. ROC Curve (solo se multiclasse e dataset sufficiente)
    if len(class_names) > 2 and y_pred_proba.shape[1] > 2 and len(y_true) > 10:
        try:
            plt.figure(figsize=(10, 8))
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
            if y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack([1-y_true_bin, y_true_bin])
            
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(min(len(class_names), y_true_bin.shape[1])):
                if np.any(y_true == i) and len(np.unique(y_true_bin[:, i])) > 1:
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
            
            if len(fpr) > 0:
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Random Classifier')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Tasso Falsi Positivi (FPR)', fontsize=14, fontweight='bold')
                plt.ylabel('Tasso Veri Positivi (TPR)', fontsize=14, fontweight='bold')
                plt.title('Curve ROC - Analisi Performance per Classe', fontsize=16, fontweight='bold', pad=20)
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

def _create_per_class_metric_plots(
    report_dict: Dict,
    class_names: List[str],
    output_dir: str
) -> List[str]:
    """Crea grafici per F1-score e Recall per ogni classe."""

    report_df = pd.DataFrame(report_dict).transpose()
    plot_df = report_df.loc[class_names] # Filtra solo le classi, esclude medie

    plot_paths = []

    metrics_to_plot = {'f1-score': 'F1 Score', 'recall': 'Recall'}
    for metric, title in metrics_to_plot.items():
        plt.figure(figsize=(12, 8))
        sns.barplot(x=plot_df.index, y=plot_df[metric])
        plt.title(f'{title} per Classe', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(title)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"per_class_{metric.replace('-','_')}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        plot_paths.append(plot_path)
        print(f"  🖼️ Grafico {title} per classe: {plot_path}")

    return plot_paths

def _format_hyperparameters_for_title(model_config: Dict = None) -> str:
    """Formatta gli iperparametri per i titoli delle visualizzazioni."""
    if not model_config:
        return ""
    
    hyperparams = model_config.get('hyperparameters', {})
    model_type = model_config.get('model_type', 'Unknown')
    
    # Estrai parametri rilevanti
    params_str = []
    params_str.append(f"Model: {model_type.upper()}")
    
    if 'epochs' in hyperparams:
        epochs = hyperparams['epochs'][0] if isinstance(hyperparams['epochs'], list) else hyperparams['epochs']
        params_str.append(f"Epochs: {epochs}")
    
    if 'batch_size' in hyperparams:
        batch_size = hyperparams['batch_size'][0] if isinstance(hyperparams['batch_size'], list) else hyperparams['batch_size']
        params_str.append(f"Batch: {batch_size}")
    
    if 'learning_rate' in hyperparams:
        lr = hyperparams['learning_rate'][0] if isinstance(hyperparams['learning_rate'], list) else hyperparams['learning_rate']
        params_str.append(f"LR: {lr}")
    
    if 'activation' in hyperparams:
        activation = hyperparams['activation'][0] if isinstance(hyperparams['activation'], list) else hyperparams['activation']
        params_str.append(f"Act: {activation}")
    
    # Parametri specifici del modello
    if model_type.lower() == 'gru' and 'gru_units' in hyperparams:
        units = hyperparams['gru_units'][0] if isinstance(hyperparams['gru_units'], list) else hyperparams['gru_units']
        params_str.append(f"Units: {units}")
    elif model_type.lower() == 'lstm' and 'lstm_units' in hyperparams:
        units = hyperparams['lstm_units'][0] if isinstance(hyperparams['lstm_units'], list) else hyperparams['lstm_units']
        params_str.append(f"Units: {units}")
    
    if 'dropout' in hyperparams:
        dropout = hyperparams['dropout'][0] if isinstance(hyperparams['dropout'], list) else hyperparams['dropout']
        if dropout > 0:
            params_str.append(f"Dropout: {dropout}")
    
    return " | ".join(params_str)

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

def create_benchmark_summary_csv(
    results_list: List[Dict],
    output_dir: str
) -> str:
    """
    Crea un file CSV riassuntivo per i risultati di un benchmark.

    Args:
        results_list: Lista di dizionari, ognuno è il report di una run.
        output_dir: Directory dove salvare il file CSV.

    Returns:
        Path del file CSV generato.
    """
    if not results_list:
        print("⚠️ La lista dei risultati è vuota, non genero il CSV.")
        return ""

    os.makedirs(output_dir, exist_ok=True)

    summary_data = []

    # Estrai tutti i nomi delle classi di attacco per le colonne del CSV
    all_attack_classes = set()
    for result in results_list:
        report = result.get('classification_report', {})
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and class_name != 'BENIGN':
                all_attack_classes.add(class_name)

    sorted_attack_classes = sorted(list(all_attack_classes))

    for result in results_list:
        model_config = result.get('model_config', {})
        hyperparams = model_config.get('hyperparameters', {})
        report = result.get('classification_report', {})
        cyber_metrics = result.get('cybersecurity_metrics', {})

        row = {
            'model_type': model_config.get('model_type', 'N/A'),
            'epochs': hyperparams.get('epochs', 'N/A'),
            'batch_size': hyperparams.get('batch_size', 'N/A'),
            'learning_rate': hyperparams.get('learning_rate', 'N/A'),
            'accuracy': report.get('accuracy', 0),
            'detection_rate': cyber_metrics.get('detection_rate', 0),
            'false_alarm_rate': cyber_metrics.get('false_alarm_rate', 0),
        }

        # Aggiungi F1-score per ogni classe di attacco
        for attack_class in sorted_attack_classes:
            row[f'f1_{attack_class}'] = report.get(attack_class, {}).get('f1-score', 0)

        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Riorganizza le colonne per una migliore leggibilità
    base_cols = ['model_type', 'epochs', 'batch_size', 'learning_rate', 'accuracy', 'detection_rate', 'false_alarm_rate']
    f1_cols = [f'f1_{ac}' for ac in sorted_attack_classes]
    df = df[base_cols + f1_cols]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir, f"benchmark_summary_{timestamp}.csv")

    df.to_csv(csv_path, index=False)

    print(f"📄 File CSV riassuntivo salvato in: {csv_path}")

    return csv_path
