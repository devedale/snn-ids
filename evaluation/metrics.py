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
    output_dir: str = "evaluation_results",
    model_config: Dict = None,
    class_loss_data: Optional[Dict] = None
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
        class_loss_data: Dati di loss per classe per epoca (opzionale)
        
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
        y_test, y_pred, y_pred_proba, cm, class_names, output_dir, model_config, class_loss_data
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
    output_dir: str,
    model_config: Dict = None,
    class_loss_data: Optional[Dict] = None
) -> List[str]:
    """Crea visualizzazioni PNG."""
    
    # Assicurati che la directory esista
    os.makedirs(output_dir, exist_ok=True)
    
    # Genera stringa con iperparametri per i titoli
    hyperparams_str = _format_hyperparameters_for_title(model_config)
    
    viz_paths = []
    
    # 0. Grafico Loss vs. Epochs per Classe (se disponibile)
    if class_loss_data:
        try:
            loss_plot_path = plot_class_loss_over_epochs(
                class_loss_data, class_names, output_dir, model_config
            )
            viz_paths.append(loss_plot_path)
        except Exception as e:
            print(f"  ⚠️ Errore durante la creazione del grafico di loss per classe: {e}")

    # 1. Matrice di Confusione Dettagliata
    num_classes = len(class_names)
    
    # Dimensioni fisse e leggibili
    plt.figure(figsize=(14, 12))
    
    # Heatmap semplice e chiara
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
    title_acc = 'Accuracy per Classe di Traffico'
    if hyperparams_str:
        title_acc += f'\n{hyperparams_str}'
    plt.title(title_acc, fontsize=14, fontweight='bold')
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

def plot_class_loss_over_epochs(
    class_losses: Dict[int, List[float]],
    class_names: List[str],
    output_dir: str,
    model_config: Dict = None
) -> str:
    """
    Crea e salva un'immagine contenente una griglia di grafici di loss per classe.

    Args:
        class_losses: Dizionario {class_index: [loss_epoch_1, ...]}
        class_names: Lista dei nomi delle classi.
        output_dir: Directory dove salvare il grafico.
        model_config: Configurazione del modello per il titolo del grafico.

    Returns:
        Path al file del grafico salvato.
    """
    num_classes = len(class_losses)
    if num_classes == 0:
        return ""

    # Determine grid size (e.g., 5x5 grid for up to 25 classes)
    ncols = 5
    nrows = int(np.ceil(num_classes / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 4 * nrows), constrained_layout=True)
    axes = axes.flatten() # Flatten to make it easy to iterate

    hyperparams_str = _format_hyperparameters_for_title(model_config)
    fig.suptitle(f'Training Loss per Classe vs. Epoche\n{hyperparams_str}', fontsize=20, fontweight='bold')

    for i, (class_idx, losses) in enumerate(class_losses.items()):
        ax = axes[i]

        # Filter out None values in case a class was missing in a fold
        valid_losses = [l for l in losses if l is not None]
        if not valid_losses:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12, alpha=0.5)
            ax.set_title(f"Classe: {class_names[class_idx]} (No Data)", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        epochs = range(1, len(valid_losses) + 1)
        ax.plot(epochs, valid_losses, marker='o', linestyle='-', markersize=4)

        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Index {class_idx}"
        ax.set_title(f"Classe: {class_name}", fontsize=10)
        ax.set_xlabel('Epoca', fontsize=8)
        ax.set_ylabel('Loss', fontsize=8)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Adjust tick font size for readability
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plot_path = os.path.join(output_dir, "class_loss_vs_epochs_subplots.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"  📉 Grafico loss per classe (subplots): {plot_path}")

    return plot_path

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
