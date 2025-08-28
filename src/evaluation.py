# -*- coding: utf-8 -*-
"""
Model Evaluation Metrics for SNN-IDS
This module provides a comprehensive system for evaluating classification models
with a special focus on cybersecurity metrics. It generates and saves various
visualizations and a detailed JSON report.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import tensorflow as tf

plt.style.use('default') # Use a clean, default plot style

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
    Performs a comprehensive evaluation of a trained model.
    It calculates standard metrics, cybersecurity-specific metrics, generates
    visualizations, and saves a detailed report.

    Args:
        model: The trained Keras model to evaluate.
        X_test: The test features.
        y_test: The test labels.
        class_names: A list of string names for the classes.
        output_dir: The directory where visualizations will be saved.
        model_config: The model's configuration (for titles and reports).
        class_loss_data: Optional data of per-class loss per epoch.

    Returns:
        A dictionary containing all calculated metrics and paths to visualizations.
    """
    print("📊 Starting comprehensive model evaluation...")
    os.makedirs(output_dir, exist_ok=True)

    # Generate predictions
    print(f"  🔮 Generating predictions on {len(X_test)} samples...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # --- Core Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    cm = confusion_matrix(y_test, y_pred)

    # --- Cybersecurity-Specific Metrics ---
    cyber_metrics = _calculate_cybersecurity_metrics(y_test, y_pred, class_names)

    # --- Visualizations ---
    viz_paths = _create_visualizations(
        y_test, y_pred, y_pred_proba, cm, class_names, output_dir, model_config, class_loss_data
    )

    # --- Final Report ---
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'basic_metrics': {
            'accuracy': float(accuracy),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist() if support is not None else []
        },
        'cybersecurity_metrics': cyber_metrics,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'visualizations': viz_paths
    }

    # Save the detailed report in the parent directory (alongside the 'visualizations' folder)
    parent_dir = os.path.dirname(output_dir)
    report_path = os.path.join(parent_dir, "evaluation_report.json")
    os.makedirs(parent_dir, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  💾 JSON report saved to: {report_path}")
    print(f"  🖼️  {len(viz_paths)} visualization(s) saved in {output_dir}")

    return report

def _calculate_cybersecurity_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Calculates metrics specifically relevant to intrusion detection systems.
    This involves treating the problem as a binary one: BENIGN vs. ATTACK.
    """
    # Identify the index for the 'BENIGN' class. It's crucial for binary metrics.
    try:
        benign_idx = class_names.index('BENIGN')
    except ValueError:
        benign_idx = 0 # Fallback if 'BENIGN' is not in the list
        print(f"  ⚠️ 'BENIGN' class not found in class_names. Assuming index {benign_idx} is the negative class.")

    # Binarize labels: 0 for BENIGN, 1 for ATTACK
    y_true_binary = (y_true != benign_idx).astype(int)
    y_pred_binary = (y_pred != benign_idx).astype(int)

    # Calculate binary confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    # Key cybersecurity metrics
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall for attacks
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate (FPR)

    # Analyze detection rate for each specific type of attack
    attack_analysis = {}
    for i, class_name in enumerate(class_names):
        if i != benign_idx:  # Only evaluate attack classes
            class_mask = (y_true == i)
            if np.any(class_mask):
                class_tp = np.sum((y_true == i) & (y_pred == i))
                class_total = np.sum(class_mask)
                detection_rate_class = class_tp / class_total
                attack_analysis[class_name] = {
                    'samples': int(class_total),
                    'detected': int(class_tp),
                    'detection_rate': float(detection_rate_class)
                }

    return {
        'detection_rate': float(detection_rate),
        'false_alarm_rate': float(false_alarm_rate),
        'binary_confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'attack_type_analysis': attack_analysis
    }

def _create_visualizations(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray,
    cm: np.ndarray, class_names: List[str], output_dir: str,
    model_config: Dict, class_loss_data: Optional[Dict]
) -> List[str]:
    """Creates and saves all visualization PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    hyperparams_str = _format_hyperparameters_for_title(model_config)
    viz_paths = []

    # 1. Detailed Confusion Matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    title = 'Detailed Confusion Matrix\n(Rows=True, Columns=Predicted)'
    if hyperparams_str: title += f'\n{hyperparams_str}'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix_detailed.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    viz_paths.append(cm_path)

    # 2. Binary Confusion Matrix (BENIGN vs. ATTACK)
    try:
        benign_idx = class_names.index('BENIGN')
    except ValueError:
        benign_idx = 0
    y_true_binary = (y_true != benign_idx).astype(int)
    y_pred_binary = (y_pred != benign_idx).astype(int)
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Reds', xticklabels=['BENIGN', 'ATTACK'], yticklabels=['BENIGN', 'ATTACK'])
    title_binary = 'Cybersecurity Confusion Matrix\n(BENIGN vs. ATTACK)'
    if hyperparams_str: title_binary += f'\n{hyperparams_str}'
    plt.title(title_binary, fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_binary_path = os.path.join(output_dir, "confusion_matrix_cybersecurity.png")
    plt.savefig(cm_binary_path, dpi=300, bbox_inches='tight')
    plt.close()
    viz_paths.append(cm_binary_path)

    # 3. Per-Class Accuracy Bar Chart
    class_accuracies = [np.mean(y_pred[y_true == i] == y_true[y_true == i]) if np.any(y_true == i) else 0.0 for i in range(len(class_names))]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), class_accuracies, color=['green' if i == benign_idx else 'red' for i in range(len(class_names))])
    title_acc = 'Accuracy per Traffic Class'
    if hyperparams_str: title_acc += f'\n{hyperparams_str}'
    plt.title(title_acc, fontsize=14, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy_per_class.png")
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close()
    viz_paths.append(acc_path)

    # 4. ROC Curves for multiclass
    if len(class_names) > 2 and y_pred_proba.shape[1] > 2:
        try:
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
            plt.figure(figsize=(10, 8))
            for i in range(y_true_bin.shape[1]):
                if np.any(y_true == i) and len(np.unique(y_true_bin[:, i])) > 1:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)', fontweight='bold')
            plt.ylabel('True Positive Rate (TPR)', fontweight='bold')
            plt.title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold')
            plt.legend(loc='lower right')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            roc_path = os.path.join(output_dir, "roc_curves.png")
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(roc_path)
        except Exception as e:
            print(f"  ⚠️ Could not generate ROC curves: {e}")

    # 5. Per-class Cross-Entropy Loss vs Epochs (aggregated across folds if provided)
    if class_loss_data and isinstance(class_loss_data, dict) and 'per_class' in class_loss_data:
        try:
            epochs = class_loss_data.get('epochs', [])
            per_class = class_loss_data.get('per_class', {})  # keys are class indices as strings
            num_classes = len(per_class)
            if num_classes > 0 and len(epochs) > 0:
                # Grid layout for subplots
                cols = min(4, num_classes)
                rows = int(np.ceil(num_classes / cols))
                plt.figure(figsize=(4*cols, 3*rows))
                for idx_str, losses in per_class.items():
                    idx = int(idx_str)
                    subplot_index = idx + 1
                    plt.subplot(rows, cols, subplot_index)
                    y_vals = np.array(losses, dtype=float)
                    # Trunca/padding to match epochs length
                    if len(y_vals) > len(epochs):
                        y_vals = y_vals[:len(epochs)]
                    elif len(y_vals) < len(epochs):
                        pad = np.full(len(epochs) - len(y_vals), np.nan)
                        y_vals = np.concatenate([y_vals, pad])
                    plt.plot(epochs, y_vals, marker='o', linewidth=1.5)
                    plt.title(class_names[idx] if idx < len(class_names) else f"Class {idx}", fontsize=10)
                    plt.xlabel('Epoch')
                    plt.ylabel('CE Loss')
                    plt.grid(alpha=0.3)
                plt.tight_layout()
                loss_grid_path = os.path.join(output_dir, 'per_class_loss_vs_epochs.png')
                plt.savefig(loss_grid_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths.append(loss_grid_path)
        except Exception as e:
            print(f"  ⚠️ Could not generate per-class loss plot: {e}")

    return viz_paths

def _format_hyperparameters_for_title(model_config: Dict = None) -> str:
    """Formats a selection of hyperparameters for use in plot titles."""
    if not model_config: return ""

    hyperparams = model_config.get('hyperparameters', {})
    model_type = model_config.get('model_type', 'Unknown')

    params_str = [f"Model: {model_type.upper()}"]

    def get_param_val(key):
        val = hyperparams.get(key)
        return val[0] if isinstance(val, list) else val

    if 'epochs' in hyperparams: params_str.append(f"Epochs: {get_param_val('epochs')}")
    if 'batch_size' in hyperparams: params_str.append(f"Batch: {get_param_val('batch_size')}")
    if 'learning_rate' in hyperparams: params_str.append(f"LR: {get_param_val('learning_rate')}")
    if 'activation' in hyperparams: params_str.append(f"Act: {get_param_val('activation')}")

    return " | ".join(params_str)
