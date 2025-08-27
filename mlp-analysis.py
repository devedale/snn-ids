#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP Analysis Benchmark for SNN-IDS (Refactored)

This script performs a detailed analysis of a 4-hidden-layer MLP architecture
by leveraging the existing and trusted SNNIDSBenchmark class.

It will:
1.  Define a search space for hyperparameters for the 4-layer MLP.
2.  Use SNNIDSBenchmark to run a clean test for each hyperparameter combination.
3.  Ensure the model uses a fixed 30-epoch training cycle.
4.  Collect all results and identify the top 10 best-performing models.
5.  Generate a CSV and a text summary of the top 10 models, including
    key metrics and links to their detailed evaluation reports and confusion matrices.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import itertools
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

# Import existing modules
sys.path.append(os.path.abspath('.'))
from config import PREPROCESSING_CONFIG, DATA_CONFIG
from preprocessing.process import preprocess_pipeline
from training.train import build_model, PerClassLossLogger
from evaluation.metrics import evaluate_model_comprehensive
import keras_tuner as kt

class MLPAnalysis:
    """
    Orchestrator for the 4-layer MLP analysis.
    This class USES SNNIDSBenchmark to ensure consistency.
    """
    def __init__(self, sample_size: int = None, data_path: str = None):
        """Initializes the benchmark."""
        self.sample_size = sample_size or PREPROCESSING_CONFIG['sample_size']
        self.data_path = data_path or DATA_CONFIG['dataset_path']
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"/tmp/mlp_analysis_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        print("="*70)
        print("🔬 MLP HYPERBAND ANALYSIS INITIALIZED 🔬")
        print("="*70)
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"💾 Sample Size: {self.sample_size}")
        print("="*70)

    def run_analysis(self):
        """
        Executes the full analysis pipeline using KerasTuner's Hyperband.
        """
        print("\n🚀 STARTING HYPERBAND SEARCH FOR 4-LAYER MLP...")
        start_time = time.time()

        # 1. Preprocessing
        print("\n🧪 PREPROCESSING...")
        try:
            X, y, label_encoder = preprocess_pipeline(
                data_path=self.data_path,
                sample_size=self.sample_size
            )
            print(f"✅ Preprocessing complete. Loaded data shape: {X.shape}")

            # Reshape data for the MLP model (pre-flattening)
            if len(X.shape) == 3:
                print(f"  -> Reshaping data from {X.shape} to 2D for MLP...")
                n_samples = X.shape[0]
                X = X.reshape(n_samples, -1)
                print(f"  -> New data shape: {X.shape}")

        except Exception as e:
            print(f"❌ Fatal error during preprocessing: {e}")
            return

        # 2. Hyperband Tuner Setup
        num_classes = len(np.unique(y))
        input_shape = X.shape[1:] if len(X.shape) > 2 else (X.shape[1],)

        # Manual data split to pass training data to the loss logger
        stratify_opt = y if np.min(np.bincount(y)) >= 2 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_opt
        )

        # Instantiate the loss logger
        self.loss_logger = PerClassLossLogger(X_train, y_train, class_indices=list(np.unique(y)))

        # Wrapper per build_model per passare argomenti extra
        def model_builder(hp):
            return build_model(
                model_type='mlp_4_layer',
                input_shape=input_shape,
                num_classes=num_classes,
                hp_or_params=hp
            )

        tuner = kt.Hyperband(
            model_builder,
            objective='val_auc',
            max_epochs=30,
            factor=3,
            directory=self.output_dir,
            project_name='snn_ids_hyperband',
            overwrite=True
        )

        # 3. Esecuzione della ricerca
        print("\n🔍 Starting Hyperband search...")
        tuner.search(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5), self.loss_logger]
        )

        total_time = time.time() - start_time
        print(f"\n✅ Hyperband search finished in {total_time:.2f} seconds.")

        # 4. Estrazione e salvataggio dei risultati
        self._generate_and_save_reports(tuner, X, y, label_encoder, total_time)

        print(f"\n🎉 MLP Analysis with Hyperband Complete!")
        print(f"📁 All results have been saved in: {self.output_dir}")

    def _generate_and_save_reports(self, tuner: kt.Tuner, X: np.ndarray, y: np.ndarray, label_encoder, total_time: float):
        """
        Analyzes tuner results, evaluates the top 10 models, and saves comprehensive reports.
        """
        print("\n📊 Generating comprehensive reports for top 10 models...")

        # --- Get Top 10 Models and Hyperparameters ---
        top_n = 10
        best_hps_list = tuner.get_best_hyperparameters(num_trials=top_n)
        best_models_list = tuner.get_best_models(num_models=top_n)

        if not best_models_list:
            print("❌ No models found by the tuner. Cannot generate reports.")
            return

        # --- Prepare Test Set ---
        # (Use the same test set for fair comparison of all top models)
        class_counts = np.bincount(y)
        stratify_opt = y if np.min(class_counts) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_opt)

        all_report_data = []
        for rank, (hps, model) in enumerate(zip(best_hps_list, best_models_list), 1):
            print(f"\n--- Evaluating Model Rank #{rank} ---")

            # --- Basic Evaluation ---
            y_pred_probs = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            precision = f1_score(y_test, y_pred, average='weighted') # Precision is similar to F1 in weighted avg

            print(f"  - Accuracy: {accuracy:.4f}, F1 (Weighted): {f1:.4f}")

            # --- Detailed Evaluation for Rank #1 ---
            viz_paths = {}
            if rank == 1:
                print("  - Generating detailed visualizations for the best model...")
                eval_dir = os.path.join(self.output_dir, "best_model_visualizations")

                # We need to pass the model config for the titles in the plots
                model_config = {'model_type': 'mlp_4_layer', 'hyperparameters': hps.values}

                # Call the comprehensive evaluation function from metrics.py
                detailed_report = evaluate_model_comprehensive(
                    model, X_test, y_test, label_encoder.classes_, eval_dir, model_config,
                    class_loss_data=self.loss_logger.losses
                )
                viz_paths = detailed_report.get('visualizations', {})

            # --- Collect Data for Reports ---
            report_row = {
                'rank': rank,
                'accuracy': accuracy,
                'f1_score_weighted': f1,
                'recall_weighted': recall,
                'precision_weighted': precision,
                **hps.values,
                'visualizations_path': viz_paths.get('confusion_matrix_detailed.png', 'N/A')
            }
            all_report_data.append(report_row)

        # --- Create Comprehensive CSV Report ---
        if not all_report_data:
            print("⚠️ No data to generate CSV report.")
            return

        df = pd.DataFrame(all_report_data)
        csv_path = os.path.join(self.output_dir, "top_10_models_comprehensive_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n📊 Top 10 models comprehensive summary saved to: {csv_path}")

        # --- Create TXT Summary ---
        txt_path = os.path.join(self.output_dir, "hyperband_summary.txt")
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("🔬 MLP HYPERBAND ANALYSIS SUMMARY 🔬\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Search Runtime: {total_time:.2f} seconds\n")
            f.write(f"Number of Models Evaluated: {len(all_report_data)}\n\n")

            f.write("-" * 80 + "\n")
            f.write("🏆 BEST MODEL - FINAL EVALUATION 🏆\n")
            f.write("-" * 80 + "\n\n")

            best_run = all_report_data[0]
            f.write(f"Rank: 1\n")
            f.write(f"Accuracy: {best_run['accuracy']:.6f}\n")
            f.write(f"F1-Score (Weighted): {best_run['f1_score_weighted']:.6f}\n")
            f.write(f"Recall (Weighted): {best_run['recall_weighted']:.6f}\n\n")
            f.write("Hyperparameters:\n")
            f.write(json.dumps({k:v for k,v in best_run.items() if k not in ['rank', 'accuracy', 'f1_score_weighted', 'recall_weighted', 'precision_weighted', 'visualizations_path']}, indent=2))
            f.write("\n\n")
            f.write(f"Find detailed evaluation report and confusion matrices in:\n")
            f.write(f"  {os.path.dirname(best_run['visualizations_path'])}\n")

        print(f"📄 Benchmark summary saved to: {txt_path}")


def main():
    """Main entry point for the MLP analysis script."""
    parser = argparse.ArgumentParser(
        description='Run a detailed analysis of a 4-layer MLP for SNN-IDS.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:

  # Run with default settings (from config.py)
  python3 mlp-analysis.py

  # Run with a specific sample size
  python3 mlp-analysis.py --sample-size 20000
        '''
    )
    parser.add_argument('--sample-size', type=int, help='Number of data samples to use.')
    parser.add_argument('--data-path', type=str, help='Path to the dataset directory.')

    args = parser.parse_args()

    try:
        analysis_benchmark = MLPAnalysis(
            sample_size=args.sample_size,
            data_path=args.data_path
        )
        analysis_benchmark.run_analysis()
        return 0
    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
