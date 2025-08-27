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
from training.train import build_model
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
        except Exception as e:
            print(f"❌ Fatal error during preprocessing: {e}")
            return

        # 2. Hyperband Tuner Setup
        num_classes = len(np.unique(y))
        input_shape = X.shape[1:] if len(X.shape) > 2 else (X.shape[1],)

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
            objective='val_accuracy',
            max_epochs=30,
            factor=3,
            directory=self.output_dir,
            project_name='snn_ids_hyperband',
            overwrite=True
        )

        # 3. Esecuzione della ricerca
        print("\n🔍 Starting Hyperband search...")
        tuner.search(X, y, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

        total_time = time.time() - start_time
        print(f"\n✅ Hyperband search finished in {total_time:.2f} seconds.")

        # 4. Estrazione e salvataggio dei risultati
        self._generate_and_save_reports(tuner, X, y, total_time)

        print(f"\n🎉 MLP Analysis with Hyperband Complete!")
        print(f"📁 All results have been saved in: {self.output_dir}")

    def _generate_and_save_reports(self, tuner: kt.Tuner, X: np.ndarray, y: np.ndarray, total_time: float):
        """Analyzes tuner results, evaluates the best model, and saves reports."""
        print("\n📊 Generating and saving reports...")

        # --- Get Best Hyperparameters and Model ---
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models(num_models=1)[0]

        if not best_model:
            print("❌ No best model found by the tuner. Cannot generate reports.")
            return

        # --- Evaluate Best Model on a Hold-out Test Set ---
        print("\n🔬 Evaluating the best model on a hold-out test set...")

        # Check if stratification is possible, otherwise split without it
        class_counts = np.bincount(y)
        if np.min(class_counts) < 2:
            print("  ⚠️  Disabling stratification for final evaluation due to rare classes (count < 2).")
            stratify_opt = None
        else:
            stratify_opt = y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_opt)

        # Re-train the model on the full training set for final evaluation
        # This ensures the model is trained on as much data as possible
        final_model = tuner.hypermodel.build(best_hps)
        final_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=0)

        y_pred_probs = final_model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        print(f"  - Final Test Accuracy: {accuracy:.4f}")
        print(f"  - Final Test F1-Score (Weighted): {f1:.4f}")
        print(f"  - Final Test Recall (Weighted): {recall:.4f}")

        # --- Create TXT Summary ---
        txt_path = os.path.join(self.output_dir, "hyperband_summary.txt")
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("🔬 MLP HYPERBAND ANALYSIS SUMMARY 🔬\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Search Runtime: {total_time:.2f} seconds\n\n")

            f.write("-" * 80 + "\n")
            f.write("🏆 BEST MODEL - FINAL EVALUATION 🏆\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"Accuracy: {accuracy:.6f}\n")
            f.write(f"F1-Score (Weighted): {f1:.6f}\n")
            f.write(f"Recall (Weighted): {recall:.6f}\n\n")

            f.write("Hyperparameters:\n")
            f.write(json.dumps(best_hps.values, indent=2))
            f.write("\n\n")

            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\n")

        print(f"📄 Benchmark summary saved to: {txt_path}")

        # --- Create CSV with Top 10 Trials ---
        trials = tuner.oracle.get_best_trials(num_trials=10)
        report_data = []
        for rank, trial in enumerate(trials, 1):
            row = {
                'rank': rank,
                'trial_id': trial.trial_id,
                'score': trial.score,
                **trial.hyperparameters.values
            }
            report_data.append(row)

        if not report_data:
            print("⚠️ No trial data to generate CSV report.")
            return

        df = pd.DataFrame(report_data)
        csv_path = os.path.join(self.output_dir, "top_10_hyperband_trials.csv")
        df.to_csv(csv_path, index=False)
        print(f"📊 Top 10 trials summary saved to: {csv_path}")


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
