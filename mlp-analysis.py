#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP Analysis Benchmark for SNN-IDS

This script performs a detailed analysis of a 4-hidden-layer MLP architecture.
It runs a hyperparameter search with a fixed 30-epoch training cycle,
tracking per-class loss at each epoch.

The script will:
1.  Define a search space for hyperparameters (learning rate, batch size, etc.).
2.  Train the 4-layer MLP for each hyperparameter combination for 30 epochs.
3.  Track and store the cross-entropy loss for each class at every epoch.
4.  Evaluate the models using standard metrics (Accuracy, F1, Recall).
5.  Generate visualizations, including confusion matrices and a new plot
    showing loss vs. epochs for each class.
6.  Produce a consolidated report with the best configuration and all results.
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
from sklearn.model_selection import train_test_split

# Import existing modules
sys.path.append(os.path.abspath('.'))
from preprocessing.process import preprocess_pipeline
from training.train import train_model
from evaluation.metrics import evaluate_model_comprehensive
from config import TRAINING_CONFIG, PREPROCESSING_CONFIG, DATA_CONFIG

class MLPAnalysisBenchmark:
    """
    Orchestrator for the 4-layer MLP analysis and hyperparameter search.
    """
    def __init__(self, sample_size: int = None, data_path: str = None):
        """Initializes the benchmark."""
        self.sample_size = sample_size or PREPROCESSING_CONFIG['sample_size']
        self.data_path = data_path or DATA_CONFIG['dataset_path']
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"mlp_analysis_results_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        self._cached_data = None

        print("="*70)
        print("🔬 MLP ANALYSIS BENCHMARK INITIALIZED 🔬")
        print("="*70)
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"💾 Sample Size: {self.sample_size}")
        print("FIXED PARAMETERS:")
        print(f"  - Architecture: 4-Hidden-Layer MLP")
        print(f"  - Epochs: 30")
        print("="*70)

    def _load_and_cache_data(self):
        """Loads and preprocesses data if not already cached."""
        if self._cached_data is None:
            print("\n🧪 Preprocessing data for the first time...")
            try:
                X, y, label_encoder = preprocess_pipeline(
                    data_path=self.data_path,
                    sample_size=self.sample_size
                )
                self._cached_data = (X, y, label_encoder)
                print("✅ Data preprocessed and cached successfully.")
            except Exception as e:
                print(f"❌ Fatal error during preprocessing: {e}")
                raise
        else:
            print("\n🧪 Using cached preprocessed data.")
        return self._cached_data

    def run_analysis(self) -> Dict:
        """
        Executes the full analysis pipeline:
        1. Hyperparameter search.
        2. Identification of the best model.
        3. Generation of a final, comprehensive report.
        """
        print("\n🚀 STARTING HYPERPARAMETER SEARCH...")
        start_time = time.time()

        # Load data
        X, y, label_encoder = self._load_and_cache_data()
        class_names = label_encoder.classes_.tolist()

        # Define the hyperparameter grid for the search
        hyperparam_grid = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [64, 128],
            'activation': ['relu', 'tanh'],
            'hidden_layer_units': [[128, 64, 32, 16], [256, 128, 64, 32]],
            'dropout': [0.2, 0.3]
        }

        keys, values = zip(*hyperparam_grid.items())
        hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        print(f"🔍 Found {len(hyperparam_combinations)} hyperparameter combinations to test.")

        all_results = []
        for i, params in enumerate(hyperparam_combinations):
            run_id = f"run_{i+1}_{int(time.time())}"
            print(f"\n--- Running Test {i+1}/{len(hyperparam_combinations)} (ID: {run_id}) ---")
            print(f"Parameters: {params}")

            result = self._run_single_test(X, y, label_encoder, params, run_id)
            all_results.append(result)

        total_time = time.time() - start_time
        print(f"\n✅ Hyperparameter search finished in {total_time:.2f} seconds.")

        # Identify best configuration
        successful_runs = [r for r in all_results if r.get('status') == 'success']
        if not successful_runs:
            print("❌ No successful runs completed. Cannot generate report.")
            return {"status": "failed", "reason": "No successful training runs."}

        best_run = max(successful_runs, key=lambda r: r.get('accuracy', 0))
        print(f"\n🏆 Best configuration found (Accuracy: {best_run['accuracy']:.4f}):")
        print(json.dumps(best_run['config'], indent=2))

        # Generate and save the final report
        final_report = self._generate_final_report(all_results, best_run, total_time)
        self._save_report(final_report)

        print(f"\n🎉 MLP Analysis Benchmark Complete!")
        print(f"📁 All results have been saved in: {self.output_dir}")

        return final_report

    def _run_single_test(self, X: np.ndarray, y: np.ndarray, label_encoder: Any, hyperparams: Dict, run_id: str) -> Dict:
        """
        Runs a single training and evaluation cycle for one set of hyperparameters.
        """
        test_start_time = time.time()

        # Add fixed parameters
        full_hyperparams = {**hyperparams, 'epochs': 30}

        # We need a separate test set for the final evaluation that the model has never seen.
        # The `train_model` function with `train_test_split` strategy already does this internally,
        # but we need an independent set here for the final comprehensive report.
        X_train_main, X_test_final, y_train_main, y_test_final = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        try:
            # Train the model
            model, _, _, per_class_losses = train_model(
                X=X_train_main,
                y=y_train_main,
                model_type='mlp_4_layer',
                validation_strategy='train_test_split', # Use a split within training for quick validation
                hyperparams=full_hyperparams,
                track_class_loss=True
            )

            # Create a dedicated output directory for this run's artifacts
            run_output_dir = os.path.join(self.output_dir, run_id)
            viz_output_dir = os.path.join(run_output_dir, "visualizations")
            os.makedirs(viz_output_dir, exist_ok=True)

            # Run comprehensive evaluation on the final, held-out test set
            evaluation_report = evaluate_model_comprehensive(
                model=model,
                X_test=X_test_final,
                y_test=y_test_final,
                class_names=label_encoder.classes_.tolist(),
                output_dir=viz_output_dir,
                model_config={'model_type': 'mlp_4_layer', 'hyperparameters': full_hyperparams},
                class_loss_data=per_class_losses
            )

            test_total_time = time.time() - test_start_time

            return {
                'run_id': run_id,
                'status': 'success',
                'total_time': test_total_time,
                'config': full_hyperparams,
                'accuracy': evaluation_report['basic_metrics']['accuracy'],
                'evaluation_report': evaluation_report,
                'per_class_loss_history': per_class_losses
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            test_total_time = time.time() - test_start_time
            return {
                'run_id': run_id,
                'status': 'error',
                'total_time': test_total_time,
                'config': full_hyperparams,
                'error_message': str(e)
            }

    def _generate_final_report(self, all_results: List[Dict], best_run: Dict, total_time: float) -> Dict:
        """Generates a comprehensive final report."""
        successful_runs = [r for r in all_results if r.get('status') == 'success']

        return {
            'benchmark_info': {
                'timestamp': self.timestamp,
                'total_execution_time_seconds': total_time,
                'total_configs_tested': len(all_results),
                'successful_configs': len(successful_runs),
                'sample_size': self.sample_size,
                'architecture': 'mlp_4_layer',
                'fixed_epochs': 30
            },
            'best_configuration': {
                'run_id': best_run['run_id'],
                'accuracy': best_run['accuracy'],
                'config': best_run['config'],
                'evaluation_summary': {
                    'f1_per_class': best_run['evaluation_report']['basic_metrics']['f1_per_class'],
                    'recall_per_class': best_run['evaluation_report']['basic_metrics']['recall_per_class'],
                    'detection_rate': best_run['evaluation_report']['cybersecurity_metrics']['detection_rate'],
                    'false_alarm_rate': best_run['evaluation_report']['cybersecurity_metrics']['false_alarm_rate']
                },
                'visualization_paths': best_run['evaluation_report']['visualizations']
            },
            'all_run_details': all_results
        }

    def _save_report(self, report: Dict):
        """Saves the final report to JSON, CSV, and TXT files."""
        # 1. Full JSON dump
        json_path = os.path.join(self.output_dir, 'mlp_analysis_complete_report.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n💾 Full JSON report saved to: {json_path}")

        # 2. Summary CSV
        try:
            csv_data = []
            for run in report['all_run_details']:
                row = {'run_id': run['run_id'], 'status': run['status']}
                row.update(run['config'])
                if run['status'] == 'success':
                    row['accuracy'] = run['accuracy']
                    row['detection_rate'] = run['evaluation_report']['cybersecurity_metrics']['detection_rate']
                    row['false_alarm_rate'] = run['evaluation_report']['cybersecurity_metrics']['false_alarm_rate']
                else:
                    row['accuracy'] = 0
                    row['detection_rate'] = 0
                    row['false_alarm_rate'] = 0
                csv_data.append(row)

            df = pd.DataFrame(csv_data)
            # Reorder columns to be more readable
            cols = ['run_id', 'status', 'accuracy', 'learning_rate', 'batch_size', 'activation', 'dropout', 'hidden_layer_units', 'detection_rate', 'false_alarm_rate']
            df_cols = [c for c in cols if c in df.columns]
            df = df[df_cols]

            csv_path = os.path.join(self.output_dir, 'mlp_analysis_summary.csv')
            df.to_csv(csv_path, index=False)
            print(f"📊 CSV summary saved to: {csv_path}")

        except Exception as e:
            print(f"⚠️ Could not save CSV summary: {e}")

        # 3. Human-readable TXT report
        txt_path = os.path.join(self.output_dir, 'mlp_analysis_summary.txt')
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("🔬 MLP ANALYSIS BENCHMARK SUMMARY 🔬\n")
            f.write("="*80 + "\n\n")

            info = report['benchmark_info']
            f.write(f"Timestamp: {info['timestamp']}\n")
            f.write(f"Total Runtime: {info['total_execution_time_seconds']:.2f} seconds\n")
            f.write(f"Sample Size: {info['sample_size']}\n")
            f.write(f"Configs Tested: {info['total_configs_tested']} ({info['successful_configs']} successful)\n\n")

            f.write("-" * 80 + "\n")
            f.write("🏆 BEST CONFIGURATION 🏆\n")
            f.write("-" * 80 + "\n\n")

            best = report['best_configuration']
            f.write(f"Run ID: {best['run_id']}\n")
            f.write(f"Accuracy: {best['accuracy']:.6f}\n\n")
            f.write("Hyperparameters:\n")
            f.write(json.dumps(best['config'], indent=2))
            f.write("\n\n")
            f.write("Key Metrics:\n")
            f.write(f"  - Detection Rate: {best['evaluation_summary']['detection_rate']:.4f}\n")
            f.write(f"  - False Alarm Rate: {best['evaluation_summary']['false_alarm_rate']:.4f}\n\n")
            f.write("Visualizations for this run are located in:\n")
            f.write(f"  - {os.path.join(self.output_dir, best['run_id'], 'visualizations')}\n")

        print(f"📄 TXT summary saved to: {txt_path}")


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
  python3 mlp-analysis.py --sample-size 50000
        '''
    )
    parser.add_argument('--sample-size', type=int, help='Number of data samples to use.')
    parser.add_argument('--data-path', type=str, help='Path to the dataset directory.')

    args = parser.parse_args()

    try:
        analysis_benchmark = MLPAnalysisBenchmark(
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
