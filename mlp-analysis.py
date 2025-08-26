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

# Import existing modules
sys.path.append(os.path.abspath('.'))
from benchmark import SNNIDSBenchmark
from config import PREPROCESSING_CONFIG, DATA_CONFIG

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
        self.output_dir = f"mlp_analysis_results_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        # The core runner is the trusted benchmark class
        self.runner = SNNIDSBenchmark(config_override={
            'sample_size': self.sample_size,
            'data_path': self.data_path
        })

        print("="*70)
        print("🔬 MLP ANALYSIS BENCHMARK INITIALIZED (Refactored) 🔬")
        print("="*70)
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"💾 Sample Size: {self.sample_size}")
        print(f"✅ Using SNNIDSBenchmark for core logic.")
        print("="*70)

    def run_analysis(self):
        """
        Executes the full analysis pipeline.
        """
        print("\n🚀 STARTING HYPERPARAMETER SEARCH FOR 4-LAYER MLP...")
        start_time = time.time()

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
            print(f"\n--- Running Test {i+1}/{len(hyperparam_combinations)} ---")

            test_config = {
                'model_type': 'mlp_4_layer',
                'hyperparameters': {**params, 'epochs': 30},
                'output_dir': self.output_dir # Pass the main output dir to the runner
            }

            # Use the trusted runner
            result = self.runner._run_single_configuration(test_config)
            all_results.append(result)

        total_time = time.time() - start_time
        print(f"\n✅ Hyperparameter search finished in {total_time:.2f} seconds.")

        # Identify best configurations and save reports
        self._generate_and_save_reports(all_results, total_time)

        print(f"\n🎉 MLP Analysis Benchmark Complete!")
        print(f"📁 All results have been saved in: {self.output_dir}")

    def _generate_and_save_reports(self, all_results: List[Dict], total_time: float):
        """Analyzes all results, finds the top 10, and saves reports."""
        successful_runs = [r for r in all_results if r.get('status') == 'success']
        if not successful_runs:
            print("❌ No successful runs completed. Cannot generate reports.")
            return

        # Sort by accuracy (descending)
        sorted_runs = sorted(successful_runs, key=lambda r: r.get('best_accuracy', 0), reverse=True)

        top_10_runs = sorted_runs[:10]

        print(f"\n🏆 Top {len(top_10_runs)} Models Found 🏆")

        # --- Create CSV Report ---
        report_data = []
        for rank, run in enumerate(top_10_runs, 1):
            config = run.get('config', {})
            hyperparams = config.get('hyperparameters', {})
            eval_report = run.get('evaluation', {}).get('report', {})
            basic_metrics = eval_report.get('basic_metrics', {})

            # Calculate mean F1 and Recall
            mean_f1 = np.mean(basic_metrics.get('f1_per_class', [0]))
            mean_recall = np.mean(basic_metrics.get('recall_per_class', [0]))

            row = {
                'rank': rank,
                'run_id': run.get('run_id'),
                'accuracy': run.get('best_accuracy', 0),
                'mean_f1_score': mean_f1,
                'mean_recall': mean_recall,
                'learning_rate': hyperparams.get('learning_rate'),
                'batch_size': hyperparams.get('batch_size'),
                'activation': hyperparams.get('activation'),
                'hidden_layer_units': str(hyperparams.get('hidden_layer_units')),
                'dropout': hyperparams.get('dropout'),
                'training_time_s': run.get('training_time'),
                'confusion_matrix_path': os.path.join(run.get('evaluation', {}).get('evaluation_dir', ''), 'confusion_matrix_detailed.png')
            }
            report_data.append(row)

        if not report_data:
            print("⚠️ No data to generate CSV report.")
            return

        df = pd.DataFrame(report_data)
        csv_path = os.path.join(self.output_dir, "top_10_models_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"📊 Top 10 models summary saved to: {csv_path}")

        # --- Create TXT Summary ---
        txt_path = os.path.join(self.output_dir, "benchmark_summary.txt")
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("🔬 MLP ANALYSIS BENCHMARK SUMMARY 🔬\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Runtime: {total_time:.2f} seconds\n")
            f.write(f"Total Configurations Tested: {len(all_results)}\n")
            f.write(f"Successful Runs: {len(successful_runs)}\n\n")

            f.write("-" * 80 + "\n")
            f.write("🏆 BEST CONFIGURATION 🏆\n")
            f.write("-" * 80 + "\n\n")

            best_run = top_10_runs[0]
            f.write(f"Rank: 1\n")
            f.write(f"Run ID: {best_run.get('run_id')}\n")
            f.write(f"Accuracy: {best_run.get('best_accuracy', 0):.6f}\n")
            f.write(f"Mean F1-Score: {report_data[0]['mean_f1_score']:.6f}\n")
            f.write(f"Mean Recall: {report_data[0]['mean_recall']:.6f}\n\n")
            f.write("Hyperparameters:\n")
            f.write(json.dumps(best_run.get('config', {}).get('hyperparameters', {}), indent=2))
            f.write("\n\n")
            f.write(f"Find detailed evaluation report and confusion matrix in:\n")
            f.write(f"  {os.path.dirname(report_data[0]['confusion_matrix_path'])}\n")

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
