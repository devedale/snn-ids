#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNN-IDS Benchmark Orchestrator
This script is the main entry point for running all tests and benchmarks for the system.
It has been refactored to be a lightweight orchestrator that delegates the heavy
lifting (preprocessing, training, evaluation) to the modules in the `src/` directory.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import itertools

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the refactored components
from config import PREPROCESSING_CONFIG, DATA_CONFIG, TRAINING_CONFIG
from src.preprocessing import preprocess_pipeline
from src.training.trainer import ModelTrainer
from src.training.models import get_model_builder
from src.evaluation import evaluate_model_comprehensive
from src.utils import zip_artifacts

# KerasTuner is an optional dependency for hyperband
try:
    import keras_tuner as kt
    import tensorflow as tf
except ImportError:
    kt = None
    tf = None

class SNNIDSBenchmark:
    """A modular and unified benchmark for the SNN-IDS."""

    def __init__(self, config_override: Dict = None):
        """Initializes the benchmark with an optional configuration override."""
        self.config_override = config_override or {}
        self.timestamp = datetime.now().isoformat()
        # This cache holds the preprocessed data (X, y, label_encoder) to avoid
        # reloading it for each run in a single benchmark execution.
        self._cached_data = None

    def _get_preprocessed_data(self) -> tuple:
        """
        Retrieves preprocessed data, using an in-memory cache if available,
        or running the preprocessing pipeline if not.
        """
        if self._cached_data is None:
            print("\n🧪 PREPROCESSING (first run)")
            prep_start = time.time()
            try:
                X, y, label_encoder = preprocess_pipeline(
                    data_path=self.config_override.get('data_path'),
                    sample_size=self.config_override.get('sample_size')
                )
                self._cached_data = (X, y, label_encoder)
                prep_time = time.time() - prep_start
                print(f"✅ Preprocessing complete in {prep_time:.2f}s")
            except Exception as e:
                print(f"❌ Fatal error during preprocessing: {e}")
                raise e
        else:
            print("\n🧪 PREPROCESSING (using cached data)")

        return self._cached_data

    def run_single_configuration(self, test_config: Dict) -> Dict:
        """
        Runs a single test cycle (preprocess, train, eval) for a given configuration.
        This is the modular core of the benchmark.
        """
        model_type = test_config['model_type']
        run_id = f"{model_type}_{int(time.time())}"
        print(f"\n\n{'='*20} STARTING TEST: {run_id} {'='*20}")
        print(f"📝 Configuration: {test_config}")

        start_time = time.time()

        try:
            # --- 1. Preprocessing ---
            X, y, label_encoder = self._get_preprocessed_data()

            # --- 2. Training ---
            print("\n🏋️ TRAINING")
            train_start = time.time()

            # Use the refactored ModelTrainer
            trainer = ModelTrainer(model_type=model_type)
            # The trainer first evaluates the model using the validation strategy
            # to get a performance metric.
            validation_accuracy, class_loss_data = trainer.train_and_evaluate(X, y)
            # Then, it trains a final model on the full dataset.
            final_model = trainer.train_final_model(X, y)

            train_time = time.time() - train_start
            print(f"✅ Training complete in {train_time:.2f}s. Validation Accuracy: {validation_accuracy:.4f}")

            # --- 3. Evaluation ---
            print("\n📊 EVALUATION")
            eval_start = time.time()

            # The evaluation now runs on the full dataset split internally
            # to get a final performance assessment.
            descriptive_name = self._generate_descriptive_folder_name(test_config, datetime.now().strftime("%Y%m%d_%H%M%S"))
            eval_dir = os.path.join(test_config.get('output_dir', 'benchmark_results'), descriptive_name, "visualizations")

            # Prepare data for the specific model type before evaluation
            from src.training.utils import prepare_data_for_model
            X_eval = prepare_data_for_model(X, model_type)

            evaluation_report = evaluate_model_comprehensive(
                model=final_model, X_test=X_eval, y_test=y,
                class_names=list(label_encoder.classes_),
                output_dir=eval_dir,
                model_config=test_config,
                class_loss_data=class_loss_data
            )
            eval_time = time.time() - eval_start
            print(f"✅ Evaluation complete in {eval_time:.2f}s.")

            # --- 4. Results ---
            total_time = time.time() - start_time
            result = {
                'run_id': run_id,
                'status': 'success',
                'total_time': total_time,
                'config': test_config,
                'training_time': train_time,
                'validation_accuracy': validation_accuracy,
                'model_path': os.path.join(TRAINING_CONFIG['output_path'], f"{model_type}_final_model.keras"),
                'evaluation': evaluation_report
            }
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {'status': 'error', 'error': str(e), 'config': test_config}

        print(f"\n{'='*20} TEST COMPLETE: {run_id} (Total Time: {result.get('total_time', 0):.2f}s) {'='*20}")
        return result

    def run_hyperband_mlp(self, max_epochs: int, final_epochs: int, batch_size: int) -> Dict[str, Any]:
        """Runs Hyperband tuning for the MLP model, then a final run with the best parameters."""
        if kt is None or tf is None:
            raise ImportError("KerasTuner and TensorFlow are required for Hyperband tuning. Please install them.")

        print("\n🔬 HYPERBAND TUNING - MLP 4-LAYER")
        X, y, label_encoder = self._get_preprocessed_data()

        # MLP requires flattened 2D data
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        from sklearn.model_selection import train_test_split
        stratify_opt = y if (np.bincount(y).min() >= 2) else None
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_opt)

        input_shape = (X_train.shape[1],)
        num_classes = len(np.unique(y))

        # Model builder function for KerasTuner
        def model_builder(hp):
            builder = get_model_builder('mlp_4_layer')
            # Define the search space for hyperparameters
            return builder(
                input_shape=input_shape,
                num_classes=num_classes,
                units_layer_1=hp.Int('units_layer_1', min_value=64, max_value=256, step=32),
                units_layer_2=hp.Int('units_layer_2', min_value=32, max_value=128, step=32),
                units_layer_3=hp.Int('units_layer_3', min_value=16, max_value=64, step=16),
                units_layer_4=hp.Int('units_layer_4', min_value=8, max_value=32, step=8),
                activation=hp.Choice('activation', values=['relu', 'tanh']),
                learning_rate=hp.Choice('learning_rate', values=[0.005, 0.001, 0.0005])
            )

        tuner = kt.Hyperband(
            model_builder,
            objective='val_accuracy',
            max_epochs=max_epochs,
            factor=3,
            directory='benchmark_results',
            project_name=f"hyperband_mlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            overwrite=True
        )

        from sklearn.utils import class_weight
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)],
            class_weight=dict(enumerate(cw))
        )

        best_hps = tuner.get_best_hyperparameters(1)[0]
        print(f"\n🏅 Hyperband finished. Best hyperparameters found: {best_hps.values}")

        # Run a final test with the best hyperparameters
        final_test_config = {
            'model_type': 'mlp_4_layer',
            'hyperparameters': {
                'epochs': final_epochs,
                'batch_size': batch_size,
                'learning_rate': best_hps.get('learning_rate'),
                'activation': best_hps.get('activation'),
                'units_layer_1': best_hps.get('units_layer_1'),
                'units_layer_2': best_hps.get('units_layer_2'),
                'units_layer_3': best_hps.get('units_layer_3'),
                'units_layer_4': best_hps.get('units_layer_4'),
            }
        }
        # We need to update the global config for the ModelTrainer to pick up these HPs
        TRAINING_CONFIG['hyperparameters']['model_specific']['mlp_4_layer'] = final_test_config['hyperparameters']

        result = self.run_single_configuration(final_test_config)
        result['test_type'] = 'hyperband_mlp_final'
        return result

    def _generate_descriptive_folder_name(self, test_config: Dict, timestamp: str) -> str:
        """Generates a descriptive folder name from the test configuration."""
        model_type = test_config.get('model_type', 'unknown')
        # This is now more complex because HPs are nested. We simplify for the folder name.
        parts = [timestamp, model_type.upper()]
        hp = test_config.get('hyperparameters', {})
        if 'epochs' in hp: parts.append(f"ep{hp['epochs'][0] if isinstance(hp['epochs'], list) else hp['epochs']}")
        if 'batch_size' in hp: parts.append(f"bs{hp['batch_size'][0] if isinstance(hp['batch_size'], list) else hp['batch_size']}")
        return "_".join(parts)

    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Saves the benchmark results to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        test_type = results.get('test_type', 'single')
        ts = int(time.time())
        results_file = os.path.join(output_dir, f"benchmark_{test_type}_{ts}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 JSON results saved to: {results_file}")

def main():
    """Main entry point for running the benchmarks."""
    parser = argparse.ArgumentParser(
        description='SNN-IDS Benchmark v3 (Refactored & Modular)',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='''
Examples:
  # 1. Run a quick smoke test to verify basic functionality
  python3 benchmark.py --model dense --smoke-test

  # 2. Run a single test for a specific model and sample size
  python3 benchmark.py --model gru --sample-size 20000

  # 3. Run the Hyperband tuner for MLP, then a final run with the best HPs
  python3 benchmark.py --hyperband-mlp --hb-max-epochs 20 --hb-final-epochs 30
        '''
    )
    
    # Mode selection
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick, lightweight smoke test.')
    parser.add_argument('--hyperband-mlp', action='store_true', help='Run Hyperband tuning for the 4-layer MLP.')
    
    # Basic configuration
    parser.add_argument('--model', type=str, help='Model type to test in a single run (e.g., dense, gru, lstm, mlp_4_layer).')
    parser.add_argument('--sample-size', type=int, help='Total number of samples to use (BENIGN + ATTACK).')
    parser.add_argument('--data-path', type=str, help='Path to the directory containing the dataset CSV files.')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Directory to save results.')

    # Hyperparameter overrides
    parser.add_argument('--epochs', type=int, help="Override the number of training epochs.")
    parser.add_argument('--batch-size', type=int, help="Override the training batch size.")
    parser.add_argument('--learning-rate', type=float, help="Override the optimizer learning rate.")

    # Hyperband-specific arguments
    parser.add_argument('--hb-max-epochs', type=int, default=20, help='Max epochs for a single Hyperband trial.')
    parser.add_argument('--hb-final-epochs', type=int, default=30, help='Epochs for the final run after tuning.')
    parser.add_argument('--hb-batch-size', type=int, default=128, help='Batch size for Hyperband tuning and final run.')
    
    args = parser.parse_args()
    
    # --- Build configuration from args ---
    config_override = {}
    if args.sample_size: config_override['sample_size'] = args.sample_size
    if args.data_path: config_override['data_path'] = args.data_path
    
    # Override hyperparameters in the global config so the trainer can see them
    if args.epochs: TRAINING_CONFIG['hyperparameters']['common']['epochs'] = [args.epochs]
    if args.batch_size: TRAINING_CONFIG['hyperparameters']['common']['batch_size'] = [args.batch_size]
    if args.learning_rate: TRAINING_CONFIG['hyperparameters']['common']['learning_rate'] = [args.learning_rate]

    benchmark = SNNIDSBenchmark(config_override)
    exit_code = 1
    
    try:
        results = None
        # Determine which mode to run
        if args.hyperband_mlp:
            results = benchmark.run_hyperband_mlp(
                max_epochs=args.hb_max_epochs,
                final_epochs=args.hb_final_epochs,
                batch_size=args.hb_batch_size
            )
        elif args.model:
            model_type = args.model.lower()
            if args.smoke_test:
                print("🔥 SMOKE TEST - Verifying basic functionality...")
                # Use smaller epoch count for smoke test
                TRAINING_CONFIG['hyperparameters']['common']['epochs'] = [2]

            test_config = {
                'model_type': model_type,
                'output_dir': args.output_dir,
                # Pass the full hyperparameter config, ModelTrainer will extract the relevant parts
                'hyperparameters': TRAINING_CONFIG['hyperparameters']
            }
            results = benchmark.run_single_configuration(test_config)
            if args.smoke_test:
                results['test_type'] = 'smoke_test'
        else:
            parser.error("You must specify a mode to run. Use --model <name> for a single run or --hyperband-mlp for tuning. Use -h for help.")

        if results:
            benchmark.save_results(results, args.output_dir)
        
        exit_code = 0 if results.get('status') == 'success' else 1

    except Exception as e:
        print(f"\n❌ Critical error during benchmark execution: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        # Always create the ZIP archive at the end
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"benchmark_run_{ts}.zip"
        zip_artifacts([args.output_dir], zip_filename)

    return exit_code

if __name__ == "__main__":
    sys.exit(main())
