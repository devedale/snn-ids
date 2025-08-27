#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark SNN-IDS
Unico entry point per tutti i test e benchmark del sistema.
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
import zipfile

# Import moduli
sys.path.append(os.path.abspath('.'))
from config import *
from preprocessing.process import preprocess_pipeline
from training.train import train_model, build_model
import keras_tuner as kt
from evaluation.metrics import evaluate_model_comprehensive

class SNNIDSBenchmark:
    """Benchmark unificato e modulare per SNN-IDS."""
    
    def __init__(self, config_override: Dict = None):
        """Inizializza benchmark con configurazione di override."""
        self.config_override = config_override or {}
        self.timestamp = datetime.now().isoformat()
        # Cache per dati pre-processati per evitare ricaricamenti
        self._cached_data = None

    def _run_single_configuration(self, test_config: Dict) -> Dict:
        """
        Esegue un singolo ciclo di test (preprocess, train, eval) per una data configurazione.
        Questo è il core modulare del benchmark.
        """
        run_id = f"{test_config['model_type']}_{int(time.time())}"
        print(f"\n\n{'='*20} INIZIO TEST: {run_id} {'='*20}")
        print(f"📝 Configurazione: {test_config}")
        
        start_time = time.time()
        
        # --- 1. Preprocessing ---
        # Il preprocessing viene eseguito solo una volta se i dati non sono già in cache
        if self._cached_data is None:
            print("\n🧪 PREPROCESSING (prima esecuzione)")
            try:
                prep_start = time.time()
                X, y, label_encoder = preprocess_pipeline(
                    data_path=test_config.get('data_path'),
                    sample_size=test_config.get('sample_size')
                )
                self._cached_data = (X, y, label_encoder)
                prep_time = time.time() - prep_start
                print(f"✅ Preprocessing completato in {prep_time:.2f}s")
            except Exception as e:
                print(f"❌ Errore fatale nel preprocessing: {e}")
                return {'status': 'error', 'stage': 'preprocessing', 'error': str(e)}
        else:
            X, y, label_encoder = self._cached_data
            print("\n🧪 PREPROCESSING (dati da cache)")

        # --- 2. Training ---
        print("\n🏋️ TRAINING")
        try:
            train_start = time.time()
            # The train_model function was modified to return per_class_losses, so we handle 4 return values now.
            model, training_log, model_path, per_class_losses = train_model(
                X=X, y=y,
                model_type=test_config['model_type'],
                hyperparams=test_config['hyperparameters'],
                track_class_loss=True # Always track loss for potential analysis
            )
            train_time = time.time() - train_start
            # The new train_model returns a log with a single entry
            best_accuracy = training_log[0]['accuracy'] if training_log else 0
            print(f"✅ Training completato in {train_time:.2f}s. Accuratezza di validazione: {best_accuracy:.4f}")
        except Exception as e:
            print(f"❌ Errore nel training: {e}")
            return {'status': 'error', 'stage': 'training', 'error': str(e), 'config': test_config}

        # --- 3. Valutazione ---
        print("\n📊 VALUTAZIONE")
        try:
            # Pass the per_class_losses to the evaluation function
            eval_result = self._run_evaluation(model, X, y, label_encoder, test_config, per_class_losses)
            print("✅ Valutazione completata.")
        except Exception as e:
            print(f"❌ Errore nella valutazione: {e}")
            eval_result = {'status': 'error', 'error': str(e)}

        # --- 4. Risultati ---
        total_time = time.time() - start_time
        result = {
            'run_id': run_id,
            'status': 'success',
            'total_time': total_time,
            'config': test_config,
            'training_time': train_time,
            'best_accuracy': best_accuracy,
            'model_path': model_path,
            'training_log': training_log,
            'evaluation': eval_result
        }
        print(f"\n{'='*20} FINE TEST: {run_id} (Totale: {total_time:.2f}s) {'='*20}")
        return result

    def run_smoke_test(self) -> Dict[str, Any]:
        """Smoke test veloce per verificare il funzionamento base."""
        print("🔥 SMOKE TEST - Verifica funzionamento base")
        smoke_config = {
            'model_type': 'dense',
            'hyperparameters': {
                'epochs': [2],
                'batch_size': [32],
                'learning_rate': [0.001],
                'activation': ['relu'],
                'gru_units': [32],
                'lstm_units': [32]
            }
        }
        # Unisci override globali
        smoke_config.update(self.config_override)
        
        result = self._run_single_configuration(smoke_config)
        result['test_type'] = 'smoke_test'
        return result

    def run_full_benchmark(self, models_to_test: List[str] = None) -> Dict[str, Any]:
        """Benchmark completo che testa più modelli e iperparametri."""
        print("🚀 BENCHMARK COMPLETO")
        
        # Modelli da testare - configurabile o default
        if models_to_test is None:
            models_to_test = self.config_override.get('models_to_test', ['dense', 'gru', 'lstm', 'mlp_4_layer'])
        
        print(f"🤖 Modelli da testare: {', '.join(models_to_test)}")
        
        # Griglia di parametri da testare
        hyperparam_grid = {
            'epochs': TRAINING_CONFIG['hyperparameters'].get('epochs', [5, 10]),
            'batch_size': TRAINING_CONFIG['hyperparameters'].get('batch_size', [32, 64]),
            'learning_rate': TRAINING_CONFIG['hyperparameters'].get('learning_rate', [0.001]),
        }
        
        # Genera tutte le combinazioni di iperparametri
        keys, values = zip(*hyperparam_grid.items())
        hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        all_results = []
        
        # Pre-processa i dati una sola volta all'inizio
        print("Pre-processing dei dati per il benchmark completo...")
        self._run_single_configuration({'model_type': 'dense', 'hyperparameters': {}}) # Chiamata fittizia per caricare i dati
        if self._cached_data is None:
            print("❌ Impossibile procedere con il benchmark: preprocessing fallito.")
            return {}

        total_configs = len(models_to_test) * len(hyperparam_combinations)
        print(f"Inizio test di {total_configs} configurazioni...")

        config_num = 0
        for model_type in models_to_test:
            for hyperparams in hyperparam_combinations:
                config_num += 1
                test_config = {
                    'sample_size': self.config_override.get('sample_size', PREPROCESSING_CONFIG['sample_size']),
                    'data_path': self.config_override.get('data_path', DATA_CONFIG['dataset_path']),
                    'model_type': model_type,
                    'hyperparameters': {**TRAINING_CONFIG['hyperparameters'], **hyperparams}
                }

                print(f"\n--- Esecuzione Configurazione {config_num}/{total_configs} ---")
                result = self._run_single_configuration(test_config)
                all_results.append(result)

        # Genera summary
        summary = self._generate_summary(all_results)
        
        final_result = {
            'timestamp': self.timestamp,
            'test_type': 'full_benchmark',
            'configuration_tests': all_results,
            'summary': summary
        }
        
        print("\n🎉 BENCHMARK COMPLETO!")
        if summary.get('best_model'):
            print(f"🏆 Miglior modello: {summary['best_model']['config']['model_type']} con accuratezza {summary['best_model']['best_accuracy']:.4f}")
        return final_result

    def run_hyperband_mlp(self, max_epochs: int = 20, final_epochs: int = 30, batch_size: int = 64) -> Dict[str, Any]:
        """Esegue Hyperband su MLP 4-layer per trovare iperparametri, poi lancia un test finale a epoche fisse."""
        print("\n🔬 HYPERBAND TUNING - MLP 4-LAYER")
        # Preprocessing (usa cache se possibile)
        if self._cached_data is None:
            X, y, label_encoder = preprocess_pipeline(
                data_path=self.config_override.get('data_path', DATA_CONFIG['dataset_path']),
                sample_size=self.config_override.get('sample_size', PREPROCESSING_CONFIG['sample_size'])
            )
            self._cached_data = (X, y, label_encoder)
        else:
            X, y, label_encoder = self._cached_data

        # Flatten per MLP
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)

        from sklearn.model_selection import train_test_split
        stratify_opt = y if (np.bincount(y).min() >= 2) else None
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_opt)

        input_shape = (X_train.shape[1],)
        num_classes = max(len(np.unique(y)), int(np.max(y)) + 1)

        def model_builder(hp):
            return build_model('mlp_4_layer', input_shape, num_classes, hp)

        out_dir = 'benchmark_results'
        project = f"hyperband_mlp4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tuner = kt.Hyperband(
            model_builder,
            objective='val_accuracy',
            max_epochs=max_epochs,
            factor=3,
            directory=out_dir,
            project_name=project,
            overwrite=True
        )

        # Class weights per mitigare sbilanciamento durante la ricerca
        from sklearn.utils import class_weight
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(cw))

        import tensorflow as tf
        tuner.search(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            class_weight=class_weight_dict,
            verbose=1
        )

        best_hps = tuner.get_best_hyperparameters(1)[0]
        best_units = [
            int(best_hps.get('units_layer_1')),
            int(best_hps.get('units_layer_2')),
            int(best_hps.get('units_layer_3')),
            int(best_hps.get('units_layer_4')),
        ]
        best_lr = best_hps.get('learning_rate')
        best_act = best_hps.get('activation')

        print(f"\n🏅 Hyperband selezionato: units={best_units} | lr={best_lr} | act={best_act}")

        # Esegue un singolo test finale con gli iperparametri selezionati (epoche fisse)
        test_config = {
            'sample_size': self.config_override.get('sample_size', PREPROCESSING_CONFIG['sample_size']),
            'data_path': self.config_override.get('data_path', DATA_CONFIG['dataset_path']),
            'model_type': 'mlp_4_layer',
            'hyperparameters': {
                'epochs': [final_epochs],
                'batch_size': [batch_size],
                'learning_rate': [best_lr],
                'activation': [best_act],
                'hidden_layer_units': best_units,
            }
        }
        result = self._run_single_configuration(test_config)
        result['test_type'] = 'hyperband_mlp4_final'
        return result

    def _run_evaluation(self, model, X, y, label_encoder, test_config, class_loss_data=None):
        """Esegue valutazione completa con visualizzazioni."""
        from sklearn.model_selection import train_test_split
        
        # Gestione dataset piccoli
        if y.size == 0: return {'status': 'error', 'error': 'Empty dataset for evaluation'}
        class_counts = np.bincount(y)
        min_class_size = np.min(class_counts) if len(class_counts) > 0 else 0
        
        stratify_opt = y if min_class_size >= 2 else None
        if stratify_opt is None:
            print("    ⚠️ Dataset piccolo o sbilanciato, split semplice senza stratificazione.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_opt
        )
            
        if len(X_test) == 0:
            print("    ⚠️ Test set vuoto, uso tutto il dataset per la valutazione.")
            X_test, y_test = X, y

        # Per modelli MLP 4-layer, appiattisci finestre 3D in vettori 2D
        if test_config.get('model_type') == 'mlp_4_layer' and len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

        # Passa un array numpy per compatibilità con funzioni che usano .tolist()
        class_names = label_encoder.classes_ if hasattr(label_encoder, 'classes_') else np.array([])

        # Crea nome directory descrittivo con iperparametri
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        descriptive_name = self._generate_descriptive_folder_name(test_config, timestamp)
        # Use the main output directory from the calling script if available
        base_output_dir = test_config.get('output_dir', 'benchmark_results')
        run_dir = os.path.join(base_output_dir, descriptive_name)
        eval_dir = os.path.join(run_dir, "visualizations")


        evaluation_report = evaluate_model_comprehensive(
            model=model, X_test=X_test, y_test=y_test, class_names=class_names, 
            output_dir=eval_dir, model_config=test_config, class_loss_data=class_loss_data
        )
        return {'status': 'success', 'evaluation_dir': eval_dir, 'report': evaluation_report}

    def _generate_descriptive_folder_name(self, test_config: Dict, timestamp: str) -> str:
        """Genera un nome di cartella descrittivo con tutti i parametri."""
        model_type = test_config.get('model_type', 'unknown')
        hyperparams = test_config.get('hyperparameters', {})
        
        # Componenti del nome
        parts = [timestamp, model_type.upper()]
        
        # Aggiungi parametri chiave
        if 'epochs' in hyperparams:
            epochs = hyperparams['epochs'][0] if isinstance(hyperparams['epochs'], list) else hyperparams['epochs']
            parts.append(f"ep{epochs}")
        
        if 'batch_size' in hyperparams:
            batch_size = hyperparams['batch_size'][0] if isinstance(hyperparams['batch_size'], list) else hyperparams['batch_size']
            parts.append(f"bs{batch_size}")
        
        if 'learning_rate' in hyperparams:
            lr = hyperparams['learning_rate'][0] if isinstance(hyperparams['learning_rate'], list) else hyperparams['learning_rate']
            parts.append(f"lr{lr}")
        
        if 'activation' in hyperparams:
            activation = hyperparams['activation'][0] if isinstance(hyperparams['activation'], list) else hyperparams['activation']
            parts.append(f"act{activation}")
        
        # Parametri specifici del modello
        if model_type.lower() == 'gru' and 'gru_units' in hyperparams:
            units = hyperparams['gru_units'][0] if isinstance(hyperparams['gru_units'], list) else hyperparams['gru_units']
            parts.append(f"units{units}")
        elif model_type.lower() == 'lstm' and 'lstm_units' in hyperparams:
            units = hyperparams['lstm_units'][0] if isinstance(hyperparams['lstm_units'], list) else hyperparams['lstm_units']
            parts.append(f"units{units}")
        
        if 'dropout' in hyperparams:
            dropout = hyperparams['dropout'][0] if isinstance(hyperparams['dropout'], list) else hyperparams['dropout']
            if dropout > 0:
                parts.append(f"drop{dropout}")
        
        return "_".join(parts)

    def save_results(self, results: Dict[str, Any], output_dir: str = "benchmark_results"):
        """Salva i risultati del benchmark in JSON e CSV."""
        os.makedirs(output_dir, exist_ok=True)
        
        test_type = results.get('test_type', 'single')
        ts = int(time.time())
        results_file = os.path.join(output_dir, f"benchmark_{test_type}_{ts}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Risultati JSON salvati: {results_file}")
        
        if 'configuration_tests' in results:
            self._save_csv_summary(results, output_dir, ts)

    def _generate_summary(self, all_results: List[Dict]) -> Dict:
        """Genera un riepilogo dai risultati di un benchmark completo."""
        successful_tests = [r for r in all_results if r.get('status') == 'success']
        if not successful_tests:
            return {}

        best_model_run = max(successful_tests, key=lambda x: x.get('best_accuracy', 0))

        return {
            'total_configs_tested': len(all_results),
            'successful_configs': len(successful_tests),
            'best_model': best_model_run,
            'total_benchmark_time': sum(r.get('total_time', 0) for r in all_results)
        }

    def _save_csv_summary(self, results: Dict, output_dir: str, timestamp: int):
        """Salva un riepilogo in formato CSV per una facile analisi."""
        try:
            import pandas as pd
            
            flat_data = []
            for test_run in results.get('configuration_tests', []):
                if test_run.get('status') != 'success':
                    continue

                config = test_run.get('config', {})
                hyperparams = config.get('hyperparameters', {})

                row = {
                    'run_id': test_run.get('run_id'),
                    'model_type': config.get('model_type'),
                    'accuracy': test_run.get('best_accuracy'),
                    'training_time_s': test_run.get('training_time'),
                    'total_time_s': test_run.get('total_time'),
                    'epochs': hyperparams.get('epochs'),
                    'batch_size': hyperparams.get('batch_size'),
                    'learning_rate': hyperparams.get('learning_rate'),
                }
                flat_data.append(row)
            
            if flat_data:
                df = pd.DataFrame(flat_data)
                csv_file = os.path.join(output_dir, f"summary_full_benchmark_{timestamp}.csv")
                df.to_csv(csv_file, index=False)
                print(f"📊 Summary CSV salvato: {csv_file}")
                
        except ImportError:
            print("⚠️ Libreria 'pandas' non trovata. Impossibile salvare il summary CSV.")
        except Exception as e:
            print(f"❌ Errore durante il salvataggio del summary CSV: {e}")

    def _zip_artifacts(self, directories_to_zip: List[str], zip_filename: str):
        """Crea un archivio ZIP degli artefatti del benchmark."""
        print(f"\n📦 Creazione archivio ZIP: {zip_filename}")
        try:
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for directory in directories_to_zip:
                    if not os.path.isdir(directory):
                        print(f"  ⚠️ La directory '{directory}' non esiste, la salto.")
                        continue

                    for root, _, files in os.walk(directory):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, start=os.path.dirname(directory))
                            zipf.write(file_path, arcname)

            print(f"✅ Archivio creato con successo.")
        except Exception as e:
            print(f"❌ Errore durante la creazione dell'archivio ZIP: {e}")

def main():
    """Entry point principale per l'esecuzione dei benchmark."""
    parser = argparse.ArgumentParser(
        description='Benchmark SNN-IDS v2 (Modulare)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Esempi di utilizzo:

  # 1. Eseguire un test rapido (smoke test) per verificare che tutto funzioni
  python3 benchmark.py --smoke-test

  # 2. Eseguire un singolo test con un modello specifico e dimensione del campione
  python3 benchmark.py --model gru --sample-size 20000

  # 3. Eseguire il benchmark completo su tutti i modelli e iperparametri di default
  python3 benchmark.py --full

  # 4. Eseguire il benchmark completo solo su modelli specifici
  python3 benchmark.py --full --models gru lstm

  # 5. Eseguire il benchmark completo solo su GRU (ottimizzazione mirata)
  python3 benchmark.py --full --models gru --sample-size 50000

  # 6. Eseguire un singolo test specificando iperparametri custom
  python3 benchmark.py --model lstm --epochs 15 --batch-size 128 --learning-rate 0.0005
        '''
    )
    
    # Argomenti principali per la selezione della modalità
    parser.add_argument('--smoke-test', action='store_true', help='Esegue uno smoke test veloce e leggero.')
    parser.add_argument('--full', action='store_true', help='Esegue il benchmark completo su più modelli e iperparametri.')
    
    # Argomenti per la configurazione di base
    parser.add_argument('--sample-size', type=int, help='Numero totale di campioni da utilizzare (BENIGN + ATTACK).')
    parser.add_argument('--data-path', type=str, help='Path alla directory contenente i file CSV del dataset.')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Directory per salvare i risultati.')

    # Argomenti per la configurazione del modello (usati in test singoli o come override)
    parser.add_argument('--model', choices=['dense', 'gru', 'lstm', 'mlp_4_layer', '4layerMLP'], help='Tipo di modello da testare in un singolo run.')
    parser.add_argument('--models', nargs='+', choices=['dense', 'gru', 'lstm', 'mlp_4_layer', '4layerMLP'], 
                       help='Lista di modelli da testare nel benchmark completo (es. --models gru lstm mlp_4_layer)')
    parser.add_argument('--epochs', type=int, help="Override del numero di epoche per il training (es. 10).")
    parser.add_argument('--batch-size', type=int, help="Override della batch size per il training (es. 64).")
    parser.add_argument('--learning-rate', type=float, help="Override del learning rate (es. 0.001).")
    parser.add_argument('--hyperband-mlp', action='store_true', help='Esegue Hyperband per MLP 4-layer e poi un run finale con epoche fisse.')
    parser.add_argument('--hb-max-epochs', type=int, default=20, help='Max epochs per Hyperband (default: 20).')
    parser.add_argument('--hb-final-epochs', type=int, default=30, help='Epoche del run finale post-tuning (default: 30).')
    parser.add_argument('--hb-batch-size', type=int, default=64, help='Batch size per Hyperband e finale (default: 64).')
    
    args = parser.parse_args()
    
    # Costruisci dizionario di override dalla linea di comando
    def _normalize_model_name(name: str) -> str:
        if not name:
            return name
        n = name.strip().lower()
        if n in ('4layermlp', 'mlp_4_layer', 'mlp4', 'mlp-4-layer', 'mlp4layer'):
            return 'mlp_4_layer'
        return n

    config_override = {}
    if args.sample_size: config_override['sample_size'] = args.sample_size
    if args.data_path: config_override['data_path'] = args.data_path
    if args.model: config_override['model_type'] = _normalize_model_name(args.model)
    if args.models: config_override['models_to_test'] = [_normalize_model_name(m) for m in args.models]
    
    # Gestione override iperparametri
    hyperparam_overrides = {}
    if args.epochs: hyperparam_overrides['epochs'] = [args.epochs]
    if args.batch_size: hyperparam_overrides['batch_size'] = [args.batch_size]
    if args.learning_rate: hyperparam_overrides['learning_rate'] = [args.learning_rate]
    if hyperparam_overrides: config_override['hyperparameters'] = hyperparam_overrides

    # Crea istanza del benchmark
    benchmark = SNNIDSBenchmark(config_override)
    
    exit_code = 1  # Default a 1 (errore)
    try:
        results = None
        if args.smoke_test:
            results = benchmark.run_smoke_test()
        elif args.hyperband_mlp:
            results = benchmark.run_hyperband_mlp(
                max_epochs=args.hb_max_epochs,
                final_epochs=args.hb_final_epochs,
                batch_size=args.hb_batch_size,
            )
        elif args.full:
            results = benchmark.run_full_benchmark()
        else:
            # Esecuzione di un test singolo di default se non specificato diversamente
            print("Nessuna modalità specificata, eseguo un test singolo con configurazione di default...")
            default_config = {
                'sample_size': config_override.get('sample_size', PREPROCESSING_CONFIG['sample_size']),
                'data_path': config_override.get('data_path', DATA_CONFIG['dataset_path']),
                'model_type': config_override.get('model_type', TRAINING_CONFIG['model_type']),
                'hyperparameters': {**TRAINING_CONFIG['hyperparameters'], **config_override.get('hyperparameters', {})}
            }
            results = benchmark._run_single_configuration(default_config)
            results['test_type'] = 'single_default'

        # Salva risultati
        if results:
            benchmark.save_results(results, args.output_dir)
        
        # Stato di uscita
        final_status = results.get('status', 'error') if 'configuration_tests' not in results else results['summary'].get('status', 'success')
        exit_code = 0 if final_status == 'success' else 1
        
    except Exception as e:
        print(f"\n❌ Errore critico durante l'esecuzione del benchmark: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    finally:
        # Crea sempre l'archivio ZIP alla fine
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"benchmark_run_{ts}.zip"
        directories_to_zip = ["benchmark_results"]
        benchmark._zip_artifacts(directories_to_zip, zip_filename)

    return exit_code



if __name__ == "__main__":
    exit(main())
