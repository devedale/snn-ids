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

# Import moduli
sys.path.append(os.path.abspath('.'))
from config import *
from preprocessing.process import preprocess_pipeline
from training.train import train_model
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
            model, training_log, model_path = train_model(
                X=X, y=y,
                model_type=test_config['model_type'],
                hyperparams=test_config['hyperparameters']
            )
            train_time = time.time() - train_start
            best_accuracy = max([run['accuracy'] for run in training_log]) if training_log else 0
            print(f"✅ Training completato in {train_time:.2f}s. Accuratezza: {best_accuracy:.4f}")
        except Exception as e:
            print(f"❌ Errore nel training: {e}")
            return {'status': 'error', 'stage': 'training', 'error': str(e), 'config': test_config}

        # --- 3. Valutazione ---
        print("\n📊 VALUTAZIONE")
        try:
            eval_result = self._run_evaluation(model, X, y, label_encoder, test_config['model_type'])
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
            'sample_size': self.config_override.get('sample_size', 5000),
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

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Benchmark completo che testa più modelli e iperparametri."""
        print("🚀 BENCHMARK COMPLETO")
        
        # Griglia di parametri da testare
        models_to_test = ['dense', 'gru', 'lstm']
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

    def _run_evaluation(self, model, X, y, label_encoder, model_type):
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

        class_names = label_encoder.classes_.tolist() if hasattr(label_encoder, 'classes_') else []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join("benchmark_results", f"{timestamp}_{model_type}_evaluation", "visualizations")

        evaluation_report = evaluate_model_comprehensive(
            model=model, X_test=X_test, y_test=y_test, class_names=class_names, output_dir=eval_dir
        )
        return {'status': 'success', 'evaluation_dir': eval_dir, 'report': evaluation_report}

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
                    'epochs': hyperparams.get('epochs', [None])[0],
                    'batch_size': hyperparams.get('batch_size', [None])[0],
                    'learning_rate': hyperparams.get('learning_rate', [None])[0],
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

  # 4. Eseguire il benchmark completo con una dimensione del campione personalizzata
  python3 benchmark.py --full --sample-size 50000

  # 5. Eseguire un singolo test specificando iperparametri custom (nota: devono essere nel formato atteso dal modulo di training)
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
    parser.add_argument('--model', choices=['dense', 'gru', 'lstm'], help='Tipo di modello da testare in un singolo run.')
    parser.add_argument('--epochs', type=int, help="Override del numero di epoche per il training (es. 10).")
    parser.add_argument('--batch-size', type=int, help="Override della batch size per il training (es. 64).")
    parser.add_argument('--learning-rate', type=float, help="Override del learning rate (es. 0.001).")
    
    args = parser.parse_args()
    
    # Costruisci dizionario di override dalla linea di comando
    config_override = {}
    if args.sample_size: config_override['sample_size'] = args.sample_size
    if args.data_path: config_override['data_path'] = args.data_path
    if args.model: config_override['model_type'] = args.model
    
    # Gestione override iperparametri
    hyperparam_overrides = {}
    if args.epochs: hyperparam_overrides['epochs'] = [args.epochs]
    if args.batch_size: hyperparam_overrides['batch_size'] = [args.batch_size]
    if args.learning_rate: hyperparam_overrides['learning_rate'] = [args.learning_rate]
    if hyperparam_overrides: config_override['hyperparameters'] = hyperparam_overrides

    # Crea istanza del benchmark
    benchmark = SNNIDSBenchmark(config_override)
    
    try:
        results = None
        if args.smoke_test:
            results = benchmark.run_smoke_test()
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
        return 0 if final_status == 'success' else 1
        
    except Exception as e:
        print(f"\n❌ Errore critico durante l'esecuzione del benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
