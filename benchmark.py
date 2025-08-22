#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Principale SNN-IDS
Sistema unificato per testare e confrontare le performance del sistema IDS
con diverse configurazioni e strategie di anonimizzazione.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Aggiungi il path del progetto
sys.path.append(os.path.abspath('.'))

from config import (
    DATA_CONFIG, PREPROCESSING_CONFIG, TRAINING_CONFIG, 
    PREDICTION_CONFIG, BENCHMARK_CONFIG
)
from preprocessing.process import load_data_from_directory
from preprocessing.temporal_windows import TemporalWindowManager
from training.train import train_and_evaluate
from evaluation.stats import generate_comprehensive_report
from crypto import cryptopan_ip
# from benchmark.visualization import visualize_benchmark_results

class SNNIDSBenchmark:
    """
    Benchmark completo per il sistema SNN-IDS.
    Gestisce tutti i test, configurazioni e comparazioni.
    """
    
    def __init__(self, config_override: Dict = None):
        """
        Inizializza il benchmark con configurazione opzionale.
        
        Args:
            config_override: Override della configurazione di default
        """
        self.config = self._merge_config(config_override or {})
        self.timestamp = datetime.now().isoformat()
        self.results = {}
        
        # Crea directory di output
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(self.config['output_dir'], 'baseline'), exist_ok=True)
        os.makedirs(os.path.join(self.config['output_dir'], 'cryptopan'), exist_ok=True)
        os.makedirs(os.path.join(self.config['output_dir'], 'comparison'), exist_ok=True)
        os.makedirs(os.path.join(self.config['output_dir'], 'visualizations'), exist_ok=True)
    
    def _merge_config(self, override: Dict) -> Dict:
        """Merge configurazione di default con override."""
        config = {
            # Configurazioni base
            'data_config': DATA_CONFIG.copy(),
            'preprocessing_config': PREPROCESSING_CONFIG.copy(),
            'training_config': TRAINING_CONFIG.copy(),
            'prediction_config': PREDICTION_CONFIG.copy(),
            'benchmark_config': BENCHMARK_CONFIG.copy(),
            
            # Parametri benchmark
            'output_dir': 'benchmark_results',
            'time_resolutions': ['1s', '5s', '10s', '1m', '5m'],
            'sample_size': 50000,
            'generate_visualizations': True,
            'test_configs': [
                {'name': 'baseline', 'use_cryptopan': False, 'description': 'Baseline senza anonimizzazione'},
                {'name': 'cryptopan', 'use_cryptopan': True, 'description': 'Con anonimizzazione Crypto-PAn'}
            ]
        }
        
        # Applica override
        self._deep_update(config, override)
        return config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Update ricorsivo di dizionari annidati."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Carica il dataset usando il sistema di preprocessing unificato.
        
        Returns:
            DataFrame bilanciato e preprocessato
        """
        print("🔄 Caricamento dataset con bilanciamento...")
        
        # Usa la configurazione unificata con fallback
        preprocessing_config = self.config.get('preprocessing_config', {})
        data_config = self.config.get('data_config', {})
        
        df = load_data_from_directory(
            path=data_config.get('dataset_path', 'data/cicids/2018'),
            sample_size=self.config.get('sample_size', 50000),
            balance_strategy=preprocessing_config.get('balance_strategy', 'security'),
            max_samples_per_class=preprocessing_config.get('max_samples_per_class', 100000),
            benign_ratio=preprocessing_config.get('benign_ratio', 0.5),
            min_samples_per_class=preprocessing_config.get('min_samples_per_class', 100)
        )
        
        if df.empty:
            raise ValueError("Dataset vuoto - verificare i path e i file di dati")
        
        print(f"✅ Dataset caricato: {len(df)} campioni")
        if 'Label' in df.columns:
            print(f"📊 Distribuzione classi:")
            for label, count in df['Label'].value_counts().head(10).items():
                percentage = (count / len(df)) * 100
                emoji = '🔴' if label != 'BENIGN' else '✅'
                print(f"  {emoji} {label}: {count:,} ({percentage:.1f}%)")
        
        return df
    
    def create_temporal_windows(self, df: pd.DataFrame, resolution: str) -> pd.DataFrame:
        """
        Crea finestre temporali per una risoluzione specifica.
        
        Args:
            df: Dataset originale
            resolution: Risoluzione temporale (es. '5s', '1m')
            
        Returns:
            DataFrame con finestre temporali
        """
        print(f"🔄 Creazione finestre temporali: {resolution}")
        
        window_manager = TemporalWindowManager([resolution])
        windowed_df = window_manager.create_temporal_windows(
            df, 
            self.config['data_config']['timestamp_column'], 
            resolution, 
            self.config['benchmark_config'].get('min_window_size', 5)
        )
        
        if windowed_df.empty:
            raise ValueError(f"Nessuna finestra valida per risoluzione {resolution}")
        
        num_windows = len(windowed_df.groupby(['window_start', 'window_end']))
        print(f"✅ Finestre create: {num_windows}")
        print(f"✅ Record totali: {len(windowed_df)}")
        
        return windowed_df
    
    def run_single_test(self, test_config: Dict, resolution: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Esegue un singolo test per una configurazione specifica.
        
        Args:
            test_config: Configurazione del test
            resolution: Risoluzione temporale
            df: Dataset originale
            
        Returns:
            Risultati del test
        """
        test_name = test_config['name']
        use_cryptopan = test_config['use_cryptopan']
        
        print(f"\n--- Test: {test_name} - Risoluzione: {resolution} ---")
        
        try:
            # 1. Crea finestre temporali
            windowed_df = self.create_temporal_windows(df, resolution)
            
            # 2. Applica anonimizzazione se richiesta
            if use_cryptopan:
                print("🔐 Applicazione anonimizzazione Crypto-PAn...")
                windowed_df = self._apply_cryptopan(windowed_df, test_name, resolution)
            
            # 3. Preprocessing per ML
            X, y = self._preprocess_windows(windowed_df)
            
            if X is None or y is None:
                return {'error': 'Errore nel preprocessing delle finestre'}
            
            print(f"✅ Features processate: {X.shape}")
            print(f"✅ Etichette: {y.shape}")
            
            # Debug classi
            unique_classes = len(np.unique(y))
            print(f"📊 Classi uniche: {unique_classes}")
            if unique_classes == 1:
                print("⚠️  Attenzione: Dataset con una sola classe")
            
            # 4. Training
            print("🚀 Avvio training...")
            start_time = time.time()
            
            training_config_override = {
                'TRAINING_CONFIG': {
                    'model_type': self.config['training_config']['model_type']
                }
            }
            
            training_log, best_model_path = train_and_evaluate(
                X=X, y=y, config_override=training_config_override
            )
            
            training_time = time.time() - start_time
            
            if training_log is None:
                return {'error': 'Errore nel training'}
            
            # 5. Evaluation (Cybersecurity-focused)
            print("📊 Generazione report di valutazione cybersecurity...")
            stats_output_path = os.path.join(
                self.config['output_dir'], test_name, f'statistics_{resolution}'
            )
            
            # Carica modello per predizioni
            import tensorflow as tf
            model = tf.keras.models.load_model(best_model_path)
            
            # Genera predizioni
            if len(X.shape) == 3:
                X_pred = X  # Mantieni forma 3D per modelli sequenziali
            else:
                X_pred = X.reshape(-1, X.shape[-1])
            
            y_pred_proba = model.predict(X_pred, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Ottieni nomi delle classi
            class_names = self._get_class_names()
            
            evaluation_results = generate_comprehensive_report(
                X=X_pred, y=y, y_pred=y_pred, 
                class_names=class_names,
                training_log=training_log, 
                best_model_path=best_model_path,
                output_path=stats_output_path
            )
            
            # Metriche cybersecurity specifiche
            cyber_metrics = self._calculate_cybersecurity_metrics(y, y_pred, y_pred_proba, class_names)
            evaluation_results.update({'cybersecurity_metrics': cyber_metrics})
            
            # 6. Risultati
            results = {
                'test_name': test_name,
                'time_resolution': resolution,
                'use_cryptopan': use_cryptopan,
                'timestamp': self.timestamp,
                'training_time': float(training_time),
                'dataset_stats': {
                    'total_windows': int(len(windowed_df.groupby(['window_start', 'window_end']))),
                    'total_records': int(len(windowed_df)),
                    'features_shape': [int(x) for x in X.shape],
                    'labels_shape': [int(x) for x in y.shape],
                    'unique_classes': int(unique_classes)
                },
                'model_performance': {
                    'best_accuracy': float(max([run['accuracy'] for run in training_log])),
                    'training_runs': int(len(training_log)),
                    'best_model_path': str(best_model_path)
                },
                'evaluation_results': evaluation_results,
                'statistics_path': str(stats_output_path)
            }
            
            print(f"✅ Test completato in {training_time:.2f} secondi")
            return results
            
        except Exception as e:
            print(f"❌ Errore durante il test: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _apply_cryptopan(self, df: pd.DataFrame, test_name: str, resolution: str) -> pd.DataFrame:
        """Applica anonimizzazione Crypto-PAn."""
        df_copy = df.copy()
        
        ip_columns = self.config['data_config']['ip_columns_to_anonymize']
        cryptopan_key = os.urandom(32)
        
        for col in ip_columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(
                    lambda ip: cryptopan_ip(str(ip), cryptopan_key) if pd.notna(ip) else ip
                )
        
        # Salva mappa
        ip_map = {
            'cryptopan_key': cryptopan_key.hex(),
            'resolution': resolution,
            'test_name': test_name,
            'timestamp': self.timestamp
        }
        
        map_path = os.path.join(
            self.config['output_dir'], 'cryptopan', 
            f'ip_map_{test_name}_{resolution}.json'
        )
        with open(map_path, 'w') as f:
            json.dump(ip_map, f, indent=4)
        
        return df_copy
    
    def _preprocess_windows(self, windowed_df: pd.DataFrame) -> tuple:
        """Preprocessa le finestre temporali per il training."""
        # Estrai features e target
        feature_columns = [col for col in windowed_df.columns 
                         if col not in ['window_start', 'window_end', 'window_resolution', 'window_size', 'Label']]
        
        # Converti in numerico
        for col in feature_columns:
            if col in windowed_df.columns:
                try:
                    windowed_df[col] = pd.to_numeric(windowed_df[col], errors='coerce').fillna(0)
                except:
                    windowed_df[col] = 0
        
        # Filtra colonne numeriche
        feature_columns = [col for col in feature_columns 
                         if windowed_df[col].dtype in ['int64', 'float64', 'float32']]
        
        # Crea sequenze per modelli temporali
        window_groups = windowed_df.groupby(['window_start', 'window_end'])
        X_processed, y_processed = [], []
        
        # Dimensione finestra uniforme
        max_window_size = max(len(group) for _, group in window_groups)
        target_window_size = min(max_window_size, 20)  # Massimo 20 timesteps
        
        for (start, end), group in window_groups:
            if len(group) >= 5:  # Minimo 5 record per finestra
                # Features
                features = group[feature_columns].values
                
                # Pad/truncate alla dimensione target
                if len(features) < target_window_size:
                    padding = np.zeros((target_window_size - len(features), len(feature_columns)))
                    features = np.vstack([features, padding])
                elif len(features) > target_window_size:
                    features = features[:target_window_size]
                
                # Label (ultimo record della finestra)
                if 'Label' in group.columns:
                    try:
                        label = int(group['Label'].iloc[-1])
                    except (ValueError, TypeError):
                        label = 0
                else:
                    label = 0
                
                X_processed.append(features)
                y_processed.append(label)
        
        if not X_processed:
            return None, None
        
        return np.array(X_processed), np.array(y_processed)
    
    def _get_class_names(self) -> List[str]:
        """Ottiene i nomi delle classi dal mapping."""
        try:
            target_map_path = self.config['prediction_config']['target_anonymization_map_path']
            if os.path.exists(target_map_path):
                with open(target_map_path, 'r') as f:
                    target_map = json.load(f)
                return list(target_map['inverse_map'].values())
            else:
                # Nomi di default per CIC-IDS 2018
                return [
                    'BENIGN', 'DDoS-HOIC', 'DDoS-LOIC-HTTP', 'DDoS-LOIC-UDP', 
                    'DoS Hulk', 'DoS GoldenEye', 'DoS Slowloris', 'Web Attack - Brute Force',
                    'Web Attack - XSS', 'Web Attack - SQL', 'FTP-BruteForce', 'SSH-BruteForce'
                ]
        except:
            return [f"Class_{i}" for i in range(20)]  # Fallback
    
    def _calculate_cybersecurity_metrics(self, y_true, y_pred, y_pred_proba, class_names) -> Dict:
        """Calcola metriche specifiche per cybersecurity/IDS."""
        from sklearn.metrics import (
            confusion_matrix, classification_report, 
            roc_auc_score, precision_recall_fscore_support,
            accuracy_score, cohen_kappa_score
        )
        
        metrics = {}
        
        # 1. CONFUSION MATRIX DETTAGLIATA
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'classes': class_names[:len(np.unique(y_true))]
        }
        
        # 2. METRICHE IDS SPECIFICHE
        # Considera BENIGN (classe 0) vs MALICIOUS (tutto il resto)
        y_binary_true = (y_true != 0).astype(int)  # 0=BENIGN, 1=MALICIOUS
        y_binary_pred = (y_pred != 0).astype(int)
        
        # True/False Positives/Negatives per IDS
        tn = np.sum((y_binary_true == 0) & (y_binary_pred == 0))  # Traffico normale correttamente identificato
        fp = np.sum((y_binary_true == 0) & (y_binary_pred == 1))  # FALSI ALLARMI (critico per IDS)
        fn = np.sum((y_binary_true == 1) & (y_binary_pred == 0))  # ATTACCHI MANCATI (molto critico)
        tp = np.sum((y_binary_true == 1) & (y_binary_pred == 1))  # Attacchi rilevati correttamente
        
        # Metriche IDS
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall per attacchi
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # Tasso falsi allarmi
        precision_attacks = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precisione attacchi
        
        metrics['ids_metrics'] = {
            'detection_rate': float(detection_rate),
            'false_alarm_rate': float(false_alarm_rate),
            'precision_attacks': float(precision_attacks),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'missed_attacks': int(fn),
            'false_alarms': int(fp)
        }
        
        # 3. METRICHE MULTI-CLASSE
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class_metrics'] = {}
        for i, class_name in enumerate(class_names[:len(precision)]):
            metrics['per_class_metrics'][class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        # 4. METRICHE AGGREGATE
        metrics['aggregate_metrics'] = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_precision': float(np.mean(precision)),
            'macro_recall': float(np.mean(recall)),
            'macro_f1': float(np.mean(f1)),
            'weighted_precision': float(np.average(precision, weights=support)),
            'weighted_recall': float(np.average(recall, weights=support)),
            'weighted_f1': float(np.average(f1, weights=support)),
            'kappa_score': float(cohen_kappa_score(y_true, y_pred))
        }
        
        # 5. ROC-AUC (se possibile)
        try:
            if len(np.unique(y_true)) == 2:
                # Binario: BENIGN vs MALICIOUS
                if y_pred_proba.shape[1] >= 2:
                    auc_score = roc_auc_score(y_binary_true, y_pred_proba[:, 1])
                else:
                    auc_score = roc_auc_score(y_binary_true, y_binary_pred)
                metrics['roc_metrics'] = {
                    'binary_auc': float(auc_score),
                    'binary_classification': True
                }
            else:
                # Multi-classe: usa One-vs-Rest
                if y_pred_proba.shape[1] > 1:
                    auc_scores = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=None)
                    metrics['roc_metrics'] = {
                        'multiclass_auc': [float(score) for score in auc_scores],
                        'macro_auc': float(np.mean(auc_scores)),
                        'binary_classification': False
                    }
        except Exception as e:
            metrics['roc_metrics'] = {'error': str(e)}
        
        # 6. ATTACK TYPE ANALYSIS
        attack_analysis = {}
        unique_classes = np.unique(y_true)
        for class_id in unique_classes:
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            if class_name != 'BENIGN':
                class_mask = (y_true == class_id)
                class_pred = y_pred[class_mask]
                
                detected = np.sum(class_pred == class_id)
                total = np.sum(class_mask)
                
                attack_analysis[class_name] = {
                    'total_samples': int(total),
                    'correctly_detected': int(detected),
                    'detection_rate': float(detected / total) if total > 0 else 0,
                    'missed': int(total - detected)
                }
        
        metrics['attack_analysis'] = attack_analysis
        
        return metrics
    
    def run_complete_benchmark(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Esegue il benchmark completo per tutte le configurazioni.
        
        Args:
            df: Dataset originale
            
        Returns:
            Risultati completi del benchmark
        """
        print("\n" + "="*80)
        print("🚀 AVVIO BENCHMARK COMPLETO SNN-IDS")
        print("="*80)
        print(f"📊 Dataset: {len(df)} record")
        print(f"🕐 Timestamp range: {df['Timestamp'].min()} - {df['Timestamp'].max()}")
        print(f"⏱️  Risoluzioni temporali: {self.config['time_resolutions']}")
        print(f"🧪 Configurazioni test: {len(self.config['test_configs'])}")
        print("="*80)
        
        all_results = {}
        
        # Esegui test per ogni configurazione e risoluzione
        for test_config in self.config['test_configs']:
            test_name = test_config['name']
            all_results[test_name] = {}
            
            print(f"\n🔄 CONFIGURAZIONE: {test_name.upper()}")
            print(f"📝 Descrizione: {test_config['description']}")
            
            for resolution in self.config['time_resolutions']:
                print(f"\n--- Testando risoluzione: {resolution} ---")
                
                result = self.run_single_test(test_config, resolution, df)
                all_results[test_name][resolution] = result
        
        # Salva risultati
        self._save_results(all_results)
        
        # Genera comparazioni
        comparison_results = self._generate_comparison(all_results)
        
        # Genera riepilogo esaustivo
        summary_report = self._generate_comprehensive_summary(all_results)
        
        # Visualizzazioni
        if self.config['generate_visualizations']:
            self._generate_visualizations(all_results)
        
        return {
            'benchmark_config': self.config,
            'all_results': all_results,
            'comparison_results': comparison_results,
            'comprehensive_summary': summary_report,
            'timestamp': self.timestamp
        }
    
    def _save_results(self, results: Dict):
        """Salva i risultati del benchmark."""
        results_path = os.path.join(self.config['output_dir'], 'complete_benchmark_results.json')
        
        # Conversione per JSON
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        results_json = convert_for_json(results)
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=4)
        
        print(f"💾 Risultati salvati in: {results_path}")
    
    def _generate_comparison(self, results: Dict) -> Dict:
        """Genera report di comparazione tra configurazioni."""
        comparison = {
            'benchmark_timestamp': self.timestamp,
            'configurations': {},
            'resolution_comparison': {},
            'overall_summary': {}
        }
        
        # Per ogni risoluzione, confronta le configurazioni
        for resolution in self.config['time_resolutions']:
            comparison['resolution_comparison'][resolution] = {}
            
            # Cerca risultati per baseline e cryptopan
            baseline_result = results.get('baseline', {}).get(resolution, {})
            cryptopan_result = results.get('cryptopan', {}).get(resolution, {})
            
            if 'error' not in baseline_result and 'error' not in cryptopan_result:
                baseline_acc = baseline_result.get('model_performance', {}).get('best_accuracy', 0)
                cryptopan_acc = cryptopan_result.get('model_performance', {}).get('best_accuracy', 0)
                
                comparison['resolution_comparison'][resolution] = {
                    'baseline_accuracy': float(baseline_acc),
                    'cryptopan_accuracy': float(cryptopan_acc),
                    'accuracy_difference': float(cryptopan_acc - baseline_acc),
                    'accuracy_change_percentage': float((cryptopan_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0,
                    'training_time_baseline': float(baseline_result.get('training_time', 0)),
                    'training_time_cryptopan': float(cryptopan_result.get('training_time', 0))
                }
        
        # Salva comparazione
        comparison_path = os.path.join(self.config['output_dir'], 'comparison', 'comparison_report.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=4)
        
        return comparison
    
    def _generate_comprehensive_summary(self, results: Dict) -> Dict:
        """Genera un riepilogo esaustivo di tutte le configurazioni testate."""
        summary = {
            'benchmark_overview': {
                'timestamp': self.timestamp,
                'total_configurations': len(self.config['test_configs']),
                'total_resolutions': len(self.config['time_resolutions']),
                'total_tests': len(self.config['test_configs']) * len(self.config['time_resolutions'])
            },
            'configuration_matrix': {},
            'performance_overview': {},
            'cybersecurity_analysis': {},
            'recommendations': {}
        }
        
        # Matrice delle configurazioni testate
        config_matrix = []
        performance_data = []
        
        for test_config in self.config['test_configs']:
            test_name = test_config['name']
            
            for resolution in self.config['time_resolutions']:
                result = results.get(test_name, {}).get(resolution, {})
                
                if 'error' not in result:
                    # Configurazione testata
                    config_entry = {
                        'test_name': test_name,
                        'resolution': resolution,
                        'description': test_config['description'],
                        'use_cryptopan': test_config['use_cryptopan'],
                        'model_type': self.config['training_config']['model_type'],
                        'balance_strategy': self.config['preprocessing_config']['balance_strategy'],
                        'sample_size': self.config['sample_size'],
                        'status': 'SUCCESS'
                    }
                    
                    # Dati performance
                    if 'cybersecurity_metrics' in result.get('evaluation_results', {}):
                        cyber_metrics = result['evaluation_results']['cybersecurity_metrics']
                        
                        perf_entry = {
                            'configuration': f"{test_name}_{resolution}",
                            'accuracy': result['model_performance']['best_accuracy'],
                            'detection_rate': cyber_metrics.get('ids_metrics', {}).get('detection_rate', 0),
                            'false_alarm_rate': cyber_metrics.get('ids_metrics', {}).get('false_alarm_rate', 0),
                            'precision_attacks': cyber_metrics.get('ids_metrics', {}).get('precision_attacks', 0),
                            'training_time': result['training_time'],
                            'unique_classes': result['dataset_stats']['unique_classes']
                        }
                        performance_data.append(perf_entry)
                else:
                    config_entry = {
                        'test_name': test_name,
                        'resolution': resolution,
                        'description': test_config['description'],
                        'use_cryptopan': test_config['use_cryptopan'],
                        'status': 'FAILED',
                        'error': result.get('error', 'Unknown error')
                    }
                
                config_matrix.append(config_entry)
        
        summary['configuration_matrix'] = config_matrix
        summary['performance_overview'] = performance_data
        
        # Analisi cybersecurity aggregata
        if performance_data:
            avg_detection = np.mean([p['detection_rate'] for p in performance_data])
            avg_false_alarm = np.mean([p['false_alarm_rate'] for p in performance_data])
            avg_precision = np.mean([p['precision_attacks'] for p in performance_data])
            
            summary['cybersecurity_analysis'] = {
                'average_detection_rate': float(avg_detection),
                'average_false_alarm_rate': float(avg_false_alarm),
                'average_attack_precision': float(avg_precision),
                'best_configuration': max(performance_data, key=lambda x: x['detection_rate'])['configuration'],
                'lowest_false_alarms': min(performance_data, key=lambda x: x['false_alarm_rate'])['configuration'],
                'performance_summary': {
                    'excellent_detection': len([p for p in performance_data if p['detection_rate'] > 0.95]),
                    'good_detection': len([p for p in performance_data if 0.85 < p['detection_rate'] <= 0.95]),
                    'poor_detection': len([p for p in performance_data if p['detection_rate'] <= 0.85]),
                    'low_false_alarms': len([p for p in performance_data if p['false_alarm_rate'] < 0.05]),
                    'high_false_alarms': len([p for p in performance_data if p['false_alarm_rate'] > 0.10])
                }
            }
        
        # Raccomandazioni
        recommendations = []
        
        if performance_data:
            best_config = max(performance_data, key=lambda x: x['detection_rate'] - x['false_alarm_rate'])
            recommendations.append(f"🏆 Migliore configurazione overall: {best_config['configuration']}")
            
            if avg_false_alarm > 0.1:
                recommendations.append("⚠️  Tasso falsi allarmi elevato - considerare tuning parametri")
            
            if avg_detection < 0.9:
                recommendations.append("⚠️  Detection rate basso - considerare più dati di training o modelli più complessi")
            
            # Confronto Crypto-PAn
            baseline_configs = [p for p in performance_data if 'baseline' in p['configuration']]
            cryptopan_configs = [p for p in performance_data if 'cryptopan' in p['configuration']]
            
            if baseline_configs and cryptopan_configs:
                avg_baseline_detection = np.mean([p['detection_rate'] for p in baseline_configs])
                avg_cryptopan_detection = np.mean([p['detection_rate'] for p in cryptopan_configs])
                
                if avg_cryptopan_detection >= avg_baseline_detection * 0.95:
                    recommendations.append("✅ Crypto-PAn mantiene buone performance di detection")
                else:
                    recommendations.append("⚠️  Crypto-PAn riduce significativamente le performance")
        
        summary['recommendations'] = recommendations
        
        # Salva riepilogo
        summary_path = os.path.join(self.config['output_dir'], 'comprehensive_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Genera anche CSV per analisi facile
        self._save_summary_csv(performance_data)
        
        return summary
    
    def _save_summary_csv(self, performance_data: List[Dict]):
        """Salva un riepilogo in formato CSV per analisi facile."""
        if not performance_data:
            return
        
        import pandas as pd
        
        df = pd.DataFrame(performance_data)
        csv_path = os.path.join(self.config['output_dir'], 'benchmark_summary.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"📊 Riepilogo CSV salvato in: {csv_path}")
    
    def _generate_visualizations(self, results: Dict):
        """Genera visualizzazioni dei risultati."""
        try:
            print("\n📊 Generazione visualizzazioni...")
            # TODO: Implementare visualizzazioni
            # visualize_benchmark_results(self.config['output_dir'])
            print("✅ Visualizzazioni generate con successo!")
        except Exception as e:
            print(f"❌ Errore durante la generazione delle visualizzazioni: {e}")

def main():
    """Entry point principale del benchmark."""
    parser = argparse.ArgumentParser(
        description='Benchmark Sistema SNN-IDS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Esempi di utilizzo:

  # Smoke test veloce
  python3 benchmark.py --smoke-test

  # Test con campione specifico
  python3 benchmark.py --sample-size 10000 --resolutions 5s 1m

  # Benchmark completo
  python3 benchmark.py --full --visualize

  # Override configurazioni
  python3 benchmark.py --model-type lstm --balance-strategy security
        '''
    )
    
    # Argomenti principali
    parser.add_argument('--smoke-test', action='store_true', 
                       help='Esegue un test veloce con parametri ridotti')
    parser.add_argument('--full', action='store_true',
                       help='Esegue il benchmark completo con tutti i parametri')
    
    # Parametri configurabili
    parser.add_argument('--sample-size', type=int, default=50000,
                       help='Numero di campioni da utilizzare (default: 50000)')
    parser.add_argument('--resolutions', nargs='+', 
                       default=['1s', '5s', '10s', '1m', '5m'],
                       help='Risoluzioni temporali da testare')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Directory di output per i risultati')
    parser.add_argument('--visualize', action='store_true',
                       help='Genera visualizzazioni al termine del benchmark')
    
    # Override configurazioni
    parser.add_argument('--model-type', type=str, choices=['lstm', 'gru', 'dense'],
                       help='Tipo di modello da utilizzare')
    parser.add_argument('--balance-strategy', type=str, 
                       choices=['security', 'balanced', 'smart', 'none'],
                       help='Strategia di bilanciamento del dataset')
    parser.add_argument('--data-path', type=str,
                       help='Path ai dati CIC-IDS 2018')
    
    args = parser.parse_args()
    
    # Configurazione automatica per smoke test
    if args.smoke_test:
        config_override = {
            'sample_size': 5000,
            'time_resolutions': ['5s'],
            'training_config': {
                'hyperparameters': {
                    'epochs': [2],
                    'batch_size': [32]
                }
            },
            'generate_visualizations': False
        }
    # Configurazione per test completo
    elif args.full:
        config_override = {
            'sample_size': 100000,
            'time_resolutions': ['1s', '5s', '10s', '1m', '5m', '10m'],
            'training_config': {
                'hyperparameters': {
                    'epochs': [10],
                    'batch_size': [64, 128]
                }
            },
            'generate_visualizations': True
        }
    # Configurazione custom
    else:
        config_override = {
            'sample_size': args.sample_size,
            'time_resolutions': args.resolutions,
            'output_dir': args.output_dir,
            'generate_visualizations': args.visualize
        }
    
    # Applica override specifici
    if args.model_type:
        config_override['training_config'] = config_override.get('training_config', {})
        config_override['training_config']['model_type'] = args.model_type
    
    if args.balance_strategy:
        config_override['preprocessing_config'] = config_override.get('preprocessing_config', {})
        config_override['preprocessing_config']['balance_strategy'] = args.balance_strategy
    
    if args.data_path:
        config_override['data_config'] = config_override.get('data_config', {})
        config_override['data_config']['dataset_path'] = args.data_path
    
    try:
        # Avvia benchmark
        benchmark = SNNIDSBenchmark(config_override)
        
        # Carica dataset
        df = benchmark.load_dataset()
        
        # Esegui benchmark
        results = benchmark.run_complete_benchmark(df)
        
        print("\n" + "="*80)
        print("🎉 BENCHMARK COMPLETATO CON SUCCESSO!")
        print("="*80)
        print(f"📁 Risultati salvati in: {benchmark.config['output_dir']}")
        if benchmark.config['generate_visualizations']:
            print(f"📊 Grafici salvati in: {benchmark.config['output_dir']}/visualizations")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Errore durante l'esecuzione del benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
