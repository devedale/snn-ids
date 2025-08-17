#!/usr/bin/env python3
"""
Result Analyzer - Modulo 3: Analisi risultati e ricostruzione
Analizza risultati SNN e ricostruisce log originali
"""

import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configurazione per analisi risultati"""
    enable_reconstruction: bool = True
    confidence_threshold: float = 0.7
    anomaly_threshold: float = 0.8
    include_temporal_analysis: bool = True
    export_format: str = "json"  # "json", "csv", "html"


@dataclass
class AnalysisResult:
    """Risultato dell'analisi"""
    total_samples: int
    anomalies_detected: int
    confidence_scores: np.ndarray
    temporal_patterns: Dict[str, Any]
    reconstruction_success: bool
    analysis_timestamp: str


class LogReconstructor:
    """
    Ricostruttore di log da dati anonimizzati
    
    Usa i mapping salvati durante il preprocessing per
    ricostruire i valori originali dai dati anonimizzati
    """
    
    def __init__(self, mappings_file: str, normalization_file: str):
        self.mappings_file = mappings_file
        self.normalization_file = normalization_file
        self.mappings = self._load_mappings()
        self.normalization_stats = self._load_normalization_stats()
        
        logger.info("LogReconstructor inizializzato")
    
    def _load_mappings(self) -> Dict[str, Any]:
        """Carica mapping di anonimizzazione"""
        try:
            with open(self.mappings_file, 'r') as f:
                mappings = json.load(f)
            logger.info(f"Caricati mapping da {self.mappings_file}")
            return mappings
        except FileNotFoundError:
            logger.warning(f"File mapping {self.mappings_file} non trovato")
            return {}
    
    def _load_normalization_stats(self) -> Dict[str, Any]:
        """Carica statistiche di normalizzazione"""
        try:
            with open(self.normalization_file, 'r') as f:
                stats = json.load(f)
            logger.info(f"Caricate statistiche da {self.normalization_file}")
            return stats
        except FileNotFoundError:
            logger.warning(f"File statistiche {self.normalization_file} non trovato")
            return {}
    
    def reconstruct_value(self, field_name: str, anonymized_value: Any) -> Any:
        """Ricostruisce un singolo valore anonimizzato"""
        
        # 1. Denormalizza se necessario
        if field_name in self.normalization_stats:
            denormalized = self._denormalize_value(field_name, anonymized_value)
        else:
            denormalized = anonymized_value
        
        # 2. Reverse mapping se disponibile
        if field_name in self.mappings:
            mapping_info = self.mappings[field_name]
            
            if mapping_info['type'] == 'one_hot':
                return self._reverse_one_hot(denormalized, mapping_info)
            elif mapping_info['type'] == 'hash_numeric':
                return self._reverse_hash(denormalized, mapping_info)
            elif mapping_info['type'] == 'ordinal':
                return self._reverse_ordinal(denormalized, mapping_info)
        
        return denormalized
    
    def _denormalize_value(self, field_name: str, normalized_value: float) -> float:
        """Denormalizza un valore"""
        stats = self.normalization_stats[field_name]
        method = stats['method']
        
        if method == 'min_max':
            min_val = stats['min_val']
            max_val = stats['max_val']
            target_range = stats['target_range']
            
            # Reverse min-max normalization
            normalized_range = normalized_value - target_range[0]
            normalized_range /= (target_range[1] - target_range[0])
            original = normalized_range * (max_val - min_val) + min_val
            
            return original
        
        elif method == 'z_score':
            mean_val = stats['mean_val']
            std_val = stats['std_val']
            
            # Reverse z-score normalization
            z_score = (normalized_value * 6) - 3  # Reverse [0,1] -> [-3,3]
            original = z_score * std_val + mean_val
            
            return original
        
        return normalized_value
    
    def _reverse_one_hot(self, values: np.ndarray, mapping_info: Dict) -> str:
        """Reverse one-hot encoding"""
        if isinstance(values, (list, np.ndarray)) and len(values) > 1:
            # Trova categoria con valore massimo
            max_idx = np.argmax(values)
            categories = mapping_info['categories']
            
            if max_idx < len(categories):
                return categories[max_idx]
        
        return "unknown"
    
    def _reverse_hash(self, hashed_value: float, mapping_info: Dict) -> str:
        """Reverse hash numerico (approssimativo)"""
        mapping = mapping_info['mapping']
        
        # Trova il valore più vicino nel mapping
        closest_original = None
        min_distance = float('inf')
        
        for original, hash_val in mapping.items():
            distance = abs(hash_val - hashed_value)
            if distance < min_distance:
                min_distance = distance
                closest_original = original
        
        return closest_original or f"[HASH_{hashed_value:.4f}]"
    
    def _reverse_ordinal(self, normalized_value: float, mapping_info: Dict) -> str:
        """Reverse ordinal encoding"""
        order = mapping_info['order']
        
        # Denormalizza a indice
        index = int(normalized_value * (len(order) - 1))
        index = max(0, min(index, len(order) - 1))
        
        return order[index]
    
    def reconstruct_log_entry(self, anonymized_entry: Dict[str, Any], 
                            feature_names: List[str]) -> Dict[str, Any]:
        """Ricostruisce un'intera entry di log"""
        reconstructed = {}
        
        for i, feature_name in enumerate(feature_names):
            # Estrai nome campo base (rimuovi prefissi)
            field_name = feature_name.replace('feat_', '').split('_win_')[0]
            
            if feature_name in anonymized_entry:
                value = anonymized_entry[feature_name]
                reconstructed[field_name] = self.reconstruct_value(field_name, value)
            else:
                reconstructed[field_name] = None
        
        return reconstructed


class ResultAnalyzer:
    """
    Analizzatore principale per risultati SNN - Modulo 3
    
    Funzionalità:
    - Analisi risultati di predizione
    - Ricostruzione log originali
    - Metriche di valutazione
    - Export report
    """
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.reconstructor = None
        self.analysis_results = None
        
        logger.info("ResultAnalyzer inizializzato")
    
    def load_model_results(self, model_export_path: str) -> Dict[str, Any]:
        """Carica risultati del modello dal modulo training"""
        with open(model_export_path, 'r') as f:
            model_data = json.load(f)
        
        logger.info(f"Caricati risultati modello da {model_export_path}")
        return model_data
    
    def load_prediction_results(self, predictions_file: str) -> np.ndarray:
        """Carica predizioni del modello SNN"""
        if predictions_file.endswith('.csv'):
            df = pd.read_csv(predictions_file)
            predictions = df.values
        elif predictions_file.endswith('.json'):
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            predictions = np.array(data)
        else:
            predictions = np.load(predictions_file)
        
        logger.info(f"Caricate {len(predictions)} predizioni")
        return predictions
    
    def setup_reconstruction(self, mappings_file: str, normalization_file: str):
        """Configura ricostruzione log"""
        if self.config.enable_reconstruction:
            self.reconstructor = LogReconstructor(mappings_file, normalization_file)
            logger.info("Ricostruzione log configurata")
    
    def analyze_predictions(self, predictions: np.ndarray, 
                          original_data: Optional[np.ndarray] = None) -> AnalysisResult:
        """Analizza predizioni SNN"""
        logger.info("Inizio analisi predizioni")
        
        # Calcola confidence scores
        if len(predictions.shape) > 1:
            confidence_scores = np.max(predictions, axis=1)
        else:
            confidence_scores = predictions
        
        # Rileva anomalie
        anomalies = confidence_scores > self.config.anomaly_threshold
        num_anomalies = np.sum(anomalies)
        
        # Analisi temporale
        temporal_patterns = {}
        if self.config.include_temporal_analysis:
            temporal_patterns = self._analyze_temporal_patterns(predictions, confidence_scores)
        
        # Ricostruzione (se abilitata)
        reconstruction_success = False
        if self.reconstructor and original_data is not None:
            reconstruction_success = self._test_reconstruction(original_data)
        
        result = AnalysisResult(
            total_samples=len(predictions),
            anomalies_detected=num_anomalies,
            confidence_scores=confidence_scores,
            temporal_patterns=temporal_patterns,
            reconstruction_success=reconstruction_success,
            analysis_timestamp=datetime.now().isoformat()
        )
        
        self.analysis_results = result
        logger.info(f"Analisi completata: {num_anomalies}/{len(predictions)} anomalie")
        return result
    
    def _analyze_temporal_patterns(self, predictions: np.ndarray, 
                                 confidence_scores: np.ndarray) -> Dict[str, Any]:
        """Analizza pattern temporali"""
        patterns = {
            'confidence_trend': self._calculate_trend(confidence_scores),
            'anomaly_clusters': self._find_anomaly_clusters(confidence_scores),
            'temporal_distribution': self._analyze_distribution(confidence_scores)
        }
        
        return patterns
    
    def _calculate_trend(self, scores: np.ndarray) -> str:
        """Calcola trend generale"""
        if len(scores) < 2:
            return "insufficient_data"
        
        # Linear regression per trend
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _find_anomaly_clusters(self, scores: np.ndarray) -> List[Dict[str, int]]:
        """Trova cluster di anomalie"""
        anomalies = scores > self.config.anomaly_threshold
        clusters = []
        
        in_cluster = False
        cluster_start = 0
        
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly and not in_cluster:
                cluster_start = i
                in_cluster = True
            elif not is_anomaly and in_cluster:
                clusters.append({
                    'start': cluster_start,
                    'end': i - 1,
                    'size': i - cluster_start
                })
                in_cluster = False
        
        # Chiudi ultimo cluster se necessario
        if in_cluster:
            clusters.append({
                'start': cluster_start,
                'end': len(anomalies) - 1,
                'size': len(anomalies) - cluster_start
            })
        
        return clusters
    
    def _analyze_distribution(self, scores: np.ndarray) -> Dict[str, float]:
        """Analizza distribuzione score"""
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'percentile_95': float(np.percentile(scores, 95)),
            'percentile_99': float(np.percentile(scores, 99))
        }
    
    def _test_reconstruction(self, sample_data: np.ndarray) -> bool:
        """Testa ricostruzione su campione"""
        try:
            # Test su primo campione
            sample = sample_data[0] if len(sample_data) > 0 else []
            feature_names = [f"feat_{i}" for i in range(len(sample))]
            
            reconstructed = self.reconstructor.reconstruct_log_entry(
                {name: val for name, val in zip(feature_names, sample)},
                feature_names
            )
            
            return len(reconstructed) > 0
        except Exception as e:
            logger.warning(f"Test ricostruzione fallito: {e}")
            return False
    
    def reconstruct_highlighted_logs(self, anomaly_indices: List[int], 
                                   original_dataset: str,
                                   feature_names: List[str]) -> List[Dict[str, Any]]:
        """Ricostruisce log evidenziati come anomali"""
        if not self.reconstructor:
            raise ValueError("Ricostruzione non configurata")
        
        # Carica dataset originale
        if Path(original_dataset).exists() and original_dataset.endswith('.csv'):
            df = pd.read_csv(original_dataset)
        else:
            # Simula dataset se non esiste (per test)
            logger.warning(f"Dataset {original_dataset} non trovato, uso dati simulati")
            df = pd.DataFrame({
                f'feat_{i}': np.random.random(len(anomaly_indices)) 
                for i in range(len(feature_names))
            })
        
        reconstructed_logs = []
        
        for idx in anomaly_indices:
            if idx < len(df):
                # Estrai riga anomala
                anomaly_row = df.iloc[idx].to_dict()
                
                # Ricostruisci valori originali
                reconstructed = self.reconstructor.reconstruct_log_entry(
                    anomaly_row, feature_names
                )
                
                # Aggiungi metadati
                reconstructed['_anomaly_index'] = idx
                reconstructed['_confidence_score'] = self.analysis_results.confidence_scores[idx] if self.analysis_results else None
                reconstructed['_timestamp'] = anomaly_row.get('timestamp', 'unknown')
                
                reconstructed_logs.append(reconstructed)
        
        logger.info(f"Ricostruiti {len(reconstructed_logs)} log anomali")
        return reconstructed_logs
    
    def export_analysis_report(self, output_path: str, 
                             reconstructed_logs: Optional[List[Dict]] = None) -> str:
        """Esporta report completo dell'analisi"""
        if not self.analysis_results:
            raise ValueError("Nessuna analisi da esportare")
        
        report = {
            'analysis_summary': {
                'total_samples': self.analysis_results.total_samples,
                'anomalies_detected': self.analysis_results.anomalies_detected,
                'anomaly_rate': self.analysis_results.anomalies_detected / self.analysis_results.total_samples,
                'analysis_timestamp': self.analysis_results.analysis_timestamp
            },
            'temporal_analysis': self.analysis_results.temporal_patterns,
            'confidence_statistics': self._analyze_distribution(self.analysis_results.confidence_scores),
            'reconstruction_status': self.analysis_results.reconstruction_success
        }
        
        # Aggiungi log ricostruiti se forniti
        if reconstructed_logs:
            report['reconstructed_anomalies'] = reconstructed_logs
        
        # Export in formato richiesto
        if self.config.export_format == "json":
            # Converti NumPy types in tipi Python nativi per JSON
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                return obj
            
            report_serializable = convert_numpy_types(report)
            
            with open(output_path, 'w') as f:
                json.dump(report_serializable, f, indent=2, ensure_ascii=False)
        elif self.config.export_format == "csv":
            # Export CSV per anomalie
            if reconstructed_logs:
                df = pd.DataFrame(reconstructed_logs)
                df.to_csv(output_path, index=False)
        
        logger.info(f"Report esportato: {output_path}")
        return output_path
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Ottiene riassunto anomalie rilevate"""
        if not self.analysis_results:
            return {}
        
        anomaly_indices = np.where(
            self.analysis_results.confidence_scores > self.config.anomaly_threshold
        )[0]
        
        return {
            'total_anomalies': int(len(anomaly_indices)),
            'anomaly_indices': [int(x) for x in anomaly_indices.tolist()],
            'anomaly_rate': float(len(anomaly_indices) / self.analysis_results.total_samples),
            'confidence_threshold': float(self.config.anomaly_threshold),
            'avg_anomaly_confidence': float(np.mean(
                self.analysis_results.confidence_scores[anomaly_indices]
            )) if len(anomaly_indices) > 0 else 0.0
        }


def main():
    """Esempio di utilizzo ResultAnalyzer"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== Result Analyzer - Modulo 3 ===\n")
    
    # Configurazione analisi
    config = AnalysisConfig(
        enable_reconstruction=True,
        confidence_threshold=0.7,
        anomaly_threshold=0.8,
        include_temporal_analysis=True
    )
    
    analyzer = ResultAnalyzer(config)
    
    try:
        # Simula predizioni per test
        dummy_predictions = np.random.random((100, 2))
        dummy_predictions[10:15] = 0.9  # Simula anomalie
        
        # Analisi
        results = analyzer.analyze_predictions(dummy_predictions)
        
        print("=== Risultati Analisi ===")
        print(f"Campioni totali: {results.total_samples}")
        print(f"Anomalie rilevate: {results.anomalies_detected}")
        print(f"Trend temporale: {results.temporal_patterns.get('confidence_trend', 'N/A')}")
        print(f"Cluster anomalie: {len(results.temporal_patterns.get('anomaly_clusters', []))}")
        
        # Riassunto anomalie
        summary = analyzer.get_anomaly_summary()
        print(f"Tasso anomalie: {summary['anomaly_rate']:.2%}")
        print(f"Confidence media anomalie: {summary['avg_anomaly_confidence']:.3f}")
        
        # Export report
        report_path = analyzer.export_analysis_report("output/analysis_report.json")
        print(f"Report salvato: {report_path}")
        
    except Exception as e:
        logger.error(f"Errore nell'analisi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
