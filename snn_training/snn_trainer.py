#!/usr/bin/env python3
"""
SNN Trainer - Modulo 2: Training SNN
Trainer per Spiking Neural Networks con framework multipli
"""

import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, fields
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configurazione per training SNN"""
    framework: str = "nengo"  # "nengo", "norse", "snntorch", "bindsnet"
    
    # Architettura rete
    input_neurons: int = 10
    hidden_layers: List[int] = None
    output_neurons: int = 2
    neuron_type: str = "lif"  # "lif", "adaptive", "izhikevich"
    
    # Training parameters
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    
    # SNN-specific
    dt: float = 0.001  # Time step
    simulation_time: float = 1.0  # Simulation duration
    encoding_method: str = "rate"  # "rate", "temporal", "population"
    
    # Output
    model_name: str = "snn_model"
    save_checkpoints: bool = True
    export_format: str = "standard"  # "standard", "onnx", "tensorflow"

    @staticmethod
    def from_yaml(config_path: str) -> "TrainingConfig":
        """Crea una TrainingConfig caricandola da file YAML.

        Ignora chiavi sconosciute e mantiene i default per chiavi mancanti.
        """
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PyYAML non installato. Installa con: pip install pyyaml"
            ) from e

        from pathlib import Path as _Path
        cfg_file = _Path(config_path)
        if not cfg_file.exists():
            raise FileNotFoundError(f"File di configurazione non trovato: {config_path}")

        with open(cfg_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        valid_names = {f.name for f in fields(TrainingConfig)}
        filtered = {k: v for k, v in data.items() if k in valid_names}
        return TrainingConfig(**filtered)


class SNNFramework(ABC):
    """Interfaccia astratta per framework SNN"""
    
    @abstractmethod
    def build_model(self, config: TrainingConfig) -> Any:
        """Costruisce il modello SNN"""
        pass
    
    @abstractmethod
    def train_model(self, model: Any, train_data: Tuple, config: TrainingConfig) -> Dict[str, Any]:
        """Allena il modello"""
        pass
    
    @abstractmethod
    def evaluate_model(self, model: Any, test_data: Tuple) -> Dict[str, Any]:
        """Valuta il modello"""
        pass
    
    @abstractmethod
    def save_model(self, model: Any, path: str, config: TrainingConfig) -> str:
        """Salva il modello"""
        pass


class NengoFramework(SNNFramework):
    """Framework Nengo per SNN"""
    
    def __init__(self):
        try:
            import nengo
            import nengo_dl
            self.nengo = nengo
            self.nengo_dl = nengo_dl
            self.available = True
            logger.info("Nengo framework disponibile")
        except ImportError:
            self.available = False
            logger.warning("Nengo non disponibile. Installare con: pip install nengo nengo-dl")
    
    def build_model(self, config: TrainingConfig) -> Any:
        """Costruisce modello Nengo"""
        if not self.available:
            raise ImportError("Nengo non disponibile")
        
        with self.nengo.Network() as net:
            # Input layer
            input_layer = self.nengo.Ensemble(
                config.input_neurons, 
                dimensions=config.input_neurons,
                neuron_type=self._get_neuron_type(config.neuron_type)
            )
            
            # Hidden layers
            hidden_layers = config.hidden_layers or [64, 32]
            layers = [input_layer]
            
            for i, n_neurons in enumerate(hidden_layers):
                hidden = self.nengo.Ensemble(
                    n_neurons,
                    dimensions=n_neurons,
                    neuron_type=self._get_neuron_type(config.neuron_type)
                )
                
                # Connetti layer precedente
                self.nengo.Connection(layers[-1], hidden)
                layers.append(hidden)
            
            # Output layer
            output_layer = self.nengo.Ensemble(
                config.output_neurons,
                dimensions=config.output_neurons,
                neuron_type=self.nengo.LIF()
            )
            self.nengo.Connection(layers[-1], output_layer)
            
            # Probe per output
            output_probe = self.nengo.Probe(output_layer, synapse=0.1)
            
        return net, output_probe
    
    def _get_neuron_type(self, neuron_type: str):
        """Ottiene tipo di neurone Nengo"""
        if neuron_type == "lif":
            return self.nengo.LIF()
        elif neuron_type == "adaptive":
            return self.nengo.AdaptiveLIF()
        else:
            return self.nengo.LIF()
    
    def train_model(self, model: Any, train_data: Tuple, config: TrainingConfig) -> Dict[str, Any]:
        """Allena modello Nengo"""
        net, probe = model
        X_train, y_train = train_data
        
        # Converti dati per Nengo
        train_inputs = self._encode_inputs(X_train, config)
        train_targets = self._encode_targets(y_train, config)
        
        # Training con nengo-dl
        with self.nengo_dl.Simulator(net, minibatch_size=config.batch_size) as sim:
            # Configura training
            sim.compile(
                optimizer='adam',
                loss='mse',
                metrics=['accuracy']
            )
            
            # Train
            history = sim.fit(
                train_inputs,
                train_targets,
                epochs=config.epochs,
                validation_split=config.validation_split,
                verbose=1
            )
        
        return {
            'history': history.history if hasattr(history, 'history') else {},
            'framework': 'nengo',
            'final_loss': history.history.get('loss', [0])[-1] if hasattr(history, 'history') else 0
        }
    
    def _encode_inputs(self, X: np.ndarray, config: TrainingConfig) -> np.ndarray:
        """Codifica input per SNN"""
        # Rate coding: valore -> frequenza spike
        timesteps = int(config.simulation_time / config.dt)
        
        if config.encoding_method == "rate":
            # Rate encoding
            encoded = np.repeat(X[:, np.newaxis, :], timesteps, axis=1)
            return encoded
        else:
            # Temporal encoding (implementazione base)
            return np.repeat(X[:, np.newaxis, :], timesteps, axis=1)
    
    def _encode_targets(self, y: np.ndarray, config: TrainingConfig) -> np.ndarray:
        """Codifica target per SNN"""
        timesteps = int(config.simulation_time / config.dt)
        
        # One-hot encoding se necessario
        if len(y.shape) == 1:
            from sklearn.preprocessing import LabelEncoder, OneHotEncoder
            y_encoded = np.eye(config.output_neurons)[y.astype(int)]
        else:
            y_encoded = y
        
        return np.repeat(y_encoded[:, np.newaxis, :], timesteps, axis=1)
    
    def evaluate_model(self, model: Any, test_data: Tuple) -> Dict[str, Any]:
        """Valuta modello Nengo"""
        net, probe = model
        X_test, y_test = test_data
        
        test_inputs = self._encode_inputs(X_test, TrainingConfig())
        
        with self.nengo_dl.Simulator(net) as sim:
            sim.run_steps(test_inputs.shape[1], data={sim.model.inputs[0]: test_inputs})
            outputs = sim.data[probe]
        
        # Calcola metriche
        predictions = np.mean(outputs, axis=1)  # Media temporale
        
        return {
            'predictions': predictions,
            'accuracy': self._calculate_accuracy(predictions, y_test),
            'mse': np.mean((predictions - y_test)**2)
        }
    
    def _calculate_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calcola accuracy"""
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(targets, axis=1) if len(targets.shape) > 1 else targets
        return np.mean(pred_classes == true_classes)
    
    def save_model(self, model: Any, path: str, config: TrainingConfig) -> str:
        """Salva modello Nengo"""
        net, probe = model
        
        # Salva network
        import pickle
        model_path = f"{path}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump((net, probe), f)
        
        # Salva configurazione
        config_path = f"{path}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'framework': 'nengo',
                'input_neurons': config.input_neurons,
                'hidden_layers': config.hidden_layers,
                'output_neurons': config.output_neurons,
                'neuron_type': config.neuron_type
            }, f, indent=2)
        
        return model_path


class MockFramework(SNNFramework):
    """Framework mock per testing senza dipendenze esterne"""
    
    def build_model(self, config: TrainingConfig) -> Any:
        """Costruisce modello mock"""
        model = {
            'type': 'mock_snn',
            'input_size': config.input_neurons,
            'hidden_layers': config.hidden_layers or [64, 32],
            'output_size': config.output_neurons,
            'weights': self._initialize_weights(config)
        }
        return model
    
    def _initialize_weights(self, config: TrainingConfig) -> Dict[str, np.ndarray]:
        """Inizializza pesi casuali"""
        hidden_layers = config.hidden_layers or [64, 32]
        weights = {}
        
        prev_size = config.input_neurons
        for i, size in enumerate(hidden_layers):
            weights[f'layer_{i}'] = np.random.normal(0, 0.1, (prev_size, size))
            prev_size = size
        
        weights['output'] = np.random.normal(0, 0.1, (prev_size, config.output_neurons))
        return weights
    
    def train_model(self, model: Any, train_data: Tuple, config: TrainingConfig) -> Dict[str, Any]:
        """Simula training"""
        X_train, y_train = train_data
        
        # Simula training con loss decrescente
        history = {
            'loss': [1.0 - i/config.epochs for i in range(config.epochs)],
            'accuracy': [i/config.epochs for i in range(config.epochs)]
        }
        
        return {
            'history': history,
            'framework': 'mock',
            'final_loss': history['loss'][-1]
        }
    
    def evaluate_model(self, model: Any, test_data: Tuple) -> Dict[str, Any]:
        """Simula valutazione"""
        X_test, y_test = test_data
        
        # Simula predizioni casuali
        predictions = np.random.random((len(X_test), model['output_size']))
        
        return {
            'predictions': predictions,
            'accuracy': 0.85,  # Accuracy simulata
            'mse': 0.15
        }
    
    def save_model(self, model: Any, path: str, config: TrainingConfig) -> str:
        """Salva modello mock"""
        model_path = f"{path}.json"
        
        # Serializza modello
        serializable_model = {
            'type': model['type'],
            'input_size': model['input_size'],
            'hidden_layers': model['hidden_layers'],
            'output_size': model['output_size'],
            'weights': {k: v.tolist() for k, v in model['weights'].items()}
        }
        
        with open(model_path, 'w') as f:
            json.dump(serializable_model, f, indent=2)
        
        return model_path


class SNNTrainer:
    """
    Trainer principale per SNN - Modulo 2
    
    Funzionalità:
    - Training con framework multipli
    - Gestione configurazioni
    - Export/import modelli
    - Metriche di valutazione
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.framework = self._get_framework()
        self.model = None
        self.training_history = None
        
        logger.info(f"SNNTrainer inizializzato con framework: {self.config.framework}")
    
    def _get_framework(self) -> SNNFramework:
        """Ottiene framework SNN richiesto"""
        if self.config.framework == "nengo":
            return NengoFramework()
        elif self.config.framework == "mock":
            return MockFramework()
        else:
            logger.warning(f"Framework {self.config.framework} non supportato, uso mock")
            return MockFramework()
    
    def load_training_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Carica dati di training da file CSV SNN"""
        df = pd.read_csv(data_path)
        
        # Separazione features e target
        timestamp_col = 'timestamp'
        feature_cols = [col for col in df.columns if col.startswith('feat_') or col not in [timestamp_col]]
        
        X = df[feature_cols].values
        
        # Target: per ora uso classificazione binaria basata su pattern
        # In futuro può essere configurabile
        y = self._generate_targets(X)
        
        logger.info(f"Caricati {len(X)} campioni con {X.shape[1]} features")
        return X, y
    
    def _generate_targets(self, X: np.ndarray) -> np.ndarray:
        """Genera target per training (anomaly detection)"""
        # Esempio: classifica basata su deviazione dalla media
        mean_vals = np.mean(X, axis=1)
        threshold = np.percentile(mean_vals, 80)  # Top 20% come anomali
        
        y = (mean_vals > threshold).astype(int)
        logger.info(f"Generati target: {np.sum(y)} anomalie su {len(y)} campioni")
        return y
    
    def train(self, data_path: str) -> Dict[str, Any]:
        """Esegue training completo"""
        logger.info("Inizio training SNN")
        
        # Carica dati
        X, y = self.load_training_data(data_path)
        
        # Split train/validation
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Aggiorna configurazione input
        self.config.input_neurons = X.shape[1]
        
        # Costruisci modello
        self.model = self.framework.build_model(self.config)
        
        # Training
        self.training_history = self.framework.train_model(
            self.model, (X_train, y_train), self.config
        )
        
        # Valutazione
        val_metrics = self.framework.evaluate_model(self.model, (X_val, y_val))
        
        results = {
            'training_history': self.training_history,
            'validation_metrics': val_metrics,
            'model_config': self.config,
            'data_shape': X.shape
        }
        
        logger.info(f"Training completato. Val accuracy: {val_metrics.get('accuracy', 0):.3f}")
        return results
    
    def save_model(self, output_dir: str = "models") -> str:
        """Salva modello trained"""
        if self.model is None:
            raise ValueError("Nessun modello da salvare. Eseguire prima il training.")
        
        Path(output_dir).mkdir(exist_ok=True)
        model_path = Path(output_dir) / self.config.model_name
        
        saved_path = self.framework.save_model(self.model, str(model_path), self.config)
        
        # Salva anche training history
        if self.training_history:
            history_path = f"{model_path}_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Modello salvato: {saved_path}")
        return saved_path
    
    def export_for_analysis(self, output_path: str) -> str:
        """Esporta modello per modulo di analisi"""
        if self.model is None:
            raise ValueError("Nessun modello da esportare")
        
        export_data = {
            'model_path': self.save_model(),
            'config': {
                'framework': self.config.framework,
                'input_neurons': self.config.input_neurons,
                'output_neurons': self.config.output_neurons,
                'model_name': self.config.model_name
            },
            'training_results': self.training_history,
            'export_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Modello esportato per analisi: {output_path}")
        return output_path


def main():
    """Esempio di utilizzo SNNTrainer"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== SNN Trainer - Modulo 2 ===\n")
    
    # Configurazione training
    config = TrainingConfig(
        framework="mock",  # Usa mock per test senza dipendenze
        hidden_layers=[32, 16],
        epochs=50,
        batch_size=16,
        learning_rate=0.001
    )
    
    trainer = SNNTrainer(config)
    
    try:
        # Simula training con dati di esempio
        # In pratica si userebbe: trainer.train("path/to/snn_dataset.csv")
        
        # Per test, generiamo dati casuali
        X_dummy = np.random.random((100, 10))
        y_dummy = np.random.randint(0, 2, 100)
        
        # Training
        results = trainer.train("dummy_data")  # Normalmente path CSV
        
        print("=== Risultati Training ===")
        print(f"Framework: {results['training_history']['framework']}")
        print(f"Final loss: {results['training_history']['final_loss']:.4f}")
        print(f"Val accuracy: {results['validation_metrics']['accuracy']:.3f}")
        
        # Salva modello
        model_path = trainer.save_model("output/models")
        print(f"Modello salvato: {model_path}")
        
        # Export per analisi
        export_path = trainer.export_for_analysis("output/model_export.json")
        print(f"Export per analisi: {export_path}")
        
    except Exception as e:
        logger.error(f"Errore nel training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
