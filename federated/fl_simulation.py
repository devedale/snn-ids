# -*- coding: utf-8 -*-
"""
Simulatore semplice di Federated Learning (FL) con supporto opzionale HE.

Modello di federazione: FedAvg.
Si assume che il training locale sia eseguito chiamando `training.train.train_model`
con subset di dati per ciascun client. Per semplicità, qui eseguiamo epoche locali
ridotte e emuliamo l'aggregazione dei pesi con media.

Nota: per reti Keras, i pesi sono liste di array. HE è applicato solo su vettori
di gradienti selezionati o, in alternativa, su feature sensibili in pre-processing
(qui forniamo hook per cifrare e sommare vettori, non tensori arbitrari).
"""

from typing import Dict, Any, List, Tuple
import numpy as np

from federated.he import HEContext, HEEncryptor


def split_dataset_iid(X: np.ndarray, y: np.ndarray, num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split IID semplice per simulazione locale."""
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    splits = np.array_split(idx, num_clients)
    return [(X[s], y[s]) for s in splits]


class FedAvgServer:
    def __init__(self, base_weights: List[np.ndarray], he_context: HEContext = None):
        self.global_weights = [w.copy() for w in base_weights]
        self.he_ctx = he_context or HEContext({"enabled": False})
        self.encryptor = HEEncryptor(self.he_ctx)

    def aggregate(self, client_weight_updates: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Media pesata semplice dei pesi (stessa dimensione)."""
        num_clients = len(client_weight_updates)
        new_weights = []
        for layer_idx in range(len(self.global_weights)):
            layer_stack = np.stack([cw[layer_idx] for cw in client_weight_updates], axis=0)
            layer_mean = np.mean(layer_stack, axis=0)
            new_weights.append(layer_mean)
        self.global_weights = new_weights
        return self.global_weights


class FedClient:
    def __init__(self, client_id: int, local_data: Tuple[np.ndarray, np.ndarray]):
        self.client_id = client_id
        self.X, self.y = local_data

    def local_train(self, build_model_fn, base_weights: List[np.ndarray], params: Dict[str, Any]) -> List[np.ndarray]:
        """Esegue un brevissimo fine-tuning locale e restituisce i pesi aggiornati."""
        import tensorflow as tf

        # Ricostruisci il modello coerente con i pesi globali
        input_shape = self.X.shape[1:] if len(self.X.shape) > 2 else (self.X.shape[1],)
        num_classes = max(len(np.unique(self.y)), np.max(self.y) + 1)
        model = build_model_fn(input_shape, num_classes)
        model.set_weights(base_weights)

        # Parametri locali ridotti
        epochs = params.get("local_epochs", 1)
        batch_size = params.get("batch_size", 32)

        model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, verbose=0)
        return model.get_weights()


