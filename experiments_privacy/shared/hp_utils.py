# -*- coding: utf-8 -*-
"""Utility per caricare e normalizzare gli iperparametri migliori trovati in Scenario A.

Ricerca l'ultimo file JSON in `experiments_privacy/results/A_central` e ne estrae:
- hyperparameters effettivi usati nella run finale di Hyperband (se presenti)
- fallback: da `config.TRAINING_CONFIG` se i campi non sono nel JSON

Espone funzioni per trasformare la struttura HP di Hyperband (flat) in `builder_kwargs`
compatibili con `src/training/models.py` (es.: `units_layer_1`, ...), e per estrarre
`epochs`, `batch_size`, `learning_rate` se disponibili.
"""

import os
import json
from typing import Dict, Any, Tuple

from config import TRAINING_CONFIG


RESULTS_DIR = os.path.join("experiments_privacy", "results", "A_central")


def _read_latest_json(results_dir: str = RESULTS_DIR) -> Dict[str, Any]:
    try:
        files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if not files:
            return {}
        files.sort()
        with open(os.path.join(results_dir, files[-1]), 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _extract_hp_from_results(data: Dict[str, Any]) -> Dict[str, Any]:
    """Prova a individuare un dict di iperparametri dai risultati del benchmark."""
    # Path 1: benchmark.run_hyperband salva best_hps sotto results['config']['hyperparameters']? Dipende dalla versione
    cfg = (data or {}).get('config') or {}
    hp = cfg.get('hyperparameters') or {}
    # Se vuoto, prova evaluation->model_config
    if not hp and isinstance(data.get('evaluation'), dict):
        eval_cfg = data['evaluation'].get('model_config') or {}
        hp = eval_cfg.get('hyperparameters') or {}
    # Se ancora vuoto, ritorna {}
    return hp or {}


def load_best_hyperparams_for_model(model_type: str = "mlp_4_layer") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Ritorna una tupla (builder_kwargs, train_kwargs).

    - builder_kwargs: parametri da passare al costruttore del modello (es. units_layer_1, activation, learning_rate)
    - train_kwargs: parametri di training come epochs, batch_size (se disponibili)

    Fallback: usa TRAINING_CONFIG se non trovati nel JSON.
    """
    data = _read_latest_json()
    hp_all = _extract_hp_from_results(data)

    # Struttura può essere già mista {common:{}, model_specific:{}} oppure flat (da Hyperband)
    if 'common' in hp_all or 'model_specific' in hp_all:
        common = hp_all.get('common', {})
        specific = (hp_all.get('model_specific', {}) or {}).get(model_type, {})
        merged = {**common, **specific}
    else:
        merged = dict(hp_all)

    # Estrai builder kwargs per mlp_4_layer
    builder_kwargs: Dict[str, Any] = {}
    train_kwargs: Dict[str, Any] = {}

    # Mappe attese dal costruttore in src/training/models.build_mlp_4_layer_model
    # Possibili chiavi da Hyperband: units_layer_1..4, activation, learning_rate
    for k in [
        'units_layer_1', 'units_layer_2', 'units_layer_3', 'units_layer_4',
        'activation', 'learning_rate'
    ]:
        if k in merged:
            builder_kwargs[k] = merged[k][0] if isinstance(merged[k], list) else merged[k]

    # Training keys
    for k in ['epochs', 'batch_size']:
        if k in merged:
            train_kwargs[k] = merged[k][0] if isinstance(merged[k], list) else merged[k]

    # Fallback se mancano chiavi
    default_common = TRAINING_CONFIG['hyperparameters']['common']
    default_specific = TRAINING_CONFIG['hyperparameters']['model_specific'].get(model_type, {})

    def _first_or_none(val):
        return val[0] if isinstance(val, list) and val else val

    # Default builder kwargs
    if 'activation' not in builder_kwargs and 'activation' in default_specific:
        builder_kwargs['activation'] = _first_or_none(default_specific['activation'])
    if 'learning_rate' not in builder_kwargs:
        # Prefer model-specific LR, else common LR
        lr_spec = default_specific.get('learning_rate')
        lr_comm = default_common.get('learning_rate')
        builder_kwargs['learning_rate'] = _first_or_none(lr_spec) if lr_spec is not None else _first_or_none(lr_comm)

    # MLP layers default
    if model_type == 'mlp_4_layer':
        if not any(k in builder_kwargs for k in ['units_layer_1','units_layer_2','units_layer_3','units_layer_4']):
            # Usa hidden_layer_units se presente
            hlu = default_specific.get('hidden_layer_units')
            if hlu:
                units = _first_or_none(hlu) or [256,128,64,32]
                if isinstance(units, (list, tuple)) and len(units) == 4:
                    builder_kwargs.update({
                        'units_layer_1': units[0],
                        'units_layer_2': units[1],
                        'units_layer_3': units[2],
                        'units_layer_4': units[3],
                    })

    # Default training kwargs
    if 'epochs' not in train_kwargs:
        ep_spec = default_specific.get('epochs')
        train_kwargs['epochs'] = int(_first_or_none(ep_spec) or 10)
    if 'batch_size' not in train_kwargs:
        bs_spec = default_specific.get('batch_size')
        bs_comm = default_common.get('batch_size')
        train_kwargs['batch_size'] = int(_first_or_none(bs_spec) or _first_or_none(bs_comm) or 64)

    return builder_kwargs, train_kwargs


