# -*- coding: utf-8 -*-
"""Scenario D: FL con Differential Privacy semplificata (clipping + rumore su pesi).
Calcola una stima approssimata di (ε, δ) per indicare il tradeoff priv-acc.
"""

from typing import Dict, Any
from experiments_privacy.shared.fl_utils import FLConfig, simulate_fl_pipeline
from experiments_privacy.shared.hp_utils import load_best_hyperparams_for_model


def _approximate_epsilon(noise_multiplier: float, steps: int, batch_size: int, dataset_size: int, delta: float) -> float:
    """Stima molto semplice di epsilon (non rigorosa, ma informativa)."""
    # Questa è una formula indicativa (RDP semplificata non implementata qui per brevità)
    # epsilon ~ O( steps * batch_size / dataset_size / noise^2 )
    denom = max(noise_multiplier ** 2, 1e-6)
    q = batch_size / max(dataset_size, 1)
    return float( (steps * q) / denom )


def run(
    num_clients: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    builder_kwargs: dict = None,
    dp_noise: float = None,
    dp_clip: float = None
) -> Dict[str, Any]:
    best_builder, best_train = load_best_hyperparams_for_model("mlp_4_layer")
    eff_builder = {**(best_builder or {}), **(builder_kwargs or {})}

    cfg = FLConfig(
        model_type="mlp_4_layer",
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs if local_epochs is not None else int(best_train.get('epochs', 2)),
        batch_size=batch_size if batch_size is not None else int(best_train.get('batch_size', 64)),
        learning_rate=float(eff_builder.get('learning_rate', 0.001)),
        use_dp=True,
        use_he=False,
        dp_noise_multiplier=0.8 if dp_noise is None else float(dp_noise),
        dp_l2_clip=1.0 if dp_clip is None else float(dp_clip),
        dp_delta=1e-5,
    )
    res = simulate_fl_pipeline(cfg, builder_kwargs=eff_builder)
    # stima epsilon usando dimensioni del dataset ricavate indirettamente
    # NB: usa i valori effettivi in cfg per evitare None da args
    steps = int(rounds) * int(num_clients) * int(cfg.local_epochs)
    dataset_size_proxy = max(10000, int(cfg.batch_size) * int(num_clients) * 100)  # proxy conservativo
    epsilon = _approximate_epsilon(cfg.dp_noise_multiplier, steps, int(cfg.batch_size), dataset_size_proxy, cfg.dp_delta)
    res["epsilon"] = epsilon
    res["delta"] = cfg.dp_delta
    res["scenario"] = "D"
    return res


