# -*- coding: utf-8 -*-
"""Scenario E: FL con DP semplificata (clipping+rumore) + stima HE (aggregazione)."""

from typing import Dict, Any
from experiments_privacy.shared.fl_utils import FLConfig, simulate_fl_pipeline
from experiments_privacy.shared.hp_utils import load_best_hyperparams_for_model

from .scenario_d_fl_dp import _approximate_epsilon


def run(
    num_clients: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    builder_kwargs: dict = None,
    dp_noise: float = None,
    dp_clip: float = None,
    he_pmd: int = 8192
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
        use_he=True,
        dp_noise_multiplier=0.8 if dp_noise is None else float(dp_noise),
        dp_l2_clip=1.0 if dp_clip is None else float(dp_clip),
        dp_delta=1e-5,
    )
    res = simulate_fl_pipeline(cfg, builder_kwargs=eff_builder, he_poly_modulus_degree=he_pmd)
    steps = int(rounds) * int(num_clients) * int(cfg.local_epochs)
    dataset_size_proxy = max(10000, int(cfg.batch_size) * int(num_clients) * 100)
    epsilon = _approximate_epsilon(cfg.dp_noise_multiplier, steps, int(cfg.batch_size), dataset_size_proxy, cfg.dp_delta)
    res["epsilon"] = epsilon
    res["delta"] = cfg.dp_delta
    res["scenario"] = "E"
    return res


