# -*- coding: utf-8 -*-
"""Scenario C: FL con stima HE (CKKS via TenSEAL) per confidenzialità in transito.
Addestramento identico a B, ma con stima di banda/overhead dei pesi cifrati.
"""

from typing import Dict, Any
from experiments_privacy.shared.fl_utils import FLConfig, simulate_fl_pipeline
from experiments_privacy.shared.hp_utils import load_best_hyperparams_for_model


def run(num_clients: int, rounds: int, local_epochs: int, batch_size: int, builder_kwargs: dict = None, he_pmd: int = 8192) -> Dict[str, Any]:
    best_builder, best_train = load_best_hyperparams_for_model("mlp_4_layer")
    eff_builder = {**(best_builder or {}), **(builder_kwargs or {})}

    cfg = FLConfig(
        model_type="mlp_4_layer",
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs if local_epochs is not None else int(best_train.get('epochs', 2)),
        batch_size=batch_size if batch_size is not None else int(best_train.get('batch_size', 64)),
        learning_rate=float(eff_builder.get('learning_rate', 0.001)),
        use_dp=False,
        use_he=True,
    )
    res = simulate_fl_pipeline(cfg, builder_kwargs=eff_builder, he_poly_modulus_degree=he_pmd)
    res["scenario"] = "C"
    return res


