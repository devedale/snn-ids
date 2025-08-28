# -*- coding: utf-8 -*-
"""Scenario B: Federated Learning semplice (FedAvg), nessuna protezione.
Usa la simulazione interna (no server Flower reale per semplicità).
"""

from typing import Dict, Any
from experiments_privacy.shared.fl_utils import FLConfig, simulate_fl_pipeline


def run(num_clients: int, rounds: int, local_epochs: int, batch_size: int, builder_kwargs: dict = None, he_pmd: int = 8192) -> Dict[str, Any]:
    cfg = FLConfig(
        model_type="mlp_4_layer",
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        use_dp=False,
        use_he=False,
    )
    res = simulate_fl_pipeline(cfg, builder_kwargs=builder_kwargs, he_poly_modulus_degree=he_pmd)
    res["scenario"] = "B"
    return res


