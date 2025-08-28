# -*- coding: utf-8 -*-
"""Scenario C: FL con stima HE (CKKS via TenSEAL) per confidenzialità in transito.
Addestramento identico a B, ma con stima di banda/overhead dei pesi cifrati.
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
        use_he=True,
    )
    res = simulate_fl_pipeline(cfg, builder_kwargs=builder_kwargs, he_poly_modulus_degree=he_pmd)
    res["scenario"] = "C"
    return res


