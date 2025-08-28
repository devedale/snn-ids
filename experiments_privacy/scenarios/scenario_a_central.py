# -*- coding: utf-8 -*-
"""Scenario A: Centralizzato baseline con Hyperband su mlp_4_layer.
Usa direttamente benchmark.py per semplicità/replicabilità.
"""

import os
import json
import subprocess
from typing import Dict, Any


def run(hb_max_epochs: int, hb_final_epochs: int, hb_batch_size: int) -> Dict[str, Any]:
    cmd = [
        "python3", "benchmark.py",
        "--hyperband",
        "--model", "mlp_4_layer",
        "--hb-max-epochs", str(hb_max_epochs),
        "--hb-final-epochs", str(hb_final_epochs),
        "--hb-batch-size", str(hb_batch_size),
        "--output-dir", "experiments_privacy/results/A_central"
    ]
    print("Eseguo:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # cerca l'ultimo json in output dir
    out_dir = "experiments_privacy/results/A_central"
    files = sorted([f for f in os.listdir(out_dir) if f.endswith('.json')])
    if not files:
        return {"scenario": "A", "status": "no_results"}
    with open(os.path.join(out_dir, files[-1]), 'r') as f:
        data = json.load(f)
    data["scenario"] = "A"
    return data


