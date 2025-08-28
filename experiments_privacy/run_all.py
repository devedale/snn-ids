#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ensure project root on path
"""
Orchestratore scenari A–E con parametri semplici e seed unificato.
Esegue:
 A: Central (Hyperband su mlp_4_layer via benchmark.py)
 B: FL
 C: FL + HE (stima banda)
 D: FL + DP (epsilon, delta)
 E: FL + HE + DP
"""

import argparse
from typing import List, Dict, Any

from experiments_privacy.scenarios.scenario_a_central import run as run_a
from experiments_privacy.scenarios.scenario_b_fl import run as run_b
from experiments_privacy.scenarios.scenario_c_fl_he import run as run_c
from experiments_privacy.scenarios.scenario_d_fl_dp import run as run_d
from experiments_privacy.scenarios.scenario_e_fl_he_dp import run as run_e
from experiments_privacy.shared.summary import summarize_and_save


def main():
    parser = argparse.ArgumentParser(description="Run privacy scenarios A–E")
    parser.add_argument("--hb-max-epochs", type=int, default=20)
    parser.add_argument("--hb-final-epochs", type=int, default=30)
    parser.add_argument("--hb-batch-size", type=int, default=12)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--dp-noise", type=float, default=None, help="Noise multiplier per DP (override)")
    parser.add_argument("--dp-clip", type=float, default=None, help="L2 clip per DP (override)")
    parser.add_argument("--he-pmd", type=int, default=8192, help="poly_modulus_degree per stima HE (8192/16384)")
    args = parser.parse_args()

    os.makedirs("experiments_privacy/results", exist_ok=True)

    results: List[Dict[str, Any]] = []

    # Scenario A
    try:
        res_a = run_a(args.hb_max_epochs, args.hb_final_epochs, args.hb_batch_size)
        results.append(res_a)
    except Exception as e:
        results.append({"scenario": "A", "status": "error", "error": str(e)})

    # Estraggo hp migliori da A (se disponibili) per passaggio a B–E
    builder_kwargs = {}
    try:
        hp = (res_a or {}).get("config", {}).get("hyperparameters", {})
        # mappa parametri noti per mlp_4_layer
        if isinstance(hp, dict):
            # accetta sia liste che scalari
            def pick(k, default=None):
                v = hp.get(k, default)
                return v[0] if isinstance(v, list) else v
            if "units_layer_1" in hp or "hidden_layer_units" in hp:
                if "hidden_layer_units" in hp:
                    units = pick("hidden_layer_units") or [256,128,64,32]
                    builder_kwargs.update({
                        "units_layer_1": units[0],
                        "units_layer_2": units[1],
                        "units_layer_3": units[2],
                        "units_layer_4": units[3],
                    })
                else:
                    builder_kwargs.update({
                        "units_layer_1": pick("units_layer_1", 256),
                        "units_layer_2": pick("units_layer_2", 128),
                        "units_layer_3": pick("units_layer_3", 64),
                        "units_layer_4": pick("units_layer_4", 32),
                    })
            if "activation" in hp:
                builder_kwargs["activation"] = pick("activation", "relu")
            if "learning_rate" in hp:
                builder_kwargs["learning_rate"] = pick("learning_rate", 0.001)
    except Exception:
        builder_kwargs = {}

    # Scenario B
    try:
        res_b = run_b(args.num_clients, args.rounds, args.local_epochs, args.batch_size, builder_kwargs, he_pmd=args.he_pmd)
        results.append(res_b)
    except Exception as e:
        import traceback; traceback.print_exc()
        results.append({"scenario": "B", "status": "error", "error": str(e)})

    # Scenario C
    try:
        res_c = run_c(args.num_clients, args.rounds, args.local_epochs, args.batch_size, builder_kwargs, he_pmd=args.he_pmd)
        results.append(res_c)
    except Exception as e:
        import traceback; traceback.print_exc()
        results.append({"scenario": "C", "status": "error", "error": str(e)})

    # Scenario D
    try:
        res_d = run_d(
            args.num_clients, args.rounds, args.local_epochs, args.batch_size,
            builder_kwargs, dp_noise=args.dp_noise, dp_clip=args.dp_clip
        )
        results.append(res_d)
    except Exception as e:
        import traceback; traceback.print_exc()
        results.append({"scenario": "D", "status": "error", "error": str(e)})

    # Scenario E
    try:
        res_e = run_e(
            args.num_clients, args.rounds, args.local_epochs, args.batch_size,
            builder_kwargs, dp_noise=args.dp_noise, dp_clip=args.dp_clip, he_pmd=args.he_pmd
        )
        results.append(res_e)
    except Exception as e:
        import traceback; traceback.print_exc()
        results.append({"scenario": "E", "status": "error", "error": str(e)})

    # Salva riepilogo
    summarize_and_save(results, "experiments_privacy/results")

    print("\nTabella riassuntiva pronta in experiments_privacy/results/summary.csv")


if __name__ == "__main__":
    main()


