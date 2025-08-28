# -*- coding: utf-8 -*-
"""Utility per tabellare e salvare risultati degli scenari A–E."""

import os
import json
from typing import Dict, Any, List
import pandas as pd


def summarize_and_save(results: List[Dict[str, Any]], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    # Trova baseline A (se presente) per confronto accuracy
    baseline_acc = None
    for r in results:
        if r.get("scenario") == "A":
            baseline_acc = r.get("validation_accuracy") or r.get("final_accuracy")
            break
    for r in results:
        scenario = r.get("scenario", "?")
        status = r.get("status", "success")
        error_msg = r.get("error")
        acc = r.get("validation_accuracy") or r.get("final_accuracy")
        training_time = r.get("training_time")
        total_time = r.get("total_time")
        eps = r.get("epsilon")
        delta = r.get("delta")
        bandwidth_mb = None
        he_total_mb = None
        round_count = r.get("rounds")
        num_clients = r.get("num_clients")
        use_dp = r.get("use_dp")
        use_he = r.get("use_he")
        round_durations = r.get("round_durations_s") or []
        round_time_mean = float(sum(round_durations) / len(round_durations)) if round_durations else None
        round_time_std = float((
            (sum([(x - (round_time_mean or 0.0)) ** 2 for x in round_durations]) / max(1, len(round_durations) - 1)) ** 0.5
        )) if round_durations else None
        if r.get("he_round_overheads"):
            # media approssimata dei megabytes per round
            vals = [x.get("approx_cipher_megabytes") for x in r["he_round_overheads"] if x.get("he_available")]
            modes = [x.get("mode") for x in r["he_round_overheads"] if x.get("he_available")]
            enc_times = [x.get("enc_time_s_total") for x in r["he_round_overheads"] if x.get("enc_time_s_total") is not None]
            agg_times = [x.get("agg_time_s") for x in r["he_round_overheads"] if x.get("agg_time_s") is not None]
            dec_times = [x.get("dec_time_s") for x in r["he_round_overheads"] if x.get("dec_time_s") is not None]
            if vals:
                bandwidth_mb = float(sum(vals) / len(vals))
                he_total_mb = float(sum([v for v in vals if v is not None]))
            he_mode = None
            if modes:
                # Preferisci 'encrypt' se presente, altrimenti primo disponibile
                he_mode = "encrypt" if any(m == "encrypt" for m in modes) else modes[0]
        else:
            he_mode = None
        acc_diff_vs_A = (None if (baseline_acc is None or acc is None) else float(acc) - float(baseline_acc))
        he_bandwidth_per_sec = None
        if he_total_mb is not None and total_time:
            try:
                he_bandwidth_per_sec = he_total_mb / float(total_time)
            except Exception:
                he_bandwidth_per_sec = None
        rows.append({
            "Scenario": scenario,
            "Status": status,
            "Error": error_msg,
            "Accuracy": acc,
            "TrainingTime_s": training_time,
            "TotalTime_s": total_time,
            "Epsilon": eps,
            "Delta": delta,
            "HE_MB_per_round": bandwidth_mb,
            "HE_MB_total": he_total_mb,
            "HE_MB_per_sec": he_bandwidth_per_sec,
            "HE_Mode": he_mode,
            "HE_EncTimeMean_s": (float(sum(enc_times)/len(enc_times)) if 'enc_times' in locals() and enc_times else None),
            "HE_AggTimeMean_s": (float(sum(agg_times)/len(agg_times)) if 'agg_times' in locals() and agg_times else None),
            "HE_DecTimeMean_s": (float(sum(dec_times)/len(dec_times)) if 'dec_times' in locals() and dec_times else None),
            "Rounds": round_count,
            "NumClients": num_clients,
            "UseDP": use_dp,
            "UseHE": use_he,
            "RoundTimeMean_s": round_time_mean,
            "RoundTimeStd_s": round_time_std,
            "Acc_vs_A": acc_diff_vs_A,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "summary.csv")
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)

    print(f"Riepilogo salvato in: {csv_path} e {json_path}")
    return csv_path


