# -*- coding: utf-8 -*-

"""
Script per la Generazione di un Dataset Sintetico di Cybersecurity con Timestamp.
"""

import pandas as pd
from faker import Faker
import random
import os
from datetime import datetime, timedelta

# Importa le configurazioni
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from config import DATA_CONFIG

def generate_synthetic_data(num_rows=2000, output_path=None):
    """
    Genera dati sintetici di rete, inclusa una colonna timestamp, e li salva in un CSV.

    Args:
        num_rows (int): Numero di righe da generare.
        output_path (str, optional): Percorso di output. Se None, usa quello da config.
    """
    if output_path is None:
        output_path = DATA_CONFIG["dataset_path"]

    print(f"Inizio la generazione di {num_rows} righe di dati sintetici...")

    fake = Faker()
    protocols = ["TCP", "UDP", "ICMP"]
    attack_types = ["normal", "dos", "probe", "r2l", "u2r"]
    attack_weights = [0.7, 0.15, 0.1, 0.03, 0.02]

    data = []
    # Inizia da un timestamp fisso e lo incrementa per ogni record
    current_time = datetime.now()

    for _ in range(num_rows):
        # Incrementa il timestamp di un intervallo casuale (da 1 a 10 secondi)
        current_time += timedelta(seconds=random.randint(1, 10))

        record = {
            DATA_CONFIG["timestamp_column"]: current_time.isoformat(),
            "ip_sorgente": fake.ipv4(),
            "ip_destinazione": fake.ipv4(),
            "porta_sorgente": random.randint(1024, 65535),
            "porta_destinazione": random.choice([80, 443, 22, 21, 53, random.randint(1024, 65535)]),
            "protocollo": random.choice(protocols),
            "byte_inviati": random.randint(60, 1500),
            "byte_ricevuti": random.randint(0, 5000), # Permettiamo 0 byte ricevuti
            "tipo_attacco": random.choices(attack_types, weights=attack_weights, k=1)[0]
        }
        data.append(record)

    df = pd.DataFrame(data)

    # Assicura che la directory di output esista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salva il DataFrame in un file CSV
    df.to_csv(output_path, index=False)

    print(f"Dataset sintetico salvato con successo in: {output_path}")

if __name__ == '__main__':
    generate_synthetic_data()
