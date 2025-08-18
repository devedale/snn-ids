# -*- coding: utf-8 -*-

"""
Script per la Generazione di un Dataset Sintetico di Cybersecurity.

Questo script crea un file CSV contenente dati di rete simulati,
utili per testare la pipeline di machine learning.
"""

import pandas as pd
from faker import Faker
import random
import os

def generate_synthetic_data(num_rows=1000, output_path="data/cybersecurity_data.csv"):
    """
    Genera dati sintetici e li salva in un file CSV.

    Args:
        num_rows (int): Il numero di righe di dati da generare.
        output_path (str): Il percorso dove salvare il file CSV.
    """
    print(f"Inizio la generazione di {num_rows} righe di dati sintetici...")

    # Inizializza Faker per generare dati finti
    fake = Faker()

    # Liste di valori possibili per i campi categorici
    protocols = ["TCP", "UDP", "ICMP"]
    attack_types = ["normal", "dos", "probe", "r2l", "u2r"]
    # Pesi per rendere 'normal' più comune
    attack_weights = [0.7, 0.1, 0.1, 0.05, 0.05]

    # Crea la lista di dati
    data = []
    for _ in range(num_rows):
        record = {
            "ip_sorgente": fake.ipv4(),
            "ip_destinazione": fake.ipv4(),
            "porta_sorgente": random.randint(1024, 65535),
            "porta_destinazione": random.choice([80, 443, 22, 21, 53, random.randint(1024, 65535)]),
            "protocollo": random.choice(protocols),
            "byte_inviati": random.randint(60, 1500),
            "byte_ricevuti": random.randint(60, 3000),
            "pacchetti_inviati": random.randint(1, 20),
            "pacchetti_ricevuti": random.randint(1, 30),
            "tipo_attacco": random.choices(attack_types, weights=attack_weights, k=1)[0]
        }
        data.append(record)

    # Crea un DataFrame pandas
    df = pd.DataFrame(data)

    # Assicura che la directory di output esista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salva il DataFrame in un file CSV
    df.to_csv(output_path, index=False)

    print(f"Dataset sintetico salvato con successo in: {output_path}")
    print("Colonne generate:", list(df.columns))

if __name__ == '__main__':
    generate_synthetic_data()
