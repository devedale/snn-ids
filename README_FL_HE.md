### Federated Learning + Homomorphic Encryption (SNN-IDS)

Esecuzioni supportate:

- Modalità standard best-config (già esistente):
  ```bash
  python3 benchmark-best-config.py
  ```

- Modalità Federated Learning:
  ```bash
  python3 benchmark-best-config.py --federated
  ```

- Federated Learning con Homomorphic Encryption (feature sensibili: IP ottetti e porte):
  ```bash
  python3 benchmark-best-config.py --federated --he
  ```

- Abilitare anche Differential Privacy sui gradienti:
  ```bash
  python3 benchmark-best-config.py --federated --he --dp
  ```

Output principali:
- Cartella `best_config_benchmark_<timestamp>/`
  - `federated_best_config_report.json` con accuracy e privacy report.
  - Gli artefatti del benchmark standard restano invariati.

Note HE:
- Se `tenseal` non è installato, viene usato un mock HE con stessa API per simulare overhead/aggregazione.
  Per usare CKKS reale, installare TenSEAL (`pip install tenseal`) e impostare `HOMOMORPHIC_CONFIG.enabled = True` o usare `--he`.


