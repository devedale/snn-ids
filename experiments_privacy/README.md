## Privacy Experiments (A–E)

Questa cartella contiene una suite minimale e riproducibile per eseguire i 5 scenari:

- A: Central (baseline)
- B: Federated Learning (FL)
- C: FL + Homomorphic Encryption (HE, CKKS via TenSEAL) [stima overhead/banda]
- D: FL + Differential Privacy (DP) semplificata (clipping+rumore su pesi)
- E: FL + HE + DP (come sopra)

Nota: la pipeline dell’esperimento A sfrutta direttamente `benchmark.py --hyperband` con `mlp_4_layer` per ricavare gli iperparametri migliori, che poi vengono riutilizzati negli scenari B–E per massimizzare semplicità e coerenza.

### Requisiti

Installa i requisiti aggiuntivi per HE e strumenti di report:

```bash
pip install -r experiments_privacy/requirements_experiments.txt
```

Richiede anche i requisiti di progetto già previsti in `requirements.txt` (TensorFlow, scikit-learn, ecc.).

### Esecuzione rapida

Per lanciare tutti gli scenari con un solo comando e ottenere un riepilogo finale:

```bash
python3 experiments_privacy/run_all.py \
  --hb-max-epochs 20 --hb-final-epochs 30 --hb-batch-size 12 \
  --num-clients 5 --rounds 10 --local-epochs 2 --batch-size 12
```

Parametri principali:
- `--hb-*`: controllano l’Hyperband (scenario A) su `mlp_4_layer`.
- `--num-clients`, `--rounds`, `--local-epochs`, `--batch-size`: controllano la simulazione FL (scenari B–E).

Output:
- Risultati e report JSON in `experiments_privacy/results/`.
- Stima banda/overhead e, quando applicabile, stima (ε, δ) per DP.

### Struttura

- `shared/`: utilità comuni (caricamento dati, builder modello MLP, simulazione FL con opzioni DP/HE).
- `scenarios/`: script specifici per ciascun scenario A–E.
- `run_all.py`: orchestratore che esegue in sequenza A→E e genera una tabella riassuntiva.

### Note su DP/HE

- DP: implementazione semplificata senza dipendenze extra (niente TensorFlow Privacy), applicando clipping L2 e rumore gaussiano ai pesi locali prima dell’aggregazione. Viene stimato un ε approssimato (indicativo) in funzione di rumore e passi.
- HE: viene stimato l’overhead di banda con TenSEAL (CKKS) su vettori di pesi campionati e un esempio di somma omomorfica; l’addestramento FL in Flower rimane in chiaro per massima semplicità operativa, ma include le stime realistiche di trasferimento se si cifrassero gli aggiornamenti.


