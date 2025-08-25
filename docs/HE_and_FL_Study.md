### Studio: Homomorphic Encryption (HE) e Federated Learning (FL) per SNN-IDS

Questo documento riassume teoria, formule essenziali, metriche e indici di confronto utilizzati negli esperimenti del progetto per HE e FL, con riferimento ai file di esecuzione (`benchmark-best-config.py`, `benchmark.py`) e agli artefatti prodotti (visualizzazioni e CSV di riepilogo).

#### Homomorphic Encryption (HE) – CKKS (float-valued)
- Obiettivo: consentire operazioni aritmetiche su dati cifrati. In CKKS i vettori reali \(\mathbf{x}\) sono codificati e cifrati in un ciphertext \(c = \text{Enc}(\mathbf{x})\). Operazioni supportate: somma e prodotto scalare approssimati.
- Proprietà: per vettori \(\mathbf{x}, \mathbf{y}\) e scalare \(a\):
  - \(\text{Dec}(c_x + c_y) \approx \mathbf{x} + \mathbf{y}\)
  - \(\text{Dec}(a \cdot c_x) \approx a \cdot \mathbf{x}\)
- Parametri principali: `poly_modulus_degree`, `coeff_mod_bit_sizes`, `scale`. Maggiore sicurezza ⇒ più overhead computazionale e memoria.
- Nel progetto: cifriamo solo feature sensibili (ottetti IP e porte) per massimizzare la confidenzialità minimizzando l’overhead. L’aggregazione dei gradienti/pesi può avvenire su ciphertext (mock/CKKS reale via TenSEAL).

#### Differential Privacy (DP) (opzionale)
- DP sui gradienti/pesi: aggiunta di rumore calibrato (es. Gaussian) alle statistiche condivise. Per una funzione \(f\) con sensibilità \(\Delta f\), il meccanismo gaussiano rilascia \(\tilde{f} = f + \mathcal{N}(0, \sigma^2)\) dove \(\sigma \propto \Delta f / \varepsilon\). Parametri: \(\varepsilon\) (privacy budget), \(\delta\). Nel codice: `dp_noise_scale` controlla l’entità del rumore (semplificato).

#### Federated Learning (FL) – FedAvg
- Dati distribuiti su client; ogni round:
  1) i client partono da pesi globali \(w_t\), addestrano localmente e producono \(w_t^{(k)}\)
  2) server aggrega: \(w_{t+1} = \sum_k p_k \cdot w_t^{(k)}\) (media pesata; nel nostro caso semplice media).
- Parametri: `num_clients`, `num_rounds`, `local_epochs`, `client_lr`/batch. Trade-off: più round/epoche ⇒ migliori prestazioni ma tempi maggiori.

#### Indici e Metriche di Confronto
- Accuratezza (global): frazione di predizioni corrette.
- F1 macro: media semplice degli F1 per classe; misura robusta in presenza di classi sbilanciate.
- Tempo di training (s): include round FL e fine-tuning finale.
- Confidenzialità (qualitativo): numero di feature protette, schema HE, eventuale DP.
- Overhead HE: confronto tempi e, se utile, memoria (non incluso ora) tra HE e no-HE a pari iperparametri.

#### Esperimenti supportati
1) Best-Config centralizzato (K-Fold): `benchmark-best-config.py` (sezioni baseline/fine-tuning). Output: report completi e top-10.
2) Confronto HE vs NO-HE in modalità FL (sweep): `python3 benchmark-best-config.py --federated-sweep [--he] [--dp]`
   - Grid snella predefinita: lr ∈ {0.008, 0.01, 0.012}, units ∈ {56, 64, 72}, epochs=3, batch=32.
   - Output: `best_config_benchmark_<ts>/federated_sweep_comparison.json` e `federated_sweep_summary.csv` con colonne chiave: he_enabled, learning_rate, gru_units, epochs, batch_eff, training_time_s, accuracy, f1_macro, eval_dir.
   - Visual: per ogni run, cartella `benchmark_results/<ts>_GRU_FL_{HE|NOHE}_.../`.

#### Come interpretare i risultati
- Confronto pari iperparametri: filtrare il CSV su (lr, units, epochs, batch_eff) e confrontare `accuracy`, `f1_macro` e `training_time_s` tra he_enabled=false/true.
- Indici consigliati:
  - Δ accuracy = acc_HE − acc_NOHE
  - Δ F1 macro = f1_HE − f1_NOHE
  - Overhead tempo = time_HE / time_NOHE (o differenza assoluta)

#### Note pratiche
- Per esperimenti HE focalizzati (senza FL): usare i runner centralizzati (best-config/progressive) e replicare la logica di confronto sugli stessi iperparametri; valutare l’integrazione HE a livello di feature preprocessing/aggregazione.
- Per FL più “realistico”: considerare evaluation client-based (leave-one-client-out) e non-IID splits.


