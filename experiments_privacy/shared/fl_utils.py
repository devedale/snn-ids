# -*- coding: utf-8 -*-
"""
Utility per eseguire simulazioni FL semplici con Flower, opzionalmente con:
- Differential Privacy (DP) via TensorFlow Privacy (DP-SGD)
- Stima Homomorphic Encryption (HE) via TenSEAL (CKKS) per overhead/banda

Obiettivi: semplicità, ripetibilità (seed), modularità.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import tensorflow as tf
import time

from config import RANDOM_CONFIG
from src.preprocessing import preprocess_pipeline
from src.training.utils import prepare_data_for_model
from src.training.models import get_model_builder


def set_global_seed(seed: int):
    import random
    import numpy as np
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@dataclass
class FLConfig:
    model_type: str = "mlp_4_layer"
    num_clients: int = 5
    rounds: int = 10
    local_epochs: int = 2
    batch_size: int = 12
    learning_rate: float = 0.001
    iid: bool = True
    use_dp: bool = False
    dp_noise_multiplier: float = 0.8
    dp_l2_clip: float = 1.0
    dp_delta: float = 1e-5
    use_he: bool = False
    # HE real encryption (CKKS) controls
    he_encrypt: bool = False
    he_poly_modulus_degree: int = 8192
    he_coeff_mod_bit_sizes: tuple = (60, 40, 40, 60)
    he_global_scale_bits: int = 40


def build_compiled_model(
    model_type: str,
    input_shape: Tuple[int, ...],
    num_classes: int,
    learning_rate: float,
    **builder_kwargs: Any
) -> tf.keras.Model:
    builder = get_model_builder(model_type)
    # Evita doppio passaggio di learning_rate se incluso in builder_kwargs
    lr_from_kwargs = builder_kwargs.pop("learning_rate", None)
    eff_lr = lr_from_kwargs if lr_from_kwargs is not None else learning_rate
    model = builder(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=eff_lr,
        **builder_kwargs
    )
    return model


def make_client_data_splits(X: np.ndarray, y: np.ndarray, num_clients: int, iid: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(RANDOM_CONFIG.get('seed', 42))
    indices = np.arange(len(X))
    if iid:
        rng.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        return [(X[idx], y[idx]) for idx in splits]
    # semplice non-IID: ordina per label e taglia a blocchi
    order = np.argsort(y)
    splits = np.array_split(order, num_clients)
    return [(X[idx], y[idx]) for idx in splits]


def train_local(model: tf.keras.Model, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, use_dp: bool, dp_noise_multiplier: float, dp_l2_clip: float, dp_delta: float) -> Dict[str, Any]:
    # Allenamento standard
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    weights = model.get_weights()
    if use_dp:
        # Sostituisce DP-SGD con un meccanismo semplice: clipping L2 e aggiunta di rumore gaussiano ai pesi
        clipped = []
        for w in weights:
            norm = np.linalg.norm(w)
            factor = 1.0
            if norm > dp_l2_clip and norm > 0:
                factor = dp_l2_clip / norm
            w_c = w * factor
            noise = np.random.normal(loc=0.0, scale=dp_noise_multiplier * max(1e-6, dp_l2_clip), size=w.shape)
            clipped.append(w_c + noise.astype(w.dtype))
        weights = clipped
    return {"weights": weights, "history": {k: [float(v) for v in val] for k, val in history.history.items()}}


def aggregate_fedavg(client_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
    agg = []
    for layer_weights in zip(*client_weights):
        agg.append(np.mean(np.stack(layer_weights, axis=0), axis=0))
    return agg


def estimate_he_communication_overhead(
    weights: List[np.ndarray],
    poly_modulus_degree: int = 8192
) -> Dict[str, Any]:
    """Stima semplicissima dell’overhead se cifrassimo i pesi via CKKS.
    Non cifra davvero durante FL (per semplicità), ma calcola dimensioni indicative.
    """
    try:
        import tenseal as ts
    except Exception as e:
        return {"he_available": False, "error": str(e)}

    vec = np.concatenate([w.flatten() for w in weights]).astype(np.float64)
    # Parametri ragionevoli/coerenti con config, con PMD configurabile
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, -1, coeff_mod_bit_sizes)
    context.global_scale = 2 ** 40
    context.generate_galois_keys()

    # Capacità slot CKKS ~ poly_modulus_degree/2
    slots = poly_modulus_degree // 2
    length = vec.shape[0]
    num_chunks = int(np.ceil(length / slots))
    # Cifra un solo chunk di riferimento per stimare dimensione
    sample_chunk = vec[:min(slots, length)].tolist()
    enc = ts.ckks_vector(context, sample_chunk)
    sample_size = len(enc.serialize())
    total_bytes = sample_size * num_chunks

    return {
        "he_available": True,
        "approx_cipher_bytes": int(total_bytes),
        "approx_cipher_megabytes": float(total_bytes) / (1024 * 1024),
        "poly_modulus_degree": int(poly_modulus_degree),
        "num_ciphertexts": int(num_chunks)
    }


def _flatten_weights(weights: List[np.ndarray]) -> tuple:
    """Concatena pesi in un singolo vettore e restituisce anche le shape per ricostruzione."""
    shapes = [w.shape for w in weights]
    vec = np.concatenate([w.flatten() for w in weights]).astype(np.float64)
    return vec, shapes


def _unflatten_vector(vec: np.ndarray, shapes: List[tuple]) -> List[np.ndarray]:
    """Ricostruisce lista pesi dalle shape originali."""
    rebuilt: List[np.ndarray] = []
    offset = 0
    for shp in shapes:
        size = int(np.prod(shp))
        rebuilt.append(vec[offset:offset + size].reshape(shp))
        offset += size
    return rebuilt


def _create_ckks_contexts(poly_modulus_degree: int, coeff_mod_bit_sizes: List[int], global_scale_bits: int):
    """Crea contesto CKKS TenSEAL completo e un contesto pubblico per cifrare lato client."""
    try:
        import tenseal as ts
    except Exception as e:
        raise RuntimeError(f"TenSEAL non disponibile: {e}")

    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, -1, list(coeff_mod_bit_sizes))
    ctx.global_scale = 2 ** global_scale_bits
    ctx.generate_galois_keys()

    # Crea una copia pubblica senza secret key
    public_bytes = ctx.serialize(save_secret_key=False)
    public_ctx = ts.context_from(public_bytes)
    return ctx, public_ctx


def _encrypt_weights_ckks(weights: List[np.ndarray], public_ctx, poly_modulus_degree: int):
    """Cifra i pesi concatenati in chunk CKKS; ritorna lista di ciphertext e metadati."""
    import tenseal as ts  # type: ignore
    vec, shapes = _flatten_weights(weights)
    slots = poly_modulus_degree // 2
    length = int(vec.shape[0])
    num_chunks = int(np.ceil(length / slots))
    chunks = []
    for i in range(num_chunks):
        s = i * slots
        e = min((i + 1) * slots, length)
        pt = vec[s:e].tolist()
        chunks.append(ts.ckks_vector(public_ctx, pt))
    return chunks, shapes, length


def _aggregate_cipher_chunks(clients_chunks: List[List[Any]]) -> List[Any]:
    """Somma omomorfica per-chunk: assume stesso numero di chunk tra i client."""
    if not clients_chunks:
        return []
    num_chunks = len(clients_chunks[0])
    agg: List[Any] = []
    for i in range(num_chunks):
        c = clients_chunks[0][i]
        # usa somma non in-place per evitare side-effects
        total = c
        for k in range(1, len(clients_chunks)):
            total = total + clients_chunks[k][i]
        agg.append(total)
    return agg


def _decrypt_aggregate_ckks(agg_chunks: List[Any], full_ctx, total_length: int) -> np.ndarray:
    """Decifra l'aggregato CKKS e concatena in un vettore della lunghezza attesa."""
    out: List[float] = []
    for ct in agg_chunks:
        # decrypt() restituisce lista float64
        out.extend(ct.decrypt())
        if len(out) >= total_length:
            break
    return np.array(out[:total_length], dtype=np.float64)


def simulate_fl_pipeline(
    fl_cfg: FLConfig,
    preprocess_override: Dict[str, Any] = None,
    builder_kwargs: Optional[Dict[str, Any]] = None,
    X_pre: Optional[np.ndarray] = None,
    y_pre: Optional[np.ndarray] = None,
    label_encoder_pre: Optional[Any] = None,
    he_poly_modulus_degree: int = 8192
) -> Dict[str, Any]:
    set_global_seed(RANDOM_CONFIG.get('seed', 42))

    if X_pre is None or y_pre is None or label_encoder_pre is None:
        X, y, label_encoder = preprocess_pipeline(
            data_path=(preprocess_override or {}).get('data_path'),
            sample_size=(preprocess_override or {}).get('sample_size')
        )
    else:
        X, y, label_encoder = X_pre, y_pre, label_encoder_pre
    X_prep = prepare_data_for_model(X, fl_cfg.model_type)
    num_classes = len(np.unique(y))
    input_shape = X_prep.shape[1:]

    # Inizializza modello globale con gestione univoca del learning_rate
    _kwargs_global = dict(builder_kwargs or {})
    _lr_from_kwargs = _kwargs_global.pop("learning_rate", None)
    _effective_lr = _lr_from_kwargs if _lr_from_kwargs is not None else fl_cfg.learning_rate
    model = build_compiled_model(
        fl_cfg.model_type, input_shape, num_classes, _effective_lr,
        **_kwargs_global
    )

    # Split dati fra client
    client_splits = make_client_data_splits(X_prep, y, fl_cfg.num_clients, iid=fl_cfg.iid)

    round_histories = []
    round_durations_s: List[float] = []
    t0 = time.time()
    he_overheads = []
    he_runtime_ctx = None
    he_public_ctx = None
    weight_shapes_cache: Optional[List[tuple]] = None
    flat_length_cache: Optional[int] = None
    if fl_cfg.use_he and fl_cfg.he_encrypt:
        try:
            he_runtime_ctx, he_public_ctx = _create_ckks_contexts(
                fl_cfg.he_poly_modulus_degree,
                list(fl_cfg.he_coeff_mod_bit_sizes) if isinstance(fl_cfg.he_coeff_mod_bit_sizes, (list, tuple)) else [60, 40, 40, 60],
                fl_cfg.he_global_scale_bits,
            )
        except Exception as e:
            # fallback: disabilita he_encrypt se impossibile
            he_runtime_ctx, he_public_ctx = None, None
            print("[WARN] Impossibile inizializzare HE reale, uso stima: ", e)
            
    for r in range(fl_cfg.rounds):
        r0 = time.time()
        client_weights = []
        client_cipher_chunks: List[List[Any]] = []
        enc_bytes_this_round: List[int] = []
        enc_time_s = 0.0
        for (Xc, yc) in client_splits:
            # clone pesi correnti
            _kwargs_client = dict(builder_kwargs or {})
            _lr_client = _kwargs_client.pop("learning_rate", None)
            _effective_lr_client = _lr_client if _lr_client is not None else fl_cfg.learning_rate
            client_model = build_compiled_model(
                fl_cfg.model_type, input_shape, num_classes, _effective_lr_client,
                **_kwargs_client
            )
            client_model.set_weights(model.get_weights())
            res = train_local(
                client_model, Xc, yc,
                epochs=fl_cfg.local_epochs,
                batch_size=fl_cfg.batch_size,
                use_dp=fl_cfg.use_dp,
                dp_noise_multiplier=fl_cfg.dp_noise_multiplier,
                dp_l2_clip=fl_cfg.dp_l2_clip,
                dp_delta=fl_cfg.dp_delta,
            )
            if fl_cfg.use_he and fl_cfg.he_encrypt and he_public_ctx is not None:
                t_enc0 = time.time()
                chunks, shapes, flat_len = _encrypt_weights_ckks(res["weights"], he_public_ctx, fl_cfg.he_poly_modulus_degree)
                enc_time_s += (time.time() - t_enc0)
                # calcola bytes approssimati
                try:
                    total_bytes = sum(len(ct.serialize()) for ct in chunks)
                except Exception:
                    total_bytes = 0
                enc_bytes_this_round.append(total_bytes)
                client_cipher_chunks.append(chunks)
                if weight_shapes_cache is None:
                    weight_shapes_cache = shapes
                    flat_length_cache = flat_len
            else:
                client_weights.append(res["weights"])

        # Aggregazione
        he_info: Dict[str, Any] = {"he_available": False}
        if fl_cfg.use_he and fl_cfg.he_encrypt and he_runtime_ctx is not None and client_cipher_chunks:
            # Somma omomorfica
            t_agg0 = time.time()
            agg_chunks = _aggregate_cipher_chunks(client_cipher_chunks)
            agg_time_s = time.time() - t_agg0
            # Decrittazione dell'aggregato
            t_dec0 = time.time()
            flat_sum = _decrypt_aggregate_ckks(agg_chunks, he_runtime_ctx, int(flat_length_cache or 0))
            dec_time_s = time.time() - t_dec0
            # Media FedAvg in plaintext (sull'aggregato)
            flat_mean = flat_sum / float(fl_cfg.num_clients)
            new_weights = _unflatten_vector(flat_mean, weight_shapes_cache or [])
            he_info = {
                "he_available": True,
                "mode": "encrypt",
                "approx_cipher_bytes": int(np.mean(enc_bytes_this_round)) if enc_bytes_this_round else None,
                "approx_cipher_megabytes": (np.mean(enc_bytes_this_round) / (1024 * 1024)) if enc_bytes_this_round else None,
                "poly_modulus_degree": int(fl_cfg.he_poly_modulus_degree),
                "num_ciphertexts": int(len(agg_chunks)) if agg_chunks else None,
                "enc_time_s_total": float(enc_time_s),
                "agg_time_s": float(agg_time_s),
                "dec_time_s": float(dec_time_s),
            }
        else:
            # Aggregazione classica e, se richiesto, sola stima HE
            new_weights = aggregate_fedavg(client_weights)
            he_info = estimate_he_communication_overhead(new_weights, poly_modulus_degree=he_poly_modulus_degree) if fl_cfg.use_he else {"he_available": False}
            he_info["mode"] = "estimate" if he_info.get("he_available") else None
        he_overheads.append(he_info)

        model.set_weights(new_weights)
        round_histories.append({"round": r + 1, "he": he_info})
        round_durations_s.append(time.time() - r0)

    # Accuracy proxy: valutiamo sul dataset completo
    loss, acc = model.evaluate(X_prep, y, verbose=0)
    total_time = time.time() - t0

    res = {
        "final_accuracy": float(acc),
        "rounds": fl_cfg.rounds,
        "num_clients": fl_cfg.num_clients,
        "use_dp": fl_cfg.use_dp,
        "use_he": fl_cfg.use_he,
        "he_round_overheads": he_overheads,
        "round_durations_s": round_durations_s,
        "total_time": float(total_time),
        "class_names": list(label_encoder.classes_),
    }
    return res


