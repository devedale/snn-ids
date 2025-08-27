# -*- coding: utf-8 -*-
"""
Astrazione leggera per Homomorphic Encryption (CKKS) focalizzata sulle feature sensibili:
- Ottetti IP: Src IP_Octet_1..4, Dst IP_Octet_1..4
- Porte: Src Port, Dst Port

Se `tenseal` non è disponibile, usa un fallback mock che simula il costo
computazionale e fornisce API compatibili per l'aggregazione sicura.
"""

from typing import Dict, List, Any, Optional

try:
    import tenseal as ts  # type: ignore
    HAVE_TENSEAL = True
except Exception:
    HAVE_TENSEAL = False


class HEContext:
    """Wrapper del contesto CKKS o mock."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        self.scale = config.get("scale", 2 ** 40)
        self.scheme = config.get("scheme", "CKKS")
        self._ctx = None

        if self.enabled and HAVE_TENSEAL:
            poly_modulus_degree = config.get("poly_modulus_degree", 8192)
            coeff_mod_bit_sizes = config.get("coeff_mod_bit_sizes", [60, 40, 40, 60])
            self._ctx = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            )
            self._ctx.generate_galois_keys()
            if config.get("enable_key_rotation", True):
                self._ctx.generate_galois_keys()
            self._ctx.global_scale = self.scale

    def is_active(self) -> bool:
        return self.enabled


class HEEncryptor:
    """Utility per cifrare/decifrare e sommare vettori (per gradienti o features)."""

    def __init__(self, he_context: HEContext):
        self.ctx = he_context

    def encrypt_vector(self, values: List[float]):
        if not self.ctx.enabled:
            return values  # plaintext
        if HAVE_TENSEAL:
            return ts.ckks_vector(self.ctx._ctx, values)
        # mock object
        return _MockCipher(values)

    def decrypt_vector(self, enc_vector):
        if not self.ctx.enabled:
            return enc_vector
        if HAVE_TENSEAL:
            return enc_vector.decrypt()
        return enc_vector.decrypt()

    def add(self, a, b):
        if not self.ctx.enabled:
            # plaintext somma
            return [x + y for x, y in zip(a, b)]
        # HE add
        a.add_(b) if hasattr(a, "add_") else a.add(b)
        return a


class _MockCipher:
    """Cifrario mock per simulare API CKKS di TenSEAL."""

    def __init__(self, values: List[float]):
        self._values = list(values)

    def add(self, other: "_MockCipher"):
        self._values = [x + y for x, y in zip(self._values, other._values)]
        return self

    def add_(self, other: "_MockCipher"):
        return self.add(other)

    def decrypt(self) -> List[float]:
        return list(self._values)


def select_sensitive_feature_indices(feature_names: List[str], features_to_encrypt: List[str]) -> List[int]:
    """Restituisce gli indici delle feature sensibili in `feature_names`."""
    targets = set(features_to_encrypt)
    return [i for i, name in enumerate(feature_names) if name in targets]


