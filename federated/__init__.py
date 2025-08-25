"""
Modulo Federated Learning + Homomorphic Encryption per SNN-IDS.

Contiene:
- he.py: astrazione per cifratura omomorfica (CKKS) con fallback mock.
- fl_simulation.py: classi per simulazione locale di FL (client/server).
- fl_benchmark.py: orchestratore benchmark federato (best-config) con report privacy.
"""

__all__ = [
    "he",
    "fl_simulation",
    "fl_benchmark",
]


