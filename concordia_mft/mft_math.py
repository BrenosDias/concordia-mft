"""MFT vector helpers."""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

from concordia_mft.constants import MFT_KEYS


def mft_dict_to_vec(d: dict) -> np.ndarray:
    return np.array([float(d[k]) for k in MFT_KEYS], dtype=np.float64)


def cos_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = norm(v1), norm(v2)
    if n1 == 0 or n2 == 0:
        return float('nan')
    return float(np.dot(v1, v2) / (n1 * n2))
