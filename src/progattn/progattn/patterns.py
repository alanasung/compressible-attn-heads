"""Programmatic QK pattern generators."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

PROGRAMS: tuple[str, ...] = (
    "previous_token",
    "positional_offset",
    "bos_attend",
    "induction",
    "positional_decay",
    "delimiter",
    "uniform",
)


def _causal_mask(n: int) -> np.ndarray:
    return np.tril(np.ones((n, n), dtype=np.float64))


def previous_token(n: int, **_: object) -> np.ndarray:
    p = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = max(0, i - 1)
        p[i, j] = 1.0
    return p


def positional_offset(n: int, offset: int = 2, **_: object) -> np.ndarray:
    p = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = max(0, i - offset)
        p[i, j] = 1.0
    return p


def bos_attend(n: int, **_: object) -> np.ndarray:
    p = np.zeros((n, n), dtype=np.float64)
    p[:, 0] = 1.0
    return p * _causal_mask(n)


def induction(n: int, **_: object) -> np.ndarray:
    # Soft previous-token + mild lookback — proxy for induction heads.
    p = previous_token(n) * 0.7 + positional_offset(n, offset=3) * 0.3
    return p


def positional_decay(n: int, decay: float = 0.8, **_: object) -> np.ndarray:
    p = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            p[i, j] = decay ** (i - j)
    row = p.sum(axis=1, keepdims=True)
    return p / np.maximum(row, 1e-8)


def delimiter(n: int, every: int = 4, **_: object) -> np.ndarray:
    p = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        anchors = [j for j in range(0, i + 1) if j % every == 0]
        if not anchors:
            anchors = [0]
        for j in anchors:
            p[i, j] = 1.0 / len(anchors)
    return p


def uniform(n: int, **_: object) -> np.ndarray:
    p = _causal_mask(n)
    row = p.sum(axis=1, keepdims=True)
    return p / np.maximum(row, 1e-8)


_GENERATORS: dict[str, Callable[..., np.ndarray]] = {
    "previous_token": previous_token,
    "positional_offset": positional_offset,
    "bos_attend": bos_attend,
    "induction": induction,
    "positional_decay": positional_decay,
    "delimiter": delimiter,
    "uniform": uniform,
}


def generate_pattern(name: str, n: int, **kwargs: object) -> np.ndarray:
    if name not in _GENERATORS:
        raise KeyError(f"unknown program {name!r}; known={list(_GENERATORS)}")
    p = _GENERATORS[name](n, **kwargs)
    # Numerical safety: rows sum to 1 under causal support.
    p = np.asarray(p, dtype=np.float64) * _causal_mask(n)
    row = p.sum(axis=1, keepdims=True)
    return p / np.maximum(row, 1e-8)
