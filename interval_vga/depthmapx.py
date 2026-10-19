"""Narrow, tested interoperability boundary for depthmapX CSV exports.

This module does not read or write depthmapX ``.graph`` files.  It only preserves
the packed grid reference used by point-map CSV exports and converts an already
computed adjacency matrix into comparable edge pairs.
"""

from __future__ import annotations

import numpy as np


DEPTHMAP_REF_STRIDE = 1 << 16


def encode_grid_ref(row: int, column: int) -> int:
    """Pack a non-negative row/column as ``column * 65536 + row``."""
    if not 0 <= row < DEPTHMAP_REF_STRIDE or not 0 <= column < DEPTHMAP_REF_STRIDE:
        raise ValueError("depthmapX grid row and column must be within [0, 65535]")
    return int(column) * DEPTHMAP_REF_STRIDE + int(row)


def decode_grid_ref(reference: int) -> tuple[int, int]:
    """Return ``(row, column)`` from a packed depthmapX point reference."""
    if reference < 0:
        raise ValueError("depthmapX grid reference must be non-negative")
    return int(reference & 0xFFFF), int(reference >> 16)


def adjacency_edges(
    cells: np.ndarray,
    adjacency: np.ndarray,
    *,
    directed: bool = False,
    threshold: float = 1.0,
) -> set[tuple[int, int]]:
    """Convert row/column-indexed adjacency into depthmapX-compatible refs."""
    cells = np.asarray(cells)
    adjacency = np.asarray(adjacency)
    count = len(cells)
    if cells.shape != (count, 2):
        raise ValueError("cells must have shape (n, 2) in row,column order")
    if adjacency.shape != (count, count):
        raise ValueError("adjacency must have shape (n, n)")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    refs = [encode_grid_ref(int(row), int(column)) for row, column in cells]
    result: set[tuple[int, int]] = set()
    for source, target in np.argwhere(adjacency >= threshold):
        if source == target:
            continue
        left, right = refs[int(source)], refs[int(target)]
        if not directed and left > right:
            left, right = right, left
        result.add((left, right))
    return result
