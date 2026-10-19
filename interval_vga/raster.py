"""Raster adapters for planar segments with vertical passability intervals."""

from __future__ import annotations

import numpy as np

from .engine import EnvironmentGrid


Segment = tuple[float, float, float, float]
OpeningSegment = tuple[float, float, float, float, float, float]


def rasterize_segments(
    walls: list[Segment],
    openings: list[OpeningSegment],
    *,
    cell_size: float,
    pad: float = 1.0,
    free_interval: tuple[float, float] = (-99.0, 99.0),
    blocked_interval: tuple[float, float] = (99.0, -99.0),
) -> EnvironmentGrid:
    """Convert wall/opening line segments into one interval environment.

    Openings are drawn after walls so an opening segment can replace the
    passability of an overlapping wall segment. Sampling and rounding match the
    legacy implementation to preserve its established results.
    """
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    geometry = list(walls) + [opening[:4] for opening in openings]
    if not geometry:
        raise ValueError("at least one wall or opening segment is required")
    xs = [value for segment in geometry for value in (segment[0], segment[2])]
    ys = [value for segment in geometry for value in (segment[1], segment[3])]
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    columns = int(np.ceil((x1 - x0) / cell_size)) + 1
    rows = int(np.ceil((y1 - y0) / cell_size)) + 1
    low = np.full((rows, columns), free_interval[0], dtype=np.float32)
    high = np.full((rows, columns), free_interval[1], dtype=np.float32)

    def draw(segment: Segment, z_min: float, z_max: float) -> None:
        ax, ay, bx, by = segment
        count = int(max(abs(bx - ax), abs(by - ay)) / cell_size * 2) + 2
        fractions = np.linspace(0.0, 1.0, count)
        column = np.rint((ax + (bx - ax) * fractions - x0) / cell_size).astype(int)
        row = np.rint((ay + (by - ay) * fractions - y0) / cell_size).astype(int)
        inside = (column >= 0) & (column < columns) & (row >= 0) & (row < rows)
        low[row[inside], column[inside]] = z_min
        high[row[inside], column[inside]] = z_max

    for wall in walls:
        draw(wall, *blocked_interval)
    for opening in openings:
        draw(opening[:4], opening[4], opening[5])
    return EnvironmentGrid(low, high, cell_size=cell_size, origin=(x0, y0))
