"""Visibility edge-generation rules shared by the 2D and 2.5D models.

The grid and graph analysis layers intentionally know nothing about school roles.
They provide an environment, observer heights, and target vertical extents; a rule
decides whether (or how strongly) each directed edge exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ObserverProfile:
    name: str
    height: float


@dataclass(frozen=True)
class TargetProfile:
    name: str
    vertical_extent: tuple[float, float]

    def __post_init__(self) -> None:
        low, high = self.vertical_extent
        if low > high:
            raise ValueError("target vertical_extent must satisfy low <= high")


class VisibilityRule(Protocol):
    """Strategy boundary for replacing only the graph edge-generation rule."""

    name: str

    def generate(
        self,
        cells: np.ndarray,
        passable_low: np.ndarray,
        passable_high: np.ndarray,
        *,
        observer_height: np.ndarray | None = None,
        target_low: np.ndarray | None = None,
        target_high: np.ndarray | None = None,
        progress: bool = False,
    ) -> np.ndarray:
        """Return an N×N adjacency/visibility matrix."""


def _validate_grid(
    cells: np.ndarray, passable_low: np.ndarray, passable_high: np.ndarray
) -> np.ndarray:
    cells = np.asarray(cells)
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError("cells must have shape (n, 2) in row,column order")
    if passable_low.shape != passable_high.shape:
        raise ValueError("passable interval arrays must have the same shape")
    if len(cells):
        rows, cols = cells[:, 0], cells[:, 1]
        if rows.min() < 0 or cols.min() < 0:
            raise ValueError("cell coordinates must be non-negative")
        if rows.max() >= passable_low.shape[0] or cols.max() >= passable_low.shape[1]:
            raise ValueError("cell coordinates fall outside the environment grid")
    return cells


def _sample_path_indices(cells: np.ndarray, source: int, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the legacy all-pairs sampling and rounding exactly."""
    rows = cells[:, 0].astype(np.float32)
    cols = cells[:, 1].astype(np.float32)
    t = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)[None, :]
    dr = rows - rows[source]
    dc = cols - cols[source]
    ri = np.rint(rows[source] + dr[:, None] * t).astype(np.int32)
    ci = np.rint(cols[source] + dc[:, None] * t).astype(np.int32)
    return ri, ci, t


def _path_steps(cells: np.ndarray) -> int:
    if len(cells) < 2:
        return 1
    rows = cells[:, 0]
    cols = cells[:, 1]
    return int(np.ceil(np.hypot(rows.max() - rows.min(), cols.max() - cols.min()))) + 1


def _progress(source: int, count: int, enabled: bool) -> None:
    if enabled and count > 400 and source % max(1, count // 10) == 0:
        print(f"    가시 계산 {source}/{count} ({source/count*100:.0f}%)", flush=True)


class TwoDVisibilityRule:
    """Planar baseline: any non-wall cell is vertically transparent.

    This isolates the Python raster/LOS baseline. It is intentionally not named
    "depthmapX exact": depthmapX uses its own bin-based spark graph algorithm.
    """

    name = "2d-planar"

    def generate(
        self,
        cells: np.ndarray,
        passable_low: np.ndarray,
        passable_high: np.ndarray,
        *,
        observer_height: np.ndarray | None = None,
        target_low: np.ndarray | None = None,
        target_high: np.ndarray | None = None,
        progress: bool = False,
    ) -> np.ndarray:
        cells = _validate_grid(cells, passable_low, passable_high)
        count = len(cells)
        visible = np.zeros((count, count), dtype=bool)
        steps = _path_steps(cells)
        blocked_grid = passable_low > passable_high
        for source in range(count):
            rows, cols, _ = _sample_path_indices(cells, source, steps)
            visible[source] = ~blocked_grid[rows, cols].any(axis=1)
            _progress(source, count, progress)
        np.fill_diagonal(visible, False)
        # Planar LOS is reciprocal. Enforce the invariant even if a future path
        # sampler changes its tie-breaking at cell boundaries.
        return visible & visible.T


class Interval25DVisibilityRule:
    """Height-aware LOS over vertical passability intervals.

    With one target sample this is byte-for-byte compatible with the previous
    ``visgraph.visibility`` calculation. More samples return a float matrix in
    [0, 1]: the fraction of the target interval visible from each observer.
    """

    name = "interval-2.5d"

    def __init__(self, target_samples: int = 1):
        if target_samples < 1:
            raise ValueError("target_samples must be >= 1")
        self.target_samples = int(target_samples)

    def generate(
        self,
        cells: np.ndarray,
        passable_low: np.ndarray,
        passable_high: np.ndarray,
        *,
        observer_height: np.ndarray | None = None,
        target_low: np.ndarray | None = None,
        target_high: np.ndarray | None = None,
        progress: bool = False,
    ) -> np.ndarray:
        cells = _validate_grid(cells, passable_low, passable_high)
        count = len(cells)
        observer = np.asarray(observer_height, dtype=np.float32)
        low = np.asarray(target_low, dtype=np.float32)
        high = low if target_high is None else np.asarray(target_high, dtype=np.float32)
        for name, values in (("observer_height", observer), ("target_low", low), ("target_high", high)):
            if values.shape != (count,):
                raise ValueError(f"{name} must have shape ({count},)")
        if np.any(low > high):
            raise ValueError("every target interval must satisfy low <= high")

        steps = _path_steps(cells)
        dtype = bool if self.target_samples == 1 else np.float32
        visible = np.zeros((count, count), dtype=dtype)
        sample_fractions = np.linspace(0.0, 1.0, self.target_samples, dtype=np.float32)

        for source in range(count):
            rows, cols, t = _sample_path_indices(cells, source, steps)
            grid_low = passable_low[rows, cols]
            grid_high = passable_high[rows, cols]
            hits = np.zeros(count, dtype=np.float32)
            for fraction in sample_fractions:
                target = low + fraction * (high - low)
                ray_height = observer[source] + (target - observer[source])[:, None] * t
                blocked = ((ray_height < grid_low) | (ray_height > grid_high)).any(axis=1)
                hits += ~blocked
            if self.target_samples == 1:
                visible[source] = hits.astype(bool)
            else:
                visible[source] = hits / self.target_samples
            _progress(source, count, progress)

        np.fill_diagonal(visible, False)
        return visible


def profiles_to_heights(
    profile_names: np.ndarray,
    observers: dict[str, ObserverProfile],
    targets: dict[str, TargetProfile],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve domain-neutral named profiles into per-cell numeric arrays."""
    names = np.asarray(profile_names)
    missing = set(names.tolist()) - (set(observers) & set(targets))
    if missing:
        raise KeyError(f"profiles missing observer or target definitions: {sorted(missing)}")
    observer = np.array([observers[name].height for name in names], dtype=np.float32)
    target_low = np.array([targets[name].vertical_extent[0] for name in names], dtype=np.float32)
    target_high = np.array([targets[name].vertical_extent[1] for name in names], dtype=np.float32)
    return observer, target_low, target_high
