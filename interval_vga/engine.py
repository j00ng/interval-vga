"""Domain-neutral public API for visibility edge generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visibility_rules import VisibilityRule


@dataclass(frozen=True)
class EnvironmentGrid:
    """Planar grid whose cells store vertically passable height intervals."""

    passable_low: np.ndarray
    passable_high: np.ndarray
    cell_size: float = 1.0
    origin: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        low = np.asarray(self.passable_low, dtype=np.float32)
        high = np.asarray(self.passable_high, dtype=np.float32)
        if low.ndim != 2 or low.shape != high.shape:
            raise ValueError("passability arrays must be equally shaped 2D grids")
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        object.__setattr__(self, "passable_low", low)
        object.__setattr__(self, "passable_high", high)

    @property
    def blocked(self) -> np.ndarray:
        return self.passable_low > self.passable_high

    def xy(self, cells: np.ndarray) -> np.ndarray:
        """Convert row,column cell indices to x,y coordinates."""
        cells = np.asarray(cells)
        x = self.origin[0] + cells[:, 1] * self.cell_size
        y = self.origin[1] + cells[:, 0] * self.cell_size
        return np.column_stack((x, y))


@dataclass(frozen=True)
class VisibilityResult:
    """Adjacency weights plus the cells they index."""

    cells: np.ndarray
    adjacency: np.ndarray
    rule_name: str

    @property
    def directed(self) -> bool:
        return not np.array_equal(self.adjacency, self.adjacency.T)

    @property
    def weighted(self) -> bool:
        return self.adjacency.dtype != np.bool_

    def binary(self, threshold: float = 1.0) -> np.ndarray:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        return self.adjacency >= threshold


class VisibilityEngine:
    """Apply a swappable edge-generation rule to one environment."""

    def __init__(self, environment: EnvironmentGrid):
        self.environment = environment

    def compute(
        self,
        cells: np.ndarray,
        rule: VisibilityRule,
        *,
        observer_height: np.ndarray | None = None,
        target_low: np.ndarray | None = None,
        target_high: np.ndarray | None = None,
        progress: bool = False,
    ) -> VisibilityResult:
        cells = np.asarray(cells, dtype=np.int32)
        adjacency = rule.generate(
            cells,
            self.environment.passable_low,
            self.environment.passable_high,
            observer_height=observer_height,
            target_low=target_low,
            target_high=target_high,
            progress=progress,
        )
        return VisibilityResult(cells=cells, adjacency=adjacency, rule_name=rule.name)
