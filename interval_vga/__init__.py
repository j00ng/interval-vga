"""General-purpose interval-based 2.5D visibility graph engine."""

from .engine import EnvironmentGrid, VisibilityEngine, VisibilityResult
from .depthmapx import adjacency_edges, decode_grid_ref, encode_grid_ref
from .raster import OpeningSegment, Segment, rasterize_segments
from visibility_rules import (
    Interval25DVisibilityRule,
    ObserverProfile,
    TargetProfile,
    TwoDVisibilityRule,
    VisibilityRule,
    profiles_to_heights,
)

__all__ = [
    "EnvironmentGrid",
    "VisibilityEngine",
    "VisibilityResult",
    "encode_grid_ref",
    "decode_grid_ref",
    "adjacency_edges",
    "Segment",
    "OpeningSegment",
    "rasterize_segments",
    "VisibilityRule",
    "TwoDVisibilityRule",
    "Interval25DVisibilityRule",
    "ObserverProfile",
    "TargetProfile",
    "profiles_to_heights",
]
