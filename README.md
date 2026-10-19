# interval-vga

`interval-vga` is an experimental, general-purpose engine for height-aware 2.5D
visibility graphs. It stores only the vertical passability interval required at
each planar grid cell instead of reconstructing a complete 3D mesh or voxel field.

The core model is domain-neutral:

```text
visibility = f(environment, observer state, target state, edge rule)
```

It can represent building interiors, public spaces, camera layouts, exhibitions,
industrial environments, accessibility studies, and other spatial visibility
problems. School layouts are included only as one research example.

## Current status

Research preview (`0.1.0-alpha`). The engine currently provides:

- a symmetric planar 2D LOS baseline;
- interval-based 2.5D LOS;
- directed observer/target-conditioned visibility;
- weighted visibility over a sampled target vertical extent;
- DXF layer semantics for walls, doors, windows, and custom openings;
- edge-set comparison against depthmapX connection exports.

DXF naming is configured outside the engine. A semantic sidecar can define a
general door/window profile, drawing-specific aliases, and exceptional per-layer
heights; see `examples/semantic-config.json`. Unresolved opening layers are
reported as `UNKNOWN_OPENING` instead of being silently treated as doors.

The project does **not** yet claim exact depthmapX edge compatibility or full-3D
equivalence. Those claims require the published benchmark described below.

## Minimal API

```python
import numpy as np
from interval_vga import EnvironmentGrid, VisibilityEngine, Interval25DVisibilityRule
from interval_vga import rasterize_segments

low = np.full((20, 20), -99.0, dtype=np.float32)
high = np.full((20, 20), 99.0, dtype=np.float32)
environment = EnvironmentGrid(low, high, cell_size=0.5)
cells = np.argwhere(~environment.blocked)

eye = np.full(len(cells), 1.6, dtype=np.float32)
target_low = np.full(len(cells), 0.8, dtype=np.float32)
target_high = np.full(len(cells), 1.8, dtype=np.float32)

result = VisibilityEngine(environment).compute(
    cells,
    Interval25DVisibilityRule(target_samples=5),
    observer_height=eye,
    target_low=target_low,
    target_high=target_high,
)
```

For line-based inputs, `rasterize_segments(walls, openings, cell_size=...)`
constructs the same `EnvironmentGrid`. Each opening is
`(x1, y1, x2, y2, z_min, z_max)`; DXF `WINDOW` and `DOOR` segments obtain those
vertical intervals from the semantic profile before rasterization.

## Model boundaries

- A passable interval is a reduced visibility model, not complete 3D geometry.
- Directionality comes from different observer and target states; light itself is
  reciprocal between identical endpoints.
- Directed graph metrics are not presented as depthmapX Integration[HH].
- Runtime is currently dominated by all-pairs LOS sampling.

## Validation gates

1. Preserve the existing single-target 2.5D result through regression tests.
2. Compare the 2D Python edge set with depthmapX `pointmap-connections-csv`.
3. Report edge precision, recall, F1, metric rank correlation, runtime, and memory.
4. Compare interval 2.5D against full 3D mesh ray casting on shared scenes.

Recovered depthmapX rules used by this implementation are documented in
`docs/depthmapx-compatibility-contract.md` and enforced by contract tests. The
current interoperability boundary is connection CSV, not the binary `.graph`
format.

## Licensing

A project license has not yet been selected. depthmapX is a separate GPLv3 project;
its executable and source are not bundled here. Select a license before public release.
