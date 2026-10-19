# depthmapX compatibility contract

This project uses recovered depthmapX behavior as an executable boundary, not as
a claim that the Python implementation is a drop-in clone.

## Preserved contracts

1. **Stable point-map geometry** — edge generation consumes an existing set of
   grid points and does not mutate its coordinates or passability grid.
2. **Edge generation before metrics** — the engine returns adjacency first.
   Connectivity and integration-like metrics remain downstream consumers.
3. **Reciprocal planar baseline** — standard 2D LOS produces an undirected,
   diagonal-free graph.
4. **Packed grid references** — connection CSV identifiers use
   `column * 65536 + row` and round-trip through the public adapter.
5. **2.5D is a separate rule** — observer/target height and opening intervals
   may change edges, but never redefine the planar baseline or its geometry.
6. **Explicit redraw boundary** — presentation and document refresh behavior is
   outside the computational engine. Core computation has no UI side effects.

These contracts are enforced by `test_depthmapx_contracts.py` and the legacy
single-target regression test in `test_visibility_rules.py`.

## Compatibility boundary

The supported interchange format is currently the depthmapX point-map connection
CSV. `interval_vga.depthmapx` packs point references and converts adjacency into
edge pairs; `compare_visibility_edges.py` reports precision, recall, F1, Jaccard,
and degree-rank correlation against a depthmapX export.

The project does **not** currently read or write `.graph` files. It also does not
claim equality with depthmapX's bin-based `sparkGraph2` implementation until a
published real-plan fixture passes the connection-edge comparison gate.

## Gate for stronger claims

A release may say "depthmapX edge-compatible for the tested fixture" only when:

- the exact DXF, grid spacing, origin, fill region, and depthmapX version are fixed;
- both pipelines expose their point references and undirected connection edges;
- node sets match, and edge precision/recall/F1 are reported rather than hidden;
- any difference is retained as a reproducible fixture and explained.

Without that evidence, the accurate description is "depthmapX-informed VGA
engine with CSV comparison tooling."
