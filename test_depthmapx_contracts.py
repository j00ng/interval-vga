"""Executable contracts recovered from the depthmapX VGA pipeline."""

import unittest

import numpy as np

from interval_vga import EnvironmentGrid, VisibilityEngine
from interval_vga.depthmapx import adjacency_edges, decode_grid_ref, encode_grid_ref
from visibility_rules import Interval25DVisibilityRule, TwoDVisibilityRule


class DepthmapXContractTests(unittest.TestCase):
    def setUp(self):
        self.low = np.full((3, 4), -99.0, dtype=np.float32)
        self.high = np.full((3, 4), 99.0, dtype=np.float32)
        self.cells = np.array([[1, 0], [1, 1], [1, 3]], dtype=np.int32)

    def test_grid_reference_round_trip_matches_connection_csv_contract(self):
        reference = encode_grid_ref(row=27, column=14)
        self.assertEqual(reference, 14 * 65536 + 27)
        self.assertEqual(decode_grid_ref(reference), (27, 14))

    def test_planar_baseline_is_an_undirected_edge_set(self):
        result = VisibilityEngine(EnvironmentGrid(self.low, self.high)).compute(
            self.cells, TwoDVisibilityRule()
        )
        np.testing.assert_array_equal(result.adjacency, result.adjacency.T)
        edges = adjacency_edges(result.cells, result.adjacency)
        self.assertTrue(all(left < right for left, right in edges))
        self.assertEqual(len(edges), 3)

    def test_edge_generation_does_not_mutate_input_geometry(self):
        cells_before = self.cells.copy()
        low_before, high_before = self.low.copy(), self.high.copy()
        height = np.full(len(self.cells), 1.5, dtype=np.float32)
        VisibilityEngine(EnvironmentGrid(self.low, self.high)).compute(
            self.cells,
            Interval25DVisibilityRule(),
            observer_height=height,
            target_low=height,
        )
        np.testing.assert_array_equal(self.cells, cells_before)
        np.testing.assert_array_equal(self.low, low_before)
        np.testing.assert_array_equal(self.high, high_before)

    def test_2d_and_25d_share_geometry_but_not_edge_semantics(self):
        self.low[1, 1], self.high[1, 1] = 1.2, 2.1
        environment = EnvironmentGrid(self.low, self.high)
        planar = VisibilityEngine(environment).compute(self.cells, TwoDVisibilityRule())
        low_height = np.full(len(self.cells), 1.1, dtype=np.float32)
        interval = VisibilityEngine(environment).compute(
            self.cells,
            Interval25DVisibilityRule(),
            observer_height=low_height,
            target_low=low_height,
        )
        np.testing.assert_array_equal(planar.cells, interval.cells)
        self.assertTrue(planar.adjacency[0, 2])
        self.assertFalse(interval.adjacency[0, 2])


if __name__ == "__main__":
    unittest.main()
