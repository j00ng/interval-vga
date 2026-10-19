import unittest

import numpy as np

from interval_vga import (
    EnvironmentGrid,
    Interval25DVisibilityRule,
    TwoDVisibilityRule,
    VisibilityEngine,
    rasterize_segments,
)


class PublicAPITests(unittest.TestCase):
    def setUp(self):
        low = np.full((3, 3), -99.0, dtype=np.float32)
        high = np.full((3, 3), 99.0, dtype=np.float32)
        low[1, 1], high[1, 1] = 1.2, 2.1
        self.environment = EnvironmentGrid(low, high, cell_size=0.5, origin=(10.0, 20.0))
        self.cells = np.array([[1, 0], [1, 1], [1, 2]], dtype=np.int32)

    def test_segment_rasterization_is_domain_neutral(self):
        environment = rasterize_segments(
            walls=[(0.0, 0.0, 1.0, 0.0)],
            openings=[(0.25, 0.0, 0.75, 0.0, 0.8, 2.0)],
            cell_size=0.25,
            pad=0.25,
        )
        self.assertEqual(environment.cell_size, 0.25)
        self.assertTrue(np.any(environment.blocked))
        self.assertTrue(np.any(environment.passable_low == 0.8))

    def test_planar_and_interval_rules_share_one_engine(self):
        engine = VisibilityEngine(self.environment)
        planar = engine.compute(self.cells, TwoDVisibilityRule())
        heights = np.full(3, 1.55, dtype=np.float32)
        interval = engine.compute(
            self.cells,
            Interval25DVisibilityRule(),
            observer_height=heights,
            target_low=heights,
        )
        self.assertEqual(planar.rule_name, "2d-planar")
        self.assertEqual(interval.rule_name, "interval-2.5d")
        self.assertFalse(planar.directed)
        self.assertFalse(interval.directed)

    def test_weighted_result_can_be_thresholded(self):
        result = VisibilityEngine(self.environment).compute(
            self.cells,
            Interval25DVisibilityRule(target_samples=5),
            observer_height=np.full(3, 1.25, dtype=np.float32),
            target_low=np.array([1.25, 1.25, 0.75], dtype=np.float32),
            target_high=np.array([1.25, 1.25, 1.75], dtype=np.float32),
        )
        self.assertTrue(result.weighted)
        self.assertEqual(result.binary(0.5).dtype, np.bool_)

    def test_environment_coordinates_are_not_building_type_specific(self):
        xy = self.environment.xy(self.cells)
        np.testing.assert_allclose(xy, [[10.0, 20.5], [10.5, 20.5], [11.0, 20.5]])


if __name__ == "__main__":
    unittest.main()
