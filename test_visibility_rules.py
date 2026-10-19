import unittest

import numpy as np

from visibility_rules import (
    Interval25DVisibilityRule,
    ObserverProfile,
    TargetProfile,
    TwoDVisibilityRule,
    profiles_to_heights,
)


def legacy_visibility(cells, low, high, eye, head):
    """Frozen copy of the pre-refactor algorithm for regression comparison."""
    count = len(cells)
    rows = cells[:, 0].astype(np.float32)
    cols = cells[:, 1].astype(np.float32)
    steps = int(np.ceil(np.hypot(rows.max() - rows.min(), cols.max() - cols.min()))) + 1
    t = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)[None, :]
    visible = np.zeros((count, count), bool)
    for source in range(count):
        dr = rows - rows[source]
        dc = cols - cols[source]
        ri = np.rint(rows[source] + dr[:, None] * t).astype(np.int32)
        ci = np.rint(cols[source] + dc[:, None] * t).astype(np.int32)
        ray = eye[source] + (head - eye[source])[:, None] * t
        blocked = ((ray < low[ri, ci]) | (ray > high[ri, ci])).any(axis=1)
        visible[source] = ~blocked
    np.fill_diagonal(visible, False)
    return visible


class VisibilityRuleTests(unittest.TestCase):
    def setUp(self):
        self.cells = np.array([[1, 0], [1, 1], [1, 2]], dtype=np.int32)
        self.low = np.full((3, 3), -99.0, dtype=np.float32)
        self.high = np.full((3, 3), 99.0, dtype=np.float32)

    def test_interval_rule_preserves_legacy_single_target_result(self):
        self.low[1, 1] = 1.2
        self.high[1, 1] = 2.1
        eye = np.array([1.55, 1.10, 1.10], dtype=np.float32)
        head = np.array([1.20, 1.20, 1.70], dtype=np.float32)
        expected = legacy_visibility(self.cells, self.low, self.high, eye, head)
        actual = Interval25DVisibilityRule().generate(
            self.cells,
            self.low,
            self.high,
            observer_height=eye,
            target_low=head,
            target_high=head,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_planar_rule_is_symmetric_and_ignores_opening_height(self):
        self.low[1, 1] = 1.8
        self.high[1, 1] = 2.1
        visible = TwoDVisibilityRule().generate(self.cells, self.low, self.high)
        self.assertTrue(visible[0, 2])
        np.testing.assert_array_equal(visible, visible.T)

    def test_wall_blocks_both_rules(self):
        self.low[1, 1] = 99.0
        self.high[1, 1] = -99.0
        planar = TwoDVisibilityRule().generate(self.cells, self.low, self.high)
        interval = Interval25DVisibilityRule().generate(
            self.cells,
            self.low,
            self.high,
            observer_height=np.full(3, 1.5, dtype=np.float32),
            target_low=np.full(3, 1.5, dtype=np.float32),
        )
        self.assertFalse(planar[0, 2])
        self.assertFalse(interval[0, 2])

    def test_same_endpoint_states_produce_reciprocal_los(self):
        self.low[1, 1] = 1.2
        self.high[1, 1] = 2.1
        height = np.full(3, 1.55, dtype=np.float32)
        visible = Interval25DVisibilityRule().generate(
            self.cells, self.low, self.high,
            observer_height=height, target_low=height,
        )
        np.testing.assert_array_equal(visible, visible.T)

    def test_target_interval_returns_visible_fraction(self):
        self.low[1, 1] = 1.0
        self.high[1, 1] = 1.5
        observer = np.full(3, 1.25, dtype=np.float32)
        target_low = np.array([1.25, 1.25, 0.75], dtype=np.float32)
        target_high = np.array([1.25, 1.25, 1.75], dtype=np.float32)
        weighted = Interval25DVisibilityRule(target_samples=5).generate(
            self.cells, self.low, self.high,
            observer_height=observer,
            target_low=target_low,
            target_high=target_high,
        )
        self.assertGreater(weighted[0, 2], 0.0)
        self.assertLess(weighted[0, 2], 1.0)

    def test_profiles_are_domain_neutral(self):
        names = np.array(["standing", "seated"])
        observers = {
            "standing": ObserverProfile("standing", 1.55),
            "seated": ObserverProfile("seated", 1.10),
        }
        targets = {
            "standing": TargetProfile("standing", (1.5, 1.8)),
            "seated": TargetProfile("seated", (1.0, 1.3)),
        }
        eye, low, high = profiles_to_heights(names, observers, targets)
        np.testing.assert_allclose(eye, [1.55, 1.10])
        np.testing.assert_allclose(low, [1.5, 1.0])
        np.testing.assert_allclose(high, [1.8, 1.3])


if __name__ == "__main__":
    unittest.main()
