import tempfile
from pathlib import Path
import unittest

from compare_visibility_edges import compare_edges, load_edges


class EdgeComparisonTests(unittest.TestCase):
    def test_undirected_loader_normalizes_reverse_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edges.csv"
            path.write_text("RefFrom,RefTo\n2,1\n1,2\n2,3\n", encoding="utf-8")
            self.assertEqual(load_edges(str(path)), {(1, 2), (2, 3)})

    def test_precision_recall_and_f1(self):
        result = compare_edges({(1, 2), (2, 3)}, {(1, 2), (3, 4)})
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 1)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)

    def test_directed_comparison_keeps_orientation(self):
        result = compare_edges({(1, 2)}, {(2, 1)}, directed=True)
        self.assertEqual(result["true_positive"], 0)
        self.assertEqual(result["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
