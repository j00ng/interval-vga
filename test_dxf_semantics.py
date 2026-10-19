import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dxf_semantics import LayerType, OpeningProfile, infer_layer_semantic, parse_dxf
from interval_vga.raster import rasterize_segments
from visibility_rules import Interval25DVisibilityRule


def line_entity(layer, x1=0, y1=0, x2=1, y2=0):
    return "\n".join([
        "0", "LINE", "8", layer,
        "10", str(x1), "20", str(y1),
        "11", str(x2), "21", str(y2),
    ])


class DXFSemanticsTests(unittest.TestCase):
    def test_layer_names_distinguish_wall_door_and_window(self):
        self.assertEqual(infer_layer_semantic("WALL_SURE").kind, "wall")
        self.assertEqual(infer_layer_semantic("DOOR").kind, "door")
        window = infer_layer_semantic("WINDOW")
        self.assertEqual(window.kind, "window")
        self.assertEqual((window.sill, window.head), (1.2, 2.1))

    def test_layer_name_can_carry_height_metadata(self):
        window = infer_layer_semantic("WINDOW_SILL_0.9_HEAD_2.4")
        self.assertEqual((window.sill, window.head), (0.9, 2.4))

    def test_sidecar_overrides_layer_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dxf = root / "scene.dxf"
            dxf.write_text("\n".join([line_entity("GLAZING"), "0", "EOF"]), encoding="utf-8")
            metadata = root / "openings.json"
            metadata.write_text(json.dumps({
                "layers": {"GLAZING": {"kind": "window", "sill": 1.05, "head": 2.25}}
            }), encoding="utf-8")
            parsed = parse_dxf(str(dxf), metadata_path=str(metadata))
            self.assertEqual(parsed.walls, [])
            self.assertEqual(parsed.openings[0][-2:], (1.05, 2.25))
            self.assertEqual(parsed.layer_semantics["GLAZING"].source, "metadata")

    def test_generic_opening_is_not_silently_treated_as_door(self):
        with tempfile.TemporaryDirectory() as directory:
            dxf = Path(directory) / "scene.dxf"
            dxf.write_text("\n".join([line_entity("OPENING"), "0", "EOF"]), encoding="utf-8")
            parsed = parse_dxf(str(dxf))
            self.assertEqual(parsed.unknown_opening_layers, ("OPENING",))
            self.assertEqual(parsed.openings, [])

    def test_alias_mapping_resolves_drawing_specific_layer_name(self):
        semantic = infer_layer_semantic("A_GLAZ", aliases={"A_GLAZ": LayerType.WINDOW})
        self.assertEqual(semantic.kind, LayerType.WINDOW)
        self.assertEqual((semantic.sill, semantic.head), (1.2, 2.1))

    def test_opening_profile_controls_vertical_intervals(self):
        profile = OpeningProfile(door=(0.1, 2.4), window=(0.8, 1.9))
        door = infer_layer_semantic("DOOR", profile=profile)
        window = infer_layer_semantic("WINDOW", profile=profile)
        self.assertEqual((door.sill, door.head), (0.1, 2.4))
        self.assertEqual((window.sill, window.head), (0.8, 1.9))

    def test_sidecar_aliases_and_profile_are_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dxf = root / "scene.dxf"
            dxf.write_text("\n".join([line_entity("A-GLAZ"), "0", "EOF"]), encoding="utf-8")
            metadata = root / "semantics.json"
            metadata.write_text(json.dumps({
                "opening_profile": {
                    "door": {"sill": 0.0, "head": 2.2},
                    "window": {"sill": 0.95, "head": 2.35},
                },
                "aliases": {"A-GLAZ": "window"},
            }), encoding="utf-8")
            parsed = parse_dxf(str(dxf), metadata_path=str(metadata))
            self.assertEqual(parsed.openings[0][-2:], (0.95, 2.35))

    def test_layer_override_can_use_profile_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dxf = root / "scene.dxf"
            dxf.write_text("\n".join([line_entity("GLAZING"), "0", "EOF"]), encoding="utf-8")
            metadata = root / "semantics.json"
            metadata.write_text(json.dumps({
                "opening_profile": {"window": [0.85, 2.25]},
                "layers": {"GLAZING": {"kind": "window"}},
            }), encoding="utf-8")
            parsed = parse_dxf(str(dxf), metadata_path=str(metadata))
            self.assertEqual(parsed.openings[0][-2:], (0.85, 2.25))

    def test_real_dxf_window_sill_changes_los(self):
        """Integration contract: DXF semantics must reach the LOS rule."""
        with tempfile.TemporaryDirectory() as directory:
            dxf = Path(directory) / "scene.dxf"
            dxf.write_text("\n".join([line_entity("WINDOW"), "0", "EOF"]), encoding="utf-8")
            parsed = parse_dxf(str(dxf))
            sill, head = parsed.openings[0][-2:]

        environment = rasterize_segments([], parsed.openings, cell_size=0.5, pad=0.5)
        self.assertEqual((sill, head), (1.2, 2.1))
        cells = np.array([[0, 2], [2, 2]], dtype=np.int32)
        rule = Interval25DVisibilityRule()
        low_view = np.full(2, 1.10, dtype=np.float32)
        high_view = np.full(2, 1.55, dtype=np.float32)
        blocked = rule.generate(
            cells, environment.passable_low, environment.passable_high,
            observer_height=low_view, target_low=low_view,
        )
        visible = rule.generate(
            cells, environment.passable_low, environment.passable_high,
            observer_height=high_view, target_low=high_view,
        )
        self.assertFalse(blocked[0, 1])
        self.assertTrue(visible[0, 1])


if __name__ == "__main__":
    unittest.main()
