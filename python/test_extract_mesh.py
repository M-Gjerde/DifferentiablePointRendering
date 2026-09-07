from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from extract_mesh import load_point_radius


class PointRadiusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "points.ply"

    def write_points(self, rows: list[tuple[float, ...]], *, with_power: bool = True) -> None:
        properties = ["x", "y", "z"] + (["power"] if with_power else [])
        self.path.write_text(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(rows)}\n"
            + "".join(f"property float {name}\n" for name in properties)
            + "end_header\n"
            + "".join(" ".join(str(value) for value in row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_lights_do_not_change_center_or_radius(self) -> None:
        surface = [(10, 0, 0, 0), (14, 0, 0, 0), (18, 0, 0, 0)]
        self.write_points(surface)
        self.assertEqual(load_point_radius(self.path), 4.0)

        for light in ((10000, 5000, -2000, 500), (-5000, 3000, 1000, 1e-8)):
            with self.subTest(light=light):
                self.write_points([light, *surface])
                self.assertEqual(load_point_radius(self.path), 4.0)

    def test_all_lights_have_no_surface_scale(self) -> None:
        self.write_points([(0, 0, 0, 500), (100, 0, 0, 1)])
        with self.assertRaisesRegex(RuntimeError, "No non-emissive points"):
            load_point_radius(self.path)

    def test_missing_power_cannot_silently_include_lights(self) -> None:
        self.write_points([(0, 0, 0), (100, 0, 0)], with_power=False)
        with self.assertRaisesRegex(ValueError, "Missing vertex power"):
            load_point_radius(self.path)


if __name__ == "__main__":
    unittest.main()
