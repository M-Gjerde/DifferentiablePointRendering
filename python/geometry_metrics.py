from __future__ import annotations

import csv
import tempfile
import warnings
from pathlib import Path
from typing import Any, Mapping


class GeometryMetricsTrail:
    """Evaluate extracted meshes once and persist a live, run-local CSV trail."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "geometry_metrics.csv"
        self._rows_by_iteration: dict[int, dict[str, Any]] = {}
        self._load_existing_rows()

    def _load_existing_rows(self) -> None:
        """Restore an existing trail so resumed runs do not discard history."""
        if not self.path.is_file():
            return

        with self.path.open("r", encoding="utf-8", newline="") as csv_file:
            for line_number, row in enumerate(csv.DictReader(csv_file), start=2):
                iteration_value = row.get("iteration")
                try:
                    if iteration_value is None:
                        raise ValueError("missing iteration")
                    iteration = int(iteration_value)
                except (TypeError, ValueError):
                    invalid_iteration = (
                        "<missing>" if iteration_value is None else iteration_value
                    )
                    warnings.warn(
                        f"Ignoring geometry metrics row {line_number} with invalid "
                        f"iteration '{invalid_iteration}': {self.path}",
                        stacklevel=2,
                    )
                    continue
                self._rows_by_iteration[iteration] = {
                    key: value for key, value in row.items() if key is not None
                }

    @property
    def latest_row(self) -> Mapping[str, Any] | None:
        if not self._rows_by_iteration:
            return None
        return self._rows_by_iteration[max(self._rows_by_iteration)]

    def _write(self) -> None:
        rows = [
            self._rows_by_iteration[iteration]
            for iteration in sorted(self._rows_by_iteration)
        ]
        fieldnames: list[str] = []
        for row in rows:
            for field_name in row:
                if field_name is not None and field_name not in fieldnames:
                    fieldnames.append(field_name)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="",
                    dir=self.run_dir,
                    prefix=".geometry_metrics.",
                    suffix=".tmp",
                    delete=False,
            ) as csv_file:
                temporary_path = Path(csv_file.name)
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            assert temporary_path is not None
            temporary_path.replace(self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def record(self, row: Mapping[str, Any], iteration: int) -> dict[str, Any]:
        persisted_row = dict(row)
        persisted_row["iteration"] = int(iteration)
        self._rows_by_iteration[int(iteration)] = persisted_row
        self._write()
        return persisted_row

    def evaluate(
            self,
            mesh_path: Path | None,
            ground_truth_path: Path | None,
            iteration: int,
            samples: int = 500_000,
            seed: int = 0,
            scale: float = 1.0,
            use_vertices: bool = False,
    ) -> Mapping[str, Any] | None:
        if mesh_path is None or ground_truth_path is None:
            return None

        mesh_path = Path(mesh_path).expanduser().resolve()
        ground_truth_path = Path(ground_truth_path).expanduser().resolve()
        if not mesh_path.is_file():
            warnings.warn(f"Skipping geometry metrics; mesh is missing: {mesh_path}", stacklevel=2)
            return None
        if not ground_truth_path.is_file():
            warnings.warn(
                f"Skipping geometry metrics; ground truth is missing: {ground_truth_path}",
                stacklevel=2,
            )
            return None

        try:
            from metrics.evaluate_runs import MeshCheckpoint, compute_geometry_rows

            rows = compute_geometry_rows(
                run_dir=self.run_dir,
                checkpoints=[MeshCheckpoint(iteration=int(iteration), mesh_path=mesh_path)],
                ground_truth_path=ground_truth_path,
                samples=int(samples),
                device_name="cpu",
                seed=int(seed),
                scale=float(scale),
                use_vertices=bool(use_vertices),
                print_each_score=False,
            )
            if not rows:
                warnings.warn(f"No geometry metrics were produced for {mesh_path}", stacklevel=2)
                return None

            return self.record(rows[0], iteration=iteration)
        except Exception as exception:
            warnings.warn(
                f"Geometry evaluation failed for {mesh_path}: {exception}",
                stacklevel=2,
            )
            return None
