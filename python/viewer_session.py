"""Local protocol for a viewer-launched training process; no run discovery/attach API."""
from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
TUNABLE_FIELDS = (
    "learning_rate_position", "learning_rate_rotation", "learning_rate_scale",
    "learning_rate_albedo", "learning_rate_opacity", "learning_rate_beta",
    "alpha_loss_weight", "depth_distort_weight", "normal_consistency_weight",
    "intra_slab_depth_weight",
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, allow_nan=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_patch(patch: Any) -> dict[str, float]:
    if not isinstance(patch, dict) or not patch or set(patch) - set(TUNABLE_FIELDS):
        raise ValueError("Unsupported training parameter patch")
    result = {}
    for key, value in patch.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and non-negative")
        result[key] = value
    return result


def write_preview(path: Path, arrays, metadata=None) -> int:
    """Binary PLY, using the same property names/order as the persistent writer."""
    import numpy as np
    positions, rotations, scales, colors, opacity, beta, power, *materials = arrays
    if len(materials) not in (0, 2):
        raise ValueError("Expected both roughness and metalness in a material preview")
    count = len(positions)
    metadata=metadata or {}
    has_age="primitive_age" in metadata
    material_names = ("roughness", "metalness") if materials else ()
    origin_column = 16 + len(material_names)
    data = np.empty((count, origin_column + 1 + has_age), dtype="<f4")
    data[:, :3] = positions
    # Training quaternions use w,x,y,z; the loader reads these names explicitly.
    data[:, 3:7] = rotations
    data[:, 7:9] = scales
    data[:, 9:12] = colors
    for column, values in [(12, opacity), (13, beta), (15, power)]:
        data[:, column] = np.asarray(values).reshape(-1)
    data[:, 14] = 0.0  # shape
    for column, values in enumerate(materials, start=16):
        data[:, column] = np.asarray(values).reshape(-1)
    data[:, origin_column] = metadata.get("densification_origin", 0)
    if has_age:
        data[:, origin_column + 1] = metadata["primitive_age"]
    names = ("x", "y", "z", "rot_w", "rot_x", "rot_y", "rot_z", "su", "sv",
             "albedo_r", "albedo_g", "albedo_b", "opacity", "beta", "shape", "power",
             *material_names, "densification_origin")
    if has_age:
        names+=("primitive_age",)
    # Match the canonical PLY names (scale/albedo names are intentionally explicit).
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % count
    header += "".join("property float %s\n" % name for name in names) + "end_header\n"
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        data.tofile(stream)
    return count


class ViewerSession:
    def __init__(self, directory: Path, interval: float = 1.0):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.interval = max(float(interval), 0.1)
        self.state = {"protocol": PROTOCOL_VERSION, "pid": os.getpid(), "state": "initializing",
                      "iteration": 0, "ack": 0, "message": "Initializing training"}
        self._last_status = 0.0
        self._last_preview = 0.0
        self._generation = 0
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._paused = False
        self.stop_requested = False
        self.save_requested = False
        self.save_mesh_requested = False
        self._write_status(force=True)

    def _write_status(self, *, force=False):
        now = time.monotonic()
        if force or now - self._last_status >= 0.25:
            self.state["updated_at"] = time.time()
            if self._error:
                self.state["preview_error"] = self._error
            atomic_json(self.directory / "status.json", self.state)
            self._last_status = now

    def configure(self, config):
        self.state.update(run_dir=str(config.output_dir.resolve()), scene=str(Path(config.scene_xml).resolve()),
                          target_iteration=config.resume_iteration_offset + config.iterations,
                          ground_truth=str(config.ground_truth) if getattr(config,"ground_truth",None) else None,
                          mesh_extraction_interval=getattr(config,"mesh_extraction_interval",0),
                          parameters={key: float(getattr(config, key)) for key in TUNABLE_FIELDS})
        self._write_status(force=True)

    @contextmanager
    def monitor_adjoint_kernels(self, renderer_module):
        # Older locally selected modules can still launch training.
        message = getattr(renderer_module, "adjoint_kernel_progress_message", None)
        if message is None:
            yield
            return

        stop = threading.Event()
        path = self.directory / "adjoint_progress.json"

        def publish_progress():
            previous = ""
            while not stop.wait(0.1):
                current = message()
                if current != previous:
                    try:
                        atomic_json(path, {"message": current})
                    except OSError:
                        return
                    previous = current

        renderer_module.enable_adjoint_kernel_progress(True)
        worker = threading.Thread(target=publish_progress, name="adjoint-progress", daemon=True)
        worker.start()
        try:
            yield
        finally:
            stop.set()
            worker.join()
            renderer_module.enable_adjoint_kernel_progress(False)
            with suppress(OSError):
                path.unlink(missing_ok=True)

    def boundary(self, config, iteration: int) -> dict[str, float]:
        """Commands are consumed at iteration boundaries, never inside kernels."""
        changed = {}
        while True:
            request_path = self.directory / "command.json"
            try:
                request = json.loads(request_path.read_text())
                sequence = int(request["sequence"])
                if sequence > int(self.state["ack"]):
                    command = request["command"]
                    if command == "tune":
                        patch = validate_patch(request.get("parameters"))
                        with (Path(self.state["run_dir"]) / "parameter_changes.jsonl").open("a") as history:
                            history.write(json.dumps({"iteration": iteration + 1, "sequence": sequence,
                                                      "parameters": patch}) + "\n")
                        for key, value in patch.items():
                            setattr(config, key, value)
                        changed.update(patch)
                        self.state["parameters"].update(patch)
                    elif command == "pause":
                        self._paused = True
                    elif command == "resume":
                        self._paused = False
                    elif command == "stop":
                        self.stop_requested = True
                        self._paused = False
                    elif command == "save":
                        include_mesh = request.get("include_mesh", False)
                        if not isinstance(include_mesh, bool):
                            raise ValueError("include_mesh must be a boolean")
                        self.save_requested = True
                        self.save_mesh_requested |= include_mesh
                    else:
                        raise ValueError("Unknown viewer command")
                    self.state.update(ack=sequence, command_error="", applied_at_iteration=iteration + 1)
            except FileNotFoundError:
                pass
            except (ValueError, KeyError, TypeError, OSError) as error:
                # A rejected command is acknowledged so it cannot wedge the session.
                self.state["command_error"] = str(error)
                if isinstance(locals().get("request"), dict):
                    try:
                        self.state["ack"] = max(int(self.state["ack"]), int(request.get("sequence", 0)))
                    except (ValueError, TypeError):
                        pass
            self.state["state"] = "stopping" if self.stop_requested else "paused" if self._paused else "running"
            self._write_status(force=self._paused or bool(changed) or self.stop_requested)
            if not self._paused:
                return changed
            time.sleep(0.1)

    def progress(self, iteration: int, loss_state: dict, point_count: int, iteration_seconds: float):
        self.state.update(iteration=int(iteration), points=int(point_count),
                          loss=float(loss_state.get("total_loss_value", 0)),
                          rgb_loss=float(loss_state.get("total_rgb_loss_value", 0)),
                          iteration_ms=1000 * iteration_seconds)
        # Non-finite numerical failures must remain reportable as valid JSON.
        for key in ("loss", "rgb_loss", "iteration_ms"):
            if not math.isfinite(self.state[key]):
                self.state[key] = None
        self._write_status()

    def preview_due(self):
        return (time.monotonic() - self._last_preview >= self.interval
                and (self._thread is None or not self._thread.is_alive()))

    def publish(self, iteration: int, tensors, *, force=False, metadata=None):
        """Own CPU copies before handing serialization to the one-slot worker."""
        if force and self._thread:
            self._thread.join()
        if not force and not self.preview_due():
            return
        import numpy as np
        arrays = tuple(np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t).copy()
                       for t in tensors)
        owned_metadata={key:np.asarray(value).copy() for key,value in (metadata or {}).items()}
        self._last_preview = time.monotonic()
        self._generation += 1
        generation = self._generation

        def work():
            temporary = self.directory / f".live-{generation}.tmp"
            destination = self.directory / f"live-{generation}.ply"
            try:
                count = write_preview(temporary, arrays, owned_metadata)
                os.replace(temporary, destination)
                atomic_json(self.directory / "live.json", {
                    "protocol": PROTOCOL_VERSION, "generation": generation,
                    "iteration": int(iteration), "points": count,
                    "pointcloud": str(destination), "scene": self.state["scene"],
                    "run_dir": self.state["run_dir"],
                })
                # Keep a grace window for readers loading the previous publication.
                for old in self.directory.glob("live-*.ply"):
                    try:
                        old_generation = int(old.stem.split("-")[-1])
                        if old_generation < generation - 2:
                            old.unlink(missing_ok=True)
                    except ValueError:
                        continue
            except Exception as error:
                self._error = str(error)
            finally:
                temporary.unlink(missing_ok=True)

        self._thread = threading.Thread(target=work, name="viewer-preview", daemon=True)
        self._thread.start()

    def finish(self, error: BaseException | None = None):
        if self._thread:
            self._thread.join()
        self.state.update(state="failed" if error else "stopped" if self.stop_requested else "completed",
                          message=str(error) if error else "Training finished")
        self._write_status(force=True)

    def finalizing(self):
        self.state.update(state="finalizing", message="Saving final reconstruction")
        self._write_status(force=True)


def materialize_workspace(workspace: Path, destination: Path) -> dict:
    """Turn an editable viewer bundle into a private scene XML for a new run."""
    import xml.etree.ElementTree as ET
    state = json.loads(Path(workspace).read_text())
    source = Path(state["source_scene"])
    tree = ET.parse(source)
    root = tree.getroot()
    # Moving XML must not change the resolution of mesh and texture references.
    for node in root.iter("string"):
        if node.get("name") == "filename":
            value = Path(node.get("value", ""))
            if not value.is_absolute():
                from_scene = source.parent / value
                from_assets = Path(state["asset_root"]) / value
                node.set("value", str((from_scene if from_scene.exists() else from_assets).resolve()))
    for entity in state["entities"]:
        shape = next((n for n in root.findall("shape") if n.get("id") == entity["name"]), None)
        if shape is None:
            if not entity.get("mesh_file"):
                continue
            shape = ET.SubElement(root, "shape", type="ply", id=entity["name"])
            ET.SubElement(shape, "string", name="filename", value=entity["mesh_file"])
        for old in list(shape.findall("transform")):
            shape.remove(old)
        transform = ET.SubElement(shape, "transform", name="to_world")
        ET.SubElement(transform, "translate", **dict(zip(("x", "y", "z"), map(str, entity["translation"]))))
        ET.SubElement(transform, "scale", **dict(zip(("x", "y", "z"), map(str, entity["scale"]))))
        x, y, z, w = entity["rotation_xyzw"]
        length = math.sqrt(x*x + y*y + z*z + w*w)
        if length > 0:
            x, y, z, w = x/length, y/length, z/length, w/length
        else:
            x, y, z, w = 0, 0, 0, 1
        sine = math.sqrt(max(0.0, 1-w*w))
        axis = (x/sine, y/sine, z/sine) if sine > 1e-8 else (0, 0, 1)
        ET.SubElement(transform, "rotate", x=str(axis[0]), y=str(axis[1]), z=str(axis[2]),
                      angle=str(math.degrees(2*math.acos(max(-1.0, min(1.0, w))))))
        if "emitter" in entity:
            emitter = shape.find("emitter")
            if emitter is None:
                emitter = ET.SubElement(shape, "emitter", type="area")
            for old in list(emitter):
                if old.get("name") in {"color", "radiance", "power"}:
                    emitter.remove(old)
            color = ", ".join(map(str, entity["emitter"]["color"]))
            ET.SubElement(emitter, "rgb", name="color", value=color)
            ET.SubElement(emitter, "rgb", name="radiance", value=color)
            ET.SubElement(emitter, "float", name="power", value=str(entity["emitter"]["power"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return state
