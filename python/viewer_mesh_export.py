"""Prepare a private viewer snapshot, then run the existing TSDF extractor.

The viewer's owned-process supervisor supplies -o and --viewer-session. No scene
or point-cloud input is rewritten, and the extractor remains the sole mesh path.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET


def ascii_surfel_copy(source: Path, destination: Path) -> Path:
    """Convert viewer binary packets to the extractor's ASCII checkpoint format."""
    types = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
             "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
             "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
             "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
    with source.open("rb") as stream:
        header: list[str] = []
        properties: list[tuple[str, str]] = []
        count = None
        element = None
        encoding = None
        for _ in range(4096):
            line = stream.readline()
            if not line:
                raise ValueError("Missing PLY end_header")
            text = line.decode("ascii").strip()
            header.append(text)
            fields = text.split()
            if fields[:1] == ["format"]:
                encoding = fields[1]
            elif fields[:1] == ["element"]:
                element = fields[1]
                if element == "vertex":
                    count = int(fields[2])
                elif int(fields[2]) != 0:
                    raise ValueError("Export input must be a surfel point cloud, not a triangle mesh")
            elif fields[:1] == ["property"] and element == "vertex":
                if len(fields) != 3 or fields[1] not in types:
                    raise ValueError("Unsupported surfel PLY property")
                properties.append((fields[2], types[fields[1]]))
            if text == "end_header":
                break
        else:
            raise ValueError("PLY header is too large")
        if header[0] != "ply" or not count or count < 0 or not properties:
            raise ValueError("Export input contains no surfels")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if encoding == "ascii":
            stream.close()
            shutil.copyfile(source, destination)
            return destination
        if encoding not in {"binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"Unsupported PLY format: {encoding}")
        import numpy as np
        endian = "<" if encoding == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + kind) for name, kind in properties])
        formats = ["%.9g" if kind == "f4" else "%.17g" if kind == "f8" else "%d"
                   for _, kind in properties]
        with destination.open("w", encoding="ascii") as output:
            output.write("\n".join("format ascii 1.0" if line.startswith("format ") else line for line in header) + "\n")
            remaining = count
            while remaining:
                rows = min(32768, remaining)
                data = stream.read(rows * dtype.itemsize)
                if len(data) != rows * dtype.itemsize:
                    raise ValueError("Truncated surfel PLY payload")
                values = np.frombuffer(data, dtype=dtype, count=rows)
                np.savetxt(output, np.column_stack([values[name] for name, _ in properties]), fmt=formats)
                remaining -= rows
    return destination


def prepare_export(workspace: Path, destination: Path, source_config: Path | None = None) -> tuple[Path, Path]:
    from viewer_session import materialize_workspace
    destination.mkdir(parents=True, exist_ok=True)
    state = json.loads(workspace.read_text())
    source = Path(state["source_scene"]).expanduser().resolve()
    # The C++ launcher copied the source XML before launching. Resolve references
    # against its original directory, then materialize the frozen entity edits.
    source_base = Path(state.get("source_scene_base", source.parent))
    assets = Path(state["asset_root"]).expanduser().resolve()
    tree = ET.parse(source)
    for node in tree.getroot().iter("string"):
        if node.get("name") == "filename":
            path = Path(node.get("value", ""))
            if not path.is_absolute():
                candidate = source_base / path
                node.set("value", str((candidate if candidate.exists() else assets / path).resolve()))
    source_copy = destination / "source_scene.xml"
    tree.write(source_copy, encoding="utf-8", xml_declaration=True)
    state["source_scene"] = str(source_copy)
    prepared_workspace = destination / "workspace.json"
    prepared_workspace.write_text(json.dumps(state, indent=2))
    scene = destination / "scene.xml"
    materialize_workspace(prepared_workspace, scene)
    points = ascii_surfel_copy(Path(state["pointcloud"]), destination / "points.ply")
    if source_config and source_config.is_file():
        config = json.loads(source_config.read_text())
    else:
        from config import OptimizationConfig, RendererSettingsConfig
        config = dict(renderer_settings=RendererSettingsConfig().as_dict(OptimizationConfig()))
    config.update(assets_root=str(assets), scene_xml=str(scene), pointcloud_ply=str(points))
    (destination / "run_config.json").write_text(json.dumps(config, indent=2))
    return destination, points


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--viewer-session", type=Path, required=True)
    args, extractor = parser.parse_known_args(argv)
    return args, extractor


def main(argv=None):
    args, extra = parse_args(argv)
    print(f"Preparing private export input in {args.output}", flush=True)
    run, points = prepare_export(args.workspace.resolve(), args.output / "input", args.run_config)
    command = [sys.executable, "-u", str(Path(__file__).with_name("extract_mesh.py")),
               "--run-dir", str(run), "--ply", str(points),
               "--mesh-output-subdir", str(args.output.resolve() / "mesh"), *extra]
    print("Starting the existing TSDF mesh extractor", flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
