#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter an ASCII PLY file by removing vertices whose 'opacity' property "
            "is below a threshold. Preserves header and updates 'element vertex N'."
        )
    )
    parser.add_argument("--in", dest="inputPath", type=Path, required=True, help="Input .ply path (ASCII).")
    parser.add_argument("--out", dest="outputPath", type=Path, required=True, help="Output .ply path (ASCII).")
    parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.01,
        help="Remove vertices with opacity < threshold.",
    )
    parser.add_argument(
        "--keep-header-comments",
        action="store_true",
        help="Keep comment lines in the header (default: keep them anyway).",
    )
    return parser.parse_args()


def split_ply_header_and_body(allLines: List[str]) -> Tuple[List[str], List[str]]:
    headerLines: List[str] = []
    bodyLines: List[str] = []

    inHeader = True
    for line in allLines:
        if inHeader:
            headerLines.append(line)
            if line.strip() == "end_header":
                inHeader = False
        else:
            bodyLines.append(line)

    if not headerLines or headerLines[-1].strip() != "end_header":
        raise ValueError("Input does not look like a valid PLY (missing 'end_header').")

    return headerLines, bodyLines


def find_vertex_property_index(headerLines: List[str], propertyName: str) -> int:
    """
    Returns 0-based index of the property within the vertex element, e.g.:
    property float x
    property float opacity   -> returns its position in the vertex property list.
    """
    insideVertexElement = False
    vertexPropertyNames: List[str] = []

    for line in headerLines:
        stripped = line.strip()
        if stripped.startswith("element "):
            parts = stripped.split()
            if len(parts) >= 3 and parts[1] == "vertex":
                insideVertexElement = True
                vertexPropertyNames.clear()
            else:
                # leaving vertex section when a new element starts
                if insideVertexElement:
                    insideVertexElement = False

        if insideVertexElement and stripped.startswith("property "):
            # expected: property <type> <name>
            parts = stripped.split()
            if len(parts) >= 3:
                vertexPropertyNames.append(parts[2])

    if not vertexPropertyNames:
        raise ValueError("Could not find any 'property' lines inside 'element vertex' in header.")

    try:
        return vertexPropertyNames.index(propertyName)
    except ValueError as exc:
        raise ValueError(f"Vertex property '{propertyName}' not found. Found: {vertexPropertyNames}") from exc


def update_vertex_count_in_header(headerLines: List[str], newVertexCount: int) -> List[str]:
    updatedHeaderLines: List[str] = []
    replaced = False

    for line in headerLines:
        stripped = line.strip()
        if stripped.startswith("element vertex "):
            updatedHeaderLines.append(f"element vertex {newVertexCount}\n")
            replaced = True
        else:
            updatedHeaderLines.append(line)

    if not replaced:
        raise ValueError("Header missing 'element vertex N' line.")

    return updatedHeaderLines


def filter_vertices_by_opacity(
    bodyLines: List[str],
    opacityColumnIndex: int,
    opacityThreshold: float,
) -> List[str]:
    keptBodyLines: List[str] = []

    for lineNumber, line in enumerate(bodyLines, start=1):
        stripped = line.strip()
        if stripped == "":
            continue

        parts = stripped.split()
        if opacityColumnIndex >= len(parts):
            raise ValueError(
                f"Vertex line {lineNumber} has only {len(parts)} columns, "
                f"but opacity index is {opacityColumnIndex}."
            )

        try:
            opacityValue = float(parts[opacityColumnIndex])
        except ValueError as exc:
            raise ValueError(f"Failed parsing opacity on vertex line {lineNumber}: '{parts[opacityColumnIndex]}'") from exc

        if opacityValue >= opacityThreshold:
            keptBodyLines.append(line if line.endswith("\n") else line + "\n")

    return keptBodyLines


def main() -> None:
    args = parse_args()

    inputPath: Path = args.inputPath
    outputPath: Path = args.outputPath
    opacityThreshold: float = float(args.opacity_threshold)

    allText = inputPath.read_text(encoding="utf-8")
    allLines = [line + "\n" for line in allText.splitlines()]

    headerLines, bodyLines = split_ply_header_and_body(allLines)

    opacityColumnIndex = find_vertex_property_index(headerLines, "opacity")
    keptBodyLines = filter_vertices_by_opacity(bodyLines, opacityColumnIndex, opacityThreshold)

    updatedHeaderLines = update_vertex_count_in_header(headerLines, newVertexCount=len(keptBodyLines))

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    outputPath.write_text("".join(updatedHeaderLines + keptBodyLines), encoding="utf-8")

    originalVertexCount = len([l for l in bodyLines if l.strip() != ""])
    keptVertexCount = len(keptBodyLines)
    removedVertexCount = originalVertexCount - keptVertexCount

    print(f"Input:  {inputPath}")
    print(f"Output: {outputPath}")
    print(f"Opacity threshold: {opacityThreshold}")
    print(f"Vertices: kept {keptVertexCount} / {originalVertexCount} (removed {removedVertexCount})")


if __name__ == "__main__":
    main()
