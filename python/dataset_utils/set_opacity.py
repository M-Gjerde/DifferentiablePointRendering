#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


SCALE_PROPERTY_NAMES = {"su", "sv"}
LEGACY_SCALE_PROPERTY_PREFIX = "scale_"


def parse_indices(raw_indices: str | None) -> list[int]:
    if raw_indices is None:
        return []

    parsed_indices: list[int] = []
    for raw_index in raw_indices.split(","):
        stripped_index = raw_index.strip()
        if stripped_index:
            parsed_indices.append(int(stripped_index))

    return parsed_indices


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set opacity and/or su/sv scale values in an ASCII 2D Gaussian splat PLY file."
        )
    )
    parser.add_argument(
        "input_ply",
        nargs="?",
        type=Path,
        help="Path to the input .ply file.",
    )
    parser.add_argument(
        "--input-ply",
        dest="input_ply_option",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-ply",
        type=Path,
        default=None,
        help="Path to the modified output .ply file. Defaults to '<input>_<modified-properties>.ply'.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file. A backup is written unless --no-backup is passed.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak backup when using --in-place.",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=None,
        help="Optional opacity value to write to the 'opacity' property.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Optional scale value to write to both 'su' and 'sv'.",
    )
    parser.add_argument(
        "--fallback-opacity-index",
        type=int,
        default=None,
        help="Fallback opacity column index if no 'opacity' property is found in the PLY header.",
    )
    parser.add_argument(
        "--fallback-scale-indices",
        type=parse_indices,
        default=[],
        help=(
            "Comma-separated fallback scale column indices if no 'su'/'sv' properties are found, "
            "for example '9,10'."
        ),
    )

    args = parser.parse_args()

    if args.input_ply is None:
        args.input_ply = args.input_ply_option

    if args.input_ply is None:
        parser.error("input_ply is required")

    if args.opacity is None and args.scale is None:
        parser.error("Pass at least one modification: --opacity VALUE and/or --scale VALUE")

    if args.in_place and args.output_ply is not None:
        parser.error("--output-ply cannot be used together with --in-place")

    return args


def default_output_path(input_ply_path: Path, opacity: float | None, scale: float | None) -> Path:
    modified_property_names: list[str] = []
    if scale is not None:
        modified_property_names.append("scale")
    if opacity is not None:
        modified_property_names.append("opacity")

    suffix_name = "_".join(modified_property_names)
    return input_ply_path.with_name(f"{input_ply_path.stem}_{suffix_name}{input_ply_path.suffix}")


def inspect_ply_header(input_ply_path: Path) -> tuple[int, dict[str, int]]:
    vertex_count = -1
    vertex_property_indices: dict[str, int] = {}
    is_ascii = False

    inside_vertex_element = False
    current_vertex_property_index = 0

    with input_ply_path.open("r", encoding="utf-8") as file_handle:
        first_line = file_handle.readline()
        if first_line.strip() != "ply":
            raise ValueError(f"File does not look like a PLY file: {input_ply_path}")

        for line in file_handle:
            stripped_line = line.strip()
            tokens = stripped_line.split()

            if len(tokens) >= 2 and tokens[0] == "format":
                if tokens[1] == "ascii":
                    is_ascii = True
                else:
                    raise ValueError(
                        f"Only ASCII PLY files are supported. Found format: {tokens[1]}"
                    )

            if len(tokens) >= 3 and tokens[0] == "element":
                inside_vertex_element = tokens[1] == "vertex"
                if inside_vertex_element:
                    vertex_count = int(tokens[2])
                    current_vertex_property_index = 0
                continue

            if inside_vertex_element and len(tokens) >= 3 and tokens[0] == "property":
                property_name = tokens[-1]
                vertex_property_indices[property_name] = current_vertex_property_index
                current_vertex_property_index += 1
                continue

            if stripped_line == "end_header":
                break

    if not is_ascii:
        raise ValueError("PLY header did not declare ASCII format.")

    if vertex_count < 0:
        raise ValueError("Could not find 'element vertex <count>' in PLY header.")

    return vertex_count, vertex_property_indices


def resolve_opacity_index(
    vertex_property_indices: dict[str, int],
    fallback_opacity_index: int | None,
) -> int:
    opacity_property_index = vertex_property_indices.get("opacity")
    if opacity_property_index is not None:
        return opacity_property_index

    if fallback_opacity_index is not None:
        print(
            "No 'opacity' property found in header. "
            f"Using fallback opacity column index {fallback_opacity_index}."
        )
        return fallback_opacity_index

    raise ValueError(
        "No 'opacity' property found in the PLY header. "
        "Either add an opacity property or pass --fallback-opacity-index."
    )


def resolve_scale_indices(
    vertex_property_indices: dict[str, int],
    fallback_scale_indices: list[int],
) -> list[int]:
    scale_property_indices = [
        property_index
        for property_name, property_index in vertex_property_indices.items()
        if property_name in SCALE_PROPERTY_NAMES
    ]

    if scale_property_indices:
        return sorted(scale_property_indices)

    # Compatibility with the older version of this helper script.
    legacy_scale_property_indices = [
        property_index
        for property_name, property_index in vertex_property_indices.items()
        if property_name == "scale" or property_name.startswith(LEGACY_SCALE_PROPERTY_PREFIX)
    ]

    if legacy_scale_property_indices:
        return sorted(legacy_scale_property_indices)

    if fallback_scale_indices:
        print(
            "No 'su'/'sv' scale properties found in header. "
            f"Using fallback scale column indices {fallback_scale_indices}."
        )
        return fallback_scale_indices

    raise ValueError(
        "No 'su'/'sv' scale properties found in the PLY header. "
        "Either add su/sv properties or pass --fallback-scale-indices, for example '9,10'."
    )


def set_requested_properties_in_ply(
    input_ply_path: Path,
    output_ply_path: Path,
    opacity: float | None,
    scale: float | None,
    fallback_opacity_index: int | None,
    fallback_scale_indices: list[int],
) -> None:
    if not input_ply_path.is_file():
        raise FileNotFoundError(f"Input PLY file does not exist: {input_ply_path}")

    vertex_count, vertex_property_indices = inspect_ply_header(input_ply_path)

    opacity_property_index: int | None = None
    if opacity is not None:
        opacity_property_index = resolve_opacity_index(
            vertex_property_indices=vertex_property_indices,
            fallback_opacity_index=fallback_opacity_index,
        )

    scale_property_indices: list[int] = []
    if scale is not None:
        scale_property_indices = resolve_scale_indices(
            vertex_property_indices=vertex_property_indices,
            fallback_scale_indices=fallback_scale_indices,
        )

    opacity_string = f"{opacity:.9g}" if opacity is not None else None
    scale_string = f"{scale:.9g}" if scale is not None else None

    output_ply_path.parent.mkdir(parents=True, exist_ok=True)

    end_header_seen = False
    modified_vertex_count = 0

    with input_ply_path.open("r", encoding="utf-8") as input_file, output_ply_path.open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        for line in input_file:
            output_file.write(line)
            if line.strip() == "end_header":
                end_header_seen = True
                break

        if not end_header_seen:
            raise ValueError("PLY header ended before 'end_header' was found.")

        for vertex_index in range(vertex_count):
            line = input_file.readline()
            if line == "":
                raise ValueError(
                    f"Unexpected end of file while reading vertex {vertex_index} of {vertex_count}."
                )

            stripped_line = line.strip()
            if not stripped_line:
                output_file.write(line)
                continue

            parts = stripped_line.split()

            if opacity_property_index is not None:
                if len(parts) <= opacity_property_index:
                    raise ValueError(
                        f"Vertex line {vertex_index} has {len(parts)} columns, "
                        f"but opacity index is {opacity_property_index}."
                    )
                assert opacity_string is not None
                parts[opacity_property_index] = opacity_string

            if scale_string is not None:
                for scale_property_index in scale_property_indices:
                    if len(parts) <= scale_property_index:
                        raise ValueError(
                            f"Vertex line {vertex_index} has {len(parts)} columns, "
                            f"but scale index is {scale_property_index}."
                        )
                    parts[scale_property_index] = scale_string

            output_file.write(" ".join(parts) + "\n")
            modified_vertex_count += 1

        for line in input_file:
            output_file.write(line)

    modified_properties: list[str] = []
    if scale is not None:
        modified_properties.append(f"scale columns {scale_property_indices}")
    if opacity is not None:
        modified_properties.append(f"opacity column {opacity_property_index}")

    print(f"Modified {modified_vertex_count} vertices.")
    print(f"Modified properties: {', '.join(modified_properties)}")
    print(f"Wrote: {output_ply_path}")


def set_requested_properties_in_place(
    input_ply_path: Path,
    opacity: float | None,
    scale: float | None,
    fallback_opacity_index: int | None,
    fallback_scale_indices: list[int],
    create_backup: bool,
) -> None:
    if create_backup:
        backup_path = input_ply_path.with_suffix(input_ply_path.suffix + ".bak")
        shutil.copy2(input_ply_path, backup_path)
        print(f"Backup written to: {backup_path}")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".ply",
        prefix=input_ply_path.stem + "_tmp_",
        dir=input_ply_path.parent,
        delete=False,
        encoding="utf-8",
    ) as temporary_file:
        temporary_output_path = Path(temporary_file.name)

    try:
        set_requested_properties_in_ply(
            input_ply_path=input_ply_path,
            output_ply_path=temporary_output_path,
            opacity=opacity,
            scale=scale,
            fallback_opacity_index=fallback_opacity_index,
            fallback_scale_indices=fallback_scale_indices,
        )
        temporary_output_path.replace(input_ply_path)
        print(f"Overwrote input file: {input_ply_path}")
    except Exception:
        if temporary_output_path.exists():
            temporary_output_path.unlink()
        raise


def main() -> None:
    args = parse_arguments()

    input_ply_path: Path = args.input_ply

    if args.in_place:
        set_requested_properties_in_place(
            input_ply_path=input_ply_path,
            opacity=args.opacity,
            scale=args.scale,
            fallback_opacity_index=args.fallback_opacity_index,
            fallback_scale_indices=args.fallback_scale_indices,
            create_backup=not args.no_backup,
        )
        return

    output_ply_path = args.output_ply
    if output_ply_path is None:
        output_ply_path = default_output_path(
            input_ply_path=input_ply_path,
            opacity=args.opacity,
            scale=args.scale,
        )

    if output_ply_path.resolve() == input_ply_path.resolve():
        raise ValueError("Output path equals input path. Use --in-place if you want to overwrite.")

    set_requested_properties_in_ply(
        input_ply_path=input_ply_path,
        output_ply_path=output_ply_path,
        opacity=args.opacity,
        scale=args.scale,
        fallback_opacity_index=args.fallback_opacity_index,
        fallback_scale_indices=args.fallback_scale_indices,
    )


if __name__ == "__main__":
    main()