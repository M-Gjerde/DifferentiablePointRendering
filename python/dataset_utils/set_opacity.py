#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set all opacity values in an ASCII surfel PLY file to a constant value."
    )
    parser.add_argument(
        "input_ply",
        type=Path,
        help="Path to the input .ply file.",
    )
    parser.add_argument(
        "--output-ply",
        type=Path,
        default=None,
        help="Path to the modified output .ply file. Defaults to '<input>_opacity1.ply'.",
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
        "--opacity-value",
        type=float,
        default=1.0,
        help="Opacity value to write. Default: 1.0.",
    )
    parser.add_argument(
        "--fallback-opacity-index",
        type=int,
        default=14,
        help="Fallback opacity column index if no 'opacity' property is found in the PLY header. Default: 14.",
    )
    return parser.parse_args()


def default_output_path(input_ply_path: Path) -> Path:
    return input_ply_path.with_name(f"{input_ply_path.stem}1{input_ply_path.suffix}")


def inspect_ply_header(input_ply_path: Path) -> tuple[list[str], int, int, bool]:
    header_lines: list[str] = []
    vertex_count = -1
    opacity_property_index = -1
    is_ascii = False

    inside_vertex_element = False
    current_vertex_property_index = 0

    with input_ply_path.open("r", encoding="utf-8") as file_handle:
        first_line = file_handle.readline()
        if first_line.strip() != "ply":
            raise ValueError(f"File does not look like a PLY file: {input_ply_path}")

        header_lines.append(first_line)

        for line in file_handle:
            header_lines.append(line)
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
                if property_name == "opacity":
                    opacity_property_index = current_vertex_property_index
                current_vertex_property_index += 1

            if stripped_line == "end_header":
                break

    if not is_ascii:
        raise ValueError("PLY header did not declare ASCII format.")

    if vertex_count < 0:
        raise ValueError("Could not find 'element vertex <count>' in PLY header.")

    return header_lines, vertex_count, opacity_property_index, is_ascii


def set_opacity_in_ply(
    input_ply_path: Path,
    output_ply_path: Path,
    opacity_value: float,
    fallback_opacity_index: int,
) -> None:
    if not input_ply_path.is_file():
        raise FileNotFoundError(f"Input PLY file does not exist: {input_ply_path}")

    header_lines, vertex_count, opacity_property_index, _ = inspect_ply_header(input_ply_path)

    if opacity_property_index < 0:
        opacity_property_index = fallback_opacity_index
        print(
            f"No 'opacity' property found in header. "
            f"Using fallback opacity column index {opacity_property_index}."
        )

    opacity_string = f"{opacity_value:.9g}"
    end_header_seen = False
    modified_vertex_count = 0

    output_ply_path.parent.mkdir(parents=True, exist_ok=True)

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
                    f"Unexpected end of file while reading vertex {vertex_index} "
                    f"of {vertex_count}."
                )

            stripped_line = line.strip()
            if not stripped_line:
                output_file.write(line)
                continue

            parts = stripped_line.split()
            if len(parts) <= opacity_property_index:
                raise ValueError(
                    f"Vertex line {vertex_index} has {len(parts)} columns, "
                    f"but opacity index is {opacity_property_index}."
                )

            parts[opacity_property_index] = opacity_string
            output_file.write(" ".join(parts) + "\n")
            modified_vertex_count += 1

        for line in input_file:
            output_file.write(line)

    print(f"Modified {modified_vertex_count} vertices.")
    print(f"Wrote: {output_ply_path}")


def set_opacity_in_place(
    input_ply_path: Path,
    opacity_value: float,
    fallback_opacity_index: int,
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
        set_opacity_in_ply(
            input_ply_path=input_ply_path,
            output_ply_path=temporary_output_path,
            opacity_value=opacity_value,
            fallback_opacity_index=fallback_opacity_index,
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
        set_opacity_in_place(
            input_ply_path=input_ply_path,
            opacity_value=args.opacity_value,
            fallback_opacity_index=args.fallback_opacity_index,
            create_backup=not args.no_backup,
        )
        return

    output_ply_path = args.output_ply
    if output_ply_path is None:
        output_ply_path = default_output_path(input_ply_path)

    set_opacity_in_ply(
        input_ply_path=input_ply_path,
        output_ply_path=output_ply_path,
        opacity_value=args.opacity_value,
        fallback_opacity_index=args.fallback_opacity_index,
    )


if __name__ == "__main__":
    main()