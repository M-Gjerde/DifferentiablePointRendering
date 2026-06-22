from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 6-panel optimization GIF from the latest optimization run.\n"
            "Panels: target | initial render | final render | optimization sequence | median depth | loss curve"
        )
    )
    parser.add_argument(
        "--optimization-output-root",
        type=Path,
        default=Path("OptimizationOutput"),
        help="Root folder containing timestamped optimization runs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit run directory. If omitted, the latest run is used.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index of camera to use among discovered camera folders.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="GIF framerate.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="optimization_summary.gif",
        help="Output GIF filename inside the run directory.",
    )
    parser.add_argument(
        "--loss-column",
        type=str,
        default=None,
        help=(
            "Optional explicit loss column from metrics.csv. "
            "Defaults to loss_total_mean, then loss_rgb_mean, then the weighted mean regularizers."
        ),
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=640,
        help="Width of each panel in the final GIF.",
    )
    parser.add_argument(
        "--panel-height",
        type=int,
        default=480,
        help="Height of each panel in the final GIF.",
    )
    parser.add_argument(
        "--title-height",
        type=int,
        default=40,
        help="Extra height reserved for a title above each panel.",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="GIF loop count. 0 = infinite.",
    )
    parser.add_argument(
        "--frame-stride", "--stride",
        type=int,
        default=1,
        help="Use every N-th render frame. Example: 10 means use frames 0, 10, 20, ...",
    )
    parser.add_argument(
        "--max-gif-frames",
        type=int,
        default=None,
        help="Optional maximum number of GIF frames. Frames are evenly sampled.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help=(
            "Zero-based index of the run to use when --run-dir is omitted. "
            "0 = latest, 1 = second latest, 2 = third latest, ..."
        ),
    )
    parser.add_argument(
        "--last-frame-hold-seconds",
        type=float,
        default=10.0,
        help="Hold the final GIF frame for this many seconds before looping.",
    )

    return parser.parse_args()


def parse_run_timestamp(run_dir_name: str) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", run_dir_name)
    if match is None:
        return None

    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None

def find_run_dir_by_index(optimization_output_root: Path, run_index: int) -> Path:
    if run_index < 0:
        raise ValueError(f"--index must be >= 0, got {run_index}")

    if not optimization_output_root.exists():
        raise FileNotFoundError(f"OptimizationOutput folder does not exist: {optimization_output_root}")

    candidate_run_dirs = []

    for child in optimization_output_root.iterdir():
        if not child.is_dir():
            continue

        metrics_csv_path = child / "metrics.csv"
        if not metrics_csv_path.exists():
            continue

        candidate_run_dirs.append(
            {
                "run_dir": child,
                "parsed_timestamp": parse_run_timestamp(child.name),
                "modified_time": metrics_csv_path.stat().st_mtime,
            }
        )

    if not candidate_run_dirs:
        raise FileNotFoundError(
            f"No run folders with metrics.csv found under: {optimization_output_root}"
        )

    candidate_run_dirs.sort(
        key=lambda item: (
            item["parsed_timestamp"] is not None,
            item["parsed_timestamp"] if item["parsed_timestamp"] is not None else datetime.min,
            item["modified_time"],
        ),
        reverse=True,
    )

    if run_index >= len(candidate_run_dirs):
        available_runs = [
            f"[{candidate_index}] {candidate['run_dir'].name}"
            for candidate_index, candidate in enumerate(candidate_run_dirs)
        ]

        raise IndexError(
            f"--index {run_index} is out of range. "
            f"Found {len(candidate_run_dirs)} run folders with metrics.csv.\n"
            "Available runs:\n" + "\n".join(available_runs)
        )

    return candidate_run_dirs[run_index]["run_dir"]


def csv_value_is_true(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def filter_metrics_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return the rows representing the global/averaged optimization metric.

    Current training writes one row per iteration with ``*_mean`` losses. For
    one-camera-per-iteration training, the first few rows only average over the
    cameras visited so far; once the camera cache is complete,
    ``loss_average_is_complete`` becomes true. Prefer only those complete
    averages, while retaining all rows for short/interrupted runs that never
    reach a complete average.

    Older metrics files may instead contain one row per camera and mark the
    aggregate row as ``camera_name == ALL_CAMERAS``. That format remains
    supported for old runs.
    """
    if dataframe.empty:
        return dataframe.copy()

    filtered = dataframe.copy()

    if "loss_average_is_complete" in filtered.columns:
        complete_mask = filtered["loss_average_is_complete"].map(csv_value_is_true)
        if bool(complete_mask.any()):
            filtered = filtered.loc[complete_mask].copy()

    if "camera_name" in filtered.columns:
        aggregate_mask = filtered["camera_name"].astype(str) == "ALL_CAMERAS"
        if bool(aggregate_mask.any()):
            filtered = filtered.loc[aggregate_mask].copy()

    if "iteration" not in filtered.columns:
        return filtered

    filtered["iteration"] = pd.to_numeric(filtered["iteration"], errors="coerce")
    filtered = filtered.loc[np.isfinite(filtered["iteration"])].copy()
    return filtered.sort_values("iteration").reset_index(drop=True)


def select_loss_column(dataframe: pd.DataFrame, explicit_loss_column: str | None) -> str:
    if explicit_loss_column is not None:
        if explicit_loss_column not in dataframe.columns:
            raise ValueError(
                f"Requested loss column '{explicit_loss_column}' not found. "
                f"Available columns: {list(dataframe.columns)}"
            )
        return explicit_loss_column

    preferred = [
        # Current averaged metrics.csv format.
        "loss_total_mean",
        "loss_rgb_mean",
        "loss_visibility_weighted_opacity_weighted_mean",
        "loss_normal_consistency_weighted_mean",
        "loss_depth_distortion_weighted_mean",
        "loss_visibility_weighted_opacity_raw_mean",
        "loss_normal_consistency_raw_mean",
        "loss_depth_distortion_raw_mean",

        # Backward compatibility with previous metrics.csv formats.
        "loss_total_sum",
        "loss_rgb_sum",
        "loss_visibility_weighted_opacity_weighted_sum",
        "loss_normal_consistency_weighted_sum",
        "loss_depth_distortion_weighted_sum",
        "loss_visibility_weighted_opacity_raw_sum",
        "loss_normal_consistency_raw_sum",
        "loss_depth_distortion_raw_sum",
        "loss_depth_distortion_sum",
        "loss_l2_window_mean",
        "loss_l2_current_camera",
        "loss_l2_window_sum_scaled",
    ]

    for column_name in preferred:
        if column_name in dataframe.columns:
            return column_name

    raise ValueError(
        "No supported loss column found. "
        f"Available columns: {list(dataframe.columns)}"
    )


def discover_camera_names(run_dir: Path) -> List[str]:
    camera_names = []

    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue

        render_dir = child / "render"
        if render_dir.is_dir():
            camera_names.append(child.name)

    if not camera_names:
        raise FileNotFoundError(f"No camera folders with a render/ subfolder found in: {run_dir}")

    return camera_names


def make_loss_curve_image(
    metrics_csv_path: Path,
    output_png_path: Path,
    explicit_loss_column: str | None,
    width: int,
    height: int,
) -> Tuple[Path, str]:
    dataframe = pd.read_csv(metrics_csv_path)
    dataframe = filter_metrics_rows(dataframe)

    if "iteration" not in dataframe.columns:
        raise ValueError("metrics.csv does not contain an 'iteration' column")

    loss_column = select_loss_column(dataframe, explicit_loss_column)
    loss_values = pd.to_numeric(dataframe[loss_column], errors="coerce")

    valid_mask = np.isfinite(dataframe["iteration"]) & np.isfinite(loss_values)
    dataframe = dataframe.loc[valid_mask].copy()
    dataframe[loss_column] = loss_values.loc[valid_mask]

    if dataframe.empty:
        raise ValueError(
            f"No finite values were found for '{loss_column}' after filtering metrics.csv."
        )

    plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    plt.plot(dataframe["iteration"], dataframe[loss_column], linewidth=2.0)
    plt.xlabel("Iteration")
    plt.ylabel(loss_column)
    plt.title("Mean loss curve" if loss_column.endswith("_mean") else "Loss curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=100)
    plt.close()

    return output_png_path, loss_column


def load_image_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")

    return Image.open(path).convert("RGB")


def make_placeholder_image(
    text: str,
    width: int,
    height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    image = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(image)

    lines = text.split("\n")
    line_heights = []

    for line in lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_heights.append(bbox[3] - bbox[1])
        except Exception:
            _, line_height = draw.textsize(line, font=font)
            line_heights.append(line_height)

    total_height = sum(line_heights) + max(0, len(lines) - 1) * 8
    y = (height - total_height) // 2

    for line, line_height in zip(lines, line_heights):
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
        except Exception:
            text_width, _ = draw.textsize(line, font=font)

        x = (width - text_width) // 2
        draw.text((x, y), line, fill=(220, 220, 220), font=font)
        y += line_height + 8

    return image


def fit_image_to_panel(image: Image.Image, panel_width: int, panel_height: int) -> Image.Image:
    src_width, src_height = image.size

    if src_width <= 0 or src_height <= 0:
        return Image.new("RGB", (panel_width, panel_height), (0, 0, 0))

    scale = min(panel_width / src_width, panel_height / src_height)
    new_width = max(1, int(round(src_width * scale)))
    new_height = max(1, int(round(src_height * scale)))

    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (panel_width, panel_height), (20, 20, 20))

    offset_x = (panel_width - new_width) // 2
    offset_y = (panel_height - new_height) // 2

    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def draw_panel_title(
    panel_image: Image.Image,
    title: str,
    panel_width: int,
    title_height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    out = Image.new("RGB", (panel_width, title_height + panel_image.height), (30, 30, 30))
    out.paste(panel_image, (0, title_height))

    draw = ImageDraw.Draw(out)

    try:
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except Exception:
        text_width, text_height = draw.textsize(title, font=font)

    x = (panel_width - text_width) // 2
    y = max(0, (title_height - text_height) // 2)

    draw.text((x, y), title, fill=(255, 255, 255), font=font)
    return out


def make_panel(
    image: Image.Image,
    title: str,
    panel_width: int,
    panel_height: int,
    title_height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    fitted = fit_image_to_panel(image, panel_width, panel_height)
    return draw_panel_title(fitted, title, panel_width, title_height, font)


def compose_grid_panels(
    rows: List[List[Image.Image]],
    background_color: tuple[int, int, int] = (10, 10, 10),
) -> Image.Image:
    if not rows:
        raise ValueError("Cannot compose an empty panel grid")

    row_widths = [sum(panel.width for panel in row) for row in rows]
    row_heights = [max(panel.height for panel in row) for row in rows]

    output_width = max(row_widths)
    output_height = sum(row_heights)

    output_image = Image.new("RGB", (output_width, output_height), background_color)

    y_offset = 0
    for row, row_height in zip(rows, row_heights):
        x_offset = 0
        for panel in row:
            output_image.paste(panel, (x_offset, y_offset))
            x_offset += panel.width

        y_offset += row_height

    return output_image


def parse_frame_index_from_name(path: Path, suffix: str) -> int | None:
    match = re.match(rf"^(\d+)_{re.escape(suffix)}\.png$", path.name)
    if match is None:
        return None

    return int(match.group(1))


def discover_render_frames(camera_render_dir: Path) -> List[Path]:
    if not camera_render_dir.exists():
        raise FileNotFoundError(f"Missing render directory: {camera_render_dir}")

    frame_paths = sorted(
        camera_render_dir.glob("*_render.png"),
        key=lambda path: (
            parse_frame_index_from_name(path, "render") is None,
            parse_frame_index_from_name(path, "render")
            if parse_frame_index_from_name(path, "render") is not None
            else path.name,
        ),
    )

    if not frame_paths:
        raise FileNotFoundError(f"No optimization render frames found in: {camera_render_dir}")

    return frame_paths

def select_render_frames_for_gif(
    render_frame_paths: List[Path],
    frame_stride: int,
    max_gif_frames: int | None,
) -> List[Path]:
    if frame_stride < 1:
        raise ValueError(f"--frame-stride must be >= 1, got {frame_stride}")

    selected_frame_paths = render_frame_paths[::frame_stride]

    if render_frame_paths[-1] not in selected_frame_paths:
        selected_frame_paths.append(render_frame_paths[-1])

    if max_gif_frames is not None:
        if max_gif_frames < 2:
            raise ValueError(f"--max-gif-frames must be >= 2, got {max_gif_frames}")

        if len(selected_frame_paths) > max_gif_frames:
            sampled_indices = np.linspace(
                0,
                len(selected_frame_paths) - 1,
                max_gif_frames,
                dtype=int,
            )

            deduplicated_frame_paths = []
            seen_indices = set()

            for sampled_index in sampled_indices:
                if sampled_index in seen_indices:
                    continue

                seen_indices.add(sampled_index)
                deduplicated_frame_paths.append(selected_frame_paths[sampled_index])

            selected_frame_paths = deduplicated_frame_paths

    return selected_frame_paths

def discover_median_depth_frames(camera_median_depth_dir: Path) -> dict[int, Path]:
    if not camera_median_depth_dir.exists():
        return {}

    median_depth_frame_paths = {}

    for median_depth_path in camera_median_depth_dir.glob("*_median_depth.png"):
        frame_index = parse_frame_index_from_name(median_depth_path, "median_depth")
        if frame_index is None:
            continue

        median_depth_frame_paths[frame_index] = median_depth_path

    return median_depth_frame_paths


def get_matching_median_depth_path(
    render_frame_path: Path,
    median_depth_frame_paths: dict[int, Path],
) -> Path | None:
    render_frame_index = parse_frame_index_from_name(render_frame_path, "render")

    if render_frame_index is None:
        return None

    return median_depth_frame_paths.get(render_frame_index)


def build_gif(
    run_dir: Path,
    camera_name: str,
    fps: float,
    output_name: str,
    loss_column: str | None,
    panel_width: int,
    panel_height: int,
    title_height: int,
    loop: int,
    frame_stride: int,
    max_gif_frames: int | None,
    last_frame_hold_seconds: float,
) -> Path:
    target_path = run_dir / f"render_target_{camera_name}.png"
    initial_path = run_dir / f"render_initial_{camera_name}.png"
    final_path = run_dir / f"render_final_{camera_name}.png"

    render_dir = run_dir / camera_name / "render"
    median_depth_dir = run_dir / camera_name / "median_depth"

    metrics_csv_path = run_dir / "metrics.csv"
    loss_curve_path = run_dir / "loss_curve_for_gif.png"

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    target_img = load_image_rgb(target_path)
    final_img = load_image_rgb(final_path)

    if initial_path.exists():
        initial_img = load_image_rgb(initial_path)
    else:
        initial_img = make_placeholder_image(
            f"Missing initial render\n{initial_path.name}",
            panel_width,
            panel_height,
            font,
        )

    all_render_frame_paths = discover_render_frames(render_dir)
    render_frame_paths = select_render_frames_for_gif(
        render_frame_paths=all_render_frame_paths,
        frame_stride=frame_stride,
        max_gif_frames=max_gif_frames,
    )

    print(
        f"GIF frame sampling: using {len(render_frame_paths)} / "
        f"{len(all_render_frame_paths)} render frames"
    )
    median_depth_frame_paths = discover_median_depth_frames(median_depth_dir)

    loss_curve_img_path, used_loss_column = make_loss_curve_image(
        metrics_csv_path=metrics_csv_path,
        output_png_path=loss_curve_path,
        explicit_loss_column=loss_column,
        width=panel_width,
        height=panel_height,
    )
    loss_curve_img = load_image_rgb(loss_curve_img_path)

    target_panel = make_panel(
        target_img,
        f"Target ({camera_name})",
        panel_width,
        panel_height,
        title_height,
        font,
    )

    initial_panel = make_panel(
        initial_img,
        f"Initial render ({camera_name})",
        panel_width,
        panel_height,
        title_height,
        font,
    )

    final_panel = make_panel(
        final_img,
        f"Final render ({camera_name})",
        panel_width,
        panel_height,
        title_height,
        font,
    )

    loss_panel = make_panel(
        loss_curve_img,
        f"Loss curve ({used_loss_column})",
        panel_width,
        panel_height,
        title_height,
        font,
    )

    no_median_depth_img = make_placeholder_image(
        "Median depth unavailable",
        panel_width,
        panel_height,
        font,
    )

    output_path = run_dir / output_name

    base_duration_ms = max(1, int(round(1000.0 / max(fps, 1.0e-6))))

    if last_frame_hold_seconds > 0.0:
        final_duration_ms = max(
            base_duration_ms,
            int(round(last_frame_hold_seconds * 1000.0)),
        )
    else:
        final_duration_ms = base_duration_ms

    gif_frames: list[Image.Image] = []
    gif_durations_ms: list[int] = []

    for frame_index, render_frame_path in enumerate(render_frame_paths):
        render_img = load_image_rgb(render_frame_path)

        optimization_panel = make_panel(
            render_img,
            f"Optimization ({render_frame_path.stem})",
            panel_width,
            panel_height,
            title_height,
            font,
        )

        median_depth_path = get_matching_median_depth_path(
            render_frame_path=render_frame_path,
            median_depth_frame_paths=median_depth_frame_paths,
        )

        if median_depth_path is not None:
            median_depth_img = load_image_rgb(median_depth_path)
            median_depth_title = f"Median depth ({median_depth_path.stem})"
        else:
            median_depth_img = no_median_depth_img
            median_depth_title = "Median depth"

        median_depth_panel = make_panel(
            median_depth_img,
            median_depth_title,
            panel_width,
            panel_height,
            title_height,
            font,
        )

        grid = compose_grid_panels(
            [
                [target_panel, initial_panel, final_panel],
                [optimization_panel, median_depth_panel, loss_panel],
            ]
        )

        gif_frames.append(grid)

        is_last_frame = frame_index == len(render_frame_paths) - 1
        gif_durations_ms.append(final_duration_ms if is_last_frame else base_duration_ms)

    if not gif_frames:
        raise RuntimeError("No GIF frames were generated")

    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=gif_durations_ms,
        loop=loop,
        optimize=False,
        disposal=2,
    )

    return output_path

def main() -> None:
    args = parse_args()

    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
    else:
        run_dir = find_run_dir_by_index(
            optimization_output_root=args.optimization_output_root.resolve(),
            run_index=args.index,
        )

    camera_names = discover_camera_names(run_dir)

    if args.camera_index < 0 or args.camera_index >= len(camera_names):
        raise IndexError(
            f"camera-index {args.camera_index} is out of range. "
            f"Available cameras ({len(camera_names)}): {camera_names}"
        )

    camera_name = camera_names[args.camera_index]
    print("Using camera:", camera_name)

    output_path = build_gif(
        run_dir=run_dir,
        camera_name=camera_name,
        fps=args.fps,
        output_name=args.output_name,
        loss_column=args.loss_column,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        title_height=args.title_height,
        loop=args.loop,
        frame_stride=args.frame_stride,
        max_gif_frames=args.max_gif_frames,
        last_frame_hold_seconds=args.last_frame_hold_seconds,
    )

    print()
    print("Done.")
    print(f"Run folder       : {run_dir}")
    print(f"Run index        : {args.index if args.run_dir is None else 'explicit --run-dir'}")
    print(f"Available cameras: {camera_names}")
    print(f"Selected camera  : [{args.camera_index}] {camera_name}")
    print(f"FPS              : {args.fps}")
    print(f"Output GIF       : {output_path}")


if __name__ == "__main__":
    main()