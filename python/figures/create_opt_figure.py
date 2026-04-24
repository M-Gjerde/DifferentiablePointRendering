from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 4-panel optimization GIF from the latest optimization run.\n"
            "Panels: target | final render | optimization sequence | loss curve"
        )
    )
    parser.add_argument(
        "--optimization-output-root",
        type=Path,
        default=Path("../Assets/OptimizationOutput"),
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
        default=8.0,
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
            "Defaults to loss_total_sum, then loss_rgb_sum, then loss_depth_distortion_sum."
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
    return parser.parse_args()


def parse_run_timestamp(run_dir_name: str) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", run_dir_name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def find_latest_run_dir(optimization_output_root: Path) -> Path:
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
    return candidate_run_dirs[0]["run_dir"]


def filter_metrics_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "camera_name" not in dataframe.columns:
        return dataframe
    mask = dataframe["camera_name"].astype(str) == "ALL_CAMERAS"
    if mask.any():
        return dataframe.loc[mask].copy()
    return dataframe.copy()


def select_loss_column(dataframe: pd.DataFrame, explicit_loss_column: str | None) -> str:
    if explicit_loss_column is not None:
        if explicit_loss_column not in dataframe.columns:
            raise ValueError(
                f"Requested loss column '{explicit_loss_column}' not found. "
                f"Available columns: {list(dataframe.columns)}"
            )
        return explicit_loss_column

    preferred = [
        "loss_total_sum",
        "loss_rgb_sum",
        "loss_depth_distortion_sum",
        "loss_l2_window_mean",
        "loss_l2_current_camera",
        "loss_l2_window_sum_scaled",
    ]
    for col in preferred:
        if col in dataframe.columns:
            return col

    raise ValueError(f"No supported loss column found. Available columns: {list(dataframe.columns)}")


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

    dataframe = dataframe.sort_values("iteration").reset_index(drop=True)
    loss_column = select_loss_column(dataframe, explicit_loss_column)

    plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    plt.plot(dataframe["iteration"], dataframe[loss_column], linewidth=2.0)
    plt.xlabel("Iteration")
    plt.ylabel(loss_column)
    plt.title("Loss curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=100)
    plt.close()

    return output_png_path, loss_column


def load_image_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    image = Image.open(path).convert("RGB")
    return image


def fit_image_to_panel(image: Image.Image, panel_width: int, panel_height: int) -> Image.Image:
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        return Image.new("RGB", (panel_width, panel_height), (0, 0, 0))

    scale = min(panel_width / src_w, panel_height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (panel_width, panel_height), (20, 20, 20))
    offset_x = (panel_width - new_w) // 2
    offset_y = (panel_height - new_h) // 2
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
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = draw.textsize(title, font=font)

    x = (panel_width - text_w) // 2
    y = max(0, (title_height - text_h) // 2)
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


def compose_row_panels(panels: List[Image.Image]) -> Image.Image:
    widths = [p.width for p in panels]
    heights = [p.height for p in panels]
    out = Image.new("RGB", (sum(widths), max(heights)), (10, 10, 10))

    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width
    return out


def discover_render_frames(camera_render_dir: Path) -> List[Path]:
    if not camera_render_dir.exists():
        raise FileNotFoundError(f"Missing render directory: {camera_render_dir}")

    frame_paths = sorted(
        camera_render_dir.glob("*_render.png"),
        key=lambda p: int(re.match(r"^(\d+)_render\.png$", p.name).group(1))
        if re.match(r"^(\d+)_render\.png$", p.name)
        else p.name,
    )

    if not frame_paths:
        raise FileNotFoundError(f"No optimization render frames found in: {camera_render_dir}")

    return frame_paths


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
) -> Path:
    target_path = run_dir / f"render_target_{camera_name}.png"
    final_path = run_dir / f"render_final_{camera_name}.png"
    render_dir = run_dir / camera_name / "render"
    metrics_csv_path = run_dir / "metrics.csv"
    loss_curve_path = run_dir / "loss_curve_for_gif.png"

    target_img = load_image_rgb(target_path)
    final_img = load_image_rgb(final_path)
    render_frame_paths = discover_render_frames(render_dir)

    loss_curve_img_path, used_loss_column = make_loss_curve_image(
        metrics_csv_path=metrics_csv_path,
        output_png_path=loss_curve_path,
        explicit_loss_column=loss_column,
        width=panel_width,
        height=panel_height,
    )
    loss_curve_img = load_image_rgb(loss_curve_img_path)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    target_panel = make_panel(
        target_img, f"Target ({camera_name})",
        panel_width, panel_height, title_height, font
    )
    final_panel = make_panel(
        final_img, f"Final render ({camera_name})",
        panel_width, panel_height, title_height, font
    )
    loss_panel = make_panel(
        loss_curve_img, f"Loss curve ({used_loss_column})",
        panel_width, panel_height, title_height, font
    )

    output_path = run_dir / output_name
    duration_sec = 1.0 / max(fps, 1e-6)

    with imageio.get_writer(output_path, mode="I", duration=duration_sec, loop=loop) as writer:
        for frame_index, render_frame_path in enumerate(render_frame_paths):
            render_img = load_image_rgb(render_frame_path)
            optimization_title = f"Optimization ({render_frame_path.stem})"
            optimization_panel = make_panel(
                render_img,
                optimization_title,
                panel_width,
                panel_height,
                title_height,
                font,
            )

            row = compose_row_panels(
                [
                    target_panel,
                    final_panel,
                    optimization_panel,
                    loss_panel,
                ]
            )
            writer.append_data(np.asarray(row, dtype=np.uint8))

    return output_path


def main() -> None:
    args = parse_args()

    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
    else:
        run_dir = find_latest_run_dir(args.optimization_output_root.resolve())

    camera_names = discover_camera_names(run_dir)

    if args.camera_index < 0 or args.camera_index >= len(camera_names):
        raise IndexError(
            f"camera-index {args.camera_index} is out of range. "
            f"Available cameras ({len(camera_names)}): {camera_names}"
        )

    camera_name = camera_names[args.camera_index]

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
    )

    print()
    print("Done.")
    print(f"Run folder       : {run_dir}")
    print(f"Available cameras: {camera_names}")
    print(f"Selected camera  : [{args.camera_index}] {camera_name}")
    print(f"FPS              : {args.fps}")
    print(f"Output GIF       : {output_path}")


if __name__ == "__main__":
    main()