from __future__ import annotations

import argparse
import json
import re
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the latest optimization run, plot the loss curve(s), "
            "and optionally re-render points_final.ply using the saved run_config.json."
        )
    )
    parser.add_argument(
        "--optimization-output-root",
        type=Path,
        required=False,
        default=Path("../Assets/OptimizationOutput"),
        help="Path to the OptimizationOutput directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit run directory. If omitted, the latest run is used.",
    )
    parser.add_argument(
        "--pybind-dir",
        type=Path,
        default=None,
        help="Path to cmake-build-pybind so that 'import pale' works.",
    )
    parser.add_argument(
        "--loss-column",
        type=str,
        default=None,
        help=(
            "Optional explicit loss column for the single-loss plot. "
            "If omitted, defaults to loss_rgb_sum, then falls back to other columns."
        ),
    )
    parser.add_argument(
        "--render-output-subdir",
        type=str,
        default="rerender_from_points_final",
        help="Subdirectory inside the run folder where rendered images are written.",
    )
    parser.add_argument(
        "--loss-output-name",
        type=str,
        default="loss_curve.png",
        help="Filename for the saved loss curve image inside the run folder.",
    )
    parser.add_argument(
        "--plot-all-losses",
        action="store_true",
        default=True,
        help=(
            "If set, create a two-panel plot: "
            "top = image/total loss, bottom = regularizer losses."
        ),
    )
    parser.add_argument(
        "--render-final",
        action="store_true",
        help=(
            "If set, also re-render points_final.ply using run_config.json. "
            "Disabled by default."
        ),
    )

    parser.add_argument(
        "--show-plots",
        dest="show_plots",
        action="store_true",
        help="Show the matplotlib plot window. Enabled by default.",
    )
    parser.add_argument(
        "--no-show-plots",
        dest="show_plots",
        action="store_false",
        help="Do not open the matplotlib plot window.",
    )
    parser.set_defaults(show_plots=True)

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
        raise FileNotFoundError(
            f"OptimizationOutput folder does not exist: {optimization_output_root}"
        )

    candidate_run_dirs: list[dict[str, Any]] = []

    for child in optimization_output_root.iterdir():
        if not child.is_dir():
            continue

        metrics_csv_path = child / "metrics.csv"
        if not metrics_csv_path.exists():
            continue

        parsed_timestamp = parse_run_timestamp(child.name)
        modified_time = metrics_csv_path.stat().st_mtime

        candidate_run_dirs.append(
            {
                "run_dir": child,
                "parsed_timestamp": parsed_timestamp,
                "modified_time": modified_time,
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
    """
    Prefer the aggregated ALL_CAMERAS rows from the newer CSV format.
    Fall back to the whole dataframe if camera_name does not exist.
    """
    if "camera_name" not in dataframe.columns:
        return dataframe

    all_cameras_mask = dataframe["camera_name"].astype(str) == "ALL_CAMERAS"
    if all_cameras_mask.any():
        return dataframe.loc[all_cameras_mask].copy()

    return dataframe.copy()


def select_loss_column(dataframe: pd.DataFrame, explicit_loss_column: str | None) -> str:
    if explicit_loss_column is not None:
        if explicit_loss_column not in dataframe.columns:
            raise ValueError(
                f"Requested loss column '{explicit_loss_column}' was not found in metrics.csv. "
                f"Available columns: {list(dataframe.columns)}"
            )
        return explicit_loss_column

    preferred_columns = [
        "loss_rgb_sum",
        "loss_total_sum",
        "loss_normal_consistency_weighted_sum",
        "loss_depth_distortion_weighted_sum",
        "loss_normal_consistency_raw_sum",
        "loss_depth_distortion_raw_sum",
        "loss_l2_window_mean",
        "loss_l2_current_camera",
        "loss_l2_window_sum_scaled",
    ]

    for column_name in preferred_columns:
        if column_name in dataframe.columns:
            return column_name

    raise ValueError(
        "Could not find a supported loss column in metrics.csv. "
        f"Available columns: {list(dataframe.columns)}"
    )


def save_loss_curve(
    metrics_csv_path: Path,
    output_png_path: Path,
    explicit_loss_column: str | None,
    plot_all_losses: bool,
    show_plots: bool,
) -> str:
    dataframe = pd.read_csv(metrics_csv_path)
    dataframe = filter_metrics_rows(dataframe)

    if "iteration" not in dataframe.columns:
        raise ValueError("metrics.csv does not contain an 'iteration' column")

    dataframe = dataframe.sort_values("iteration").reset_index(drop=True)

    if plot_all_losses:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.2, 1.0]},
        )

        ax_top = axes[0]
        ax_bottom = axes[1]

        top_columns = [
            column_name
            for column_name in [
                "loss_rgb_sum",
                "loss_total_sum",
            ]
            if column_name in dataframe.columns
        ]

        bottom_columns_weighted = [
            column_name
            for column_name in [
                "loss_depth_distortion_weighted_sum",
                "loss_normal_consistency_weighted_sum",
            ]
            if column_name in dataframe.columns
        ]

        bottom_columns_raw = [
            column_name
            for column_name in [
                "loss_depth_distortion_raw_sum",
                "loss_normal_consistency_raw_sum",
            ]
            if column_name in dataframe.columns
        ]

        if not top_columns and not bottom_columns_weighted and not bottom_columns_raw:
            fallback_column = select_loss_column(dataframe, explicit_loss_column)
            top_columns = [fallback_column]

        style_map = {
            "loss_rgb_sum": dict(color="tab:blue", linewidth=2.5, alpha=1.0),
            "loss_total_sum": dict(color="tab:orange", linewidth=2.0, alpha=0.95),
            "loss_depth_distortion_weighted_sum": dict(
                color="tab:red", linewidth=1.8, alpha=0.95
            ),
            "loss_normal_consistency_weighted_sum": dict(
                color="tab:green", linewidth=1.8, alpha=0.95
            ),
            "loss_depth_distortion_raw_sum": dict(
                color="tab:red", linewidth=1.2, alpha=0.65, linestyle="--"
            ),
            "loss_normal_consistency_raw_sum": dict(
                color="tab:green", linewidth=1.2, alpha=0.65, linestyle="--"
            ),
        }

        for column_name in top_columns:
            ax_top.plot(
                dataframe["iteration"],
                dataframe[column_name],
                label=column_name,
                **style_map.get(column_name, {}),
            )

        for column_name in bottom_columns_weighted + bottom_columns_raw:
            ax_bottom.plot(
                dataframe["iteration"],
                dataframe[column_name],
                label=column_name,
                **style_map.get(column_name, {}),
            )

        ax_top.set_ylabel("Image / Total loss")
        ax_top.set_title(f"Optimization losses\n{metrics_csv_path.parent.name}")
        ax_top.grid(True)
        if top_columns:
            ax_top.legend()

        ax_bottom.set_xlabel("Iteration")
        ax_bottom.set_ylabel("Regularizer loss")
        ax_bottom.grid(True)
        if bottom_columns_weighted or bottom_columns_raw:
            ax_bottom.legend()

        plt.tight_layout()
        plt.savefig(output_png_path, dpi=200)

        if show_plots:
            plt.show()

        plt.close(fig)

        plotted_columns = top_columns + bottom_columns_weighted + bottom_columns_raw
        return ", ".join(plotted_columns)

    loss_column_name = select_loss_column(dataframe, explicit_loss_column)

    fig = plt.figure(figsize=(12, 5))
    plt.plot(
        dataframe["iteration"],
        dataframe[loss_column_name],
        linewidth=2.2,
        color="tab:blue",
    )
    plt.xlabel("Iteration")
    plt.ylabel(loss_column_name)
    plt.title(f"{loss_column_name} over iterations\n{metrics_csv_path.parent.name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=200)

    if show_plots:
        plt.show()

    plt.close(fig)

    return loss_column_name


def resolve_pybind_dir(pybind_dir_argument: Path | None) -> Path:
    if pybind_dir_argument is not None:
        if not pybind_dir_argument.exists():
            raise FileNotFoundError(f"Provided --pybind-dir does not exist: {pybind_dir_argument}")
        return pybind_dir_argument.resolve()

    if "PALE_PYBIND_DIR" in os.environ:
        environment_candidate = Path(os.environ["PALE_PYBIND_DIR"])
        if environment_candidate.exists():
            return environment_candidate.resolve()

    script_dir = Path(__file__).resolve().parent
    relative_candidates = [
        script_dir / "cmake-build-pybind",
        script_dir.parent / "cmake-build-pybind",
        script_dir.parent.parent / "cmake-build-pybind",
    ]

    for candidate_path in relative_candidates:
        if candidate_path.exists():
            return candidate_path.resolve()

    raise FileNotFoundError(
        "Could not resolve the pybind directory for 'pale'. "
        "Pass --pybind-dir explicitly."
    )


def import_pale(pybind_dir: Path):
    if str(pybind_dir) not in sys.path:
        sys.path.insert(0, str(pybind_dir))

    try:
        import pale
    except ModuleNotFoundError as exception:
        raise ModuleNotFoundError(
            f"Failed to import 'pale' even after adding pybind dir to sys.path: {pybind_dir}"
        ) from exception

    return pale


def load_run_config(run_config_path: Path) -> dict:
    if not run_config_path.exists():
        raise FileNotFoundError(
            f"Missing run_config.json: {run_config_path}\n"
            "You need to save the optimization/run config into each run folder."
        )

    with open(run_config_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def get_forward_rgb(rendered_images: dict, camera_name: str) -> np.ndarray:
    camera_output = rendered_images[camera_name]

    if isinstance(camera_output, dict):
        if "image" not in camera_output:
            raise KeyError(
                f"Rendered output for camera '{camera_name}' is a dict but does not contain 'image'. "
                f"Available keys: {list(camera_output.keys())}"
            )
        image_numpy = np.asarray(camera_output["image"], dtype=np.float32)
    else:
        image_numpy = np.asarray(camera_output, dtype=np.float32)

    if image_numpy.ndim != 3:
        raise RuntimeError(
            f"Unexpected image shape for camera '{camera_name}': {image_numpy.shape}"
        )

    return np.clip(image_numpy[..., :3], 0.0, 1.0)


def render_points_final(
    pale_module,
    run_dir: Path,
    run_config: dict,
    render_output_subdir: str,
) -> list[Path]:
    assets_root = Path(run_config["assets_root"])
    scene_xml = run_config["scene_xml"]
    renderer_settings = dict(run_config["renderer_settings"])

    points_final_ply_path = run_dir / "points_final.ply"
    if not points_final_ply_path.exists():
        raise FileNotFoundError(f"Missing points_final.ply: {points_final_ply_path}")

    render_output_dir = run_dir / render_output_subdir
    render_output_dir.mkdir(parents=True, exist_ok=True)

    renderer_settings["primal_shadow_rays"] = 64
    renderer_settings["forward_passes"] = 10

    renderer = pale_module.Renderer(
        str(assets_root),
        str(scene_xml),
        str(points_final_ply_path),
        renderer_settings,
    )

    rendered_images = renderer.render_forward()
    camera_names = renderer.get_camera_names()

    saved_image_paths: list[Path] = []

    for camera_name in camera_names:
        if camera_name not in rendered_images:
            print(f"Warning: rendered output for camera '{camera_name}' was not found. Skipping.")
            continue

        rgb_image = get_forward_rgb(rendered_images, camera_name)
        output_png_path = render_output_dir / f"{camera_name}.png"

        plt.imsave(output_png_path, rgb_image)
        saved_image_paths.append(output_png_path)

    return saved_image_paths


def main() -> None:
    args = parse_args()

    optimization_output_root = args.optimization_output_root.resolve()

    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
    else:
        run_dir = find_latest_run_dir(optimization_output_root)

    metrics_csv_path = run_dir / "metrics.csv"
    loss_curve_output_path = run_dir / args.loss_output_name

    if not metrics_csv_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv: {metrics_csv_path}")

    loss_column_name = save_loss_curve(
        metrics_csv_path=metrics_csv_path,
        output_png_path=loss_curve_output_path,
        explicit_loss_column=args.loss_column,
        plot_all_losses=args.plot_all_losses,
        show_plots=args.show_plots,
    )

    saved_render_paths: list[Path] = []
    run_config_path: Path | None = None
    pybind_dir: Path | None = None

    if args.render_final:
        run_config_path = run_dir / "run_config.json"
        pybind_dir = resolve_pybind_dir(args.pybind_dir)
        pale_module = import_pale(pybind_dir)
        run_config = load_run_config(run_config_path)

        saved_render_paths = render_points_final(
            pale_module=pale_module,
            run_dir=run_dir,
            run_config=run_config,
            render_output_subdir=args.render_output_subdir,
        )

    print()
    print("Done.")
    print(f"Run folder          : {run_dir}")
    print(f"Metrics file        : {metrics_csv_path}")
    print(f"Loss column(s) used : {loss_column_name}")
    print(f"Loss curve written  : {loss_curve_output_path}")
    print(f"Show plots          : {args.show_plots}")
    print(f"Rendering enabled   : {args.render_final}")

    if args.render_final:
        print(f"Run config          : {run_config_path}")
        print(f"Pybind dir          : {pybind_dir}")
        print("Rendered images:")
        for saved_render_path in saved_render_paths:
            print(f"  {saved_render_path}")
    else:
        print("Rendered images     : skipped")


if __name__ == "__main__":
    main()