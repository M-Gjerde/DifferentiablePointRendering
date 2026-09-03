from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from config import OptimizationConfig, RendererSettingsConfig
from io_utils import configure_paths_from_dataset_folder, resolve_output_dir, save_run_config


def _recreate_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir in {Path("/"), Path.home().resolve(), Path.cwd().resolve()}:
        raise ValueError(f"Refusing to delete unsafe output directory: {output_dir}")

    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path exists but is not a directory: {output_dir}")
        print(f"Clearing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    return output_dir


def _copy_scene_xml(scene_xml: str | Path, output_dir: Path) -> None:
    source = Path(scene_xml).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Scene XML file does not exist: {source}")

    destination = output_dir / source.name
    shutil.copy2(source, destination)
    print(f"Copied scene XML       : {destination}")


def prepare_run(config: OptimizationConfig, renderer_settings: RendererSettingsConfig) -> None:
    configure_paths_from_dataset_folder(config)
    config.assets_root = Path(config.assets_root).expanduser().resolve()

    if config.ground_truth is not None:
        config.ground_truth = config.ground_truth.expanduser().resolve()
        if not config.ground_truth.is_file():
            raise FileNotFoundError(f"--gt does not exist: {config.ground_truth}")

    output_root = resolve_output_dir(
        config.output_dir,
        output_dir_is_explicit=config.output_dir_is_explicit,
    )
    run_name = (
        f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_"
        f"lr{config.learning_rate_position:.3g}_"
        f"it{config.iterations}_{Path(config.scene_xml).stem}"
    )
    config.output_dir = output_root if config.output_dir_is_explicit else output_root / run_name
    if config.output_dir_is_explicit:
        run_name = config.output_dir.name

    config.output_dir = _recreate_output_dir(config.output_dir)
    _copy_scene_xml(config.scene_xml, config.output_dir)
    save_run_config(config.output_dir, config, renderer_settings, run_name)


def print_run_configuration(
    config: OptimizationConfig,
    camera_ids: list[str],
    main_camera: str,
) -> None:
    rows = (
        ("assets_root", config.assets_root),
        ("scene_xml", config.scene_xml),
        ("pointcloud", config.pointcloud_ply),
        ("dataset_path", config.dataset_path),
        ("target_color_space", f"{config.target_color_space} -> linear sRGB training"),
        ("shared_slab_lighting", config.share_local_layer_direct_lighting),
        ("iterations", config.iterations),
        ("resume_iteration_offset", config.resume_iteration_offset),
        ("final_global_iteration", config.resume_iteration_offset + config.iterations),
        ("lr_base", config.learning_rate),
        ("lr_position", config.learning_rate_position),
        ("lr_rotation", config.learning_rate_rotation),
        ("lr_scale", config.learning_rate_scale),
        ("lr_color", config.learning_rate_albedo),
        ("lr_opacity", config.learning_rate_opacity),
        ("lr_beta", config.learning_rate_beta),
        (
            "global_lr_decay",
            f"{config.use_global_lr_decay} "
            f"({config.global_lr_scale_init} -> {config.global_lr_scale_final})",
        ),
        (
            "position_lr_decay",
            f"{config.use_position_lr_decay} "
            f"({config.position_lr_scale_init} -> {config.position_lr_scale_final})",
        ),
        ("lr_decay_timeline", f"start={config.lr_decay_start_iteration}, steps={config.lr_decay_max_steps}"),
        ("ssim_weight", config.ssim_weight),
        ("ssim_window/sigma", f"{config.ssim_window_size} / {config.ssim_sigma}"),
        ("depth_distort_weight", config.depth_distort_weight),
        ("opacity_prior_weight", config.opacity_prior_weight),
        ("densification_interval", config.densification_interval),
        ("prune_interval", config.prune_interval),
        ("densify_grad_abs_min", config.densification_grad_abs_min),
        ("densify_grad_abs_min_final", config.densification_grad_abs_min_final),
        ("curvature_violation_threshold", config.curvature_violation_threshold),
        (
            "densify_grad_abs_min_decay",
            f"{config.densification_grad_abs_min_decay_start_iteration} -> "
            f"{config.densification_grad_abs_min_decay_end_iteration}",
        ),
        ("mesh_extraction_interval", config.mesh_extraction_interval),
        ("save_final_mesh", config.save_final_mesh),
        ("geometry_gt", config.ground_truth),
        ("metrics_viewer", config.enable_metrics),
        ("image_preview", config.enable_image_preview),
        ("optimizer", config.optimizer_type),
        ("run_output_dir", config.output_dir),
        ("cameras", camera_ids),
        ("main camera", main_camera),
    )
    print("Starting optimization with configuration:")
    for label, value in rows:
        print(f"  {label:<27}: {value}")


def _start_companion(name: str, script: Path, arguments: list[str]) -> subprocess.Popen | None:
    if not script.is_file():
        print(f"Warning: {script.name} not found at {script}.")
        return None
    try:
        process = subprocess.Popen([sys.executable, str(script), *arguments])
    except Exception as exception:
        print(f"Warning: could not start {name} ({exception}). Continuing without it.")
        return None
    print(f"Started {name}: {script}")
    return process


def start_companions(config: OptimizationConfig) -> list[subprocess.Popen]:
    python_dir = Path(__file__).parent
    specifications = []
    if config.enable_image_preview:
        specifications.append((
            "image preview",
            python_dir / "image_preview.py",
            [
                "--output-path", str(config.output_dir.resolve()),
                "--refresh-ms", "200",
                "--parent-pid", str(os.getpid()),
            ],
        ))
    if config.enable_metrics:
        specifications.append((
            "metrics viewer",
            python_dir / "analyze" / "view_metrics_live.py",
            ["--run-dir", str(config.output_dir.resolve())],
        ))
    return [
        process
        for name, script, arguments in specifications
        if (process := _start_companion(name, script, arguments)) is not None
    ]


def stop_companions(processes: Iterable[subprocess.Popen]) -> None:
    for process in reversed(list(processes)):
        if process.poll() is not None:
            continue
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
