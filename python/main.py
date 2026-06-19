from __future__ import annotations

import time
import sys
import subprocess
from pathlib import Path
import os
import shutil

import pale
from config import RendererSettingsConfig, parse_args
from io_utils import *
from training import run_optimization
from render_hooks import get_training_camera_names


def recreate_output_dir(output_dir: Path) -> Path:
    resolved_output_dir = output_dir.expanduser().resolve()

    unsafe_paths = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
    }

    if resolved_output_dir in unsafe_paths:
        raise ValueError(f"Refusing to delete unsafe output directory: {resolved_output_dir}")

    if resolved_output_dir.exists():
        if not resolved_output_dir.is_dir():
            raise NotADirectoryError(f"Output path exists but is not a directory: {resolved_output_dir}")

        print(f"Clearing existing output directory: {resolved_output_dir}")
        shutil.rmtree(resolved_output_dir)

    resolved_output_dir.mkdir(parents=True, exist_ok=False)
    return resolved_output_dir

def copy_scene_xml_to_run_folder(scene_xml: str | Path, output_dir: Path) -> Path:
    scene_xml_path = Path(scene_xml).expanduser().resolve()

    if not scene_xml_path.is_file():
        raise FileNotFoundError(f"Scene XML file does not exist: {scene_xml_path}")

    output_scene_xml_path = output_dir / scene_xml_path.name
    shutil.copy2(scene_xml_path, output_scene_xml_path)

    print(f"Copied scene XML       : {output_scene_xml_path}")
    return output_scene_xml_path

def main() -> None:
    config = parse_args()
    configure_paths_from_dataset_folder(config)

    config.assets_root = Path(config.assets_root).expanduser().resolve()

    renderer_settings = RendererSettingsConfig()

    base_output_dir = resolve_output_dir(
        config.output_dir,
        output_dir_is_explicit=config.output_dir_is_explicit,
    )
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    scene_short = Path(config.scene_xml).stem

    run_folder_name = (
        f"{timestamp}_"
        f"lr{config.learning_rate_position:.3g}_"
        f"it{config.iterations}_"
        f"{scene_short}"
    )

    if config.output_dir_is_explicit:
        config.output_dir = base_output_dir
        run_folder_name = config.output_dir.name
    else:
        config.output_dir = base_output_dir / run_folder_name

    config.output_dir = recreate_output_dir(config.output_dir)

    saved_scene_xml_path = copy_scene_xml_to_run_folder(
        scene_xml=config.scene_xml,
        output_dir=config.output_dir,
    )

    save_run_config(
        output_dir=config.output_dir,
        config=config,
        renderer_settings=renderer_settings,
        run_folder_name=run_folder_name,
    )

    # ------------------------------------------------------------------
    # 1. Initialize renderer once
    # ------------------------------------------------------------------
    renderer = pale.Renderer(
        str(config.assets_root),
        config.scene_xml,
        config.pointcloud_ply,
        renderer_settings.as_dict(config),
    )

    camera_ids = get_training_camera_names(renderer)
    if len(camera_ids) == 0:
        raise RuntimeError("No cameras found in scene.")

    main_camera = camera_ids[0]

    print("Starting optimization with configuration:")
    print(f"  assets_root          : {config.assets_root}")
    print(f"  scene_xml            : {config.scene_xml}")
    print(f"  pointcloud           : {config.pointcloud_ply}")
    print(f"  dataset_path         : {config.dataset_path}")
    print(f"  iterations           : {config.iterations}")
    print(f"  lr_base              : {config.learning_rate}")
    print(f"  lr_position          : {config.learning_rate_position}")
    print(f"  lr_rotation          : {config.learning_rate_rotation}")
    print(f"  lr_scale             : {config.learning_rate_scale}")
    print(f"  lr_color             : {config.learning_rate_albedo}")
    print(f"  lr_opacity           : {config.learning_rate_opacity}")
    print(f"  lr_beta              : {config.learning_rate_beta}")
    print(f"  depth_distort_weight : {config.depth_distort_weight}")
    print(f"  optimizer            : {config.optimizer_type}")
    print(f"  run_output_dir       : {config.output_dir}")
    print(f"  cameras              : {camera_ids}")
    print(f"  main camera          : {main_camera}")

    # ------------------------------------------------------------------
    # 2. Launch external image preview script (non-blocking)
    # ------------------------------------------------------------------
    image_preview_script = Path(__file__).parent / "image_preview.py"
    image_preview_process = None

    dataset_path_full = Path(config.dataset_path).expanduser().resolve()

    if image_preview_script.exists():
        preview_args = [
            sys.executable,
            str(image_preview_script),
            "--output-path",
            str(config.output_dir.resolve()),
            "--refresh-ms",
            "200",
            "--parent-pid",
            str(os.getpid()),
        ]

        try:
            image_preview_process = subprocess.Popen(preview_args)
            print(f"Started image preview : {image_preview_script}")
            print(f"  dataset-path        : {dataset_path_full}")
            print(f"  output-path         : {config.output_dir}")
            print(f"  camera-name         : {main_camera}")
        except Exception as exception:
            print(f"Warning: could not start image preview ({exception}). Continuing without preview.")
    else:
        print(f"Warning: image_preview.py not found at {image_preview_script}. No live preview will be shown.")

    # ------------------------------------------------------------------
    # 3. Run optimization (reusing the same renderer)
    # ------------------------------------------------------------------
    try:
        run_optimization(renderer, config, renderer_settings)
    finally:
        if image_preview_process is not None:
            if image_preview_process.poll() is None:
                image_preview_process.terminate()
                try:
                    image_preview_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    image_preview_process.kill()
                    image_preview_process.wait()


if __name__ == "__main__":
    main()