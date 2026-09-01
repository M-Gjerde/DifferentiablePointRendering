from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = REPO_ROOT / "Assets"

PARAMETERS = {
    "translation_x",
    "translation_y",
    "translation_z",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "scale_u",
    "scale_v",
    "opacity",
    "beta",
    "albedo_r",
    "albedo_g",
    "albedo_b",
}

BOUNDED_PARAMETERS: dict[str, tuple[float, float]] = {
    "scale_u": (1.0e-8, math.inf),
    "scale_v": (1.0e-8, math.inf),
    "opacity": (0.0, 1.0),
    "albedo_r": (0.0, 1.0),
    "albedo_g": (0.0, 1.0),
    "albedo_b": (0.0, 1.0),
}

OBJECTIVE_TYPES = {"linear", "l2", "l2_ssim"}

REQUIRED_SETTINGS = {
    "bounces",
    "forward_passes",
    "adjoint_bounces",
    "adjoint_passes",
    "enable_adjoint_shadow_rays",
    "adjoint_shadow_path_rays",
    "share_local_layer_direct_lighting",
    "minimum_projected_footprint",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_suite(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite_path = path.resolve()
    suite = json.loads(suite_path.read_text())
    if int(suite.get("schema_version", 0)) != 2:
        raise ValueError(f"{suite_path}: expected schema_version=2")

    defaults = suite.get("defaults", {})
    raw_cases = suite.get("cases", [])
    if not isinstance(defaults, dict) or not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"{suite_path}: defaults must be an object and cases must be a non-empty list")

    resolved_cases: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError(f"{suite_path}: every case must be an object")
        case = _deep_merge(defaults, raw_case)
        validate_case(case, suite_path)
        name = str(case["name"])
        if name in seen_names:
            raise ValueError(f"{suite_path}: duplicate case name '{name}'")
        seen_names.add(name)
        resolved_cases.append(case)
    return suite, resolved_cases


def validate_case(case: dict[str, Any], source: Path | None = None) -> None:
    origin = f"{source}: " if source is not None else ""
    for key in ("name", "stage", "scene", "ply", "camera", "parameter", "index", "values", "epsilons"):
        if key not in case:
            raise ValueError(f"{origin}case is missing '{key}': {case}")

    name = str(case["name"])
    if not name or any(character.isspace() for character in name):
        raise ValueError(f"{origin}case name must be non-empty and contain no whitespace: '{name}'")
    if str(case["parameter"]) not in PARAMETERS:
        raise ValueError(f"{origin}{name}: unsupported parameter '{case['parameter']}'")
    if int(case["index"]) < 0:
        raise ValueError(f"{origin}{name}: index must be non-negative")

    values = [float(value) for value in case["values"]]
    epsilons = [float(epsilon) for epsilon in case["epsilons"]]
    if not values or not epsilons:
        raise ValueError(f"{origin}{name}: values and epsilons must be non-empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{origin}{name}: values must be finite")
    if not all(math.isfinite(epsilon) and epsilon > 0.0 for epsilon in epsilons):
        raise ValueError(f"{origin}{name}: epsilons must be finite and positive")
    if epsilons != sorted(epsilons, reverse=True):
        raise ValueError(f"{origin}{name}: list epsilons from largest to smallest")

    bounds = BOUNDED_PARAMETERS.get(str(case["parameter"]))
    if bounds is not None:
        lower, upper = bounds
        largest_epsilon = max(epsilons)
        for value in values:
            if value - largest_epsilon < lower or value + largest_epsilon > upper:
                raise ValueError(
                    f"{origin}{name}: central stencil around {value} +/- {largest_epsilon} "
                    f"exceeds bounds [{lower}, {upper}]"
                )

    settings = case.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(f"{origin}{name}: settings must be an object")
    missing_settings = REQUIRED_SETTINGS - settings.keys()
    if missing_settings:
        raise ValueError(f"{origin}{name}: settings missing {sorted(missing_settings)}")
    if int(settings["forward_passes"]) <= 0 or int(settings["adjoint_passes"]) <= 0:
        raise ValueError(f"{origin}{name}: forward_passes and adjoint_passes must be positive")
    if int(settings["adjoint_shadow_path_rays"]) <= 0:
        raise ValueError(f"{origin}{name}: adjoint_shadow_path_rays must be positive")
    q_null = float(settings.get("adjoint_q_null", 0.5))
    q_reflect = float(settings.get("adjoint_q_reflect", 0.5))
    if not (0.0 <= q_null < 1.0 and 0.0 < q_reflect <= 1.0):
        raise ValueError(f"{origin}{name}: adjoint q probabilities must be in their open sampling ranges")
    if not math.isclose(q_null + q_reflect, 1.0, rel_tol=0.0, abs_tol=1.0e-7):
        raise ValueError(
            f"{origin}{name}: current null/reflect sampler requires q_null + q_reflect = 1"
        )

    objective = case.get("objective", {})
    if str(objective.get("type", "linear")) not in OBJECTIVE_TYPES:
        raise ValueError(f"{origin}{name}: objective.type must be one of {sorted(OBJECTIVE_TYPES)}")

    check = case.get("check")
    if not isinstance(check, dict):
        raise ValueError(f"{origin}{name}: check must be an object")
    for key in (
        "relative_tolerance",
        "absolute_tolerance",
        "minimum_signal",
        "fd_consistency_relative_tolerance",
        "repeatability_tolerance",
    ):
        value = float(check.get(key, -1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{origin}{name}: check.{key} must be finite and non-negative")


def scene_paths(case: dict[str, Any]) -> tuple[Path, Path]:
    scene_name = str(case["scene"])
    scene_directory = ASSETS_ROOT / "GradientTests" / scene_name
    return scene_directory / f"{scene_name}.xml", scene_directory / f"{case['ply']}.ply"


def _set_parameter(renderer: Any, parameter: str, value: float, index: int) -> None:
    if parameter == "opacity":
        renderer.set_point_opacity(opacity=value, index=index)
    elif parameter == "beta":
        renderer.set_point_beta(beta=value, index=index)
    elif parameter.startswith("translation_"):
        renderer.set_point_translation(translation=value, axis="xyz".index(parameter[-1]), index=index)
    elif parameter.startswith("rotation_"):
        renderer.set_point_rotation_degrees(rotation_deg=value, axis="xyz".index(parameter[-1]), index=index)
    elif parameter == "scale_u":
        renderer.set_point_scale(scale=value, axis=0, index=index)
    elif parameter == "scale_v":
        renderer.set_point_scale(scale=value, axis=1, index=index)
    elif parameter.startswith("albedo_"):
        renderer.set_point_albedo(intensity=value, axis="rgb".index(parameter[-1]), index=index)
    else:
        raise ValueError(f"Unsupported parameter '{parameter}'")


def _extract_gradient(gradients: dict[str, Any], parameter: str, index: int) -> float:
    if parameter.startswith("translation_"):
        return float(np.asarray(gradients["position"])[index, "xyz".index(parameter[-1])])
    if parameter.startswith("rotation_"):
        radians_gradient = float(np.asarray(gradients["rotation"])[index, "xyz".index(parameter[-1])])
        return radians_gradient * (math.pi / 180.0)
    if parameter == "scale_u":
        return float(np.asarray(gradients["scale"])[index, 0])
    if parameter == "scale_v":
        return float(np.asarray(gradients["scale"])[index, 1])
    if parameter.startswith("albedo_"):
        return float(np.asarray(gradients["albedo"])[index, "rgb".index(parameter[-1])])
    if parameter == "opacity":
        return float(np.asarray(gradients["opacity"])[index])
    if parameter == "beta":
        return float(np.asarray(gradients["beta"])[index])
    raise ValueError(f"Unsupported parameter '{parameter}'")


def _render_rgb(renderer: Any, camera: str) -> np.ndarray:
    images = renderer.render_forward()
    if camera not in images:
        raise KeyError(f"Camera '{camera}' not found; renderer returned {sorted(images.keys())}")
    rgb = np.asarray(images[camera]["raw"], dtype=np.float32)[..., :3]
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.all(np.isfinite(rgb)):
        raise RuntimeError(f"Invalid rendered RGB for camera '{camera}': shape={rgb.shape}")
    return np.ascontiguousarray(rgb)


@dataclass(frozen=True)
class Objective:
    kind: str
    payload: np.ndarray
    ssim_weight: float = 0.0
    ssim_window_size: int = 5
    ssim_sigma: float = 0.75

    def evaluate(self, image: np.ndarray, gradient: bool) -> tuple[float, np.ndarray | None]:
        if self.kind == "linear":
            value = float(np.sum(image.astype(np.float64) * self.payload.astype(np.float64)))
            return value, self.payload if gradient else None
        if self.kind == "l2":
            residual = image - self.payload
            value = float(0.5 * np.mean(residual.astype(np.float64) ** 2))
            image_gradient = np.ascontiguousarray(residual / float(residual.size), dtype=np.float32)
            return value, image_gradient if gradient else None
        if self.kind == "l2_ssim":
            from losses import compute_l2_ssim_loss_and_grad

            value, image_gradient, _ = compute_l2_ssim_loss_and_grad(
                image,
                self.payload,
                ssim_weight=self.ssim_weight,
                window_size=self.ssim_window_size,
                sigma=self.ssim_sigma,
            )
            return float(value), np.ascontiguousarray(image_gradient, dtype=np.float32) if gradient else None
        raise AssertionError(self.kind)


def _spatial_pattern(shape: tuple[int, int, int]) -> np.ndarray:
    height, width, channels = shape
    if channels != 3:
        raise ValueError(f"Expected RGB shape, got {shape}")
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    return 1.0 + 0.31 * x - 0.23 * y + 0.13 * x * y + 0.07 * np.sin(3.0 * x + 2.0 * y)


def make_objective(shape: tuple[int, int, int], specification: dict[str, Any]) -> Objective:
    kind = str(specification.get("type", "linear"))
    spatial = _spatial_pattern(shape)
    if kind == "linear":
        channel_weights = np.asarray(specification.get("channel_weights", [1.0, -0.65, 0.35]), dtype=np.float32)
        if channel_weights.shape != (3,):
            raise ValueError("linear objective channel_weights must contain three values")
        cotangent = spatial[..., None] * channel_weights[None, None, :]
        cotangent /= float(np.prod(shape))
        return Objective(kind="linear", payload=np.ascontiguousarray(cotangent, dtype=np.float32))

    target_channels = np.asarray(specification.get("target_channels", [0.12, 0.18, 0.08]), dtype=np.float32)
    if target_channels.shape != (3,):
        raise ValueError("target_channels must contain three values")
    target = target_channels[None, None, :] + 0.035 * spatial[..., None]
    target = np.ascontiguousarray(target, dtype=np.float32)
    if kind == "l2":
        return Objective(kind="l2", payload=target)
    if kind == "l2_ssim":
        return Objective(
            kind="l2_ssim",
            payload=target,
            ssim_weight=float(specification.get("ssim_weight", 0.2)),
            ssim_window_size=int(specification.get("window_size", 5)),
            ssim_sigma=float(specification.get("sigma", 0.75)),
        )
    raise ValueError(f"Unsupported objective '{kind}'")


def _combined_tolerance(analytic: float, finite_difference: float, check: dict[str, Any]) -> float:
    scale = max(abs(analytic), abs(finite_difference))
    return float(check["absolute_tolerance"]) + float(check["relative_tolerance"]) * scale


def _relative_error(analytic: float, finite_difference: float, floor: float) -> float:
    return abs(analytic - finite_difference) / max(abs(analytic), abs(finite_difference), floor)


def compare_gradients(
    analytic: float,
    finite_difference: float,
    check: dict[str, Any],
    allow_zero_signal: bool = False,
) -> dict[str, Any]:
    finite = math.isfinite(analytic) and math.isfinite(finite_difference)
    signal = max(abs(analytic), abs(finite_difference)) if finite else math.inf
    signal_pass = allow_zero_signal or (finite and signal >= float(check["minimum_signal"]))
    absolute_error = abs(analytic - finite_difference) if finite else math.inf
    tolerance = _combined_tolerance(analytic, finite_difference, check) if finite else 0.0
    return {
        "finite": finite,
        "signal": signal,
        "signal_pass": signal_pass,
        "absolute_error": absolute_error,
        "relative_error": _relative_error(
            analytic,
            finite_difference,
            max(float(check["minimum_signal"]), 1.0e-30),
        ) if finite else math.inf,
        "tolerance": tolerance,
        "pass": finite and signal_pass and absolute_error <= tolerance,
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_rgb_artifact(
    image: np.ndarray,
    base_path: Path,
    *,
    display_scale: float = 1.0,
) -> tuple[Path, Path]:
    """Save exact linear RGB and a viewable, sRGB-encoded preview."""
    from PIL import Image

    rgb = np.ascontiguousarray(image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.all(np.isfinite(rgb)):
        raise ValueError(f"Cannot save invalid RGB artifact: shape={rgb.shape}")
    if not math.isfinite(display_scale) or display_scale <= 0.0:
        raise ValueError(f"display_scale must be finite and positive, got {display_scale}")

    base_path.parent.mkdir(parents=True, exist_ok=True)
    array_path = base_path.with_suffix(".npy")
    preview_path = base_path.with_suffix(".png")
    np.save(array_path, rgb)

    linear = np.maximum(rgb.astype(np.float64) * display_scale, 0.0)
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    preview = np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(preview, mode="RGB").save(preview_path)
    return array_path, preview_path


def run_case(case: dict[str, Any], output_directory: Path) -> dict[str, Any]:
    import pale

    validate_case(case)
    scene_xml, pointcloud_ply = scene_paths(case)
    if not scene_xml.is_file() or not pointcloud_ply.is_file():
        raise FileNotFoundError(f"Missing fixture: scene={scene_xml}, ply={pointcloud_ply}")

    output_directory.mkdir(parents=True, exist_ok=True)
    settings = deepcopy(case["settings"])
    renderer = pale.Renderer(str(ASSETS_ROOT), str(scene_xml), str(pointcloud_ply), settings)
    camera = str(case["camera"])
    if camera not in list(renderer.get_camera_names()):
        raise ValueError(f"{case['name']}: camera '{camera}' is unavailable")

    point_parameters = renderer.get_point_parameters()
    point_count = int(np.asarray(point_parameters["position"]).shape[0])
    index = int(case["index"])
    if index >= point_count:
        raise ValueError(f"{case['name']}: index {index} is outside point count {point_count}")

    parameter = str(case["parameter"])
    values = [float(value) for value in case["values"]]
    epsilons = [float(epsilon) for epsilon in case["epsilons"]]
    check = case["check"]
    allow_zero_signal = bool(check.get("allow_zero_signal", False))
    rows: list[dict[str, Any]] = []
    value_summaries: list[dict[str, Any]] = []
    image_artifacts: list[dict[str, Any]] = []
    images_directory = output_directory / "images"

    def save_image(label: str, image: np.ndarray, *, display_scale: float = 1.0) -> dict[str, Any]:
        array_path, preview_path = _save_rgb_artifact(
            image,
            images_directory / label,
            display_scale=display_scale,
        )
        artifact = {
            "label": label,
            "linear_npy": str(array_path.relative_to(output_directory)),
            "preview_png": str(preview_path.relative_to(output_directory)),
            "display_scale": display_scale,
        }
        image_artifacts.append(artifact)
        return artifact

    for value_index, value in enumerate(values):
        _set_parameter(renderer, parameter, value, index)
        baseline = _render_rgb(renderer, camera)
        repeated = _render_rgb(renderer, camera)
        artifact_prefix = f"value_{value_index:02d}"
        baseline_artifact = save_image(f"{artifact_prefix}_baseline", baseline)
        repeated_artifact = save_image(f"{artifact_prefix}_repeat", repeated)
        repeatability_max_abs = float(np.max(np.abs(repeated.astype(np.float64) - baseline.astype(np.float64))))
        repeatability_pass = math.isfinite(repeatability_max_abs) and (
            repeatability_max_abs <= float(check["repeatability_tolerance"])
        )

        objective = make_objective(tuple(baseline.shape), case.get("objective", {}))
        objective_value, image_gradient = objective.evaluate(baseline, gradient=True)
        assert image_gradient is not None
        gradients, _ = renderer.render_backward({camera: np.ascontiguousarray(image_gradient, dtype=np.float32)})
        analytic = _extract_gradient(gradients, parameter, index)

        finite_differences: list[float] = []
        row_passes: list[bool] = []
        perturbation_artifacts: list[dict[str, Any]] = []
        for epsilon_index, epsilon in enumerate(epsilons):
            _set_parameter(renderer, parameter, value + epsilon, index)
            plus_image = _render_rgb(renderer, camera)
            plus_objective, _ = objective.evaluate(plus_image, gradient=False)

            _set_parameter(renderer, parameter, value - epsilon, index)
            minus_image = _render_rgb(renderer, camera)
            minus_objective, _ = objective.evaluate(minus_image, gradient=False)

            epsilon_prefix = f"{artifact_prefix}_epsilon_{epsilon_index:02d}"
            plus_artifact = save_image(f"{epsilon_prefix}_plus", plus_image)
            minus_artifact = save_image(f"{epsilon_prefix}_minus", minus_image)
            absolute_difference = np.abs(
                plus_image.astype(np.float64) - minus_image.astype(np.float64)
            ).astype(np.float32)
            maximum_pixel_difference = float(np.max(absolute_difference))
            difference_display_scale = (
                1.0 / maximum_pixel_difference if maximum_pixel_difference > 0.0 else 1.0
            )
            difference_artifact = save_image(
                f"{epsilon_prefix}_abs_difference",
                absolute_difference,
                display_scale=difference_display_scale,
            )
            perturbation_artifacts.append(
                {
                    "epsilon": epsilon,
                    "plus": plus_artifact,
                    "minus": minus_artifact,
                    "absolute_difference": difference_artifact,
                    "maximum_pixel_difference": maximum_pixel_difference,
                }
            )

            finite_difference = float((plus_objective - minus_objective) / (2.0 * epsilon))
            finite_differences.append(finite_difference)
            comparison = compare_gradients(analytic, finite_difference, check, allow_zero_signal)
            if not math.isfinite(objective_value):
                comparison["finite"] = False
                comparison["pass"] = False
            row_passes.append(bool(comparison["pass"]))
            rows.append(
                {
                    "case": case["name"],
                    "stage": case["stage"],
                    "value_index": value_index,
                    "value": value,
                    "epsilon": epsilon,
                    "objective": objective_value,
                    "analytic_grad": analytic,
                    "fd_grad": finite_difference,
                    "absolute_error": comparison["absolute_error"],
                    "relative_error": comparison["relative_error"],
                    "tolerance": comparison["tolerance"],
                    "signal": comparison["signal"],
                    "repeatability_max_abs": repeatability_max_abs,
                    "maximum_pixel_difference": maximum_pixel_difference,
                    "plus_preview_png": plus_artifact["preview_png"],
                    "minus_preview_png": minus_artifact["preview_png"],
                    "absolute_difference_preview_png": difference_artifact["preview_png"],
                    "pass": comparison["pass"],
                }
            )

        _set_parameter(renderer, parameter, value, index)
        finite_difference_spread = max(finite_differences) - min(finite_differences)
        consistency_scale = max(
            abs(analytic),
            *(abs(number) for number in finite_differences),
            float(check["minimum_signal"]),
        )
        consistency_relative = abs(finite_difference_spread) / consistency_scale
        consistency_pass = (
            len(finite_differences) == 1
            or consistency_relative <= float(check["fd_consistency_relative_tolerance"])
        )
        value_pass = repeatability_pass and consistency_pass and all(row_passes)
        value_summaries.append(
            {
                "value": value,
                "objective": objective_value,
                "analytic_grad": analytic,
                "fd_gradients": finite_differences,
                "fd_consistency_relative": consistency_relative,
                "fd_consistency_pass": consistency_pass,
                "repeatability_max_abs": repeatability_max_abs,
                "repeatability_pass": repeatability_pass,
                "baseline": baseline_artifact,
                "repeat": repeated_artifact,
                "perturbations": perturbation_artifacts,
                "pass": value_pass,
            }
        )

    case_pass = all(summary["pass"] for summary in value_summaries)
    maximum_relative_error = max(float(row["relative_error"]) for row in rows)
    maximum_absolute_error = max(float(row["absolute_error"]) for row in rows)
    result = {
        "schema_version": 2,
        "name": case["name"],
        "stage": case["stage"],
        "scene": case["scene"],
        "ply": case["ply"],
        "camera": camera,
        "parameter": parameter,
        "index": index,
        "settings": settings,
        "objective": case.get("objective", {}),
        "check": check,
        "values": value_summaries,
        "image_artifacts": image_artifacts,
        "maximum_relative_error": maximum_relative_error,
        "maximum_absolute_error": maximum_absolute_error,
        "pass": case_pass,
    }
    _write_csv(output_directory / "samples.csv", rows)
    (output_directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def format_result(result: dict[str, Any]) -> str:
    status = "PASS" if result["pass"] else "FAIL"
    lines = [
        f"[{status}] {result['stage']}/{result['name']} ",
        f"  fixture={result['scene']}/{result['ply']} parameter={result['parameter']}[{result['index']}]",
        f"  max_abs={result['maximum_absolute_error']:.3e} "
        f"max_rel={100.0 * result['maximum_relative_error']:.2f}%",
    ]
    for value in result["values"]:
        lines.append(
            f"  value={value['value']:+.6g} analytic={value['analytic_grad']:+.6e} "
            f"fd={','.join(f'{number:+.6e}' for number in value['fd_gradients'])} "
            f"fd_spread={100.0 * value['fd_consistency_relative']:.2f}% "
            f"repeat={value['repeatability_max_abs']:.1e} "
            f"{'PASS' if value['pass'] else 'FAIL'}"
        )
    return "\n".join(lines)
