from __future__ import annotations

from typing import Any, Sequence


METRICS_COLUMNS = (
    "iteration",
    "active_camera_name",
    "active_camera_count",
    "loss_average_camera_count",
    "loss_average_expected_camera_count",
    "loss_average_is_complete",
    "loss_rgb_mean",
    "loss_rgb_l2_mean",
    "loss_rgb_dssim_mean",
    "loss_depth_distortion_raw_mean",
    "loss_depth_distortion_weighted_mean",
    "loss_normal_consistency_raw_mean",
    "loss_normal_consistency_weighted_mean",
    "loss_opacity_prior_raw_mean",
    "loss_opacity_prior_weighted_mean",
    "loss_intra_slab_depth_raw_mean",
    "loss_intra_slab_depth_weighted_mean",
    "loss_curvature_scale_raw_mean",
    "loss_curvature_scale_weighted_mean",
    "loss_total_mean",
    "num_points",
    "densification_new_points",
    "densification_clone_points",
    "densification_split_points",
    "densification_position_split_points",
    "densification_curvature_split_points",
    "densification_clone_points_total",
    "densification_split_points_total",
    "densification_position_split_points_total",
    "densification_curvature_split_points_total",
    "densification_clone_points_active",
    "densification_split_points_active",
    "densification_position_split_points_active",
    "densification_curvature_split_points_active",
    "prune_scale_area_points",
    "prune_inactive_transport_points",
    "iteration_time_sec",
    "total_time_sec",
    "grad_position_renderer_norm",
    "grad_position_renderer_max",
    "grad_position_surface_regularizer_norm",
    "grad_position_surface_regularizer_max",
    "grad_position_total_norm",
    "grad_position_total_max",
    "grad_opacity_total_norm",
    "grad_opacity_total_max",
)


class MetricsCSVWriter:
    """Validate every optimization row against the canonical CSV schema."""

    def __init__(self, csv_writer: Any, column_names: Sequence[str] = METRICS_COLUMNS) -> None:
        self._csv_writer = csv_writer
        self._column_names = tuple(column_names)

    def writerow(self, row_values: Sequence[Any]) -> None:
        row_values = list(row_values)
        if len(row_values) != len(self._column_names):
            raise ValueError(
                f"Metrics header has {len(self._column_names)} columns, "
                f"but row has {len(row_values)} values"
            )
        self._csv_writer.writerow(row_values)
