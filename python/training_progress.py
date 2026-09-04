from __future__ import annotations

from collections.abc import Mapping
from typing import Any


WEIGHTED_LOSS_FIELDS = (
    ("depth", "total_depth_distortion_loss_weighted"),
    ("normal", "total_normal_loss_weighted"),
    ("opacity", "total_opacity_prior_loss_weighted"),
    ("slab", "total_intra_slab_depth_loss_weighted"),
    ("curvature", "total_curvature_scale_loss_weighted"),
)


def make_training_progress_postfix(
        loss_state: Mapping[str, Any],
        geometry_row: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Build the compact values shown beside the training progress bar."""
    postfix = {
        "loss": f"{float(loss_state['total_loss_value']):.3e}",
        "rgb": f"{float(loss_state['total_rgb_loss_value']):.3e}",
    }

    for label, field_name in WEIGHTED_LOSS_FIELDS:
        value = float(loss_state.get(field_name, 0.0))
        if value != 0.0:
            postfix[label] = f"{value:.3e}"

    if geometry_row is not None:
        for label, field_name in (
                ("CD", "cd"),
                ("acc", "accuracy"),
                ("comp", "completion"),
        ):
            value = geometry_row.get(field_name)
            if isinstance(value, (str, int, float)):
                postfix[label] = f"{float(value):.3e}"

    return postfix
