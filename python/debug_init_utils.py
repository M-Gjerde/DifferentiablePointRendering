from __future__ import annotations

import numpy as np


def normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return array / norms


def add_debug_noise_to_initial_parameters(
    positions: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
    *,
    seed_positions: int = 7,
    seed_colors: int = 12345,
    noise_sigma_translation: float = 0.07,
    noise_sigma_tangent: float = 0.05,
    noise_sigma_scale: float = 0.03,
    noise_sigma_color: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Debug utility: add Gaussian noise to initial parameters.

    Returns new (positions, tangent_u, tangent_v, scales, colors) arrays.
    """

    # --- Positions ---
    rng_pos = np.random.default_rng(seed_positions)
    noisy_positions = positions.copy()
    noisy_positions += rng_pos.normal(
        loc=0.0,
        scale=noise_sigma_translation,
        size=noisy_positions.shape,
    )
    noisy_positions = noisy_positions.astype(np.float32)

    # --- Tangents ---
    noisy_tangent_u = tangent_u.copy()
    noisy_tangent_v = tangent_v.copy()

    noisy_tangent_u += rng_pos.normal(
        0.0,
        noise_sigma_tangent,
        noisy_tangent_u.shape,
    )
    noisy_tangent_v += rng_pos.normal(
        0.0,
        noise_sigma_tangent,
        noisy_tangent_v.shape,
    )

    noisy_tangent_u = normalize_rows(noisy_tangent_u).astype(np.float32)
    noisy_tangent_v = normalize_rows(noisy_tangent_v).astype(np.float32)

    # --- Scales ---
    noisy_scales = scales.copy()
    noisy_scales += rng_pos.normal(
        0.0,
        noise_sigma_scale,
        noisy_scales.shape,
    )
    noisy_scales = noisy_scales.astype(np.float32)

    # --- Colors (grayscale noise per point) ---
    rng_col = np.random.default_rng(seed_colors)
    noisy_colors = colors.copy()
    gray_noise = rng_col.normal(
        0.0, noise_sigma_color, size=(noisy_colors.shape[0], 1)
    )
    noisy_colors += gray_noise
    noisy_colors = noisy_colors.astype(np.float32)

    return (
        noisy_positions,
        noisy_tangent_u,
        noisy_tangent_v,
        noisy_scales,
        noisy_colors,
    )
