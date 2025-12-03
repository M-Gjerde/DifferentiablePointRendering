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
    opacities: np.ndarray,
    betas: np.ndarray,
    *,
    seed_positions: int = 16,
    seed_colors: int = 12345,
    noise_sigma_translation: float = 0.0,
    noise_sigma_tangent: float = 0.0,
    noise_sigma_scale: float = 0.00,
    noise_sigma_color: float = 0.0,
    noise_sigma_opacity: float = 0.00,
    noise_sigma_beta: float = 0.0,
    index: int = -1,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Debug utility: add Gaussian noise to initial parameters.

    If index == -1: perturb all rows.
    If index >= 0: perturb only the given row index.

    Returns new (positions, tangent_u, tangent_v, scales, colors, opacities, betas) arrays.
    """
    n_points = positions.shape[0]
    if index >= n_points:
        raise IndexError(f"index {index} out of range for {n_points} points")

    # Helper to get slice & size for noise
    if index < 0:
        row_slice = slice(None)  # all rows
        n_rows = n_points
    else:
        row_slice = slice(index, index + 1)
        n_rows = 1

    # --- Positions ---
    rng_pos = np.random.default_rng(seed_positions)
    noisy_positions = positions.copy()
    noisy_positions[row_slice] += rng_pos.normal(
        loc=0.0,
        scale=noise_sigma_translation,
        size=(n_rows, noisy_positions.shape[1]),
    )
    noisy_positions = noisy_positions.astype(np.float32)

    # --- Tangents ---
    noisy_tangent_u = tangent_u.copy()
    noisy_tangent_v = tangent_v.copy()

    noisy_tangent_u[row_slice] += rng_pos.normal(
        0.0,
        noise_sigma_tangent,
        (n_rows, noisy_tangent_u.shape[1]),
    )
    noisy_tangent_v[row_slice] += rng_pos.normal(
        0.0,
        noise_sigma_tangent,
        (n_rows, noisy_tangent_v.shape[1]),
    )

    # Keep behavior: normalize all rows (cheap and consistent)
    noisy_tangent_u = normalize_rows(noisy_tangent_u).astype(np.float32)
    noisy_tangent_v = normalize_rows(noisy_tangent_v).astype(np.float32)

    # --- Scales ---
    noisy_scales = scales.copy()
    noisy_scales[row_slice] += rng_pos.normal(
        0.0,
        noise_sigma_scale,
        (n_rows, noisy_scales.shape[1]),
    )
    noisy_scales = noisy_scales.astype(np.float32)

    # --- Colors (grayscale noise per point) ---
    rng_col = np.random.default_rng(seed_colors)
    noisy_colors = colors.copy()
    gray_noise = rng_col.normal(
        0.0,
        noise_sigma_color,
        size=(n_rows, 1),
    )
    noisy_colors[row_slice] += gray_noise
    noisy_colors = noisy_colors.astype(np.float32)

    # --- Opacities ---
    rng_opacity = np.random.default_rng(seed_colors)
    noisy_opacities = opacities.copy()
    opacity_noise = rng_opacity.normal(
        0.0,
        noise_sigma_opacity,
        size=noisy_opacities[row_slice].shape,
    )
    noisy_opacities[row_slice] += opacity_noise
    noisy_opacities = noisy_opacities.astype(np.float32)

    # --- Betas ---
    rng_beta = np.random.default_rng(seed_colors)
    noisy_betas = betas.copy()
    beta_noise = rng_beta.normal(
        0.0,  # your mean
        noise_sigma_beta,
        size=noisy_betas[row_slice].shape,
    )
    noisy_betas[row_slice] += beta_noise
    noisy_betas = noisy_betas.astype(np.float32)

    return (
        noisy_positions,
        noisy_tangent_u,
        noisy_tangent_v,
        noisy_scales,
        noisy_colors,
        noisy_opacities,
        noisy_betas,
    )
