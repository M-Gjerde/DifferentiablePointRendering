from PIL import Image
import numpy as np


def load_image_rgb(path: str) -> np.ndarray:
    """
    Load an image, convert to RGB, and normalize to [0,1].
    """
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def compute_mse(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """
    Compute mean squared error between two images.
    """
    if image_a.shape != image_b.shape:
        raise ValueError("Images must have the same shape")
    return float(np.mean((image_a - image_b) ** 2))


def compute_error_std(
    image_a: np.ndarray,
    image_b: np.ndarray,
) -> tuple[float, float]:
    """
    Compute standard deviation of:
      - squared error
      - absolute error
    over all pixels and channels.
    """
    if image_a.shape != image_b.shape:
        raise ValueError("Images must have the same shape")

    difference = image_a - image_b

    squared_error = difference ** 2
    absolute_error = np.abs(difference)

    std_squared_error = float(np.std(squared_error))
    std_absolute_error = float(np.std(absolute_error))

    return std_squared_error, std_absolute_error


# Paths
target_path = "/home/magnus/CLionProjects/DifferentiablePointRendering/python/output/mse_comparison/render_target_DatasetCam_003.png"
ours_path = "/home/magnus/CLionProjects/DifferentiablePointRendering/python/output/mse_comparison/render_final_DatasetCam_003.png"
gs_path = "/home/magnus/CLionProjects/DifferentiablePointRendering/python/output/mse_comparison/2dgs_render_final.png"

# Load images
target = load_image_rgb(target_path)
ours = load_image_rgb(ours_path)
gs = load_image_rgb(gs_path)

# Compute MSE
mse_ours = compute_mse(ours, target)
mse_gs = compute_mse(gs, target)

# Compute STD
std_sq_ours, std_abs_ours = compute_error_std(ours, target)
std_sq_gs, std_abs_gs = compute_error_std(gs, target)

print(f"MSE (Ours vs Target):  {mse_ours:.6e}")
print(f"STD squared error:     {std_sq_ours:.6e}")
print(f"STD absolute error:    {std_abs_ours:.6e}")
print()
print(f"MSE (2DGS vs Target):  {mse_gs:.6e}")
print(f"STD squared error:     {std_sq_gs:.6e}")
print(f"STD absolute error:    {std_abs_gs:.6e}")
