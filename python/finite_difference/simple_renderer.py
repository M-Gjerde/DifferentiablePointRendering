# main.py
from pathlib import Path
import argparse

import numpy as np
import pale


from finite_difference.finite_diff_helpers import save_rgb_preview_png, finite_difference_opacity, write_fd_images
from io_utils import load_target_image
from losses import compute_l2_grad, compute_l2_loss
import matplotlib.pyplot as plt


# This cell only defines a reusable plotting helper for your sweep CSV.
# No data is plotted here.

def plot_fd_linear_sweep_from_csv(csv_path: str, output_png: str = None):
    """
    Plot eps vs FD gradient from a sweep CSV with columns: eps,fd_grad,rel_err.
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    eps = data[:, 0]
    fd_grad = data[:, 2]

    plt.figure()
    plt.plot(eps, fd_grad)
    plt.xlabel("epsilon")
    plt.ylabel("relative error")
    plt.title("Finite-difference opacity sweep")

    if output_png is not None:
        plt.savefig(output_png, bbox_inches="tight")
    else:
        plt.show()

print("Helper function 'plot_fd_linear_sweep_from_csv' defined.")


def run_opacity_fd_linear_sweep(
    renderer,
    camera: str,
    target_image: np.ndarray,
    analytic_grad_scalar: float,
    index: int,
    output_dir: Path,
    eps_low: float = 0.05,
    eps_high: float = 0.5,
    num_eps: int = 50,
    num_avg: int = 1,
    rel_err_threshold: float = 0.05,      # 5%
    fd_variation_threshold: float = 0.02, # 2% neighbor-to-neighbor change
) -> None:

    eps_values = np.linspace(eps_low, eps_high, num_eps, dtype=np.float64)

    fd_mean = np.zeros(num_eps, dtype=np.float64)
    fd_std  = np.zeros(num_eps, dtype=np.float64)
    rel_errs = np.zeros(num_eps, dtype=np.float64)

    for i, eps in enumerate(eps_values):
        fd_samples = np.zeros(num_avg, dtype=np.float64)

        for k in range(num_avg):
            rgb_minus, rgb_plus, _ = finite_difference_opacity(
                renderer, index, float(eps), camera
            )

            loss_minus = compute_l2_loss(rgb_minus, target_image)
            loss_plus  = compute_l2_loss(rgb_plus, target_image)

            fd_samples[k] = (loss_plus - loss_minus) / (2.0 * float(eps))

        fd_mean[i] = float(fd_samples.mean())
        fd_std[i]  = float(fd_samples.std(ddof=1))

        rel_errs[i] = np.abs(fd_mean[i] - analytic_grad_scalar) / (
            np.abs(analytic_grad_scalar) + 1e-12
        )

        print(
            f"eps={eps: .6f}  "
            f"FD(mean)={fd_mean[i]: .8e}  "
            f"std={fd_std[i]: .3e}  "
            f"rel_err={rel_errs[i]: .3%}"
        )

    # ---- Plateau detection (on the mean) ----
    denom = np.maximum(np.abs(fd_mean[:-1]), 1e-12)
    neighbor_change = np.abs(fd_mean[1:] - fd_mean[:-1]) / denom

    stable = np.hstack([neighbor_change < fd_variation_threshold, False])
    good = (rel_errs < rel_err_threshold)
    plateau_mask = good & stable

    # find longest contiguous plateau segment
    best_len = 0
    best_start = None
    best_end = None

    start = None
    for i, ok in enumerate(plateau_mask):
        if ok and start is None:
            start = i
        if (not ok or i == len(plateau_mask) - 1) and start is not None:
            end = i if ok and i == len(plateau_mask) - 1 else i - 1
            length = end - start + 1
            if length > best_len:
                best_len = length
                best_start, best_end = start, end
            start = None

    # ---- Save CSV (now includes std) ----
    csv_path = output_dir / "fd_opacity_linear_sweep.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("eps,fd_mean,fd_std,rel_err\n")
        for eps, m, s, r in zip(eps_values, fd_mean, fd_std, rel_errs):
            f.write(f"{eps},{m},{s},{r}\n")

    # ---- Plot (mean curve + 1σ band) ----
    png_path = output_dir / "fd_opacity_linear_sweep.png"

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(eps_values, fd_mean, label="FD mean")
    plt.fill_between(
        eps_values,
        fd_mean - fd_std,
        fd_mean + fd_std,
        alpha=0.25,
        label="±1 std",
    )
    plt.axhline(analytic_grad_scalar, linestyle="--", linewidth=1, label="Analytic")
    plt.xlabel("epsilon")
    plt.ylabel("FD gradient")
    plt.title("Finite-difference opacity sweep (averaged)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    if best_start is not None:
        eps_a = eps_values[best_start]
        eps_b = eps_values[best_end]
        print(f"\nPlateau candidate: eps in [{eps_a:.6f}, {eps_b:.6f}]")
        print(f"Suggested eps: {(0.5*(eps_a+eps_b)):.6f}")
    else:
        print("\nNo clear plateau found in [0.005, 0.5].")
        print("If so, increase samples or enforce common random numbers.")

    print(f"\nWrote:")
    print(f"  {csv_path}")
    print(f"  {png_path}")




def main(args) -> None:
    renderer_settings = {
        "photons": 1e6,
        "bounces": 4,
        "forward_passes": 100,
        "gather_passes": 1,
        "adjoint_bounces": 1,
        "adjoint_passes": 100,
        "logging": 3
    }

    assets_root = Path(__file__).parent.parent.parent / "Assets"
    scene_xml = args.scene + ".xml"
    pointcloud_ply = args.ply + ".ply"

    output_dir = (
        Path(__file__).parent / "Output" / args.scene / "opacity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = pale.Renderer(str(assets_root), scene_xml, pointcloud_ply, renderer_settings)

    camera = args.camera
    target_image = load_target_image(output_dir / Path(camera + "_target.png"))

    # Render ONCE (baseline)
    rendered_images = renderer.render_forward()
    rendered_image = rendered_images[camera]
    save_rgb_preview_png(rendered_image, output_dir / Path(camera + "_rendered.png"))

    # Analytic gradient of scalar loss wrt opacity[index]
    loss_grad_image = compute_l2_grad(rendered_image, target_image)
    gradients, _adjoint_images = renderer.render_backward({camera: loss_grad_image})

    analytic_opacity_grads = np.asarray(gradients["opacity"], dtype=np.float32).squeeze()
    if np.ndim(analytic_opacity_grads) == 0:
        analytic_grad_scalar = float(analytic_opacity_grads)
    else:
        analytic_grad_scalar = float(analytic_opacity_grads[args.index])

    print("AN Gradient:", analytic_grad_scalar)

    run_opacity_fd_linear_sweep(
        renderer=renderer,
        camera=camera,
        target_image=target_image,
        analytic_grad_scalar=analytic_grad_scalar,
        index=(args.index if args.index >= 0 else 0),
        output_dir=output_dir
    )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite-difference gradient visualization for Pale renderer."
    )
    parser.add_argument(
        "--ply",
        type=str,
        default="initial",
        help="Points (PLY without extension). Default: 'initial'.",
    )

    parser.add_argument(
        "--scene",
        type=str,
        default="empty",
        help="Which scene file to use (without extension). Default: empty",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Gaussian index to perturb (>=0 for single, -1 for all). Default: -1.",
    )

    parser.add_argument(
        "--param",
        type=str,
        choices=["translation", "rotation", "scale", "translation_rotation", "opacity"],
        default="translation",
    )

    parser.add_argument(
        "--axis",
        type=str,
        choices=["x", "y", "z"],
        default="axis of choice",
        help="Which axis to finite-difference: 'translation', 'rotation', or 'scale'.",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Where to output files",
        default="output"
    )
    parser.add_argument(
        "--camera",
        type=str,
        help="Which camera (In the xml file) to render from",
        default="camera1"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)