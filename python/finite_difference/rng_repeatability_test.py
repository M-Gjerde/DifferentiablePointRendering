# rng_repeatability_test.py
#
# Renders once to define the "target" (render #1), then renders again and compares.
# This isolates whether RNG / MC noise is identical between repeated renders.
#
# Usage:
#   python rng_repeatability_test.py --scene empty --ply initial --camera camera1 --seed 42
#
import argparse
from pathlib import Path

import numpy as np
import pale

from losses import compute_l2_loss


def render_once(renderer: "pale.Renderer", camera: str) -> np.ndarray:
    images = renderer.render_forward()
    img = np.asarray(images[camera + "_raw"], dtype=np.float32)[..., :3]  # (H,W,3)
    return img


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def maxabs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def main(args: argparse.Namespace) -> None:
    renderer_settings = {
        "photons": float(args.photons),
        "bounces": int(args.bounces),
        "forward_passes": int(args.forward_passes),
        "gather_passes": int(args.gather_passes),
        "adjoint_bounces": 1,
        "adjoint_passes": 1,
        "logging": int(args.logging),
        "seed": int(args.seed),
    }

    assets_root = Path(__file__).resolve().parents[2] / "Assets"

    scene_path = Path(args.scene).parent
    scene_xml = assets_root / "GradientTests" / f"{args.scene}" / f"{args.scene}.xml"
    pointcloud_ply = assets_root / "GradientTests" / f"{args.scene}"  / scene_path / f"{args.ply}.ply"

    print("Scene XML:", scene_xml)
    print("PLY:", pointcloud_ply)
    print("Camera:", args.camera)
    print("Seed:", args.seed)

    renderer = pale.Renderer(str(assets_root), str(scene_xml), str(pointcloud_ply), renderer_settings)

    # Render #1 defines the target
    img1 = render_once(renderer, args.camera)
    target = img1

    # Render #2 should match #1 if RNG is reset/identical
    img2 = render_once(renderer, args.camera)

    L1 = float(compute_l2_loss(img1, target))  # should be exactly 0 (or extremely close)
    L2 = float(compute_l2_loss(img2, target))  # should be 0 if img2 == img1

    dimg = img2 - img1

    print("\n--- Repeatability (same renderer instance) ---")
    print(f"Loss1 (img1 vs img1): {L1:.10e}")
    print(f"Loss2 (img2 vs img1): {L2:.10e}")
    print(f"Image RMS(diff):      {rms(dimg):.10e}")
    print(f"Image max|diff|:      {maxabs(dimg):.10e}")

    # Optional: render more times and compare to the first
    if args.repeats > 2:
        print(f"\n--- {args.repeats} renders (same instance), all vs first ---")
        for i in range(3, args.repeats + 1):
            im = render_once(renderer, args.camera)
            di = im - img1
            Li = float(compute_l2_loss(im, target))
            print(
                f"#{i}: Loss={Li:.10e}, RMS(diff)={rms(di):.10e}, max|diff|={maxabs(di):.10e}"
            )

    # Optional: recreate renderer (same seed) and compare first images across instances
    if args.test_new_instance:
        renderer2 = pale.Renderer(str(assets_root), str(scene_xml), str(pointcloud_ply), renderer_settings)
        imgA = render_once(renderer2, args.camera)
        dA = imgA - img1
        LA = float(compute_l2_loss(imgA, target))
        print("\n--- Repeatability (new renderer instance, same seed) ---")
        print(f"Loss (imgA vs img1):  {LA:.10e}")
        print(f"Image RMS(diff):      {rms(dA):.10e}")
        print(f"Image max|diff|:      {maxabs(dA):.10e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Pale RNG repeatability test (target = first render)")
    p.add_argument("--ply", type=str, default="initial")
    p.add_argument("--scene", type=str, default="empty")
    p.add_argument("--camera", type=str, default="camera1")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--photons", type=float, default=1e6)
    p.add_argument("--bounces", type=int, default=4)
    p.add_argument("--forward_passes", type=int, default=10)
    p.add_argument("--gather_passes", type=int, default=1)
    p.add_argument("--logging", type=int, default=3)

    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--test_new_instance", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())