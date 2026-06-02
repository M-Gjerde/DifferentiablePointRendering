#!/usr/bin/env python3
"""
pfm_gradient_viewer.py
Usage:
    python pfm_gradient_viewer.py <image.pfm> [--dpi 100] [--flipud]
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# ----------------- PFM Reader -----------------

def read_pfm(path: str):
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").strip()
        if header not in ("PF", "Pf"):
            raise ValueError("Not a PFM file")

        # read dimensions (skip comments)
        line = f.readline().decode("ascii").strip()
        while line.startswith("#") or len(line) == 0:
            line = f.readline().decode("ascii").strip()
        try:
            width, height = map(int, line.split())
        except Exception:
            raise ValueError("Malformed PFM dimensions")

        # scale: sign encodes endianness
        scale = float(f.readline().decode("ascii").strip())
        little = scale < 0
        dtype = "<f4" if little else ">f4"

        channels = 3 if header == "PF" else 1
        data = np.frombuffer(f.read(), dtype=dtype)
        expected = width * height * channels
        if data.size != expected:
            raise ValueError(f"Unexpected data size: got {data.size}, expected {expected}")
        data = data.reshape((height, width, channels)).astype(np.float32)
        return data  # top-left at [0,0]

# ----------------- Mapping -----------------

def to_scalar(img_hwC: np.ndarray, mode: str):
    if img_hwC.shape[2] == 1:
        return img_hwC[..., 0]
    r, g, b = img_hwC[..., 0], img_hwC[..., 1], img_hwC[..., 2]
    if mode == "Luma":
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    if mode == "R":
        return r
    if mode == "G":
        return g
    if mode == "B":
        return b
    return r

def normalize_signed(v: np.ndarray, abs_quantile: float, gain: float):
    v = np.asarray(v, dtype=np.float32)
    abs_q = np.quantile(np.abs(v), abs_quantile)
    denom = max(abs_q, 1e-12) / max(gain, 1e-6)
    x = np.clip(v / denom, -1.0, 1.0)
    return x

# ----------------- Viewer -----------------

def view_pfm_signed(pfm_path: str, dpi: int = 100, flipud: bool = False):
    img = read_pfm(pfm_path)
    if flipud:
        img = np.flipud(img)

    # report ranges
    if img.shape[2] == 1:
        ch_min, ch_max = float(np.min(img[..., 0])), float(np.max(img[..., 0]))
        print(f"[PFM] Channel min/max: {ch_min:.6g} {ch_max:.6g}")
    else:
        rmin, rmax = float(np.min(img[..., 0])), float(np.max(img[..., 0]))
        gmin, gmax = float(np.min(img[..., 1])), float(np.max(img[..., 1]))
        bmin, bmax = float(np.min(img[..., 2])), float(np.max(img[..., 2]))
        print(f"[PFM] R min/max: {rmin:.6g} {rmax:.6g}")
        print(f"[PFM] G min/max: {gmin:.6g} {gmax:.6g}")
        print(f"[PFM] B min/max: {bmin:.6g} {bmax:.6g}")

    init_mode = "Luma"
    init_q = 0.99
    init_gain = 1.0

    scalar = to_scalar(img, init_mode)
    norm = normalize_signed(scalar, init_q, init_gain)

    # --- Fixed figure size: 16:10 inches ---
    FIG_W_IN, FIG_H_IN = 16.0, 10.0
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=dpi)

    # Layout fractions within fixed figure
    # Bottom controls band occupies ~10% height
    controls_h = 0.12
    left_pad = 0.06
    right_cbar_w = 0.04
    gap_w = 0.02
    top_pad = 0.06

    # Image axes occupies remaining area on the left
    ax_img = fig.add_axes([
        left_pad,
        controls_h,
        1.0 - left_pad - gap_w - right_cbar_w - 0.04,  # width
        1.0 - controls_h - top_pad                      # height
    ])

    # Keep image aspect. Fit image entirely without distortion.
    # Letterboxing will appear if aspect differs from axes box.
    H, W = scalar.shape
    im_artist = ax_img.imshow(norm, cmap="seismic", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax_img.set_aspect('equal')      # preserve pixel aspect
    ax_img.set_title(pfm_path, pad=6)
    ax_img.axis("off")

    # Colorbar on right
    cax = fig.add_axes([
        1.0 - right_cbar_w - 0.02,
        controls_h,
        right_cbar_w,
        1.0 - controls_h - top_pad
    ])
    cbar = plt.colorbar(im_artist, cax=cax)
    cbar.set_label("Signed value (blue=neg, red=pos)")

    # Sliders
    s_left, s_width = left_pad, 0.88
    s1_bottom = controls_h * 0.55
    s2_bottom = controls_h * 0.20

    ax_q    = fig.add_axes([s_left, s1_bottom, s_width, 0.03])
    ax_gain = fig.add_axes([s_left, s2_bottom, s_width, 0.03])

    s_q    = Slider(ax=ax_q,    label="Abs-quantile", valmin=0.9, valmax=1.0, valinit=init_q,   valstep=0.001)
    s_gain = Slider(ax=ax_gain, label="Gain",         valmin=0.1, valmax=50.0,  valinit=init_gain, valstep=0.1)

    # Channel mode (if RGB), fixed small panel in the image area
    current_mode = {"val": init_mode}
    if img.shape[2] == 3:
        ax_mode = fig.add_axes([
            left_pad + 0.005,
            controls_h + 0.005,
            0.10,
            0.18
        ])
        mode_selector = RadioButtons(ax_mode, ("Luma", "R", "G", "B"), active=0)
        def on_mode(label):
            current_mode["val"] = label
            update(None)
        mode_selector.on_clicked(on_mode)

    # Update
    def update(_):
        sc = to_scalar(img, current_mode["val"])
        nm = normalize_signed(sc, s_q.val, s_gain.val)
        im_artist.set_data(nm)
        fig.canvas.draw_idle()

    s_q.on_changed(update)
    s_gain.on_changed(update)

    plt.show()

# ----------------- CLI -----------------

def main():
    p = argparse.ArgumentParser(description="PFM signed viewer with fixed 16:10 figure size.")
    p.add_argument("pfm", help="Path to .pfm image")
    p.add_argument("--dpi", type=int, default=100, help="Figure DPI")
    p.add_argument("--flipud", action="store_true", help="Flip image vertically after load")
    args = p.parse_args()

    view_pfm_signed(args.pfm, dpi=args.dpi, flipud=args.flipud)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python pfm_gradient_viewer.py <image.pfm> [--dpi 100] [--flipud]")
        sys.exit(1)
    main()
