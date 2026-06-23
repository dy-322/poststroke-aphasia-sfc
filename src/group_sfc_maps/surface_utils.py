from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colorbar import ColorbarBase
from nilearn import datasets, plotting
from neurocat.plotting import draw_and_save as draw_surface


ROI_NUMBER = 450
SURFACE_CMAP = "viridis"


def compute_padded_color_range(arrays: list[np.ndarray], pad_fraction: float = 0.04) -> tuple[float, float]:
    finite_parts = []
    for values in arrays:
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            finite_parts.append(finite)

    if not finite_parts:
        raise ValueError("No finite values available for color range")

    finite_values = np.concatenate(finite_parts)
    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    pad = max((vmax - vmin) * pad_fraction, 1e-3)
    return vmin - pad, vmax + pad


def trim_white_edges(image: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    rgb = image[..., :3]
    mask = np.any(rgb < threshold, axis=2)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1, :]


def add_scaled_image_axis(
    fig: plt.Figure,
    image: np.ndarray,
    left: float,
    center_y: float,
    width: float,
    fig_size: tuple[float, float],
) -> plt.Axes:
    fig_width, fig_height = fig_size
    image_aspect = image.shape[1] / image.shape[0]
    height = width * fig_width / (image_aspect * fig_height)
    ax = fig.add_axes([left, center_y - height / 2, width, height])
    ax.imshow(image)
    ax.axis("off")
    return ax


def save_roi_surface_map(
    values: np.ndarray,
    title: str,
    color_range: tuple[float, float],
    output_dir: Path,
    filename_stem: str,
    atlas_path: Path,
    cmap: str = SURFACE_CMAP,
    colorbar_label: str = "Mean SFC",
    colorbar_range: tuple[float, float] | None = None,
    colorbar_cmap=None,
) -> None:
    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata().astype(int)

    subcortical_values = values[:50]
    cortical_values = values[50:]

    roi_vol = np.full(atlas_data.shape, np.nan, dtype=float)
    for roi_label in range(1, 51):
        roi_vol[atlas_data == roi_label] = subcortical_values[roi_label - 1]
    roi_img = nib.Nifti1Image(roi_vol, atlas_img.affine, atlas_img.header)

    vmin, vmax = color_range
    bg_img = datasets.load_mni152_template(resolution=1)
    fig_nilearn = plt.figure(figsize=(40, 8))
    plotting.plot_stat_map(
        roi_img,
        bg_img=bg_img,
        threshold=0.0,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        display_mode="z",
        cut_coords=[-25, -20, -10, 0, 10, 20, 22],
        draw_cross=False,
        colorbar=False,
        figure=fig_nilearn,
        black_bg=False,
        annotate=False,
    )

    temp_dir = Path(tempfile.gettempdir())
    safe_stem = filename_stem.replace(" ", "_")
    temp_nilearn = temp_dir / f"{safe_stem}_subcortical.png"
    temp_surface_base = temp_dir / f"{safe_stem}_surface"
    temp_surface = temp_surface_base.with_suffix(".png")

    fig_nilearn.savefig(temp_nilearn, bbox_inches="tight", dpi=260, pad_inches=0)
    plt.close(fig_nilearn)

    draw_surface(
        layer1=cortical_values,
        colorbar=cmap,
        color_range=color_range,
        fig_name=str(temp_surface_base),
        layout="row",
        outline=True,
        trim=False,
        tp="fslr",
        mesh="veryinflated",
        save_png_only=True,
        display=False,
        size=(6000, 1000),
    )

    img_surface = trim_white_edges(mpimg.imread(temp_surface))
    img_nilearn = trim_white_edges(mpimg.imread(temp_nilearn))

    fig_size = (22, 10)
    fig = plt.figure(figsize=fig_size)
    fig.patch.set_facecolor("white")
    fig.text(0.035, 0.955, title, fontsize=18, ha="left", va="top")

    add_scaled_image_axis(fig, img_surface, left=0.035, center_y=0.70, width=0.84, fig_size=fig_size)
    add_scaled_image_axis(fig, img_nilearn, left=0.035, center_y=0.27, width=0.84, fig_size=fig_size)

    ax_cbar = fig.add_axes([0.91, 0.14, 0.018, 0.73])
    cbar_vmin, cbar_vmax = colorbar_range if colorbar_range is not None else color_range
    cbar_cmap = colorbar_cmap if colorbar_cmap is not None else cmap
    norm = mpl.colors.Normalize(vmin=cbar_vmin, vmax=cbar_vmax)
    cbar = ColorbarBase(ax_cbar, cmap=cbar_cmap, norm=norm, orientation="vertical")
    cbar.ax.set_title(colorbar_label, fontsize=9, pad=8)
    ax_cbar.set_yticks([cbar_vmin, (cbar_vmin + cbar_vmax) / 2, cbar_vmax])
    ax_cbar.tick_params(labelsize=8)

    fig.savefig(output_dir / f"{filename_stem}.pdf", dpi=300, facecolor="white")
    fig.savefig(output_dir / f"{filename_stem}.png", dpi=300, facecolor="white")
    plt.close(fig)

    temp_surface.unlink(missing_ok=True)
    temp_nilearn.unlink(missing_ok=True)
