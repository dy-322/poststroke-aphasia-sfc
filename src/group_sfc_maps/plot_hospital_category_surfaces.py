from __future__ import annotations

import re
import tempfile
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["pdf.fonttype"] = 42

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colorbar import ColorbarBase
from nilearn import datasets, plotting
from neurocat.plotting import draw_and_save as ds


DATA_ROOT = Path(
    r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main"
    r"\aphasia_connectioncsv_timeseries_paired_20260507"
    r"\aphasia_connectioncsv_timeseries_paired_20260507"
)
SFC_TXT_DIR = DATA_ROOT / "sfc_pca_450_t288" / "sfc_txt_files"
OUTPUT_DIR = DATA_ROOT / "sfc_pca_450_t288" / "group_plots" / "plot_final_style_categories"
AVERAGE_TXT_DIR = OUTPUT_DIR / "average_txt"
FILTER_XLSX = Path(r"D:\download\失语筛选.xlsx")
ATLAS_PATH = Path(
    r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_MNI152NLin6Asym_1mm.nii.gz"
)

ROI_NUMBER = 450
GLOBAL_VMIN = 0.15
GLOBAL_VMAX = 0.50
CMAP = "RdBu_r"

NORMAL_LABEL = "Normal stroke"
ALL_APHASIA_LABEL = "All aphasia"

SUBTYPE_SLUGS = {
    "完全性失语": "global_aphasia",
    "皮质下失语": "subcortical_aphasia",
    "命名性失语": "anomic_aphasia",
    "传导性失语": "conduction_aphasia",
    "运动性失语 (Broca)": "broca_aphasia",
    "感觉性失语 (Wernicke)": "wernicke_aphasia",
    "经皮质感觉性失语": "transcortical_sensory_aphasia",
}
SUBTYPE_DISPLAY_LABELS = {
    "完全性失语": "Global aphasia",
    "皮质下失语": "Subcortical aphasia",
    "命名性失语": "Anomic aphasia",
    "传导性失语": "Conduction aphasia",
    "运动性失语 (Broca)": "Broca aphasia",
    "感觉性失语 (Wernicke)": "Wernicke aphasia",
    "经皮质感觉性失语": "Transcortical sensory aphasia",
}


def trim_white_edges(img_array: np.ndarray) -> np.ndarray:
    mask = np.any(img_array < 1, axis=-1)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    if len(row_indices) == 0 or len(col_indices) == 0:
        return img_array

    return np.ascontiguousarray(
        img_array[
            row_indices[0] : row_indices[-1] + 1,
            col_indices[0] : col_indices[-1] + 1,
        ]
    )


def subject_from_sfc_file(path: Path) -> str:
    prefix = "sub-"
    suffix = "_sfc_values_pca.txt"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected SFC filename: {name}")
    return name[len(prefix) : -len(suffix)]


def primary_case_id(subject_id: str) -> str:
    parts = subject_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse case id from subject id: {subject_id}")
    return parts[1].split("-")[0]


def load_sfc_txt(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text().splitlines():
        if "\t" not in line or line.startswith("ROI") or line.startswith("-"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        try:
            values.append(float(parts[1]))
        except ValueError:
            values.append(np.nan)

    arr = np.asarray(values, dtype=float)
    if arr.shape[0] != ROI_NUMBER:
        raise ValueError(f"{path.name}: expected {ROI_NUMBER} ROI values, got {arr.shape[0]}")
    return arr


def safe_slug(label: str) -> str:
    mapped = SUBTYPE_SLUGS.get(label)
    if mapped:
        return mapped
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    return slug or "unknown_subtype"


def is_normal_category(label: str) -> bool:
    return "normal" in label.lower()


def load_case_categories() -> dict[str, str]:
    df = pd.read_excel(FILTER_XLSX, sheet_name="Sheet1")
    required = {"病历号", "失语类别"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{FILTER_XLSX} missing columns: {sorted(missing)}")

    case_to_category: dict[str, str] = {}
    for _, row in df.dropna(subset=["病历号", "失语类别"]).iterrows():
        case_id = str(int(row["病历号"]))
        category = str(row["失语类别"]).strip()
        case_to_category[case_id] = category
    return case_to_category


def load_group_values() -> tuple[pd.DataFrame, OrderedDict[str, dict[str, object]]]:
    case_to_category = load_case_categories()
    groups: OrderedDict[str, dict[str, object]] = OrderedDict()
    groups["normal_stroke"] = {"label": NORMAL_LABEL, "source_category": "Normal Storke", "values": []}
    groups["all_aphasia"] = {"label": ALL_APHASIA_LABEL, "source_category": "all_aphasia", "values": []}

    rows = []
    for path in sorted(SFC_TXT_DIR.glob("*_sfc_values_pca.txt")):
        subject_id = subject_from_sfc_file(path)
        case_id = primary_case_id(subject_id)
        if case_id not in case_to_category:
            raise ValueError(f"Subject {subject_id} case id {case_id} not found in {FILTER_XLSX}")

        category = case_to_category[case_id]
        values = load_sfc_txt(path)

        if subject_id.startswith("normalStroke_") or is_normal_category(category):
            group_slug = "normal_stroke"
            groups[group_slug]["values"].append(values)
        else:
            groups["all_aphasia"]["values"].append(values)
            group_slug = safe_slug(category)
            if group_slug not in groups:
                groups[group_slug] = {
                    "label": SUBTYPE_DISPLAY_LABELS.get(category, category),
                    "source_category": category,
                    "values": [],
                }
            groups[group_slug]["values"].append(values)

        rows.append(
            {
                "subject_id": subject_id,
                "case_id": case_id,
                "excel_category": category,
                "primary_group": group_slug,
                "mean_sfc": float(np.nanmean(values)),
                "median_sfc": float(np.nanmedian(values)),
                "n_finite_roi": int(np.isfinite(values).sum()),
                "sfc_file": str(path),
            }
        )

    summary = pd.DataFrame(rows)
    groups = OrderedDict((key, value) for key, value in groups.items() if value["values"])
    return summary, groups


def save_average_txt(group_slug: str, values: np.ndarray) -> Path:
    AVERAGE_TXT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AVERAGE_TXT_DIR / f"{group_slug}_average_sfc.txt"
    with output_path.open("w", encoding="utf-8") as f:
        f.write("ROI_Index\tSFC_Value\n")
        for roi_idx, value in enumerate(values, start=1):
            f.write(f"{roi_idx}\t{value:.6f}\n")
    return output_path


def save_plot_final_style_surface(group_slug: str, label: str, sfc_values: np.ndarray) -> Path:
    sfc_first50 = sfc_values[:50]
    sfc_rest = sfc_values[50:]

    atlas_img = nib.load(str(ATLAS_PATH))
    atlas_data = atlas_img.get_fdata().astype(int)
    roi_vol = np.full(atlas_data.shape, np.nan, dtype=float)
    for lbl in range(1, 51):
        roi_vol[atlas_data == lbl] = sfc_first50[lbl - 1]
    roi_img = nib.Nifti1Image(roi_vol, atlas_img.affine, atlas_img.header)

    bg_img = datasets.load_mni152_template(resolution=1)
    bg_data = bg_img.get_fdata()
    bg_img_final = nib.Nifti1Image(bg_data, bg_img.affine, bg_img.header)

    fig_nilearn = plt.figure(figsize=(40, 8))
    plotting.plot_stat_map(
        roi_img,
        bg_img=bg_img_final,
        threshold=0.0,
        cmap=CMAP,
        vmax=GLOBAL_VMAX,
        vmin=GLOBAL_VMIN,
        display_mode="z",
        cut_coords=[-25, -20, -10, 0, 10, 20, 22],
        draw_cross=False,
        colorbar=False,
        figure=fig_nilearn,
        black_bg=False,
        annotate=False,
    )

    temp_dir = Path(tempfile.gettempdir())
    temp_nilearn = temp_dir / f"{group_slug}_plot_final_nilearn.png"
    temp_surface_base = temp_dir / f"{group_slug}_plot_final_surface"
    temp_surface = temp_surface_base.with_suffix(".png")

    fig_nilearn.savefig(temp_nilearn, bbox_inches="tight", dpi=300, pad_inches=0)
    plt.close(fig_nilearn)

    ds(
        layer1=sfc_rest,
        colorbar=CMAP,
        color_range=(GLOBAL_VMIN, GLOBAL_VMAX),
        fig_name=str(temp_surface_base),
        layout="row",
        outline=True,
        trim=True,
        tp="fslr",
        mesh="veryinflated",
        save_png_only=True,
        display=False,
        size=(6000, 1000),
    )

    img_surface = mpimg.imread(temp_surface)
    img_nilearn = mpimg.imread(temp_nilearn)
    img_surface_trimmed = trim_white_edges(img_surface)
    img_nilearn_trimmed = trim_white_edges(img_nilearn)

    fig_merged = plt.figure(figsize=(70, 25))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig_merged,
        width_ratios=[97, 3],
        hspace=0.1,
        wspace=0.05,
    )

    ax1 = fig_merged.add_subplot(gs[0, 0])
    ax1.imshow(img_surface_trimmed)
    ax1.axis("off")

    ax2 = fig_merged.add_subplot(gs[1, 0])
    ax2.imshow(img_nilearn_trimmed)
    ax2.axis("off")

    ax_cbar = fig_merged.add_subplot(gs[:, 1])
    norm = mpl.colors.Normalize(vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX)
    cbar = ColorbarBase(ax_cbar, cmap=CMAP, norm=norm, orientation="vertical")
    ax_cbar.set_yticks([GLOBAL_VMIN, GLOBAL_VMAX])
    cbar.set_label("SFC Values", rotation=270, labelpad=30, fontsize=70, fontname="Arial")
    ax_cbar.tick_params(labelsize=50, direction="in", length=8, width=8)
    for tick_label in ax_cbar.get_yticklabels():
        tick_label.set_fontname("Arial")

    fig_merged.suptitle(f"{label} (n={GROUP_COUNTS[group_slug]})", fontsize=70, fontname="Arial", y=0.98)
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{group_slug}_surface_t288.pdf"
    fig_merged.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig_merged)

    temp_surface.unlink(missing_ok=True)
    temp_nilearn.unlink(missing_ok=True)
    return output_path


GROUP_COUNTS: dict[str, int] = {}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, groups = load_group_values()
    summary.to_csv(OUTPUT_DIR / "subject_category_mapping.csv", index=False, encoding="utf-8-sig")

    global GROUP_COUNTS
    GROUP_COUNTS = {slug: len(info["values"]) for slug, info in groups.items()}

    mean_table = pd.DataFrame({"roi_index": np.arange(1, ROI_NUMBER + 1)})
    count_rows = []
    output_rows = []
    for group_slug, info in groups.items():
        matrix = np.vstack(info["values"])
        mean_values = np.nanmean(matrix, axis=0)
        mean_table[f"{group_slug}_mean_sfc"] = mean_values
        mean_table[f"{group_slug}_n"] = np.sum(np.isfinite(matrix), axis=0)

        average_txt = save_average_txt(group_slug, mean_values)
        surface_pdf = save_plot_final_style_surface(group_slug, str(info["label"]), mean_values)

        count_rows.append(
            {
                "group": group_slug,
                "label": info["label"],
                "source_category": info["source_category"],
                "n_subjects": GROUP_COUNTS[group_slug],
            }
        )
        output_rows.append(
            {
                "group": group_slug,
                "label": info["label"],
                "n_subjects": GROUP_COUNTS[group_slug],
                "average_txt": str(average_txt),
                "surface_pdf": str(surface_pdf),
            }
        )
        print(f"Saved {group_slug}: n={GROUP_COUNTS[group_slug]} -> {surface_pdf}")

    mean_table.to_csv(OUTPUT_DIR / "category_mean_sfc_values.csv", index=False)
    pd.DataFrame(count_rows).to_csv(OUTPUT_DIR / "category_counts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(output_rows).to_csv(OUTPUT_DIR / "surface_outputs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"vmin": GLOBAL_VMIN, "vmax": GLOBAL_VMAX, "cmap": CMAP}]).to_csv(
        OUTPUT_DIR / "surface_colorbar.csv", index=False
    )

    print("\nCategory counts:")
    print(pd.DataFrame(count_rows).to_string(index=False))
    print(f"\nSaved plot_final-style surfaces to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
