from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from scipy import stats


DATA_ROOT = Path(
    r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main"
    r"\aphasia_connectioncsv_timeseries_paired_20260507"
    r"\aphasia_connectioncsv_timeseries_paired_20260507"
)
SFC_RESULT_DIR = DATA_ROOT / "sfc_pca_450_t288"
SFC_TXT_DIR = SFC_RESULT_DIR / "sfc_txt_files"
OUTPUT_DIR = SFC_RESULT_DIR / "group_plots"
ATLAS_LABEL_FILE = Path(
    r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
)

ROI_NUMBER = 450
NETWORK_ORDER = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis"]
GROUP_ORDER = ["normal_stroke", "aphasia"]
GROUP_LABELS = {"normal_stroke": "Normal stroke", "aphasia": "Aphasia"}
GROUP_COLORS = {"normal_stroke": "#4C78A8", "aphasia": "#E45756"}


def subject_from_sfc_file(path: Path) -> str:
    prefix = "sub-"
    suffix = "_sfc_values_pca.txt"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected SFC filename: {name}")
    return name[len(prefix) : -len(suffix)]


def infer_group(subject_id: str) -> str:
    if subject_id.startswith("aphasia_"):
        return "aphasia"
    if subject_id.startswith("normalStroke_"):
        return "normal_stroke"
    raise ValueError(f"Cannot infer group from subject id: {subject_id}")


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


def parse_cortical_network_map(label_file: Path) -> dict[str, np.ndarray]:
    """Return network -> 0-based indices in the 450-length SFC vector."""
    raw_lines = [line.strip() for line in label_file.read_text().splitlines() if line.strip()]
    network_to_indices: dict[str, list[int]] = {network: [] for network in NETWORK_ORDER}

    idx = 0
    while idx + 1 < len(raw_lines):
        roi_name = raw_lines[idx]
        data_line = raw_lines[idx + 1]
        idx += 2

        parts = data_line.split()
        if not parts:
            continue
        try:
            label = int(parts[0])
        except ValueError:
            continue

        if label <= 50 or not roi_name.startswith("7Networks"):
            continue
        name_parts = roi_name.split("_")
        if len(name_parts) < 3:
            continue
        network = name_parts[2]
        if network in network_to_indices:
            network_to_indices[network].append(label - 1)

    missing = [network for network, indices in network_to_indices.items() if not indices]
    if missing:
        raise RuntimeError(f"No ROI indices parsed for networks: {missing}")

    return {network: np.asarray(indices, dtype=int) for network, indices in network_to_indices.items()}


def build_network_subject_table(network_map: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for path in sorted(SFC_TXT_DIR.glob("*_sfc_values_pca.txt")):
        subject_id = subject_from_sfc_file(path)
        group = infer_group(subject_id)
        values = load_sfc_txt(path)

        row = {
            "subject_id": subject_id,
            "group": group,
            "group_label": GROUP_LABELS[group],
            "sfc_file": str(path),
        }
        for network in NETWORK_ORDER:
            network_values = values[network_map[network]]
            row[f"{network}_median"] = float(np.nanmedian(network_values))
            row[f"{network}_mean"] = float(np.nanmean(network_values))
            row[f"{network}_n_roi"] = int(np.isfinite(network_values).sum())
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No SFC txt files found in {SFC_TXT_DIR}")
    return df


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def plot_network_boxes(df: pd.DataFrame, value_suffix: str = "median") -> pd.DataFrame:
    plt.rcParams["font.family"] = "Arial"
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 4, figsize=(11, 6.2))
    axes = axes.flatten()
    stats_rows = []

    for network_idx, network in enumerate(NETWORK_ORDER):
        ax = axes[network_idx]
        y_col = f"{network}_{value_suffix}"
        data_by_group = []

        for pos, group in enumerate(GROUP_ORDER):
            data = df.loc[df["group"] == group, y_col].dropna().to_numpy(dtype=float)
            data_by_group.append(data)

            ax.boxplot(
                data,
                positions=[pos],
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                showcaps=False,
                medianprops={"color": "black", "linewidth": 1.4},
                boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.3},
                whiskerprops={"color": "black", "linewidth": 1.1},
                capprops={"linewidth": 0},
            )
            jitter_x = rng.normal(pos, 0.06, size=len(data))
            ax.scatter(
                jitter_x,
                data,
                s=24,
                color=GROUP_COLORS[group],
                alpha=0.82,
                edgecolors="none",
                zorder=3,
            )

        normal_data, aphasia_data = data_by_group
        if len(normal_data) > 0 and len(aphasia_data) > 0:
            stat, p_value = stats.mannwhitneyu(normal_data, aphasia_data, alternative="two-sided")
            stars = significance_stars(float(p_value))
            y_min = float(np.nanmin(np.concatenate(data_by_group)))
            y_max = float(np.nanmax(np.concatenate(data_by_group)))
            y_range = max(y_max - y_min, 1e-6)
            y = y_max + y_range * 0.08
            ax.plot([0, 1], [y, y], color="black", lw=1.1, clip_on=False)
            ax.text(0.5, y + y_range * 0.02, stars, ha="center", va="bottom", fontsize=10)
            ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * 0.28)

            stats_rows.append(
                {
                    "network": network,
                    "normal_stroke_n": len(normal_data),
                    "aphasia_n": len(aphasia_data),
                    "normal_stroke_median": float(np.nanmedian(normal_data)),
                    "aphasia_median": float(np.nanmedian(aphasia_data)),
                    "mannwhitney_u": float(stat),
                    "p_value": float(p_value),
                    "significance": stars,
                }
            )

        ax.set_title(network, fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal\nstroke", "Aphasia"], fontsize=9)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

    legend_ax = axes[-1]
    legend_ax.axis("off")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=GROUP_COLORS[group],
            markersize=9,
            label=GROUP_LABELS[group],
            linestyle="None",
        )
        for group in GROUP_ORDER
    ]
    legend_ax.legend(handles=handles, loc="center", frameon=True, title="Group")

    fig.suptitle("Network-Level SFC by Group", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"network_level_sfc_{value_suffix}_boxplots.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"network_level_sfc_{value_suffix}_boxplots.pdf", bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(stats_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    network_map = parse_cortical_network_map(ATLAS_LABEL_FILE)
    network_counts = pd.DataFrame(
        [{"network": network, "n_roi": len(indices)} for network, indices in network_map.items()]
    )
    network_counts.to_csv(OUTPUT_DIR / "network_roi_counts.csv", index=False)

    network_df = build_network_subject_table(network_map)
    network_df.to_csv(OUTPUT_DIR / "subject_network_sfc_summary.csv", index=False)
    stats_df = plot_network_boxes(network_df, value_suffix="median")
    stats_df.to_csv(OUTPUT_DIR / "network_level_group_mannwhitney.csv", index=False)

    print(f"Loaded {len(network_df)} subjects")
    print(network_df.groupby("group")["subject_id"].count().to_string())
    print(f"Network boxplots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
