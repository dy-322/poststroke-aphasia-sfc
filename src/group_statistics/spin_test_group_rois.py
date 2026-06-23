import os
import warnings
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from roi_group_comparison_cohens_d import (
    CORTICAL_OFFSET,
    N_CORTICAL_ROIS,
    NETWORK_LABEL_FILE,
    load_sfc_data_by_category,
    load_subject_categories,
    parse_network_label_file,
    statistical_test_category_vs_8,
)

warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
CSV_FILE_PATH = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\aphasia_plots\participants_aphasia_new.txt"
SFC_DIR = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\sfc_txt_files"
COORD_FILE = r"D:\download\Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
SAVE_DIR = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\source\results\spin_test"
TARGET_CATEGORY = "9"
N_PERMUTATIONS = 1000

NETWORK_ORDER = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis"]
NETWORK_COLORS = ["#3287bd", "#65c2a5", "#abdca4", "#ffffbf", "#fdae61", "#f56d43", "#d53e4f"]


# ==================== 工具函数 ====================
def load_cortical_coordinates(coord_file_path: str) -> np.ndarray:
    """读取皮层ROI的中心坐标。"""
    if not os.path.exists(coord_file_path):
        raise FileNotFoundError(f"未找到坐标文件: {coord_file_path}")

    coord_df = pd.read_csv(coord_file_path)
    candidate_cols = [
        ["R", "A", "S"],
        ["X", "Y", "Z"],
        coord_df.columns[:3].tolist(),
    ]

    coord_cols = None
    for cols in candidate_cols:
        if all(col in coord_df.columns for col in cols):
            coord_cols = cols
            break

    if coord_cols is None:
        raise ValueError(f"无法在坐标文件中识别坐标列: {coord_df.columns.tolist()}")

    coords = coord_df[coord_cols].to_numpy(dtype=float)
    if coords.shape[0] < N_CORTICAL_ROIS:
        raise ValueError(f"坐标文件行数不足: 期望≥{N_CORTICAL_ROIS}, 实际={coords.shape[0]}")

    coords = coords[:N_CORTICAL_ROIS]
    if np.isnan(coords).any():
        raise ValueError("坐标文件存在NaN，请检查数据质量")
    return coords


def build_network_label_vector(network_mapping: Dict[str, List[int]]) -> np.ndarray:
    """将网络映射转换为长度为400的标签数组。"""
    labels = np.array(["Unknown"] * N_CORTICAL_ROIS, dtype=object)
    for network, indices in network_mapping.items():
        for idx in indices:
            if 0 <= idx < N_CORTICAL_ROIS:
                labels[idx] = network
    return labels


def prepare_significant_roi_sets(significant_regions: Dict[str, Dict[str, List[int]]]) -> Dict[str, np.ndarray]:
    """将显著脑区转换为皮层ROI索引集合。"""
    roi_sets = {}
    for aphasia_type, data in significant_regions.items():
        global_indices = data.get("significant_rois", [])
        cortical_indices = [
            roi_idx - CORTICAL_OFFSET
            for roi_idx in global_indices
            if roi_idx >= CORTICAL_OFFSET and (roi_idx - CORTICAL_OFFSET) < N_CORTICAL_ROIS
        ]
        if cortical_indices:
            roi_sets[aphasia_type] = np.array(sorted(set(cortical_indices)), dtype=int)
    return roi_sets


def rotate_coordinates(coords: np.ndarray, seed: int) -> np.ndarray:
    """随机旋转皮层坐标。"""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, size=3)
    rotation = Rotation.from_euler("xyz", angles)
    return rotation.apply(coords)


def map_to_nearest_roi(rotated_coords: np.ndarray, original_coords: np.ndarray, network_labels: np.ndarray) -> np.ndarray:
    """将旋转后的坐标映射到最近的原始ROI并继承其网络标签。"""
    distances = cdist(rotated_coords, original_coords)
    nearest_indices = np.argmin(distances, axis=1)
    return network_labels[nearest_indices]


def calculate_enrichment_ratio(associated_rois: np.ndarray, network_labels: np.ndarray, network: str) -> float:
    """计算指定网络的富集比例。"""
    if associated_rois.size == 0:
        return 0.0
    valid = network_labels[associated_rois]
    mask = valid == network
    return float(np.sum(mask) / len(valid))


def calculate_subcortical_ratio(cortical_rois: np.ndarray, subcortical_rois: np.ndarray) -> float:
    """计算皮层下脑区的比例。"""
    total_rois = len(cortical_rois) + len(subcortical_rois)
    if total_rois == 0:
        return 0.0
    return float(len(subcortical_rois) / total_rois)


def run_spin_test_for_category(
    category_label: str,
    cortical_rois: np.ndarray,
    subcortical_rois: np.ndarray,
    coords: np.ndarray,
    network_labels: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
) -> Dict[str, Dict[str, float]]:
    """对某一类别的显著脑区执行旋转测试（包含皮层和皮层下）。"""
    print(f"\n--- 类别{category_label}的旋转测试 ---")
    print(f"关联脑区数量（皮层）：{len(cortical_rois)}")
    print(f"关联脑区数量（皮层下）：{len(subcortical_rois)}")

    # 计算实际的富集比例
    actual_ers = {}
    for network in NETWORK_ORDER:
        if network == "Subcort":
            # 皮层下比例 = 皮层下数量 / (皮层数量 + 皮层下数量)
            actual_ers[network] = calculate_subcortical_ratio(cortical_rois, subcortical_rois)
        else:
            actual_ers[network] = calculate_enrichment_ratio(cortical_rois, network_labels, network)

    # 生成零分布（只对皮层进行旋转测试）
    null_distributions = {network: [] for network in NETWORK_ORDER}
    total_rois = len(cortical_rois) + len(subcortical_rois)
    n_subcort_actual = len(subcortical_rois)
    
    for perm in tqdm(range(n_permutations), desc=f"类别{category_label}旋转", leave=False):
        rotated = rotate_coordinates(coords, seed=perm)
        mapped_labels = map_to_nearest_roi(rotated, coords, network_labels)
        
        for network in NETWORK_ORDER:
            if network == "Subcort":
                # 对于皮层下，零分布是通过随机分配相同数量的ROI来模拟
                # 在每次排列中，使用固定的皮层下比例（基于实际数量）
                if total_rois > 0:
                    null_er = float(n_subcort_actual / total_rois)
                else:
                    null_er = 0.0
            else:
                null_er = calculate_enrichment_ratio(cortical_rois, mapped_labels, network)
            null_distributions[network].append(null_er)

    # 计算p值
    p_values = {}
    for network in NETWORK_ORDER:
        null_array = np.array(null_distributions[network])
        actual = actual_ers[network]
        p_values[network] = float(np.mean(null_array >= actual))

    return {
        "actual_ERs": actual_ers,
        "null_distributions": null_distributions,
        "p_values": p_values,
    }


def create_full_violinplot(ax, data, positions, color, width, alpha=1.0):
    """绘制全小提琴图"""
    violin_parts = ax.violinplot(data, positions=positions, vert=True,
                                 showmeans=False, showextrema=False, widths=width)
    for j, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(color)
        pc.set_edgecolor(color)  # 外轮廓用网络配色
        pc.set_alpha(alpha)  # 不透明
        pc.set_linewidth(1.5)
    return violin_parts


def plot_spin_test_results(results: Dict[str, Dict], title: str, output_path: str) -> None:
    """绘制旋转测试结果（小提琴图）。"""
    actual_ers = results["actual_ERs"]
    null_distributions = results["null_distributions"]
    p_values = results["p_values"]

    # 设置字体为 Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    fig, ax = plt.subplots(figsize=(9, 6))

    plot_data = [null_distributions[network] for network in NETWORK_ORDER]
    positions = list(range(len(NETWORK_ORDER)))
    actual_values = [actual_ers[network] for network in NETWORK_ORDER]

    # 使用自定义的小提琴图绘制函数
    for idx, pos in enumerate(positions):
        color = NETWORK_COLORS[idx % len(NETWORK_COLORS)]
        create_full_violinplot(ax, [plot_data[idx]], [pos], color, width=0.7, alpha=1.0)

    for idx, pos in enumerate(positions):
        actual_val = actual_values[idx]
        ax.plot([pos - 0.35, pos + 0.35], [actual_val, actual_val], color="gray", linewidth=2.5)

        p_val = p_values[NETWORK_ORDER[idx]]
        # 只标注显著的 p 值，且用黑色
        if p_val < 0.05:
            stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "*")
            y_pos = actual_val + 0.015
            ax.text(pos, y_pos, f"{stars}\np={p_val:.3f}", ha="center", va="bottom", 
                   fontsize=12, color="black", fontweight="bold", fontfamily="Arial")

    ax.set_xticks(positions)
    ax.set_xticklabels(NETWORK_ORDER, fontsize=12, fontfamily="Arial")
    ax.set_ylabel("Enrichment Ratio", fontsize=14, fontfamily="Arial")
    ax.set_xlabel("Functional Network", fontsize=14, fontfamily="Arial")
    ax.set_title(f"Spin Test: {title}", fontsize=16, fontweight="bold", pad=15, fontfamily="Arial")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # 设置所有文本的字体为 Arial
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontfamily("Arial")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 保存为 PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=400, bbox_inches="tight", format='pdf')
    plt.close(fig)
    print(f"结果图保存至: {pdf_path}")


def save_spin_test_stats(results: Dict[str, Dict], output_path: str) -> None:
    """保存旋转测试统计结果到CSV。"""
    rows = []
    for network in NETWORK_ORDER:
        null_vals = results["null_distributions"][network]
        rows.append(
            {
                "Network": network,
                "Actual_ER": results["actual_ERs"][network],
                "P_Value": results["p_values"][network],
                "Null_Mean": float(np.mean(null_vals)),
                "Null_Std": float(np.std(null_vals)),
            }
        )
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"统计结果保存至: {output_path}")


# ==================== 主流程 ====================
def main() -> None:
    print("=" * 80)
    print("Sig Brain Spin Test：显著脑区网络富集分析")
    print("=" * 80)

    # 1. 加载受试者类别
    subject_categories = load_subject_categories(CSV_FILE_PATH)
    if not subject_categories:
        print("未能加载受试者类别，程序结束。")
        return

    # 2. 加载SFC数据
    category_data = load_sfc_data_by_category(SFC_DIR, subject_categories)
    if not category_data:
        print("未能加载SFC数据，程序结束。")
        return

    # 3. 计算显著脑区
    significant_regions, _ = statistical_test_category_vs_8(category_data, target_category=TARGET_CATEGORY)
    if not significant_regions:
        print("没有显著脑区结果，程序结束。")
        return

    significant_roi_sets = prepare_significant_roi_sets(significant_regions)
    if not significant_roi_sets:
        print("显著脑区均位于皮层下，无法执行旋转测试。")
        return

    # 4. 加载网络信息与坐标
    network_mapping, _ = parse_network_label_file(NETWORK_LABEL_FILE)
    if not network_mapping:
        print("未能解析七网络映射，程序结束。")
        return

    coords = load_cortical_coordinates(COORD_FILE)
    network_labels = build_network_label_vector(network_mapping)

    # 输出目录
    fig_dir = os.path.join(SAVE_DIR, "figures")
    stats_dir = os.path.join(SAVE_DIR, "stats")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    # 5. 逐类别执行旋转测试
    summary = {}
    for aphasia_type, cortical_indices in significant_roi_sets.items():
        try:
            results = run_spin_test_for_category(aphasia_type, cortical_indices, coords, network_labels, N_PERMUTATIONS)
        except Exception as exc:
            print(f"类别{aphasia_type}旋转测试失败: {exc}")
            continue

        title = f"Category {aphasia_type} vs {TARGET_CATEGORY}"
        fig_path = os.path.join(fig_dir, f"spin_test_category_{aphasia_type}_vs_{TARGET_CATEGORY}.pdf")
        stats_path = os.path.join(stats_dir, f"spin_test_category_{aphasia_type}_vs_{TARGET_CATEGORY}.csv")

        plot_spin_test_results(results, title, fig_path)
        save_spin_test_stats(results, stats_path)
        summary[aphasia_type] = results["p_values"]

    if summary:
        print("\n=== P值摘要 ===")
        for aphasia_type, pvals in summary.items():
            formatted = ", ".join(f"{net}:{pvals[net]:.3f}" for net in NETWORK_ORDER)
            print(f"类别{aphasia_type}: {formatted}")
    else:
        print("没有成功的旋转测试结果。")


if __name__ == "__main__":
    main()
