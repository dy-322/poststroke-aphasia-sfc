# -*- coding: utf-8 -*-
"""
Spin Test（旋转测试）分析脚本
用于验证关联脑区在功能网络中的富集程度是否显著

原理：
1. 通过随机旋转脑区空间位置，生成空分布（null distribution）
2. 将实际富集指标与空分布对比，计算P值
3. 验证富集效应是否为真实效应（非随机偶然）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 配置参数 ====================
# CSV文件路径
BASE_PATH = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\images"
CSV_FILES = [
    ("roi_sfc_naming_subscore_correlations.csv", "Naming Subscore"),
    ("roi_sfc_comprehension_subscore_correlations.csv", "Comprehension Subscore"),
    ("roi_sfc_wab_r_aq_correlations.csv", "WAB-R AQ"),
    ("roi_sfc_repetition_subscore_correlations.csv", "Repetition Subscore")
]

# 坐标文件路径
COORD_FILE = r"D:\download\Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"

# 网络定义和顺序
NETWORK_ORDER = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis"]
NETWORK_COLORS = ["#3287bd", "#65c2a5", "#abdca4", "#ffffbf", "#fdae61", "#f56d43", "#d53e4f"]
NETWORK_NAME_MAP = {
    'Cont': 'Cont',
    'Default': 'Default',
    'DorsAttn': 'DorsAttn',
    'Limbic': 'Limbic',
    'SalVentAttn': 'SalVentAttn',
    'SomMot': 'SomMot',
    'Vis': 'Vis'
}

# Spin Test参数
N_PERMUTATIONS = 500  # 旋转次数
SUBCORTICAL_COUNT = 50  # 皮层下ROI数量
CORRELATION_THRESHOLD = None  # 相关性阈值，None表示使用所有有网络信息的ROI


# ==================== 2. 辅助函数 ====================
def cartesian_to_spherical(coords):
    """将笛卡尔坐标转换为球面坐标"""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1, 1))  # 极角（colatitude）
    phi = np.arctan2(y, x)  # 方位角（azimuth）
    return np.column_stack([r, theta, phi])


def spherical_to_cartesian(spherical_coords):
    """将球面坐标转换为笛卡尔坐标"""
    r, theta, phi = spherical_coords[:, 0], spherical_coords[:, 1], spherical_coords[:, 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack([x, y, z])


def rotate_coordinates(coords, seed=None):
    """
    随机旋转三维坐标（保持球面拓扑结构）
    
    参数:
        coords: (N, 3) 数组，笛卡尔坐标
        seed: 随机种子
    
    返回:
        rotated_coords: (N, 3) 数组，旋转后的坐标
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成随机旋转矩阵（使用欧拉角）
    angles = np.random.uniform(0, 2 * np.pi, size=3)
    rotation = Rotation.from_euler('xyz', angles)
    
    # 应用旋转
    rotated_coords = rotation.apply(coords)
    
    return rotated_coords


def map_to_nearest_roi(rotated_coords, original_coords, original_network_labels):
    """
    将旋转后的坐标映射到最近的原始ROI，获取其网络标签
    
    参数:
        rotated_coords: (N, 3) 旋转后的坐标
        original_coords: (M, 3) 原始ROI坐标
        original_network_labels: (M,) 原始ROI的网络标签
    
    返回:
        mapped_labels: (N,) 映射后的网络标签
    """
    # 计算距离矩阵
    distances = cdist(rotated_coords, original_coords)
    
    # 找到每个旋转后ROI的最近原始ROI
    nearest_indices = np.argmin(distances, axis=1)
    
    # 获取对应的网络标签
    mapped_labels = original_network_labels[nearest_indices]
    
    return mapped_labels


def calculate_enrichment_ratio(associated_rois, network_labels, network_name):
    """
    计算富集指标（Enrichment Ratio, ER）
    
    参数:
        associated_rois: 关联脑区索引数组
        network_labels: 所有ROI的网络标签数组
        network_name: 目标网络名称
    
    返回:
        ER: 富集比例
    """
    if len(associated_rois) == 0:
        return 0.0
    
    # 统计关联脑区中属于目标网络的数量
    network_mask = network_labels[associated_rois] == network_name
    network_count = np.sum(network_mask)
    
    # 计算比例
    ER = network_count / len(associated_rois)
    
    return ER


def load_coordinates_and_networks(csv_path, coord_file_path):
    """
    加载坐标和网络信息
    
    参数:
        csv_path: CSV文件路径
        coord_file_path: 坐标文件路径
    
    返回:
        coords: (N, 3) 皮层ROI坐标数组（只包含有网络信息的ROI）
        network_labels: (N,) 网络标签数组
        associated_rois: 关联脑区索引（在皮层ROI中的索引）
        roi_index_mapping: 原始ROI索引到皮层ROI索引的映射
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 只保留有网络信息的ROI（即皮层ROI，索引从51开始）
    df_cortical = df[df['Network'].notna() & (df['Network'] != '')].copy()
    
    # 重置索引，使其从0开始
    df_cortical.reset_index(drop=True, inplace=True)
    
    print(f"  CSV中皮层ROI数量: {len(df_cortical)}")
    print(f"  ROI_Index范围: {df_cortical['ROI_Index'].min()} - {df_cortical['ROI_Index'].max()}")
    
    # 读取坐标文件
    coord_df = pd.read_csv(coord_file_path)
    print(f"  坐标文件总行数: {len(coord_df)}")
    print(f"  坐标文件列名: {coord_df.columns.tolist()}")
    
    # 识别坐标列
    coord_cols = None
    if 'R' in coord_df.columns and 'A' in coord_df.columns and 'S' in coord_df.columns:
        coord_cols = ['R', 'A', 'S']
        print(f"  使用RAS坐标系统")
    elif 'X' in coord_df.columns and 'Y' in coord_df.columns and 'Z' in coord_df.columns:
        coord_cols = ['X', 'Y', 'Z']
        print(f"  使用XYZ坐标系统")
    else:
        # 尝试使用前3列作为坐标
        coord_cols = coord_df.columns[:3].tolist()
        print(f"  使用前3列作为坐标: {coord_cols}")
    
    # 提取皮层ROI的坐标
    # 注意：CSV中ROI_Index从1开始，前50个是皮层下（索引1-50），皮层ROI从51开始（索引51-450）
    # 坐标文件如果只有400行，说明只包含400个皮层ROI
    # 因此，CSV中ROI_Index=51对应坐标文件中的行索引=0（0-based）
    # 映射关系：coord_row_idx = roi_idx - 51
    
    # 确保坐标列为数值类型
    if coord_cols:
        for col in coord_cols:
            if col in coord_df.columns:
                coord_df[col] = pd.to_numeric(coord_df[col], errors='coerce')
    
    cortical_coords = []
    cortical_networks = []
    valid_indices = []
    
    for idx, row in df_cortical.iterrows():
        roi_idx = row['ROI_Index']  # CSV中的ROI_Index（从1开始，皮层ROI从51开始）
        network = row['Network']
        
        # CSV中ROI_Index=51-450对应坐标文件的行0-399（假设坐标文件只有400个皮层ROI）
        # 如果坐标文件有450行（包含前50个皮层下），则coord_row_idx = roi_idx - 1
        # 如果坐标文件只有400行（只包含皮层ROI），则coord_row_idx = roi_idx - 51
        
        # 先尝试：假设坐标文件只有400行皮层ROI（不包含前50个皮层下）
        if len(coord_df) == 400:
            coord_row_idx = roi_idx - 51  # ROI_Index 51 -> 行0, ROI_Index 52 -> 行1, ...
        else:
            # 如果坐标文件包含所有450行，则直接使用roi_idx - 1
            coord_row_idx = roi_idx - 1
        
        # 检查索引是否在有效范围内
        if coord_row_idx >= 0 and coord_row_idx < len(coord_df):
            try:
                # 提取坐标
                if coord_cols:
                    coord_values = coord_df.iloc[coord_row_idx][coord_cols]
                else:
                    coord_values = coord_df.iloc[coord_row_idx, :3]
                
                # 转换为numpy数组并确保为数值类型
                coord_list = []
                for val in coord_values:
                    try:
                        num_val = float(val)
                        coord_list.append(num_val)
                    except (ValueError, TypeError):
                        coord_list.append(np.nan)
                
                coord = np.array(coord_list)
                
                # 检查坐标是否有效（非NaN且为数值类型）
                if len(coord) == 3 and np.all(np.isfinite(coord)):
                    cortical_coords.append(coord)
                    cortical_networks.append(network)
                    valid_indices.append(idx)
            except Exception as e:
                print(f"  警告: 无法提取ROI_Index={roi_idx}的坐标: {e}")
                continue
    
    if len(cortical_coords) == 0:
        raise ValueError("未找到有效的皮层ROI坐标")
    
    cortical_coords = np.array(cortical_coords)
    cortical_networks = np.array(cortical_networks)
    
    print(f"  成功加载 {len(cortical_coords)} 个皮层ROI的坐标")
    
    # 确定关联脑区集：使用P_Value_FDR_Corrected < 0.05的ROI
    # valid_indices对应df_cortical中有有效坐标的ROI索引
    df_valid = df_cortical.iloc[valid_indices].copy()
    
    # 筛选P_Value_FDR_Corrected < 0.05的ROI
    if 'P_Value_FDR_Corrected' in df_valid.columns:
        # 确保P值为数值类型
        df_valid['P_Value_FDR_Corrected'] = pd.to_numeric(df_valid['P_Value_FDR_Corrected'], errors='coerce')
        
        # 筛选显著ROI
        significant_mask = df_valid['P_Value_FDR_Corrected'] < 0.05
        significant_indices = np.where(significant_mask)[0]  # 在valid_indices中的索引
        
        associated_rois = significant_indices
        print(f"  使用P_Value_FDR_Corrected < 0.05筛选，关联脑区数量: {len(associated_rois)}")
        
        if len(associated_rois) == 0:
            print(f"  警告: 没有找到P_Value_FDR_Corrected < 0.05的ROI")
    else:
        raise ValueError("CSV文件中缺少P_Value_FDR_Corrected列")
    
    return cortical_coords, cortical_networks, associated_rois, {}


def run_spin_test(csv_path, coord_file_path, n_permutations=N_PERMUTATIONS):
    """
    执行Spin Test
    
    参数:
        csv_path: CSV文件路径
        coord_file_path: 坐标文件路径
        n_permutations: 旋转次数
    
    返回:
        results: 字典，包含实际ER、空分布、P值等信息
    """
    print(f"\n正在加载数据: {csv_path}")
    
    # 加载坐标和网络信息
    coords, network_labels, associated_rois, _ = load_coordinates_and_networks(csv_path, coord_file_path)
    
    print(f"  皮层ROI数量: {len(coords)}")
    print(f"  关联脑区数量: {len(associated_rois)}")
    print(f"  网络分布: {dict(zip(*np.unique(network_labels, return_counts=True)))}")
    
    # 计算实际ER（针对每个网络）
    actual_ERs = {}
    for network in NETWORK_ORDER:
        ER = calculate_enrichment_ratio(associated_rois, network_labels, network)
        actual_ERs[network] = ER
        if ER > 0:
            print(f"  实际 {network} ER: {ER:.4f}")
    
    # 生成空分布
    print(f"\n正在执行 {n_permutations} 次旋转...")
    null_distributions = {network: [] for network in NETWORK_ORDER}
    
    for perm in tqdm(range(n_permutations), desc="旋转进度"):
        # 随机旋转坐标
        rotated_coords = rotate_coordinates(coords, seed=perm)
        
        # 将旋转后的坐标映射回最近的原始ROI
        mapped_labels = map_to_nearest_roi(rotated_coords, coords, network_labels)
        
        # 计算随机ER
        for network in NETWORK_ORDER:
            ER_random = calculate_enrichment_ratio(associated_rois, mapped_labels, network)
            null_distributions[network].append(ER_random)
    
    # 计算P值
    p_values = {}
    for network in NETWORK_ORDER:
        null_ERs = np.array(null_distributions[network])
        actual_ER = actual_ERs[network]
        
        # P值：空分布中 >= 实际ER的比例
        p_value = np.mean(null_ERs >= actual_ER)
        p_values[network] = p_value
    
    results = {
        'actual_ERs': actual_ERs,
        'null_distributions': null_distributions,
        'p_values': p_values
    }
    
    return results


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


def plot_spin_test_results(results, title, output_path):
    """
    绘制Spin Test结果（小提琴图）
    
    参数:
        results: run_spin_test返回的结果字典
        title: 图表标题
        output_path: 输出文件路径
    """
    actual_ERs = results['actual_ERs']
    null_distributions = results['null_distributions']
    p_values = results['p_values']
    
    # 设置字体为 Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # 准备数据用于绘图
    plot_data = []
    positions = []
    actual_values = []
    
    for i, network in enumerate(NETWORK_ORDER):
        if network not in null_distributions or len(null_distributions[network]) == 0:
            continue
        
        null_ERs = null_distributions[network]
        actual_ER = actual_ERs[network]
        
        plot_data.append(null_ERs)
        positions.append(i)
        actual_values.append(actual_ER)
    
    # 使用自定义的小提琴图绘制函数
    for idx, pos in enumerate(positions):
        color = NETWORK_COLORS[idx % len(NETWORK_COLORS)]
        create_full_violinplot(ax, [plot_data[idx]], [pos], color, width=0.7, alpha=1.0)
    
    # 绘制实际ER值（水平横线）
    for idx, pos in enumerate(positions):
        actual_val = actual_values[idx]
        network_name = NETWORK_ORDER[pos]
        ax.plot([pos - 0.35, pos + 0.35], [actual_val, actual_val], 
                color="gray", linewidth=2.5)
        
        p_val = p_values[network_name]
        # 只标注显著的 p 值，且用黑色
        if p_val < 0.05:
            stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "*")
            y_pos = actual_val + 0.015
            ax.text(pos, y_pos, f"{stars}\np={p_val:.3f}", ha="center", va="bottom", 
                   fontsize=12, color="black", fontweight="bold", fontfamily="Arial")
    
    # 设置x轴
    ax.set_xticks(positions)
    ax.set_xticklabels([NETWORK_ORDER[i] for i in positions], fontsize=12, fontfamily="Arial")
    ax.set_xlabel('Functional Network', fontsize=14, fontfamily="Arial")
    
    # 设置y轴
    ax.set_ylabel('Enrichment Ratio', fontsize=14, fontfamily="Arial")
    ax.set_ylim([0, max([max(d) if len(d) > 0 else 0 for d in plot_data] + actual_values) * 1.2])
    
    # 添加网格
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 设置标题
    ax.set_title(f'Spin Test: {title}', fontsize=16, fontweight='bold', pad=15, fontfamily="Arial")
    
    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 设置所有文本的字体为 Arial
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontfamily("Arial")
    
    plt.tight_layout()
    
    # 保存为 PDF
    pdf_path = output_path.replace('.png', '.pdf')
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    plt.savefig(pdf_path, dpi=400, bbox_inches='tight', format='pdf')
    print(f"结果图保存至: {pdf_path}")
    plt.close(fig)


# ==================== 3. 主程序 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("Spin Test（旋转测试）分析")
    print("=" * 60)
    
    # 检查坐标文件是否存在
    if not os.path.exists(COORD_FILE):
        print(f"错误：坐标文件不存在: {COORD_FILE}")
        return
    
    # 对每个CSV文件执行Spin Test
    for csv_file, title in CSV_FILES:
        csv_path = os.path.join(BASE_PATH, csv_file)
        
        if not os.path.exists(csv_path):
            print(f"警告：CSV文件不存在: {csv_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理文件: {csv_file}")
        print(f"标题: {title}")
        print(f"{'='*60}")
        
        try:
            # 执行Spin Test
            results = run_spin_test(csv_path, COORD_FILE, n_permutations=N_PERMUTATIONS)
            
            # 打印结果摘要
            print("\n结果摘要:")
            print("-" * 60)
            for network in NETWORK_ORDER:
                actual_ER = results['actual_ERs'][network]
                p_val = results['p_values'][network]
                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
                print(f"{network:12s}: ER = {actual_ER:.4f}, P = {p_val:.4f} {sig}")
            
            # 绘制结果
            output_filename = csv_file.replace('.csv', '_spin_test.png')
            output_path = os.path.join(BASE_PATH, output_filename)
            plot_spin_test_results(results, title, output_path)
            
            # 保存统计结果到CSV
            stats_df = pd.DataFrame({
                'Network': NETWORK_ORDER,
                'Actual_ER': [results['actual_ERs'][net] for net in NETWORK_ORDER],
                'P_Value': [results['p_values'][net] for net in NETWORK_ORDER],
                'Null_Mean': [np.mean(results['null_distributions'][net]) for net in NETWORK_ORDER],
                'Null_Std': [np.std(results['null_distributions'][net]) for net in NETWORK_ORDER]
            })
            stats_filename = csv_file.replace('.csv', '_spin_test_stats.csv')
            stats_path = os.path.join(BASE_PATH, stats_filename)
            stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
            print(f"统计结果已保存至: {stats_path}")
            
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("所有分析完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

