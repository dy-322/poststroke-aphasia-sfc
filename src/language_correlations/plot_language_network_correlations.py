# ----------相关性森林图（按Network分组，包含皮层下）----------
# Python版本要求：3.6
# 需要安装的库：numpy pandas matplotlib scipy statsmodels
# 导入必要的库

import numpy as np  # 数值计算库
import pandas as pd  # 数据处理库
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，确保PNG文件能正确保存
import matplotlib.pyplot as plt  # 绘图库
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import sem  # 标准误
from statsmodels.stats.multitest import multipletests  # FDR校正
import os

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
# 确保PNG文件正确保存
plt.rcParams['figure.max_open_warning'] = 0

# ==================== 1. 数据准备 ====================
# 定义CSV文件路径
base_path = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\images"
csv_files = [
    ("roi_sfc_naming_subscore_correlations.csv", "Naming Subscore"),
    ("roi_sfc_comprehension_subscore_correlations.csv", "Comprehension Subscore"),
    ("roi_sfc_wab_r_aq_correlations.csv", "WAB-R AQ"),
    ("roi_sfc_repetition_subscore_correlations.csv", "Repetition Subscore")
]

# 定义网络顺序（7个网络 + 皮层下）
network_order = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis", "Subcortex"]

# 定义颜色方案（与sig_brain_spintest.py一致）
# 7个网络的颜色 + 皮层下的颜色（使用灰色）
network_colors = ["#3287bd", "#65c2a5", "#abdca4", "#ffffbf", "#fdae61", "#f56d43", "#d53e4f", "#808080"]

# 皮层下ROI的索引范围（前50个ROI）
SUBCORTICAL_OFFSET = 50

# ==================== 2. 辅助函数 ====================
def identify_subcortical_regions(df):
    """识别皮层下区域：Network为空或不在7个网络中的ROI"""
    subcortical_mask = (
        df['Network'].isna() | 
        (df['Network'] == '') |
        (~df['Network'].isin(["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis"]))
    )
    return subcortical_mask

def calculate_forest_plot_stats(data_points):
    """计算森林图所需的统计量"""
    if len(data_points) == 0:
        return None, None, None, None
    
    mean_val = np.mean(data_points)
    std_val = np.std(data_points, ddof=1)  # 样本标准差
    sem_val = sem(data_points)  # 标准误
    ci_95_lower = mean_val - 1.96 * sem_val
    ci_95_upper = mean_val + 1.96 * sem_val
    
    return mean_val, std_val, sem_val, (ci_95_lower, ci_95_upper)

# ==================== 3. 森林图绘制函数 ====================
def compute_global_network_order(loaded_dfs):
    """跨所有数据集计算每个网络的全局均值，返回按均值从大到小排序的网络列表"""
    network_means = {net: [] for net in network_order}
    for csv_file, (df, title) in loaded_dfs.items():
        df_work = df.copy()
        subcortical_mask = identify_subcortical_regions(df_work)
        df_work.loc[subcortical_mask, 'Network'] = 'Subcortex'
        df_filtered = df_work[df_work['Network'].notna() & (df_work['Network'] != '')].copy()
        for network in network_order:
            network_data = df_filtered[df_filtered['Network'] == network]
            if len(network_data) > 0:
                data_points = network_data['Correlation'].values
                data_points = data_points[~np.isnan(data_points)]
                if len(data_points) > 0:
                    network_means[network].append(np.mean(data_points))
    global_means = {net: (np.mean(vals) if vals else 0) for net, vals in network_means.items()}
    sorted_order = sorted(network_order, key=lambda x: global_means[x], reverse=True)
    print("Forest plot y-axis order (by global mean):")
    for net in sorted_order:
        print(f"  {net}: {global_means[net]:.4f}")
    return sorted_order


def plot_forest_correlation_by_network(df, title, output_filename, sorted_network_order=None):
    """绘制按Network分组的相关性森林图（包含半小提琴图）"""
    plot_order = sorted_network_order if sorted_network_order is not None else network_order
    # 构建网络颜色映射（保持与原始network_order的对应关系）
    color_map = {net: network_colors[i] for i, net in enumerate(network_order)}

    # 创建数据副本
    df_work = df.copy()

    # 识别皮层下区域
    subcortical_mask = identify_subcortical_regions(df_work)
    df_work.loc[subcortical_mask, 'Network'] = 'Subcortex'

    # 过滤掉Network仍为空的数据
    df_filtered = df_work[df_work['Network'].notna() & (df_work['Network'] != '')].copy()

    if len(df_filtered) == 0:
        print(f"警告：{title} 没有有效的Network数据")
        return

    # 创建图形和坐标轴（缩小尺寸）
    fig, ax = plt.subplots(figsize=(9, 6))

    # 设置类别在y轴上的位置（从上到下）
    positions = np.arange(len(plot_order))
    positions_dict = {net: pos for pos, net in enumerate(plot_order)}

    # 绘制0线（highlight）- 深灰色虚线
    ax.axvline(x=0, color='#404040', linestyle='--', linewidth=2, alpha=0.7, zorder=0)

    # ==================== 3. 绘制图形 ====================
    for network in plot_order:
        # 获取当前网络的数据值
        network_data = df_filtered[df_filtered['Network'] == network]
        
        if len(network_data) == 0:
            # 如果没有数据，跳过但保留位置（y轴标签仍会显示）
            continue
        
        data_points = network_data['Correlation'].values
        
        # 过滤掉NaN值
        data_points = data_points[~np.isnan(data_points)]
        
        if len(data_points) == 0:
            continue
        
        color = color_map[network]
        y_pos = positions_dict[network]
        
        # 计算统计量
        mean_val, std_val, sem_val, ci_95 = calculate_forest_plot_stats(data_points)
        
        if mean_val is None:
            continue
        
        # ----------------- 3.1 绘制均值点和置信区间（森林图核心） -----------------
        # 绘制置信区间线
        ax.plot([ci_95[0], ci_95[1]], [y_pos, y_pos], 
                color=color, linewidth=2.5, alpha=0.8, zorder=2)
        
        # 绘制均值点（菱形），边框颜色比箱体颜色更深
        # 将颜色转换为RGB，然后加深
        import matplotlib.colors as mcolors
        rgb = mcolors.hex2color(color)
        # 加深颜色（乘以0.6）
        darker_color = tuple(max(0, min(1, c * 0.6)) for c in rgb)
        ax.scatter([mean_val], [y_pos], color=color, s=150, 
                  edgecolor=darker_color, linewidth=2, zorder=3, marker='D')
        
        # 添加统计信息文本（可选）
        # info_text = f"n={len(data_points)}\nμ={mean_val:.3f}"
        # ax.text(mean_val + 0.05, y_pos, info_text, 
        #         fontsize=9, va='center', ha='left', fontfamily='Arial')
    
    # ==================== 4. 图表美化 ====================
    ax.set_yticks(positions)
    ax.set_yticklabels(plot_order, fontsize=14, fontfamily='Arial')
    ax.set_xlabel('Correlation Coefficient', fontsize=15, fontfamily='Arial', fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, fontfamily='Arial')
    
    # 设置x轴格式
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # 添加网格线
    ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
    
    # 显示边框
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(2)
        ax.spines[spine].set_color('black')
    
    # 反转y轴，使第一个网络在顶部
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # 保存图形为PDF
    output_path = os.path.join(base_path, output_filename.replace('.png', '.pdf'))
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 使用fig对象保存为PDF格式
        fig.savefig(output_path, dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='pdf')
        print(f"图片已保存至: {output_path}")
        # 验证文件是否存在且大小大于0
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                print(f"文件大小: {file_size} 字节")
            else:
                print(f"警告: 文件大小为0，可能保存失败")
        else:
            print(f"错误: 文件未创建")
    except Exception as e:
        print(f"保存图片时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close(fig)

# ==================== 4. 雷达图绘制函数 ====================
def get_radar_values(df):
    """从df提取每个网络FDR显著脑区数量，返回values列表（顺序同network_order）"""
    df_work = df.copy()
    subcortical_mask = identify_subcortical_regions(df_work)
    df_work.loc[subcortical_mask, 'Network'] = 'Subcortex'
    df_filtered = df_work[df_work['Network'].notna() & (df_work['Network'] != '')].copy()

    if 'P_Value_FDR_Corrected' not in df_filtered.columns:
        p_value_col = None
        if 'P_Value_Uncorrected' in df_filtered.columns:
            p_value_col = 'P_Value_Uncorrected'
        elif 'P_Value' in df_filtered.columns:
            p_value_col = 'P_Value'
        if p_value_col is None:
            return [0] * len(network_order)
        valid_mask = ~np.isnan(df_filtered[p_value_col])
        valid_p_values = df_filtered.loc[valid_mask, p_value_col].values
        if len(valid_p_values) == 0:
            return [0] * len(network_order)
        _, p_corrected, _, _ = multipletests(valid_p_values, alpha=0.05, method='fdr_bh')
        df_filtered['P_Value_FDR_Corrected'] = np.nan
        df_filtered.loc[valid_mask, 'P_Value_FDR_Corrected'] = p_corrected

    network_counts = {}
    for network in network_order:
        network_data = df_filtered[df_filtered['Network'] == network]
        if len(network_data) > 0:
            sig_mask = (network_data['P_Value_FDR_Corrected'].notna() &
                        (network_data['P_Value_FDR_Corrected'] < 0.05))
            network_counts[network] = int(sig_mask.sum())
        else:
            network_counts[network] = 0
    return [network_counts[net] for net in network_order]


def get_network_roi_counts(df):
    """返回每个网络的全部 ROI 数量（不论是否显著），顺序同 network_order"""
    df_work = df.copy()
    subcortical_mask = identify_subcortical_regions(df_work)
    df_work.loc[subcortical_mask, 'Network'] = 'Subcortex'
    df_filtered = df_work[df_work['Network'].notna() & (df_work['Network'] != '')].copy()
    return [int((df_filtered['Network'] == net).sum()) for net in network_order]


def plot_radar_significant_regions(df, title, output_filename, color=None,
                                   global_min=None, global_max=None):
    """绘制FDR校正后显著脑区数量的雷达图（风格同plot_radar.py，无网络标注）"""
    import matplotlib.patches as mpatches

    values = get_radar_values(df)
    roi_counts = get_network_roi_counts(df)
    num_vars = len(network_order)

    # 全局归一化范围（跨4个分数统一，由调用方传入）
    data_min = global_min if global_min is not None else min(values)
    data_max = global_max if global_max is not None else max(values)
    if data_max == data_min:
        data_max = data_min + 1

    # 角度 & 半径参数（与plot_radar.py一致）
    angles = np.linspace(np.pi / 2, -1.5 * np.pi, num_vars, endpoint=False)
    base_radius = 20
    max_radius  = 65
    radius_range = max_radius - base_radius

    # ---------- 创建图形 ----------
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 内参考圆（虚线）
    circle = plt.Circle(
        (0, 0), base_radius, transform=ax.transData._b,
        facecolor='white', edgecolor='black', linestyle='--',
        linewidth=1, alpha=1.0, zorder=1)
    ax.add_patch(circle)

    # 20%参考圆（虚线）
    twenty_percent_radius = base_radius + 0.2 * radius_range
    twenty_percent_circle = plt.Circle(
        (0, 0), twenty_percent_radius, transform=ax.transData._b,
        facecolor='none', edgecolor='black', linestyle='--',
        linewidth=1, alpha=1.0, zorder=1)
    ax.add_patch(twenty_percent_circle)

    # 外圆实线（linewidth=2.5，与plot_radar.py一致）
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(theta, [max_radius] * len(theta), color='black', linewidth=2.5, zorder=1)

    # 辐射角度线
    for angle in angles:
        ax.plot([angle, angle], [base_radius, max_radius],
                color='lightgray', linestyle='--', linewidth=0.8, zorder=0)

    # ---------- 数据映射 & 绘制 ----------
    if color is None:
        color = "#7895C1"

    line_radii = [base_radius + (v - data_min) / (data_max - data_min) * radius_range
                  for v in values]
    closed_angles = np.append(angles, angles[0])
    closed_radii  = np.append(line_radii, line_radii[0])

    ax.plot(closed_angles, closed_radii, color=color, linewidth=2.5, zorder=1)
    ax.fill(closed_angles, closed_radii, color=color, alpha=0.3, zorder=1)
    ax.scatter(angles, line_radii, color=color, s=50, zorder=2)

    # ---------- 前两名百分比标注：显著脑区数 ÷ 该网络总脑区数 ----------
    sorted_indices = np.argsort(values)[::-1]
    top2_indices = sorted_indices[:2]
    for rank, idx in enumerate(top2_indices):
        angle  = angles[idx]
        radius = line_radii[idx]
        denom  = roi_counts[idx] if roi_counts[idx] > 0 else 1
        pct    = values[idx] / denom * 100
        offset = 5 if rank == 0 else -5
        ax.text(angle, radius + offset, f'{pct:.1f}%',
                ha='center', va='center', fontsize=24,
                fontfamily='Arial', weight='bold', color=color, zorder=3)

    # ---------- 隐藏极坐标默认元素 ----------
    ax.set_ylim(0, 100)
    ax.spines['polar'].set_visible(False)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticks([])

    # ---------- 图例 ----------
    group_patch = mpatches.Patch(color=color, alpha=0.6, label=title)
    fig.legend(
        handles=[group_patch],
        loc='lower center', bbox_to_anchor=(0.5, 0.08),
        frameon=False, ncol=1,
        prop={'family': 'Arial', 'weight': 'bold', 'size': 8},
        columnspacing=1.2, handletextpad=0.5)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

    # ---------- 保存 ----------
    radar_filename = output_filename.replace('_correlation_forest_plot.png', '_radar_plot.pdf')
    output_path = os.path.join(base_path, radar_filename)
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, format='pdf',
                    bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
        print(f"Radar saved: {output_path}")
        print(f"  counts: {dict(zip(network_order, values))}")
    except Exception as e:
        print(f"Error saving radar: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close(fig)


def plot_radar_combined(loaded_dfs, radar_colors, global_min, global_max):
    """将4个分数的雷达图叠加绘制在同一张图上"""
    import matplotlib.patches as mpatches

    num_vars = len(network_order)
    angles = np.linspace(np.pi / 2, -1.5 * np.pi, num_vars, endpoint=False)
    base_radius = 20
    max_radius  = 65
    radius_range = max_radius - base_radius

    data_min = global_min
    data_max = global_max
    if data_max == data_min:
        data_max = data_min + 1

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 内参考圆（虚线）
    circle = plt.Circle(
        (0, 0), base_radius, transform=ax.transData._b,
        facecolor='white', edgecolor='black', linestyle='--',
        linewidth=1, alpha=1.0, zorder=1)
    ax.add_patch(circle)

    # 20%参考圆（虚线）
    twenty_percent_radius = base_radius + 0.2 * radius_range
    twenty_percent_circle = plt.Circle(
        (0, 0), twenty_percent_radius, transform=ax.transData._b,
        facecolor='none', edgecolor='black', linestyle='--',
        linewidth=1, alpha=1.0, zorder=1)
    ax.add_patch(twenty_percent_circle)

    # 外圆实线
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(theta, [max_radius] * len(theta), color='black', linewidth=1, zorder=1)

    # 辐射角度线
    for angle in angles:
        ax.plot([angle, angle], [base_radius, max_radius],
                color='lightgray', linestyle='--', linewidth=0.8, zorder=0)

    legend_handles = []
    for csv_file, title in csv_files:
        if csv_file not in loaded_dfs:
            continue
        df, title = loaded_dfs[csv_file]
        color = radar_colors.get(title, "#7895C1")
        values = get_radar_values(df)
        roi_counts = get_network_roi_counts(df)

        line_radii = [base_radius + (v - data_min) / (data_max - data_min) * radius_range
                      for v in values]
        closed_angles = np.append(angles, angles[0])
        closed_radii  = np.append(line_radii, line_radii[0])

        ax.plot(closed_angles, closed_radii, color=color, linewidth=2, zorder=2)
        ax.scatter(angles, line_radii, color=color, s=50, zorder=3)

        # 标注最大值：显著脑区数 ÷ 该网络总脑区数
        max_idx = int(np.argmax(values))
        denom = roi_counts[max_idx] if roi_counts[max_idx] > 0 else 1
        pct = values[max_idx] / denom * 100
        angle_max = angles[max_idx]
        radius_max = line_radii[max_idx]
        ax.text(angle_max, radius_max + 7, f'{pct:.1f}%',
                ha='center', va='center', fontsize=11,
                fontfamily='Arial', weight='bold', color=color, zorder=4)

        legend_handles.append(mpatches.Patch(color=color, alpha=0.8, label=title))

    ax.set_ylim(0, 100)
    ax.spines['polar'].set_visible(False)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticks([])

    fig.legend(
        handles=legend_handles,
        loc='lower center', bbox_to_anchor=(0.5, 0.05),
        frameon=False, ncol=2,
        prop={'family': 'Arial', 'weight': 'bold', 'size': 10},
        columnspacing=1.2, handletextpad=0.5)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.2)

    output_path = os.path.join(base_path, "radar_combined_all_scores.pdf")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, format='pdf',
                    bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
        print(f"Combined radar saved: {output_path}")
    except Exception as e:
        print(f"Error saving combined radar: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close(fig)


def plot_forest_combined(loaded_dfs, sorted_network_order, score_colors):
    """将4个分数的森林图叠加绘制在同一张图上（CI线 + 均值菱形点）"""
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines

    plot_order = sorted_network_order
    n_networks = len(plot_order)
    net_spacing = 1.2         # 网络间距（单位：y轴刻度）
    offsets = [-0.20, -0.10, 0.10, 0.20]   # 网络内4个分数的偏移，更紧凑

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.axvline(x=0, color='#404040', linestyle='--', linewidth=2, alpha=0.7, zorder=3)
    ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)

    # 交替行背景（按网络间距调整宽度）
    for i in range(n_networks):
        if i % 2 == 0:
            ax.axhspan(i * net_spacing - net_spacing / 2,
                       i * net_spacing + net_spacing / 2,
                       color='#f5f5f5', zorder=0)

    legend_handles = []
    for score_idx, (csv_file, title) in enumerate(csv_files):
        if csv_file not in loaded_dfs:
            continue
        df, title = loaded_dfs[csv_file]
        color = score_colors.get(title, "#7895C1")

        df_work = df.copy()
        subcortical_mask = identify_subcortical_regions(df_work)
        df_work.loc[subcortical_mask, 'Network'] = 'Subcortex'
        df_filtered = df_work[df_work['Network'].notna() & (df_work['Network'] != '')].copy()

        rgb = mcolors.to_rgb(color)
        darker_color = tuple(max(0, min(1, c * 0.6)) for c in rgb)

        for net_idx, network in enumerate(plot_order):
            network_data = df_filtered[df_filtered['Network'] == network]
            if len(network_data) == 0:
                continue
            data_points = network_data['Correlation'].values
            data_points = data_points[~np.isnan(data_points)]
            if len(data_points) == 0:
                continue

            mean_val, _, _, ci_95 = calculate_forest_plot_stats(data_points)
            if mean_val is None:
                continue

            y_pos = net_idx * net_spacing + offsets[score_idx]
            ax.plot([ci_95[0], ci_95[1]], [y_pos, y_pos],
                    color=color, linewidth=2, alpha=0.85, zorder=2)
            ax.scatter([mean_val], [y_pos], color=color, s=80,
                       edgecolor=darker_color, linewidth=1.5, zorder=3, marker='D')

        handle = mlines.Line2D([], [], color=color, marker='D', markersize=8,
                               markeredgecolor=darker_color, markeredgewidth=1.5,
                               linewidth=2, label=title)
        legend_handles.append(handle)

    ax.set_yticks(np.arange(n_networks) * net_spacing)
    ax.set_yticklabels(plot_order, fontsize=13, fontfamily='Arial')
    ax.set_xlabel('Correlation Coefficient', fontsize=14, fontfamily='Arial', fontweight='bold')
    ax.set_title('SC-FC Coupling Correlations by Network', fontsize=15,
                 fontweight='bold', pad=16, fontfamily='Arial')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['left'].set_color('black')

    ax.invert_yaxis()
    ax.set_xlim(-0.25, 0.05)
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1),
              frameon=False,
              prop={'family': 'Arial', 'size': 10})

    plt.tight_layout()

    output_path = os.path.join(base_path, "forest_combined_all_scores.pdf")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, format='pdf',
                    bbox_inches='tight', facecolor='white', edgecolor='none', dpi=600)
        print(f"Combined forest saved: {output_path}")
    except Exception as e:
        print(f"Error saving combined forest: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close(fig)


if __name__ == "__main__":
    # --- 第一遍：计算全局 min/max（统一4个分数的雷达图显示范围）---
    all_radar_values = []
    loaded_dfs = {}
    for csv_file, title in csv_files:
        csv_path = os.path.join(base_path, csv_file)
        if not os.path.exists(csv_path):
            print(f"Warning: file not found {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path)
            if csv_file == "roi_sfc_repetition_subscore_correlations.csv":
                nan_count = df['Network'].isna().sum()
                if nan_count > 0:
                    df.loc[df['Network'].isna(), 'Network'] = 'Subcortex'
            loaded_dfs[csv_file] = (df, title)
            all_radar_values.extend(get_radar_values(df))
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    global_min = min(all_radar_values) if all_radar_values else 0
    global_max = max(all_radar_values) if all_radar_values else 1
    print(f"Global radar range: [{global_min}, {global_max}]")

    # --- 计算森林图 y 轴全局排序（跨4个分数按均值从大到小）---
    sorted_network_order = compute_global_network_order(loaded_dfs)

    # --- 第二遍：绘图 ---
    radar_colors = {
        "Naming Subscore":        "#fdae61",   # SalVentAttn
        "Comprehension Subscore": "#3287bd",   # Cont
        "WAB-R AQ":               "#f56d43",   # SomMot
        "Repetition Subscore":    "#abdca4",   # DorsAttn
    }

    for csv_file, title in csv_files:
        if csv_file not in loaded_dfs:
            continue
        df, title = loaded_dfs[csv_file]
        try:
            print(f"\nProcessing: {csv_file}")
            output_filename = csv_file.replace('.csv', '_correlation_forest_plot.png')
            plot_forest_correlation_by_network(df, title, output_filename,
                                               sorted_network_order=sorted_network_order)
            radar_color = radar_colors.get(title, "#7895C1")
            plot_radar_significant_regions(df, title, output_filename,
                                           color=radar_color,
                                           global_min=global_min,
                                           global_max=global_max)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()

    # --- 合并雷达图（4个分数叠加）---
    print("\nGenerating combined radar plot...")
    plot_radar_combined(loaded_dfs, radar_colors, global_min, global_max)

    # --- 合并森林图（4个分数叠加）---
    print("\nGenerating combined forest plot...")
    plot_forest_combined(loaded_dfs, sorted_network_order, radar_colors)

    print("\nDone.")
