from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import to_rgb, to_hex
from scipy import stats


CSV_PATHS = [
    r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\single_subject_results_t288\Global_network_sfc_median.csv",
    r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\single_subject_results_t288\Conduction_network_sfc_median.csv",
    r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\single_subject_results_t288\None_network_sfc_median.csv",
    r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\single_subject_results_t288\Anomic_network_sfc_median.csv",
    r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\single_subject_results_t288\Broca_network_sfc_median.csv",
    r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\single_subject_results_t288\9_network_sfc_median.csv",
]

# 失语症类型标签（从文件名提取，例如 "Global_network_sfc_median.csv" -> "Global"）
def extract_label_from_path(csv_path):
    """从CSV路径提取标签名称"""
    stem = Path(csv_path).stem  # 例如 "Global_network_sfc_median"
    # 去掉 "_network_sfc_median" 后缀
    label = stem.replace('_network_sfc_median', '')
    # 如果标签是"9"，保持为"9"，否则首字母大写
    if label == '9':
        return "Healthy"

    if label == "None":
        return "Normal Stroke"
        
    return label.capitalize()

APHASIA_LABELS = [extract_label_from_path(p) for p in CSV_PATHS]
CATEGORIES = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis"]


def lighten_color(color, factor=0.5):
    """将颜色变浅，factor 越小颜色越浅（0-1之间）"""
    rgb = np.array(to_rgb(color))
    # 向白色(1,1,1)方向混合
    light_rgb = rgb + (1 - rgb) * (1 - factor)
    return to_hex(light_rgb)


def get_significance_stars(p_value):
    """根据p值返回显著性星号"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'  # not significant


def add_significance_annotation(ax, x1, x2, y, stars, color='black'):
    """在两个箱型图之间添加显著性标注 - 只用一条水平直线"""
    # 只绘制一条水平直线
    ax.plot([x1, x2], [y, y], lw=1.2, c=color)
    ax.text((x1 + x2) / 2, y + 0.002, stars, ha='center', va='bottom', fontsize=9, fontfamily='Arial')


def plot_all_network_subplots(csv_paths, categories, output_path=None):
    """
    绘制多子图布局的箱线图，每个网络一个子图，包含显著性分析。
    """
    # 散点颜色配置 - 使用之前的颜色方案
    aphasia_colors = ["#7895C1", "#A8CBDF", "#992224", "#EF8B67", "#E3625D", "#B54764"]
    
    # 根据传入的csv_paths计算标签
    aphasia_labels = [extract_label_from_path(p) for p in csv_paths]
    
    # 创建颜色字典（用于散点）
    color_dict = {}
    for i, label in enumerate(aphasia_labels):
        if label == 'Healthy':
            color_dict[label] = '#808080'  # 灰色
        else:
            color_dict[label] = aphasia_colors[i] if i < len(aphasia_colors) else '#808080'
    
    # 预先载入数据
    dataframes = {}
    for csv_path, label in zip(csv_paths, aphasia_labels):
        df = pd.read_csv(csv_path)
        dataframes[label] = df
    
    # 确定绘图顺序：Healthy在第一位
    desired_order = ['Healthy', 'Normal Stroke', 'Anomic', 'Broca', 'Conduction', 'Global']
    plot_order = [label for label in desired_order if label in aphasia_labels]
    for label in aphasia_labels:
        if label not in plot_order:
            plot_order.append(label)
    
    # 设置字体
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # 创建子图布局：2行4列（7个网络 + 1个图例位置）
    n_networks = len(categories)
    n_cols = 4
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    axes = axes.flatten()
    
    rng = np.random.default_rng(seed=42)
    
    # 获取Healthy组的数据用于显著性分析
    healthy_data_dict = {}
    if 'Healthy' in dataframes:
        for cat in categories:
            healthy_data_dict[cat] = dataframes['Healthy'][cat].dropna().values
    
    # 绘制每个网络的子图
    for cat_idx, category in enumerate(categories):
        ax = axes[cat_idx]
        
        positions = list(range(len(plot_order)))
        max_y = -np.inf
        min_y = np.inf
        
        # 存储每组数据用于显著性分析
        data_by_group = {}
        
        # 绘制每个组的箱型图
        for pos, label in enumerate(plot_order):
            if label not in dataframes:
                continue
            
            data_points = dataframes[label][category].dropna().values
            if data_points.size == 0:
                continue
            
            data_by_group[label] = data_points
            max_y = max(max_y, np.max(data_points))
            min_y = min(min_y, np.min(data_points))
            
            # 散点颜色
            scatter_color = color_dict[label]
            
            # 箱型图：透明填充、黑色边框、无cap线
            ax.boxplot(
                data_points,
                positions=[pos],
                widths=0.6,
                patch_artist=True,
                showfliers=False,
                notch=False,
                showcaps=False,  # 不显示cap线
                medianprops={'color': 'black', 'linewidth': 1.5},
                boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 1.5},
                whiskerprops={'color': 'black', 'linewidth': 1.2},
                capprops={'color': 'black', 'linewidth': 0},  # cap线宽度为0
            )
            
            # 添加散点（使用色系颜色）
            ax.scatter(
                rng.normal(pos, 0.08, len(data_points)),
                data_points,
                color=scatter_color,
                alpha=0.8,
                s=18,
                edgecolors='none',
                zorder=3,
            )
        
        # 设置子图属性（先设置Y轴范围，只覆盖数据范围）
        ax.set_title(category, fontsize=14, fontweight='bold', fontfamily='Arial')
        ax.set_xticks(positions)
        ax.set_xticklabels([])  # 不显示x轴标签，用图例代替
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='y', labelsize=10)
        
        # 设置y轴范围，只覆盖数据本身（不为显著性线预留刻度）
        if max_y > min_y:
            y_margin_data = (max_y - min_y) * 0.08  # 数据上下方的小边距
            ax.set_ylim(min_y - y_margin_data, max_y + y_margin_data)
        
        # 添加显著性分析（与Healthy组比较）- 在设置ylim之后，允许绘制超出范围
        if 'Healthy' in data_by_group and len(data_by_group) > 1:
            healthy_data = data_by_group['Healthy']
            y_range = max_y - min_y
            y_offset = y_range * 0.08  # 显著性标注的高度增量
            current_y = max_y + y_range * 0.03  # 起始位置靠近数据
            
            for pos, label in enumerate(plot_order):
                if label == 'Healthy' or label not in data_by_group:
                    continue
                
                other_data = data_by_group[label]
                
                # 使用Mann-Whitney U检验（非参数检验，适合小样本）
                try:
                    stat, p_value = stats.mannwhitneyu(healthy_data, other_data, alternative='two-sided')
                    stars = get_significance_stars(p_value)
                    
                    if stars != 'ns':  # 只显示显著的结果
                        healthy_pos = plot_order.index('Healthy')
                        # clip_on=False 允许显著性线绘制在轴外
                        ax.plot([healthy_pos, pos], [current_y, current_y], lw=1.2, c='black', clip_on=False)
                        ax.text((healthy_pos + pos) / 2, current_y + 0.002, stars, ha='center', va='bottom', 
                               fontsize=9, fontfamily='Arial', clip_on=False)
                        current_y += y_offset
                except Exception:
                    pass
        
        # 设置坐标轴样式
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.2)
        
        ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.6)
    
    # 隐藏多余的子图（第8个位置用于图例）
    for idx in range(n_networks, n_rows * n_cols):
        axes[idx].axis('off')
    
    # 在最后一个位置创建图例
    legend_ax = axes[-1]
    legend_handles = []
    for label in plot_order:
        legend_color = color_dict[label]
        # 使用圆形标记代替方块，与散点一致
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=legend_color, markersize=10,
                                         label=label, linestyle='None'))
    
    legend_ax.legend(handles=legend_handles, loc='center', frameon=True, 
                     fontsize=11, prop={'family': 'Arial'},
                     title='Type', title_fontsize=12,
                     fancybox=True, shadow=False,
                     edgecolor='gray', labelspacing=1.2)
    legend_ax.set_title('', fontsize=1)  # 清除标题
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none', dpi=300,
                   metadata={'Creator': 'Python matplotlib'})
    plt.close()


def plot_mean_sfc_boxplot(csv_paths, categories, output_path=None):
    """
    绘制所有网络SFC均值的箱型图，按失语症类型分组（Healthy, Normal Stroke, 其他）
    """
    # 颜色映射：与主图保持一致
    aphasia_colors = ["#7895C1","#A8CBDF","#992224","#EF8B67","#E3625D","#B54764"]
    
    # 根据传入的csv_paths计算标签
    aphasia_labels = [extract_label_from_path(p) for p in csv_paths]
    
    # 创建颜色字典：根据原始索引映射颜色
    color_dict = {}
    for i, (csv_path, label) in enumerate(zip(csv_paths, aphasia_labels)):
        if label == 'Healthy':
            color_dict[label] = '#808080'  # 灰色
        else:
            # 使用原始索引对应的颜色
            color_dict[label] = aphasia_colors[i] if i < len(aphasia_colors) else '#808080'
    
    # 读取数据并计算每个受试者的所有网络SFC均值
    mean_data = {}  # {label: [mean_values]}
    
    for csv_path, label in zip(csv_paths, aphasia_labels):
        df = pd.read_csv(csv_path)
        # 计算每个受试者的所有网络SFC均值
        network_cols = [c for c in categories if c in df.columns]
        if network_cols:
            mean_values = df[network_cols].mean(axis=1).dropna().values
            if label not in mean_data:
                mean_data[label] = []
            mean_data[label].extend(mean_values.tolist())
    
    # 设置所有字体为Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(seed=42)
    
    # 按顺序组织数据：Healthy, Normal Stroke, 其他
    plot_order = []
    plot_data = []
    plot_colors = []
    
    # 先添加Healthy
    if 'Healthy' in mean_data:
        plot_order.append('Healthy')
        plot_data.append(mean_data['Healthy'])
        plot_colors.append(color_dict['Healthy'])
    
    # 再添加Normal Stroke
    if 'Normal Stroke' in mean_data:
        plot_order.append('Normal Stroke')
        plot_data.append(mean_data['Normal Stroke'])
        plot_colors.append(color_dict['Normal Stroke'])
    
    # 最后添加其他组（排除Healthy和Normal Stroke）
    for label in aphasia_labels:
        if label not in ['Healthy', 'Normal Stroke'] and label in mean_data:
            plot_order.append(label)
            plot_data.append(mean_data[label])
            plot_colors.append(color_dict[label])
    
    # 绘制箱型图
    positions = list(range(len(plot_order)))
    healthy_pos = None
    
    for pos_idx, (pos, data, label, color) in enumerate(zip(positions, plot_data, plot_order, plot_colors)):
        if len(data) == 0:
            continue
        
        is_healthy = (label == 'Healthy')
        if is_healthy:
            healthy_pos = pos
            border_color = '#808080'
            fill_color = lighten_color(border_color, factor=0.5)
        else:
            border_color = color
            fill_color = lighten_color(border_color, factor=0.5)
        
        ax.boxplot(
            data,
            positions=[pos],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            notch=False,
            medianprops={'color': border_color, 'linewidth': 1.8},
            boxprops={'facecolor': fill_color, 'edgecolor': border_color, 'linewidth': 1.8},
            whiskerprops={'color': border_color, 'linewidth': 1.6},
            capprops={'color': border_color, 'linewidth': 1.6},
        )
        
        # 添加散点
        scatter_color = border_color
        ax.scatter(
            rng.normal(pos, 0.05, len(data)),
            data,
            color=scatter_color,
            alpha=0.75,
            s=20,
            edgecolors='none',
            zorder=3,
        )
    
    # 添加Healthy组的灰色背景
    if healthy_pos is not None:
        box_width = 0.5
        spacing = 0.1
        x_start = healthy_pos - box_width / 2 - spacing
        x_end = healthy_pos + box_width / 2 + spacing
        ax.axvspan(x_start, x_end, facecolor='lightgray', alpha=0.3, zorder=0)
    
    # 设置x轴标签
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_order, rotation=15, fontsize=12, fontfamily='Arial')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    
    # 设置刻度标签字体为Arial
    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
    
    ax.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.8)
    ax.set_ylabel('Mean SFC across all networks', fontsize=12, fontfamily='Arial')
    
    # 设置坐标轴范围
    ax.set_xlim(-0.5, len(plot_order) - 0.5)
    
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.4)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none', 
                   metadata={'Creator': 'Python matplotlib'})
    plt.close()


def plot_all_network(csv_paths, categories, output_path=None, legend_loc="upper right"):
    """
    在同一坐标轴上绘制多个网络的箱线图+半小提琴+散点组合，每个网络包含不同失语症类型。
    """
    # aphasia_colors = ["#d36a87", "#ea9979", "#83b6b5", "#bcdfa7", "#a596ee", "#9bb55e", "#fcd744"]
    aphasia_colors = ["#7895C1","#A8CBDF","#992224","#EF8B67","#E3625D","#B54764"]
    if len(aphasia_colors) < len(csv_paths):
        raise ValueError("提供的颜色数量不足以覆盖所有失语症类型。")

    group_gap = 0.9  # 不同网络之间的间隔
    inner_spacing = 0.2  # 同一网络下失语症类别之间的间距
    total_group_width = inner_spacing * (len(csv_paths) - 1) + 1

    # 根据传入的csv_paths计算标签
    aphasia_labels = [extract_label_from_path(p) for p in csv_paths]
    
    # 预先载入并检查数据，确保每个网络都有对应失语症的列
    dataframes = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        missing_cols = [c for c in categories if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{csv_path} 缺少列: {missing_cols}")
        dataframes.append(df)

    # 设置所有字体为Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    fig_width = max(12, len(categories) * 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    rng = np.random.default_rng(seed=42)

    # 存储"9"组在每个网络中的位置范围，用于绘制背景条
    healthy_group_ranges = []  # [(x_start, x_end), ...] 每个网络一个

    group_centers = []
    for cat_idx, category in enumerate(categories):
        group_offset = cat_idx * (total_group_width + group_gap)
        group_centers.append(group_offset + (total_group_width - 1) / 2)

        # 先找到"Healthy"组的位置
        healthy_group_pos = None
        for aph_idx, (df, aphasia_label) in enumerate(zip(dataframes, aphasia_labels)):
            if aphasia_label == 'Healthy':
                healthy_group_pos = group_offset + aph_idx * inner_spacing
                break
        
        # 如果找到"Healthy"组，记录其x范围
        if healthy_group_pos is not None:
            box_width = 0.16
            spacing = inner_spacing * 0.5  # 额外的spacing，增加范围
            x_start = healthy_group_pos - box_width / 2 - spacing
            x_end = healthy_group_pos + box_width / 2 + spacing
            healthy_group_ranges.append((x_start, x_end))

        for aph_idx, (df, aphasia_label) in enumerate(zip(dataframes, aphasia_labels)):
            data_points = df[category].dropna().values
            if data_points.size == 0:
                continue

            base_pos = group_offset + aph_idx * inner_spacing
            box_pos = base_pos

            # 判断是否为"Healthy"组（健康对照组）
            is_healthy_group = (aphasia_label == 'Healthy')
            
            # 设置颜色：边框和中位线使用原色，填充使用浅色
            # 如果是"9"组，使用灰色
            if is_healthy_group:
                border_color = '#808080'  # 灰色
                fill_color = lighten_color(border_color, factor=0.5)  # 浅灰色填充
            else:
                border_color = aphasia_colors[aph_idx]
                fill_color = lighten_color(border_color, factor=0.5)  # 浅色填充

            ax.boxplot(
                data_points,
                positions=[box_pos],
                widths=0.16,
                patch_artist=True,
                showfliers=False,
                notch=False,
                medianprops={'color': border_color, 'linewidth': 1.8},
                boxprops={'facecolor': fill_color, 'edgecolor': border_color, 'linewidth': 1.8},
                whiskerprops={'color': border_color, 'linewidth': 1.6},
                capprops={'color': border_color, 'linewidth': 1.6},
            )

            # 散点颜色也使用边框颜色
            scatter_color = border_color if is_healthy_group else aphasia_colors[aph_idx]
            ax.scatter(
                rng.normal(box_pos, 0.05, len(data_points)),
                data_points,
                color=scatter_color,
                alpha=0.75,
                s=20,
                edgecolors='none',
                zorder=3,
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(categories, rotation=15, fontsize=12, fontfamily='Arial')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    # 设置刻度标签字体为Arial
    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
    ax.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.8)
    
    # 设置坐标轴范围
    right_limit = (len(categories) - 1) * (total_group_width + group_gap) + total_group_width
    ax.set_xlim(-0.6, right_limit - 0.4)
    
    # 在所有箱型图和坐标轴设置完成后，添加"Healthy"组的灰色背景条
    # 这样可以确保背景条覆盖整个y轴范围
    if healthy_group_ranges:
        for x_start, x_end in healthy_group_ranges:
            # 使用axvspan绘制灰色背景条，覆盖整个y轴范围
            ax.axvspan(x_start, x_end, 
                      facecolor='lightgray', alpha=0.3, zorder=0)

    # 创建图例，放在顶部水平排列
    legend_handles = []
    for i in range(len(csv_paths)):
        if aphasia_labels[i] == 'Healthy':
            # "Healthy"组使用灰色
            legend_color = '#808080'
        else:
            legend_color = aphasia_colors[i]
        legend_handles.append(mpatches.Patch(facecolor=legend_color, label=aphasia_labels[i]))
    
    # 将图例放在图的顶部，水平排列，宽度与图窗一致
    # 使用fig.legend可以让图例跨越整个图窗宽度
    legend = fig.legend(handles=legend_handles, loc='upper center', 
                        bbox_to_anchor=(0.5, 0.98), 
                        frameon=False, ncol=len(csv_paths), fontsize=11, 
                        prop={'family': 'Arial'},
                        columnspacing=1.5, handletextpad=0.5)

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.4)
    
    # 设置y轴标签字体
    ax.set_ylabel(ax.get_ylabel(), fontfamily='Arial') if ax.get_ylabel() else None

    # 调整布局，为顶部图例预留空间
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if output_path:
        # 保存为高质量PDF
        plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none', 
                   metadata={'Creator': 'Python matplotlib'})
    plt.close()  # 关闭图形以释放内存


def plot_summary_boxplot(csv_paths, categories, output_path=None):
    """
    绘制汇总箱线图：Healthy, Normal Stroke, Aphasia（合并所有失语症类型）
    """
    # 颜色配置
    group_colors = {
        'Healthy': '#808080',       # 灰色
        'Normal Stroke': '#7895C1', # 蓝色
        'Aphasia': '#E3625D',       # 红色
    }
    
    # 根据传入的csv_paths计算标签
    aphasia_labels = [extract_label_from_path(p) for p in csv_paths]
    
    # 读取数据并分组
    all_data = {
        'Healthy': [],
        'Normal Stroke': [],
        'Aphasia': [],  # 合并所有失语症类型
    }
    
    for csv_path, label in zip(csv_paths, aphasia_labels):
        df = pd.read_csv(csv_path)
        # 计算每个受试者的所有网络SFC均值
        network_cols = [c for c in categories if c in df.columns]
        if network_cols:
            mean_values = df[network_cols].mean(axis=1).dropna().values
            
            if label == 'Healthy':
                all_data['Healthy'].extend(mean_values.tolist())
            elif label == 'Normal Stroke':
                all_data['Normal Stroke'].extend(mean_values.tolist())
            else:
                # 所有失语症类型合并到Aphasia
                all_data['Aphasia'].extend(mean_values.tolist())
    
    # 设置字体
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    fig, ax = plt.subplots(figsize=(5, 6))
    rng = np.random.default_rng(seed=42)
    
    plot_order = ['Healthy', 'Normal Stroke', 'Aphasia']
    positions = list(range(len(plot_order)))
    
    max_y = -np.inf
    min_y = np.inf
    data_by_group = {}
    
    # 绘制每个组的箱型图
    for pos, label in enumerate(plot_order):
        data_points = np.array(all_data[label])
        if data_points.size == 0:
            continue
        
        data_by_group[label] = data_points
        max_y = max(max_y, np.max(data_points))
        min_y = min(min_y, np.min(data_points))
        
        scatter_color = group_colors[label]
        
        # 箱型图：透明填充、黑色边框、无cap线
        ax.boxplot(
            data_points,
            positions=[pos],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            notch=False,
            showcaps=False,
            medianprops={'color': 'black', 'linewidth': 1.5},
            boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 1.5},
            whiskerprops={'color': 'black', 'linewidth': 1.2},
            capprops={'color': 'black', 'linewidth': 0},
        )
        
        # 添加散点
        ax.scatter(
            rng.normal(pos, 0.08, len(data_points)),
            data_points,
            color=scatter_color,
            alpha=0.8,
            s=25,
            edgecolors='none',
            zorder=3,
        )
    
    # 添加显著性分析（与Healthy组比较）
    if 'Healthy' in data_by_group:
        healthy_data = data_by_group['Healthy']
        y_range = max_y - min_y
        y_offset = y_range * 0.12
        current_y = max_y + y_range * 0.05
        healthy_pos = 0
        
        for pos, label in enumerate(plot_order):
            if label == 'Healthy' or label not in data_by_group:
                continue
            
            other_data = data_by_group[label]
            
            try:
                stat, p_value = stats.mannwhitneyu(healthy_data, other_data, alternative='two-sided')
                stars = get_significance_stars(p_value)
                
                if stars != 'ns':
                    add_significance_annotation(ax, healthy_pos, pos, current_y, stars)
                    current_y += y_offset
            except Exception:
                pass
    
    # 设置子图属性
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_order, fontsize=12, fontfamily='Arial')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', labelsize=11)
    ax.set_ylabel('Mean SFC', fontsize=12, fontfamily='Arial')
    
    # 设置y轴范围
    if max_y > min_y:
        y_margin = (max_y - min_y) * 0.35
        ax.set_ylim(min_y - (max_y - min_y) * 0.05, max_y + y_margin)
    
    # 设置坐标轴样式
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.2)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.6)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none', dpi=300,
                   metadata={'Creator': 'Python matplotlib'})
    plt.close()


def plot_grouped_network_boxplot(csv_paths, categories, output_path=None):
    """
    分组凹口箱线图：X轴为7个网络，每个位置包含6个类别的 notch 箱线图。
    - 类别顺序：Healthy, Normal Stroke, Anomic, Conduction, Global, Broca（Broca最后）
    - 配色：Anomic/Conduction/Global/Broca 使用 sig_brain_spintest 的网络配色（非蓝色）
    - 显著性：每个网络组内，各类别 vs Healthy 做 Mann-Whitney U 检验并标注
    """
    # sig_brain_spintest.py 的网络配色（去掉蓝色 #3287bd），随机选4个给4种失语症
    # 可用: #65c2a5, #abdca4, #fdae61, #f56d43, #d53e4f（跳过 #ffffbf 太浅）
    category_colors = {
        'Healthy':       '#7f7f7f',   # 灰
        'Normal Stroke': '#4c78a8',   # 蓝（保留）
        'Anomic':        '#65c2a5',   # 青绿
        'Conduction':    '#abdca4',   # 浅绿
        'Global':        '#f56d43',   # 橙红
        'Broca':         '#d53e4f',   # 深红
    }

    aphasia_labels = [extract_label_from_path(p) for p in csv_paths]

    dataframes = {}
    for csv_path, label in zip(csv_paths, aphasia_labels):
        df = pd.read_csv(csv_path)
        dataframes[label] = df

    # Broca 放最后
    desired_order = ['Healthy', 'Normal Stroke', 'Anomic', 'Conduction', 'Global', 'Broca']
    plot_order = [label for label in desired_order if label in dataframes]
    for label in aphasia_labels:
        if label not in plot_order:
            plot_order.append(label)

    n_groups = len(categories)
    n_subgroups = len(plot_order)
    box_width = 0.10
    inner_gap = 0.02
    group_gap = 0.55
    group_width = n_subgroups * box_width + (n_subgroups - 1) * inner_gap

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

    fig_w = max(14, n_groups * 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    group_centers = []
    global_max = -np.inf
    global_min = np.inf

    for net in categories:
        for label in plot_order:
            vals = (dataframes[label][net].dropna().values
                    if net in dataframes[label].columns else np.array([]))
            if vals.size:
                global_max = max(global_max, vals.max())
                global_min = min(global_min, vals.min())

    # ── 绘制箱线图 ──
    bg_labels = ['Healthy', 'Normal Stroke']   # 需要浅灰背景的类别
    bg_padding = inner_gap * 0.5               # 背景块左右各扩展的量

    for g_idx, net in enumerate(categories):
        g_start = g_idx * (group_width + group_gap)
        g_center = g_start + group_width / 2
        group_centers.append(g_center)

        # 为 Healthy 和 Normal Stroke 绘制浅灰背景
        for bg_label in bg_labels:
            if bg_label not in plot_order:
                continue
            bg_c_idx = plot_order.index(bg_label)
            bg_pos = g_start + bg_c_idx * (box_width + inner_gap)
            ax.axvspan(bg_pos - box_width / 2 - bg_padding,
                       bg_pos + box_width / 2 + bg_padding,
                       facecolor='#EBEBEB', alpha=0.55, zorder=0, linewidth=0)

        for c_idx, label in enumerate(plot_order):
            pos = g_start + c_idx * (box_width + inner_gap)
            vals = (dataframes[label][net].dropna().values
                    if net in dataframes[label].columns else np.array([]))
            if vals.size == 0:
                continue

            ax.boxplot(
                vals,
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                notch=True,
                showcaps=False,
                medianprops=dict(color='black', linewidth=1.0),
                boxprops=dict(facecolor=category_colors.get(label, '#999999'),
                              edgecolor='black', linewidth=0.7),
                whiskerprops=dict(color='black', linewidth=0.7),
                capprops=dict(linewidth=0),
            )

    # ── 组内显著性标注：每个网络 x 组内，各类别 vs Healthy ──
    y_range = (global_max - global_min) if global_max > global_min else 0.2
    y_step = y_range * 0.065   # 每条线间距

    for g_idx, net in enumerate(categories):
        g_start = g_idx * (group_width + group_gap)

        if 'Healthy' not in dataframes or net not in dataframes['Healthy'].columns:
            continue
        healthy_data = dataframes['Healthy'][net].dropna().values
        if healthy_data.size == 0:
            continue

        healthy_c_idx = plot_order.index('Healthy')
        healthy_pos = g_start + healthy_c_idx * (box_width + inner_gap)

        # 该组局部最大值（用于定位第一条线的起点）
        local_max = max(
            (dataframes[label][net].dropna().values.max()
             for label in plot_order
             if net in dataframes[label].columns
             and dataframes[label][net].dropna().size > 0),
            default=global_max
        )
        current_y = local_max + y_range * 0.03

        for label in plot_order:
            if label == 'Healthy':
                continue
            if net not in dataframes[label].columns:
                continue
            other_data = dataframes[label][net].dropna().values
            if other_data.size == 0:
                continue
            try:
                _, p = stats.mannwhitneyu(healthy_data, other_data,
                                          alternative='two-sided')
                stars = get_significance_stars(p)
                if stars != 'ns':
                    c_idx = plot_order.index(label)
                    other_pos = g_start + c_idx * (box_width + inner_gap)
                    ax.plot([healthy_pos, other_pos],
                            [current_y, current_y],
                            lw=1.0, c='#808080', clip_on=False)
                    ax.text((healthy_pos + other_pos) / 2,
                            current_y + y_range * 0.002,
                            stars, ha='center', va='bottom',
                            fontsize=8, color='#555555',
                            fontfamily='Arial', clip_on=False)
                    current_y += y_step
            except Exception:
                pass

    # ── x 轴标签 ──
    ax.set_xticks(group_centers)
    ax.set_xticklabels(categories, rotation=20, ha='right',
                       fontsize=12, fontfamily='Arial')

    # ── 图例（图底横向） ──
    handles = [
        mpatches.Patch(facecolor=category_colors.get(label, '#999999'),
                       edgecolor='black',
                       label='Healthy Control' if label == 'Healthy' else label,
                       linewidth=0.5)
        for label in plot_order
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.16),
              ncol=len(plot_order), fontsize=10, frameon=False,
              columnspacing=1.0, handletextpad=0.4,
              prop={'family': 'Arial'})

    ax.set_ylabel('S-F coupling', fontsize=14, fontfamily='Arial')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.tick_params(axis='y', labelsize=11)

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']:
        ax.spines[s].set_linewidth(1.2)

    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.6)
    total_x = (n_groups - 1) * (group_width + group_gap) + group_width
    ax.set_xlim(-group_gap / 2, total_x + group_gap / 2)

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight',
                    facecolor='white', edgecolor='none', dpi=300,
                    metadata={'Creator': 'Python matplotlib'})
    plt.close()


# 分组箱线图（参考图样式）
plot_grouped_network_boxplot(CSV_PATHS, CATEGORIES,
                             output_path='all_networks_grouped.pdf')

# 子图布局（原来的）
# plot_all_network_subplots(CSV_PATHS, CATEGORIES, output_path='all_networks_subplots.pdf')

# 绘制汇总图（Healthy, Normal Stroke, Aphasia）
# plot_summary_boxplot(CSV_PATHS, CATEGORIES, output_path='summary_boxplot.pdf')

# 保留原始函数调用（如需要）
# plot_all_network(CSV_PATHS, CATEGORIES, output_path='all_networks.pdf')
