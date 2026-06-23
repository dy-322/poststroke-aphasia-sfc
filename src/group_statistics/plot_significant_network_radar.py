import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# ========== 显著脑区数量雷达图 ==========
# 读取显著脑区数量CSV文件
csv_file = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\source\results\significance_analysis\network_significant_regions_count.csv"
df_significant = pd.read_csv(csv_file)

# 将空的aphasia_type填充为Non-aphasic Stroke
df_significant['aphasia_type'] = df_significant['aphasia_type'].fillna('Non-aphasic Stroke')

# 定义七个网络列和皮层下
network_columns_sig = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis", "Subcortex"]
num_vars_sig = len(network_columns_sig)

# 获取所有组别（过滤掉空值）
groups = df_significant['aphasia_type'].dropna().tolist()

# 定义颜色映射（与plot_all_network.py保持一致）
# plot_all_network.py中的颜色顺序：Global, Conduction, None, Anomic, Broca
aphasia_color_map = {
    "Global": "#7895C1",
    "Conduction": "#A8CBDF",
    "Non-aphasic Stroke":  "#808080",
    "Anomic": "#EF8B67",
    "Broca": "#E3625D"
}

# 为每个组别分配颜色
group_colors = {}
for group in groups:
    if group in aphasia_color_map:
        group_colors[group] = aphasia_color_map[group]
    else:
        # 如果组别不在映射中，使用默认颜色
        group_colors[group] = "#808080"
        print(f"警告：组别 '{group}' 未找到颜色映射，使用默认灰色")

# 准备所有组别的数据
data_sets_sig = {}
for group in groups:
    filtered_df = df_significant[df_significant['aphasia_type'] == group]
    if len(filtered_df) > 0:
        row = filtered_df.iloc[0]
        # 前7个是网络列，最后一个是皮层下
        data_list = [row[col] for col in network_columns_sig[:-1]]  # 前7个网络
        # 添加皮层下数据（使用subcortical_significant列）
        if 'subcortical_significant' in row:
            data_list.append(row['subcortical_significant'])
        else:
            data_list.append(0)  # 如果没有皮层下数据，设为0
        data_sets_sig[group] = data_list
    else:
        print(f"警告：未找到组别 '{group}' 的数据")

# 计算所有数据的最小值和最大值（全局范围，用于固定所有雷达图的显示范围）
all_values_sig = []
for data in data_sets_sig.values():
    all_values_sig.extend(data)
data_min_sig = min(all_values_sig)
data_max_sig = max(all_values_sig)
data_range_sig = data_max_sig - data_min_sig

# 设置半径范围
base_radius_sig = 20
max_radius_for_lines_sig = 65
radius_range_sig = max_radius_for_lines_sig - base_radius_sig

# 生成角度（从顶部开始，顺时针排列）
angles_sig = np.linspace(np.pi / 2, -1.5 * np.pi, num_vars_sig, endpoint=False)

# 为每个组别单独生成雷达图
for group in groups:
    # 创建图形（调整大小以适应图例）
    fig_sig, ax_sig = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 绘制参考圆圈
    circle_sig = plt.Circle(
        (0, 0), base_radius_sig, transform=ax_sig.transData._b,
        facecolor='white', edgecolor='black', linestyle='--',
        linewidth=1, alpha=1.0, zorder=1)
    ax_sig.add_patch(circle_sig)
    
    # 参考线：对应映射范围的20%位置
    twenty_percent_radius_sig = base_radius_sig + 0.2 * radius_range_sig
    twenty_percent_circle_sig = plt.Circle(
        (0, 0), twenty_percent_radius_sig, transform=ax_sig.transData._b,
        facecolor='none', edgecolor='black', linestyle='--',
        linewidth=1, alpha=1.0, zorder=1
    )
    ax_sig.add_patch(twenty_percent_circle_sig)
    theta_sig = np.linspace(0, 2 * np.pi, 100)
    ax_sig.plot(theta_sig, [max_radius_for_lines_sig] * len(theta_sig),
                color='black', linewidth=2.5, zorder=1)
    
    # 绘制角度线
    for angle in angles_sig:
        ax_sig.plot(
            [angle, angle],
            [base_radius_sig, max_radius_for_lines_sig],
            color='lightgray',
            linestyle='--',
            linewidth=0.8,
            zorder=0
        )
    
    # 绘制当前组别的数据
    data = data_sets_sig[group]
    color = group_colors[group]
    
    # 计算数据点半径：将数据值映射到base_radius到max_radius_for_lines（使用全局范围）
    line_radii = [base_radius_sig + (value - data_min_sig) / (data_max_sig - data_min_sig) * radius_range_sig 
                 for value in data]
    closed_angles = np.append(angles_sig, angles_sig[0])
    closed_radii = np.append(line_radii, line_radii[0])
    
    # 绘制线条和填充区域（带透明度填充）
    ax_sig.plot(closed_angles, closed_radii, color=color, linewidth=2.5, zorder=1, label=group)
    ax_sig.fill(closed_angles, closed_radii, color=color, alpha=0.3, zorder=1)
    ax_sig.scatter(angles_sig, line_radii, color=color, s=50, zorder=2)

    # 标注最大值和次大值的百分比
    sorted_indices = np.argsort(data)[::-1]
    top2_indices = sorted_indices[:2]
    total = sum(data) if sum(data) > 0 else 1
    for rank, idx in enumerate(top2_indices):
        angle = angles_sig[idx]
        radius = line_radii[idx]
        pct = data[idx] / total * 100
        offset = 5 if rank == 0 else -5
        ax_sig.text(
            angle, radius + offset, f'{pct:.1f}%',
            ha='center', va='center', fontsize=24,
            fontfamily='Arial', weight='bold', color=color, zorder=3)

    # 添加网络标签
    label_radius_category_sig = 90
    for i, (angle, label) in enumerate(zip(angles_sig, network_columns_sig)):
        angle_deg = (np.rad2deg(angle) + 360) % 360
        rotation = angle_deg
        if 0 < angle_deg <= 180:
            rotation -= 90
        if 180 < angle_deg < 360:
            rotation += 90
        ax_sig.text(
            angle, label_radius_category_sig, label,
            ha='center', va='center', fontsize=24,
            fontfamily='Arial', rotation=rotation,
            weight='bold', color='black')

    # 参考圆圈数值标注（标在右侧 angle=0 方向）
    # label_angle = 0
    # # 内圆：对应 data_min_sig
    # ax_sig.text(label_angle, base_radius_sig, f'{int(data_min_sig)}',
    #             ha='left', va='center', fontsize=24,
    #             fontfamily='Arial', color='gray', zorder=3)
    # # 20%圆：对应 data_min_sig + 0.2*(data_max_sig-data_min_sig)
    # val_20pct = data_min_sig + 0.2 * (data_max_sig - data_min_sig)
    # ax_sig.text(label_angle, twenty_percent_radius_sig, f'{val_20pct:.1f}',
    #             ha='left', va='center', fontsize=24,
    #             fontfamily='Arial', color='gray', zorder=3)
    # # 外圆：对应 data_max_sig
    # ax_sig.text(label_angle, max_radius_for_lines_sig, f'{int(data_max_sig)}',
    #             ha='left', va='center', fontsize=24,
    #             fontfamily='Arial', color='gray', zorder=3)

    # 隐藏极坐标轴线和网格
    ax_sig.set_ylim(0, 100)
    ax_sig.spines['polar'].set_visible(False)
    ax_sig.grid(False)
    ax_sig.set_yticklabels([])
    ax_sig.set_xticks([])

    # 生成图例（只显示当前组别）
    group_patch_sig = mpatches.Patch(color=group_colors[group], alpha=0.6, label=group)

    # 放置图例（调整位置和大小，避免超出画布）
    fig_sig.legend(
        handles=[group_patch_sig],
        loc='lower center', bbox_to_anchor=(0.5, 0.08),
        frameon=False, ncol=1,
        prop={'family': 'Arial', 'weight': 'bold', 'size': 8},
        columnspacing=1.2,
        handletextpad=0.5
    )
    
    # 调整画布边距，保存图片
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
    output_filename = f'network_significant_regions_radar_{group}.pdf'
    plt.savefig(output_filename, format='pdf', 
                bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    plt.close(fig_sig)
    
    print(f"已生成 {group} 类别的雷达图: {output_filename}")

print(f"\n所有显著脑区数量雷达图已生成完成！共生成 {len(groups)} 个雷达图")
print(f"所有雷达图使用统一的显示范围: [{data_min_sig}, {data_max_sig}]")
