#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑区不对称性综合分析
使用AI不对称性指标：AI = (L-R) / abs((L+R)/2)

包含以下分析：
1. 脑区级别折线图（6个失语症类型子图）- 展示相对于健康受试者的差值
2. 表面图展示（每个失语症类型的不对称性）
3. 网络层面柱状图（7个网络+皮层下）
4. Lesion层面双轴散点图（dorsal vs ventral）
5. Lesion volume相关性分析
"""

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['pdf.fonttype'] = 42

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
import os
import re
from typing import Dict, List, Tuple
import seaborn as sns
import tempfile
from neurocat.plotting import draw_and_save as ds

# ==================== 不对称性指标计算函数 ====================

# Relation和Cohen's d指标已删除，只保留AI指标


def compute_asymmetry_ai(sfc_values, pairs):
    """
    计算AI不对称性指标: AI = (L-R) / abs((L+R)/2)

    参数:
    - sfc_values: SFC值数组 (450,)
    - pairs: 左右配对列表 [(left_idx, right_idx), ...]

    返回:
    - ai_values: 每个脑区对的AI值列表
    - mean_ai: 平均AI值
    """
    ai_values = []

    for left_idx, right_idx in pairs:
        left_val = sfc_values[left_idx]
        right_val = sfc_values[right_idx]

        if not np.isnan(left_val) and not np.isnan(right_val):
            denominator = abs((left_val + right_val) / 2)
            if denominator > 1e-10:  # 避免除零
                ai = (left_val - right_val) / denominator
                ai_values.append(ai)

    if len(ai_values) == 0:
        return [], np.nan

    return ai_values, np.mean(ai_values)


def compute_roi_level_asymmetry(sfc_values, pairs):
    """
    计算每个脑区对的AI不对称性指标

    参数:
    - sfc_values: SFC值数组 (450,)
    - pairs: 左右配对列表

    返回:
    - asymmetry_values: 每个脑区对的AI值 (225,)
    """
    asymmetry_values = []

    for left_idx, right_idx in pairs:
        left_val = sfc_values[left_idx]
        right_val = sfc_values[right_idx]

        if np.isnan(left_val) or np.isnan(right_val):
            asymmetry_values.append(np.nan)
            continue

        denominator = abs((left_val + right_val) / 2)
        if denominator > 1e-10:
            ai = (left_val - right_val) / denominator
            asymmetry_values.append(ai)
        else:
            asymmetry_values.append(np.nan)

    return np.array(asymmetry_values)



# ==================== 数据加载函数 ====================

def get_hemisphere_pairs():
    """获取左右半球的配对索引"""
    pairs = []
    # 皮层下区域配对（25对）
    for i in range(25):
        left_idx = i + 25
        right_idx = i
        pairs.append((left_idx, right_idx))
    # 皮层区域配对（200对）
    for i in range(200):
        left_idx = i + 50
        right_idx = i + 250
        pairs.append((left_idx, right_idx))
    return pairs


def load_sfc_file(filepath, roi_number=450):
    """加载单个受试者的SFC文件"""
    try:
        vals = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() and '\t' in line:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        try:
                            int(parts[0])
                            val = float(parts[1])
                            vals.append(val)
                        except ValueError:
                            if parts[1].strip().lower() == 'nan':
                                vals.append(np.nan)
        if len(vals) == roi_number:
            return np.array(vals)
        else:
            return None
    except Exception as e:
        print(f"加载文件失败 {filepath}: {e}")
        return None


def load_subject_categories(csv_file_path):
    """从TSV文件读取受试者失语类别映射"""
    try:
        df = pd.read_csv(csv_file_path, sep='\t')
        required_cols = {'participant_id', 'wab_type'}
        missing_cols = required_cols.difference(df.columns)
        if missing_cols:
            print(f"数据文件缺少必要列: {missing_cols}")
            return {}
        
        subject_categories = {}
        for _, row in df.iterrows():
            participant_id = str(row['participant_id'])
            aphasia_type = row['wab_type']
            
            if not isinstance(participant_id, str):
                continue
            
            participant_id = participant_id.strip()
            if not participant_id:
                continue
            
            if pd.isna(aphasia_type) or str(aphasia_type).strip() == '':
                aphasia_type = 'None'
            else:
                aphasia_type = str(aphasia_type).strip()
            
            if aphasia_type in {"TranscorticalMotor", "Wernicke"}:
                continue
            
            subject_categories[participant_id] = aphasia_type
        
        print(f"从TSV成功解析 {len(subject_categories)} 个受试者类别映射")
        return subject_categories
    
    except Exception as e:
        print(f"读取TSV时发生错误: {str(e)}")
        return {}


def parse_network_from_label(label_file_path):
    """解析标签文件，提取每个ROI的网络归属和半球信息"""
    roi_info = []
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
    
    current_name = None
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if i % 2 == 0:
            current_name = line
        else:
            parts = line.split()
            if len(parts) >= 1:
                roi_index = int(parts[0])
                network = 'Subcortical'
                hemisphere = None
                if current_name:
                    if '-lh' in current_name.lower():
                        hemisphere = 'L'
                    elif '-rh' in current_name.lower():
                        hemisphere = 'R'
                    if '7Networks' in current_name:
                        if '_LH_' in current_name:
                            hemisphere = 'L'
                        elif '_RH_' in current_name:
                            hemisphere = 'R'
                        match = re.search(r'7Networks_[LR]H_([A-Za-z]+)(?:_|$)', current_name)
                        if match:
                            network = match.group(1)
                roi_info.append({
                    'index': roi_index,
                    'name': current_name,
                    'network': network,
                    'hemisphere': hemisphere
                })
    return roi_info


def get_network_hemisphere_indices(roi_info):
    """获取每个网络的左右半球ROI索引"""
    network_indices = {}
    for roi in roi_info:
        network = roi['network']
        hemisphere = roi['hemisphere']
        idx = roi['index'] - 1
        if network not in network_indices:
            network_indices[network] = {'L': [], 'R': []}
        if hemisphere == 'L':
            network_indices[network]['L'].append(idx)
        elif hemisphere == 'R':
            network_indices[network]['R'].append(idx)
    return network_indices


# ==================== 绘图函数 ====================

def plot_roi_line_charts(category_data, pairs, roi_info, metric, output_path):
    """
    绘制脑区级别的折线图（失语症类型子图，不包括健康受试者）
    展示相对于健康受试者的差值
    按照网络分组，同一网络的ROI用该网络的平均值代替
    布局：1行5列，x/y轴交换，添加网络颜色标注
    """
    # 检查是否有健康受试者数据
    if 'Healthy' not in category_data:
        print("警告：没有健康受试者数据，无法计算差值")
        return

    # 构建ROI到网络的映射
    roi_to_network = {}
    for roi in roi_info:
        idx = roi['index'] - 1
        network = roi['network']
        hemisphere = roi['hemisphere']
        roi_to_network[idx] = {'network': network, 'hemisphere': hemisphere}

    # 为每个pair分配网络标签
    pair_networks = []
    for left_idx, right_idx in pairs:
        if left_idx in roi_to_network:
            network = roi_to_network[left_idx]['network']
        else:
            network = 'Unknown'
        pair_networks.append(network)

    # 计算健康受试者每个网络的平均SFC
    healthy_data = category_data['Healthy']
    healthy_network_left = {}
    healthy_network_right = {}

    for network in set(pair_networks):
        network_left_vals = []
        network_right_vals = []
        for pair_idx, (left_idx, right_idx) in enumerate(pairs):
            if pair_networks[pair_idx] == network:
                for sfc in healthy_data:
                    if not np.isnan(sfc[left_idx]):
                        network_left_vals.append(sfc[left_idx])
                    if not np.isnan(sfc[right_idx]):
                        network_right_vals.append(sfc[right_idx])

        healthy_network_left[network] = np.mean(network_left_vals) if network_left_vals else np.nan
        healthy_network_right[network] = np.mean(network_right_vals) if network_right_vals else np.nan

    # 准备绘图顺序（排除Healthy）
    plot_order = []
    if 'Normal Stroke' in category_data:
        plot_order.append('Normal Stroke')
    for label in category_data.keys():
        if label not in ['Healthy', 'Normal Stroke']:
            plot_order.append(label)

    # 只取前5个类型
    plot_order = plot_order[:5]
    n_groups = len(plot_order)

    # 网络颜色映射 - 前50个ROI用灰色，后面7个网络用指定颜色
    network_colors = {
        'Subcortical': '#999999',  # 灰色用于前50个roi index
        'Cont': '#1d52a1',
        'Default': '#716db2',
        'DorsAttn': '#65c8cc',
        'Limbic': '#72c15a',
        'SalVentAttn': '#f0e94b',
        'SomMot': '#f3793b',
        'Vis': '#c6519f',
        'Unknown': '#f3878d'
    }

    fig, axes = plt.subplots(1, n_groups, figsize=(3*n_groups, 6))
    if n_groups == 1:
        axes = [axes]

    for idx, label in enumerate(plot_order):
        ax = axes[idx]
        if label not in category_data:
            continue

        sfc_data_list = category_data[label]

        # 计算该类别每个网络的平均SFC
        category_network_left = {}
        category_network_right = {}

        for network in set(pair_networks):
            network_left_vals = []
            network_right_vals = []
            for pair_idx, (left_idx, right_idx) in enumerate(pairs):
                if pair_networks[pair_idx] == network:
                    for sfc in sfc_data_list:
                        if not np.isnan(sfc[left_idx]):
                            network_left_vals.append(sfc[left_idx])
                        if not np.isnan(sfc[right_idx]):
                            network_right_vals.append(sfc[right_idx])

            category_network_left[network] = np.mean(network_left_vals) if network_left_vals else np.nan
            category_network_right[network] = np.mean(network_right_vals) if network_right_vals else np.nan

        # 计算每个pair的差值（不使用网络平均）
        left_diff = []
        right_diff = []

        for pair_idx, (left_idx, right_idx) in enumerate(pairs):
            # 计算该类别的平均值
            left_vals = [sfc[left_idx] for sfc in sfc_data_list if not np.isnan(sfc[left_idx])]
            right_vals = [sfc[right_idx] for sfc in sfc_data_list if not np.isnan(sfc[right_idx])]

            left_mean = np.mean(left_vals) if left_vals else np.nan
            right_mean = np.mean(right_vals) if right_vals else np.nan

            # 计算健康受试者的平均值
            healthy_left_vals = [sfc[left_idx] for sfc in healthy_data if not np.isnan(sfc[left_idx])]
            healthy_right_vals = [sfc[right_idx] for sfc in healthy_data if not np.isnan(sfc[right_idx])]

            healthy_left_mean = np.mean(healthy_left_vals) if healthy_left_vals else np.nan
            healthy_right_mean = np.mean(healthy_right_vals) if healthy_right_vals else np.nan

            left_diff.append(left_mean - healthy_left_mean)
            right_diff.append(right_mean - healthy_right_mean)

        # 十点移动平均
        def moving_average(data, window=10):
            data_array = np.array(data)
            smoothed = np.convolve(data_array, np.ones(window)/window, mode='same')
            return smoothed

        left_diff_smoothed = moving_average(left_diff, window=10)
        right_diff_smoothed = moving_average(right_diff, window=10)

        roi_indices = np.arange(len(pairs))

        # 绘制折线（x/y轴交换），使用更深的颜色，添加点标记
        ax.plot(left_diff_smoothed, roi_indices, '-o', color='#C0392B', linewidth=1.2,
                label='Left Hemisphere', alpha=0.9, markersize=2, markeredgewidth=0)
        ax.plot(right_diff_smoothed, roi_indices, '-o', color='#2874A6', linewidth=1.2,
                label='Right Hemisphere', alpha=0.9, markersize=2, markeredgewidth=0)

        # 绘制y轴颜色条代替网络标签
        # 找到每个网络的范围
        network_ranges = {}
        current_network = None
        start_idx = 0

        for i, network in enumerate(pair_networks + [None]):
            if network != current_network:
                if current_network is not None:
                    network_ranges[current_network] = (start_idx, i - 1)
                current_network = network
                start_idx = i

        # 在y轴左侧绘制颜色条
        for network, (start, end) in network_ranges.items():
            color = network_colors.get(network, '#CCCCCC')
            # 在y轴范围内绘制颜色块
            ax.axhspan(start, end, xmin=-0.02, xmax=0, color=color,
                      alpha=0.8, clip_on=False, linewidth=0)

        # 移除y轴刻度和标签
        ax.set_yticks([])
        ax.set_ylabel('')

        ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title(f'{label}\n(n={len(sfc_data_list)})', fontsize=10, fontweight='bold')
        ax.set_xlabel('SFC Difference', fontsize=9)
        ax.legend(loc='best', fontsize=8)
        ax.tick_params(labelsize=8)

        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"脑区折线图已保存: {output_path}")


def plot_surface_maps(category_data, pairs, metric, output_dir):
    """
    绘制脑表面图展示（每个失语症类型的AI不对称性，包括健康受试者）
    展示皮层和皮层下区域，所有类型使用统一的color range（关于0对称）
    """
    import nibabel as nib
    import matplotlib.image as mpimg
    from matplotlib import gridspec
    from matplotlib.colorbar import ColorbarBase
    from nilearn import datasets, plotting

    # 准备绘图顺序（包括Healthy）
    plot_order = []
    if 'Healthy' in category_data:
        plot_order.append('Healthy')
    if 'Normal Stroke' in category_data:
        plot_order.append('Normal Stroke')
    for label in category_data.keys():
        if label not in ['Healthy', 'Normal Stroke']:
            plot_order.append(label)

    # Atlas路径
    atlas_path = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_MNI152NLin6Asym_1mm.nii.gz"

    # ========== 第一步：计算所有类别的全局color range ==========
    print("\n计算全局color range...")
    all_valid_ai = []
    
    for label in plot_order:
        if label not in category_data:
            continue

        sfc_data_list = category_data[label]

        # 计算每个ROI对的平均AI值
        roi_ai_values = []
        for pair_idx, (left_idx, right_idx) in enumerate(pairs):
            ai_values_for_pair = []
            for sfc in sfc_data_list:
                left_val = sfc[left_idx]
                right_val = sfc[right_idx]
                if not np.isnan(left_val) and not np.isnan(right_val):
                    denominator = abs((left_val + right_val) / 2)
                    if denominator > 1e-10:
                        ai = (left_val - right_val) / denominator
                        ai_values_for_pair.append(ai)

            if ai_values_for_pair:
                roi_ai_values.append(np.mean(ai_values_for_pair))
            else:
                roi_ai_values.append(np.nan)

        # 分离皮层下和皮层区域
        subcortical_ai = np.array(roi_ai_values[:25])
        cortical_ai = np.array(roi_ai_values[25:])

        # 收集所有有效的AI值
        valid_ai = np.concatenate([cortical_ai[~np.isnan(cortical_ai)],
                                   subcortical_ai[~np.isnan(subcortical_ai)]])
        if len(valid_ai) > 0:
            all_valid_ai.extend(valid_ai)

    # 计算全局color range（关于0对称）
    if len(all_valid_ai) == 0:
        print("警告：没有有效的AI值")
        return
    
    global_ai_abs_max = max(abs(np.min(all_valid_ai)), abs(np.max(all_valid_ai)))
    global_color_range = (-global_ai_abs_max, global_ai_abs_max)
    print(f"全局AI范围: {global_color_range}")

    # ========== 第二步：为每个失语症类型绘制图像（使用全局color range）==========
    for label in plot_order:
        if label not in category_data:
            continue

        sfc_data_list = category_data[label]

        # 计算每个ROI对的平均AI值
        roi_ai_values = []
        for pair_idx, (left_idx, right_idx) in enumerate(pairs):
            ai_values_for_pair = []
            for sfc in sfc_data_list:
                left_val = sfc[left_idx]
                right_val = sfc[right_idx]
                if not np.isnan(left_val) and not np.isnan(right_val):
                    denominator = abs((left_val + right_val) / 2)
                    if denominator > 1e-10:
                        ai = (left_val - right_val) / denominator
                        ai_values_for_pair.append(ai)

            if ai_values_for_pair:
                roi_ai_values.append(np.mean(ai_values_for_pair))
            else:
                roi_ai_values.append(np.nan)

        # 分离皮层下和皮层区域
        # pairs结构：前25对是皮层下，后200对是皮层
        subcortical_ai = np.array(roi_ai_values[:25])  # 前25个皮层下区域
        cortical_ai = np.array(roi_ai_values[25:])  # 后200个皮层区域

        if len(cortical_ai) != 200:
            print(f"警告：{label} 的皮层AI值数量不正确: {len(cortical_ai)}")
            continue

        # 将200个AI值扩展为400个值（左右半球各200个，都使用相同的AI值）
        # 这样ds()函数可以正确显示左右两个半球
        cortical_ai_full = np.concatenate([cortical_ai, cortical_ai])  # 400个值

        # 使用全局color range
        color_range = global_color_range
        print(f"{label} 使用全局AI范围: {color_range}")

        try:
            temp_dir = tempfile.gettempdir()

            # 1. 绘制皮层表面图
            temp_surface = os.path.join(temp_dir, f'temp_surface_{label}.png')
            ds(
                layer1=cortical_ai_full,  # 使用400个值（左右半球）
                colorbar='PRGn',
                color_range=color_range,
                fig_name=temp_surface.replace('.png', ''),
                layout='row',
                outline=False,  # 关闭边界线
                trim=True,
                tp='fslr',
                mesh='inflated',  # 使用inflated而不是veryininflateded，减少膨胀
                save_png_only=True,
                display=False,
                size=(6000, 1000)
            )

            # 2. 绘制皮层下图像
            # 将subcortical_ai映射到atlas的前50个ROI
            atlas_img = nib.load(atlas_path)
            atlas_data = atlas_img.get_fdata().astype(int)
            roi_vol = np.full(atlas_data.shape, np.nan, dtype=float)

            # subcortical_ai有25个值，对应25对ROI
            # 需要映射到atlas的前50个标签（1-50）
            for i in range(25):
                # 左半球（标签26-50）对应subcortical_ai的左侧
                # 右半球（标签1-25）对应subcortical_ai的右侧
                # 但AI是(L-R)/mean，所以每对用同一个AI值
                roi_vol[atlas_data == (i + 1)] = subcortical_ai[i]  # 右半球
                roi_vol[atlas_data == (i + 26)] = subcortical_ai[i]  # 左半球

            roi_img = nib.Nifti1Image(roi_vol, atlas_img.affine, atlas_img.header)

            bg_img = datasets.load_mni152_template(resolution=1)
            bg_data = bg_img.get_fdata()
            fig_nilearn = plt.figure(figsize=(40, 8))

            bg_img_final = nib.Nifti1Image(bg_data, bg_img.affine, bg_img.header)
            nilearn_disp = plotting.plot_stat_map(
                roi_img,
                bg_img=bg_img_final,
                threshold=0.0,
                cmap='PRGn',
                vmax=color_range[1],
                vmin=color_range[0],
                display_mode='z',
                cut_coords=[-25, -20, -10, 0, 10, 20, 22],
                draw_cross=False,
                colorbar=False,
                figure=fig_nilearn,
                black_bg=False,
                annotate=False,
            )

            temp_nilearn = os.path.join(temp_dir, f'temp_nilearn_{label}.png')
            fig_nilearn.savefig(temp_nilearn, bbox_inches='tight', dpi=300, pad_inches=0)
            plt.close(fig_nilearn)

            # 3. 读取并修剪图像
            img_surface = mpimg.imread(temp_surface)
            img_nilearn = mpimg.imread(temp_nilearn)

            def trim_white_edges(img_array):
                mask = np.any(img_array < 1, axis=-1)
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                row_indices = np.where(rows)[0]
                col_indices = np.where(cols)[0]
                if len(row_indices) == 0 or len(col_indices) == 0:
                    return img_array
                cropped_img = img_array[row_indices[0]:row_indices[-1] + 1,
                              col_indices[0]:col_indices[-1] + 1]
                return np.ascontiguousarray(cropped_img)

            img_surface_trimmed = trim_white_edges(img_surface)
            img_nilearn_trimmed = trim_white_edges(img_nilearn)

            # 4. 合并图像并添加colorbar
            fig_merged = plt.figure(figsize=(70, 25))
            gs = gridspec.GridSpec(2, 2, figure=fig_merged,
                                   width_ratios=[97, 3],
                                   hspace=0.1, wspace=0.05)

            ax1 = fig_merged.add_subplot(gs[0, 0])
            ax1.imshow(img_surface_trimmed)
            ax1.axis('off')
            ax1.set_title(f'{label} - Cortical (n={len(sfc_data_list)})',
                         fontsize=60, fontweight='bold', pad=20)

            ax2 = fig_merged.add_subplot(gs[1, 0])
            ax2.imshow(img_nilearn_trimmed)
            ax2.axis('off')
            ax2.set_title(f'{label} - Subcortical',
                         fontsize=60, fontweight='bold', pad=20)

            # 添加colorbar
            ax_cbar = fig_merged.add_subplot(gs[:, 1])
            norm = plt.Normalize(vmin=color_range[0], vmax=color_range[1])
            cbar = ColorbarBase(ax_cbar, cmap='PRGn', norm=norm, orientation='vertical')
            cbar.set_label('AI Asymmetry', rotation=270, labelpad=30, fontsize=70, fontname='Arial')
            ax_cbar.tick_params(labelsize=50, direction='in', length=8, width=80)
            for tick_label in ax_cbar.get_yticklabels():
                tick_label.set_fontname('Arial')

            plt.tight_layout()

            output_path = os.path.join(output_dir, f'surface_map_{label}_{metric}.pdf')
            fig_merged.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
            plt.close(fig_merged)

            # 删除临时文件
            if os.path.exists(temp_surface):
                os.remove(temp_surface)
            if os.path.exists(temp_nilearn):
                os.remove(temp_nilearn)

            print(f"表面图已保存: {output_path}")

        except Exception as e:
            print(f"绘制{label}的表面图时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"所有表面图绘制完成")


def plot_network_bar_charts(category_network_asymmetry, metric, output_path):
    """
    绘制网络层面的柱状图 - 所有失语症类型合并到单一坐标轴
    X轴分组：Anomic -> Broca -> Conduction -> Global -> Normal Stroke -> Healthy
    每组内8个网络柱，按网络颜色着色
    图宽180mm，高度为原始比例的一半；图例横向排列于图底部
    """
    networks = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis", "Subcortical"]

    # 网络颜色映射 - 与折线图y轴颜色一致
    network_colors = {
        'Subcortical': '#999999',
        'Cont': '#1d52a1',
        'Default': '#716db2',
        'DorsAttn': '#65c8cc',
        'Limbic': '#72c15a',
        'SalVentAttn': '#f0e94b',
        'SomMot': '#f3793b',
        'Vis': '#c6519f'
    }

    # 按指定顺序排列，仅保留数据中存在的分组
    desired_order = ['Anomic', 'Broca', 'Conduction', 'Global', 'Normal Stroke', 'Healthy']
    plot_order = [g for g in desired_order if g in category_network_asymmetry]
    # 追加数据中有但不在预设顺序里的分组
    for label in category_network_asymmetry.keys():
        if label not in plot_order:
            plot_order.append(label)

    n_networks = len(networks)

    # 每根柱宽度与分组间距
    bar_width = 0.09
    group_gap = 0.35   # 组间额外间隔

    # 图宽180mm；高度为原始比例的一半
    fig_w_in = 180 / 25.4               # ≈ 7.09 英寸
    fig_h_in = fig_w_in * (7 / 12) / 2  # 原始比例再减半 ≈ 2.07 英寸
    fig, ax = plt.subplots(1, 1, figsize=(fig_w_in, fig_h_in))

    group_centers = []
    for i, label in enumerate(plot_order):
        if label not in category_network_asymmetry:
            continue

        network_data = category_network_asymmetry[label]
        group_start = i * (n_networks * bar_width + group_gap)

        for j, network in enumerate(networks):
            if network in network_data and len(network_data[network]) > 0:
                mean_val = np.mean(network_data[network])
            else:
                mean_val = 0
            x_pos = group_start + j * bar_width
            color = network_colors.get(network, '#999999')
            ax.bar(x_pos, mean_val, width=bar_width * 0.88, color=color, alpha=0.85)

        center = group_start + (n_networks - 1) * bar_width / 2
        group_centers.append(center)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(plot_order, fontsize=8)
    ax.set_title('Network-Level SFC Asymmetry Index Across Aphasia Subtypes',
                 fontsize=9, fontweight='bold', pad=4)
    ax.set_ylabel(f'Asymmetry Index ({metric.upper()})', fontsize=8, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylim(-0.15, 0.15)
    ax.tick_params(labelsize=8)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # 网络图例 - 横向排列于图底部
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=network_colors[net], alpha=0.85, label=net)
        for net in networks
    ]
    fig.legend(handles=legend_elements, fontsize=7, loc='lower center',
               ncol=len(networks), frameon=False,
               handlelength=1.0, handleheight=0.8,
               bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)   # 为底部图例留出空间
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"网络柱状图已保存: {output_path}")


def plot_lesion_dorsal_ventral_scatter(lesion_data, metric, output_path):
    """
    绘制lesion层面的双轴散点图
    X轴：Dorsal Stream的不对称指标
    Y轴：Ventral Stream的不对称指标
    点：不同的失语症亚型着色（包括健康人）
    """
    aphasia_colors = {
        'Healthy': '#808080',
        'Normal Stroke': '#7895C1',
        'Anomic': '#A8CBDF',
        'Broca': '#992224',
        'Conduction': '#EF8B67',
        'Global': '#E3625D',
        'TranscorticalSensory': '#B54764'
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    for aphasia_type, data in lesion_data.items():
        if 'dorsal_asym' not in data or 'ventral_asym' not in data:
            continue

        dorsal_vals = data['dorsal_asym']
        ventral_vals = data['ventral_asym']

        color = aphasia_colors.get(aphasia_type, '#999999')
        ax.scatter(dorsal_vals, ventral_vals, c=color, s=80, alpha=0.7,
                   edgecolors='black', linewidth=0.5, label=aphasia_type)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel(f'Dorsal Stream Asymmetry ({metric})', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Ventral Stream Asymmetry ({metric})', fontsize=13, fontweight='bold')
    ax.set_title(f'Dorsal vs Ventral Stream Asymmetry ({metric})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Lesion双轴散点图已保存: {output_path}")


def plot_lesion_volume_correlation(lesion_volume_data, metric, output_dir):
    """
    绘制lesion占比相关性分析散点图（两张图：dorsal和ventral）
    X轴：Lesion占比
    Y轴：不对称指标
    着色：按失语症亚型着色
    """
    aphasia_colors = {
        'Healthy': '#808080',
        'Normal Stroke': '#7895C1',
        'Anomic': '#A8CBDF',
        'Broca': '#992224',
        'Conduction': '#EF8B67',
        'Global': '#E3625D',
        'TranscorticalSensory': '#B54764'
    }

    # 绘制两张图
    for lesion_type in ['dorsal', 'ventral']:
        fig, ax = plt.subplots(figsize=(10, 8))

        for aphasia_type, data in lesion_volume_data.items():
            ratio_key = f'{lesion_type}_ratio'
            if ratio_key not in data or 'asymmetry' not in data:
                continue

            ratios = data[ratio_key]
            asymmetries = data['asymmetry']
            
            # 过滤掉无效数据
            valid_indices = [(i, r, a) for i, (r, a) in enumerate(zip(ratios, asymmetries)) 
                           if not (np.isnan(r) or np.isnan(a))]
            
            if not valid_indices:
                continue
            
            valid_ratios = [r for _, r, _ in valid_indices]
            valid_asymmetries = [a for _, _, a in valid_indices]

            color = aphasia_colors.get(aphasia_type, '#999999')
            ax.scatter(valid_ratios, valid_asymmetries, c=color, s=80, alpha=0.7,
                       edgecolors='black', linewidth=0.5, label=aphasia_type)

        # 添加回归线
        all_ratios = []
        all_asymmetries = []
        for data in lesion_volume_data.values():
            ratio_key = f'{lesion_type}_ratio'
            if ratio_key in data and 'asymmetry' in data:
                for r, a in zip(data[ratio_key], data['asymmetry']):
                    if not (np.isnan(r) or np.isnan(a)):
                        all_ratios.append(r)
                        all_asymmetries.append(a)

        if len(all_ratios) > 2:
            # 计算相关性
            r, p_val = stats.pearsonr(all_ratios, all_asymmetries)
            
            # 绘制回归线
            z = np.polyfit(all_ratios, all_asymmetries, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(all_ratios), max(all_ratios), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label='Regression')

            # 显示相关性信息
            ax.text(0.05, 0.95, f'r={r:.3f}, p={p_val:.4f}\nn={len(all_ratios)}',
                    transform=ax.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            print(f"{lesion_type.capitalize()} lesion相关性: r={r:.3f}, p={p_val:.4f}, n={len(all_ratios)}")

        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(f'{lesion_type.capitalize()} Lesion Percentage', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'Asymmetry Index ({metric})', fontsize=13, fontweight='bold')
        ax.set_title(f'{lesion_type.capitalize()} Lesion Percentage vs Asymmetry ({metric})',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        output_path = output_dir.replace('.pdf', f'_{lesion_type}.pdf')
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Lesion {lesion_type} 相关性图已保存: {output_path}")


# ==================== 数据处理和主函数 ====================

def load_all_subjects_data(sfc_dir, subject_categories, pairs, roi_number=450):
    """加载所有受试者的SFC数据"""
    category_data = {}
    
    for fname in os.listdir(sfc_dir):
        if not fname.endswith("_sfc_values_pca.txt"):
            continue
        
        filepath = os.path.join(sfc_dir, fname)
        stem = fname[:-len("_sfc_values_pca.txt")]
        
        csv_key = None
        tokens = stem.replace('_', '-').split('-')
        for token in tokens:
            if token.startswith('M') and len(token) > 1 and token[1:].isdigit():
                csv_key = f"sub-{token}"
                break
        
        is_healthy = (csv_key is None)
        
        if is_healthy:
            aphasia_type = 'Healthy'
        else:
            aphasia_type = subject_categories.get(csv_key)
            if aphasia_type is None:
                continue
            if aphasia_type == 'None':
                aphasia_type = 'Normal Stroke'
        
        sfc_values = load_sfc_file(filepath, roi_number)
        if sfc_values is None:
            continue
        
        if aphasia_type not in category_data:
            category_data[aphasia_type] = []
        category_data[aphasia_type].append(sfc_values)
    
    print(f"\n加载了 {sum(len(v) for v in category_data.values())} 个受试者的数据")
    for aphasia_type, data_list in category_data.items():
        print(f"  {aphasia_type}: {len(data_list)} 个受试者")
    
    return category_data


def compute_all_asymmetries(category_data, pairs):
    """计算所有受试者的AI不对称性指标"""
    category_asymmetry = {}

    for aphasia_type, sfc_list in category_data.items():
        all_asymmetries = []

        for sfc_values in sfc_list:
            _, mean_ai = compute_asymmetry_ai(sfc_values, pairs)
            if not np.isnan(mean_ai):
                all_asymmetries.append(mean_ai)

        category_asymmetry[aphasia_type] = np.array(all_asymmetries)

    return category_asymmetry


def compute_network_asymmetries(category_data, network_indices, pairs):
    """计算网络层面的AI不对称性"""
    category_network_asymmetry = {}

    for aphasia_type, sfc_list in category_data.items():
        network_asymmetry = {net: [] for net in network_indices.keys()}

        for sfc_values in sfc_list:
            for network, indices in network_indices.items():
                left_indices = indices['L']
                right_indices = indices['R']

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                # 创建网络内的配对
                network_pairs = [(l, r) for l, r in zip(left_indices, right_indices)]

                _, mean_ai = compute_asymmetry_ai(sfc_values, network_pairs)
                if not np.isnan(mean_ai):
                    network_asymmetry[network].append(mean_ai)

        category_network_asymmetry[aphasia_type] = network_asymmetry

    return category_network_asymmetry


def extract_network_rois(template_info_file, network_name):
    """从disconnectome_mediation_analysis.py提取dorsal和ventral ROI"""
    if not os.path.exists(template_info_file):
        raise FileNotFoundError(f"Template file not found: {template_info_file}")
    
    with open(template_info_file, 'r') as f:
        lines = f.readlines()
    
    network_rois = []
    
    if network_name == 'DorsAttn':
        target_patterns = ['DorsAttn', 'PUT-VA-lh', 'PUT-DA-lh', 'PUT-VP-lh', 'PUT-DP-lh',
                         'CAU-VA-lh', 'CAU-DA-lh', 'CAU-body-lh', 'CAU-tail-lh']
    elif network_name == 'SalVentAttn':
        target_patterns = ['HIP-head-lh', 'HIP-body-lh', 'HIP-tail-lh',
                         'THA-VPm-lh', 'THA-VPl-lh', 'THA-VAi-lh', 'THA-VAs-lh',
                         'THA-DAm-lh', 'THA-DAl-lh', 'THA-DP-lh']
    else:
        raise ValueError(f"Unknown network: {network_name}")
    
    for idx, line in enumerate(lines):
        line = line.strip()
        for pattern in target_patterns:
            if (pattern in line and '_LH_' in line) or (pattern in line and '-lh' in line):
                network_rois.append(idx)
                break
    
    return network_rois


def load_lesion_data(lesion_dir, sfc_dir, subject_categories, template_info_file, pairs):
    """加载lesion数据用于dorsal/ventral分析（使用AI指标，包括健康人）"""
    dorsal_rois = extract_network_rois(template_info_file, 'DorsAttn')
    ventral_rois = extract_network_rois(template_info_file, 'SalVentAttn')

    lesion_data = {}

    for fname in os.listdir(sfc_dir):
        if not fname.endswith("_sfc_values_pca.txt"):
            continue

        filepath = os.path.join(sfc_dir, fname)
        stem = fname[:-len("_sfc_values_pca.txt")]

        csv_key = None
        subject_id = None
        tokens = stem.replace('_', '-').split('-')
        for token in tokens:
            if token.startswith('M') and len(token) > 1 and token[1:].isdigit():
                csv_key = f"sub-{token}"
                subject_id = token
                break

        # 判断是否为健康受试者
        is_healthy = (csv_key is None)

        if is_healthy:
            aphasia_type = 'Healthy'
        else:
            aphasia_type = subject_categories.get(csv_key)
            if aphasia_type is None:
                continue
            if aphasia_type == 'None':
                aphasia_type = 'Normal Stroke'

        sfc_values = load_sfc_file(filepath, 450)
        if sfc_values is None:
            continue

        # 计算dorsal和ventral的不对称性（使用AI）
        dorsal_pairs = [(l, l+200) for l in dorsal_rois if l < 250]
        ventral_pairs = [(l, l+200) for l in ventral_rois if l < 250]

        _, dorsal_asym = compute_asymmetry_ai(sfc_values, dorsal_pairs)
        _, ventral_asym = compute_asymmetry_ai(sfc_values, ventral_pairs)

        if np.isnan(dorsal_asym) or np.isnan(ventral_asym):
            continue

        # 加载lesion volume（健康人的lesion volume为0）
        lesion_volume = 0
        if not is_healthy and subject_id:
            possible_ids = [subject_id, f'M{subject_id}']
            for test_id in possible_ids:
                subject_dir = os.path.join(lesion_dir, test_id)
                all_regions_file = os.path.join(subject_dir, f"{test_id}_all_regions.npy")
                if os.path.exists(all_regions_file):
                    try:
                        region_masks = np.load(all_regions_file, allow_pickle=True).item()
                        lesion_data_arr = region_masks.get('lesion')
                        if lesion_data_arr is not None:
                            lesion_volume = np.sum(lesion_data_arr > 0)
                            break
                    except:
                        pass

        if aphasia_type not in lesion_data:
            lesion_data[aphasia_type] = {
                'dorsal_asym': [],
                'ventral_asym': [],
                'lesion_volume': []
            }

        lesion_data[aphasia_type]['dorsal_asym'].append(dorsal_asym)
        lesion_data[aphasia_type]['ventral_asym'].append(ventral_asym)
        lesion_data[aphasia_type]['lesion_volume'].append(lesion_volume)

    return lesion_data


def calculate_roi_lesion_percentage(lesion_mask, atlas_img, roi_indices):
    """
    计算指定ROI区域的lesion占比
    
    参数:
    - lesion_mask: lesion的二值mask (3D array)
    - atlas_img: atlas的nifti图像
    - roi_indices: 要计算的ROI索引列表（从0开始，对应atlas中的标签）
    
    返回:
    - lesion_percentage: lesion在这些ROI中的占比（lesion体素数 / ROI总体素数）
    """
    import nibabel as nib
    
    if lesion_mask is None or atlas_img is None:
        return 0.0
    
    # 加载atlas数据
    if isinstance(atlas_img, str):
        atlas_data = nib.load(atlas_img).get_fdata().astype(int)
    else:
        atlas_data = atlas_img.get_fdata().astype(int)
    
    # 创建ROI mask（包含所有指定的ROI）
    roi_mask = np.zeros_like(atlas_data, dtype=bool)
    for roi_idx in roi_indices:
        # atlas标签从1开始，roi_idx从0开始，所以需要+1
        # 但是对于左半球ROI（索引50-249），对应atlas标签51-250
        # 对于右半球ROI（索引250-449），对应atlas标签251-450
        # 对于皮层下ROI（索引0-49），对应atlas标签1-50
        atlas_label = roi_idx + 1
        roi_mask |= (atlas_data == atlas_label)
    
    # 计算ROI区域的总体素数
    roi_total_voxels = np.sum(roi_mask)
    
    if roi_total_voxels == 0:
        return 0.0
    
    # 计算lesion与ROI重叠的体素数
    lesion_in_roi = np.sum(lesion_mask & roi_mask)
    
    # 计算占比
    lesion_percentage = lesion_in_roi / roi_total_voxels
    
    return lesion_percentage


def prepare_lesion_volume_data(category_data, lesion_dir, subject_categories,
                                sfc_dir, pairs, template_info_file):
    """准备lesion volume相关性数据（使用AI指标，计算dorsal和ventral的lesion占比）"""
    import nibabel as nib
    
    # 获取dorsal和ventral的ROI索引
    dorsal_rois = extract_network_rois(template_info_file, 'DorsAttn')
    ventral_rois = extract_network_rois(template_info_file, 'SalVentAttn')
    
    print(f"Dorsal ROIs数量: {len(dorsal_rois)}")
    print(f"Ventral ROIs数量: {len(ventral_rois)}")
    
    # 加载atlas
    atlas_path = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_MNI152NLin6Asym_1mm.nii.gz"
    if not os.path.exists(atlas_path):
        print(f"警告: Atlas文件不存在: {atlas_path}")
        return {}
    
    atlas_img = nib.load(atlas_path)
    print(f"成功加载atlas: {atlas_path}")

    lesion_volume_data = {}

    for fname in os.listdir(sfc_dir):
        if not fname.endswith("_sfc_values_pca.txt"):
            continue

        filepath = os.path.join(sfc_dir, fname)
        stem = fname[:-len("_sfc_values_pca.txt")]

        csv_key = None
        subject_id = None
        tokens = stem.replace('_', '-').split('-')
        for token in tokens:
            if token.startswith('M') and len(token) > 1 and token[1:].isdigit():
                csv_key = f"sub-{token}"
                subject_id = token
                break

        if csv_key is None:
            continue

        aphasia_type = subject_categories.get(csv_key)
        if aphasia_type is None:
            continue
        if aphasia_type == 'None':
            aphasia_type = 'Normal Stroke'

        sfc_values = load_sfc_file(filepath, 450)
        if sfc_values is None:
            continue

        # 计算整体不对称性（使用AI）
        _, asymmetry = compute_asymmetry_ai(sfc_values, pairs)

        if np.isnan(asymmetry):
            continue

        # 加载lesion数据并计算dorsal和ventral占比
        dorsal_ratio = 0
        ventral_ratio = 0
        possible_ids = [subject_id, f'M{subject_id}']
        
        for test_id in possible_ids:
            subject_dir = os.path.join(lesion_dir, test_id)
            all_regions_file = os.path.join(subject_dir, f"{test_id}_all_regions.npy")
            
            if os.path.exists(all_regions_file):
                try:
                    region_masks = np.load(all_regions_file, allow_pickle=True).item()
                    lesion_data_arr = region_masks.get('lesion')
                    
                    if lesion_data_arr is not None:
                        # 将lesion数据转换为二值mask
                        lesion_mask = lesion_data_arr > 0
                        
                        # 计算dorsal和ventral区域的lesion占比
                        dorsal_ratio = calculate_roi_lesion_percentage(
                            lesion_mask, atlas_img, dorsal_rois
                        )
                        ventral_ratio = calculate_roi_lesion_percentage(
                            lesion_mask, atlas_img, ventral_rois
                        )
                        
                        print(f"受试者 {test_id}: dorsal_ratio={dorsal_ratio:.4f}, ventral_ratio={ventral_ratio:.4f}")
                        break
                        
                except Exception as e:
                    print(f"处理受试者 {test_id} 的lesion数据时出错: {e}")
                    pass

        if aphasia_type not in lesion_volume_data:
            lesion_volume_data[aphasia_type] = {
                'dorsal_ratio': [],
                'ventral_ratio': [],
                'asymmetry': []
            }

        lesion_volume_data[aphasia_type]['dorsal_ratio'].append(dorsal_ratio)
        lesion_volume_data[aphasia_type]['ventral_ratio'].append(ventral_ratio)
        lesion_volume_data[aphasia_type]['asymmetry'].append(asymmetry)

    return lesion_volume_data


def load_disconnection_and_clinical_data(disconnectome_results_file):
    """
    从disconnectome中介分析结果中加载disconnection和临床分数数据
    
    返回:
    - DataFrame包含subject_id, 7个disconnection变量, 4个临床分数, 和AI
    """
    if not os.path.exists(disconnectome_results_file):
        print(f"警告: Disconnectome结果文件不存在: {disconnectome_results_file}")
        return None
    
    df = pd.read_csv(disconnectome_results_file)
    print(f"成功加载disconnectome结果文件，共 {len(df)} 个受试者")
    
    return df


def correlate_disconnection_with_ai(category_data, pairs, disconnectome_results_file, output_dir):
    """
    计算7个disconnection变量与AI的相关性
    
    参数:
    - category_data: 所有受试者的SFC数据
    - pairs: 左右半球配对
    - disconnectome_results_file: disconnectome中介分析结果文件路径
    - output_dir: 输出目录
    """
    print("\n" + "="*80)
    print("Disconnection变量与AI相关性分析")
    print("="*80)
    
    # 加载disconnectome数据
    disconn_df = load_disconnection_and_clinical_data(disconnectome_results_file)
    if disconn_df is None:
        return
    
    # 定义7个disconnection变量
    disconnection_vars = [
        {'name': 'af_disconnection', 'label': 'AF Disconnection'},
        {'name': 'ilf_disconnection', 'label': 'ILF Disconnection'},
        {'name': 'uf_disconnection', 'label': 'UF Disconnection'},
        {'name': 'af_ilf_disconnection', 'label': 'AF+ILF Disconnection'},
        {'name': 'af_uf_disconnection', 'label': 'AF+UF Disconnection'},
        {'name': 'ilf_uf_disconnection', 'label': 'ILF+UF Disconnection'},
        {'name': 'all_tracts_disconnection', 'label': 'All Tracts Disconnection'}
    ]
    
    # 计算每个受试者的AI
    subject_ai_map = {}
    for aphasia_type, sfc_list in category_data.items():
        # 从文件名中提取subject_id（需要匹配disconn_df中的subject_id）
        # 这里需要重新加载以获取subject_id
        pass
    
    # 重新从SFC文件加载，同时记录subject_id
    sfc_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\sfc_txt_files"
    
    for fname in os.listdir(sfc_dir):
        if not fname.endswith("_sfc_values_pca.txt"):
            continue
        
        filepath = os.path.join(sfc_dir, fname)
        stem = fname[:-len("_sfc_values_pca.txt")]
        
        # 提取subject_id
        subject_id = None
        tokens = stem.replace('_', '-').split('-')
        for token in tokens:
            if token.startswith('M') and len(token) > 1 and token[1:].isdigit():
                subject_id = token[1:]  # 去掉M前缀，与disconn_df匹配
                break
        
        if subject_id is None:
            continue
        
        # 加载SFC数据
        sfc_values = load_sfc_file(filepath, 450)
        if sfc_values is None:
            continue
        
        # 计算AI
        _, ai = compute_asymmetry_ai(sfc_values, pairs)
        if not np.isnan(ai):
            subject_ai_map[subject_id] = ai
    
    print(f"成功计算 {len(subject_ai_map)} 个受试者的AI")
    
    # 将AI添加到disconn_df
    disconn_df['asymmetry_ai'] = disconn_df['subject_id'].apply(
        lambda x: subject_ai_map.get(str(x).strip(), np.nan)
    )
    
    # 计算相关性
    correlation_results = []
    
    for disc_var in disconnection_vars:
        var_name = disc_var['name']
        var_label = disc_var['label']
        
        if var_name not in disconn_df.columns:
            print(f"警告: {var_name} 不在数据中")
            continue
        
        # 提取有效数据
        X = disconn_df[var_name].values
        Y = disconn_df['asymmetry_ai'].values
        
        valid_mask = ~(np.isnan(X) | np.isnan(Y))
        X_valid = X[valid_mask]
        Y_valid = Y[valid_mask]
        
        if len(X_valid) < 3:
            print(f"警告: {var_label} 有效数据不足")
            continue
        
        # 计算Pearson相关
        r, p = stats.pearsonr(X_valid, Y_valid)
        
        correlation_results.append({
            'variable': var_label,
            'variable_name': var_name,
            'r': r,
            'p': p,
            'n': len(X_valid)
        })
        
        print(f"{var_label:30s} r={r:7.4f}, p={p:.4f}, n={len(X_valid)}")
        
        # 如果r > 0.2，绘制散点图
        if abs(r) > 0.2:
            plot_correlation_scatter(
                X_valid, Y_valid, var_label, 'Asymmetry Index (AI)',
                r, p, len(X_valid),
                os.path.join(output_dir, f'correlation_{var_name}_ai.pdf')
            )
    
    # 保存相关性结果到CSV
    corr_df = pd.DataFrame(correlation_results)
    corr_file = os.path.join(output_dir, 'disconnection_ai_correlations.csv')
    corr_df.to_csv(corr_file, index=False)
    print(f"\nDisconnection-AI相关性结果已保存: {corr_file}")
    
    return corr_df


def correlate_clinical_with_ai(category_data, pairs, disconnectome_results_file, output_dir):
    """
    计算4个临床分数与AI的相关性
    
    参数:
    - category_data: 所有受试者的SFC数据
    - pairs: 左右半球配对
    - disconnectome_results_file: disconnectome中介分析结果文件路径
    - output_dir: 输出目录
    """
    print("\n" + "="*80)
    print("临床分数与AI相关性分析")
    print("="*80)
    
    # 加载disconnectome数据
    disconn_df = load_disconnection_and_clinical_data(disconnectome_results_file)
    if disconn_df is None:
        return
    
    # 定义4个临床分数变量
    clinical_vars = [
        {'name': 'repetition_subscore', 'label': 'Repetition'},
        {'name': 'comprehension_subscore', 'label': 'Comprehension'},
        {'name': 'naming_subscore', 'label': 'Naming'},
        {'name': 'wab_r_aq', 'label': 'WAB AQ'}
    ]
    
    # 计算每个受试者的AI
    subject_ai_map = {}
    sfc_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\sfc_txt_files"
    
    for fname in os.listdir(sfc_dir):
        if not fname.endswith("_sfc_values_pca.txt"):
            continue
        
        filepath = os.path.join(sfc_dir, fname)
        stem = fname[:-len("_sfc_values_pca.txt")]
        
        # 提取subject_id
        subject_id = None
        tokens = stem.replace('_', '-').split('-')
        for token in tokens:
            if token.startswith('M') and len(token) > 1 and token[1:].isdigit():
                subject_id = token[1:]  # 去掉M前缀
                break
        
        if subject_id is None:
            continue
        
        # 加载SFC数据
        sfc_values = load_sfc_file(filepath, 450)
        if sfc_values is None:
            continue
        
        # 计算AI
        _, ai = compute_asymmetry_ai(sfc_values, pairs)
        if not np.isnan(ai):
            subject_ai_map[subject_id] = ai
    
    print(f"成功计算 {len(subject_ai_map)} 个受试者的AI")
    
    # 将AI添加到disconn_df
    disconn_df['asymmetry_ai'] = disconn_df['subject_id'].apply(
        lambda x: subject_ai_map.get(str(x).strip(), np.nan)
    )
    
    # 计算相关性
    correlation_results = []
    
    for clin_var in clinical_vars:
        var_name = clin_var['name']
        var_label = clin_var['label']
        
        if var_name not in disconn_df.columns:
            print(f"警告: {var_name} 不在数据中")
            continue
        
        # 提取有效数据
        X = disconn_df[var_name].values
        Y = disconn_df['asymmetry_ai'].values
        
        valid_mask = ~(np.isnan(X) | np.isnan(Y))
        X_valid = X[valid_mask]
        Y_valid = Y[valid_mask]
        
        if len(X_valid) < 3:
            print(f"警告: {var_label} 有效数据不足")
            continue
        
        # 计算Pearson相关
        r, p = stats.pearsonr(X_valid, Y_valid)
        
        correlation_results.append({
            'variable': var_label,
            'variable_name': var_name,
            'r': r,
            'p': p,
            'n': len(X_valid)
        })
        
        print(f"{var_label:30s} r={r:7.4f}, p={p:.4f}, n={len(X_valid)}")
        
        # 如果r > 0.2，绘制散点图
        if abs(r) > 0.2:
            plot_correlation_scatter(
                X_valid, Y_valid, var_label, 'Asymmetry Index (AI)',
                r, p, len(X_valid),
                os.path.join(output_dir, f'correlation_{var_name}_ai.pdf')
            )
    
    # 保存相关性结果到CSV
    corr_df = pd.DataFrame(correlation_results)
    corr_file = os.path.join(output_dir, 'clinical_ai_correlations.csv')
    corr_df.to_csv(corr_file, index=False)
    print(f"\n临床分数-AI相关性结果已保存: {corr_file}")
    
    return corr_df


def plot_correlation_scatter(X, Y, x_label, y_label, r, p, n, output_path):
    """
    绘制相关性散点图
    
    参数:
    - X: 自变量数据
    - Y: 因变量数据
    - x_label: X轴标签
    - y_label: Y轴标签
    - r: 相关系数
    - p: p值
    - n: 样本量
    - output_path: 输出路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制散点
    ax.scatter(X, Y, c='#2874A6', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # 绘制回归线
    z = np.polyfit(X, Y, 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(np.min(X), np.max(X), 100)
    ax.plot(x_line, p_fit(x_line), "r--", linewidth=2, alpha=0.8, label='Regression')
    
    # 添加相关性信息
    ax.text(0.05, 0.95, f'r={r:.3f}, p={p:.4f}\nn={n}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
    ax.set_title(f'{x_label} vs {y_label}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  散点图已保存: {output_path}")


def save_all_data_to_csv(category_asymmetry, category_network_asymmetry,
                         lesion_data, lesion_volume_data, output_dir):
    """保存所有数据到CSV文件（AI指标）"""

    # 1. 保存整体不对称性数据
    rows = []
    for aphasia_type, values in category_asymmetry.items():
        for i, val in enumerate(values):
            rows.append({
                'aphasia_type': aphasia_type,
                'subject_index': i + 1,
                'asymmetry_ai': val
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'overall_asymmetry_ai.csv'), index=False)
    print(f"整体不对称性数据已保存")

    # 2. 保存网络层面不对称性数据
    rows = []
    for aphasia_type, network_data in category_network_asymmetry.items():
        for network, values in network_data.items():
            for i, val in enumerate(values):
                rows.append({
                    'aphasia_type': aphasia_type,
                    'network': network,
                    'subject_index': i + 1,
                    'asymmetry_ai': val
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'network_asymmetry_ai.csv'), index=False)
    print(f"网络不对称性数据已保存")

    # 3. 保存lesion dorsal/ventral数据
    rows = []
    for aphasia_type, data in lesion_data.items():
        for i in range(len(data['dorsal_asym'])):
            rows.append({
                'aphasia_type': aphasia_type,
                'subject_index': i + 1,
                'dorsal_asymmetry_ai': data['dorsal_asym'][i],
                'ventral_asymmetry_ai': data['ventral_asym'][i],
                'lesion_volume': data['lesion_volume'][i]
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'lesion_dorsal_ventral_ai.csv'), index=False)
    print(f"Lesion dorsal/ventral数据已保存")

    # 4. 保存lesion volume相关性数据
    rows = []
    for aphasia_type, data in lesion_volume_data.items():
        for i in range(len(data.get('asymmetry', []))):
            rows.append({
                'aphasia_type': aphasia_type,
                'subject_index': i + 1,
                'dorsal_lesion_ratio': data['dorsal_ratio'][i],
                'ventral_lesion_ratio': data['ventral_ratio'][i],
                'asymmetry_ai': data['asymmetry'][i]
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'lesion_volume_correlation_ai.csv'), index=False)
    print(f"Lesion volume相关性数据已保存")


def main():
    """主函数"""
    # 路径配置
    label_file_path = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
    csv_file_path = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\aphasia_plots\participants_aphasia_new.txt"
    sfc_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\sfc_txt_files"
    lesion_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\extracted_regions"
    output_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\asymmetry_analysis_results"

    roi_number = 450

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("脑区不对称性综合分析 (AI指标)")
    print("="*80)

    # 加载基础数据
    print("\n=== 加载基础数据 ===")
    pairs = get_hemisphere_pairs()
    subject_categories = load_subject_categories(csv_file_path)
    roi_info = parse_network_from_label(label_file_path)
    network_indices = get_network_hemisphere_indices(roi_info)

    print("\n=== 加载所有受试者SFC数据 ===")
    category_data = load_all_subjects_data(sfc_dir, subject_categories, pairs, roi_number)

    # 使用AI指标进行分析
    metric = 'ai'
    print(f"\n{'='*80}")
    print(f"分析指标: AI")
    print(f"{'='*80}")

    # 1. 计算整体不对称性
    print(f"\n[1/7] 计算整体不对称性...")
    category_asymmetry = compute_all_asymmetries(category_data, pairs)

    # 2. 计算网络层面不对称性
    print(f"\n[2/7] 计算网络层面不对称性...")
    category_network_asymmetry = compute_network_asymmetries(
        category_data, network_indices, pairs
    )

    # 3. 绘制脑区级别折线图
    print(f"\n[3/7] 绘制脑区级别折线图...")
    plot_roi_line_charts(
        category_data, pairs, roi_info, metric,
        os.path.join(output_dir, f'roi_line_charts_{metric}.pdf')
    )

    # # 4. 绘制表面图
    # print(f"\n[4/7] 绘制表面图...")
    # plot_surface_maps(
    #     category_data, pairs, metric, output_dir
    # )

    # 5. 绘制网络层面柱状图
    print(f"\n[5/7] 绘制网络层面柱状图...")
    plot_network_bar_charts(
        category_network_asymmetry, metric,
        os.path.join(output_dir, f'network_bar_charts_{metric}.pdf')
    )

    # # 6. 加载lesion数据并绘制dorsal/ventral散点图
    # print(f"\n[6/7] 绘制lesion dorsal/ventral散点图...")
    # lesion_data = load_lesion_data(
    #     lesion_dir, sfc_dir, subject_categories,
    #     label_file_path, pairs
    # )
    # plot_lesion_dorsal_ventral_scatter(
    #     lesion_data, metric,
    #     os.path.join(output_dir, f'lesion_dorsal_ventral_{metric}.pdf')
    # )

    # # 7. 绘制lesion volume相关性图
    # print(f"\n[7/7] 绘制lesion volume相关性图...")
    # lesion_volume_data = prepare_lesion_volume_data(
    #     category_data, lesion_dir, subject_categories,
    #     sfc_dir, pairs, label_file_path
    # )
    # plot_lesion_volume_correlation(
    #     lesion_volume_data, metric,
    #     os.path.join(output_dir, f'lesion_volume_correlation_{metric}.pdf')
    # )

    # # 8. 保存所有数据到CSV
    # print(f"\n[8/10] 保存数据到CSV...")
    # save_all_data_to_csv(
    #     category_asymmetry, category_network_asymmetry,
    #     lesion_data, lesion_volume_data, output_dir
    # )

    # # 9. Disconnection变量与AI相关性分析
    # print(f"\n[9/10] Disconnection变量与AI相关性分析...")
    # disconnectome_results_file = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\disconnectome_mediation_results\subject_measures_all_scores.csv"
    # correlate_disconnection_with_ai(
    #     category_data, pairs, disconnectome_results_file, output_dir
    # )

    # # 10. 临床分数与AI相关性分析
    # print(f"\n[10/10] 临床分数与AI相关性分析...")
    # correlate_clinical_with_ai(
    #     category_data, pairs, disconnectome_results_file, output_dir
    # )

    print(f"\n{'='*80}")
    print(f"所有分析完成！")
    print(f"结果保存在: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
