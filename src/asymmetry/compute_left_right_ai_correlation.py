"""
计算SFC的对称指数（Symmetry Index, AI）

新的AI定义：左脑和右脑的皮尔逊相关系数
- 每个脑区的左脑和同一脑区的右脑为一对散点
- 450个脑区，共225对（皮层下25对 + 皮层200对）
- 计算这225对散点的皮尔逊相关系数作为对称指标

AI = Pearson_r(左脑SFC向量, 右脑SFC向量)
- AI接近1表示高度对称（左右脑区SFC模式一致）
- AI接近0表示无相关性
- AI为负值表示反向相关

脑区结构：
- 前50个ROI（1-50）：皮层下区域
  - 1-25：右半球 (rh) -> 索引0-24
  - 26-50：左半球 (lh) -> 索引25-49
  - 配对：左(索引25-49) 对应 右(索引0-24)
- 后400个ROI（51-450）：Schaefer皮层分区
  - 51-250：左半球 (LH) -> 索引50-249
  - 251-450：右半球 (RH) -> 索引250-449
  - 配对：左(索引50-249) 对应 右(索引250-449)

功能：
1. 计算每个受试者整体的AI（皮尔逊相关系数）
2. 按网络计算AI
3. 按皮层下结构计算AI
"""

import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 设置全局字体
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['pdf.fonttype'] = 42

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
import os
import pandas as pd
import re


# ==================== 索引配对工具函数 ====================

def get_hemisphere_pairs():
    """
    获取左右半球的配对索引。
    
    返回:
    - pairs: list of tuples, [(left_idx, right_idx), ...]
      所有225对左右脑区的0-based索引
    """
    pairs = []
    
    # 皮层下区域配对（25对）
    # 右半球：索引0-24
    # 左半球：索引25-49
    # 配对：左(索引i+25) 对应 右(索引i)，i从0到24
    for i in range(25):
        left_idx = i + 25   # 左半球索引 (25-49)
        right_idx = i       # 右半球索引 (0-24)
        pairs.append((left_idx, right_idx))
    
    # 皮层区域配对（200对）
    # 左半球：索引50-249（200个ROI）
    # 右半球：索引250-449（200个ROI）
    # 配对：左(索引i+50) 对应 右(索引i+250)，i从0到199
    for i in range(200):
        left_idx = i + 50    # 左半球索引 (50-249)
        right_idx = i + 250  # 右半球索引 (250-449)
        pairs.append((left_idx, right_idx))
    
    print(f"共创建 {len(pairs)} 对左右脑区配对")
    print(f"  皮层下配对: 25对 (索引25-49 vs 0-24)")
    print(f"  皮层配对: 200对 (索引50-249 vs 250-449)")
    
    return pairs


def get_hemisphere_indices():
    """
    获取左右半球的ROI索引列表。
    
    返回:
    - left_indices: 左半球ROI索引列表（0-based）
    - right_indices: 右半球ROI索引列表（0-based）
    """
    # 皮层下区域
    subcortical_right = list(range(0, 25))      # 索引0-24
    subcortical_left = list(range(25, 50))      # 索引25-49
    
    # 皮层区域
    cortical_left = list(range(50, 250))        # 索引50-249
    cortical_right = list(range(250, 450))      # 索引250-449
    
    left_indices = subcortical_left + cortical_left
    right_indices = subcortical_right + cortical_right
    
    print(f"左半球ROI数量: {len(left_indices)} (皮层下: {len(subcortical_left)}, 皮层: {len(cortical_left)})")
    print(f"右半球ROI数量: {len(right_indices)} (皮层下: {len(subcortical_right)}, 皮层: {len(cortical_right)})")
    
    return left_indices, right_indices


# ==================== AI计算函数 ====================

def compute_subject_ai_correlation(sfc_values, pairs):
    """
    计算单个受试者的整体AI值（皮尔逊相关系数）。
    
    AI = Pearson_r(左脑SFC向量, 右脑SFC向量)
    
    参数:
    - sfc_values: 受试者的SFC值数组 (450,)
    - pairs: 左右配对列表 [(left_idx, right_idx), ...]
    
    返回:
    - ai: 皮尔逊相关系数
    - p_value: p值
    - n_valid: 有效配对数
    """
    left_values = []
    right_values = []
    
    for left_idx, right_idx in pairs:
        left_val = sfc_values[left_idx]
        right_val = sfc_values[right_idx]
        
        # 只保留两边都不是NaN的配对
        if not np.isnan(left_val) and not np.isnan(right_val):
            left_values.append(left_val)
            right_values.append(right_val)
    
    n_valid = len(left_values)
    
    if n_valid < 3:  # 至少需要3个点计算相关系数
        return np.nan, np.nan, n_valid
    
    # 计算皮尔逊相关系数
    r, p_value = stats.pearsonr(left_values, right_values)
    
    return r, p_value, n_valid


def compute_network_ai_correlation(sfc_values, network_indices):
    """
    按网络计算AI值（皮尔逊相关系数）。
    
    对于每个网络，获取该网络内的左右脑区配对，计算相关系数。
    
    参数:
    - sfc_values: SFC值数组 (450,)
    - network_indices: {network_name: {'L': [indices], 'R': [indices]}}
    
    返回:
    - network_ai: {network_name: (r, p_value, n_valid)}
    """
    network_ai = {}
    
    for network, indices in network_indices.items():
        left_indices = indices['L']
        right_indices = indices['R']
        
        # 确保左右数量相等
        n_pairs = min(len(left_indices), len(right_indices))
        
        if n_pairs == 0:
            network_ai[network] = (np.nan, np.nan, 0)
            continue
        
        left_values = []
        right_values = []
        
        # 按顺序配对
        for i in range(n_pairs):
            left_val = sfc_values[left_indices[i]]
            right_val = sfc_values[right_indices[i]]
            
            if not np.isnan(left_val) and not np.isnan(right_val):
                left_values.append(left_val)
                right_values.append(right_val)
        
        n_valid = len(left_values)
        
        if n_valid < 3:
            network_ai[network] = (np.nan, np.nan, n_valid)
            continue
        
        r, p_value = stats.pearsonr(left_values, right_values)
        network_ai[network] = (r, p_value, n_valid)
    
    return network_ai


def compute_subcortical_ai_correlation(sfc_values, structure_indices):
    """
    按皮层下结构计算AI值（皮尔逊相关系数）。
    
    参数:
    - sfc_values: SFC值数组 (450,)
    - structure_indices: {structure_name: {'L': [indices], 'R': [indices], 'function': str}}
    
    返回:
    - structure_ai: {structure_name: (r, p_value, n_valid)}
    """
    structure_ai = {}
    
    for structure, info in structure_indices.items():
        left_indices = info['L']
        right_indices = info['R']
        
        n_pairs = min(len(left_indices), len(right_indices))
        
        if n_pairs == 0:
            structure_ai[structure] = (np.nan, np.nan, 0)
            continue
        
        left_values = []
        right_values = []
        
        for i in range(n_pairs):
            left_val = sfc_values[left_indices[i]]
            right_val = sfc_values[right_indices[i]]
            
            if not np.isnan(left_val) and not np.isnan(right_val):
                left_values.append(left_val)
                right_values.append(right_val)
        
        n_valid = len(left_values)
        
        if n_valid < 3:
            structure_ai[structure] = (np.nan, np.nan, n_valid)
            continue
        
        r, p_value = stats.pearsonr(left_values, right_values)
        structure_ai[structure] = (r, p_value, n_valid)
    
    return structure_ai


# ==================== 网络和结构定义 ====================

def parse_network_from_label(label_file_path):
    """
    解析标签文件，提取每个ROI的网络归属和半球信息。
    
    返回:
    - roi_info: list of dict
    """
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
    """
    获取每个网络的左右半球ROI索引。
    
    返回:
    - network_indices: {network_name: {'L': [indices], 'R': [indices]}}
    """
    network_indices = {}
    
    for roi in roi_info:
        network = roi['network']
        hemisphere = roi['hemisphere']
        idx = roi['index'] - 1  # 转为0-based
        
        if network not in network_indices:
            network_indices[network] = {'L': [], 'R': []}
        
        if hemisphere == 'L':
            network_indices[network]['L'].append(idx)
        elif hemisphere == 'R':
            network_indices[network]['R'].append(idx)
    
    print("\n网络ROI统计:")
    for network, indices in network_indices.items():
        print(f"  {network}: 左半球 {len(indices['L'])} 个, 右半球 {len(indices['R'])} 个")
    
    return network_indices


def get_subcortical_structure_indices():
    """
    获取皮层下结构的左右半球索引。
    
    返回:
    - structure_indices: {structure_name: {'L': [indices], 'R': [indices], 'function': str}}
    """
    structure_indices = {
        'Hippocampus': {
            'R': [i - 1 for i in [1, 2, 3, 4]],
            'L': [i - 1 for i in [26, 27, 28, 29]],
            'function': 'Memory'
        },
        'Thalamus': {
            'R': [i - 1 for i in [5, 6, 7, 8, 9, 10, 21]],
            'L': [i - 1 for i in [30, 31, 32, 33, 34, 35, 46]],
            'function': 'Relay'
        },
        'Putamen': {
            'R': [i - 1 for i in [11, 12, 13, 14]],
            'L': [i - 1 for i in [36, 37, 38, 39]],
            'function': 'Motor'
        },
        'Caudate': {
            'R': [i - 1 for i in [15, 16, 17, 18]],
            'L': [i - 1 for i in [40, 41, 42, 43]],
            'function': 'Cognition'
        },
        'Amygdala': {
            'R': [i - 1 for i in [19, 20]],
            'L': [i - 1 for i in [44, 45]],
            'function': 'Emotion'
        },
        'NAc': {
            'R': [i - 1 for i in [22, 23]],
            'L': [i - 1 for i in [47, 48]],
            'function': 'Reward'
        },
        'GP': {
            'R': [i - 1 for i in [24, 25]],
            'L': [i - 1 for i in [49, 50]],
            'function': 'Motor regulation'
        }
    }
    
    print("\n皮层下结构ROI统计:")
    for structure, info in structure_indices.items():
        print(f"  {structure}: 左 {len(info['L'])} 个, 右 {len(info['R'])} 个 ({info['function']})")
    
    return structure_indices


# ==================== 数据加载函数 ====================

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
            
            # 跳过特定类别
            if aphasia_type in {"TranscorticalMotor", "Wernicke"}:
                continue

            subject_categories[participant_id] = aphasia_type

        print(f"从TSV成功解析 {len(subject_categories)} 个受试者类别映射")
        return subject_categories

    except Exception as e:
        print(f"读取TSV时发生错误: {str(e)}")
        return {}


def load_all_subjects_ai(sfc_dir, subject_categories, pairs, roi_number=450):
    """
    加载所有受试者的SFC数据并计算AI值（皮尔逊相关系数）。
    
    返回:
    - category_ai_data: {aphasia_type: [(ai, p_value, n_valid), ...]}
    """
    category_ai_data = {}
    
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
        
        # 计算AI（皮尔逊相关系数）
        ai, p_value, n_valid = compute_subject_ai_correlation(sfc_values, pairs)
        
        if not np.isnan(ai):
            if aphasia_type not in category_ai_data:
                category_ai_data[aphasia_type] = []
            category_ai_data[aphasia_type].append((ai, p_value, n_valid))
    
    print("\n按类别加载的受试者AI数据:")
    for aphasia_type, ai_list in category_ai_data.items():
        print(f"  {aphasia_type}: {len(ai_list)} 个受试者")
    
    return category_ai_data


def load_all_subjects_network_ai(sfc_dir, subject_categories, network_indices, roi_number=450):
    """
    加载所有受试者的SFC数据并计算每个网络的AI值。
    
    返回:
    - category_network_ai: {aphasia_type: {network: [(r, p, n), ...]}}
    """
    category_network_ai = {}
    networks = list(network_indices.keys())
    
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
        
        # 计算每个网络的AI
        network_ai = compute_network_ai_correlation(sfc_values, network_indices)
        
        if aphasia_type not in category_network_ai:
            category_network_ai[aphasia_type] = {net: [] for net in networks}
        
        for network, (r, p, n) in network_ai.items():
            if not np.isnan(r):
                category_network_ai[aphasia_type][network].append((r, p, n))
    
    print("\n按类别和网络加载的受试者AI数据:")
    for aphasia_type, network_data in category_network_ai.items():
        counts = [len(network_data[net]) for net in networks]
        print(f"  {aphasia_type}: {counts[0]} 个受试者")
    
    return category_network_ai


def load_all_subjects_subcortical_ai(sfc_dir, subject_categories, structure_indices, roi_number=450):
    """
    加载所有受试者的SFC数据并计算每个皮层下结构的AI值。
    
    返回:
    - category_structure_ai: {aphasia_type: {structure: [(r, p, n), ...]}}
    """
    category_structure_ai = {}
    structures = list(structure_indices.keys())
    
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
        
        # 计算每个皮层下结构的AI
        structure_ai = compute_subcortical_ai_correlation(sfc_values, structure_indices)
        
        if aphasia_type not in category_structure_ai:
            category_structure_ai[aphasia_type] = {s: [] for s in structures}
        
        for structure, (r, p, n) in structure_ai.items():
            if not np.isnan(r):
                category_structure_ai[aphasia_type][structure].append((r, p, n))
    
    print("\n按类别和皮层下结构加载的受试者AI数据:")
    for aphasia_type, structure_data in category_structure_ai.items():
        counts = [len(structure_data[s]) for s in structures]
        print(f"  {aphasia_type}: {counts[0]} 个受试者")
    
    return category_structure_ai


# ==================== 新增：加载原始SFC数据用于散点图 ====================

def load_all_subjects_sfc_data(sfc_dir, subject_categories, pairs, roi_number=450):
    """
    加载所有受试者的原始SFC数据（用于散点图）。
    
    返回:
    - category_sfc_data: {aphasia_type: [(left_values, right_values), ...]}
    """
    category_sfc_data = {}
    
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
        
        # 提取左右脑SFC配对值
        left_values = []
        right_values = []
        
        for left_idx, right_idx in pairs:
            left_val = sfc_values[left_idx]
            right_val = sfc_values[right_idx]
            
            if not np.isnan(left_val) and not np.isnan(right_val):
                left_values.append(left_val)
                right_values.append(right_val)
        
        if len(left_values) > 0:
            if aphasia_type not in category_sfc_data:
                category_sfc_data[aphasia_type] = []
            category_sfc_data[aphasia_type].append((np.array(left_values), np.array(right_values)))
    
    return category_sfc_data


# ==================== 新增可视化方案 ====================

def plot_lr_scatter_by_group(category_sfc_data, category_ai_data, output_path):
    """
    方案1: 左右脑散点图（分面板，每组一个子图）
    
    每个子图展示该组所有受试者的左脑SFC vs 右脑SFC散点，
    添加回归线和r值标注。
    """
    aphasia_colors = ["#7895C1", "#A8CBDF", "#992224", "#EF8B67", "#E3625D", "#B54764"]
    
    # 设置绘图顺序
    plot_order = []
    if 'Healthy' in category_sfc_data:
        plot_order.append('Healthy')
    if 'Normal Stroke' in category_sfc_data:
        plot_order.append('Normal Stroke')
    for label in category_sfc_data.keys():
        if label not in plot_order:
            plot_order.append(label)
    
    # 创建颜色字典
    color_dict = {}
    color_idx = 0
    for label in plot_order:
        if label == 'Healthy':
            color_dict[label] = '#808080'
        else:
            color_dict[label] = aphasia_colors[color_idx % len(aphasia_colors)]
            color_idx += 1
    
    n_groups = len(plot_order)
    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_groups == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 计算全局范围
    all_left = []
    all_right = []
    for label in plot_order:
        for left_vals, right_vals in category_sfc_data[label]:
            all_left.extend(left_vals)
            all_right.extend(right_vals)
    
    global_min = min(min(all_left), min(all_right))
    global_max = max(max(all_left), max(all_right))
    margin = (global_max - global_min) * 0.05
    
    for idx, label in enumerate(plot_order):
        ax = axes[idx]
        color = color_dict[label]
        
        # 收集该组所有受试者的数据
        all_left_group = []
        all_right_group = []
        
        for left_vals, right_vals in category_sfc_data[label]:
            all_left_group.extend(left_vals)
            all_right_group.extend(right_vals)
        
        all_left_group = np.array(all_left_group)
        all_right_group = np.array(all_right_group)
        
        # 绘制散点
        ax.scatter(all_left_group, all_right_group, c=color, alpha=0.3, s=8, 
                   edgecolors='none', rasterized=True)
        
        # 添加回归线
        if len(all_left_group) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(all_left_group, all_right_group)
            x_line = np.array([global_min - margin, global_max + margin])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=color, linewidth=2, linestyle='-', label=f'r = {r_value:.3f}')
        
        # 添加对角线（完美对称线）
        diag_line = np.array([global_min - margin, global_max + margin])
        ax.plot(diag_line, diag_line, color='gray', linewidth=1.5, linestyle='--', alpha=0.5, label='y = x')
        
        # 计算该组的平均AI
        if label in category_ai_data:
            ai_values = [r for r, p, n in category_ai_data[label]]
            mean_ai = np.mean(ai_values)
            n_subjects = len(ai_values)
            ax.set_title(f'{label}\n(n={n_subjects}, mean r={mean_ai:.3f})', 
                        fontsize=12, fontweight='bold', fontfamily='Arial')
        else:
            ax.set_title(label, fontsize=12, fontweight='bold', fontfamily='Arial')
        
        ax.set_xlabel('Left Hemisphere SFC', fontsize=10, fontfamily='Arial')
        ax.set_ylabel('Right Hemisphere SFC', fontsize=10, fontfamily='Arial')
        ax.set_xlim(global_min - margin, global_max + margin)
        ax.set_ylim(global_min - margin, global_max + margin)
        ax.set_aspect('equal')
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(n_groups, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    
    print(f"\n左右脑散点图已保存: {output_path}")


def plot_forest_plot(category_ai_data, output_path):
    """
    方案2: 森林图（Forest Plot）
    
    展示每组的AI均值和95%置信区间。
    """
    aphasia_colors = ["#7895C1", "#A8CBDF", "#992224", "#EF8B67", "#E3625D", "#B54764"]
    
    # 设置绘图顺序
    plot_order = []
    if 'Healthy' in category_ai_data:
        plot_order.append('Healthy')
    if 'Normal Stroke' in category_ai_data:
        plot_order.append('Normal Stroke')
    for label in category_ai_data.keys():
        if label not in plot_order:
            plot_order.append(label)
    
    # 创建颜色字典
    color_dict = {}
    color_idx = 0
    for label in plot_order:
        if label == 'Healthy':
            color_dict[label] = '#808080'
        else:
            color_dict[label] = aphasia_colors[color_idx % len(aphasia_colors)]
            color_idx += 1
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_positions = list(range(len(plot_order)))
    
    for pos, label in enumerate(plot_order):
        if label not in category_ai_data:
            continue
        
        ai_values = np.array([r for r, p, n in category_ai_data[label]])
        n = len(ai_values)
        mean = np.mean(ai_values)
        std = np.std(ai_values, ddof=1)
        se = std / np.sqrt(n)
        
        # 95% 置信区间
        ci_lower = mean - 1.96 * se
        ci_upper = mean + 1.96 * se
        
        color = color_dict[label]
        
        # 绘制置信区间线
        ax.plot([ci_lower, ci_upper], [pos, pos], color=color, linewidth=2, solid_capstyle='round')
        
        # 绘制均值点
        ax.scatter([mean], [pos], color=color, s=120, zorder=5, edgecolors='white', linewidths=1.5)
        
        # 添加数值标注（在点旁边）
        ax.text(mean + 0.02, pos, f'r={mean:.3f}', 
                va='center', ha='left', fontsize=9, fontfamily='Arial', color=color)
        
        # 添加样本量（在y轴标签旁边）
        ax.text(-0.02, pos, f'n={n}', va='center', ha='right', fontsize=9, fontfamily='Arial')
    
    # 添加Healthy组的参考线
    if 'Healthy' in category_ai_data:
        healthy_mean = np.mean([r for r, p, n in category_ai_data['Healthy']])
        ax.axvline(x=healthy_mean, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Healthy mean (r={healthy_mean:.3f})')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_order, fontsize=12, fontfamily='Arial')
    ax.set_xlabel('Symmetry Index (Pearson r) with 95% CI', fontsize=13, fontfamily='Arial')
    ax.set_title('Forest Plot: AI by Aphasia Type', fontsize=14, fontweight='bold', fontfamily='Arial')
    
    ax.set_xlim(0, 0.6)  # 设置x轴范围到0.6
    ax.set_ylim(-0.5, len(plot_order) - 0.5)
    ax.invert_yaxis()  # 让Healthy在最上面
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)  # 图例放右上角
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    
    print(f"\n森林图已保存: {output_path}")


def plot_radar_chart(category_network_ai, output_path):
    """
    方案3: 雷达图（按网络展示各组的AI模式）
    
    每个失语症类型一条线，展示在各网络上的AI均值。
    """
    aphasia_colors = ["#7895C1", "#A8CBDF", "#992224", "#EF8B67", "#E3625D", "#B54764"]
    
    networks = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis", "Subcortical"]
    
    # 设置绘图顺序
    plot_order = []
    if 'Healthy' in category_network_ai:
        plot_order.append('Healthy')
    if 'Normal Stroke' in category_network_ai:
        plot_order.append('Normal Stroke')
    for label in category_network_ai.keys():
        if label not in plot_order:
            plot_order.append(label)
    
    # 创建颜色字典
    color_dict = {}
    color_idx = 0
    for label in plot_order:
        if label == 'Healthy':
            color_dict[label] = '#808080'
        else:
            color_dict[label] = aphasia_colors[color_idx % len(aphasia_colors)]
            color_idx += 1
    
    # 计算每组每个网络的平均AI
    group_network_means = {}
    for label in plot_order:
        if label not in category_network_ai:
            continue
        
        network_means = []
        for network in networks:
            if network in category_network_ai[label]:
                ai_list = category_network_ai[label][network]
                if len(ai_list) > 0:
                    mean_r = np.mean([r for r, p, n in ai_list])
                    network_means.append(mean_r)
                else:
                    network_means.append(np.nan)
            else:
                network_means.append(np.nan)
        
        group_network_means[label] = network_means
    
    # 创建雷达图
    n_networks = len(networks)
    angles = np.linspace(0, 2 * np.pi, n_networks, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for label in plot_order:
        if label not in group_network_means:
            continue
        
        values = group_network_means[label]
        values += values[:1]  # 闭合
        
        color = color_dict[label]
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color, markersize=6)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(networks, fontsize=11, fontfamily='Arial')
    
    # 设置y轴范围（最外圈为0.4）
    ax.set_ylim(0, 0.4)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.set_yticklabels(['0.1', '0.2', '0.3', '0.4'], fontsize=9, fontfamily='Arial')
    
    ax.set_title('Network AI Pattern by Aphasia Type', fontsize=14, fontweight='bold', 
                 fontfamily='Arial', pad=20)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10, 
              title='Aphasia Type', title_fontsize=11)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    
    print(f"\n雷达图已保存: {output_path}")


def plot_boxplot_with_significance(category_ai_data, output_path):
    """
    方案4: 带显著性标注的组间比较箱型图
    
    在箱型图上添加与Healthy组比较的显著性星号和效应量。
    """
    aphasia_colors = ["#7895C1", "#A8CBDF", "#992224", "#EF8B67", "#E3625D", "#B54764"]
    
    # 提取AI值
    all_data = {}
    for aphasia_type, ai_list in category_ai_data.items():
        ai_values = [r for r, p, n in ai_list]
        all_data[aphasia_type] = np.array(ai_values)
    
    # 设置绘图顺序
    plot_order = []
    if 'Healthy' in all_data:
        plot_order.append('Healthy')
    if 'Normal Stroke' in all_data:
        plot_order.append('Normal Stroke')
    for label in all_data.keys():
        if label not in plot_order:
            plot_order.append(label)
    
    # 创建颜色字典
    color_dict = {}
    color_idx = 0
    for label in plot_order:
        if label == 'Healthy':
            color_dict[label] = '#808080'
        else:
            color_dict[label] = aphasia_colors[color_idx % len(aphasia_colors)]
            color_idx += 1
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    positions = list(range(len(plot_order)))
    rng = np.random.default_rng(seed=42)
    
    # 获取Healthy组数据（用于比较）
    healthy_data = all_data.get('Healthy', None)
    
    # 存储统计结果
    stats_results = []
    
    for pos, label in enumerate(plot_order):
        if label not in all_data:
            continue
        
        data_points = all_data[label]
        color = color_dict[label]
        
        # 绘制箱型图
        bp = ax.boxplot(data_points, positions=[pos], widths=0.6,
                       patch_artist=True, showfliers=False, 
                       medianprops={'color': 'black', 'linewidth': 1.5},
                       boxprops={'facecolor': 'white', 'edgecolor': color, 'linewidth': 1.5},
                       whiskerprops={'color': color, 'linewidth': 1.2},
                       capprops={'color': color, 'linewidth': 1.2})
        
        # 添加散点
        jitter = rng.normal(0, 0.08, len(data_points))
        ax.scatter(pos + jitter, data_points, color=color, alpha=0.6, s=25, 
                   edgecolors='none', zorder=3)
        
        # 与Healthy组比较
        if healthy_data is not None and label != 'Healthy' and len(data_points) > 2:
            # Mann-Whitney U检验
            stat, p_value = stats.mannwhitneyu(healthy_data, data_points, alternative='two-sided')
            
            # Cohen's d 效应量
            pooled_std = np.sqrt((np.var(healthy_data, ddof=1) + np.var(data_points, ddof=1)) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(healthy_data) - np.mean(data_points)) / pooled_std
            else:
                cohens_d = 0
            
            stats_results.append({
                'group': label,
                'p_value': p_value,
                'cohens_d': cohens_d
            })
    
    # 添加显著性标注
    if healthy_data is not None:
        healthy_pos = plot_order.index('Healthy') if 'Healthy' in plot_order else None
        max_val = max([np.max(all_data[label]) for label in plot_order if label in all_data])
        
        y_offset = max_val * 0.05
        current_y = max_val + y_offset
        
        for result in stats_results:
            label = result['group']
            p = result['p_value']
            d = result['cohens_d']
            pos = plot_order.index(label)
            
            # 显著性星号
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            else:
                stars = 'ns'
            
            # 绘制连接线和标注
            if healthy_pos is not None:
                line_y = current_y
                ax.plot([healthy_pos, pos], [line_y, line_y], 'k-', linewidth=1)
                ax.plot([healthy_pos, healthy_pos], [line_y - y_offset*0.3, line_y], 'k-', linewidth=1)
                ax.plot([pos, pos], [line_y - y_offset*0.3, line_y], 'k-', linewidth=1)
                
                # 标注文字
                mid_x = (healthy_pos + pos) / 2
                ax.text(mid_x, line_y + y_offset*0.2, f'{stars}\nd={d:.2f}', 
                       ha='center', va='bottom', fontsize=9, fontfamily='Arial')
                
                current_y += y_offset * 1.5
    
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_order, fontsize=11, fontfamily='Arial', rotation=15, ha='right')
    ax.set_ylabel('Symmetry Index (Pearson r)', fontsize=13, fontfamily='Arial')
    ax.set_xlabel('Aphasia Type', fontsize=13, fontfamily='Arial')
    ax.set_title('AI Comparison with Significance (vs Healthy)', fontsize=14, fontweight='bold', fontfamily='Arial')
    
    # 添加参考线
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.2, alpha=0.5)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    
    print(f"\n带显著性标注的箱型图已保存: {output_path}")
    
    # 打印统计结果
    print("\n=== 与Healthy组比较的统计结果 ===")
    for result in stats_results:
        print(f"  {result['group']}: p={result['p_value']:.4f}, Cohen's d={result['cohens_d']:.3f}")


def plot_pairwise_heatmap(category_ai_data, output_path):
    """
    方案5: 组间两两比较热力图
    
    展示组间两两比较的效应量（Cohen's d）和显著性。
    """
    # 提取AI值
    all_data = {}
    for aphasia_type, ai_list in category_ai_data.items():
        ai_values = [r for r, p, n in ai_list]
        all_data[aphasia_type] = np.array(ai_values)
    
    # 设置顺序
    groups = []
    if 'Healthy' in all_data:
        groups.append('Healthy')
    if 'Normal Stroke' in all_data:
        groups.append('Normal Stroke')
    for label in all_data.keys():
        if label not in groups:
            groups.append(label)
    
    n_groups = len(groups)
    
    # 计算两两比较的效应量和p值
    effect_matrix = np.zeros((n_groups, n_groups))
    p_matrix = np.ones((n_groups, n_groups))
    
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i == j:
                effect_matrix[i, j] = 0
                p_matrix[i, j] = 1
            elif i < j:
                data1 = all_data[group1]
                data2 = all_data[group2]
                
                if len(data1) > 2 and len(data2) > 2:
                    # Mann-Whitney U检验
                    stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    p_matrix[i, j] = p_value
                    p_matrix[j, i] = p_value
                    
                    # Cohen's d
                    pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
                    if pooled_std > 0:
                        d = (np.mean(data1) - np.mean(data2)) / pooled_std
                    else:
                        d = 0
                    effect_matrix[i, j] = d
                    effect_matrix[j, i] = -d  # 对称
    
    # 创建热力图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 效应量热力图
    ax1 = axes[0]
    max_abs = np.max(np.abs(effect_matrix))
    im1 = ax1.imshow(effect_matrix, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs, aspect='auto')
    
    ax1.set_xticks(range(n_groups))
    ax1.set_yticks(range(n_groups))
    ax1.set_xticklabels(groups, fontsize=10, fontfamily='Arial', rotation=45, ha='right')
    ax1.set_yticklabels(groups, fontsize=10, fontfamily='Arial')
    ax1.set_title("Cohen's d Effect Size", fontsize=13, fontweight='bold', fontfamily='Arial')
    
    # 添加数值标注
    for i in range(n_groups):
        for j in range(n_groups):
            if i != j:
                text = ax1.text(j, i, f'{effect_matrix[i, j]:.2f}',
                               ha='center', va='center', fontsize=9, fontfamily='Arial')
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("Cohen's d", fontsize=11, fontfamily='Arial')
    
    # p值热力图（使用-log10变换）
    ax2 = axes[1]
    
    # 创建显著性标注矩阵
    sig_matrix = np.zeros((n_groups, n_groups), dtype=object)
    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                sig_matrix[i, j] = '-'
            else:
                p = p_matrix[i, j]
                if p < 0.001:
                    sig_matrix[i, j] = '***'
                elif p < 0.01:
                    sig_matrix[i, j] = '**'
                elif p < 0.05:
                    sig_matrix[i, j] = '*'
                else:
                    sig_matrix[i, j] = 'ns'
    
    # -log10(p) 转换，处理对角线
    log_p_matrix = -np.log10(np.clip(p_matrix, 1e-10, 1))
    np.fill_diagonal(log_p_matrix, 0)
    
    im2 = ax2.imshow(log_p_matrix, cmap='Reds', vmin=0, vmax=4, aspect='auto')
    
    ax2.set_xticks(range(n_groups))
    ax2.set_yticks(range(n_groups))
    ax2.set_xticklabels(groups, fontsize=10, fontfamily='Arial', rotation=45, ha='right')
    ax2.set_yticklabels(groups, fontsize=10, fontfamily='Arial')
    ax2.set_title('-log₁₀(p-value)', fontsize=13, fontweight='bold', fontfamily='Arial')
    
    # 添加显著性标注
    for i in range(n_groups):
        for j in range(n_groups):
            text = ax2.text(j, i, sig_matrix[i, j],
                           ha='center', va='center', fontsize=10, fontfamily='Arial',
                           color='white' if log_p_matrix[i, j] > 2 else 'black')
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('-log₁₀(p)', fontsize=11, fontfamily='Arial')
    # 添加显著性阈值线
    cbar2.ax.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=1)
    cbar2.ax.axhline(y=-np.log10(0.01), color='black', linestyle='--', linewidth=1)
    cbar2.ax.axhline(y=-np.log10(0.001), color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    
    print(f"\n组间比较热力图已保存: {output_path}")


# ==================== 新增：左右脑SFC对比图 ====================

def load_all_subjects_lr_sfc(sfc_dir, subject_categories, network_indices, roi_number=450):
    """
    加载所有受试者按网络分的左右脑SFC均值。
    
    每个受试者在每个网络（7个皮层网络 + 1个皮层下）有一个左脑均值和一个右脑均值。
    所以每个受试者有8个左脑点和8个右脑点。
    
    返回:
    - category_lr_sfc: {aphasia_type: {'L': [网络均值列表], 'R': [网络均值列表]}}
      每个类别中所有受试者的所有网络的左右脑均值
    """
    category_lr_sfc = {}
    
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
        
        # 初始化
        if aphasia_type not in category_lr_sfc:
            category_lr_sfc[aphasia_type] = {'L': [], 'R': []}
        
        # 计算每个网络的左右脑SFC均值
        for network, indices in network_indices.items():
            left_indices = indices['L']
            right_indices = indices['R']
            
            if len(left_indices) > 0:
                left_sfc = sfc_values[left_indices]
                left_mean = np.nanmean(left_sfc)
                if not np.isnan(left_mean):
                    category_lr_sfc[aphasia_type]['L'].append(left_mean)
            
            if len(right_indices) > 0:
                right_sfc = sfc_values[right_indices]
                right_mean = np.nanmean(right_sfc)
                if not np.isnan(right_mean):
                    category_lr_sfc[aphasia_type]['R'].append(right_mean)
    
    # 打印统计信息
    print("\n各类别左右脑SFC数据点数量（每受试者8个网络）:")
    for aphasia_type, data in category_lr_sfc.items():
        n_subjects = len(data['L']) // 8 if len(data['L']) > 0 else 0
        print(f"  {aphasia_type}: 左脑 {len(data['L'])} 个点 (~{n_subjects} 受试者), 右脑 {len(data['R'])} 个点")
    
    return category_lr_sfc


def plot_lr_sfc_comparison(category_lr_sfc, output_path):
    """
    绘制左右脑SFC对比图（单个图）。
    
    风格：半小提琴 + 箱体 + 散点
    x轴是失语症类别，每个类别有左右脑两个箱型图（距离较近）。
    """
    # 颜色配置
    left_color = '#E74C3C'   # 红色系 - 左脑
    right_color = '#3498DB'  # 蓝色系 - 右脑
    
    # 设置绘图顺序
    plot_order = []
    if 'Healthy' in category_lr_sfc:
        plot_order.append('Healthy')
    if 'Normal Stroke' in category_lr_sfc:
        plot_order.append('Normal Stroke')
    for label in category_lr_sfc.keys():
        if label not in plot_order:
            plot_order.append(label)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    rng = np.random.default_rng(seed=42)
    
    # 组内左右偏移（距离更近）
    offset = 0.12
    violin_width = 0.22
    violin_alpha = 0.35
    
    positions = list(range(len(plot_order)))
    max_val = -np.inf
    min_val = np.inf
    sig_results = []
    
    for pos, label in enumerate(plot_order):
        if label not in category_lr_sfc:
            continue
        
        left_data = np.array(category_lr_sfc[label]['L'])
        right_data = np.array(category_lr_sfc[label]['R'])
        
        if len(left_data) == 0 or len(right_data) == 0:
            continue
        
        max_val = max(max_val, np.max(left_data), np.max(right_data))
        min_val = min(min_val, np.min(left_data), np.min(right_data))
        
        # 左脑位置
        left_pos = pos - offset
        # 右脑位置
        right_pos = pos + offset
        
        # 绘制左脑半小提琴
        if len(left_data) >= 3:
            try:
                violin_l = ax.violinplot([left_data], positions=[left_pos], vert=True,
                                        showmeans=False, showextrema=False, widths=violin_width)
                for pc in violin_l['bodies']:
                    path = pc.get_paths()[0]
                    vertices = path.vertices
                    # 保留左半部分
                    vertices[:, 0] = np.where(vertices[:, 0] < left_pos, vertices[:, 0], left_pos)
                    path.vertices = vertices
                    pc.set_facecolor(left_color)
                    pc.set_edgecolor(left_color)
                    pc.set_alpha(violin_alpha)
            except:
                pass
        
        # 绘制右脑半小提琴
        if len(right_data) >= 3:
            try:
                violin_r = ax.violinplot([right_data], positions=[right_pos], vert=True,
                                        showmeans=False, showextrema=False, widths=violin_width)
                for pc in violin_r['bodies']:
                    path = pc.get_paths()[0]
                    vertices = path.vertices
                    # 保留右半部分
                    vertices[:, 0] = np.where(vertices[:, 0] > right_pos, vertices[:, 0], right_pos)
                    path.vertices = vertices
                    pc.set_facecolor(right_color)
                    pc.set_edgecolor(right_color)
                    pc.set_alpha(violin_alpha)
            except:
                pass
        
        # 绘制左脑箱体图
        if len(left_data) > 0:
            median_l = np.median(left_data)
            q1_l = np.percentile(left_data, 25)
            q3_l = np.percentile(left_data, 75)
            box_width = 0.06
            ax.add_patch(plt.Rectangle((left_pos - box_width/2, q1_l),
                                      box_width, q3_l - q1_l,
                                      facecolor='white', edgecolor=left_color, 
                                      linewidth=1.5, zorder=3))
            ax.plot([left_pos - box_width*0.4, left_pos + box_width*0.4], [median_l, median_l],
                   color=left_color, linewidth=2, zorder=4)
            ax.plot([left_pos, left_pos], [q1_l, np.min(left_data)], color=left_color, linewidth=1.2, zorder=2)
            ax.plot([left_pos, left_pos], [q3_l, np.max(left_data)], color=left_color, linewidth=1.2, zorder=2)
        
        # 绘制右脑箱体图
        if len(right_data) > 0:
            median_r = np.median(right_data)
            q1_r = np.percentile(right_data, 25)
            q3_r = np.percentile(right_data, 75)
            box_width = 0.06
            ax.add_patch(plt.Rectangle((right_pos - box_width/2, q1_r),
                                      box_width, q3_r - q1_r,
                                      facecolor='white', edgecolor=right_color, 
                                      linewidth=1.5, zorder=3))
            ax.plot([right_pos - box_width*0.4, right_pos + box_width*0.4], [median_r, median_r],
                   color=right_color, linewidth=2, zorder=4)
            ax.plot([right_pos, right_pos], [q1_r, np.min(right_data)], color=right_color, linewidth=1.2, zorder=2)
            ax.plot([right_pos, right_pos], [q3_r, np.max(right_data)], color=right_color, linewidth=1.2, zorder=2)
        
        # 绘制散点（每个受试者8个网络的均值点）
        scatter_x_l = rng.normal(left_pos - 0.06, 0.015, len(left_data))
        ax.scatter(scatter_x_l, left_data, color=left_color, alpha=0.4, s=12, edgecolors='none', zorder=2)
        
        scatter_x_r = rng.normal(right_pos + 0.06, 0.015, len(right_data))
        ax.scatter(scatter_x_r, right_data, color=right_color, alpha=0.4, s=12, edgecolors='none', zorder=2)
        
        # Mann-Whitney U检验（左右脑比较，独立样本）
        if len(left_data) >= 3 and len(right_data) >= 3:
            try:
                stat, p_value = stats.mannwhitneyu(left_data, right_data, alternative='two-sided')
            except:
                p_value = 1.0
            
            sig_results.append({
                'label': label,
                'pos': pos,
                'p_value': p_value,
                'left_mean': np.mean(left_data),
                'right_mean': np.mean(right_data),
                'n_left': len(left_data),
                'n_right': len(right_data)
            })
    
    # 添加显著性标注
    if max_val > min_val:
        y_range = max_val - min_val
        sig_y = max_val + y_range * 0.05
        
        for result in sig_results:
            p = result['p_value']
            pos = result['pos']
            
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            else:
                stars = ''
            
            if stars:
                # 绘制连接线
                left_pos = pos - offset
                right_pos = pos + offset
                ax.plot([left_pos, left_pos, right_pos, right_pos], 
                       [sig_y - y_range*0.015, sig_y, sig_y, sig_y - y_range*0.015],
                       color='black', linewidth=1)
                ax.text(pos, sig_y + y_range*0.01, stars, ha='center', va='bottom', 
                       fontsize=11, fontfamily='Arial')
    
    # 设置坐标轴
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_order, fontsize=12, fontfamily='Arial', rotation=15, ha='right')
    ax.set_ylabel('Mean SFC per Network', fontsize=13, fontfamily='Arial')
    ax.set_xlabel('Aphasia Type', fontsize=13, fontfamily='Arial')
    
    if max_val > min_val:
        y_margin = (max_val - min_val) * 0.15
        ax.set_ylim(min_val - y_margin * 0.3, max_val + y_margin)
    
    ax.set_xlim(-0.5, len(plot_order) - 0.5)
    ax.set_title('Left vs Right Hemisphere SFC Comparison (8 Networks)', fontsize=14, fontweight='bold', fontfamily='Arial')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.2)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=left_color, alpha=0.6, label='Left Hemisphere'),
        Patch(facecolor=right_color, alpha=0.6, label='Right Hemisphere')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True, edgecolor='gray')
    
    # 添加显著性说明
    ax.text(0.02, 0.02, '* p<0.05, ** p<0.01, *** p<0.001 (Mann-Whitney U test)', 
            transform=ax.transAxes, fontsize=9, fontfamily='Arial', style='italic')
    
    plt.tight_layout()
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    
    print(f"\n左右脑SFC对比图已保存: {output_path}")


# ==================== 可视化函数 ====================

def create_half_violin(ax, data, position, color, width=0.4, alpha=0.3, horizontal=True):
    """创建半小提琴图"""
    if len(data) < 3:
        return None
    
    try:
        violin_parts = ax.violinplot([data], positions=[position], vert=not horizontal,
                                     showmeans=False, showextrema=False, widths=width)
        
        for pc in violin_parts['bodies']:
            path = pc.get_paths()[0]
            vertices = path.vertices
            
            if horizontal:
                vertices[:, 1] = np.where(vertices[:, 1] > position, vertices[:, 1], position)
            else:
                vertices[:, 0] = np.where(vertices[:, 0] > position, vertices[:, 0], position)
            
            path.vertices = vertices
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(alpha)
            pc.set_linewidth(1.5)
        
        return violin_parts
    except Exception:
        return None


def draw_boxplot_on_violin(ax, data, pos, color, horizontal=True):
    """在半小提琴图上绘制简化箱体图"""
    if len(data) == 0:
        return
    
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    whislo = np.min(data)
    whishi = np.max(data)
    
    box_width = 0.12
    
    if horizontal:
        ax.add_patch(plt.Rectangle((q1, pos - box_width/2),
                                   q3 - q1, box_width,
                                   facecolor='white', edgecolor=color, 
                                   linewidth=1.5, zorder=3))
        ax.plot([median, median], [pos - box_width*0.4, pos + box_width*0.4],
                color=color, linewidth=2, zorder=4)
        ax.plot([q1, whislo], [pos, pos], color=color, linewidth=1.2, zorder=2)
        ax.plot([q3, whishi], [pos, pos], color=color, linewidth=1.2, zorder=2)
        ax.plot([whislo, whislo], [pos - box_width/3, pos + box_width/3], 
                color=color, linewidth=1.2)
        ax.plot([whishi, whishi], [pos - box_width/3, pos + box_width/3], 
                color=color, linewidth=1.2)
    else:
        ax.add_patch(plt.Rectangle((pos - box_width/2, q1),
                                   box_width, q3 - q1,
                                   facecolor='white', edgecolor=color, 
                                   linewidth=1.5, zorder=3))
        ax.plot([pos - box_width*0.4, pos + box_width*0.4], [median, median],
                color=color, linewidth=2, zorder=4)
        ax.plot([pos, pos], [q1, whislo], color=color, linewidth=1.2, zorder=2)
        ax.plot([pos, pos], [q3, whishi], color=color, linewidth=1.2, zorder=2)
        ax.plot([pos - box_width/3, pos + box_width/3], [whislo, whislo], 
                color=color, linewidth=1.2)
        ax.plot([pos - box_width/3, pos + box_width/3], [whishi, whishi], 
                color=color, linewidth=1.2)


def plot_subject_ai_boxplot(category_ai_data, output_path, horizontal=True):
    """
    绘制每个受试者整体AI值的箱型图（皮尔逊相关系数）。
    """
    aphasia_colors = ["#7895C1", "#A8CBDF", "#992224", "#EF8B67", "#E3625D", "#B54764"]
    
    # 提取AI值（只取r值）
    all_data = {}
    for aphasia_type, ai_list in category_ai_data.items():
        ai_values = [r for r, p, n in ai_list]
        all_data[aphasia_type] = np.array(ai_values)
    
    # 设置绘图顺序
    plot_order = []
    if 'Healthy' in all_data:
        plot_order.append('Healthy')
    if 'Normal Stroke' in all_data:
        plot_order.append('Normal Stroke')
    for label in all_data.keys():
        if label not in plot_order:
            plot_order.append(label)
    
    # 创建颜色字典
    color_dict = {}
    color_idx = 0
    for label in plot_order:
        if label == 'Healthy':
            color_dict[label] = '#808080'
        else:
            color_dict[label] = aphasia_colors[color_idx % len(aphasia_colors)]
            color_idx += 1
    
    fig, ax = plt.subplots(figsize=(10, 7))
    rng = np.random.default_rng(seed=42)
    
    positions = list(range(len(plot_order)))
    max_val = -np.inf
    min_val = np.inf
    data_by_group = {}
    healthy_pos = None
    
    violin_width = 0.5
    violin_alpha = 0.35
    
    for pos, label in enumerate(plot_order):
        if label not in all_data:
            continue
        
        data_points = all_data[label]
        if len(data_points) == 0:
            continue
        
        data_by_group[label] = data_points
        max_val = max(max_val, np.max(data_points))
        min_val = min(min_val, np.min(data_points))
        
        if label == 'Healthy':
            healthy_pos = pos
        
        color = color_dict[label]
        
        # 绘制半小提琴
        create_half_violin(ax, data_points, pos, color, 
                         width=violin_width, alpha=violin_alpha, horizontal=horizontal)
        
        # 绘制箱体图
        draw_boxplot_on_violin(ax, data_points, pos, color, horizontal=horizontal)
        
        # 绘制散点
        if horizontal:
            scatter_y = rng.normal(pos - violin_width * 0.35, 0.03, len(data_points))
            ax.scatter(data_points, scatter_y, color=color, alpha=0.6, s=20, 
                      edgecolors='none', zorder=2)
        else:
            scatter_x = rng.normal(pos - violin_width * 0.35, 0.03, len(data_points))
            ax.scatter(scatter_x, data_points, color=color, alpha=0.6, s=20, 
                      edgecolors='none', zorder=2)
    
    # 添加Healthy组的灰色背景
    if healthy_pos is not None:
        spacing = 0.15
        if horizontal:
            y_start = healthy_pos - violin_width / 2 - spacing
            y_end = healthy_pos + violin_width / 2 + spacing
            ax.axhspan(y_start, y_end, facecolor='lightgray', alpha=0.3, zorder=0)
        else:
            x_start = healthy_pos - violin_width / 2 - spacing
            x_end = healthy_pos + violin_width / 2 + spacing
            ax.axvspan(x_start, x_end, facecolor='lightgray', alpha=0.3, zorder=0)
    
    # 添加参考线（r=1表示完全对称）
    if horizontal:
        ax.axvline(x=1.0, color='green', linestyle='--', linewidth=1.2, alpha=0.7, zorder=1, label='Perfect Symmetry')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.0, alpha=0.5, zorder=1, label='No Correlation')
    else:
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.0, alpha=0.5, zorder=1)
    
    # 设置坐标轴
    if horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels(plot_order, fontsize=11, fontfamily='Arial')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='x', labelsize=10)
        if max_val > min_val:
            x_margin = (max_val - min_val) * 0.15
            ax.set_xlim(min_val - x_margin, min(max_val + x_margin, 1.05))
        ax.set_ylim(-0.5, len(plot_order) - 0.5)
        ax.set_xlabel('Symmetry Index (Pearson r)', fontsize=13, fontfamily='Arial')
        ax.set_ylabel('Aphasia Type', fontsize=13, fontfamily='Arial')
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_order, rotation=45, ha='right', fontsize=11, fontfamily='Arial')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='y', labelsize=10)
        if max_val > min_val:
            y_margin = (max_val - min_val) * 0.15
            ax.set_ylim(min_val - y_margin, min(max_val + y_margin, 1.05))
        ax.set_xlim(-0.5, len(plot_order) - 0.5)
        ax.set_ylabel('Symmetry Index (Pearson r)', fontsize=13, fontfamily='Arial')
        ax.set_xlabel('Aphasia Type', fontsize=13, fontfamily='Arial')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.2)
    
    if horizontal:
        ax.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.5)
    else:
        ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    
    print(f"\n受试者AI箱型图已保存: {output_path}")
    
    # 打印统计信息
    print("\n=== AI统计信息（皮尔逊相关系数）===")
    for label in plot_order:
        if label in data_by_group:
            data = data_by_group[label]
            print(f"{label}: n={len(data)}, mean={np.mean(data):.4f}, "
                  f"std={np.std(data):.4f}, median={np.median(data):.4f}")


def plot_network_ai_boxplot(category_network_ai, output_path, horizontal=True):
    """
    绘制每个网络的AI值半小提琴图（多子图布局）。
    """
    aphasia_colors = ["#7895C1", "#A8CBDF", "#992224", "#EF8B67", "#E3625D", "#B54764"]
    
    networks = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis", "Subcortical"]
    
    plot_order = []
    if 'Healthy' in category_network_ai:
        plot_order.append('Healthy')
    if 'Normal Stroke' in category_network_ai:
        plot_order.append('Normal Stroke')
    for label in category_network_ai.keys():
        if label not in plot_order:
            plot_order.append(label)
    
    color_dict = {}
    color_idx = 0
    for label in plot_order:
        if label == 'Healthy':
            color_dict[label] = '#808080'
        else:
            color_dict[label] = aphasia_colors[color_idx % len(aphasia_colors)]
            color_idx += 1
    
    n_networks = len(networks)
    n_cols = 4
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8))
    axes = axes.flatten()
    
    rng = np.random.default_rng(seed=42)
    
    violin_width = 0.5
    violin_alpha = 0.35
    
    for net_idx, network in enumerate(networks):
        ax = axes[net_idx]
        
        positions = list(range(len(plot_order)))
        max_val = -np.inf
        min_val = np.inf
        data_by_group = {}
        healthy_pos = None
        
        for pos, label in enumerate(plot_order):
            if label not in category_network_ai:
                continue
            if network not in category_network_ai[label]:
                continue
            
            # 提取r值
            ai_list = category_network_ai[label][network]
            data_points = np.array([r for r, p, n in ai_list])
            if len(data_points) == 0:
                continue
            
            data_by_group[label] = (pos, data_points)
            max_val = max(max_val, np.max(data_points))
            min_val = min(min_val, np.min(data_points))
            
            if label == 'Healthy':
                healthy_pos = pos
        
        if healthy_pos is not None:
            spacing = 0.15
            if horizontal:
                y_start = healthy_pos - violin_width / 2 - spacing
                y_end = healthy_pos + violin_width / 2 + spacing
                ax.axhspan(y_start, y_end, facecolor='lightgray', alpha=0.3, zorder=0)
            else:
                x_start = healthy_pos - violin_width / 2 - spacing
                x_end = healthy_pos + violin_width / 2 + spacing
                ax.axvspan(x_start, x_end, facecolor='lightgray', alpha=0.3, zorder=0)
        
        for label, (pos, data_points) in data_by_group.items():
            color = color_dict[label]
            
            create_half_violin(ax, data_points, pos, color, 
                             width=violin_width, alpha=violin_alpha, horizontal=horizontal)
            
            draw_boxplot_on_violin(ax, data_points, pos, color, horizontal=horizontal)
            
            if horizontal:
                scatter_y = rng.normal(pos - violin_width * 0.35, 0.03, len(data_points))
                ax.scatter(data_points, scatter_y, color=color, alpha=0.6, s=10, 
                          edgecolors='none', zorder=2)
            else:
                scatter_x = rng.normal(pos - violin_width * 0.35, 0.03, len(data_points))
                ax.scatter(scatter_x, data_points, color=color, alpha=0.6, s=10, 
                          edgecolors='none', zorder=2)
        
        # 添加参考线
        if horizontal:
            ax.axvline(x=1.0, color='green', linestyle='--', linewidth=1.0, alpha=0.6, zorder=1)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.4, zorder=1)
        else:
            ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.0, alpha=0.6, zorder=1)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.4, zorder=1)
        
        if horizontal:
            ax.set_yticks(positions)
            ax.set_yticklabels(plot_order, fontsize=9, fontfamily='Arial')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='x', labelsize=9)
            if max_val > min_val:
                x_margin = (max_val - min_val) * 0.15
                ax.set_xlim(min_val - x_margin, min(max_val + x_margin, 1.05))
            ax.set_ylim(-0.5, len(plot_order) - 0.5)
        else:
            ax.set_xticks(positions)
            ax.set_xticklabels(plot_order, rotation=45, ha='right', fontsize=9, fontfamily='Arial')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='y', labelsize=9)
            if max_val > min_val:
                y_margin = (max_val - min_val) * 0.15
                ax.set_ylim(min_val - y_margin, min(max_val + y_margin, 1.05))
            ax.set_xlim(-0.5, len(plot_order) - 0.5)
        
        ax.set_title(network, fontsize=12, fontweight='bold', fontfamily='Arial')
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.2)
        
        if horizontal:
            ax.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.5)
        else:
            ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
    
    for idx in range(n_networks, n_rows * n_cols):
        axes[idx].axis('off')
    
    if horizontal:
        fig.text(0.5, 0.02, 'Symmetry Index (Pearson r)', ha='center', fontsize=12, fontfamily='Arial')
    else:
        fig.text(0.02, 0.5, 'Symmetry Index (Pearson r)', va='center', rotation='vertical', fontsize=12, fontfamily='Arial')
    
    plt.tight_layout(rect=[0.03, 0.05, 1, 0.98])
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    
    print(f"\n网络AI多子图已保存: {output_path}")


def plot_subcortical_ai_subplots(category_structure_ai, output_path, horizontal=True):
    """
    绘制皮层下结构的AI值半小提琴图（多子图布局）。
    """
    aphasia_colors = ["#7895C1", "#A8CBDF", "#992224", "#EF8B67", "#E3625D", "#B54764"]
    
    structures = ["Hippocampus", "Thalamus", "Putamen", "Caudate", "Amygdala", "NAc", "GP"]
    
    plot_order = []
    if 'Healthy' in category_structure_ai:
        plot_order.append('Healthy')
    if 'Normal Stroke' in category_structure_ai:
        plot_order.append('Normal Stroke')
    for label in category_structure_ai.keys():
        if label not in plot_order:
            plot_order.append(label)
    
    color_dict = {}
    color_idx = 0
    for label in plot_order:
        if label == 'Healthy':
            color_dict[label] = '#808080'
        else:
            color_dict[label] = aphasia_colors[color_idx % len(aphasia_colors)]
            color_idx += 1
    
    n_structures = len(structures)
    n_cols = 4
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8))
    axes = axes.flatten()
    
    rng = np.random.default_rng(seed=42)
    
    violin_width = 0.5
    violin_alpha = 0.35
    
    for struct_idx, structure in enumerate(structures):
        ax = axes[struct_idx]
        
        positions = list(range(len(plot_order)))
        max_val = -np.inf
        min_val = np.inf
        data_by_group = {}
        healthy_pos = None
        
        for pos, label in enumerate(plot_order):
            if label not in category_structure_ai:
                continue
            if structure not in category_structure_ai[label]:
                continue
            
            ai_list = category_structure_ai[label][structure]
            data_points = np.array([r for r, p, n in ai_list])
            if len(data_points) == 0:
                continue
            
            data_by_group[label] = (pos, data_points)
            max_val = max(max_val, np.max(data_points))
            min_val = min(min_val, np.min(data_points))
            
            if label == 'Healthy':
                healthy_pos = pos
        
        if healthy_pos is not None:
            spacing = 0.15
            if horizontal:
                y_start = healthy_pos - violin_width / 2 - spacing
                y_end = healthy_pos + violin_width / 2 + spacing
                ax.axhspan(y_start, y_end, facecolor='lightgray', alpha=0.3, zorder=0)
            else:
                x_start = healthy_pos - violin_width / 2 - spacing
                x_end = healthy_pos + violin_width / 2 + spacing
                ax.axvspan(x_start, x_end, facecolor='lightgray', alpha=0.3, zorder=0)
        
        for label, (pos, data_points) in data_by_group.items():
            color = color_dict[label]
            
            create_half_violin(ax, data_points, pos, color, 
                             width=violin_width, alpha=violin_alpha, horizontal=horizontal)
            
            draw_boxplot_on_violin(ax, data_points, pos, color, horizontal=horizontal)
            
            if horizontal:
                scatter_y = rng.normal(pos - violin_width * 0.35, 0.03, len(data_points))
                ax.scatter(data_points, scatter_y, color=color, alpha=0.6, s=10, 
                          edgecolors='none', zorder=2)
            else:
                scatter_x = rng.normal(pos - violin_width * 0.35, 0.03, len(data_points))
                ax.scatter(scatter_x, data_points, color=color, alpha=0.6, s=10, 
                          edgecolors='none', zorder=2)
        
        if horizontal:
            ax.axvline(x=1.0, color='green', linestyle='--', linewidth=1.0, alpha=0.6, zorder=1)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.4, zorder=1)
        else:
            ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.0, alpha=0.6, zorder=1)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.4, zorder=1)
        
        if horizontal:
            ax.set_yticks(positions)
            ax.set_yticklabels(plot_order, fontsize=9, fontfamily='Arial')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='x', labelsize=9)
            if max_val > min_val:
                x_margin = (max_val - min_val) * 0.15
                ax.set_xlim(min_val - x_margin, min(max_val + x_margin, 1.05))
            ax.set_ylim(-0.5, len(plot_order) - 0.5)
        else:
            ax.set_xticks(positions)
            ax.set_xticklabels(plot_order, rotation=45, ha='right', fontsize=9, fontfamily='Arial')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='y', labelsize=9)
            if max_val > min_val:
                y_margin = (max_val - min_val) * 0.15
                ax.set_ylim(min_val - y_margin, min(max_val + y_margin, 1.05))
            ax.set_xlim(-0.5, len(plot_order) - 0.5)
        
        ax.set_title(structure, fontsize=12, fontweight='bold', fontfamily='Arial')
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.2)
        
        if horizontal:
            ax.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.5)
        else:
            ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
    
    # 最后一个位置用于图例
    legend_ax = axes[-1]
    legend_ax.axis('off')
    
    legend_handles = []
    for label in plot_order:
        legend_color = color_dict[label]
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=legend_color, markersize=10,
                                         label=label, linestyle='None'))
    
    legend_ax.legend(handles=legend_handles, loc='center', frameon=True, 
                     fontsize=11, prop={'family': 'Arial'},
                     title='Aphasia Type', title_fontsize=12,
                     fancybox=True, shadow=False, edgecolor='gray', labelspacing=1.2)
    
    if horizontal:
        fig.text(0.5, 0.02, 'Symmetry Index (Pearson r)', ha='center', fontsize=12, fontfamily='Arial')
    else:
        fig.text(0.02, 0.5, 'Symmetry Index (Pearson r)', va='center', rotation='vertical', fontsize=12, fontfamily='Arial')
    
    plt.tight_layout(rect=[0.03, 0.05, 1, 0.98])
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    
    print(f"\n皮层下结构AI多子图已保存: {output_path}")


# ==================== 保存函数 ====================

def save_subject_ai_to_csv(category_ai_data, output_path):
    """保存受试者AI数据到CSV"""
    rows = []
    for aphasia_type, ai_list in category_ai_data.items():
        for i, (r, p, n) in enumerate(ai_list):
            rows.append({
                'aphasia_type': aphasia_type,
                'subject_index': i + 1,
                'AI_pearson_r': r,
                'p_value': p,
                'n_valid_pairs': n
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"受试者AI值已保存到: {output_path}")


def save_network_ai_to_csv(category_network_ai, output_path, networks):
    """保存网络AI数据到CSV"""
    rows = []
    for aphasia_type, network_data in category_network_ai.items():
        for network in networks:
            if network in network_data:
                for i, (r, p, n) in enumerate(network_data[network]):
                    rows.append({
                        'aphasia_type': aphasia_type,
                        'network': network,
                        'subject_index': i + 1,
                        'AI_pearson_r': r,
                        'p_value': p,
                        'n_valid_pairs': n
                    })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"网络AI值已保存到: {output_path}")


def save_subcortical_ai_to_csv(category_structure_ai, output_path, structures):
    """保存皮层下结构AI数据到CSV"""
    rows = []
    for aphasia_type, structure_data in category_structure_ai.items():
        for structure in structures:
            if structure in structure_data:
                for i, (r, p, n) in enumerate(structure_data[structure]):
                    rows.append({
                        'aphasia_type': aphasia_type,
                        'structure': structure,
                        'subject_index': i + 1,
                        'AI_pearson_r': r,
                        'p_value': p,
                        'n_valid_pairs': n
                    })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"皮层下结构AI值已保存到: {output_path}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 路径配置
    label_file_path = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
    csv_file_path = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\aphasia_plots\participants_aphasia_new.txt"
    sfc_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\sfc_txt_files"
    output_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\single_subject_results_t288\symmetry_index_correlation"
    
    roi_number = 450
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== 1. 计算每个受试者的整体AI ====================
    print("\n" + "="*60)
    print("1. 计算每个受试者的整体对称指数 (皮尔逊相关系数)")
    print("="*60)
    
    print("\n=== 创建左右脑区配对 ===")
    pairs = get_hemisphere_pairs()
    
    print("\n=== 加载受试者类别信息 ===")
    subject_categories = load_subject_categories(csv_file_path)
    
    print("\n=== 加载SFC数据并计算每个受试者的AI值 ===")
    category_ai_data = load_all_subjects_ai(
        sfc_dir, subject_categories, pairs, roi_number
    )
    
    if not category_ai_data:
        print("未找到任何受试者数据！")
        return
    
    # 保存到CSV
    csv_output = os.path.join(output_dir, "subject_AI_correlation_values.csv")
    save_subject_ai_to_csv(category_ai_data, csv_output)
    
    # 绘制箱型图
    print("\n=== 绘制受试者AI箱型图 ===")
    boxplot_output = os.path.join(output_dir, "AI_boxplot_by_aphasia_type.pdf")
    plot_subject_ai_boxplot(category_ai_data, boxplot_output, horizontal=True)
    
    # 森林图
    print("\n=== 绘制森林图 ===")
    forest_output = os.path.join(output_dir, "AI_forest_plot.pdf")
    plot_forest_plot(category_ai_data, forest_output)
    
    # ==================== 2. 按网络计算AI ====================
    print("\n" + "="*60)
    print("2. 按网络计算对称指数")
    print("="*60)
    
    print("\n=== 解析标签文件获取网络信息 ===")
    roi_info = parse_network_from_label(label_file_path)
    print(f"共解析 {len(roi_info)} 个ROI")
    
    network_indices = get_network_hemisphere_indices(roi_info)
    
    print("\n=== 计算每个受试者每个网络的AI值 ===")
    category_network_ai = load_all_subjects_network_ai(
        sfc_dir, subject_categories, network_indices, roi_number
    )
    
    if category_network_ai:
        # 保存到CSV
        networks = list(network_indices.keys())
        network_csv = os.path.join(output_dir, "network_AI_correlation_values.csv")
        save_network_ai_to_csv(category_network_ai, network_csv, networks)
        
        # 绘制多子图
        print("\n=== 绘制网络AI多子图 ===")
        network_output = os.path.join(output_dir, "AI_boxplot_by_network_subplots.pdf")
        plot_network_ai_boxplot(category_network_ai, network_output, horizontal=True)
        
        # 方案3: 雷达图
        print("\n=== 绘制网络AI雷达图 (方案3) ===")
        radar_output = os.path.join(output_dir, "AI_network_radar_chart.pdf")
        plot_radar_chart(category_network_ai, radar_output)
    
    # 新增：左右脑SFC对比图（按8个网络分）
    print("\n=== 加载左右脑SFC数据（按网络分）===")
    category_lr_sfc = load_all_subjects_lr_sfc(
        sfc_dir, subject_categories, network_indices, roi_number
    )
    
    print("\n=== 绘制左右脑SFC对比图 ===")
    lr_comparison_output = os.path.join(output_dir, "LR_SFC_comparison.pdf")
    plot_lr_sfc_comparison(category_lr_sfc, lr_comparison_output)
    
    # ==================== 3. 按皮层下结构计算AI ====================
    print("\n" + "="*60)
    print("3. 按皮层下结构计算对称指数")
    print("="*60)
    
    print("\n=== 获取皮层下结构索引 ===")
    structure_indices = get_subcortical_structure_indices()
    
    print("\n=== 计算每个受试者每个皮层下结构的AI值 ===")
    category_structure_ai = load_all_subjects_subcortical_ai(
        sfc_dir, subject_categories, structure_indices, roi_number
    )
    
    if category_structure_ai:
        # 保存到CSV
        structures = list(structure_indices.keys())
        subcortical_csv = os.path.join(output_dir, "subcortical_structure_AI_correlation_values.csv")
        save_subcortical_ai_to_csv(category_structure_ai, subcortical_csv, structures)
        
        # 绘制多子图
        print("\n=== 绘制皮层下结构AI多子图 ===")
        subcortical_output = os.path.join(output_dir, "AI_boxplot_by_subcortical_structure_subplots.pdf")
        plot_subcortical_ai_subplots(category_structure_ai, subcortical_output, horizontal=True)
    
    print(f"\n" + "="*60)
    print(f"=== 完成 ===")
    print(f"所有结果已保存到: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
