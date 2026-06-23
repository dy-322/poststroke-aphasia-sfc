import numpy as np
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
import pandas as pd
import tempfile
import nibabel as nib
from matplotlib import gridspec
from matplotlib.colorbar import ColorbarBase
from nilearn import plotting, datasets
from surfplot import Plot
from brainspace.datasets import load_parcellation
from brainspace.mesh.mesh_io import read_surface
from neuromaps.datasets import fetch_fslr
from statsmodels.stats.multitest import multipletests
from neurocat.plotting import draw_and_save as ds
import warnings

CORTICAL_OFFSET = 50  # 前50个ROI为皮层下
N_TOTAL_ROIS = 450
N_CORTICAL_ROIS = N_TOTAL_ROIS - CORTICAL_OFFSET
SUBCORT_ATLAS_PATH = r"D:\download\subcortex_visualization-main\subcortex_visualization-main\atlas_info\Melbourne_S3\Tian_Subcortex_S3_3T_1mm.nii.gz"
BG_IMG_PATH = r"D:\download\MNI152_T1_1mm_brain.nii.gz"
COMBINED_ATLAS_PATH = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_MNI152NLin6Asym_1mm.nii.gz"
NETWORK_LABEL_FILE = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"

warnings.filterwarnings('ignore')


# 从plot_sfc_browser.py导入必要的函数
def load_subject_categories(csv_file_path):
    """从新的 TSV 文件读取受试者（只对含 M 的受试者）失语类别映射，与 plot_sfc_browser.py 保持一致"""
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
            
            # 检查 participant_id 是否有效
            if not isinstance(participant_id, str):
                continue
            
            participant_id = participant_id.strip()
            if not participant_id:
                continue
            
            # 处理 aphasia_type：如果是 NaN 或空字符串，将其设置为 "None" 类别
            if pd.isna(aphasia_type) or str(aphasia_type).strip() == '':
                aphasia_type = 'None'
            else:
                aphasia_type = str(aphasia_type).strip()
            
            # 跳过特定类别，但保留 "None" 类别
            if aphasia_type in {"TranscorticalMotor", "Wernicke"}:
                continue

            subject_categories[participant_id] = aphasia_type

        # 统计各类别的数量
        category_counts = {}
        for aphasia_type in subject_categories.values():
            category_counts[aphasia_type] = category_counts.get(aphasia_type, 0) + 1
        
        print(f"从 TSV 成功解析 {len(subject_categories)} 个受试者类别映射")
        print(f"类别分布: {category_counts}")
        if 'None' in category_counts:
            print(f"注意: 找到 {category_counts['None']} 个 'None' 类别的受试者")
        
        return subject_categories

    except Exception as e:
        print(f"读取 TSV 时发生错误: {str(e)}")
        return {}




def _extract_subject_metadata_from_fname(fname):
    """
    根据文件名提取受试者基础信息：
    - subject_label: 原始标签（例如 sub-sub-3126）
    - csv_key: 若为失语受试者，返回用于 CSV 匹配的 sub-MXXXX；健康受试者为 None
    """
    if not fname.endswith("_sfc_values_pca.txt"):
        return None

    stem = fname[:-len("_sfc_values_pca.txt")]
    subject_label = stem

    # 拆分并寻找带 M 的 token，例如 sub-sub-M2014 -> M2014
    csv_key = None
    tokens = stem.replace('_', '-').split('-')
    for token in tokens:
        if token.startswith('M') and token[1:].isdigit():
            csv_key = f"sub-{token}"
            break

    return {
        'subject_label': subject_label,
        'csv_key': csv_key,
    }


def load_all_subjects_from_sfc_dir(sfc_dir):
    """扫描 SFC 目录，返回受试者列表及其 CSV 匹配信息"""
    subjects = []

    for fname in os.listdir(sfc_dir):
        metadata = _extract_subject_metadata_from_fname(fname)
        if metadata is None:
            continue
        metadata['filename'] = fname
        metadata['filepath'] = os.path.join(sfc_dir, fname)
        metadata['is_healthy'] = metadata['csv_key'] is None
        subjects.append(metadata)

    subject_labels = [item['subject_label'] for item in subjects]
    print(f"找到 {len(subject_labels)} 个受试者: {sorted(subject_labels)}")
    return subjects


def load_sfc_data_by_category(sfc_dir, subject_categories):
    """结合目录扫描结果与 CSV 分类，读取各受试者的 SFC 数据，与 plot_sfc_browser.py 保持一致"""
    roi_number = 450
    category_data = {}
    
    # 首先从SFC目录获取所有受试者列表
    subject_entries = load_all_subjects_from_sfc_dir(sfc_dir)
    
    for entry in subject_entries:
        subject_label = entry['subject_label']
        filepath = entry['filepath']

        if entry['is_healthy']:
            aphasia_type = '9'
        else:
            csv_key = entry['csv_key']
            aphasia_type = subject_categories.get(csv_key)
            if aphasia_type is None:
                print(f"Warning: CSV 中缺少 {csv_key} 的分类信息，跳过 {subject_label}")
                continue

        if not os.path.exists(filepath):
            print(f"Warning: 文件不存在 {filepath}")
            continue

        try:
            sfc_values = []
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for line in lines[3:]:
                    if line.strip() and '\t' in line:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            try:
                                sfc_val = float(parts[1])
                                sfc_values.append(sfc_val)
                            except ValueError:
                                if parts[1].strip().lower() == 'nan':
                                    sfc_values.append(np.nan)
                                else:
                                    print(f"Warning: 无法解析 {subject_label} 的数值: {parts[1]}")
                                    sfc_values.append(np.nan)

            if len(sfc_values) == roi_number:
                if aphasia_type not in category_data:
                    category_data[aphasia_type] = []
                category_data[aphasia_type].append({
                    'subject_id': subject_label,
                    'sfc_values': np.array(sfc_values)
                })
                print(f"Loaded SFC data for {subject_label}, category {aphasia_type}")
            else:
                print(f"Warning: {subject_label} ROI 数量为 {len(sfc_values)}，应为 {roi_number}")

        except Exception as e:
            print(f"Error loading SFC data for {subject_label}: {str(e)}")

    # 显示加载的类别统计
    print(f"\n按类别加载的 SFC 数据统计:")
    for aphasia_type, subjects in category_data.items():
        print(f"  类别 '{aphasia_type}': {len(subjects)} 个受试者")
    if 'None' in category_data:
        print(f"  注意: 'None' 类别包含 {len(category_data['None'])} 个受试者")

    return category_data


def compute_category_average_sfc(category_data):
    """计算每个类别的平均SFC值"""
    category_averages = {}

    for aphasia_type, subjects in category_data.items():
        if subjects:
            # 收集所有受试者的SFC数据
            all_sfc_arrays = [subject['sfc_values'] for subject in subjects]

            # 转为二维数组: (n_subjects, n_rois)
            all_sfc_matrix = np.vstack(all_sfc_arrays)

            # 计算平均值（忽略NaN值）
            avg_sfc = np.nanmean(all_sfc_matrix, axis=0)

            category_averages[aphasia_type] = {
                'average_sfc': avg_sfc,
                'n_subjects': len(subjects),
                'subject_ids': [subject['subject_id'] for subject in subjects],
                'all_sfc_matrix': all_sfc_matrix  # 保存原始数据用于统计检验
            }

            print(f"失语症类型{aphasia_type}: {len(subjects)}个受试者，平均SFC已计算")

    return category_averages


def statistical_test_category_vs_8(category_data, target_category='9', alpha=0.05):
    """
    直接比较每个类别与类别8的SFC差异，进行ROI级别的组间统计检验

    参数:
    - category_data: 包含所有类别原始SFC数据的字典
    - target_category: 目标比较类别（默认为'9'）
    - alpha: 显著性水平（默认为0.05）

    返回:
    - significant_regions: 每个类别与类别9有显著差异的脑区
    - p_values: 每个类别每个脑区的p值
    """

    if target_category not in category_data:
        print(f"错误：目标类别{target_category}不在数据中")
        return None, None

    # 获取目标类别的所有受试者SFC数据
    target_subjects = category_data[target_category]
    target_data = np.vstack([subject['sfc_values'] for subject in target_subjects])

    significant_regions = {}
    p_values = {}

    print(f"\n=== 与类别{target_category}的统计比较 ===")
    print(f"类别{target_category}的受试者数量: {len(target_subjects)}")

    for aphasia_type, subjects in category_data.items():
        if aphasia_type == target_category:
            continue

        print(f"\n比较类别{aphasia_type} vs 类别{target_category}")
        print(f"类别{aphasia_type}的受试者数量: {len(subjects)}")

        # 获取当前类别的所有受试者SFC数据
        current_data = np.vstack([subject['sfc_values'] for subject in subjects])
        roi_p_values = []
        significant_rois = []

        # 对每个ROI进行统计检验
        for roi_idx in range(current_data.shape[1]):
            current_roi_data = current_data[:, roi_idx]
            target_roi_data = target_data[:, roi_idx]

            current_valid = current_roi_data[~np.isnan(current_roi_data)]
            target_valid = target_roi_data[~np.isnan(target_roi_data)]

            if len(current_valid) < 3 or len(target_valid) < 3:
                # 样本量太小，无法进行统计检验
                p_val = 1.0
            else:
                # 使用Mann-Whitney U检验（非参数检验）
                try:
                    statistic, p_val = stats.mannwhitneyu(
                        current_valid, target_valid,
                        alternative='two-sided'
                    )
                except:
                    p_val = 1.0

            roi_p_values.append(p_val)

            if p_val < alpha:
                significant_rois.append(roi_idx)

        # 多重比较校正（FDR）
        roi_p_values = np.array(roi_p_values)
        rejected, p_corrected, _, _ = multipletests(roi_p_values, alpha=alpha, method='fdr_bh')

        # 更新显著脑区（使用校正后的p值）
        significant_rois_corrected = np.where(rejected)[0].tolist()

        significant_regions[aphasia_type] = {
            'significant_rois': significant_rois_corrected,
            'uncorrected_rois': significant_rois,
            'n_significant': len(significant_rois_corrected),
            'n_uncorrected': len(significant_rois)
        }

        p_values[aphasia_type] = {
            'p_values': roi_p_values,
            'p_corrected': p_corrected
        }

        print(f"类别{aphasia_type} vs 类别{target_category}:")
        print(f"  未校正显著脑区数量: {len(significant_rois)}")
        print(f"  FDR校正后显著脑区数量: {len(significant_rois_corrected)}")
        if len(significant_rois_corrected) > 0:
            print(
                f"  显著脑区索引: {significant_rois_corrected[:10]}{'...' if len(significant_rois_corrected) > 10 else ''}")

    return significant_regions, p_values


def create_significance_map(significant_regions, n_rois=400):
    """
    创建显著差异脑区的二值化地图
    """
    significance_maps = {}

    for aphasia_type, data in significant_regions.items():
        # 创建二值化地图：1表示显著差异，0表示无显著差异
        sig_map = np.zeros(n_rois)
        sig_map[data['significant_rois']] = 1
        significance_maps[aphasia_type] = sig_map

    return significance_maps


def calculate_cohens_d_for_significant_regions(category_data, significant_regions, target_category='9'):
    """
    为显著脑区计算Cohen's d效应量
    
    参数:
    - category_data: 包含所有类别原始SFC数据的字典
    - significant_regions: 显著脑区信息
    - target_category: 目标比较类别（默认为'9'）
    
    返回:
    - cohens_d_results: 每个类别的Cohen's d效应量结果
    """
    
    if target_category not in category_data:
        print(f"错误：目标类别{target_category}不在数据中")
        return None
    
    print(f"\n=== 计算显著脑区的Cohen's d效应量 ===")
    
    # 准备目标类别的数据
    target_subjects = category_data[target_category]
    target_data = np.vstack([subject['sfc_values'] for subject in target_subjects])
    
    print(f"目标类别{target_category}的受试者数量: {len(target_subjects)}")
    
    cohens_d_results = {}
    
    for aphasia_type, sig_data in significant_regions.items():
        if aphasia_type == target_category:
            continue
            
        print(f"\n计算类别{aphasia_type} vs 类别{target_category}的Cohen's d...")
        
        # 准备当前类别的数据
        current_subjects = category_data[aphasia_type]
        current_data = np.vstack([subject['sfc_values'] for subject in current_subjects])
        
        print(f"类别{aphasia_type}的受试者数量: {len(current_subjects)}")
        
        # 获取显著脑区索引
        significant_rois = sig_data['significant_rois']
        effect_vector = np.full(N_TOTAL_ROIS, np.nan)
        
        if len(significant_rois) == 0:
            print(f"类别{aphasia_type}没有显著脑区，跳过Cohen's d计算")
            cohens_d_results[aphasia_type] = {
                'cohens_d_values': [],
                'significant_rois': [],
                'n_significant': 0,
                'effect_vector': effect_vector,
                'subject_ids': [subject['subject_id'] for subject in current_subjects]
            }
            continue
        
        # 计算显著脑区的Cohen's d
        cohens_d_values = []
        valid_rois = []
        
        for roi_idx in significant_rois:
            if roi_idx >= current_data.shape[1] or roi_idx >= target_data.shape[1]:
                continue
                
            # 获取当前ROI的数据
            current_roi_data = current_data[:, roi_idx]
            target_roi_data = target_data[:, roi_idx]
            
            # 去除NaN值
            current_valid = current_roi_data[~np.isnan(current_roi_data)]
            target_valid = target_roi_data[~np.isnan(target_roi_data)]
            
            if len(current_valid) < 2 or len(target_valid) < 2:
                continue
            
            # 计算Cohen's d
            # Cohen's d = (mean1 - mean2) / pooled_standard_deviation
            mean_current = np.mean(current_valid)
            mean_target = np.mean(target_valid)
            
            # 计算合并标准差
            n_current = len(current_valid)
            n_target = len(target_valid)
            
            var_current = np.var(current_valid, ddof=1)  # 样本方差
            var_target = np.var(target_valid, ddof=1)
            
            pooled_std = np.sqrt(((n_current - 1) * var_current + (n_target - 1) * var_target) / 
                               (n_current + n_target - 2))
            
            if pooled_std > 0:
                cohens_d = (mean_current - mean_target) / pooled_std
                cohens_d_values.append(cohens_d)
                valid_rois.append(roi_idx)
                effect_vector[roi_idx] = cohens_d
        
        cohens_d_results[aphasia_type] = {
            'cohens_d_values': cohens_d_values,
            'significant_rois': valid_rois,
            'n_significant': len(valid_rois),
            'effect_vector': effect_vector,
            'subject_ids': [subject['subject_id'] for subject in current_subjects]
        }
        
        if len(cohens_d_values) > 0:
            print(f"类别{aphasia_type} vs 类别{target_category}:")
            print(f"  有效显著脑区数量: {len(cohens_d_values)}")
            print(f"  平均Cohen's d: {np.mean(cohens_d_values):.3f}")
            print(f"  Cohen's d范围: [{np.min(cohens_d_values):.3f}, {np.max(cohens_d_values):.3f}]")
            
            # 效应量解释
            positive_effects = np.sum(np.array(cohens_d_values) > 0)
            negative_effects = np.sum(np.array(cohens_d_values) < 0)
            print(f"  正效应脑区数量 (当前类别>目标类别): {positive_effects}")
            print(f"  负效应脑区数量 (当前类别<目标类别): {negative_effects}")
        else:
            print(f"类别{aphasia_type}没有有效的Cohen's d计算")
    
    return cohens_d_results


def calculate_cohens_d_for_all_regions(category_data, target_category='9'):
    """
    为所有脑区计算Cohen's d效应量（不仅仅是显著脑区）
    
    参数:
    - category_data: 包含所有类别原始SFC数据的字典
    - target_category: 目标比较类别（默认为'9'）
    
    返回:
    - cohens_d_results: 每个类别的Cohen's d效应量结果
    """
    
    if target_category not in category_data:
        print(f"错误：目标类别{target_category}不在数据中")
        return None
    
    print(f"\n=== 计算所有脑区的Cohen's d效应量 ===")
    
    # 准备目标类别的数据
    target_subjects = category_data[target_category]
    target_data = np.vstack([subject['sfc_values'] for subject in target_subjects])
    
    print(f"目标类别{target_category}的受试者数量: {len(target_subjects)}")
    
    cohens_d_results = {}
    
    for aphasia_type, subjects in category_data.items():
        if aphasia_type == target_category:
            continue
            
        print(f"\n计算类别{aphasia_type} vs 类别{target_category}的所有脑区Cohen's d...")
        
        # 准备当前类别的数据
        current_data = np.vstack([subject['sfc_values'] for subject in subjects])
        
        print(f"类别{aphasia_type}的受试者数量: {len(subjects)}")
        
        # 为所有脑区计算Cohen's d
        effect_vector = np.full(N_TOTAL_ROIS, np.nan)
        cohens_d_values = []
        valid_rois = []
        
        for roi_idx in range(N_TOTAL_ROIS):
            if roi_idx >= current_data.shape[1] or roi_idx >= target_data.shape[1]:
                continue
                
            # 获取当前ROI的数据
            current_roi_data = current_data[:, roi_idx]
            target_roi_data = target_data[:, roi_idx]
            
            # 去除NaN值
            current_valid = current_roi_data[~np.isnan(current_roi_data)]
            target_valid = target_roi_data[~np.isnan(target_roi_data)]
            
            if len(current_valid) < 2 or len(target_valid) < 2:
                continue
            
            # 计算Cohen's d
            # Cohen's d = (mean1 - mean2) / pooled_standard_deviation
            mean_current = np.mean(current_valid)
            mean_target = np.mean(target_valid)
            
            # 计算合并标准差
            n_current = len(current_valid)
            n_target = len(target_valid)
            
            var_current = np.var(current_valid, ddof=1)  # 样本方差
            var_target = np.var(target_valid, ddof=1)
            
            pooled_std = np.sqrt(((n_current - 1) * var_current + (n_target - 1) * var_target) / 
                               (n_current + n_target - 2))
            
            if pooled_std > 0:
                cohens_d = (mean_current - mean_target) / pooled_std
                cohens_d_values.append(cohens_d)
                valid_rois.append(roi_idx)
                effect_vector[roi_idx] = cohens_d
        
        cohens_d_results[aphasia_type] = {
            'cohens_d_values': cohens_d_values,
            'all_rois': valid_rois,
            'n_rois': len(valid_rois),
            'effect_vector': effect_vector,
            'subject_ids': [subject['subject_id'] for subject in subjects]
        }
        
        if len(cohens_d_values) > 0:
            print(f"类别{aphasia_type} vs 类别{target_category}:")
            print(f"  有效脑区数量: {len(cohens_d_values)}")
            print(f"  平均Cohen's d: {np.mean(cohens_d_values):.3f}")
            print(f"  Cohen's d范围: [{np.min(cohens_d_values):.3f}, {np.max(cohens_d_values):.3f}]")
            
            # 效应量解释
            positive_effects = np.sum(np.array(cohens_d_values) > 0)
            negative_effects = np.sum(np.array(cohens_d_values) < 0)
            print(f"  正效应脑区数量 (当前类别>目标类别): {positive_effects}")
            print(f"  负效应脑区数量 (当前类别<目标类别): {negative_effects}")
        else:
            print(f"类别{aphasia_type}没有有效的Cohen's d计算")
    
    return cohens_d_results


def calculate_cohens_d_for_all_regions(category_data, p_values_dict=None, target_category='9'):
    """
    为所有脑区计算Cohen's d效应量（不仅限于显著脑区），并包含p值信息
    
    参数:
    - category_data: 包含所有类别原始SFC数据的字典
    - p_values_dict: p值字典（可选），包含每个类别的p值信息
    - target_category: 目标比较类别（默认为'9'）
    
    返回:
    - cohens_d_results: 每个类别的Cohen's d效应量结果（包含p值）
    """
    
    if target_category not in category_data:
        print(f"错误：目标类别{target_category}不在数据中")
        return None
    
    print(f"\n=== 计算所有脑区的Cohen's d效应量 ===")
    
    # 准备目标类别的数据
    target_subjects = category_data[target_category]
    target_data = np.vstack([subject['sfc_values'] for subject in target_subjects])
    
    print(f"目标类别{target_category}的受试者数量: {len(target_subjects)}")
    
    cohens_d_results = {}
    
    for aphasia_type in category_data.keys():
        if aphasia_type == target_category:
            continue
            
        print(f"\n计算类别{aphasia_type} vs 类别{target_category}的所有脑区Cohen's d...")
        
        # 准备当前类别的数据
        current_subjects = category_data[aphasia_type]
        current_data = np.vstack([subject['sfc_values'] for subject in current_subjects])
        
        print(f"类别{aphasia_type}的受试者数量: {len(current_subjects)}")
        
        effect_vector = np.full(N_TOTAL_ROIS, np.nan)
        p_value_vector = np.full(N_TOTAL_ROIS, np.nan)
        p_corrected_vector = np.full(N_TOTAL_ROIS, np.nan)
        cohens_d_values = []
        valid_rois = []
        
        # 获取p值信息（如果提供）
        p_info = None
        if p_values_dict and aphasia_type in p_values_dict:
            p_info = p_values_dict[aphasia_type]
        
        # 对所有ROI进行Cohen's d计算
        for roi_idx in range(min(current_data.shape[1], target_data.shape[1])):
            # 获取当前ROI的数据
            current_roi_data = current_data[:, roi_idx]
            target_roi_data = target_data[:, roi_idx]
            
            # 去除NaN值
            current_valid = current_roi_data[~np.isnan(current_roi_data)]
            target_valid = target_roi_data[~np.isnan(target_roi_data)]
            
            if len(current_valid) < 2 or len(target_valid) < 2:
                continue
            
            # 计算Cohen's d
            mean_current = np.mean(current_valid)
            mean_target = np.mean(target_valid)
            
            # 计算合并标准差
            n_current = len(current_valid)
            n_target = len(target_valid)
            
            var_current = np.var(current_valid, ddof=1)
            var_target = np.var(target_valid, ddof=1)
            
            pooled_std = np.sqrt(((n_current - 1) * var_current + (n_target - 1) * var_target) / 
                               (n_current + n_target - 2))
            
            if pooled_std > 0:
                cohens_d = (mean_current - mean_target) / pooled_std
                cohens_d_values.append(cohens_d)
                valid_rois.append(roi_idx)
                effect_vector[roi_idx] = cohens_d
                
                # 添加p值信息
                if p_info is not None:
                    if roi_idx < len(p_info['p_values']):
                        p_value_vector[roi_idx] = p_info['p_values'][roi_idx]
                    if roi_idx < len(p_info['p_corrected']):
                        p_corrected_vector[roi_idx] = p_info['p_corrected'][roi_idx]
        
        cohens_d_results[aphasia_type] = {
            'cohens_d_values': cohens_d_values,
            'valid_rois': valid_rois,
            'n_valid': len(valid_rois),
            'effect_vector': effect_vector,
            'p_value_vector': p_value_vector,
            'p_corrected_vector': p_corrected_vector,
            'subject_ids': [subject['subject_id'] for subject in current_subjects]
        }
        
        if len(cohens_d_values) > 0:
            print(f"类别{aphasia_type} vs 类别{target_category}:")
            print(f"  有效脑区数量: {len(cohens_d_values)}")
            print(f"  平均Cohen's d: {np.mean(cohens_d_values):.3f}")
            print(f"  Cohen's d范围: [{np.min(cohens_d_values):.3f}, {np.max(cohens_d_values):.3f}]")
            
            # 效应量解释
            positive_effects = np.sum(np.array(cohens_d_values) > 0)
            negative_effects = np.sum(np.array(cohens_d_values) < 0)
            print(f"  正效应脑区数量 (当前类别>目标类别): {positive_effects}")
            print(f"  负效应脑区数量 (当前类别<目标类别): {negative_effects}")
        else:
            print(f"类别{aphasia_type}没有有效的Cohen's d计算")
    
    return cohens_d_results


def trim_white_edges(img_array):
    """去除图像白边。"""
    if img_array is None:
        return None
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


def parse_network_label_file(label_file_path):
    """解析七网络标签文件，返回 {network_name: [cortical_roi_idx]}。"""
    if not label_file_path or not os.path.exists(label_file_path):
        print(f"提示：未找到七网络标签文件 {label_file_path}")
        return {}, []
    try:
        with open(label_file_path, 'r') as f:
            raw_lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"读取网络标签文件失败: {str(e)}")
        return {}, []

    if len(raw_lines) % 2 != 0:
        print("警告：网络标签文件存在不完整条目")

    network_to_indices = {}
    idx = 0
    while idx + 1 < len(raw_lines):
        roi_name = raw_lines[idx]
        data_line = raw_lines[idx + 1]
        idx += 2

        parts = data_line.split()
        if not parts:
            continue
        try:
            roi_label = int(parts[0])
        except ValueError:
            continue

        if roi_label <= CORTICAL_OFFSET:
            continue
        if not roi_name.startswith("7Networks"):
            continue

        name_parts = roi_name.split('_')
        if len(name_parts) < 3:
            continue
        network_name = name_parts[2]
        # 将51-450映射到0-399（皮层ROI索引）
        cortical_index = roi_label - 51
        if 0 <= cortical_index < N_CORTICAL_ROIS:
            network_to_indices.setdefault(network_name, []).append(cortical_index)

    network_names = sorted(network_to_indices.keys())
    if network_names:
        print(f"解析七网络成功，共{len(network_names)}个网络")
    else:
        print("警告：未能解析到任何网络映射")
    return network_to_indices, network_names


def compute_network_cohens_d(effect_vector, network_to_indices):
    """计算七个网络的平均Cohen's d。"""
    network_means = {}
    if effect_vector is None or not network_to_indices:
        return network_means
    for network, cortical_indices in network_to_indices.items():
        global_indices = [
            CORTICAL_OFFSET + idx for idx in cortical_indices
            if 0 <= idx < N_CORTICAL_ROIS and (CORTICAL_OFFSET + idx) < len(effect_vector)
        ]
        if not global_indices:
            network_means[network] = np.nan
            continue
        values = effect_vector[global_indices]
        if values.size == 0 or np.all(np.isnan(values)):
            network_means[network] = np.nan
        else:
            network_means[network] = float(np.nanmean(values))
    return network_means


def determine_color_range(cohens_d_results, significant_regions=None):
    """
    根据所有类别自动确定对称颜色范围，用于皮层下图和colorbar。
    如果提供了显著脑区信息，则只基于显著脑区的值计算。
    """
    max_abs = 0.0
    for aphasia_type, data in cohens_d_results.items():
        effect_vector = data.get('effect_vector')
        if effect_vector is None:
            continue
        
        # 如果提供了显著脑区信息，只取显著脑区的值
        if significant_regions and aphasia_type in significant_regions:
            significant_rois = significant_regions[aphasia_type].get('significant_rois', [])
            if len(significant_rois) > 0:
                # 只取显著脑区的值
                sig_values = []
                for roi_idx in significant_rois:
                    if 0 <= roi_idx < len(effect_vector):
                        if not np.isnan(effect_vector[roi_idx]):
                            sig_values.append(effect_vector[roi_idx])
                if len(sig_values) > 0:
                    cand = float(np.nanmax(np.abs(sig_values)))
                    if cand > max_abs:
                        max_abs = cand
        else:
            # 如果没有显著脑区信息，取所有值
            valid = effect_vector[~np.isnan(effect_vector)]
            if valid.size > 0:
                cand = float(np.nanmax(np.abs(valid)))
                if cand > max_abs:
                    max_abs = cand
    if max_abs == 0:
        max_abs = 0.1
    return [-max_abs, max_abs]


def determine_surface_color_range(cohens_d_results, significant_regions=None):
    """
    根据所有类别的皮层数据确定颜色范围（全局最小值和最大值），用于表面图。
    如果提供了显著脑区信息，则只基于显著脑区的值计算。
    """
    all_cortical_values = []
    for aphasia_type, data in cohens_d_results.items():
        effect_vector = data.get('effect_vector')
        if effect_vector is None:
            continue
        
        # 如果提供了显著脑区信息，只取显著脑区的值
        if significant_regions and aphasia_type in significant_regions:
            significant_rois = significant_regions[aphasia_type].get('significant_rois', [])
            if len(significant_rois) > 0:
                # 只取显著脑区中的皮层数据
                for roi_idx in significant_rois:
                    if roi_idx >= CORTICAL_OFFSET and roi_idx < len(effect_vector):
                        if not np.isnan(effect_vector[roi_idx]):
                            all_cortical_values.append(effect_vector[roi_idx])
        else:
            # 如果没有显著脑区信息，取所有皮层数据
            cortical_values = effect_vector[CORTICAL_OFFSET:]
            valid = cortical_values[~np.isnan(cortical_values)]
            if valid.size > 0:
                all_cortical_values.extend(valid.tolist())
    
    if len(all_cortical_values) == 0:
        return [-0.1, 0.1]
    
    vmin = float(np.min(all_cortical_values))
    vmax = float(np.max(all_cortical_values))
    
    # 如果最小值和最大值相同，设置一个小的范围
    if vmin >= vmax:
        vmax = vmin + 0.1
    
    return [vmin, vmax]


def create_white_magma_colormap(base_cmap='magma_r', white_fraction=0.1):
    """
    创建一个自定义colormap，在最小值处是白色，然后过渡到base_cmap的颜色。
    
    参数:
    - base_cmap: 基础colormap名称
    - white_fraction: 白色区域占colormap的比例（0-1之间）
    
    返回:
    - 自定义colormap
    """
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    
    # 获取基础colormap（兼容新旧版本matplotlib）
    try:
        base = plt.get_cmap(base_cmap)
    except:
        import matplotlib.cm as cm
        base = cm.get_cmap(base_cmap)
    
    # 创建颜色数组
    n_colors = 256
    colors = []
    
    # 白色部分
    n_white = int(n_colors * white_fraction)
    for i in range(n_white):
        colors.append((1.0, 1.0, 1.0, 1.0))  # 白色
    
    # 过渡到base colormap
    n_base = n_colors - n_white
    for i in range(n_base):
        # 从0开始映射到base colormap
        colors.append(base(i / n_base))
    
    return LinearSegmentedColormap.from_list('white_' + base_cmap, colors, N=n_colors)


def create_effect_size_plot(effect_vector,
                            color_range,
                            output_path,
                            title,
                            atlas_img_path=COMBINED_ATLAS_PATH,
                            cmap='magma_r',
                            surface_color_range=None,
                            significant_rois=None):
    """
    绘制表面图+皮层下图的Cohen's d，使用magma_r配色，保存为PDF。
    如果提供了显著脑区列表，则只显示显著脑区。
    
    参数:
    - effect_vector: Cohen's d向量
    - color_range: 用于皮层下图和colorbar的颜色范围
    - surface_color_range: 用于表面图的颜色范围（如果为None，则使用color_range）
    - significant_rois: 显著脑区的索引列表（如果提供，则只显示这些脑区）
    """
    if effect_vector is None or len(effect_vector) == 0:
        return False

    # 如果提供了显著脑区列表，创建掩码：显著脑区保留原值，非显著脑区设置为特殊值（用于显示白色）
    white_value = None
    if significant_rois is not None and len(significant_rois) > 0:
        # 计算一个比color range最小值更小的值，用于表示非显著脑区（将显示为白色）
        # 使用surface_color_range来计算white_value，因为表面图使用这个range
        surface_range = surface_color_range if surface_color_range is not None else color_range
        white_value = surface_range[0] - (surface_range[1] - surface_range[0]) * 0.1  # 比最小值小10%的范围
        
        masked_effect_vector = np.full(len(effect_vector), white_value, dtype=float)  # 默认填充为白色值
        for roi_idx in significant_rois:
            if 0 <= roi_idx < len(effect_vector):
                if not np.isnan(effect_vector[roi_idx]):
                    masked_effect_vector[roi_idx] = effect_vector[roi_idx]
        effect_vector = masked_effect_vector
    else:
        # 如果没有显著脑区列表，使用原始值
        pass

    cortical_values = effect_vector[CORTICAL_OFFSET:]
    subcortical_values = effect_vector[:CORTICAL_OFFSET]
    if np.all(np.isnan(cortical_values)) and np.all(np.isnan(subcortical_values)):
        print("提示：该类别没有可视化所需的Cohen's d，已跳过")
        return False

    # 设置matplotlib字体为Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # 准备皮层下数据
    if not atlas_img_path or not os.path.exists(atlas_img_path):
        raise FileNotFoundError(f"未找到皮层下atlas: {atlas_img_path}")

    atlas_img = nib.load(atlas_img_path)
    atlas_data = atlas_img.get_fdata().astype(int)
    roi_vol = np.full(atlas_data.shape, np.nan, dtype=float)
    max_sub = min(CORTICAL_OFFSET, len(subcortical_values))
    
    # 计算皮层下的white_value
    subcort_white_value = None
    if white_value is not None:
        subcort_white_value = color_range[0] - (color_range[1] - color_range[0]) * 0.1
    
    for lbl in range(1, max_sub + 1):
        roi_idx = lbl - 1  # 皮层下ROI索引（0-49）
        if not np.isnan(subcortical_values[roi_idx]):
            # 显著脑区，使用原值
            roi_vol[atlas_data == lbl] = subcortical_values[roi_idx]
        elif subcort_white_value is not None:
            # 非显著脑区，填充为白色值
            roi_vol[atlas_data == lbl] = subcort_white_value

    roi_img = nib.Nifti1Image(roi_vol, atlas_img.affine, atlas_img.header)
    bg_img = datasets.load_mni152_template(resolution=1)
    bg_data = bg_img.get_fdata()
    bg_img_final = nib.Nifti1Image(bg_data, bg_img.affine, bg_img.header)

    # 绘制皮层下（nilearn）
    # 如果有white_value，扩展color range以包含它，并创建自定义colormap
    plot_subcort_range = color_range.copy()
    plot_subcort_cmap = cmap
    if subcort_white_value is not None:
        plot_subcort_range[0] = subcort_white_value
        plot_subcort_cmap = create_white_magma_colormap(base_cmap=cmap, white_fraction=0.05)
    
    fig_nilearn = plt.figure(figsize=(40, 8))
    nilearn_disp = plotting.plot_stat_map(
        roi_img,
        bg_img=bg_img_final,
        threshold=0.0,
        cmap=plot_subcort_cmap,
        vmax=plot_subcort_range[1],
        vmin=plot_subcort_range[0],
        display_mode='z',
        cut_coords=[-25, -20, -10, 0, 10, 20, 22],
        draw_cross=False,
        annotate=False,
        colorbar=False,
        figure=fig_nilearn,
        black_bg=False,
    )

    # 保存nilearn图像到临时文件
    temp_dir = tempfile.gettempdir()
    tmp_nilearn = os.path.join(temp_dir, f'temp_nilearn_{os.getpid()}.png')
    fig_nilearn.savefig(tmp_nilearn, bbox_inches='tight', dpi=300, pad_inches=0)
    nilearn_disp.close()
    plt.close(fig_nilearn)

    # 读取nilearn图像尺寸
    img_nilearn_temp = mpimg.imread(tmp_nilearn)
    nilearn_width = img_nilearn_temp.shape[1]
    nilearn_height = img_nilearn_temp.shape[0]

    # 使用neurocat绘制表面图
    temp_surface = os.path.join(temp_dir, f'temp_surface_{os.getpid()}.png')
    
    # 如果提供了surface_color_range，使用它；否则使用color_range
    surface_range = surface_color_range if surface_color_range is not None else color_range
    
    # 如果有white_value，扩展color range以包含它，并创建自定义colormap
    plot_surface_range = surface_range.copy()
    plot_cmap = cmap
    if white_value is not None and white_value < surface_range[0]:
        plot_surface_range[0] = white_value  # 扩展range以包含white_value
        # 创建自定义colormap，在最小值处是白色
        plot_cmap = create_white_magma_colormap(base_cmap=cmap, white_fraction=0.05)

    ds(
        layer1=cortical_values,  # 400个皮层ROI
        colorbar=plot_cmap,
        color_range=plot_surface_range,  # 使用扩展后的color range
        fig_name=temp_surface.replace('.png', ''),
        layout='row',
        outline=True,
        trim=True,
        tp='fslr',
        mesh='veryinflated',
        save_png_only=True,
        display=False,
        size=(6000, 1000)
    )

    # 读取图像并修剪白边
    img_surface = trim_white_edges(mpimg.imread(temp_surface))
    img_nilearn = trim_white_edges(mpimg.imread(tmp_nilearn))

    # 合并图像：表面图在上，皮层下图在下，colorbar在右侧
    fig_merged = plt.figure(figsize=(70, 25))
    gs = gridspec.GridSpec(2, 2, figure=fig_merged,
                           width_ratios=[97, 3],
                           hspace=0.1, wspace=0.05)

    ax1 = fig_merged.add_subplot(gs[0, 0])
    ax1.imshow(img_surface)
    ax1.axis('off')

    ax2 = fig_merged.add_subplot(gs[1, 0])
    ax2.imshow(img_nilearn)
    ax2.axis('off')

    # 添加colorbar（使用皮层下图的range和colormap，因为它们包含white_value）
    ax_cbar = fig_merged.add_subplot(gs[:, 1])
    vmin, vmax = plot_subcort_range
    if vmin >= vmax:
        vmax = vmin + 1e-6
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = ColorbarBase(ax_cbar, cmap=plot_subcort_cmap, norm=norm, orientation='vertical')
    
    # 设置刻度（不显示white_value，只显示实际的数据范围）
    # 使用原始的color_range来设置刻度，而不是包含white_value的range
    tick_vmin, tick_vmax = color_range
    if tick_vmin < 0 < tick_vmax:
        ticks = [tick_vmin, 0, tick_vmax]
    else:
        ticks = [tick_vmin, tick_vmax]
    ax_cbar.set_yticks(ticks)
    
    # 设置字体为Arial
    cbar_label = "Cohen's d (显著脑区)" if significant_rois is not None and len(significant_rois) > 0 else "Cohen's d"
    cbar.set_label(cbar_label, rotation=270, labelpad=30, fontsize=40, fontname='Arial')
    ax_cbar.tick_params(labelsize=42, direction='in', length=8, width=2)
    
    # 设置刻度标签字体为Arial
    for label in ax_cbar.get_yticklabels():
        label.set_fontname('Arial')

    plt.tight_layout()
    
    # 确保输出路径为PDF格式
    if not output_path.endswith('.pdf'):
        pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    else:
        pdf_path = output_path
    
    # 保存为PDF格式
    fig_merged.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close(fig_merged)

    # 清理临时文件
    try:
        os.remove(temp_surface)
        os.remove(tmp_nilearn)
    except:
        pass

    print(f"Cohen's d图像已保存至: {pdf_path}")
    return True


def create_significant_regions_effect_size_plot(effect_vector,
                                                significant_rois,
                                                color_range,
                                                output_path,
                                                title,
                                                atlas_img_path=COMBINED_ATLAS_PATH,
                                                cmap='RdBu_r'):
    """
    只对显著脑区绘制Cohen's d效应量图（表面图+皮层下图）
    
    参数:
    - effect_vector: 所有脑区的Cohen's d向量
    - significant_rois: 显著脑区的索引列表
    - color_range: 颜色范围 [vmin, vmax]
    - output_path: 输出路径
    - title: 标题
    - atlas_img_path: 皮层下atlas路径
    - cmap: 颜色映射
    """
    if effect_vector is None or len(effect_vector) == 0:
        return False
    
    if significant_rois is None or len(significant_rois) == 0:
        print("提示：该类别没有显著脑区，已跳过")
        return False
    
    # 创建掩码：只保留显著脑区的值，非显著脑区设为NaN
    masked_effect_vector = np.full(len(effect_vector), np.nan, dtype=float)
    for roi_idx in significant_rois:
        if 0 <= roi_idx < len(effect_vector):
            if not np.isnan(effect_vector[roi_idx]):
                masked_effect_vector[roi_idx] = effect_vector[roi_idx]
    
    cortical_values = masked_effect_vector[CORTICAL_OFFSET:]
    subcortical_values = masked_effect_vector[:CORTICAL_OFFSET]
    
    if np.all(np.isnan(cortical_values)) and np.all(np.isnan(subcortical_values)):
        print("提示：该类别显著脑区没有可视化所需的Cohen's d，已跳过")
        return False

    # 准备皮层下数据
    if not atlas_img_path or not os.path.exists(atlas_img_path):
        raise FileNotFoundError(f"未找到皮层下atlas: {atlas_img_path}")

    atlas_img = nib.load(atlas_img_path)
    atlas_data = atlas_img.get_fdata().astype(int)
    roi_vol = np.full(atlas_data.shape, np.nan, dtype=float)
    max_sub = min(CORTICAL_OFFSET, len(subcortical_values))
    for lbl in range(1, max_sub + 1):
        if not np.isnan(subcortical_values[lbl - 1]):
            roi_vol[atlas_data == lbl] = subcortical_values[lbl - 1]

    roi_img = nib.Nifti1Image(roi_vol, atlas_img.affine, atlas_img.header)
    bg_img = datasets.load_mni152_template(resolution=1)
    bg_data = bg_img.get_fdata()
    bg_img_final = nib.Nifti1Image(bg_data, bg_img.affine, bg_img.header)

    # 绘制皮层下（nilearn）
    fig_nilearn = plt.figure(figsize=(40, 8))
    nilearn_disp = plotting.plot_stat_map(
        roi_img,
        bg_img=bg_img_final,
        threshold=0.0,
        cmap=cmap,
        vmax=color_range[1],
        vmin=color_range[0],
        display_mode='z',
        cut_coords=[-25, -20, -10, 0, 10, 20, 22],
        draw_cross=False,
        annotate=False,
        colorbar=False,
        figure=fig_nilearn,
        black_bg=False,
    )

    # 保存nilearn图像到临时文件
    temp_dir = tempfile.gettempdir()
    tmp_nilearn = os.path.join(temp_dir, f'temp_nilearn_sig_{os.getpid()}.png')
    fig_nilearn.savefig(tmp_nilearn, bbox_inches='tight', dpi=300, pad_inches=0)
    nilearn_disp.close()
    plt.close(fig_nilearn)

    # 读取nilearn图像尺寸
    img_nilearn_temp = mpimg.imread(tmp_nilearn)
    nilearn_width = img_nilearn_temp.shape[1]
    nilearn_height = img_nilearn_temp.shape[0]

    # 使用neurocat绘制表面图（只显示显著脑区）
    temp_surface = os.path.join(temp_dir, f'temp_surface_sig_{os.getpid()}.png')
    surface_height = int(nilearn_width * 0.2)
    surface_size = (nilearn_width, surface_height)

    ds(
        layer1=cortical_values,  # 400个皮层ROI，非显著脑区为NaN
        colorbar=cmap,
        color_range=color_range,
        fig_name=temp_surface.replace('.png', ''),
        layout='row',
        outline=True,
        trim=True,
        tp='fslr',
        mesh='veryinflated',
        save_png_only=True,
        display=False,
        size=(6000, 1000)
    )

    # 读取图像并修剪白边
    img_surface = trim_white_edges(mpimg.imread(temp_surface))
    img_nilearn = trim_white_edges(mpimg.imread(tmp_nilearn))

    # 合并图像
    fig_merged = plt.figure(figsize=(70, 25))
    gs = gridspec.GridSpec(2, 2, figure=fig_merged,
                           width_ratios=[97, 3],
                           hspace=0.1, wspace=0.05)

    ax1 = fig_merged.add_subplot(gs[0, 0])
    ax1.imshow(img_surface)
    ax1.axis('off')

    ax2 = fig_merged.add_subplot(gs[1, 0])
    ax2.imshow(img_nilearn)
    ax2.axis('off')

    # 添加colorbar
    ax_cbar = fig_merged.add_subplot(gs[:, 1])
    vmin, vmax = color_range
    if vmin >= vmax:
        vmax = vmin + 1e-6
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = ColorbarBase(ax_cbar, cmap=cmap, norm=norm, orientation='vertical')
    
    # 设置刻度
    if vmin < 0 < vmax:
        ticks = [vmin, 0, vmax]
    else:
        ticks = [vmin, vmax]
    ax_cbar.set_yticks(ticks)
    cbar.set_label("Cohen's d (显著脑区)", rotation=270, labelpad=30, fontsize=40)
    ax_cbar.tick_params(labelsize=42, direction='in', length=8, width=2)

    plt.tight_layout()
    fig_merged.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig_merged)

    # 清理临时文件
    try:
        os.remove(temp_surface)
        os.remove(tmp_nilearn)
    except:
        pass

    print(f"显著脑区Cohen's d图像已保存至: {output_path}")
    return True


def calculate_subject_level_cohens_d(category_data, target_category='9'):
    """
    为每个受试者计算每个脑区的Cohen's d（相对于目标类别）。
    
    参数:
    - category_data: 包含所有类别原始SFC数据的字典
    - target_category: 目标比较类别（默认为'9'）
    
    返回:
    - subject_cohens_d: 字典，{subject_id: {'aphasia_type': str, 'cohens_d_vector': np.array}}
    """
    if target_category not in category_data:
        print(f"错误：目标类别{target_category}不在数据中")
        return {}
    
    print(f"\n=== 计算每个受试者的Cohen's d效应量 ===")
    
    # 准备目标类别的数据
    target_subjects = category_data[target_category]
    target_data = np.vstack([subject['sfc_values'] for subject in target_subjects])
    
    # 计算目标类别在每个脑区的均值和标准差
    target_means = np.nanmean(target_data, axis=0)
    target_stds = np.nanstd(target_data, axis=0, ddof=1)
    
    # 避免除零
    target_stds[target_stds == 0] = np.nan
    
    print(f"目标类别{target_category}的受试者数量: {len(target_subjects)}")
    
    subject_cohens_d = {}
    
    # 遍历所有类别（除了目标类别）
    for aphasia_type, subjects in category_data.items():
        if aphasia_type == target_category:
            continue
        
        print(f"\n计算类别{aphasia_type}的受试者Cohen's d...")
        
        for subject in subjects:
            subject_id = subject['subject_id']
            sfc_values = subject['sfc_values']
            
            # 计算每个脑区的Cohen's d
            # Cohen's d = (subject_value - target_mean) / target_std
            cohens_d_vector = (sfc_values - target_means) / target_stds
            
            subject_cohens_d[subject_id] = {
                'aphasia_type': aphasia_type,
                'cohens_d_vector': cohens_d_vector
            }
        
        print(f"  类别{aphasia_type}: {len(subjects)}个受试者")
    
    print(f"总共计算了{len(subject_cohens_d)}个受试者的Cohen's d")
    return subject_cohens_d


def compute_subject_network_cohens_d(subject_cohens_d, network_to_indices, network_names):
    """
    为每个受试者计算7个网络的平均Cohen's d。
    
    参数:
    - subject_cohens_d: 每个受试者的Cohen's d向量
    - network_to_indices: 网络到皮层ROI索引的映射
    - network_names: 网络名称列表
    
    返回:
    - subject_network_data: 字典，{subject_id: {'aphasia_type': str, network1: val, ...}}
    """
    subject_network_data = {}
    
    for subject_id, data in subject_cohens_d.items():
        aphasia_type = data['aphasia_type']
        cohens_d_vector = data['cohens_d_vector']
        
        row = {
            'subject_id': subject_id,
            'aphasia_type': aphasia_type
        }
        
        # 计算每个网络的平均Cohen's d
        for network in network_names:
            if network not in network_to_indices:
                row[network] = np.nan
                continue
            
            # 获取该网络对应的全局ROI索引（考虑皮层偏移）
            cortical_indices = network_to_indices[network]
            global_indices = [
                CORTICAL_OFFSET + idx for idx in cortical_indices
                if 0 <= idx < N_CORTICAL_ROIS and (CORTICAL_OFFSET + idx) < len(cohens_d_vector)
            ]
            
            if not global_indices:
                row[network] = np.nan
                continue
            
            # 获取这些ROI的Cohen's d值
            network_values = cohens_d_vector[global_indices]
            
            # 计算平均值（忽略NaN）
            if np.all(np.isnan(network_values)):
                row[network] = np.nan
            else:
                row[network] = float(np.nanmean(network_values))
        
        subject_network_data[subject_id] = row
    
    return subject_network_data


def save_cohens_d_csv(subject_network_data, network_names, save_path):
    """导出每个受试者的7个网络平均Cohen's d。"""
    if not subject_network_data:
        print("没有受试者数据可写入CSV，已跳过")
        return None
    
    rows = []
    for subject_id, data in subject_network_data.items():
        row = {
            'subject_id': data['subject_id'],
            'aphasia_type': data['aphasia_type']
        }
        # 添加7个网络的平均Cohen's d
        for network in network_names:
            row[network] = data.get(network, np.nan)
        rows.append(row)
    
    # 确保列顺序：subject_id, aphasia_type, 然后按网络名称排序
    ordered_cols = ['subject_id', 'aphasia_type'] + sorted(network_names)
    df = pd.DataFrame(rows)
    
    # 重新排序列
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[ordered_cols]
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"受试者网络Cohen's d CSV已保存至: {save_path}")
    print(f"  共{len(rows)}行，{len(ordered_cols)}列")
    return save_path


def count_significant_regions_by_network(significant_regions, network_to_indices, network_names, save_path):
    """
    统计每个组别的每个网络有多少个显著脑区，保存为CSV
    
    参数:
    - significant_regions: 显著脑区信息字典
    - network_to_indices: 网络到皮层ROI索引的映射
    - network_names: 网络名称列表
    - save_path: CSV保存路径
    
    返回:
    - csv保存路径
    """
    print(f"\n=== 统计每个组别每个网络的显著脑区个数 ===")
    
    if not significant_regions or not network_to_indices:
        print("无显著脑区信息或网络映射，跳过统计")
        return None
    
    rows = []
    
    for aphasia_type, sig_data in significant_regions.items():
        significant_rois = sig_data['significant_rois']
        
        row = {
            'aphasia_type': aphasia_type,
            'total_significant': len(significant_rois)
        }
        
        # 统计每个网络的显著脑区数量
        for network in network_names:
            if network not in network_to_indices:
                row[network] = 0
                continue
            
            # 获取该网络对应的皮层ROI索引（0-399）
            cortical_indices = network_to_indices[network]
            # 转换为全局ROI索引（50-449）
            global_indices = [CORTICAL_OFFSET + idx for idx in cortical_indices]
            
            # 统计显著脑区中属于该网络的数量
            count = sum(1 for roi_idx in significant_rois if roi_idx in global_indices)
            row[network] = count
        
        rows.append(row)
        
        print(f"类别{aphasia_type}:")
        print(f"  总显著脑区数: {row['total_significant']}")
        for network in sorted(network_names):
            if row[network] > 0:
                print(f"  {network}: {row[network]}个显著脑区")
    
    # 创建DataFrame并保存
    ordered_cols = ['aphasia_type', 'total_significant'] + sorted(network_names)
    df = pd.DataFrame(rows)
    
    # 确保所有列都存在
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = 0
    
    df = df[ordered_cols]
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n网络显著脑区统计CSV已保存至: {save_path}")
    print(f"  共{len(rows)}行，{len(ordered_cols)}列")
    
    return save_path


def count_significant_regions_by_network(significant_regions, network_to_indices, network_names, save_path):
    """
    统计每个组别每个网络的显著脑区个数
    
    参数:
    - significant_regions: 显著脑区信息字典
    - network_to_indices: 网络到皮层ROI索引的映射
    - network_names: 网络名称列表
    - save_path: CSV保存路径
    
    返回:
    - DataFrame: 统计结果
    """
    if not significant_regions or not network_to_indices:
        print("没有显著脑区或网络映射数据，跳过统计")
        return None
    
    print(f"\n=== 统计每个组别每个网络的显著脑区个数 ===")
    
    rows = []
    
    for aphasia_type, sig_data in significant_regions.items():
        significant_rois = sig_data['significant_rois']
        
        # 将显著的全局ROI索引转换为皮层索引（减去CORTICAL_OFFSET）
        # 注意：只统计皮层ROI（索引 >= CORTICAL_OFFSET）
        significant_cortical_indices = set()
        for global_idx in significant_rois:
            if global_idx >= CORTICAL_OFFSET:
                cortical_idx = global_idx - CORTICAL_OFFSET
                if 0 <= cortical_idx < N_CORTICAL_ROIS:
                    significant_cortical_indices.add(cortical_idx)
        
        row = {'aphasia_type': aphasia_type}
        
        # 统计每个网络的显著脑区数
        total_sig_cortical = len(significant_cortical_indices)
        
        for network in sorted(network_names):
            if network not in network_to_indices:
                row[network] = 0
                continue
            
            # 获取该网络的皮层ROI索引
            network_indices = set(network_to_indices[network])
            
            # 计算交集：该网络中有多少个显著脑区
            sig_in_network = len(network_indices.intersection(significant_cortical_indices))
            row[network] = sig_in_network
        
        # 添加总计和皮层下显著脑区数
        row['total_cortical_significant'] = total_sig_cortical
        
        # 统计皮层下显著脑区数（索引 < CORTICAL_OFFSET）
        subcortical_sig = sum(1 for idx in significant_rois if idx < CORTICAL_OFFSET)
        row['subcortical_significant'] = subcortical_sig
        
        row['total_significant'] = sig_data['n_significant']
        
        rows.append(row)
        
        print(f"类别{aphasia_type}:")
        print(f"  总显著脑区数: {row['total_significant']}")
        print(f"  皮层显著脑区数: {total_sig_cortical}")
        print(f"  皮层下显著脑区数: {subcortical_sig}")
        for network in sorted(network_names):
            print(f"  {network}: {row[network]}")
    
    # 创建DataFrame
    ordered_cols = ['aphasia_type'] + sorted(network_names) + ['total_cortical_significant', 'subcortical_significant', 'total_significant']
    df = pd.DataFrame(rows)
    
    # 重新排序列
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[ordered_cols]
    
    # 保存CSV，确保正确处理"none"等特殊类别名称
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 将aphasia_type列转换为字符串，避免pandas将"none"识别为NaN
    df['aphasia_type'] = df['aphasia_type'].astype(str)
    df.to_csv(save_path, index=False, na_rep='')
    print(f"\n显著脑区网络统计CSV已保存至: {save_path}")
    print(f"  共{len(rows)}行，{len(ordered_cols)}列")
    
    return df


def main():
    """主函数：分析不同失语者类别与类别9的显著差异脑区"""

    csv_file_path = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\aphasia_plots\participants_aphasia_new.txt"
    sfc_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\sfc_txt_files"
    # D:\sc - fc_coupling\scfc - coupling - main\scfc - coupling - main\batch_results_pca_450_t288\sfc_txt_files\sub - sub - M2298_sfc_values_pca.txt
    save_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\source\results\significance_analysis"
    target_category = '9'

    print("=== 失语者类别与类别9的显著差异脑区分析 ===")

    print("\n1. 加载受试者类别信息...")
    subject_categories = load_subject_categories(csv_file_path)
    if not subject_categories:
        print("无法加载受试者类别信息，退出程序")
        return

    category_counts = {}
    for _, category in subject_categories.items():
        category_counts[category] = category_counts.get(category, 0) + 1

    print("\n失语者类别分布:")
    for category, count in sorted(category_counts.items()):
        print(f"  类别{category}: {count}个受试者")

    print("\n2. 按类别加载SFC数据...")
    category_data = load_sfc_data_by_category(sfc_dir, subject_categories)
    if not category_data:
        print("无法加载SFC数据，退出程序")
        return

    print("\n3. 进行ROI级别的组间统计检验...")
    significant_regions, p_values_dict = statistical_test_category_vs_8(category_data, target_category=target_category)
    if significant_regions is None:
        print("统计检验失败，退出程序")
        return

    print("\n4. 计算所有脑区的Cohen's d效应量（包含p值）...")
    cohens_d_results = calculate_cohens_d_for_all_regions(
        category_data, p_values_dict=p_values_dict, target_category=target_category
    )
    if cohens_d_results is None:
        print("Cohen's d计算失败，退出程序")
        return

    network_mapping, network_names = parse_network_label_file(NETWORK_LABEL_FILE)
    
    # 统计每个组别每个网络的显著脑区个数
    print("\n5. 统计每个组别每个网络的显著脑区个数...")
    network_sig_count_path = os.path.join(save_dir, "network_significant_regions_count.csv")
    count_significant_regions_by_network(significant_regions, network_mapping, network_names, network_sig_count_path)
    
    effect_plot_dir = os.path.join(save_dir, "cohens_d_plots")
    os.makedirs(effect_plot_dir, exist_ok=True)

    # 自动确定颜色范围（基于显著脑区）
    color_range = determine_color_range(cohens_d_results, significant_regions)  # 用于皮层下图和colorbar（对称范围，基于显著脑区）
    surface_color_range = determine_surface_color_range(cohens_d_results, significant_regions)  # 用于表面图（基于显著脑区的全局最小最大值）
    print(f"\n自动确定的颜色范围（皮层下图和colorbar，基于显著脑区）: [{color_range[0]:.3f}, {color_range[1]:.3f}]")
    print(f"自动确定的颜色范围（表面图，基于显著脑区）: [{surface_color_range[0]:.3f}, {surface_color_range[1]:.3f}]")

    print("\n6. 绘制显著脑区的Cohen's d效应量图（只显示显著脑区）...")
    for aphasia_type, data in cohens_d_results.items():
        effect_vector = data.get('effect_vector')
        if effect_vector is None or np.all(np.isnan(effect_vector)):
            print(f"类别{aphasia_type}缺少可视化所需的Cohen's d，已跳过")
            data['network_cohens_d'] = {}
            continue

        # 获取该类别的显著脑区
        significant_rois = []
        if aphasia_type in significant_regions:
            significant_rois = significant_regions[aphasia_type].get('significant_rois', [])
        
        if len(significant_rois) == 0:
            print(f"类别{aphasia_type}没有显著脑区，跳过绘图")
            data['network_cohens_d'] = {}
            continue

        if network_mapping:
            data['network_cohens_d'] = compute_network_cohens_d(effect_vector, network_mapping)
        else:
            data['network_cohens_d'] = {}

        output_path = os.path.join(
            effect_plot_dir,
            f"cohens_d_category_{aphasia_type}_vs_{target_category}.pdf"
        )
        title = f"Cohen's d (显著脑区): Category {aphasia_type} vs Category {target_category}"

        # 使用magma_r配色方案，只显示显著脑区
        # 表面图使用全局最小最大值，皮层下图和colorbar使用对称范围
        create_effect_size_plot(effect_vector, color_range, output_path, title, cmap='magma_r', 
                               surface_color_range=surface_color_range,
                               significant_rois=significant_rois)

    print("\n7. 计算每个受试者的Cohen's d（所有脑区）...")
    subject_cohens_d = calculate_subject_level_cohens_d(category_data, target_category=target_category)
    
    if not subject_cohens_d:
        print("无法计算受试者级别的Cohen's d，跳过CSV导出")
    else:
        print("\n8. 计算每个受试者的7个网络平均Cohen's d...")
        if not network_mapping or not network_names:
            print("警告：无法解析网络映射，跳过网络级Cohen's d计算")
        else:
            subject_network_data = compute_subject_network_cohens_d(
                subject_cohens_d, network_mapping, network_names
            )
            
            print("\n9. 导出受试者网络Cohen's d CSV...")
            csv_path = os.path.join(save_dir, "subject_network_cohens_d.csv")
            save_cohens_d_csv(subject_network_data, network_names, csv_path)

    print("\n=== 分析总结 ===")
    print(f"结果已保存至: {save_dir}")
    if os.path.exists(effect_plot_dir):
        print(f"  Cohen's d图像目录: {effect_plot_dir}")
    
    # 检查CSV文件路径（可能在步骤8中定义）
    csv_path = os.path.join(save_dir, "subject_network_cohens_d.csv")
    if os.path.exists(csv_path):
        print(f"  受试者网络Cohen's d CSV文件: {csv_path}")

    for aphasia_type, sig_data in significant_regions.items():
        print(f"\n类别{aphasia_type} vs 类别{target_category}:")
        print(f"  FDR校正后显著差异脑区数量: {sig_data['n_significant']}")
        print(f"  未校正显著差异脑区数量: {sig_data['n_uncorrected']}")

        if sig_data['n_significant'] > 0:
            print(f"  FDR校正后显著脑区索引: {sig_data['significant_rois']}")
            if cohens_d_results and aphasia_type in cohens_d_results:
                cohens_d_data = cohens_d_results[aphasia_type]
                if cohens_d_data.get('n_valid', 0) > 0:
                    cohens_d_values = cohens_d_data['cohens_d_values']
                    # 获取显著脑区的Cohen's d值
                    significant_rois = sig_data['significant_rois']
                    sig_cohens_d = []
                    sig_p_values = []
                    sig_p_corrected = []
                    for roi_idx in significant_rois:
                        if roi_idx < len(cohens_d_data['effect_vector']):
                            cohen_val = cohens_d_data['effect_vector'][roi_idx]
                            if not np.isnan(cohen_val):
                                sig_cohens_d.append(cohen_val)
                            # 获取p值
                            if 'p_value_vector' in cohens_d_data and roi_idx < len(cohens_d_data['p_value_vector']):
                                p_val = cohens_d_data['p_value_vector'][roi_idx]
                                if not np.isnan(p_val):
                                    sig_p_values.append(p_val)
                            if 'p_corrected_vector' in cohens_d_data and roi_idx < len(cohens_d_data['p_corrected_vector']):
                                p_corr = cohens_d_data['p_corrected_vector'][roi_idx]
                                if not np.isnan(p_corr):
                                    sig_p_corrected.append(p_corr)
                    
                    print(f"  Cohen's d效应量统计（显著脑区）:")
                    if sig_cohens_d:
                        print(f"    平均Cohen's d: {np.mean(sig_cohens_d):.3f}")
                        print(f"    Cohen's d范围: [{np.min(sig_cohens_d):.3f}, {np.max(sig_cohens_d):.3f}]")
                    if sig_p_corrected:
                        print(f"    平均FDR校正p值: {np.mean(sig_p_corrected):.4f}")
                        print(f"    FDR校正p值范围: [{np.min(sig_p_corrected):.4f}, {np.max(sig_p_corrected):.4f}]")
                    
                    print(f"  Cohen's d效应量统计（所有脑区）:")
                    print(f"    平均Cohen's d: {np.mean(cohens_d_values):.3f}")
                    print(f"    Cohen's d范围: [{np.min(cohens_d_values):.3f}, {np.max(cohens_d_values):.3f}]")
                    positive_effects = np.sum(np.array(cohens_d_values) > 0)
                    negative_effects = np.sum(np.array(cohens_d_values) < 0)
                    print(f"    正效应脑区数量: {positive_effects}")
                    print(f"    负效应脑区数量: {negative_effects}")

        if sig_data['n_uncorrected'] > 0:
            print(f"  未校正显著脑区索引: {sig_data['uncorrected_rois']}")

        if sig_data['n_significant'] == 0 and sig_data['n_uncorrected'] == 0:
            print("  无显著差异脑区")


if __name__ == "__main__":
    main()

