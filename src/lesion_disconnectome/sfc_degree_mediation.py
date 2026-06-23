# -*- coding: utf-8 -*-
"""
中介分析代码
分析SFC在SC/FC与WAB-r评分之间的介导作用

分析1: SC -> SFC -> WAB-r
  路径a: SC -> SFC
  路径b: SFC -> WAB-r (控制SC)
  路径c: SC -> WAB-r (总效应)
  路径c': SC -> WAB-r (控制SFC，直接效应)

分析2: FC -> SFC -> WAB-r
  路径a: FC -> SFC
  路径b: SFC -> WAB-r (控制FC)
  路径c: FC -> WAB-r (总效应)
  路径c': FC -> WAB-r (控制SFC，直接效应)

SC和FC先转换为度（degree）指标：将连接矩阵二值化后计算每个ROI的连接数
这样SC和FC的维度能与SFC对齐（都是ROI级别的值）
"""

import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
from nilearn.connectome import ConnectivityMeasure
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置路径 ====================
SFC_TXT_DIR = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\sfc_txt_files"
SC_BASE_DIR = r"D:\controlPaperReview\data\complete_subject_withsubcortex"
WAB_SCORE_PATH = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\aphasia_plots\participants_aphasia_new.txt"

# 仅分析 wab_r_aq 评分
SCORE_COLUMNS = ['wab_r_aq']


# ==================== 辅助函数 ====================
def load_sc_matrix(subject_identifier):
    """
    加载受试者的SC矩阵
    
    参数:
        subject_identifier: 可包含数字或完整字符串的受试者ID
    
    返回:
        sc_matrix: SC连接矩阵，如果文件不存在则返回None
    """
    subject_digits = extract_subject_digits(subject_identifier)
    if subject_digits is None:
        return None
    
    candidate_folders = [
        f"sub-{subject_digits}",
        f"sub-M{subject_digits}",
        f"M{subject_digits}",
        str(subject_identifier),
    ]
    
    for folder in candidate_folders:
        sc_path = os.path.join(SC_BASE_DIR, folder, "schaefer400_connectome.csv")
        if not os.path.exists(sc_path):
            continue
        
        try:
            sc_matrix = pd.read_csv(sc_path, header=None).values
            if sc_matrix.shape[0] != sc_matrix.shape[1]:
                print(f"警告: 受试者 {folder} 的SC矩阵不是方阵: {sc_matrix.shape}")
                return None
            return sc_matrix
        except Exception as e:
            print(f"读取受试者 {folder} 的SC文件失败: {e}")
            return None
    
    return None


def calculate_sc_degree(sc_matrix, threshold_value=None, subject_id=None, debug=False):
    """
    从SC矩阵中计算每个ROI的度（degree）
    先将矩阵二值化（0/1），然后计算每个节点的连接数
    
    参数:
        sc_matrix: SC连接矩阵
        threshold_value: 用于二值化的固定阈值（绝对值），如果为None则使用默认值0.0
                         使用固定阈值，与FC度计算保持一致
        subject_id: 受试者ID（用于调试输出）
        debug: 是否输出调试信息
    
    返回:
        sc_degrees: 每个ROI的度值数组，如果矩阵为None则返回None
    """
    if sc_matrix is None:
        return None
    
    # 如果未指定阈值，使用默认值0.0（即非零即连接）
    if threshold_value is None:
        threshold_value = 0.0
    
    # 创建副本
    matrix_copy = sc_matrix.copy()
    
    # 将对角线设为0
    np.fill_diagonal(matrix_copy, 0)
    
    if debug and subject_id:
        print(f"\n[SC度计算 - {subject_id}]")
        print(f"  SC矩阵形状: {sc_matrix.shape}")
        print(f"  SC矩阵统计: min={np.min(matrix_copy):.4f}, max={np.max(matrix_copy):.4f}, mean={np.mean(matrix_copy):.4f}, std={np.std(matrix_copy):.4f}")
        print(f"  SC矩阵非零元素比例: {np.count_nonzero(matrix_copy) / matrix_copy.size:.4f}")
    
    # 对于SC矩阵，使用绝对值阈值来二值化（与FC保持一致的计算方式）
    # 计算上三角部分的绝对值，用于统计信息
    triu_indices = np.triu_indices_from(matrix_copy, k=1)
    upper_triangle_abs = np.abs(matrix_copy[triu_indices])
    
    if debug and subject_id:
        print(f"  上三角绝对值统计: min={np.min(upper_triangle_abs):.4f}, max={np.max(upper_triangle_abs):.4f}, mean={np.mean(upper_triangle_abs):.4f}")
        if len(upper_triangle_abs) > 0 and np.max(upper_triangle_abs) > 0:
            non_zero_values = upper_triangle_abs[upper_triangle_abs > 0]
            if len(non_zero_values) > 0:
                p50 = np.percentile(non_zero_values, 50)
                p80 = np.percentile(non_zero_values, 80)
                print(f"  上三角绝对值百分位数: 50%={p50:.4f}, 80%={p80:.4f}")
            else:
                print(f"  上三角绝对值百分位数: 50%=0.0000, 80%=0.0000")
    
    if debug and subject_id:
        print(f"  使用的固定阈值（绝对值）: {threshold_value:.4f}")
        print(f"  超过阈值的连接数: {np.sum(upper_triangle_abs > threshold_value)}")
        print(f"  超过阈值的连接比例: {np.sum(upper_triangle_abs > threshold_value) / len(upper_triangle_abs):.4f}")
    
    # 二值化：绝对值大于阈值的设为1，否则为0（与FC保持一致）
    binary_matrix = (np.abs(matrix_copy) > threshold_value).astype(float)
    
    if debug and subject_id:
        print(f"  二值化后非零元素数量: {np.count_nonzero(binary_matrix)}")
        print(f"  二值化后非零元素比例: {np.count_nonzero(binary_matrix) / binary_matrix.size:.4f}")
    
    # 计算每个节点的度（每行的非零元素数量，即连接数）
    degrees = np.sum(binary_matrix, axis=1)
    
    if debug and subject_id:
        print(f"  SC度统计: min={np.min(degrees):.2f}, max={np.max(degrees):.2f}, mean={np.mean(degrees):.2f}, std={np.std(degrees):.2f}")
        print(f"  SC度前5个值: {degrees[:5]}")
    
    return degrees


def extract_subject_digits(identifier):
    """
    提取受试者ID中的数字部分，例如sub-2748 -> 2748
    """
    matches = re.findall(r'\d+', str(identifier))
    return matches[0] if matches else None


def format_subject_label(subject_digits):
    """
    根据数字ID生成标准化标签（sub-XXXX）
    """
    return f"sub-{subject_digits}" if subject_digits else None


def load_wab_scores(wab_file_path):
    """
    从participants_aphasia_new.txt加载wab_r_aq分数
    """
    if not os.path.exists(wab_file_path):
        raise FileNotFoundError(f"无法找到WAB分数文件: {wab_file_path}")
    
    wab_df = pd.read_csv(wab_file_path, sep='\t')
    score_map = {}
    
    for _, row in wab_df.iterrows():
        subject_digits = extract_subject_digits(row.get('participant_id', ''))
        if not subject_digits:
            continue
        # 优先尝试wab_r_aq，如果不存在则尝试wab_aq
        score = pd.to_numeric(row.get('wab_r_aq', row.get('wab_aq', np.nan)), errors='coerce')
        if pd.isna(score):
            continue
        score_map[subject_digits] = score
    
    return score_map


def load_sfc_values(sfc_file_path):
    """
    从*_sfc_values_pca.txt文件中解析ROI级别的SFC值
    """
    values = []
    
    try:
        with open(sfc_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith(('Subject', 'Method', 'ROI_Index', '-')):
                    continue
                
                parts = re.split(r'\s+', stripped)
                if len(parts) < 2 or not parts[0].isdigit():
                    continue
                
                try:
                    value = float(parts[1])
                except ValueError:
                    continue
                
                values.append(value)
    except Exception as e:
        print(f"读取SFC文件 {sfc_file_path} 失败: {e}")
        return None
    
    if not values:
        print(f"警告: {sfc_file_path} 未找到有效的SFC值")
        return None
    
    return np.array(values)


def load_fc_timeseries(subject_digits):
    """
    从受试者文件夹中加载BOLD时间序列
    """
    if subject_digits is None:
        return None
    
    candidate_folders = [
        f"sub-{subject_digits}",
        f"sub-M{subject_digits}",
        f"M{subject_digits}",
        subject_digits
    ]
    
    for folder in candidate_folders:
        folder_path = Path(SC_BASE_DIR) / folder
        if not folder_path.exists():
            continue
        bold_files = sorted(folder_path.glob("*task-rest_bold*.txt"))
        if not bold_files:
            continue
        bold_path = bold_files[0]
        try:
            timeseries = np.loadtxt(bold_path)
            if timeseries.ndim == 1:
                timeseries = timeseries[:, np.newaxis]
            return timeseries
        except Exception as e:
            print(f"读取BOLD文件 {bold_path} 失败: {e}")
            return None
    
    return None


def compute_fc_matrix(timeseries_data):
    """
    根据时间序列计算FC矩阵（Fisher z转换的相关矩阵）
    """
    if timeseries_data is None:
        return None
    
    try:
        conn_measure = ConnectivityMeasure(kind='correlation', standardize=True)
        fc = conn_measure.fit_transform([timeseries_data])[0]
        fc = np.clip(fc, -0.999999, 0.999999)
        fc = np.arctanh(fc)
        return fc
    except Exception as e:
        print(f"计算FC矩阵失败: {e}")
        return None


def calculate_fc_degree(fc_matrix, threshold_value=None, subject_id=None, debug=False):
    """
    从FC矩阵中计算每个ROI的度（degree）
    先将矩阵二值化（0/1），然后计算每个节点的连接数
    
    参数:
        fc_matrix: FC连接矩阵（Fisher z转换后的相关矩阵）
        threshold_value: 用于二值化的固定阈值（绝对值），如果为None则使用默认值0.5
                         使用固定阈值而不是百分位数，这样不同受试者会有不同的连接数
        subject_id: 受试者ID（用于调试输出）
        debug: 是否输出调试信息
    
    返回:
        fc_degrees: 每个ROI的度值数组，如果矩阵为None则返回None
    """
    if fc_matrix is None:
        return None
    
    # 如果未指定阈值，使用默认值0.5（Fisher z转换后的相关值）
    if threshold_value is None:
        threshold_value = 0.5
    
    # 创建副本
    matrix_copy = fc_matrix.copy()
    
    # 将对角线设为0
    np.fill_diagonal(matrix_copy, 0)
    
    if debug and subject_id:
        print(f"\n[FC度计算 - {subject_id}]")
        print(f"  FC矩阵形状: {fc_matrix.shape}")
        print(f"  FC矩阵统计: min={np.min(matrix_copy):.4f}, max={np.max(matrix_copy):.4f}, mean={np.mean(matrix_copy):.4f}, std={np.std(matrix_copy):.4f}")
        print(f"  FC矩阵非零元素比例: {np.count_nonzero(matrix_copy) / matrix_copy.size:.4f}")
    
    # 对于FC矩阵，使用绝对值阈值来二值化
    # 计算上三角部分的绝对值，用于统计信息
    triu_indices = np.triu_indices_from(matrix_copy, k=1)
    upper_triangle_abs = np.abs(matrix_copy[triu_indices])
    
    if debug and subject_id:
        print(f"  上三角绝对值统计: min={np.min(upper_triangle_abs):.4f}, max={np.max(upper_triangle_abs):.4f}, mean={np.mean(upper_triangle_abs):.4f}")
        print(f"  上三角绝对值百分位数: 50%={np.percentile(upper_triangle_abs, 50):.4f}, 80%={np.percentile(upper_triangle_abs, 80):.4f}, 90%={np.percentile(upper_triangle_abs, 90):.4f}")
    
    if debug and subject_id:
        print(f"  使用的固定阈值（绝对值）: {threshold_value:.4f}")
        print(f"  超过阈值的连接数: {np.sum(upper_triangle_abs > threshold_value)}")
        print(f"  超过阈值的连接比例: {np.sum(upper_triangle_abs > threshold_value) / len(upper_triangle_abs):.4f}")
    
    # 二值化：绝对值大于阈值的设为1，否则为0
    binary_matrix = (np.abs(matrix_copy) > threshold_value).astype(float)
    
    if debug and subject_id:
        print(f"  二值化后非零元素数量: {np.count_nonzero(binary_matrix)}")
        print(f"  二值化后非零元素比例: {np.count_nonzero(binary_matrix) / binary_matrix.size:.4f}")
    
    # 计算每个节点的度（每行的非零元素数量，即连接数）
    degrees = np.sum(binary_matrix, axis=1)
    
    if debug and subject_id:
        print(f"  FC度统计: min={np.min(degrees):.2f}, max={np.max(degrees):.2f}, mean={np.mean(degrees):.2f}, std={np.std(degrees):.2f}")
        print(f"  FC度前5个值: {degrees[:5]}")
    
    return degrees


def build_subject_dataframe():
    """
    遍历sfc_txt_files目录，整合SFC、SC度和FC度数据
    返回的数据中，每个受试者包含每个ROI的SFC值、SC度和FC度
    """
    sfc_dir = Path(SFC_TXT_DIR)
    if not sfc_dir.exists():
        print(f"错误: 找不到SFC目录 {SFC_TXT_DIR}")
        return pd.DataFrame()
    
    wab_scores = load_wab_scores(WAB_SCORE_PATH)
    print(f"成功加载 {len(wab_scores)} 条WAB AQ分数")
    
    records = []
    missing_score = 0
    missing_sfc = 0
    missing_sc = 0
    missing_fc = 0
    dimension_mismatch = 0
    
    sfc_files = sorted(sfc_dir.glob("*_sfc_values_pca.txt"))
    print(f"在 {SFC_TXT_DIR} 找到 {len(sfc_files)} 个SFC文件")
    
    for sfc_file in sfc_files:
        subject_digits = extract_subject_digits(sfc_file.stem)
        if not subject_digits:
            print(f"警告: 无法从文件名 {sfc_file.name} 中解析受试者ID")
            continue
        
        subject_label = format_subject_label(subject_digits)
        
        sfc_values = load_sfc_values(sfc_file)
        if sfc_values is None:
            missing_sfc += 1
            continue
        
        wab_score = wab_scores.get(subject_digits)
        if wab_score is None or pd.isna(wab_score):
            missing_score += 1
            continue
        
        sc_matrix = load_sc_matrix(subject_digits)
        sc_degrees = None
        if sc_matrix is None:
            missing_sc += 1
        else:
            # 只对前3个受试者输出调试信息
            debug_sc = (len(records) < 3)
            # 使用固定阈值0.0来二值化SC矩阵（绝对值>0.0的连接，即非零连接）
            # 使用固定阈值，与FC度计算保持一致
            sc_degrees = calculate_sc_degree(sc_matrix, threshold_value=0.0, subject_id=subject_label, debug=debug_sc)
            if sc_degrees is not None and len(sc_degrees) != len(sfc_values):
                dimension_mismatch += 1
                print(f"警告: 受试者 {subject_label} SC维度({len(sc_degrees)})与SFC维度({len(sfc_values)})不匹配")
                sc_degrees = None
        
        timeseries = load_fc_timeseries(subject_digits)
        fc_degrees = None
        if timeseries is None:
            missing_fc += 1
        else:
            fc_matrix = compute_fc_matrix(timeseries)
            if fc_matrix is not None:
                # 只对前3个受试者输出调试信息
                debug_fc = (len(records) < 3)
                # 使用固定阈值0.5来二值化FC矩阵（绝对值>0.5的连接）
                # 使用固定阈值而不是百分位数，这样不同受试者会有不同的连接数
                fc_degrees = calculate_fc_degree(fc_matrix, threshold_value=0.5, subject_id=subject_label, debug=debug_fc)
                if fc_degrees is not None and len(fc_degrees) != len(sfc_values):
                    dimension_mismatch += 1
                    print(f"警告: 受试者 {subject_label} FC维度({len(fc_degrees)})与SFC维度({len(sfc_values)})不匹配")
                    fc_degrees = None
            else:
                missing_fc += 1
        
        records.append({
            'subject_id': subject_label,
            'sfc_values': sfc_values,  # ROI级别的SFC值数组
            'sc_degrees': sc_degrees,  # ROI级别的SC度数组
            'fc_degrees': fc_degrees,  # ROI级别的FC度数组
            'wab_r_aq': wab_score
        })
    
    print(f"SFC解析成功的受试者: {len(records)}")
    print(f"缺失WAB AQ分数: {missing_score}")
    print(f"缺失SFC值: {missing_sfc}")
    print(f"缺失SC矩阵: {missing_sc}")
    print(f"缺失FC时间序列: {missing_fc}")
    print(f"维度不匹配: {dimension_mismatch}")
    
    return pd.DataFrame(records)


def fit_regression_model(y, predictors, predictor_names):
    """
    拟合回归模型并返回统计量
    
    参数:
        y: 因变量数组
        predictors: 预测变量列表（每个元素是一个数组）
        predictor_names: 预测变量名称列表
    
    返回:
        包含模型统计信息的字典
    """
    if len(predictors) == 0:
        return {
            'n': 0,
            'error': '没有预测变量',
            'predictors': predictor_names
        }
    
    predictors = np.column_stack(predictors)
    valid_mask = ~np.isnan(y)
    for i in range(predictors.shape[1]):
        valid_mask &= ~np.isnan(predictors[:, i])
    
    y_clean = y[valid_mask]
    X_clean = predictors[valid_mask]
    
    n = len(y_clean)
    if n <= predictors.shape[1] + 5:
        return {
            'n': n,
            'error': '样本量太小，无法稳定估计',
            'predictors': predictor_names
        }
    
    # 检查预测变量的变异性
    for i in range(X_clean.shape[1]):
        if np.std(X_clean[:, i]) < 1e-10:
            print(f"警告: 预测变量 {predictor_names[i]} 的变异性极小（std={np.std(X_clean[:, i]):.2e}），可能导致回归问题")
    
    X_with_const = add_constant(X_clean)
    model = OLS(y_clean, X_with_const).fit()
    residuals = model.resid
    sse = float(np.sum(residuals ** 2))

    # 计算标准化系数（beta）：beta_std = beta * (std_x / std_y)
    std_y = np.std(y_clean)
    std_x = np.std(X_clean, axis=0)
    
    coef_info = {}
    # 检查模型参数数量（包括常数项）
    # model.params[0]是常数项，model.params[1:]是预测变量的系数
    if len(model.params) < len(predictor_names) + 1:
        # 如果参数数量不足，可能是某些预测变量被移除了（如无变异）
        print(f"警告: 模型参数数量({len(model.params)})少于预期({len(predictor_names) + 1})")
        # 尝试匹配可用的参数
        for idx, name in enumerate(predictor_names):
            param_idx = idx + 1  # +1因为第一个是常数项
            if param_idx < len(model.params):
                raw_coef = float(model.params[param_idx])
                std_coef = np.nan
                if param_idx - 1 < len(std_x) and std_y > 0 and std_x[param_idx - 1] > 0:
                    std_coef = raw_coef * (std_x[param_idx - 1] / std_y)
                coef_info[name] = {
                    'coef': raw_coef,
                    'p': float(model.pvalues[param_idx]) if param_idx < len(model.pvalues) else np.nan,
                    'se': float(model.bse[param_idx]) if param_idx < len(model.bse) else np.nan,
                    'std_coef': float(std_coef) if not np.isnan(std_coef) else np.nan
                }
            else:
                coef_info[name] = {
                    'coef': np.nan,
                    'p': np.nan,
                    'se': np.nan,
                    'std_coef': np.nan
                }
    else:
        for idx, name in enumerate(predictor_names, start=1):
            if idx < len(model.params):
                raw_coef = float(model.params[idx])
                std_coef = np.nan
                if idx - 1 < len(std_x) and std_y > 0 and std_x[idx - 1] > 0:
                    std_coef = raw_coef * (std_x[idx - 1] / std_y)
                coef_info[name] = {
                    'coef': raw_coef,
                    'p': float(model.pvalues[idx]) if idx < len(model.pvalues) else np.nan,
                    'se': float(model.bse[idx]) if idx < len(model.bse) else np.nan,
                    'std_coef': float(std_coef) if not np.isnan(std_coef) else np.nan
                }
            else:
                coef_info[name] = {
                    'coef': np.nan,
                    'p': np.nan,
                    'se': np.nan,
                    'std_coef': np.nan
                }
    
    return {
        'n': n,
        'model': model,
        'predictors': predictor_names,
        'sse': sse,
        'p': len(model.params),
        'r2': float(model.rsquared),
        'r2_adj': float(model.rsquared_adj),
        'coef_info': coef_info
    }


def perform_mediation_analysis(x, m, y, x_name='X', m_name='M', y_name='Y'):
    """
    执行中介分析（Baron & Kenny方法）
    
    参数:
        x: 自变量数组
        m: 中介变量数组
        y: 因变量数组
        x_name: 自变量名称
        m_name: 中介变量名称
        y_name: 因变量名称
    
    返回:
        包含所有路径系数和中介效应统计的字典
    """
    # 检查数据有效性
    if np.std(x) < 1e-10:
        print(f"警告: 自变量 {x_name} 的变异性极小，可能无法进行回归分析")
    if np.std(m) < 1e-10:
        print(f"警告: 中介变量 {m_name} 的变异性极小，可能无法进行回归分析")
    
    # 路径a: X -> M
    path_a_result = fit_regression_model(m, [x], [x_name])
    
    # 路径b: M -> Y (控制X)
    path_b_result = fit_regression_model(y, [m, x], [m_name, x_name])
    
    # 路径c: X -> Y (总效应)
    path_c_result = fit_regression_model(y, [x], [x_name])
    
    # 路径c': X -> Y (控制M，直接效应)
    path_c_prime_result = fit_regression_model(y, [x, m], [x_name, m_name])
    
    results = {
        'path_a': path_a_result,
        'path_b': path_b_result,
        'path_c': path_c_result,
        'path_c_prime': path_c_prime_result
    }
    
    # 计算中介效应
    if 'error' not in path_a_result and 'error' not in path_b_result:
        a_coef = path_a_result['coef_info'][x_name]['coef']
        b_coef = path_b_result['coef_info'][m_name]['coef']
        indirect_effect = a_coef * b_coef
        
        # Sobel检验
        a_se = path_a_result['coef_info'][x_name]['se']
        b_se = path_b_result['coef_info'][m_name]['se']
        
        if not np.isnan(a_se) and not np.isnan(b_se) and a_se > 0 and b_se > 0:
            sobel_se = np.sqrt(b_coef**2 * a_se**2 + a_coef**2 * b_se**2)
            if sobel_se > 0:
                sobel_z = indirect_effect / sobel_se
                sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
            else:
                sobel_z = np.nan
                sobel_p = np.nan
        else:
            sobel_z = np.nan
            sobel_p = np.nan
        
        results['indirect_effect'] = indirect_effect
        results['sobel_z'] = sobel_z
        results['sobel_p'] = sobel_p
    else:
        results['indirect_effect'] = np.nan
        results['sobel_z'] = np.nan
        results['sobel_p'] = np.nan
    
    # 直接效应和总效应
    if 'error' not in path_c_prime_result:
        results['direct_effect'] = path_c_prime_result['coef_info'][x_name]['coef']
    else:
        results['direct_effect'] = np.nan
    
    if 'error' not in path_c_result:
        results['total_effect'] = path_c_result['coef_info'][x_name]['coef']
    else:
        results['total_effect'] = np.nan

    # 中介比例
    if not np.isnan(results.get('indirect_effect', np.nan)) and not np.isnan(results.get('total_effect', np.nan)) and results.get('total_effect', 0) != 0:
        results['mediation_proportion'] = results['indirect_effect'] / results['total_effect']
    else:
        results['mediation_proportion'] = np.nan

    return results


def validate_mediation_prerequisites(path_c_result, x_name='X', y_name='Y', alpha=0.05):
    """
    检查中介分析前提：总效应是否显著
    """
    prerequisite = {
        'total_effect_significant': False,
        'p_value': np.nan,
        'note': ''
    }
    if path_c_result and 'error' not in path_c_result:
        p_val = path_c_result['coef_info'][x_name]['p']
        prerequisite['p_value'] = p_val
        prerequisite['total_effect_significant'] = (p_val < alpha)
        if prerequisite['total_effect_significant']:
            prerequisite['note'] = f"✓ 总效应 {x_name}->{y_name} 显著 (p={p_val:.4f})"
        else:
            prerequisite['note'] = (f"⚠️ 总效应 {x_name}->{y_name} 未达显著 (p={p_val:.4f})，"
                                    "可能存在竞争中介或完全中介。")
    else:
        prerequisite['note'] = "⚠️ 无法计算总效应，数据不足或模型错误"
    return prerequisite


def print_mediation_results(mediation_result, x_name, m_name, y_name):
    """
    打印中介分析结果
    
    参数:
        mediation_result: perform_mediation_analysis返回的结果字典
        x_name: 自变量名称
        m_name: 中介变量名称
        y_name: 因变量名称
    """
    print(f"\n{'='*80}")
    print(f"中介分析: {x_name} -> {m_name} -> {y_name}")
    print(f"{'='*80}\n")
    
    # 路径a: X -> M
    path_a = mediation_result['path_a']
    print("【路径a: X -> M】")
    if 'error' in path_a:
        print(f"  错误: {path_a['error']}")
    else:
        print(f"  β_{x_name} -> {m_name}: {path_a['coef_info'][x_name]['coef']:.4f}")
        print(f"  p值: {path_a['coef_info'][x_name]['p']:.4f}")
        print(f"  R²: {path_a['r2']:.4f}")
        print(f"  n: {path_a['n']}")
    
    # 路径b: M -> Y (控制X)
    path_b = mediation_result['path_b']
    print("\n【路径b: M -> Y (控制X)】")
    if 'error' in path_b:
        print(f"  错误: {path_b['error']}")
    else:
        print(f"  β_{m_name} -> {y_name}: {path_b['coef_info'][m_name]['coef']:.4f}")
        print(f"  p值: {path_b['coef_info'][m_name]['p']:.4f}")
        print(f"  β_{x_name} -> {y_name} (控制M): {path_b['coef_info'][x_name]['coef']:.4f}")
        print(f"  p值: {path_b['coef_info'][x_name]['p']:.4f}")
        print(f"  R²: {path_b['r2']:.4f}")
        print(f"  n: {path_b['n']}")
    
    # 路径c: X -> Y (总效应)
    path_c = mediation_result['path_c']
    print("\n【路径c: X -> Y (总效应)】")
    if 'error' in path_c:
        print(f"  错误: {path_c['error']}")
    else:
        print(f"  β_{x_name} -> {y_name}: {path_c['coef_info'][x_name]['coef']:.4f}")
        print(f"  p值: {path_c['coef_info'][x_name]['p']:.4f}")
        print(f"  R²: {path_c['r2']:.4f}")
        print(f"  n: {path_c['n']}")
    
    # 路径c': X -> Y (控制M，直接效应)
    path_c_prime = mediation_result['path_c_prime']
    print("\n【路径c': X -> Y (控制M，直接效应)】")
    if 'error' in path_c_prime:
        print(f"  错误: {path_c_prime['error']}")
    else:
        print(f"  β_{x_name} -> {y_name}: {path_c_prime['coef_info'][x_name]['coef']:.4f}")
        print(f"  p值: {path_c_prime['coef_info'][x_name]['p']:.4f}")
        print(f"  R²: {path_c_prime['r2']:.4f}")
        print(f"  n: {path_c_prime['n']}")
    
    # 中介效应
    print("\n【中介效应】")
    indirect = mediation_result.get('indirect_effect', np.nan)
    direct = mediation_result.get('direct_effect', np.nan)
    total = mediation_result.get('total_effect', np.nan)
    
    print(f"  间接效应 (a × b): {indirect:.4f}")
    print(f"  直接效应 (c'): {direct:.4f}")
    print(f"  总效应 (c): {total:.4f}")
    
    if not np.isnan(indirect) and not np.isnan(total) and total != 0:
        prop_mediated = indirect / total
        print(f"  中介比例: {prop_mediated:.4f} ({prop_mediated*100:.2f}%)")
        mediation_result['mediation_proportion'] = prop_mediated
    
    # Sobel检验
    sobel_z = mediation_result.get('sobel_z', np.nan)
    sobel_p = mediation_result.get('sobel_p', np.nan)
    print(f"\n【Sobel检验】")
    if not np.isnan(sobel_z):
        print(f"  Z值: {sobel_z:.4f}")
        print(f"  p值: {sobel_p:.4f}")
        if sobel_p < 0.05:
            print(f"  结果: 中介效应显著 (p < 0.05)")
        else:
            print(f"  结果: 中介效应不显著 (p >= 0.05)")
    else:
        print(f"  无法计算Sobel统计量")

    # Bootstrap置信区间
    if 'bootstrap_ci' in mediation_result:
        ci_low, ci_high = mediation_result.get('bootstrap_ci', (np.nan, np.nan))
        print(f"\n【Bootstrap间接效应95% CI】")
        print(f"  [{ci_low:.4f}, {ci_high:.4f}]")
        if not np.isnan(ci_low) and not np.isnan(ci_high):
            if ci_low > 0 or ci_high < 0:
                print("  结论: 间接效应显著（CI不跨0）")
            else:
                print("  结论: 间接效应不显著（CI跨0）")

    # 前提检查
    if 'prerequisite' in mediation_result:
        print(f"\n【前提检查】")
        print(f"  {mediation_result['prerequisite'].get('note', '')}")

    # 标准化系数
    if 'path_a' in mediation_result and 'coef_info' in mediation_result['path_a']:
        def _safe_std(dct):
            val = dct.get('std_coef', np.nan) if isinstance(dct, dict) else np.nan
            try:
                return float(val)
            except Exception:
                return np.nan

        pa = mediation_result['path_a']['coef_info'].get(x_name, {})
        pb = mediation_result['path_b']['coef_info'].get(m_name, {}) if 'coef_info' in mediation_result.get('path_b', {}) else {}
        pc = mediation_result['path_c']['coef_info'].get(x_name, {}) if 'coef_info' in mediation_result.get('path_c', {}) else {}
        pc_prime = mediation_result['path_c_prime']['coef_info'].get(x_name, {}) if 'coef_info' in mediation_result.get('path_c_prime', {}) else {}
        print(f"\n【标准化β系数】")
        print(f"  路径a β_std: {_safe_std(pa):.4f}")
        print(f"  路径b β_std: {_safe_std(pb):.4f}")
        print(f"  路径c β_std: {_safe_std(pc):.4f}")
        print(f"  路径c' β_std: {_safe_std(pc_prime):.4f}")
    
    print(f"\n{'='*80}\n")


def perform_mediation_analysis_with_bootstrap(x, m, y, x_name='X', m_name='M', y_name='Y', n_bootstrap=5000, random_state=42):
    """
    执行中介分析并使用Bootstrap估计间接效应置信区间
    """
    base_result = perform_mediation_analysis(x, m, y, x_name=x_name, m_name=m_name, y_name=y_name)

    rng = np.random.default_rng(random_state)
    boot_indirect = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), len(x))
        x_b = x[idx]
        m_b = m[idx]
        y_b = y[idx]

        path_a_b = fit_regression_model(m_b, [x_b], [x_name])
        path_b_b = fit_regression_model(y_b, [m_b, x_b], [m_name, x_name])

        if 'error' in path_a_b or 'error' in path_b_b:
            continue

        a_coef = path_a_b['coef_info'][x_name]['coef']
        b_coef = path_b_b['coef_info'][m_name]['coef']
        boot_indirect.append(a_coef * b_coef)

    if boot_indirect:
        ci_low, ci_high = np.percentile(boot_indirect, [2.5, 97.5])
    else:
        ci_low, ci_high = (np.nan, np.nan)

    base_result['bootstrap_indirect'] = boot_indirect
    base_result['bootstrap_ci'] = (float(ci_low), float(ci_high))
    base_result['n_bootstrap'] = n_bootstrap
    return base_result


def perform_correlation_analysis(df, output_dir, score_col='wab_r_aq'):
    """
    对中介分析中的所有X变量和M变量与Y变量进行相关性分析
    X变量: SC度、FC度
    M变量: SFC
    Y变量: WAB-r评分
    """
    if df.empty or score_col not in df:
        print("相关性分析失败：缺少必要列")
        return pd.DataFrame()
    
    print(f"\n{'='*80}")
    print(f"相关性分析: 所有自变量（SC、FC、SFC）与因变量（{score_col}）")
    print(f"{'='*80}\n")
    
    # 定义变量
    variables = {
        'SC度': df['sc_degree_mean_norm'].values,
        'FC度': df['fc_degree_mean_norm'].values,
        'SFC': df['sfc_mean'].values,
        score_col: df[score_col].values
    }
    
    # 计算相关系数
    correlation_results = []
    
    for var_name in ['SC度', 'FC度', 'SFC']:
        x = variables[var_name]
        y = variables[score_col]
        
        # 移除缺失值
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) > 2:
            # 计算Pearson相关系数
            r, p = stats.pearsonr(x_clean, y_clean)
            
            correlation_results.append({
                '自变量': var_name,
                '因变量': score_col,
                'r': float(r),
                'p值': float(p),
                '样本量': len(x_clean)
            })
            
            # 打印结果
            print(f"{var_name} 与 {score_col} 的相关性:")
            print(f"  r = {r:.4f}")
            print(f"  p = {p:.4f}")
            print(f"  n = {len(x_clean)}")
            if p < 0.001:
                print(f"  显著性: *** (p < 0.001)")
            elif p < 0.01:
                print(f"  显著性: ** (p < 0.01)")
            elif p < 0.05:
                print(f"  显著性: * (p < 0.05)")
            else:
                print(f"  显著性: n.s. (p >= 0.05)")
            print()
        else:
            print(f"警告: {var_name} 与 {score_col} 的有效样本量不足")
            correlation_results.append({
                '自变量': var_name,
                '因变量': score_col,
                'r': np.nan,
                'p值': np.nan,
                '样本量': len(x_clean)
            })
    
    # 保存结果
    if correlation_results:
        corr_df = pd.DataFrame(correlation_results)
        output_path = os.path.join(output_dir, 'correlation_analysis_results.csv')
        corr_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"相关性分析结果已保存至: {output_path}")
        print(f"{'='*80}\n")
    
    return pd.DataFrame(correlation_results)


def perform_model_comparison(df, output_dir, score_col='wab_r_aq'):
    """
    比较5个竞争模型并输出结果
    模型1: WAB ~ SC
    模型2: WAB ~ FC
    模型3: WAB ~ SFC
    模型4: WAB ~ SC + FC
    模型5: WAB ~ SC + FC + SFC
    """
    if df.empty or score_col not in df:
        print("模型比较失败：缺少必要列")
        return pd.DataFrame()

    models = []

    def fit_and_collect(name, X):
        model = OLS(df[score_col], add_constant(X)).fit()
        return {
            'model': name,
            'r2': float(model.rsquared),
            'r2_adj': float(model.rsquared_adj),
            'aic': float(model.aic),
            'bic': float(model.bic),
            'n': int(model.nobs)
        }

    models.append(fit_and_collect('SC_only', df['sc_degree_mean_norm']))
    models.append(fit_and_collect('FC_only', df['fc_degree_mean_norm']))
    models.append(fit_and_collect('SFC_only', df['sfc_mean']))

    sc_fc = np.column_stack([df['sc_degree_mean_norm'], df['fc_degree_mean_norm']])
    models.append(fit_and_collect('SC_FC', sc_fc))

    sc_fc_sfc = np.column_stack([df['sc_degree_mean_norm'], df['fc_degree_mean_norm'], df['sfc_mean']])
    models.append(fit_and_collect('SC_FC_SFC', sc_fc_sfc))

    df_models = pd.DataFrame(models)
    df_models = df_models.sort_values(by='r2_adj', ascending=False)
    best_adj_r2 = df_models['r2_adj'].iloc[0]
    df_models['delta_adj_r2'] = df_models['r2_adj'] - best_adj_r2

    output_path = os.path.join(output_dir, 'model_comparison_results.csv')
    df_models.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n模型比较（按调整R²降序）:")
    for _, row in df_models.iterrows():
        print(f"  {row['model']:12s} adj-R²={row['r2_adj']:.4f}, AIC={row['aic']:.2f}, Δadj-R²={row['delta_adj_r2']:.4f}")
    print(f"模型比较结果已保存至: {output_path}")

    return df_models


# ==================== 主程序 ====================
def main():
    print("开始整合SFC、SC度与FC度数据...")

    df = build_subject_dataframe()

    if df.empty:
        print("无可用受试者，终止分析。")
        return

    # 将ROI级别的数据汇总到受试者水平（计算平均值）
    print("\n将ROI级别的数据汇总到受试者水平...")
    
    def safe_mean(arr):
        """安全计算平均值，处理None和NaN"""
        if arr is None:
            return np.nan
        if isinstance(arr, (list, np.ndarray)):
            arr = np.array(arr)
            arr = arr[~np.isnan(arr)]
            return float(np.mean(arr)) if len(arr) > 0 else np.nan
        return np.nan
    
    df['sfc_mean'] = df['sfc_values'].apply(safe_mean)
    df['sc_degree_mean'] = df['sc_degrees'].apply(safe_mean)
    df['fc_degree_mean'] = df['fc_degrees'].apply(safe_mean)
    
    # 筛选有效数据
    valid_df = df.dropna(subset=['sfc_mean', 'sc_degree_mean', 'fc_degree_mean', 'wab_r_aq'])
    
    if valid_df.empty:
        print("所有受试者的数据存在缺失，无法进行分析。")
        return

    print(f"\n成功整合 {len(valid_df)} 名受试者的完整数据")
    
    # 显示前几个受试者的详细度值信息
    print("\n前5个受试者的度值详情:")
    for idx in range(min(5, len(valid_df))):
        row = valid_df.iloc[idx]
        sc_deg = row['sc_degrees']
        fc_deg = row['fc_degrees']
        if sc_deg is not None and isinstance(sc_deg, np.ndarray):
            print(f"  {row['subject_id']}: SC度均值={row['sc_degree_mean']:.2f} (范围: {np.min(sc_deg):.2f}-{np.max(sc_deg):.2f})")
        if fc_deg is not None and isinstance(fc_deg, np.ndarray):
            print(f"  {row['subject_id']}: FC度均值={row['fc_degree_mean']:.2f} (范围: {np.min(fc_deg):.2f}-{np.max(fc_deg):.2f})")
    
    print("\n归一化前的数据摘要:")
    print(f"  平均SFC: mean={valid_df['sfc_mean'].mean():.4f}, std={valid_df['sfc_mean'].std():.4f}")
    print(f"  SC度均值: mean={valid_df['sc_degree_mean'].mean():.4f}, std={valid_df['sc_degree_mean'].std():.4f}")
    print(f"  SC度均值范围: [{valid_df['sc_degree_mean'].min():.4f}, {valid_df['sc_degree_mean'].max():.4f}]")
    print(f"  FC度均值: mean={valid_df['fc_degree_mean'].mean():.4f}, std={valid_df['fc_degree_mean'].std():.4f}")
    print(f"  FC度均值范围: [{valid_df['fc_degree_mean'].min():.4f}, {valid_df['fc_degree_mean'].max():.4f}]")
    print(f"  FC度均值唯一值数量: {valid_df['fc_degree_mean'].nunique()}")
    if valid_df['fc_degree_mean'].nunique() <= 5:
        print(f"  FC度均值唯一值: {sorted(valid_df['fc_degree_mean'].unique())}")
    print(f"  WAB-r AQ: mean={valid_df['wab_r_aq'].mean():.4f}, std={valid_df['wab_r_aq'].std():.4f}")
    
    # 对SC度和FC度进行Min-Max归一化（缩放到0-1范围）
    print("\n对SC度和FC度进行归一化...")
    
    def normalize_minmax(series):
        """Min-Max归一化：将值缩放到0-1范围"""
        min_val = series.min()
        max_val = series.max()
        if max_val > min_val:
            return (series - min_val) / (max_val - min_val)
        else:
            # 如果所有值都相同，返回0.5（中间值）
            return pd.Series([0.5] * len(series), index=series.index)
    
    # 保存原始值（用于显示）
    sc_degree_min = valid_df['sc_degree_mean'].min()
    sc_degree_max = valid_df['sc_degree_mean'].max()
    fc_degree_min = valid_df['fc_degree_mean'].min()
    fc_degree_max = valid_df['fc_degree_mean'].max()
    
    # 进行归一化
    valid_df['sc_degree_mean_norm'] = normalize_minmax(valid_df['sc_degree_mean'])
    valid_df['fc_degree_mean_norm'] = normalize_minmax(valid_df['fc_degree_mean'])
    
    print(f"  SC度归一化范围: [{sc_degree_min:.4f}, {sc_degree_max:.4f}] -> [0, 1]")
    print(f"  FC度归一化范围: [{fc_degree_min:.4f}, {fc_degree_max:.4f}] -> [0, 1]")
    
    print("\n归一化后的数据摘要:")
    print(f"  SC度均值(归一化): mean={valid_df['sc_degree_mean_norm'].mean():.4f}, std={valid_df['sc_degree_mean_norm'].std():.4f}")
    print(f"  FC度均值(归一化): mean={valid_df['fc_degree_mean_norm'].mean():.4f}, std={valid_df['fc_degree_mean_norm'].std():.4f}")

    # 标准化（z-score），用于标准化系数与模型比较
    valid_df['sfc_mean_z'] = stats.zscore(valid_df['sfc_mean'])
    valid_df['sc_degree_mean_norm_z'] = stats.zscore(valid_df['sc_degree_mean_norm'])
    valid_df['fc_degree_mean_norm_z'] = stats.zscore(valid_df['fc_degree_mean_norm'])
    valid_df['wab_r_aq_z'] = stats.zscore(valid_df['wab_r_aq'])

    # 竞争模型比较
    model_compare_df = perform_model_comparison(valid_df, SFC_TXT_DIR, score_col='wab_r_aq')

    # 对每个评分进行中介分析
    all_results = []

    for score_col in SCORE_COLUMNS:
        if score_col not in valid_df.columns:
            print(f"警告: 评分列 {score_col} 不存在，跳过")
            continue

        print(f"\n\n{'#'*80}")
        print(f"分析评分: {score_col}")
        print(f"{'#'*80}")

        # 提取变量（使用归一化后的度值）
        sfc_values = valid_df['sfc_mean'].values
        sc_degree_values = valid_df['sc_degree_mean_norm'].values
        fc_degree_values = valid_df['fc_degree_mean_norm'].values
        y_values = valid_df[score_col].values

        # 分析1: SC -> SFC -> WAB-r
        print(f"\n{'='*80}")
        print(f"分析1: SC -> SFC -> {score_col}")
        print(f"{'='*80}")
        mediation_sc = perform_mediation_analysis_with_bootstrap(
            sc_degree_values, 
            sfc_values, 
            y_values,
            x_name='SC',
            m_name='SFC',
            y_name=score_col,
            n_bootstrap=5000
        )
        mediation_sc['prerequisite'] = validate_mediation_prerequisites(mediation_sc['path_c'], x_name='SC', y_name=score_col)
        print_mediation_results(mediation_sc, 'SC', 'SFC', score_col)

        # 分析2: FC -> SFC -> WAB-r
        print(f"\n{'='*80}")
        print(f"分析2: FC -> SFC -> {score_col}")
        print(f"{'='*80}")
        mediation_fc = perform_mediation_analysis_with_bootstrap(
            fc_degree_values,
            sfc_values,
            y_values,
            x_name='FC',
            m_name='SFC',
            y_name=score_col,
            n_bootstrap=5000
        )
        mediation_fc['prerequisite'] = validate_mediation_prerequisites(mediation_fc['path_c'], x_name='FC', y_name=score_col)
        print_mediation_results(mediation_fc, 'FC', 'SFC', score_col)

        # 保存结果
        sc_path_a = mediation_sc['path_a']
        sc_path_b = mediation_sc['path_b']
        sc_path_c = mediation_sc['path_c']
        sc_path_c_prime = mediation_sc['path_c_prime']
        
        fc_path_a = mediation_fc['path_a']
        fc_path_b = mediation_fc['path_b']
        fc_path_c = mediation_fc['path_c']
        fc_path_c_prime = mediation_fc['path_c_prime']
        
        all_results.append({
            'score_name': score_col,
            # SC中介分析
            'sc_path_a_coef': sc_path_a['coef_info']['SC']['coef'] if 'error' not in sc_path_a else np.nan,
            'sc_path_a_p': sc_path_a['coef_info']['SC']['p'] if 'error' not in sc_path_a else np.nan,
            'sc_path_a_r2': sc_path_a.get('r2', np.nan) if 'error' not in sc_path_a else np.nan,
            'sc_path_a_std': sc_path_a['coef_info']['SC'].get('std_coef', np.nan) if 'error' not in sc_path_a else np.nan,
            'sc_path_b_coef': sc_path_b['coef_info']['SFC']['coef'] if 'error' not in sc_path_b else np.nan,
            'sc_path_b_p': sc_path_b['coef_info']['SFC']['p'] if 'error' not in sc_path_b else np.nan,
            'sc_path_b_r2': sc_path_b.get('r2', np.nan) if 'error' not in sc_path_b else np.nan,
            'sc_path_b_std': sc_path_b['coef_info']['SFC'].get('std_coef', np.nan) if 'error' not in sc_path_b else np.nan,
            'sc_path_c_coef': sc_path_c['coef_info']['SC']['coef'] if 'error' not in sc_path_c else np.nan,
            'sc_path_c_p': sc_path_c['coef_info']['SC']['p'] if 'error' not in sc_path_c else np.nan,
            'sc_path_c_r2': sc_path_c.get('r2', np.nan) if 'error' not in sc_path_c else np.nan,
            'sc_path_c_std': sc_path_c['coef_info']['SC'].get('std_coef', np.nan) if 'error' not in sc_path_c else np.nan,
            'sc_path_c_prime_coef': sc_path_c_prime['coef_info']['SC']['coef'] if 'error' not in sc_path_c_prime else np.nan,
            'sc_path_c_prime_p': sc_path_c_prime['coef_info']['SC']['p'] if 'error' not in sc_path_c_prime else np.nan,
            'sc_path_c_prime_r2': sc_path_c_prime.get('r2', np.nan) if 'error' not in sc_path_c_prime else np.nan,
            'sc_path_c_prime_std': sc_path_c_prime['coef_info']['SC'].get('std_coef', np.nan) if 'error' not in sc_path_c_prime else np.nan,
            'sc_indirect_effect': mediation_sc.get('indirect_effect', np.nan),
            'sc_direct_effect': mediation_sc.get('direct_effect', np.nan),
            'sc_total_effect': mediation_sc.get('total_effect', np.nan),
            'sc_mediation_prop': mediation_sc.get('mediation_proportion', np.nan),
            'sc_sobel_z': mediation_sc.get('sobel_z', np.nan),
            'sc_sobel_p': mediation_sc.get('sobel_p', np.nan),
            'sc_boot_ci_low': mediation_sc.get('bootstrap_ci', (np.nan, np.nan))[0] if mediation_sc.get('bootstrap_ci') else np.nan,
            'sc_boot_ci_high': mediation_sc.get('bootstrap_ci', (np.nan, np.nan))[1] if mediation_sc.get('bootstrap_ci') else np.nan,
            'sc_prereq_p': mediation_sc.get('prerequisite', {}).get('p_value', np.nan),
            'sc_prereq_pass': mediation_sc.get('prerequisite', {}).get('total_effect_significant', False),
            'sc_n': sc_path_a.get('n', np.nan),
            # FC中介分析
            'fc_path_a_coef': fc_path_a['coef_info']['FC']['coef'] if 'error' not in fc_path_a else np.nan,
            'fc_path_a_p': fc_path_a['coef_info']['FC']['p'] if 'error' not in fc_path_a else np.nan,
            'fc_path_a_r2': fc_path_a.get('r2', np.nan) if 'error' not in fc_path_a else np.nan,
            'fc_path_a_std': fc_path_a['coef_info']['FC'].get('std_coef', np.nan) if 'error' not in fc_path_a else np.nan,
            'fc_path_b_coef': fc_path_b['coef_info']['SFC']['coef'] if 'error' not in fc_path_b else np.nan,
            'fc_path_b_p': fc_path_b['coef_info']['SFC']['p'] if 'error' not in fc_path_b else np.nan,
            'fc_path_b_r2': fc_path_b.get('r2', np.nan) if 'error' not in fc_path_b else np.nan,
            'fc_path_b_std': fc_path_b['coef_info']['SFC'].get('std_coef', np.nan) if 'error' not in fc_path_b else np.nan,
            'fc_path_c_coef': fc_path_c['coef_info']['FC']['coef'] if 'error' not in fc_path_c else np.nan,
            'fc_path_c_p': fc_path_c['coef_info']['FC']['p'] if 'error' not in fc_path_c else np.nan,
            'fc_path_c_r2': fc_path_c.get('r2', np.nan) if 'error' not in fc_path_c else np.nan,
            'fc_path_c_std': fc_path_c['coef_info']['FC'].get('std_coef', np.nan) if 'error' not in fc_path_c else np.nan,
            'fc_path_c_prime_coef': fc_path_c_prime['coef_info']['FC']['coef'] if 'error' not in fc_path_c_prime else np.nan,
            'fc_path_c_prime_p': fc_path_c_prime['coef_info']['FC']['p'] if 'error' not in fc_path_c_prime else np.nan,
            'fc_path_c_prime_r2': fc_path_c_prime.get('r2', np.nan) if 'error' not in fc_path_c_prime else np.nan,
            'fc_path_c_prime_std': fc_path_c_prime['coef_info']['FC'].get('std_coef', np.nan) if 'error' not in fc_path_c_prime else np.nan,
            'fc_indirect_effect': mediation_fc.get('indirect_effect', np.nan),
            'fc_direct_effect': mediation_fc.get('direct_effect', np.nan),
            'fc_total_effect': mediation_fc.get('total_effect', np.nan),
            'fc_mediation_prop': mediation_fc.get('mediation_proportion', np.nan),
            'fc_sobel_z': mediation_fc.get('sobel_z', np.nan),
            'fc_sobel_p': mediation_fc.get('sobel_p', np.nan),
            'fc_boot_ci_low': mediation_fc.get('bootstrap_ci', (np.nan, np.nan))[0] if mediation_fc.get('bootstrap_ci') else np.nan,
            'fc_boot_ci_high': mediation_fc.get('bootstrap_ci', (np.nan, np.nan))[1] if mediation_fc.get('bootstrap_ci') else np.nan,
            'fc_prereq_p': mediation_fc.get('prerequisite', {}).get('p_value', np.nan),
            'fc_prereq_pass': mediation_fc.get('prerequisite', {}).get('total_effect_significant', False),
            'fc_n': fc_path_a.get('n', np.nan),
        })

    # 保存所有结果到CSV
    print("\n保存结果到CSV...")
    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(
        SFC_TXT_DIR,
        'mediation_analysis_results.csv'
    )
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存至: {output_path}")

    # 创建简化版结果表
    summary_data = []
    for r in all_results:
        # 计算中介比例
        sc_prop = np.nan
        if not np.isnan(r['sc_indirect_effect']) and not np.isnan(r['sc_total_effect']) and r['sc_total_effect'] != 0:
            sc_prop = r['sc_indirect_effect'] / r['sc_total_effect']
        
        fc_prop = np.nan
        if not np.isnan(r['fc_indirect_effect']) and not np.isnan(r['fc_total_effect']) and r['fc_total_effect'] != 0:
            fc_prop = r['fc_indirect_effect'] / r['fc_total_effect']
        
        summary_data.append({
            '评分': r['score_name'],
            # SC中介分析
            'SC路径a系数': f"{r['sc_path_a_coef']:.4f}",
            'SC路径a_p值': f"{r['sc_path_a_p']:.4f}",
            'SC路径b系数': f"{r['sc_path_b_coef']:.4f}",
            'SC路径b_p值': f"{r['sc_path_b_p']:.4f}",
            'SC路径c系数(总效应)': f"{r['sc_path_c_coef']:.4f}",
            'SC路径c_p值': f"{r['sc_path_c_p']:.4f}",
            'SC路径c\'系数(直接效应)': f"{r['sc_path_c_prime_coef']:.4f}",
            'SC路径c\'_p值': f"{r['sc_path_c_prime_p']:.4f}",
            'SC间接效应': f"{r['sc_indirect_effect']:.4f}",
            'SC直接效应': f"{r['sc_direct_effect']:.4f}",
            'SC总效应': f"{r['sc_total_effect']:.4f}",
            'SC中介比例': f"{sc_prop:.4f}" if not np.isnan(sc_prop) else "NA",
            'SC Bootstrap CI低': f"{r['sc_boot_ci_low']:.4f}",
            'SC Bootstrap CI高': f"{r['sc_boot_ci_high']:.4f}",
            'SC路径a β_std': f"{r['sc_path_a_std']:.4f}",
            'SC路径b β_std': f"{r['sc_path_b_std']:.4f}",
            'SC路径c β_std': f"{r['sc_path_c_std']:.4f}",
            'SC路径c\' β_std': f"{r['sc_path_c_prime_std']:.4f}",
            'SC_Sobel_Z': f"{r['sc_sobel_z']:.4f}",
            'SC_Sobel_p': f"{r['sc_sobel_p']:.4f}",
            # FC中介分析
            'FC路径a系数': f"{r['fc_path_a_coef']:.4f}",
            'FC路径a_p值': f"{r['fc_path_a_p']:.4f}",
            'FC路径b系数': f"{r['fc_path_b_coef']:.4f}",
            'FC路径b_p值': f"{r['fc_path_b_p']:.4f}",
            'FC路径c系数(总效应)': f"{r['fc_path_c_coef']:.4f}",
            'FC路径c_p值': f"{r['fc_path_c_p']:.4f}",
            'FC路径c\'系数(直接效应)': f"{r['fc_path_c_prime_coef']:.4f}",
            'FC路径c\'_p值': f"{r['fc_path_c_prime_p']:.4f}",
            'FC间接效应': f"{r['fc_indirect_effect']:.4f}",
            'FC直接效应': f"{r['fc_direct_effect']:.4f}",
            'FC总效应': f"{r['fc_total_effect']:.4f}",
            'FC中介比例': f"{fc_prop:.4f}" if not np.isnan(fc_prop) else "NA",
            'FC Bootstrap CI低': f"{r['fc_boot_ci_low']:.4f}",
            'FC Bootstrap CI高': f"{r['fc_boot_ci_high']:.4f}",
            'FC路径a β_std': f"{r['fc_path_a_std']:.4f}",
            'FC路径b β_std': f"{r['fc_path_b_std']:.4f}",
            'FC路径c β_std': f"{r['fc_path_c_std']:.4f}",
            'FC路径c\' β_std': f"{r['fc_path_c_prime_std']:.4f}",
            'FC_Sobel_Z': f"{r['fc_sobel_z']:.4f}",
            'FC_Sobel_p': f"{r['fc_sobel_p']:.4f}",
            '样本量': f"{int(r['sc_n'])}" if not np.isnan(r['sc_n']) else "NA"
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(
            SFC_TXT_DIR,
            'mediation_analysis_summary.csv'
        )
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"摘要结果已保存至: {summary_path}")
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()

