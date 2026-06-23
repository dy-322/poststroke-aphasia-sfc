import numpy as np
import os
import scipy.stats as stats
import glob
import re
import pandas as pd
import argparse
from nilearn.connectome import ConnectivityMeasure
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist, squareform
import scipy.io as sio
import matplotlib.pyplot as plt

# 设置路径
base_data_path = os.path.join(os.getcwd(), "data", "prepared_subjects")
# base_data_path = r"D:\controlPaperReview\data\complete_subjects_nosubcotex"
# centroid_file = r"D:\download\Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"

# 创建结果目录
result_dir = os.path.join(os.getcwd(), "batch_results_pca_wSub")
os.makedirs(result_dir, exist_ok=True)
sfc_txt_dir = None  # 在 main 中按方法名设置
images_dir = os.path.join(result_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# ROI数量将在运行时根据参数确定
roi_number = None  # 将在main函数中设置

# TR映射文件路径
TR_CSV_FILE = os.path.join(os.getcwd(), "data", "subject_tr.csv")
_tr_cache = None  # 缓存TR映射字典


def load_centroid_coordinates(centroid_file):
    """读取脑区质心坐标"""
    print(f"Loading centroid coordinates: {centroid_file}")
    df = pd.read_csv(centroid_file)
    
    # 假设坐标列名为 'x', 'y', 'z' 或类似
    coord_cols = [col for col in df.columns if col.lower() in ['x', 'y', 'z']]
    if len(coord_cols) != 3:
        # 如果没找到标准列名，尝试其他可能的列名
        coord_cols = df.columns[-3:].tolist()  # 取最后3列作为坐标
    
    coordinates = df[coord_cols].values
    print(f"Loaded {len(coordinates)} ROI coordinates")
    return coordinates


def compute_euclidean_distance_matrix(coordinates):
    """计算脑区之间的欧式距离矩阵"""
    print("Computing Euclidean distance matrix...")
    distances = pdist(coordinates, metric='euclidean')
    distance_matrix = squareform(distances)
    return distance_matrix


def load_structural_matrices(mat_file):
    """从mat文件中提取结构矩阵"""
    print(f"Loading structural matrices: {mat_file}")
    mat_data = sio.loadmat(mat_file)
    
    # 提取mats变量
    if 'mats' in mat_data:
        mats = mat_data['mats']
        print(f"Loaded mats shape: {mats.shape}")
        
        # 如果是3D数组，需要重新整形
        if len(mats.shape) == 3:
            # 判断是 (n_matrices, roi_number, roi_number) 还是 (roi_number, roi_number, n_matrices)
            # 如果第一维等于第二维，且第三维不同，则是 (roi_number, roi_number, n_matrices)
            if mats.shape[0] == mats.shape[1] and mats.shape[2] != mats.shape[0]:
                # (roi_number, roi_number, n_matrices) -> (n_matrices, roi_number, roi_number)
                mats = np.transpose(mats, (2, 0, 1))
                print(f"Transposed from (roi, roi, n) to (n, roi, roi): {mats.shape}")
            # 否则假设已经是 (n_matrices, roi_number, roi_number) 格式
        elif len(mats.shape) == 2:
            # 如果是2D，可能是单个矩阵
            mats = mats.reshape(1, mats.shape[0], mats.shape[1])
        
        print(f"Final mats shape: {mats.shape}")
        return mats
    else:
        raise ValueError("'mats' variable not found in mat file")


def load_original_sc(sc_file):
    """读取原始SC矩阵"""
    print(f"Loading original SC matrix: {sc_file}")
    sc = np.loadtxt(sc_file, delimiter=',')
    if sc.shape != (roi_number, roi_number):
        raise ValueError(f"SC matrix shape mismatch: expected ({roi_number},{roi_number}), got {sc.shape}")
    return sc


def load_fc_timeseries(fc_file, remove_first_n=10):
    """加载FC时间序列并去除前N个时间点"""
    print(f"Loading FC timeseries: {fc_file}")
    data = []
    with open(fc_file, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:

                # 检查是否是"Row X:"格式
                if line.startswith('Row ') and ':' in line:
                    # 提取冒号后的数据部分
                    data_part = line.split(':', 1)[1].strip()
                    row = np.array(data_part.split(','), dtype=float)
                else:
                    # 标准格式，按空格分割
                    row = np.array(line.split(), dtype=float)

                # if len(row) != roi_number:
                #     raise ValueError(f"FC timeseries row length mismatch: expected {roi_number}, got {len(row)}")
                data.append(row)

    
    data = np.array(data).T
    print(f"Original timeseries shape: {data.shape}")

    # import pdb;pdb.set_trace()
    
    # 去除前N个时间点
    if data.shape[0] > remove_first_n:
        data = data[remove_first_n:, :]
        print(f"After removing first {remove_first_n} timepoints: {data.shape}")
    else:
        print(f"Warning: timeseries too short ({data.shape[0]} timepoints), cannot remove {remove_first_n}")
    
    return data


def load_tr_mapping():
    """加载TR映射字典"""
    global _tr_cache
    if _tr_cache is not None:
        return _tr_cache
    
    _tr_cache = {}
    if os.path.exists(TR_CSV_FILE):
        print(f"Loading TR mapping from: {TR_CSV_FILE}")
        tr_df = pd.read_csv(TR_CSV_FILE)
        for _, row in tr_df.iterrows():
            subject_id = row['subject_id']
            tr_value = float(row['tr'])
            _tr_cache[subject_id] = tr_value
        print(f"Loaded TR values for {len(_tr_cache)} subjects")
    else:
        print(f"Warning: TR mapping file not found: {TR_CSV_FILE}")
    
    return _tr_cache


def get_subject_tr(subject_id):
    """根据subject_id获取TR值"""
    # 如果subject_id中没有'M'，则TR=2
    if 'M' not in subject_id:
        return 2.0
    
    # 如果subject_id中有'M'，从CSV文件中查找
    tr_mapping = load_tr_mapping()
    full_subject_id = f"sub-{subject_id}" if not subject_id.startswith("sub-") else subject_id
    
    if full_subject_id in tr_mapping:
        return tr_mapping[full_subject_id]
    else:
        # 如果找不到，使用默认值（可能需要警告）
        print(f"Warning: TR not found for subject {subject_id}, using default TR=2.0")
        return 2.0


def truncate_timeseries_by_real_time(timeseries, target_real_time, tr_value):
    """根据真实时间截断时间序列
    真实时间 = TR × 时间点数
    
    Args:
        timeseries: 时间序列数据
        target_real_time: 目标真实时间（秒）
        tr_value: TR值（秒）
    
    Returns:
        截断后的时间序列
    """
    if target_real_time is None or target_real_time <= 0:
        return timeseries
    
    # 根据真实时间计算应该保留的时间点数
    target_timepoints = int(target_real_time / tr_value)
    
    if target_timepoints <= 0:
        raise ValueError(f"Target real time ({target_real_time}s) with TR ({tr_value}s) results in {target_timepoints} timepoints (must be > 0)")
    
    return truncate_timeseries_length(timeseries, target_timepoints)


def truncate_timeseries_length(timeseries, target_length):
    """将时间序列截断到统一长度"""
    if target_length is None or target_length <= 0:
        return timeseries
    
    if timeseries.ndim == 1:
        timeseries = timeseries[:, np.newaxis]
    
    n_timepoints = timeseries.shape[0]
    
    if n_timepoints < target_length:
        raise ValueError(
            f"Timeseries length ({n_timepoints}) is shorter than target length ({target_length})"
        )
    
    if n_timepoints > target_length:
        print(f"Truncating timeseries from {n_timepoints} to {target_length} timepoints")
        timeseries = timeseries[:target_length, :]
    
    return timeseries


def compute_fc_matrix(timeseries_data, apply_fisher_z=True):
    """计算FC矩阵（Fisher z转换的Pearson相关系数）"""
    print("Computing FC matrix...")
    
    conn_measure = ConnectivityMeasure(kind='correlation', standardize=True)
    fc = conn_measure.fit_transform([timeseries_data])[0]
    
    # 应用Fisher z变换
    if apply_fisher_z:
        # 避免+-1导致的无穷大
        fc = np.clip(fc, -0.999999, 0.999999)
        fc = np.arctanh(fc)
        print("Applied Fisher z-transform to FC matrix")

    print(fc.shape)
    
    return fc


def prepare_structural_features(structural_matrices, original_sc, distance_matrix):
    """准备结构特征：组合结构矩阵、原始SC和距离矩阵"""
    print("Preparing structural features...")
    
    # 展平所有结构矩阵
    n_matrices = structural_matrices.shape[0]
    n_rois = roi_number
    expected_size = n_rois * n_rois
    
    # 验证所有矩阵的形状
    print(f"Expected flattened size: {expected_size}")
    print(f"Original SC shape: {original_sc.shape}, flattened size: {original_sc.size}")
    print(f"Distance matrix shape: {distance_matrix.shape}, flattened size: {distance_matrix.size}")
    print(f"Structural matrices shape: {structural_matrices.shape}")
    
    # 确保所有矩阵都是 (n_rois, n_rois) 形状
    if original_sc.shape != (n_rois, n_rois):
        raise ValueError(f"Original SC shape mismatch: expected ({n_rois}, {n_rois}), got {original_sc.shape}")
    
    if distance_matrix.shape != (n_rois, n_rois):
        raise ValueError(f"Distance matrix shape mismatch: expected ({n_rois}, {n_rois}), got {distance_matrix.shape}")
    
    # 验证结构矩阵的形状
    for i in range(n_matrices):
        if structural_matrices[i].shape != (n_rois, n_rois):
            raise ValueError(f"Structural matrix {i} shape mismatch: expected ({n_rois}, {n_rois}), got {structural_matrices[i].shape}")
    
    # 存储所有结构特征
    structural_features = []
    
    # 添加原始SC
    sc_flat = original_sc.flatten()
    structural_features.append(sc_flat)
    print(f"Added original SC: shape {sc_flat.shape}")
    
    # 添加距离矩阵
    dist_flat = distance_matrix.flatten()
    structural_features.append(dist_flat)
    print(f"Added distance matrix: shape {dist_flat.shape}")
    
    # 添加所有结构矩阵
    for i in range(n_matrices):
        mat_flat = structural_matrices[i].flatten()
        structural_features.append(mat_flat)
        if i < 3:  # 只打印前3个的详细信息
            print(f"Added structural matrix {i}: shape {mat_flat.shape}")
    
    # 验证所有展平后的数组长度一致
    sizes = [len(feat) for feat in structural_features]
    if len(set(sizes)) > 1:
        raise ValueError(f"Inconsistent feature sizes: {sizes}")
    
    # 转换为numpy数组 (n_features, n_rois^2)
    # 使用明确的数据类型和形状
    structural_features = np.vstack(structural_features).astype(np.float64)
    print(f"Structural features shape: {structural_features.shape}")
    
    return structural_features


def apply_pca_to_structural_features(structural_features, n_components=4, explained_variance_threshold=0.8):
    """对结构特征应用PCA降维"""
    print(f"Applying PCA with {n_components} components...")
    
    # 确保输入是数值数组
    if structural_features.dtype == object:
        raise ValueError("Structural features is an object array. Check that all matrices have the same shape.")
    
    # 转置以便PCA处理 (n_rois^2, n_features)
    features_for_pca = structural_features.T
    
    print(f"Features for PCA shape: {features_for_pca.shape}, dtype: {features_for_pca.dtype}")
    
    # 处理无穷大值和NaN值
    print("Cleaning infinite and NaN values...")
    
    # 统计无穷大值和NaN值的数量
    inf_count = np.sum(np.isinf(features_for_pca))
    nan_count = np.sum(np.isnan(features_for_pca))
    print(f"Found {inf_count} infinite values and {nan_count} NaN values")
    
    # 将无穷大值替换为最大有限值
    finite_values = features_for_pca[np.isfinite(features_for_pca)]
    if len(finite_values) > 0:
        max_finite = np.max(finite_values)
        min_finite = np.min(finite_values)
        print(f"Finite value range: [{min_finite:.6f}, {max_finite:.6f}]")
        
        # 替换无穷大值
        features_for_pca = np.where(np.isinf(features_for_pca), 
                                   np.sign(features_for_pca) * max_finite, 
                                   features_for_pca)
        
        # 替换NaN值为0
        features_for_pca = np.where(np.isnan(features_for_pca), 0, features_for_pca)
        
        print("Replaced infinite values with finite bounds and NaN values with 0")
    else:
        print("Warning: No finite values found!")
        features_for_pca = np.zeros_like(features_for_pca)
    
    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_for_pca)
    
    # 应用PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_scaled)
    
    # 检查解释方差
    explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
    print(f"Explained variance ratio: {explained_variance_ratio:.4f}")
    
    if explained_variance_ratio < explained_variance_threshold:
        print(f"Warning: Explained variance ({explained_variance_ratio:.4f}) is less than threshold ({explained_variance_threshold})")
    
    # 转置回 (n_components, n_rois^2)
    pca_result = pca_result.T
    
    print(f"PCA result shape: {pca_result.shape}")
    return pca_result, pca, scaler


def compute_sfc_pca_method(fc_matrix, structural_pca_features):
    """使用PCA方法计算SFC值"""
    print("Computing SFC using PCA method...")
    
    n_rois = roi_number
    n_components = structural_pca_features.shape[0]
    sfc_values = []
    
    for roi_idx in range(n_rois):
        # 提取该ROI的FC轮廓（行向量）
        fc_profile = fc_matrix[roi_idx, :]
        
        # 提取该ROI对应的结构PCA特征
        # structural_pca_features shape: (n_components, n_rois^2)
        # 需要重新整形为 (n_components, n_rois, n_rois)
        structural_pca_matrix = structural_pca_features.reshape(n_components, n_rois, n_rois)
        
        # 提取该ROI对应的结构特征（每行对应一个主成分）
        roi_structural_features = []
        for pc_idx in range(n_components):
            roi_structural_features.append(structural_pca_matrix[pc_idx, roi_idx, :])
        
        roi_structural_features = np.array(roi_structural_features).T  # (n_rois, n_components)
        
        # 移除自连接
        fc_profile_no_self = np.delete(fc_profile, roi_idx)
        structural_features_no_self = np.delete(roi_structural_features, roi_idx, axis=0)
        
        # 过滤非有限值
        finite_mask = np.isfinite(fc_profile_no_self) & np.all(np.isfinite(structural_features_no_self), axis=1)
        fc_valid = fc_profile_no_self[finite_mask]
        structural_valid = structural_features_no_self[finite_mask]
        
        if len(fc_valid) < 2 or structural_valid.shape[1] == 0:
            sfc_values.append(np.nan)
            continue
        
        # 多元线性回归

        reg = LinearRegression()
        reg.fit(structural_valid, fc_valid)
        predicted_fc = reg.predict(structural_valid)

        # 计算预测FC与实际FC的Pearson相关系数
        if len(predicted_fc) > 1 and np.std(predicted_fc) > 0 and np.std(fc_valid) > 0:
            corr, _ = stats.pearsonr(predicted_fc, fc_valid)
            sfc_values.append(corr)
        else:
            sfc_values.append(np.nan)

    
    print(f"Computed SFC values for {len(sfc_values)} ROIs")
    return sfc_values


def find_subject_files():
    """查找所有受试者的文件"""
    print("Finding subject files...")
    
    # 查找所有受试者文件夹
    subject_dirs = glob.glob(os.path.join(base_data_path, "*"))
    subjects_info = []
    
    for subject_dir in subject_dirs:
        if not os.path.isdir(subject_dir):
            continue
            
        # 从路径中提取受试者ID
        subject_id = os.path.basename(subject_dir)
        
        # 构建文件路径
        mat_file = os.path.join(subject_dir, "mats.mat")
        fc_file = os.path.join(subject_dir, "fc_400_template.csv")
        sc_file = os.path.join(subject_dir, "schaefer400_connectome.csv")

        if not os.path.exists(fc_file):

            fc_pattern = os.path.join(subject_dir, "*task-rest*.txt")
            fc_files = glob.glob(fc_pattern)
            if fc_files:
                fc_file = fc_files[0]  # 使用第一个匹配的文件
                print(f"Found alternative FC file for {subject_id}: {fc_file}")
            else:
                print(f"No FC file containing '400' found for {subject_id}")




        
        # 检查文件是否存在
        if all(os.path.exists(f) for f in [mat_file, fc_file, sc_file]):
            subjects_info.append({
                'subject_id': subject_id,
                'mat_file': mat_file,
                'fc_file': fc_file,
                'sc_file': sc_file
            })
            print(f"Found subject {subject_id}")
        else:
            missing_files = [f for f in [mat_file, fc_file, sc_file] if not os.path.exists(f)]
            print(f"Warning: Subject {subject_id} missing files: {missing_files}")
    
    print(f"Total subjects found: {len(subjects_info)}")
    return subjects_info


def save_sfc_to_txt(subject_id, sfc_values):
    """保存单个受试者的SFC值到txt文件"""
    filename = f"sub-{subject_id}_sfc_values_pca.txt"
    filepath = os.path.join(sfc_txt_dir, filename)
    

    with open(filepath, 'w') as f:
        f.write(f"Subject: sub-{subject_id}\n")
        f.write(f"Method: PCA\n")
        f.write(f"ROI_Index\tSFC_Value\n")
        f.write("-" * 30 + "\n")

        for roi_idx, sfc_val in enumerate(sfc_values):
            if np.isnan(sfc_val):
                f.write(f"{roi_idx+1}\tNaN\n")
            else:
                f.write(f"{roi_idx+1}\t{sfc_val:.6f}\n")

    print(f"Subject {subject_id} SFC values saved to: {filepath}")
    return filepath



def process_single_subject(
    subject_info,
    distance_matrix,
    target_real_time,
    n_components=4,
    explained_variance_threshold=0.8,
    remove_first_n=10,
):
    """处理单个受试者"""
    subject_id = subject_info['subject_id']
    print(f"\nProcessing subject: {subject_id}")
    
    # 获取该subject的TR值
    tr_value = get_subject_tr(subject_id)
    print(f"Subject {subject_id} TR: {tr_value}s")
    
    # 根据真实时间计算目标时间点数
    if target_real_time is not None and target_real_time > 0:
        target_timepoints = int(target_real_time / tr_value)
        print(f"Target real time: {target_real_time}s, target timepoints: {target_timepoints}")
    else:
        target_timepoints = None

    # 加载数据
    structural_matrices = load_structural_matrices(subject_info['mat_file'])
    original_sc = load_original_sc(subject_info['sc_file'])
    if subject_info['fc_file'].endswith('.csv'):

        fc_timeseries = load_fc_timeseries(subject_info['fc_file'], remove_first_n=remove_first_n)
        print(f"=={subject_id}====fc 形状：{fc_timeseries.shape}")

    else:
        fc_timeseries = np.loadtxt(subject_info['fc_file'])
        if fc_timeseries.shape[0] > remove_first_n:
            fc_timeseries = fc_timeseries[remove_first_n:, :]
            print(f"After removing first {remove_first_n} timepoints: {fc_timeseries.shape}")
        else:
            print(f"Warning: timeseries too short ({fc_timeseries.shape[0]} timepoints), cannot remove {remove_first_n}")
        print(f"=={subject_id}====fc 形状：{fc_timeseries.shape}")

    # 根据真实时间截断时间序列
    fc_timeseries = truncate_timeseries_by_real_time(fc_timeseries, target_real_time, tr_value)
    fc_matrix = compute_fc_matrix(fc_timeseries, apply_fisher_z=True)

    # 准备结构特征
    structural_features = prepare_structural_features(
        structural_matrices, original_sc, distance_matrix
    )

    # 应用PCA
    structural_pca_features, pca, scaler = apply_pca_to_structural_features(
        structural_features,
        n_components=n_components,
        explained_variance_threshold=explained_variance_threshold,
    )

    # 计算SFC
    sfc_values = compute_sfc_pca_method(fc_matrix, structural_pca_features)

    # 保存结果
    txt_filepath = save_sfc_to_txt(subject_id, sfc_values)

    return {
        'subject_id': subject_id,
        'sfc_values': sfc_values,
        'sfc_txt_file': txt_filepath,
        'pca_explained_variance': np.sum(pca.explained_variance_ratio_),
        'n_structural_matrices': structural_matrices.shape[0]
    }
        

        



def save_results(results):
    """保存结果"""
    print("Saving results...")
    
    # 保存SC-FC coupling结果
    sfc_df = pd.DataFrame([
        {
            'subject_id': result['subject_id'],
            'sfc_txt_file': result.get('sfc_txt_file', None),
            'method': 'PCA',
            'pca_explained_variance': result.get('pca_explained_variance', None),
            'n_structural_matrices': result.get('n_structural_matrices', None)
        }
        for result in results if result is not None
    ])
    
    sfc_save_path = os.path.join(result_dir, 'scfc_coupling_results_pca.csv')
    sfc_df.to_csv(sfc_save_path, index=False)
    print(f"SC-FC coupling results saved to: {sfc_save_path}")
    
    # 打印统计信息
    valid_results = [r for r in results if r is not None]
    print(f"\n=== Processing Summary ===")
    print(f"Total subjects processed: {len(valid_results)}")
    print(f"Failed subjects: {len(results) - len(valid_results)}")
    
    if valid_results:
        explained_variances = [r.get('pca_explained_variance', 0) for r in valid_results]
        print(f"Average PCA explained variance: {np.mean(explained_variances):.4f}")
        print(f"Min PCA explained variance: {np.min(explained_variances):.4f}")
        print(f"Max PCA explained variance: {np.max(explained_variances):.4f}")


def main():
    global roi_number, base_data_path, sfc_txt_dir, result_dir, images_dir, TR_CSV_FILE, _tr_cache
    """主函数"""
    parser = argparse.ArgumentParser(description="Batch SC-FC coupling computation using PCA method")
    parser.add_argument("--roi_number", type=int, default=450, help="Number of ROIs")
    parser.add_argument("--n_components", type=int, default=4, help="Number of PCA components")
    parser.add_argument("--explained_variance_threshold", type=float, default=0.8, 
                       help="Minimum explained variance threshold for PCA")
    parser.add_argument("--target_real_time", type=float, default=288.0,
                       help="Target real time in seconds to keep for each subject (<=0 to disable). Real time = TR × timepoints")
    parser.add_argument("--base_data_path", type=str, default=base_data_path,
                       help="Subject directory root. Each child directory must contain mats.mat, SC csv, and timeseries.")
    parser.add_argument("--result_dir", type=str, default=None,
                       help="Output directory. Defaults to batch_results_pca_<roi>_t288 in the current directory.")
    parser.add_argument("--distance_matrix_file", type=str,
                       default=os.path.join("data", "euclidean_distance_matrix_450.mat"),
                       help="MAT file containing distance_matrix.")
    parser.add_argument("--tr_csv_file", type=str, default=TR_CSV_FILE,
                       help="Optional CSV file with subject_id,tr columns. Missing subjects use TR=2.0.")
    parser.add_argument("--remove_first_n", type=int, default=10,
                       help="Number of initial timeseries timepoints to remove before truncation.")
    args = parser.parse_args()
    
    # 设置全局ROI数量
    roi_number = args.roi_number
    base_data_path = args.base_data_path
    TR_CSV_FILE = args.tr_csv_file
    _tr_cache = None
    
    target_real_time = args.target_real_time if args.target_real_time > 0 else None
    target_time_tag = f"time{target_real_time:.1f}s" if target_real_time else "timefull"

    print(f"Starting batch SC-FC coupling computation using PCA method (ROI number: {roi_number}, target real time: {target_time_tag})...")
    
    # 创建输出目录
    result_dir = args.result_dir or os.path.join(os.getcwd(), f"batch_results_pca_{roi_number}_t288")
    os.makedirs(result_dir, exist_ok=True)
    
    sfc_txt_dir = os.path.join(result_dir, "sfc_txt_files")
    os.makedirs(sfc_txt_dir, exist_ok=True)
    
    # 更新images目录路径
    images_dir = os.path.join(result_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 加载质心坐标并计算距离矩阵
    # coordinates = load_centroid_coordinates(centroid_file)
    # distance_matrix = compute_euclidean_distance_matrix(coordinates)
    distance_matrix = sio.loadmat(args.distance_matrix_file)["distance_matrix"]
    
    # 查找受试者文件
    subjects_info = find_subject_files()
    
    if not subjects_info:
        print("No valid subjects found!")
        return
    
    # 处理所有受试者
    results = []
    for i, subject_info in enumerate(subjects_info):
        print(f"\nProcessing subject {i+1}/{len(subjects_info)}")
        result = process_single_subject(
            subject_info,
            distance_matrix,
            target_real_time,
            n_components=args.n_components,
            explained_variance_threshold=args.explained_variance_threshold,
            remove_first_n=args.remove_first_n,
        )
        results.append(result)
    
    # 保存结果
    save_results(results)
    
    print(f"\nBatch processing completed!")
    print(f"Each subject's {roi_number} ROI SFC values saved to: {sfc_txt_dir}")
    print(f"Truncation strategy: Based on real time ({target_time_tag}) with subject-specific TR values")


if __name__ == "__main__":
    main()
