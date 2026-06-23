import numpy as np
import os
import pandas as pd

def load_all_sfc_arrays(sfc_dir,roi_number):
    """从目录中加载所有受试者的 SFC 数组（长度应为400，允许含NaN）。"""
    arrays = []

    for fname in os.listdir(sfc_dir):
        if fname.endswith("_sfc_values_pca_t288.txt"):
            fpath = os.path.join(sfc_dir, fname)

            vals = []
            with open(fpath, 'r') as f:
                lines = f.readlines()
                for line in lines[3:]:
                    if line.strip() and '\t' in line:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            try:
                                vals.append(float(parts[1]))
                            except ValueError:
                                if parts[1].strip().lower() == 'nan':
                                    vals.append(np.nan)
                                else:
                                    vals.append(np.nan)
            if len(vals) == roi_number:
                arrays.append(np.array(vals))


    return arrays


def compute_overall_average_sfc(sfc_dir,roi_number):
    """
    计算目录下所有受试者的整体平均 SFC（忽略 NaN）。
    返回 avg_sfc (400,)、有效计数 valid_counts (400,) 和值范围 [min, max]
    """
    all_mat = load_all_sfc_arrays(sfc_dir,roi_number)
    if not all_mat:
        print("No SFC arrays found in directory.")
        return None, None, None

    # 将列表转换为numpy数组
    all_mat = np.vstack(all_mat)
    valid_counts = np.sum(~np.isnan(all_mat), axis=0)
    avg_sfc = np.nanmean(all_mat, axis=0)

    # 计算值范围
    sfc_min = np.nanmin(avg_sfc)
    sfc_max = np.nanmax(avg_sfc)

    print(
        f"Overall SFC: subjects={all_mat.shape[0]}, ROI valid count range=({valid_counts.min()}-{valid_counts.max()})")
    print(f"Overall SFC value range: [{sfc_min:.4f}, {sfc_max:.4f}]")

    all_nan_indices = np.where(valid_counts == 0)[0]
    if len(all_nan_indices) > 0:
        print(f"Overall all-NaN ROIs across all subjects: {all_nan_indices}")
    return avg_sfc, valid_counts, [sfc_min, sfc_max]




def load_subject_categories(csv_file_path):
    """从新的 TSV 文件读取受试者（只对含 M 的受试者）失语类别映射"""
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


def load_sfc_data_by_category(subject_entries, subject_categories, roi_number):
    """结合目录扫描结果与 CSV 分类，读取各受试者的 SFC 数据"""
    category_data = {}
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
    """
    计算每个类别的平均SFC值，同时计算每个类别的值范围
    """
    category_averages = {}

    for aphasia_type, subjects in category_data.items():
        if subjects:
            # 收集所有受试者的SFC数据
            all_sfc_arrays = [subject['sfc_values'] for subject in subjects]

            # 转为二维数组: (n_subjects, n_rois)
            all_sfc_matrix = np.vstack(all_sfc_arrays)

            # 统计每个ROI的有效（非NaN）计数
            valid_counts = np.sum(~np.isnan(all_sfc_matrix), axis=0)
            all_nan_indices = np.where(valid_counts == 0)[0]
            some_valid_indices = np.where((valid_counts > 0) & (valid_counts < all_sfc_matrix.shape[0]))[0]

            # 计算平均值（忽略NaN值）
            avg_sfc = np.nanmean(all_sfc_matrix, axis=0)

            # 计算该类别的值范围（忽略NaN）
            sfc_min = np.nanmin(avg_sfc)
            sfc_max = np.nanmax(avg_sfc)
            value_range = [sfc_min, sfc_max]

            print(
                f"Aphasia type {aphasia_type}: ROI valid counts summary -> min={valid_counts.min()}, max={valid_counts.max()}, mean={valid_counts.mean():.2f}")
            print(f"Aphasia type {aphasia_type}: SFC value range: [{sfc_min:.4f}, {sfc_max:.4f}]")

            if len(all_nan_indices) > 0:
                print(f"All-NaN ROIs across subjects ({len(all_nan_indices)}): {all_nan_indices}")
            else:
                print("No ROIs are all-NaN across subjects for this category.")
            if len(some_valid_indices) > 0:
                print(f"ROIs with partial NaNs (some subjects missing) count: {len(some_valid_indices)}")

            # 计算并打印 NaN 值的数量和百分比
            nan_count = np.sum(np.isnan(avg_sfc))
            total_rois = len(avg_sfc)
            print(f"Number of NaN values in avg_sfc: {nan_count}/{total_rois}")

            category_averages[aphasia_type] = {
                'average_sfc': avg_sfc,
                'n_subjects': len(subjects),
                'subject_ids': [subject['subject_id'] for subject in subjects],
                'value_range': value_range  # 存储该类别的值范围
            }

            print(f"Aphasia type {aphasia_type}: {len(subjects)} subjects, average SFC computed")

    return category_averages


def save_category_average_sfc(category_averages, save_dir):
    """
    将每个失语症类别的平均 SFC (长度为 ROI 数) 保存为 TXT 文件。
    文件命名为 {aphasia_type}_average_sfc.txt。
    """
    if not category_averages:
        print("No category averages to save.")
        return

    os.makedirs(save_dir, exist_ok=True)

    for aphasia_type, data in category_averages.items():
        avg_sfc = data.get('average_sfc')
        if avg_sfc is None:
            print(f"Skipping {aphasia_type}: no average SFC available.")
            continue

        output_path = os.path.join(save_dir, f"{aphasia_type}_average_sfc.txt")
        with open(output_path, 'w') as f:
            f.write("ROI_Index\tAverage_SFC\n")
            for idx, value in enumerate(avg_sfc):
                if np.isnan(value):
                    f.write(f"{idx}\tNaN\n")
                else:
                    f.write(f"{idx}\t{value:.6f}\n")
        print(f"Saved average SFC for aphasia type {aphasia_type} to: {output_path}")


# --------------------- 新增：七个网络箱型图功能 ---------------------
def parse_network_atlas(atlas_file_path):
    """
    解析包含皮层下50个ROI + Schaefer400皮层ROI的标签文件，
    返回皮层 ROI (400个) 到七个网络的映射。

    文件格式示例（两行一组）:
        7Networks_LH_Vis_1
        51 120 17 128 255

    参数:
    - atlas_file_path: TXT标签文件路径

    返回:
    - roi_to_network: 字典，皮层ROI索引(0-based, 0-399) -> 网络名称
    - network_names: 七个网络名称列表
    """
    roi_to_network = {}

    try:
        with open(atlas_file_path, 'r') as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        if len(raw_lines) % 2 != 0:
            print("Warning: Atlas label file has unmatched lines, some entries may be skipped.")

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

            # 只解析皮层ROI（编号51-450，对应400个Schaefer分区）
            if roi_label <= 50:
                continue

            if not roi_name.startswith("7Networks"):
                continue

            network_parts = roi_name.split('_')
            if len(network_parts) < 3:
                continue
            network_name = network_parts[2]

            cortical_index = roi_label - 51  # 将51-450映射到0-399
            if 0 <= cortical_index < 400:
                roi_to_network[cortical_index] = network_name

    except Exception as e:
        print(f"Error parsing atlas file: {str(e)}")
        return None, None

    if not roi_to_network:
        print("No cortical ROI-network mappings were parsed.")
        return None, None

    network_names = sorted(set(roi_to_network.values()))

    print(f"Parsed atlas: {len(roi_to_network)} cortical ROIs mapped to {len(network_names)} networks")
    print(f"Networks found: {network_names}")

    return roi_to_network, network_names


def compute_network_sfc_by_subject(category_data, roi_to_network, network_names, cortical_offset=50):
    """
    计算每个受试者在七个网络上的平均 SFC，并按失语症类型返回。

    参数:
    - category_data: {aphasia_type: [{'subject_id': str, 'sfc_values': np.ndarray}, ...]}
    - roi_to_network: dict，ROI (0-based, 400 cortical) -> 网络名称
    - network_names: 网络名称列表
    - cortical_offset: 皮层 ROI 在 sfc_values 中的起始索引（默认50）

    返回:
    - dict: {aphasia_type: [ {'subject_id': str, network1: val, ...}, ... ] }
    """
    if not category_data:
        return {}

    # 预先计算每个网络对应的 sfc_values 索引（考虑皮层偏移）
    network_indices = {
        network: [cortical_offset + roi_idx for roi_idx, net in roi_to_network.items() if net == network]
        for network in network_names
    }

    network_sfc_by_category = {}

    for aphasia_type, subjects in category_data.items():
        rows = []
        for subject in subjects:
            sfc_values = subject['sfc_values']
            row = {'subject_id': subject['subject_id']}

            for network, indices in network_indices.items():
                if not indices:
                    row[network] = np.nan
                    continue

                # 过滤掉越界索引
                valid_indices = [idx for idx in indices if idx < len(sfc_values)]
                if not valid_indices:
                    row[network] = np.nan
                    continue

                network_values = sfc_values[valid_indices]
                if np.all(np.isnan(network_values)):
                    row[network] = np.nan
                else:
                    row[network] = float(np.nanmedian(network_values))

            rows.append(row)

        network_sfc_by_category[aphasia_type] = rows

    return network_sfc_by_category


def save_network_sfc_csvs(network_sfc_by_category, network_names, save_dir):
    """
    将每个失语症类别的七个网络 SFC 均值保存为 CSV 文件。
    每个 CSV 包含 subject_id + 七个网络列。
    """
    if not network_sfc_by_category:
        print("No network-level SFC data to save.")
        return

    os.makedirs(save_dir, exist_ok=True)

    for aphasia_type, rows in network_sfc_by_category.items():
        if not rows:
            print(f"No network data for {aphasia_type}, skipping CSV export.")
            continue

        df = pd.DataFrame(rows)
        ordered_cols = ['subject_id'] + network_names
        for col in ordered_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[ordered_cols]

        csv_path = os.path.join(save_dir, f"{aphasia_type}_network_sfc_median.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved network-level SFC CSV for aphasia type {aphasia_type} to: {csv_path}")


def filter_subjects_from_tsv(subjects_metadata, csv_file_path, save_dir):
    """
    从TSV文件中筛选出在SFC文件夹中存在的受试者，保存为新的CSV。
    
    参数:
    - subjects_metadata: 从SFC目录扫描得到的受试者列表
    - csv_file_path: 原始受试者信息TSV文件路径
    - save_dir: 保存输出CSV的目录
    """
    try:
        # 读取原始TSV文件
        df_all = pd.read_csv(csv_file_path, sep='\t')
        print(f"从TSV文件读取到 {len(df_all)} 条受试者记录")
        
        # 提取SFC文件夹中的失语受试者ID（只提取含M的受试者）
        sfc_subject_ids = []
        healthy_count = 0
        for entry in subjects_metadata:
            if not entry['is_healthy'] and entry['csv_key']:
                sfc_subject_ids.append(entry['csv_key'])
            elif entry['is_healthy']:
                healthy_count += 1
        
        print(f"SFC文件夹中找到 {len(sfc_subject_ids)} 个失语受试者（含M）")
        print(f"SFC文件夹中找到 {healthy_count} 个健康受试者（不含M，不在TSV中）")
        
        # 从TSV中筛选出存在于SFC文件夹中的受试者
        df_filtered = df_all[df_all['participant_id'].isin(sfc_subject_ids)]
        
        # 保存筛选后的CSV
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, "filtered_participants.csv")
        df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n筛选后的受试者信息已保存到: {output_path}")
        print(f"筛选出 {len(df_filtered)} 条记录（在SFC文件夹中存在）")
        print(f"未匹配 {len(sfc_subject_ids) - len(df_filtered)} 个受试者（在SFC文件夹中但不在TSV中）")
        
        # 显示筛选后的失语类型分布
        if 'wab_type' in df_filtered.columns:
            type_counts = df_filtered['wab_type'].value_counts()
            print("\n筛选后的失语类型分布:")
            for aphasia_type, count in type_counts.items():
                print(f"  {aphasia_type}: {count} 个受试者")
        
        return df_filtered
        
    except Exception as e:
        print(f"筛选受试者信息时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# --------------------- 主流程：按类别画图 ---------------------
if __name__ == "__main__":
    roi_number = 450
    csv_file_path = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\aphasia_plots\participants_aphasia_new.txt"
    sfc_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\sfc_txt_files"

    # "D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_time288s\sfc_txt_files\sub-sub-2748_sfc_values_pca_t288.txt"


    # 1. 首先从SFC目录获取所有受试者列表
    print("加载所有的sfc...")
    subjects_metadata = load_all_subjects_from_sfc_dir(sfc_dir)

    # 2. 加载受试者类别信息
    print("加载受试者信息..")
    subject_categories = load_subject_categories(csv_file_path)

    # 3. 显示匹配情况
    aphasia_with_category = []
    aphasia_without_category = []
    for entry in subjects_metadata:
        if entry['is_healthy']:
            aphasia_with_category.append(entry['subject_label'])
        else:
            csv_key = entry['csv_key']
            if csv_key in subject_categories:
                aphasia_with_category.append(entry['subject_label'])
            else:
                aphasia_without_category.append(entry['subject_label'])

    print(f"Found {len(subjects_metadata)} subjects in SFC directory")
    print(f"Found {len(subject_categories)} aphasia subjects in CSV")
    print(f"Matched {len(aphasia_with_category)} subjects (healthy + 有类型的失语)")
    if aphasia_without_category:
        print(f"Subjects without category info (will be skipped): {aphasia_without_category}")

    # 4. 筛选并保存存在于SFC文件夹中的受试者信息
    output_dir = r"single_subject_results_t288"
    os.makedirs(output_dir, exist_ok=True)
    print("\n从TSV筛选SFC文件夹中存在的受试者...")
    filtered_df = filter_subjects_from_tsv(subjects_metadata, csv_file_path, output_dir)

    # 5. 按类别加载SFC数据
    print("\nLoading SFC data by category...")
    category_data = load_sfc_data_by_category(subjects_metadata, subject_categories, roi_number)

    # 6. 计算每个类别的平均SFC（包含值范围）
    print("Computing average SFC for each category...")
    category_averages = compute_category_average_sfc(category_data)

    # 7. 保存每个失语类型的平均 SFC
    print(f"Saving average SFC per aphasia type to: {output_dir}")
    save_category_average_sfc(category_averages, output_dir)

    # 8. 解析七网络分区并导出网络级 SFC CSV
    atlas_file_path = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
    if os.path.exists(atlas_file_path):
        roi_to_network, network_names = parse_network_atlas(atlas_file_path)
        if roi_to_network and network_names:
            network_sfc_by_category = compute_network_sfc_by_subject(
                category_data, roi_to_network, network_names, cortical_offset=50
            )
            save_network_sfc_csvs(network_sfc_by_category, network_names, output_dir)
        else:
            print("Failed to parse network atlas; skipping network-level CSV export.")
    else:
        print(f"Atlas file not found: {atlas_file_path}. Skipping network-level CSV export.")

    # 9. 显示统计信息
    print("\nSummary:")
    for aphasia_type, data in category_averages.items():
        valid_sfc = data['average_sfc'][~np.isnan(data['average_sfc'])]
        print(f"Aphasia type {aphasia_type}: {data['n_subjects']} subjects, "
              f"mean SFC: {np.nanmean(valid_sfc):.4f}, "
              f"std SFC: {np.nanstd(valid_sfc):.4f}, "
              f"value range: [{data['value_range'][0]:.4f}, {data['value_range'][1]:.4f}]")
