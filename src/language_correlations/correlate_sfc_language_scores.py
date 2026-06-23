import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
from matplotlib.colorbar import ColorbarBase
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
import tempfile
import nibabel as nib
from nilearn import plotting, datasets
from statsmodels.stats.multitest import multipletests
from nilearn.surface import load_surf_mesh, load_surf_data
from nilearn.datasets import fetch_surf_fsaverage
from brainspace.datasets import load_parcellation
from brainspace.mesh.mesh_io import read_surface
from neuromaps.datasets import fetch_fslr
from neurocat.plotting import draw_and_save as ds
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义常量
NETWORKS = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
SUBCORTICAL_ROI_COUNT = 50
DEFAULT_TOTAL_ROIS = 450
PARTICIPANT_INFO_PATH = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\aphasia_plots\participants_aphasia_new.txt"
COMBINED_ATLAS_PATH = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_MNI152NLin6Asym_1mm.nii.gz"

def load_network_mapping(network_csv_path, subcortical_offset=SUBCORTICAL_ROI_COUNT):

    """从CSV文件加载网络到ROI索引的映射"""
    try:
        # 读取网络属性CSV文件
        network_df = pd.read_csv(network_csv_path)
        
        # 创建ROI索引到网络的映射
        network_mapping = {}
        
        for _, row in network_df.iterrows():
            roi_label = int(row['ROI Label'])
            roi_name = row['ROI Name']
            roi_index = roi_label + subcortical_offset  # 皮层索引需要偏移50个皮层下ROI
            
            # 从ROI名称中提取网络信息
            if 'Vis' in roi_name:
                network_mapping[roi_index] = 'Vis'
            elif 'SomMot' in roi_name:
                network_mapping[roi_index] = 'SomMot'
            elif 'DorsAttn' in roi_name:
                network_mapping[roi_index] = 'DorsAttn'
            elif 'SalVentAttn' in roi_name:
                network_mapping[roi_index] = 'SalVentAttn'
            elif 'Limbic' in roi_name:
                network_mapping[roi_index] = 'Limbic'
            elif 'Cont' in roi_name:
                network_mapping[roi_index] = 'Cont'
            elif 'Default' in roi_name:
                network_mapping[roi_index] = 'Default'
        
        print(f"✅ 成功加载网络映射，包含 {len(network_mapping)} 个ROI")
        return network_mapping
        
    except Exception as e:
        print(f"❌ 加载网络映射时出错：{e}")
        return None


def load_fallback_wab_scores(participant_file_path):
    """从participants_aphasia文件加载WAB AQ备选分数"""
    fallback_scores = {}
    if not participant_file_path or not os.path.exists(participant_file_path):
        print("⚠️ 未提供有效的participants文件路径，无法加载WAB AQ备选分数")
        return fallback_scores

    try:
        participants_df = pd.read_csv(participant_file_path, sep='\t')
    except Exception as e:
        print(f"⚠️ 加载participants文件失败: {e}")
        return fallback_scores

    if 'participant_id' not in participants_df.columns or 'wab_aq' not in participants_df.columns:
        print("⚠️ participants文件缺少participant_id或wab_aq列，无法提供备选分数")
        return fallback_scores

    for _, row in participants_df.iterrows():
        participant_id = str(row['participant_id'])
        wab_aq = row['wab_aq']
        if pd.isna(wab_aq):
            continue

        numeric_part = ''.join(filter(str.isdigit, participant_id))
        if numeric_part:
            fallback_scores[numeric_part] = wab_aq

    print(f"✅ 已加载 {len(fallback_scores)} 个WAB AQ备选分数")
    return fallback_scores


def trim_white_edges(img_array):
    """去除图像白边（与sig_brain_region.py保持一致）"""
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


def normalize_subject_identifier(value):
    """标准化受试者ID（提取数字部分）"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value_str = str(value).strip()
    for prefix in ['sub-M', 'sub-', 'M']:
        if value_str.startswith(prefix):
            value_str = value_str[len(prefix):]
            break
    digits = ''.join(filter(str.isdigit, value_str))
    return digits or value_str


def determine_symmetric_color_range(values, default_range=0.5):
    """根据数据自动确定对称颜色范围"""
    if values is None or len(values) == 0:
        return [-default_range, default_range]

    valid_vals = np.array(values)[~np.isnan(values)]
    if valid_vals.size == 0:
        return [-default_range, default_range]

    max_abs = float(np.nanmax(np.abs(valid_vals)))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = default_range
    return [-max_abs, max_abs]



def create_white_extended_colormap(base_cmap='RdBu_r', white_fraction=0.05):
    """
    创建一个自定义colormap，在最小值处是白色，然后过渡到base_cmap的颜色。
    用于将非显著脑区显示为白色。
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    # 获取基础colormap
    try:
        base = plt.get_cmap(base_cmap)
    except:
        import matplotlib.cm as cm
        base = cm.get_cmap(base_cmap)
    
    # 创建颜色数组
    n_colors = 256
    color_list = []
    
    # 白色部分（用于非显著脑区）
    n_white = int(n_colors * white_fraction)
    for i in range(n_white):
        color_list.append((1.0, 1.0, 1.0, 1.0))  # 白色
    
    # 过渡到base colormap
    n_base = n_colors - n_white
    for i in range(n_base):
        color_list.append(base(i / n_base))
    
    return LinearSegmentedColormap.from_list('white_' + base_cmap, color_list, N=n_colors)


def create_correlation_combined_plot(
    roi_correlations,
    roi_p_values,
    output_dir,
    score_column_name,
    threshold=0.0,
    roi_number=DEFAULT_TOTAL_ROIS,
    cmap='RdBu_r',
    fdr_alpha=0.05
):
    """按sig_brain_region风格绘制皮层+皮层下组合图（仅显示FDR校正后显著的脑区）
    
    使用neurocat绘制表面图，与sig_brain_region.py保持一致的样式
    非显著脑区会填充为白色，脑区边界为灰色
    """
    if roi_correlations is None or len(roi_correlations) == 0:
        print("⚠️ 缺少ROI相关性数据，无法绘制组合图")
        return None

    if roi_p_values is None or len(roi_p_values) == 0:
        print("⚠️ 缺少ROI p值数据，无法进行FDR校正")
        return None

    total_rois = len(roi_correlations)
    if total_rois < SUBCORTICAL_ROI_COUNT:
        print("⚠️ ROI数量不足，无法绘制皮层下图像")
        return None

    # 进行FDR校正
    valid_mask = ~np.isnan(roi_p_values)
    valid_p_values = roi_p_values[valid_mask]
    
    if len(valid_p_values) == 0:
        print("⚠️ 没有有效的p值数据，无法进行FDR校正")
        return None
    
    # FDR校正（Benjamini-Hochberg方法）
    rejected, p_corrected, _, _ = multipletests(
        valid_p_values, 
        alpha=fdr_alpha, 
        method='fdr_bh'
    )
    
    # 创建完整的拒绝数组（保持原始索引）
    roi_rejected = np.zeros(len(roi_p_values), dtype=bool)
    roi_rejected[valid_mask] = rejected
    
    # 统计FDR校正后的显著脑区数量
    n_fdr_significant = np.sum(roi_rejected)
    print(f"📊 FDR校正统计: {n_fdr_significant}/{len(valid_p_values)} 个脑区在FDR校正后显著 (α={fdr_alpha})")
    
    # 如果还有相关性阈值要求，进一步过滤
    if threshold > 0:
        threshold_mask = np.abs(roi_correlations) >= threshold
        roi_rejected = roi_rejected & threshold_mask
    
    # 统计最终要显示的脑区数量
    n_final = np.sum(roi_rejected)
    print(f"📊 最终显示脑区数量: {n_final} (FDR校正后显著且|r|≥{threshold})")
    
    if n_final == 0:
        print("⚠️ FDR校正后没有显著的脑区，跳过绘图")
        return None

    # 计算颜色范围（基于显著脑区的值）
    significant_values = roi_correlations[roi_rejected]
    color_range = determine_symmetric_color_range(significant_values)
    if np.isclose(color_range[0], color_range[1]):
        delta = max(abs(color_range[0]) * 0.1, 0.1)
        color_range = [color_range[0] - delta, color_range[1] + delta]

    # 为非显著脑区创建一个特殊的白色值（比colormap范围更小）
    white_value = color_range[0] - (color_range[1] - color_range[0]) * 0.1
    
    # 创建填充后的值数组：显著脑区保留原值，非显著脑区填充白色值
    filled_values = np.full(total_rois, white_value, dtype=float)
    for i in range(total_rois):
        if roi_rejected[i] and not np.isnan(roi_correlations[i]):
            filled_values[i] = roi_correlations[i]

    cortical_values = filled_values[SUBCORTICAL_ROI_COUNT:total_rois]
    subcortical_values = filled_values[:SUBCORTICAL_ROI_COUNT]

    cortical_roi_number = total_rois - SUBCORTICAL_ROI_COUNT
    if cortical_roi_number <= 0:
        print("⚠️ 皮层ROI数量无效，无法绘制表面图")
        return None

    # 设置matplotlib字体
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # 创建扩展的colormap，在最小值处显示白色
    plot_color_range = [white_value, color_range[1]]
    plot_cmap = create_white_extended_colormap(base_cmap=cmap, white_fraction=0.05)

    # 使用neurocat绘制表面图
    temp_dir = tempfile.gettempdir()
    temp_surface = os.path.join(temp_dir, f'temp_surface_{os.getpid()}.png')
    
    ds(
        layer1=cortical_values,  # 400个皮层ROI
        colorbar=plot_cmap,
        color_range=plot_color_range,
        fig_name=temp_surface.replace('.png', ''),
        layout='row',
        outline='gray',  # 使用灰色轮廓
        tp='fslr',
        mesh='veryinflated',
        save_png_only=True,
        display=False,
        size=(6000, 1000)
    )

    # 绘制皮层下图（nilearn）
    if not os.path.exists(COMBINED_ATLAS_PATH):
        raise FileNotFoundError(f"未找到组合atlas: {COMBINED_ATLAS_PATH}")

    atlas_img = nib.load(COMBINED_ATLAS_PATH)
    atlas_data = atlas_img.get_fdata().astype(int)
    roi_vol = np.full(atlas_data.shape, np.nan, dtype=float)
    max_sub = min(SUBCORTICAL_ROI_COUNT, len(subcortical_values))
    for lbl in range(1, max_sub + 1):
        roi_vol[atlas_data == lbl] = subcortical_values[lbl - 1]

    roi_img = nib.Nifti1Image(roi_vol, atlas_img.affine, atlas_img.header)
    bg_img = datasets.load_mni152_template(resolution=1)
    bg_data = bg_img.get_fdata()
    bg_img_final = nib.Nifti1Image(bg_data, bg_img.affine, bg_img.header)

    fig_nilearn = plt.figure(figsize=(40, 8))
    nilearn_disp = plotting.plot_stat_map(
        roi_img,
        bg_img=bg_img_final,
        threshold=0.0,
        cmap=plot_cmap,
        vmax=plot_color_range[1],
        vmin=plot_color_range[0],
        display_mode='z',
        cut_coords=[-25, -20, -10, 0, 10, 20, 22],
        draw_cross=False,
        annotate=False,
        colorbar=False,
        figure=fig_nilearn,
        black_bg=False,
    )

    # 保存nilearn图像到临时文件
    tmp_nilearn = os.path.join(temp_dir, f'temp_nilearn_{os.getpid()}.png')
    fig_nilearn.savefig(tmp_nilearn, bbox_inches='tight', dpi=300, pad_inches=0)
    nilearn_disp.close()
    plt.close(fig_nilearn)

    # 读取图像并修剪白边
    img_surface = trim_white_edges(mpimg.imread(temp_surface))
    img_nilearn = trim_white_edges(mpimg.imread(tmp_nilearn))

    # 合并图像：表面图在上，皮层下图在下，colorbar在右侧
    fig_combined = plt.figure(figsize=(70, 25))
    gs = gridspec.GridSpec(2, 2, figure=fig_combined,
                           width_ratios=[97, 3],
                           hspace=0.1, wspace=0.05)

    ax_surface = fig_combined.add_subplot(gs[0, 0])
    ax_surface.imshow(img_surface)
    ax_surface.axis('off')

    ax_subcortical = fig_combined.add_subplot(gs[1, 0])
    ax_subcortical.imshow(img_nilearn)
    ax_subcortical.axis('off')

    # 添加colorbar（使用原始的color_range，不包含white_value）
    ax_cbar = fig_combined.add_subplot(gs[:, 1])
    vmin, vmax = color_range
    if vmin >= vmax:
        vmax = vmin + 1e-6
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = ColorbarBase(ax_cbar, cmap=cmap, norm=norm, orientation='vertical')
    
    # 设置刻度（只显示实际数据范围的刻度）
    if vmin < 0 < vmax:
        ticks = [vmin, 0, vmax]
    else:
        ticks = np.linspace(vmin, vmax, 5)
    ax_cbar.set_yticks(ticks)
    cbar.set_label('SFC-分数相关系数 (r)', rotation=270, labelpad=30, fontsize=40, fontname='Arial')
    ax_cbar.tick_params(labelsize=42, direction='in', length=8, width=2)
    
    # 设置刻度标签字体
    for label in ax_cbar.get_yticklabels():
        label.set_fontname('Arial')

    plt.tight_layout()

    # 保存为PNG和PDF
    combined_path_png = os.path.join(
        output_dir,
        f'sfc_{score_column_name}_surface_subcortical_combined.png'
    )
    combined_path_pdf = os.path.join(
        output_dir,
        f'sfc_{score_column_name}_surface_subcortical_combined.pdf'
    )
    
    fig_combined.savefig(combined_path_png, dpi=300, bbox_inches='tight')
    fig_combined.savefig(combined_path_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.close(fig_combined)

    # 清理临时文件
    try:
        os.remove(temp_surface)
        os.remove(tmp_nilearn)
    except:
        pass

    print(f"✅ 表面+皮层下合成图已保存至：{combined_path_png}")
    print(f"✅ 表面+皮层下合成图(PDF)已保存至：{combined_path_pdf}")
    return combined_path_png


def create_subject_network_summary(
    all_subjects_data,
    network_mapping,
    clinical_data,
    subject_id_col,
    score_columns
):
    """根据受试者数据计算各网络平均SFC，并附加多分数列"""
    if not all_subjects_data:
        return None

    score_lookup = {}
    if clinical_data is not None and subject_id_col in clinical_data.columns:
        for _, row in clinical_data.iterrows():
            normalized_id = normalize_subject_identifier(row[subject_id_col])
            if not normalized_id:
                continue
            score_lookup[normalized_id] = {
                col: row[col] if col in clinical_data.columns else np.nan
                for col in score_columns
            }

    records = []
    for subject_df in all_subjects_data:
        subject_id = str(subject_df['subject_id'].iloc[0])
        normalized_id = normalize_subject_identifier(subject_id)
        network_values = {network: [] for network in NETWORKS}

        for _, roi_row in subject_df.iterrows():
            roi_index = int(roi_row['ROI_Index'])
            network = network_mapping.get(roi_index)
            if not network:
                continue
            sfc_val = roi_row['SFC_Value']
            if pd.isna(sfc_val):
                continue
            network_values[network].append(sfc_val)

        record = {'subject_id': subject_id}
        for network in NETWORKS:
            values = network_values[network]
            record[network] = float(np.mean(values)) if values else np.nan

        for col in score_columns:
            subject_scores = score_lookup.get(normalized_id, {})
            record[col] = subject_scores.get(col, np.nan) if subject_scores else np.nan

        records.append(record)

    if not records:
        return None

    return pd.DataFrame(records)


def load_sfc_data(sfc_file_path,roi_number):
    """加载单个受试者的SFC数据"""
    try:
        # 使用文件读取方式读取SFC数据
        sfc_values = []
        roi_indices = []
        
        with open(sfc_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines[3:]:  # 跳过前3行头部信息
                if line.strip() and '\t' in line:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        try:
                            roi_index = int(parts[0])
                            sfc_val = float(parts[1])
                            roi_indices.append(roi_index)
                            sfc_values.append(sfc_val)
                        except ValueError:
                            # 处理NaN值
                            if parts[1].strip().lower() == 'nan':
                                roi_indices.append(len(roi_indices) + 1)
                                sfc_values.append(np.nan)
                            else:
                                print(f"Warning: Could not parse SFC value: {parts[1]}")
                                continue
        
        # 创建DataFrame
        df = pd.DataFrame({
            'ROI_Index': roi_indices,
            'SFC_Value': sfc_values
        })
        
        # 清理数据
        df = df.dropna(subset=['SFC_Value'])
        df['ROI_Index'] = pd.to_numeric(df['ROI_Index'], errors='coerce')
        df['SFC_Value'] = pd.to_numeric(df['SFC_Value'], errors='coerce')
        
        # 移除无效的ROI索引
        df = df[df['ROI_Index'].between(1, roi_number)]
        # import pdb;pdb.set_trace()
        print(f"✅ 成功加载 {len(df)} 个有效ROI的SFC数据")
        return df
        
    except Exception as e:
        print(f"加载 {sfc_file_path} 时出错：{e}")
        return None

def load_all_subjects_sfc_score(
    sfc_folder_path,
    clinical_file_path,
    score_column=None,
    roi_number=DEFAULT_TOTAL_ROIS,
    participant_file_path=PARTICIPANT_INFO_PATH,
    clinical_data=None,
    subject_id_col=None,
):
    """加载所有受试者的SFC数据和指定分数"""
    all_subjects_data = []
    
    # 获取所有SFC文件
    sfc_files = list(Path(sfc_folder_path).glob("*_sfc_values_pca.txt"))
    
    print(f"找到 {len(sfc_files)} 个SFC文件")
    
    # 加载临床数据
    if clinical_data is None:
        try:
            clinical_data = pd.read_csv(clinical_file_path)
            print(f"成功加载临床数据，包含 {len(clinical_data)} 行")
            print("临床数据列名:", clinical_data.columns.tolist())
        except Exception as e:
            print(f"加载临床数据失败: {e}")
            return None
    else:
        print(f"使用预加载的临床数据，包含 {len(clinical_data)} 行")
    
    # 查找分数列和受试者ID列
    score_col = None
    subject_id_col = None
    
    # 如果用户指定了分数列，直接使用
    if score_column and score_column in clinical_data.columns:
        score_col = score_column
        print(f"✅ 使用用户指定的分数列: {score_col}")
    else:
        # 自动查找分数列
        if score_column:
            print(f"⚠️ 用户指定的分数列 '{score_column}' 不存在")
            print("尝试自动查找可用的分数列...")
        
        # 查找可能的分数列
        score_candidates = []
        for col in clinical_data.columns:
            if any(keyword in col.lower() for keyword in ['score', 'subscore', 'rating', 'wab', 'aq']):
                score_candidates.append(col)
        
        if score_candidates:
            print(f"找到以下可能的分数列: {score_candidates}")
            # 优先选择包含'subscore'的列
            for col in score_candidates:
                if 'subscore' in col.lower():
                    score_col = col
                    break
            # 如果没有找到subscore，选择第一个
            if score_col is None:
                score_col = score_candidates[0]
            print(f"✅ 自动选择分数列: {score_col}")
        else:
            print("❌ 未找到任何可能的分数列")
            print("可用列:", clinical_data.columns.tolist())
            return None
    
    # 查找受试者ID列
    if subject_id_col is None:
        for col in clinical_data.columns:
            if 'subject_id' in col.lower():
                subject_id_col = col
                break
    
    if subject_id_col is None:
        print("❌ 未找到受试者ID列")
        print("可用列:", clinical_data.columns.tolist())
        return None
    
    print(f"使用列: {subject_id_col} 作为受试者ID, {score_col} 作为分数列")
    
    # 检查分数列的数据
    score_data = clinical_data[score_col].dropna()
    print(f"✅ 分数数据: {len(score_data)} 个有效值")
    print(f"   分数范围: {score_data.min():.1f} - {score_data.max():.1f}")
    print(f"   分数均值: {score_data.mean():.1f}")
    
    fallback_scores = (
        load_fallback_wab_scores(participant_file_path)
        if score_column == "wab_r_aq"
        else {}
    )

    for sfc_file in sfc_files:
        # subject_id = sfc_file.stem.replace('_sfc_values_pca', '')
        import re
        subject_id = re.findall(r'\d+', sfc_file.stem)[0]
        
        # 从临床数据中查找对应的分数
        # print(subject_id)

        subject_score = None
        for _, row in clinical_data.iterrows():
            clinical_subject_id = str(row[subject_id_col]).strip()
            # 处理subject_id格式（移除可能的sub-M前缀）
            if clinical_subject_id.startswith('sub-M'):
                clinical_subject_id = clinical_subject_id[5:]
            elif clinical_subject_id.startswith('M'):
                clinical_subject_id = clinical_subject_id[1:]
            
            if clinical_subject_id == subject_id.replace('sub-M', ''):
                subject_score = row[score_col]
                break
        
        if (subject_score is None or pd.isna(subject_score)) and fallback_scores:
            digits_only = ''.join(filter(str.isdigit, subject_id))
            if digits_only in fallback_scores:
                subject_score = fallback_scores[digits_only]
                print(f"ℹ️ 使用备选WAB AQ分数 (participants文件) -> 受试者{subject_id}: {subject_score}")

        if subject_score is None or pd.isna(subject_score):
            print(f"警告: 未找到受试者 {subject_id} 的分数 (临床与备选文件均无记录)")
            continue
        
        # 加载SFC数据
        sfc_data = load_sfc_data(sfc_file,roi_number)
        # import pdb;pdb.set_trace()
        if sfc_data is None:
            continue
        
        # 添加受试者ID和分数
        sfc_data['subject_id'] = subject_id
        sfc_data['subject_score'] = subject_score
        
        all_subjects_data.append(sfc_data)
    
    print(f"✅ 成功加载 {len(all_subjects_data)} 个受试者的SFC和分数数据")
    return all_subjects_data

def compute_roi_correlations(all_subjects_data, network_mapping, score_column_name,roi_number):
    """计算每个ROI的SFC和指定分数的相关性"""
    # 初始化结果数组
    roi_correlations = np.full(roi_number, np.nan)
    roi_p_values = np.full(roi_number, np.nan)
    network_correlations = {network: [] for network in NETWORKS}
    
    print(f"🔍 计算每个ROI的SFC-{score_column_name}相关性...")
    
    for roi_idx in range(1, roi_number+1):  # ROI索引从1开始
        # 收集该ROI的所有受试者数据
        roi_sfc_values = []
        roi_scores = []
        
        for subject_data in all_subjects_data:
            roi_data = subject_data[subject_data['ROI_Index'] == roi_idx]
            if not roi_data.empty:
                sfc_val = roi_data.iloc[0]['SFC_Value']
                score_val = roi_data.iloc[0]['subject_score']
                
                if not pd.isna(sfc_val) and not pd.isna(score_val):
                    roi_sfc_values.append(sfc_val)
                    roi_scores.append(score_val)
        
        # 计算相关性
        if len(roi_sfc_values) >= 3:  # 至少需要3个数据点
            try:
                # 使用Spearman相关系数
                corr, p_val = stats.spearmanr(roi_sfc_values, roi_scores)
                roi_correlations[roi_idx-1] = corr  # 转换为0-399索引
                roi_p_values[roi_idx-1] = p_val
                
                # 按网络分类存储相关性
                if roi_idx in network_mapping:
                    network = network_mapping[roi_idx]
                    network_correlations[network].append(corr)
                
                if roi_idx % 50 == 0:  # 每50个ROI打印一次进度
                    print(f"  ROI {roi_idx}: r={corr:.3f}, p={p_val:.3f}, n={len(roi_sfc_values)}")
                    
            except Exception as e:
                print(f"ROI {roi_idx} 相关性计算失败: {e}")
        else:
            if roi_idx % 50 == 0:
                print(f"  ROI {roi_idx}: 数据不足 (n={len(roi_sfc_values)})")
    
    print(f"✅ 相关性计算完成，有效ROI数量: {np.sum(~np.isnan(roi_correlations))}")
    
    return roi_correlations, roi_p_values, network_correlations

def visualize_subcortical_correlation_plot(*args, **kwargs):
    raise NotImplementedError("此函数已弃用，请使用 create_correlation_combined_plot 替代")

def plot_network_correlation_boxplots(network_correlations, output_dir, score_column_name):
    """绘制网络相关性箱型图"""
    plt.figure(figsize=(15, 8))
    
    # 准备数据
    plot_data = []
    for network in NETWORKS:
        if network_correlations[network]:  # 只绘制有数据的网络
            values = network_correlations[network]
            plot_data.extend([(network, value) for value in values])
    
    if not plot_data:
        print("⚠️ 没有可用的网络相关性数据")
        return
    
    plot_df = pd.DataFrame(plot_data, columns=['Network', 'Correlation'])
    
    # 绘制箱型图
    sns.boxplot(data=plot_df, x='Network', y='Correlation', palette='Set3')
    plt.title(f'7个网络的SFC-{score_column_name}相关性分布', fontsize=16, fontweight='bold')
    plt.xlabel('网络', fontsize=14)
    plt.ylabel(f'SFC-{score_column_name}相关系数', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    for i, network in enumerate(NETWORKS):
        if network_correlations[network]:
            values = network_correlations[network]
            mean_val = np.mean(values)
            std_val = np.std(values)
            plt.text(i, plt.ylim()[1] * 0.95, 
                    f'n={len(values)}\nμ={mean_val:.3f}\nσ={std_val:.3f}',
                    ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'network_sfc_{score_column_name}_correlation_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 网络相关性箱型图已保存至：{output_path}")

def compute_subject_average_sfc(all_subjects_data):
    """计算每个受试者的平均SFC值"""
    subject_averages = []
    
    for subject_data in all_subjects_data:
        subject_id = subject_data['subject_id'].iloc[0]
        subject_score = subject_data['subject_score'].iloc[0]
        
        # 计算该受试者的平均SFC（忽略NaN值）
        valid_sfc = subject_data['SFC_Value'].dropna()
        if len(valid_sfc) > 0:
            avg_sfc = valid_sfc.mean()
            subject_averages.append({
                'subject_id': subject_id,
                'average_sfc': avg_sfc,
                'score': subject_score,
                'n_valid_rois': len(valid_sfc)
            })
    
    return subject_averages


def save_all_subjects_sfc_data(all_subjects_data, output_dir, score_column_name, roi_number):
    """保存所有受试者的ROI级别SFC数据到CSV（用于复现）"""
    if not all_subjects_data:
        print("⚠️ 没有受试者数据可保存")
        return None
    
    # 创建一个宽格式的DataFrame，每行是一个受试者，列是ROI索引
    rows = []
    for subject_data in all_subjects_data:
        subject_id = subject_data['subject_id'].iloc[0]
        subject_score = subject_data['subject_score'].iloc[0]
        
        row = {
            'subject_id': subject_id,
            f'{score_column_name}': subject_score
        }
        
        # 添加每个ROI的SFC值
        for _, roi_row in subject_data.iterrows():
            roi_idx = int(roi_row['ROI_Index'])
            sfc_val = roi_row['SFC_Value']
            row[f'ROI_{roi_idx}'] = sfc_val
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 确保ROI列按顺序排列
    roi_cols = [f'ROI_{i}' for i in range(1, roi_number + 1)]
    existing_roi_cols = [col for col in roi_cols if col in df.columns]
    ordered_cols = ['subject_id', score_column_name] + existing_roi_cols
    df = df[[col for col in ordered_cols if col in df.columns]]
    
    output_path = os.path.join(output_dir, f'all_subjects_roi_sfc_{score_column_name}.csv')
    df.to_csv(output_path, index=False)
    print(f"✅ 所有受试者ROI级别SFC数据已保存至：{output_path}")
    print(f"   共 {len(rows)} 个受试者，{len(existing_roi_cols)} 个ROI")
    return output_path


def save_network_correlation_summary(network_correlations, output_dir, score_column_name):
    """保存网络级别相关性汇总到CSV（用于复现）"""
    rows = []
    for network in NETWORKS:
        if network_correlations[network]:
            values = network_correlations[network]
            rows.append({
                'Network': network,
                'n_ROIs': len(values),
                'Mean_Correlation': np.mean(values),
                'Std_Correlation': np.std(values),
                'Min_Correlation': np.min(values),
                'Max_Correlation': np.max(values),
                'Median_Correlation': np.median(values)
            })
    
    if not rows:
        print("⚠️ 没有网络相关性数据可保存")
        return None
    
    df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, f'network_sfc_{score_column_name}_correlation_summary.csv')
    df.to_csv(output_path, index=False)
    print(f"✅ 网络级别相关性汇总已保存至：{output_path}")
    return output_path


def save_significant_rois_summary(roi_correlations, roi_p_values, network_mapping, output_dir, score_column_name, p_threshold=0.05):
    """保存FDR校正后显著脑区的详细信息到CSV（用于复现）"""
    # FDR校正
    valid_mask = ~np.isnan(roi_p_values)
    valid_p_values = roi_p_values[valid_mask]
    
    if len(valid_p_values) == 0:
        print("⚠️ 没有有效的p值数据")
        return None
    
    rejected, p_corrected, _, _ = multipletests(
        valid_p_values, 
        alpha=p_threshold, 
        method='fdr_bh'
    )
    
    # 创建完整的校正后p值数组
    roi_p_corrected = np.full_like(roi_p_values, np.nan)
    roi_p_corrected[valid_mask] = p_corrected
    
    # 创建拒绝假设的完整数组
    roi_rejected = np.zeros(len(roi_p_values), dtype=bool)
    roi_rejected[valid_mask] = rejected
    
    # 收集显著脑区信息
    rows = []
    for roi_idx in range(len(roi_correlations)):
        if roi_rejected[roi_idx]:
            roi_number = roi_idx + 1  # 1-based编号
            rows.append({
                'ROI_Index': roi_number,
                'Correlation': roi_correlations[roi_idx],
                'P_Value_Uncorrected': roi_p_values[roi_idx],
                'P_Value_FDR_Corrected': roi_p_corrected[roi_idx],
                'Network': network_mapping.get(roi_number, 'Unknown'),
                'Direction': '正相关' if roi_correlations[roi_idx] > 0 else '负相关',
                'Significance': '***' if roi_p_corrected[roi_idx] < 0.001 else '**' if roi_p_corrected[roi_idx] < 0.01 else '*'
            })
    
    if not rows:
        print("⚠️ FDR校正后没有显著脑区")
        return None
    
    # 按相关性绝对值排序
    df = pd.DataFrame(rows)
    df['Abs_Correlation'] = df['Correlation'].abs()
    df = df.sort_values('Abs_Correlation', ascending=False).drop('Abs_Correlation', axis=1)
    df = df.reset_index(drop=True)
    df.index = df.index + 1  # 排名从1开始
    df.index.name = 'Rank'
    
    output_path = os.path.join(output_dir, f'significant_rois_sfc_{score_column_name}_fdr.csv')
    df.to_csv(output_path)
    print(f"✅ FDR校正后显著脑区详情已保存至：{output_path}")
    print(f"   共 {len(rows)} 个显著脑区")
    return output_path

def plot_subject_sfc_score_scatter(subject_averages, output_dir, score_column_name):
    """绘制每个受试者平均SFC与评分的散点图"""
    if not subject_averages:
        print("⚠️ 没有可用的受试者平均SFC数据")
        return
    
    # 提取数据
    avg_sfc_values = [s['average_sfc'] for s in subject_averages]
    scores = [s['score'] for s in subject_averages]
    n_valid_rois = [s['n_valid_rois'] for s in subject_averages]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 散点图：平均SFC vs 评分
    scatter = ax1.scatter(avg_sfc_values, scores, c=n_valid_rois, 
                         cmap='viridis', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    # 计算相关性
    corr_coef, p_value = stats.spearmanr(avg_sfc_values, scores)
    
    # 添加趋势线
    z = np.polyfit(avg_sfc_values, scores, 1)
    p = np.poly1d(z)
    ax1.plot(avg_sfc_values, p(avg_sfc_values), "r--", alpha=0.8, linewidth=2)
    
    ax1.set_xlabel(f'平均SFC值', fontsize=12)
    ax1.set_ylabel(f'{score_column_name}分数', fontsize=12)
    ax1.set_title(f'受试者平均SFC与{score_column_name}相关性\nr={corr_coef:.3f}, p={p_value:.3f}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('有效ROI数量', fontsize=10)
    
    # 2. 有效ROI数量分布
    ax2.hist(n_valid_rois, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('有效ROI数量', fontsize=12)
    ax2.set_ylabel('受试者数量', fontsize=12)
    ax2.set_title(f'有效ROI数量分布\n(平均: {np.mean(n_valid_rois):.1f})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=np.mean(n_valid_rois), color='red', linestyle='--', alpha=0.7, 
                label=f'平均值: {np.mean(n_valid_rois):.1f}')
    ax2.legend()
    
    plt.suptitle(f'受试者平均SFC与{score_column_name}关系分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'subject_avg_sfc_{score_column_name}_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 受试者平均SFC散点图已保存至：{output_path}")
    
    # 打印统计信息
    print(f"\n📊 受试者平均SFC统计:")
    print(f"  受试者数量: {len(subject_averages)}")
    print(f"  平均SFC范围: {np.min(avg_sfc_values):.4f} - {np.max(avg_sfc_values):.4f}")
    print(f"  平均SFC均值: {np.mean(avg_sfc_values):.4f}")
    print(f"  平均SFC标准差: {np.std(avg_sfc_values):.4f}")
    print(f"  有效ROI数量范围: {np.min(n_valid_rois)} - {np.max(n_valid_rois)}")
    print(f"  与{score_column_name}相关性: r={corr_coef:.4f}, p={p_value:.4f}")
    
    return corr_coef, p_value

def print_high_correlation_rois(roi_correlations, roi_p_values, network_mapping, score_column_name, correlation_threshold=0.3, p_threshold=0.05):
    """打印高相关性的脑区编号、相关性和网络信息（使用FDR校正）"""
    print(f"\n🔥 高相关性脑区分析 (|r|≥{correlation_threshold} 且 FDR校正后p<{p_threshold}):")
    print("=" * 80)
    
    # 首先进行FDR校正（Benjamini-Hochberg方法）
    valid_mask = ~np.isnan(roi_p_values)
    valid_p_values = roi_p_values[valid_mask]
    
    if len(valid_p_values) == 0:
        print("❌ 没有有效的p值数据")
        return
    
    # 对所有有效的p值进行FDR校正
    rejected, p_corrected, _, _ = multipletests(
        valid_p_values, 
        alpha=p_threshold, 
        method='fdr_bh'  # Benjamini-Hochberg FDR校正
    )
    
    # 创建完整的校正后p值数组（保持原始索引）
    roi_p_corrected = np.full_like(roi_p_values, np.nan)
    roi_p_corrected[valid_mask] = p_corrected
    
    # 创建拒绝假设的完整数组
    roi_rejected = np.zeros(len(roi_p_values), dtype=bool)
    roi_rejected[valid_mask] = rejected
    
    # 统计校正前后的显著脑区数量
    uncorrected_significant = np.sum((roi_p_values < p_threshold) & valid_mask)
    corrected_significant = np.sum(roi_rejected)
    
    print(f"📊 FDR校正统计:")
    print(f"  总有效脑区数: {len(valid_p_values)}")
    print(f"  未校正显著脑区数 (p<{p_threshold}): {uncorrected_significant}")
    print(f"  FDR校正后显著脑区数 (FDR p<{p_threshold}): {corrected_significant}")
    
    # 找到满足条件的脑区（使用校正后的p值）
    high_corr_mask = (
        (np.abs(roi_correlations) >= correlation_threshold) & 
        roi_rejected & 
        (~np.isnan(roi_correlations))
    )
    high_corr_indices = np.where(high_corr_mask)[0]
    
    if len(high_corr_indices) == 0:
        print(f"❌ 没有找到满足条件的高相关性脑区 (|r|≥{correlation_threshold} 且 FDR校正后p<{p_threshold})")
        return
    
    # 按相关性绝对值排序
    high_corr_values = roi_correlations[high_corr_indices]
    sorted_order = np.argsort(np.abs(high_corr_values))[::-1]  # 降序排列
    
    print(f"\n📊 找到 {len(high_corr_indices)} 个高相关性脑区 (FDR校正后显著):")
    print(f"{'排名':<4} {'ROI编号':<8} {'相关性':<10} {'原始P值':<12} {'FDR校正P值':<14} {'网络':<12} {'方向':<8}")
    print("-" * 90)
    
    for rank, idx in enumerate(sorted_order, 1):
        roi_idx = high_corr_indices[idx] + 1  # 切换回1-based编号
        corr_val = high_corr_values[idx]
        p_val_uncorrected = roi_p_values[roi_idx - 1]
        p_val_corrected = roi_p_corrected[roi_idx - 1]
        network = network_mapping.get(roi_idx, 'Unknown')
        
        # 判断相关性方向
        direction = "正相关" if corr_val > 0 else "负相关"
        
        print(f"{rank:<4} {roi_idx:<8} {corr_val:<10.4f} {p_val_uncorrected:<12.6f} {p_val_corrected:<14.6f} {network:<12} {direction:<8}")
    
    # 按网络统计
    print(f"\n📈 按网络统计高相关性脑区 (FDR校正后):")
    network_counts = {}
    for idx in high_corr_indices:
        roi_idx = idx + 1
        network = network_mapping.get(roi_idx, 'Unknown')
        if network not in network_counts:
            network_counts[network] = {'count': 0, 'correlations': []}
        network_counts[network]['count'] += 1
        network_counts[network]['correlations'].append(roi_correlations[idx])
    
    for network, data in network_counts.items():
        avg_corr = np.mean(data['correlations'])
        print(f"  {network}: {data['count']} 个脑区, 平均相关性: {avg_corr:.4f}")
    
    # 打印前10个最高相关性的脑区详细信息
    print(f"\n🏆 前10个最高相关性脑区详细信息 (FDR校正后):")
    print("-" * 90)
    for rank, idx in enumerate(sorted_order[:10], 1):
        roi_idx = high_corr_indices[idx] + 1
        corr_val = high_corr_values[idx]
        p_val_uncorrected = roi_p_values[roi_idx - 1]
        p_val_corrected = roi_p_corrected[roi_idx - 1]
        network = network_mapping.get(roi_idx, 'Unknown')
        direction = "正相关" if corr_val > 0 else "负相关"
        
        print(f"第{rank}名: ROI {roi_idx}")
        print(f"  相关性系数: {corr_val:.6f} ({direction})")
        print(f"  原始P值: {p_val_uncorrected:.6f}")
        print(f"  FDR校正P值: {p_val_corrected:.6f}")
        print(f"  所属网络: {network}")
        print(f"  显著性: {'***' if p_val_corrected < 0.001 else '**' if p_val_corrected < 0.01 else '*' if p_val_corrected < 0.05 else 'ns'}")
        print()

def create_correlation_summary_plot(roi_correlations, roi_p_values, output_dir, score_column_name, total_rois=DEFAULT_TOTAL_ROIS):
    """创建相关性总结图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 相关性分布直方图
    valid_corr = roi_correlations[~np.isnan(roi_correlations)]
    ax1.hist(valid_corr, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('SFC-WAB相关系数')
    ax1.set_ylabel('脑区数量')
    ax1.set_title(f'SFC-WAB相关性分布 (有效脑区: {len(valid_corr)}/{total_rois})')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='无相关性')
    ax1.legend()
    
    # 2. 相关性排序图
    sorted_indices = np.argsort(roi_correlations)
    sorted_corr = roi_correlations[sorted_indices]
    ax2.plot(range(len(sorted_corr)), sorted_corr, 'o-', markersize=2, linewidth=1)
    ax2.set_xlabel('脑区排序')
    ax2.set_ylabel('SFC-WAB相关系数')
    ax2.set_title('SFC-WAB相关性排序')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 3. P值分布直方图
    valid_p = roi_p_values[~np.isnan(roi_p_values)]
    ax3.hist(valid_p, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax3.set_xlabel('P值')
    ax3.set_ylabel('脑区数量')
    ax3.set_title(f'P值分布 (有效脑区: {len(valid_p)}/{total_rois})')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='显著性阈值 (p=0.05)')
    ax3.legend()
    
    # 4. 显著性脑区统计
    significant_rois = np.sum(roi_p_values < 0.05)
    valid_p_count = np.sum(~np.isnan(roi_p_values))
    
    labels = ['显著相关\n(p<0.05)', '非显著相关\n(p≥0.05)']
    sizes = [significant_rois, valid_p_count - significant_rois]
    colors = ['lightcoral', 'lightblue']
    
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title(f'显著性脑区统计\n(总脑区: {valid_p_count})')
    
    plt.suptitle(f'SFC-{score_column_name}相关性分析总结', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'sfc_{score_column_name}_correlation_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 相关性总结图已保存至：{output_path}")

def main():
    """主函数"""
    print("🚀 开始分析ROI级别的SFC-分数相关性...")
    roi_number = DEFAULT_TOTAL_ROIS
    
    # 设置路径
    # sfc_folder = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_100\sfc_txt_files"
    # sfc_folder = rf"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_{roi_number}\sfc_txt_files\spearman"
    sfc_folder = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\sfc_txt_files"
    # sfc_folder =
    cortical_roi_number = roi_number - SUBCORTICAL_ROI_COUNT
    network_csv = rf"D:\download\Schaefer2018_{cortical_roi_number}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
    # clinical_file = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\merged_file_sfc.csv"
    clinical_file = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\matched_subjects_data.csv"
    output_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\images"
    # D:\sc - fc_coupling\scfc - coupling - main\scfc - coupling - main\batch_results_400\sfc_txt_files\spearman\sub - M2216_sfc_values.txt
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    score_columns = [
        "repetition_subscore",
        "comprehension_subscore",
        "naming_subscore",
        "wab_r_aq",
    ]
    print(f"\n🎯 将依次分析以下分数列: {', '.join(score_columns)}")

    try:
        clinical_data = pd.read_csv(clinical_file)
        print(f"✅ 已预加载临床数据: {len(clinical_data)} 条记录")
    except Exception as exc:
        print(f"❌ 预加载临床数据失败: {exc}")
        return

    subject_id_col = None
    for col in clinical_data.columns:
        if 'subject_id' in col.lower():
            subject_id_col = col
            break

    if subject_id_col is None:
        print("❌ 无法在临床数据中找到受试者ID列，程序退出")
        return

    missing_score_cols = [col for col in score_columns if col not in clinical_data.columns]
    if missing_score_cols:
        print(f"⚠️ 临床数据缺少以下分数列，将以NaN填充: {', '.join(missing_score_cols)}")
    
    # 1. 加载网络映射
    print("\n🗺️ 加载网络映射...")
    network_mapping = load_network_mapping(network_csv)
    
    if network_mapping is None:
        print("❌ 网络映射加载失败，程序退出")
        return
    
    for score_column in score_columns:
        print("\n" + "=" * 80)
        print(f"开始分析分数列: {score_column}")
        print("=" * 80)
        
        # 2. 加载所有受试者的SFC和分数数据
        print("📊 加载SFC和分数数据...")
        all_subjects_data = load_all_subjects_sfc_score(
            sfc_folder,
            clinical_file,
            score_column,
            roi_number,
            clinical_data=clinical_data,
            subject_id_col=subject_id_col
        )
        
        if all_subjects_data is None or len(all_subjects_data) == 0:
            print(f"❌ 未能加载任何SFC和分数数据，跳过 {score_column}")
            continue
        
        # print(f"✅ 成功加载 {len(all_subjects_data)} 个受试者的数据")
        
        # 3. 计算每个ROI的SFC-分数相关性
        print("🔍 计算ROI级别的相关性...")
        roi_correlations, roi_p_values, network_correlations = compute_roi_correlations(
            all_subjects_data, network_mapping, score_column,roi_number
        )
        
        # 4. 绘制表面与皮层下相关性图（sig_brain_region合成样式，仅显示FDR校正后显著的脑区）
        print("🎨 绘制表面+皮层下相关性图 (sig_brain_region样式, FDR校正后显著)...")
        correlation_threshold = 0.0
        try:
            print(f"📊 使用相关性阈值: |r|≥{correlation_threshold}")
            create_correlation_combined_plot(
                roi_correlations,
                roi_p_values,
                output_dir,
                score_column,
                threshold=correlation_threshold,
                roi_number=roi_number,
                fdr_alpha=0.05
            )
        except Exception as e:
            print(f"⚠️ 表面图绘制失败: {e}")
            print("跳过表面图绘制，继续其他分析...")

        # 5. 保存受试者网络级SFC数据
        print("🧾 生成受试者网络级SFC汇总CSV...")
        subject_network_df = create_subject_network_summary(
            all_subjects_data,
            network_mapping,
            clinical_data,
            subject_id_col,
            score_columns
        )
        if subject_network_df is not None:
            network_csv_path = os.path.join(
                output_dir,
                f'sfc_{score_column}_subject_network_summary.csv'
            )
            subject_network_df.to_csv(network_csv_path, index=False)
            print(f"✅ 受试者网络SFC数据已保存至：{network_csv_path}")
        else:
            print("⚠️ 无可导出的受试者网络SFC数据")
        
        # 7. 打印高相关性脑区信息
        print("🔥 分析高相关性脑区...")
        print_high_correlation_rois(roi_correlations, roi_p_values, network_mapping, score_column, correlation_threshold=0.0, p_threshold=0.05)
        
        # 8. 保存所有受试者的ROI级别SFC数据（用于复现）
        print("💾 保存所有受试者ROI级别SFC数据...")
        save_all_subjects_sfc_data(all_subjects_data, output_dir, score_column, roi_number)
        
        # 9. 保存网络级别相关性汇总
        print("💾 保存网络级别相关性汇总...")
        save_network_correlation_summary(network_correlations, output_dir, score_column)
        
        # 10. 保存FDR校正后显著脑区详情
        print("💾 保存FDR校正后显著脑区详情...")
        save_significant_rois_summary(roi_correlations, roi_p_values, network_mapping, output_dir, score_column, p_threshold=0.05)
        
        # 11. 保存相关性数据（包含FDR校正后的p值）
        print("💾 保存相关性数据...")
        
        # 计算FDR校正后的p值
        valid_mask = ~np.isnan(roi_p_values)
        valid_p_values = roi_p_values[valid_mask]
        roi_p_corrected = np.full_like(roi_p_values, np.nan)
        
        if len(valid_p_values) > 0:
            _, p_corrected, _, _ = multipletests(
                valid_p_values, 
                alpha=0.05, 
                method='fdr_bh'  # Benjamini-Hochberg FDR校正
            )
            roi_p_corrected[valid_mask] = p_corrected
        
        correlation_df = pd.DataFrame({
            'ROI_Index': range(1, roi_number+1),
            'Correlation': roi_correlations,
            'P_Value_Uncorrected': roi_p_values,
            'P_Value_FDR_Corrected': roi_p_corrected
        })
        
        # 添加网络信息
        correlation_df['Network'] = correlation_df['ROI_Index'].map(network_mapping)
        
        output_csv = os.path.join(output_dir, f'roi_sfc_{score_column}_correlations.csv')
        correlation_df.to_csv(output_csv, index=False)
        print(f"✅ 相关性数据已保存至：{output_csv} (包含FDR校正后的p值)")
        
        # 10. 打印统计信息
        print("\n📊 分析结果总结:")
        valid_corr = roi_correlations[~np.isnan(roi_correlations)]
        significant_rois_uncorrected = np.sum(roi_p_values < 0.05)
        
        # 计算FDR校正后的显著脑区数量
        valid_mask = ~np.isnan(roi_p_values)
        valid_p_values = roi_p_values[valid_mask]
        significant_rois_fdr = 0
        if len(valid_p_values) > 0:
            rejected, _, _, _ = multipletests(
                valid_p_values, 
                alpha=0.05, 
                method='fdr_bh'
            )
            significant_rois_fdr = np.sum(rejected)
        
        print(f"  总脑区数量: {roi_number}")
        print(f"  有效脑区数量: {len(valid_corr)}")
        print(f"  显著相关脑区数量 (未校正, p<0.05): {significant_rois_uncorrected}")
        print(f"  显著相关脑区数量 (FDR校正后, p<0.05): {significant_rois_fdr}")
        print(f"  平均相关系数: {np.mean(valid_corr):.4f}")
        print(f"  相关系数标准差: {np.std(valid_corr):.4f}")
        
        print("\n各网络统计:")
        for network in NETWORKS:
            if network_correlations[network]:
                values = network_correlations[network]
                print(f"  {network}: n={len(values)}, μ={np.mean(values):.4f}, σ={np.std(values):.4f}")
        
        print(f"🎉 分数列 {score_column} 的ROI级别SFC相关性分析完成！")

if __name__ == "__main__":
    main()
