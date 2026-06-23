"""
Regional SFC Z-score Analysis - Final Version
Uses pre-loaded healthy controls matrix for efficiency
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import nibabel as nib

# Configuration
SFC_DIR = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\sfc_txt_files"
EXTRACTED_REGIONS_DIR = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\extracted_regions"
CLINICAL_DATA_FILE = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\aphasia_plots\participants_aphasia_new.txt"
BRAIN_TEMPLATE_FILE = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_MNI152NLin6Asym_1mm.nii.gz"
OUTPUT_DIR = "regional_sfc_zscore_results_final"
HEALTHY_MATRIX_FILE = "healthy_controls_sfc_matrix.npy"
ROI_CACHE_FILE = "subject_roi_cache.npy"  # Cache file for ROI indexes
HEALTHY_MEAN_SFC_FILE = "healthy_mean_sfc_template.npy"  # Healthy mean SFC template
# 绘图顺序：Anomic → Broca → Conduction → Global → Non-aphasic
# 注意：临床表里 non-aphasic 在本脚本中以 'None' 存储
APHASIA_TYPES = ['Anomic', 'Broca', 'Conduction', 'Global', 'None']

BRAIN_TEMPLATE = None


def load_brain_template():
    global BRAIN_TEMPLATE
    if BRAIN_TEMPLATE is None:
        print("Loading brain template...")
        img = nib.load(BRAIN_TEMPLATE_FILE)
        BRAIN_TEMPLATE = img.get_fdata().astype(int)
        print(f"  Template shape: {BRAIN_TEMPLATE.shape}")
    return BRAIN_TEMPLATE


def load_sfc_values(sfc_file):
    sfc_values = []
    with open(sfc_file, 'r') as f:
        lines = f.readlines()
        for line in lines[3:]:
            if line.strip() and '\t' in line:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        sfc_values.append(float(parts[1]))
                    except ValueError:
                        sfc_values.append(np.nan)
    return np.array(sfc_values)


def extract_lesion_rois(lesion_mask, brain_template, overlap_threshold=0.9):
    lesion_rois = []
    unique_rois = np.unique(brain_template)
    unique_rois = unique_rois[unique_rois > 0]
    for roi_id in unique_rois:
        roi_mask = (brain_template == roi_id)
        roi_volume = np.sum(roi_mask)
        if roi_volume == 0:
            continue
        overlap = np.sum(lesion_mask & roi_mask)
        if overlap / roi_volume >= overlap_threshold:
            lesion_rois.append(roi_id)
    return lesion_rois


def extract_perilesional_rois(lesion_mask, brain_template, overlap_threshold=0.9):
    perilesional_rois = []
    unique_rois = np.unique(brain_template)
    unique_rois = unique_rois[unique_rois > 0]
    for roi_id in unique_rois:
        roi_mask = (brain_template == roi_id)
        roi_volume = np.sum(roi_mask)
        if roi_volume == 0:
            continue
        overlap = np.sum(lesion_mask & roi_mask)
        overlap_ratio = overlap / roi_volume
        if 0 < overlap_ratio < overlap_threshold:
            perilesional_rois.append(roi_id)
    return perilesional_rois


def extract_remote_rois(lesion_mask, brain_template):
    remote_mask = np.flip(lesion_mask, axis=0)
    remote_mask = remote_mask & (~lesion_mask)
    return extract_lesion_rois(remote_mask, brain_template, 0.9)


def load_or_create_roi_cache(brain_template, clinical_dict):
    """
    Load ROI cache if exists, otherwise create it
    Cache stores: {subject_id: {'lesion': [...], 'perilesional': [...], 'remote': [...]}}
    """
    if os.path.exists(ROI_CACHE_FILE):
        print(f"Loading ROI cache from {ROI_CACHE_FILE}...")
        roi_cache = np.load(ROI_CACHE_FILE, allow_pickle=True).item()
        print(f"  Loaded cache for {len(roi_cache)} subjects")
        return roi_cache

    print("Creating ROI cache (this will take some time)...")
    roi_cache = {}

    for subject_dir in os.listdir(EXTRACTED_REGIONS_DIR):
        subject_path = os.path.join(EXTRACTED_REGIONS_DIR, subject_dir)
        if not os.path.isdir(subject_path):
            continue

        regions_file = os.path.join(subject_path, f"{subject_dir}_all_regions.npy")
        if not os.path.exists(regions_file):
            continue

        matched_id = None
        if f"sub-{subject_dir}" in clinical_dict:
            matched_id = f"sub-{subject_dir}"
        else:
            for cid in clinical_dict.keys():
                if cid.replace('sub-', '') == subject_dir:
                    matched_id = cid
                    break
        if matched_id is None:
            continue

        try:
            region_masks = np.load(regions_file, allow_pickle=True).item()
            lesion_mask = region_masks.get('lesion')
            if lesion_mask is None or np.sum(lesion_mask) == 0:
                continue

            lesion_rois = extract_lesion_rois(lesion_mask, brain_template)
            perilesional_rois = extract_perilesional_rois(lesion_mask, brain_template)
            remote_rois = extract_remote_rois(lesion_mask, brain_template)

            roi_cache[matched_id] = {
                'lesion': lesion_rois,
                'perilesional': perilesional_rois,
                'remote': remote_rois
            }
            print(f"  Cached {matched_id}")
        except Exception as e:
            print(f"  Error caching {subject_dir}: {e}")
            continue

    np.save(ROI_CACHE_FILE, roi_cache)
    print(f"Saved ROI cache to {ROI_CACHE_FILE}")
    return roi_cache


def compute_cohend_for_rois(patient_sfc, healthy_matrix, roi_list):
    """
    Compute Cohen's d effect size between patient and healthy controls
    Patient: all SFC values in the ROIs (not averaged)
    Healthy: all SFC values from all subjects in the same ROIs (not averaged)
    """
    if len(roi_list) == 0:
        return np.nan, np.nan
    roi_indices = np.array(roi_list) - 1
    roi_indices = roi_indices[(roi_indices >= 0) & (roi_indices < 450)]
    if len(roi_indices) == 0:
        return np.nan, np.nan

    # Patient data: all SFC values in these ROIs
    patient_values = patient_sfc[roi_indices]
    patient_values = patient_values[~np.isnan(patient_values)]

    # Healthy controls data: all SFC values from all subjects in these ROIs
    healthy_values = healthy_matrix[:, roi_indices].flatten()
    healthy_values = healthy_values[~np.isnan(healthy_values)]

    if len(patient_values) == 0 or len(healthy_values) == 0:
        return np.nan, np.nan

    # Calculate means
    patient_mean = np.mean(patient_values)
    healthy_mean = np.mean(healthy_values)

    # Calculate pooled standard deviation
    patient_std = np.std(patient_values, ddof=1)
    healthy_std = np.std(healthy_values, ddof=1)
    n1 = len(patient_values)
    n2 = len(healthy_values)

    pooled_std = np.sqrt(((n1 - 1) * patient_std**2 + (n2 - 1) * healthy_std**2) / (n1 + n2 - 2))

    # Calculate Cohen's d
    if pooled_std > 0:
        cohens_d = (patient_mean - healthy_mean) / pooled_std
    else:
        cohens_d = np.nan

    return patient_mean, cohens_d


def compute_cohend_vs_mean_template(patient_sfc, healthy_mean_sfc, roi_list):
    """
    Compute Cohen's d between patient ROI values and healthy mean template ROI values
    Patient: all SFC values in the ROIs
    Healthy template: mean SFC values for the same ROIs (from 25 healthy subjects)
    """
    if len(roi_list) == 0:
        return np.nan, np.nan
    roi_indices = np.array(roi_list) - 1
    roi_indices = roi_indices[(roi_indices >= 0) & (roi_indices < 450)]
    if len(roi_indices) == 0:
        return np.nan, np.nan

    # Patient data: all SFC values in these ROIs
    patient_values = patient_sfc[roi_indices]
    patient_values = patient_values[~np.isnan(patient_values)]

    # Healthy template data: mean SFC values for these ROIs
    healthy_template_values = healthy_mean_sfc[roi_indices]
    healthy_template_values = healthy_template_values[~np.isnan(healthy_template_values)]

    if len(patient_values) == 0 or len(healthy_template_values) == 0:
        return np.nan, np.nan

    # Calculate means
    patient_mean = np.mean(patient_values)
    healthy_template_mean = np.mean(healthy_template_values)

    # Calculate standard deviations
    patient_std = np.std(patient_values, ddof=1)
    healthy_template_std = np.std(healthy_template_values, ddof=1)
    n1 = len(patient_values)
    n2 = len(healthy_template_values)

    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * patient_std**2 + (n2 - 1) * healthy_template_std**2) / (n1 + n2 - 2))

    # Calculate Cohen's d
    if pooled_std > 0:
        cohens_d = (patient_mean - healthy_template_mean) / pooled_std
    else:
        cohens_d = np.nan

    return patient_mean, cohens_d


def load_clinical_data():
    df = pd.read_csv(CLINICAL_DATA_FILE, sep='\t')
    clinical_dict = {}
    for _, row in df.iterrows():
        participant_id = str(row['participant_id']).strip()
        aphasia_type = row['wab_type']
        if pd.isna(aphasia_type) or str(aphasia_type).strip() == '':
            aphasia_type = 'None'
        else:
            aphasia_type = str(aphasia_type).strip()
        # 统一 non-aphasic 的常见写法到 'None'
        aph_lower = aphasia_type.lower()
        if aph_lower in {
            'non-aphasic stroke', 'non aphasic stroke',
            'non-aphasic', 'non aphasic',
            'normal stroke non-aphasia', 'normal stroke non aphasia',
            'no aphasia', 'no-aphasia'
        }:
            aphasia_type = 'None'
        if aphasia_type in APHASIA_TYPES:
            clinical_dict[participant_id] = aphasia_type
    return clinical_dict


def process_all_patients(healthy_matrix, roi_cache):
    """Process all patients using cached ROI data"""
    print("\nProcessing patients...")
    clinical_dict = load_clinical_data()
    all_results = []

    for subject_dir in os.listdir(EXTRACTED_REGIONS_DIR):
        subject_path = os.path.join(EXTRACTED_REGIONS_DIR, subject_dir)
        if not os.path.isdir(subject_path):
            continue

        sfc_files = [f for f in os.listdir(SFC_DIR)
                     if subject_dir in f and f.endswith("_sfc_values_pca.txt")]
        if not sfc_files:
            continue

        matched_id = None
        if f"sub-{subject_dir}" in clinical_dict:
            matched_id = f"sub-{subject_dir}"
        else:
            for cid in clinical_dict.keys():
                if cid.replace('sub-', '') == subject_dir:
                    matched_id = cid
                    break
        if matched_id is None or matched_id not in roi_cache:
            continue

        print(f"  {subject_dir} ({matched_id})")

        try:
            patient_sfc = load_sfc_values(os.path.join(SFC_DIR, sfc_files[0]))
            if len(patient_sfc) != 450:
                continue
        except Exception as e:
            print(f"    Error: {e}")
            continue

        # Get ROIs from cache
        lesion_rois = roi_cache[matched_id]['lesion']
        perilesional_rois = roi_cache[matched_id]['perilesional']
        remote_rois = roi_cache[matched_id]['remote']

        lesion_raw, lesion_d = compute_cohend_for_rois(patient_sfc, healthy_matrix, lesion_rois)
        peri_raw, peri_d = compute_cohend_for_rois(patient_sfc, healthy_matrix, perilesional_rois)
        remote_raw, remote_d = compute_cohend_for_rois(patient_sfc, healthy_matrix, remote_rois)

        all_results.append({
            'subject_id': matched_id,
            'aphasia_type': clinical_dict[matched_id],
            'lesion_sfc_raw': lesion_raw,
            'lesion_sfc_cohend': lesion_d,
            'perilesional_sfc_raw': peri_raw,
            'perilesional_sfc_cohend': peri_d,
            'remote_sfc_raw': remote_raw,
            'remote_sfc_cohend': remote_d
        })

    return pd.DataFrame(all_results)


def process_all_patients_with_mean_template(healthy_mean_sfc, roi_cache):
    """Process all patients using healthy mean SFC template"""
    print("\nProcessing patients with mean template...")
    clinical_dict = load_clinical_data()
    all_results = []

    for subject_dir in os.listdir(EXTRACTED_REGIONS_DIR):
        subject_path = os.path.join(EXTRACTED_REGIONS_DIR, subject_dir)
        if not os.path.isdir(subject_path):
            continue

        sfc_files = [f for f in os.listdir(SFC_DIR)
                     if subject_dir in f and f.endswith("_sfc_values_pca.txt")]
        if not sfc_files:
            continue

        matched_id = None
        if f"sub-{subject_dir}" in clinical_dict:
            matched_id = f"sub-{subject_dir}"
        else:
            for cid in clinical_dict.keys():
                if cid.replace('sub-', '') == subject_dir:
                    matched_id = cid
                    break
        if matched_id is None or matched_id not in roi_cache:
            continue

        print(f"  {subject_dir} ({matched_id})")

        try:
            patient_sfc = load_sfc_values(os.path.join(SFC_DIR, sfc_files[0]))
            if len(patient_sfc) != 450:
                continue
        except Exception as e:
            print(f"    Error: {e}")
            continue

        # Get ROIs from cache
        lesion_rois = roi_cache[matched_id]['lesion']
        perilesional_rois = roi_cache[matched_id]['perilesional']
        remote_rois = roi_cache[matched_id]['remote']

        lesion_raw, lesion_d = compute_cohend_vs_mean_template(patient_sfc, healthy_mean_sfc, lesion_rois)
        peri_raw, peri_d = compute_cohend_vs_mean_template(patient_sfc, healthy_mean_sfc, perilesional_rois)
        remote_raw, remote_d = compute_cohend_vs_mean_template(patient_sfc, healthy_mean_sfc, remote_rois)

        all_results.append({
            'subject_id': matched_id,
            'aphasia_type': clinical_dict[matched_id],
            'lesion_sfc_raw': lesion_raw,
            'lesion_sfc_cohend': lesion_d,
            'perilesional_sfc_raw': peri_raw,
            'perilesional_sfc_cohend': peri_d,
            'remote_sfc_raw': remote_raw,
            'remote_sfc_cohend': remote_d
        })

    return pd.DataFrame(all_results)


def plot_results(data_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams['font.family'] = 'Arial'
    region_colors = {'Lesion': '#E3625D', 'Perilesional': '#7895C1', 'Remote': '#808080'}
    
    fig, axes = plt.subplots(1, len(APHASIA_TYPES), figsize=(15, 5), sharey=True)
    if len(APHASIA_TYPES) == 1:
        axes = [axes]
    
    rng = np.random.default_rng(seed=42)
    
    for idx, aphasia_type in enumerate(APHASIA_TYPES):
        ax = axes[idx]
        type_data = data_df[data_df['aphasia_type'] == aphasia_type]
        
        if len(type_data) == 0:
            ax.set_title(f"{aphasia_type}\n(n=0)", fontsize=12, fontweight='bold')
            ax.axis('off')
            continue
        
        regions = ['Lesion', 'Perilesional', 'Remote']
        positions = [0, 1, 2]
        data_by_region = {r: type_data[f"{r.lower()}_sfc_cohend"].dropna().values for r in regions}

        for _, row in type_data.iterrows():
            vals = [row['lesion_sfc_cohend'], row['perilesional_sfc_cohend'], row['remote_sfc_cohend']]
            if not any(pd.isna(vals)):
                ax.plot([0, 1, 2], vals, color='gray', alpha=0.2, linewidth=0.5, zorder=1)
        
        for pos, region in zip(positions, regions):
            data_points = data_by_region[region]
            if len(data_points) == 0:
                continue
            ax.boxplot(data_points, positions=[pos], widths=0.5, patch_artist=True, showfliers=False,
                      medianprops={'color': 'black', 'linewidth': 1.5},
                      boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 1.5},
                      whiskerprops={'color': 'black', 'linewidth': 1.2},
                      capprops={'color': 'black', 'linewidth': 0})
            ax.scatter(rng.normal(pos, 0.08, len(data_points)), data_points,
                      color=region_colors[region], alpha=0.8, s=20, edgecolors='none', zorder=3)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
        ax.set_title(f"{aphasia_type}\n(n={len(type_data)})", fontsize=12, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(regions, rotation=15, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.6)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    
    axes[0].set_ylabel("Cohen's d (vs healthy controls)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regional_sfc_cohend_by_aphasia.pdf'),
                format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved plot to {output_dir}")
    plt.close()


def plot_results_mean_template(data_df, output_dir):
    """Plot results using mean template comparison"""
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams['font.family'] = 'Arial'
    region_colors = {'Lesion': '#E3625D', 'Perilesional': '#7895C1', 'Remote': '#808080'}

    rng = np.random.default_rng(seed=42)
    # CSV 里 non-aphasic 的 aphasia_type 实际存成了 NaN，统一填为 'None' 再绘图
    data_df = data_df.copy()
    data_df['aphasia_type'] = data_df['aphasia_type'].fillna('None')
    display_name = {'None': 'Non-aphasic'}

    # 单一坐标轴：把不同失语症类别按组排在同一张图上
    regions = ['Lesion', 'Perilesional', 'Remote']
    n_regions = len(regions)
    region_step = 1.0  # 恢复更紧凑的组内间距
    gap = 1.0  # 恢复更紧凑的组间间距

    # 整体尺寸缩小一些（单图）
    fig_w = 1.7 * len(APHASIA_TYPES) + 3.0
    fig, ax = plt.subplots(figsize=(fig_w, 3.8))

    # 为每个类别计算 x 基准位置，并绘制交替背景
    group_centers = []
    group_labels = []
    group_spans = []

    for idx, aphasia_type in enumerate(APHASIA_TYPES):
        base = idx * (n_regions * region_step + gap * region_step)
        x_positions = [base + i * region_step for i in range(n_regions)]
        center = base + (n_regions - 1) * region_step / 2
        group_centers.append(center)
        group_labels.append(display_name.get(aphasia_type, aphasia_type))
        # 背景跨度（覆盖该组的 3 个 region）
        left = base - 0.5 * region_step
        right = base + (n_regions - 1) * region_step + 0.5 * region_step
        group_spans.append((left, right))
        ax.axvspan(left, right, facecolor=('#f2f2f2' if (idx % 2 == 0) else 'white'), zorder=-5)

        type_data = data_df[data_df['aphasia_type'] == aphasia_type]

        # 连接线：每个被试 3 个点
        for _, row in type_data.iterrows():
            vals = [row['lesion_sfc_cohend'], row['perilesional_sfc_cohend'], row['remote_sfc_cohend']]
            if not any(pd.isna(vals)):
                ax.plot(x_positions, vals, color='gray', alpha=0.2, linewidth=0.5, zorder=1)

        # boxplot + scatter（每个 region）
        for pos, region in zip(x_positions, regions):
            data_points = type_data[f"{region.lower()}_sfc_cohend"].dropna().values
            if len(data_points) == 0:
                continue
            ax.boxplot(
                data_points, positions=[pos], widths=0.55, patch_artist=True, showfliers=False,
                medianprops={'color': 'black', 'linewidth': 1.5},
                boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 1.2},
                whiskerprops={'color': 'black', 'linewidth': 1.0},
                capprops={'color': 'black', 'linewidth': 0}
            )
            ax.scatter(
                rng.normal(pos, 0.08, len(data_points)), data_points,
                color=region_colors[region], alpha=0.85, s=18, edgecolors='none', zorder=3
            )

        # 每组上方居中：失语症类型 + 样本量
        ax.text(
            center, 1.07, f"{display_name.get(aphasia_type, aphasia_type)}",
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )
        ax.text(
            center, 1.02, f"n={len(type_data)}",
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    ax.set_ylabel("Cohen's d (vs healthy mean template)", fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.6)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # 不在 x 轴底部显示类别（类别标签已放到上方）
    ax.set_xticks([])

    # 在每组内部标出 Lesion/Perilesional/Remote（次级标签）
    y0, y1 = ax.get_ylim()
    label_y = y0 - 0.06 * (y1 - y0)
    for idx, (left, right) in enumerate(group_spans):
        base = idx * (n_regions * region_step + gap * region_step)
        for i, region in enumerate(regions):
            ax.text(base + i * region_step, label_y, region, ha='center', va='top', fontsize=9, rotation=0)

    # 给底部留一点空间放次级标签
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(os.path.join(output_dir, 'regional_sfc_cohend_by_aphasia_mean_template.pdf'),
                format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved mean template plot to {output_dir}")
    plt.close()


def main():
    print("=" * 60)
    print("Regional SFC Z-score Analysis")
    print("=" * 60)

    # Load healthy controls matrix
    print("\nLoading healthy controls matrix...")
    healthy_matrix = np.load(HEALTHY_MATRIX_FILE)
    print(f"  Loaded: {healthy_matrix.shape[0]} healthy controls")

    # Create or load healthy mean SFC template
    if os.path.exists(HEALTHY_MEAN_SFC_FILE):
        print(f"\nLoading healthy mean SFC template from {HEALTHY_MEAN_SFC_FILE}...")
        healthy_mean_sfc = np.load(HEALTHY_MEAN_SFC_FILE)
        print(f"  Template shape: {healthy_mean_sfc.shape}")
    else:
        print("\nCreating healthy mean SFC template...")
        healthy_mean_sfc = np.nanmean(healthy_matrix, axis=0)
        np.save(HEALTHY_MEAN_SFC_FILE, healthy_mean_sfc)
        print(f"  Saved template to {HEALTHY_MEAN_SFC_FILE}")
        print(f"  Template shape: {healthy_mean_sfc.shape}")

    # Load brain template and clinical data
    brain_template = load_brain_template()
    clinical_dict = load_clinical_data()

    # Load or create ROI cache
    roi_cache = load_or_create_roi_cache(brain_template, clinical_dict)

    # Process patients with original method
    print("\n" + "=" * 60)
    print("Method 1: Cohen's d vs all healthy controls")
    print("=" * 60)
    data_df = process_all_patients(healthy_matrix, roi_cache)
    print(f"\nProcessed {len(data_df)} patients")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_df.to_csv(os.path.join(OUTPUT_DIR, 'regional_sfc_zscore_data.csv'), index=False)
    print(f"\nSaved data to {OUTPUT_DIR}")

    # Create plots
    print("\nCreating plots...")
    plot_results(data_df, OUTPUT_DIR)

    # Process patients with mean template method
    print("\n" + "=" * 60)
    print("Method 2: Cohen's d vs healthy mean template")
    print("=" * 60)
    data_df_mean = process_all_patients_with_mean_template(healthy_mean_sfc, roi_cache)
    print(f"\nProcessed {len(data_df_mean)} patients")

    # Save results
    data_df_mean.to_csv(os.path.join(OUTPUT_DIR, 'regional_sfc_zscore_data_mean_template.csv'), index=False)
    print(f"\nSaved mean template data to {OUTPUT_DIR}")

    # Create plots
    print("\nCreating mean template plots...")
    plot_results_mean_template(data_df_mean, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
