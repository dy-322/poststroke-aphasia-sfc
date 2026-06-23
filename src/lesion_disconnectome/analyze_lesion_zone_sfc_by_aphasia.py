"""
Regional SFC Analysis by Aphasia Type

This script analyzes SFC differences across three spatial regions:
1. Lesion (within lesion mask)
2. Perilesional (10-20mm from lesion border)
3. Remote (contralateral mirror of lesion)

Grouped by 5 aphasia type categories with box plot visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
from scipy.ndimage import distance_transform_edt, binary_dilation
import nibabel as nib
import pickle
import re


# ==================== Configuration ====================
SFC_DIR = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\sfc_txt_files"
EXTRACTED_REGIONS_DIR = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\extracted_regions"
CLINICAL_DATA_FILE = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\aphasia_plots\participants_aphasia_new.txt"
BRAIN_TEMPLATE_FILE = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_MNI152NLin6Asym_1mm.nii.gz"
OUTPUT_DIR = "regional_sfc_analysis_results"
CACHE_FILE = "regional_rois_cache.pkl"  # Cache file for ROI extraction

# Aphasia types to include (5 categories)
APHASIA_TYPES = ['Broca', 'Anomic', 'Conduction', 'Global', 'None']

# Distance range for perilesional region (in mm, assuming 1mm voxel size)
PERILESIONAL_MIN_DIST = 10
PERILESIONAL_MAX_DIST = 20

# Global brain template (loaded once)
BRAIN_TEMPLATE = None


# ==================== Helper Functions ====================
def normalize_subject_id(subject_id):
    """Extract normalized subject ID for matching"""
    if subject_id is None:
        return None
    value_str = str(subject_id).strip()
    if not value_str:
        return None

    match = re.search(r'(?:M)?\d+', value_str, re.IGNORECASE)
    if match:
        token = match.group(0).upper()
        return token

    digits = ''.join(ch for ch in value_str if ch.isdigit())
    if digits:
        return digits
    return value_str


def load_brain_template():
    """Load brain template (ROI atlas) once at the beginning"""
    global BRAIN_TEMPLATE
    if BRAIN_TEMPLATE is None:
        print(f"Loading brain template from {BRAIN_TEMPLATE_FILE}...")
        img = nib.load(BRAIN_TEMPLATE_FILE)
        BRAIN_TEMPLATE = img.get_fdata().astype(int)
        print(f"Brain template loaded: shape {BRAIN_TEMPLATE.shape}")
    return BRAIN_TEMPLATE


def load_roi_cache():
    """Load cached ROI extraction results"""
    if os.path.exists(CACHE_FILE):
        print(f"Loading ROI cache from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        print(f"Cache loaded: {len(cache)} subjects")
        return cache
    return {}


def save_roi_cache(cache):
    """Save ROI extraction results to cache"""
    print(f"Saving ROI cache to {CACHE_FILE}...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Cache saved: {len(cache)} subjects")


def load_clinical_data(clinical_file):
    """Load clinical data with aphasia types"""
    df = pd.read_csv(clinical_file, sep='\t')
    clinical_dict = {}
    
    for _, row in df.iterrows():
        participant_id = str(row['participant_id']).strip()
        aphasia_type = row['wab_type']
        
        if pd.isna(aphasia_type) or str(aphasia_type).strip() == '':
            aphasia_type = 'None'
        else:
            aphasia_type = str(aphasia_type).strip()
        
        # Only include specified aphasia types
        if aphasia_type in APHASIA_TYPES:
            clinical_dict[participant_id] = aphasia_type
    
    print(f"Loaded clinical data for {len(clinical_dict)} subjects")
    type_counts = {}
    for atype in clinical_dict.values():
        type_counts[atype] = type_counts.get(atype, 0) + 1
    print(f"Aphasia type distribution: {type_counts}")
    
    return clinical_dict


def load_sfc_values(sfc_file):
    """Load SFC values from file"""
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
    """Extract ROIs with >90% overlap with lesion mask"""
    lesion_rois = []
    unique_rois = np.unique(brain_template)
    unique_rois = unique_rois[unique_rois > 0]

    for roi_id in unique_rois:
        roi_mask = (brain_template == roi_id)
        roi_volume = np.sum(roi_mask)
        if roi_volume == 0:
            continue
        overlap = np.sum(lesion_mask & roi_mask)
        overlap_ratio = overlap / roi_volume
        if overlap_ratio >= overlap_threshold:
            lesion_rois.append(roi_id)
    return lesion_rois


def extract_perilesional_rois(lesion_mask, brain_template, overlap_threshold=0.9):
    """Extract ROIs with some overlap but <90% with lesion"""
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
    """Extract ROIs in contralateral mirror of lesion"""
    remote_mask = np.flip(lesion_mask, axis=0)
    remote_mask = remote_mask & (~lesion_mask)
    return extract_lesion_rois(remote_mask, brain_template, overlap_threshold=0.9)


def extract_rois_for_subject(subject_dir, regions_file, brain_template, roi_cache):
    """Extract ROIs for a subject, using cache if available"""
    # Check cache first
    if subject_dir in roi_cache:
        return roi_cache[subject_dir]

    # Load region masks
    try:
        region_masks = np.load(regions_file, allow_pickle=True).item()
        lesion_mask = region_masks.get('lesion')

        if lesion_mask is None or np.sum(lesion_mask) == 0:
            return None
    except Exception as e:
        print(f"  Error loading regions: {e}")
        return None

    # Extract ROIs
    lesion_rois = extract_lesion_rois(lesion_mask, brain_template)
    perilesional_rois = extract_perilesional_rois(lesion_mask, brain_template)
    remote_rois = extract_remote_rois(lesion_mask, brain_template)

    # Cache the results
    roi_data = {
        'lesion_rois': lesion_rois,
        'perilesional_rois': perilesional_rois,
        'remote_rois': remote_rois
    }
    roi_cache[subject_dir] = roi_data

    return roi_data


def calculate_mean_sfc_for_rois(sfc_values, roi_list):
    """Calculate mean SFC for a list of ROI indices"""
    if len(roi_list) == 0:
        return np.nan

    # ROI labels start from 1, SFC array is 0-indexed
    sfc_indices = np.array(roi_list) - 1
    sfc_indices = sfc_indices[(sfc_indices >= 0) & (sfc_indices < len(sfc_values))]

    if len(sfc_indices) == 0:
        return np.nan

    return np.nanmean(sfc_values[sfc_indices])


def process_single_subject(subject_id, subject_dir, sfc_file, regions_file, clinical_dict, brain_template, roi_cache):
    """Process a single subject and extract regional SFC values"""
    # Check if subject has clinical data
    if subject_id not in clinical_dict:
        return None

    aphasia_type = clinical_dict[subject_id]

    # Extract ROIs (using cache if available)
    roi_data = extract_rois_for_subject(subject_dir, regions_file, brain_template, roi_cache)
    if roi_data is None:
        print(f"  Warning: Could not extract ROIs for {subject_id}")
        return None

    lesion_rois = roi_data['lesion_rois']
    perilesional_rois = roi_data['perilesional_rois']
    remote_rois = roi_data['remote_rois']

    # Load SFC values
    try:
        sfc_values = load_sfc_values(sfc_file)
        if len(sfc_values) != 450:
            print(f"  Warning: {subject_id} has {len(sfc_values)} SFC values, expected 450")
            return None
    except Exception as e:
        print(f"  Error loading SFC for {subject_id}: {e}")
        return None

    # Calculate mean SFC for each region
    lesion_sfc = calculate_mean_sfc_for_rois(sfc_values, lesion_rois)
    perilesional_sfc = calculate_mean_sfc_for_rois(sfc_values, perilesional_rois)
    remote_sfc = calculate_mean_sfc_for_rois(sfc_values, remote_rois)

    return {
        'subject_id': subject_id,
        'aphasia_type': aphasia_type,
        'lesion_sfc': lesion_sfc,
        'perilesional_sfc': perilesional_sfc,
        'remote_sfc': remote_sfc,
        'lesion_rois': len(lesion_rois),
        'perilesional_rois': len(perilesional_rois),
        'remote_rois': len(remote_rois)
    }


def collect_all_subjects_data():
    """Collect regional SFC data for all subjects"""
    print("=" * 60)
    print("Starting Regional SFC Analysis")
    print("=" * 60)

    # Load brain template
    brain_template = load_brain_template()

    # Load ROI cache
    roi_cache = load_roi_cache()

    # Load clinical data
    clinical_dict = load_clinical_data(CLINICAL_DATA_FILE)

    # Find all subjects with SFC files and region masks
    all_data = []

    print(f"\nScanning {EXTRACTED_REGIONS_DIR}...")
    subject_dirs = os.listdir(EXTRACTED_REGIONS_DIR)
    print(f"Found {len(subject_dirs)} directories")

    for subject_dir in subject_dirs:
        subject_path = os.path.join(EXTRACTED_REGIONS_DIR, subject_dir)
        if not os.path.isdir(subject_path):
            continue

        # Find the all_regions.npy file
        regions_file = os.path.join(subject_path, f"{subject_dir}_all_regions.npy")
        if not os.path.exists(regions_file):
            continue

        # Find corresponding SFC file
        sfc_files = []
        for fname in os.listdir(SFC_DIR):
            if subject_dir in fname and fname.endswith("_sfc_values_pca.txt"):
                sfc_files.append(os.path.join(SFC_DIR, fname))

        if not sfc_files:
            continue

        sfc_file = sfc_files[0]

        # Match with clinical data
        matched_id = None
        if f"sub-{subject_dir}" in clinical_dict:
            matched_id = f"sub-{subject_dir}"
        else:
            for clinical_id in clinical_dict.keys():
                clinical_id_clean = clinical_id.replace('sub-', '')
                if clinical_id_clean == subject_dir:
                    matched_id = clinical_id
                    break

        if matched_id is None:
            continue

        print(f"Processing {subject_dir} (matched to {matched_id})...")
        result = process_single_subject(matched_id, subject_dir, sfc_file, regions_file,
                                       clinical_dict, brain_template, roi_cache)

        if result is not None:
            all_data.append(result)

    # Save updated cache
    save_roi_cache(roi_cache)

    print(f"\nSuccessfully processed {len(all_data)} subjects")
    return all_data


def get_significance_stars(p_value):
    """Return significance stars based on p-value"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def plot_regional_sfc_by_aphasia(data_df, output_dir):
    """Create box plots showing regional SFC differences by aphasia type"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set font
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # Color scheme for regions
    region_colors = {
        'Lesion': '#E3625D',
        'Perilesional': '#7895C1',
        'Remote': '#808080'
    }
    
    # Create figure
    fig, axes = plt.subplots(1, len(APHASIA_TYPES), figsize=(15, 5), sharey=True)
    
    if len(APHASIA_TYPES) == 1:
        axes = [axes]
    
    rng = np.random.default_rng(seed=42)
    
    # Plot each aphasia type
    for idx, aphasia_type in enumerate(APHASIA_TYPES):
        ax = axes[idx]
        
        # Filter data for this aphasia type
        type_data = data_df[data_df['aphasia_type'] == aphasia_type]
        
        if len(type_data) == 0:
            ax.set_title(f"{aphasia_type}\n(n=0)", fontsize=12, fontweight='bold')
            ax.axis('off')
            continue
        
        # Prepare data for three regions
        regions = ['Lesion', 'Perilesional', 'Remote']
        positions = [0, 1, 2]
        
        data_by_region = {}
        for region in regions:
            col_name = f"{region.lower()}_sfc"
            data_by_region[region] = type_data[col_name].dropna().values

        # Draw connecting lines for each subject (before plotting boxes and points)
        for _, row in type_data.iterrows():
            lesion_val = row['lesion_sfc']
            peri_val = row['perilesional_sfc']
            remote_val = row['remote_sfc']

            if not (pd.isna(lesion_val) or pd.isna(peri_val) or pd.isna(remote_val)):
                ax.plot([0, 1, 2], [lesion_val, peri_val, remote_val],
                       color='gray', alpha=0.2, linewidth=0.5, zorder=1)

        # Plot box plots for each region
        for pos, region in zip(positions, regions):
            data_points = data_by_region[region]
            
            if len(data_points) == 0:
                continue
            
            color = region_colors[region]
            
            # Box plot
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
            
            # Scatter points
            ax.scatter(
                rng.normal(pos, 0.08, len(data_points)),
                data_points,
                color=color,
                alpha=0.8,
                s=20,
                edgecolors='none',
                zorder=3,
            )
        
        # Statistical testing between regions
        if len(data_by_region['Lesion']) > 0 and len(data_by_region['Perilesional']) > 0:
            try:
                stat, p_val = stats.mannwhitneyu(
                    data_by_region['Lesion'], 
                    data_by_region['Perilesional'], 
                    alternative='two-sided'
                )
                stars = get_significance_stars(p_val)
                if stars != 'ns':
                    y_max = max(np.max(data_by_region['Lesion']), np.max(data_by_region['Perilesional']))
                    y_pos = y_max + 0.02
                    ax.plot([0, 1], [y_pos, y_pos], 'k-', lw=1.2)
                    ax.text(0.5, y_pos + 0.005, stars, ha='center', va='bottom', fontsize=9)
            except:
                pass
        
        # Set subplot properties
        ax.set_title(f"{aphasia_type}\n(n={len(type_data)})", fontsize=12, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(regions, rotation=15, fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Grid and spines
        ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.6)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.2)
    
    # Set y-axis label on first subplot
    axes[0].set_ylabel('Mean SFC', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'regional_sfc_by_aphasia.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()
    
    # Create legend separately
    fig_legend, ax_legend = plt.subplots(figsize=(6, 2))
    ax_legend.axis('off')
    
    legend_handles = []
    for region, color in region_colors.items():
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10,
                                        label=region, linestyle='None'))
    
    ax_legend.legend(handles=legend_handles, loc='center', frameon=True,
                    fontsize=12, ncol=3, prop={'family': 'Arial'})
    
    legend_path = os.path.join(output_dir, 'regional_legend.pdf')
    plt.savefig(legend_path, format='pdf', bbox_inches='tight')
    plt.close()


def save_results_to_csv(data_df, output_dir):
    """Save analysis results to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw data
    csv_path = os.path.join(output_dir, 'regional_sfc_data.csv')
    data_df.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")
    
    # Create summary statistics
    summary_data = []
    
    for aphasia_type in APHASIA_TYPES:
        type_data = data_df[data_df['aphasia_type'] == aphasia_type]
        
        if len(type_data) == 0:
            continue
        
        for region in ['lesion', 'perilesional', 'remote']:
            col_name = f"{region}_sfc"
            values = type_data[col_name].dropna()
            
            summary_data.append({
                'aphasia_type': aphasia_type,
                'region': region,
                'n_subjects': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'regional_sfc_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    
    return summary_df


def perform_statistical_tests(data_df, output_dir):
    """Perform statistical tests comparing regions within each aphasia type"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for aphasia_type in APHASIA_TYPES:
        type_data = data_df[data_df['aphasia_type'] == aphasia_type]
        
        if len(type_data) < 3:
            continue
        
        # Compare Lesion vs Perilesional
        lesion_vals = type_data['lesion_sfc'].dropna().values
        peri_vals = type_data['perilesional_sfc'].dropna().values
        remote_vals = type_data['remote_sfc'].dropna().values
        
        comparisons = [
            ('Lesion', 'Perilesional', lesion_vals, peri_vals),
            ('Lesion', 'Remote', lesion_vals, remote_vals),
            ('Perilesional', 'Remote', peri_vals, remote_vals)
        ]
        
        for region1, region2, vals1, vals2 in comparisons:
            if len(vals1) > 0 and len(vals2) > 0:
                try:
                    stat, p_val = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
                    results.append({
                        'aphasia_type': aphasia_type,
                        'comparison': f"{region1} vs {region2}",
                        'n1': len(vals1),
                        'n2': len(vals2),
                        'statistic': stat,
                        'p_value': p_val,
                        'significance': get_significance_stars(p_val)
                    })
                except:
                    pass
    
    stats_df = pd.DataFrame(results)
    stats_path = os.path.join(output_dir, 'statistical_tests.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved statistical tests to {stats_path}")
    
    return stats_df


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("Regional SFC Analysis by Aphasia Type")
    print("=" * 60)
    
    # Collect data from all subjects
    all_data = collect_all_subjects_data()
    
    if len(all_data) == 0:
        print("No data collected. Exiting.")
        return
    
    # Convert to DataFrame
    data_df = pd.DataFrame(all_data)
    
    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)
    print(f"Total subjects: {len(data_df)}")
    print("\nSubjects per aphasia type:")
    for aphasia_type in APHASIA_TYPES:
        count = len(data_df[data_df['aphasia_type'] == aphasia_type])
        print(f"  {aphasia_type}: {count}")
    
    # Save results to CSV
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    summary_df = save_results_to_csv(data_df, OUTPUT_DIR)
    
    # Perform statistical tests
    stats_df = perform_statistical_tests(data_df, OUTPUT_DIR)
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    plot_regional_sfc_by_aphasia(data_df, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
