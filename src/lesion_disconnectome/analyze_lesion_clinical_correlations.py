#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Correlation Analysis Script
Analyzes correlations between lesion variables, AI asymmetry variables, SFC variables, and clinical scores
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import glob
import re
from pathlib import Path
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("Warning: pingouin not installed. Will use manual partial correlation.")
from sklearn.linear_model import LinearRegression

# Set matplotlib parameters
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42

def load_data():
    """Load all required data"""
    print("="*80)
    print("Loading data...")
    print("="*80)
    
    # 1. Load disconnectome data (contains 7 disconnection variables and clinical scores)
    disconn_file = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\disconnectome_mediation_results\subject_measures_all_scores.csv"
    disconn_df = pd.read_csv(disconn_file)
    print(f"Loaded disconnectome data: {len(disconn_df)} subjects")
    
    # 2. Load network lesion percentage data (7 networks) - optional
    network_lesion_file = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\disconnectome_mediation_results\network_lesion_percentages.csv"
    if os.path.exists(network_lesion_file):
        network_lesion_df = pd.read_csv(network_lesion_file)
        print(f"Loaded network lesion data: {len(network_lesion_df)} subjects")
    else:
        print(f"Warning: Network lesion file not found, will calculate it...")
        network_lesion_df = calculate_network_lesion_percentages()
    
    # 3. Load aphasia type data
    aphasia_type_file = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\single_subject_results_t288\filtered_participants.csv"
    aphasia_type_df = None
    if os.path.exists(aphasia_type_file):
        aphasia_type_df = pd.read_csv(aphasia_type_file)
        # Extract subject_id from participant_id (remove 'sub-M' prefix)
        aphasia_type_df['subject_id'] = aphasia_type_df['participant_id'].str.replace('sub-M', '', regex=False)
        # Keep only subject_id and wab_type columns
        aphasia_type_df = aphasia_type_df[['subject_id', 'wab_type']]
        print(f"Loaded aphasia type data: {len(aphasia_type_df)} subjects")
    else:
        print(f"Warning: Aphasia type file not found")
    
    return disconn_df, network_lesion_df, aphasia_type_df

def calculate_network_lesion_percentages():
    """Calculate lesion percentages for 7 networks"""
    print("\nCalculating network lesion percentages...")
    
    import nibabel as nib
    from scipy.ndimage import zoom
    
    lesion_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\extracted_regions"
    atlas_path = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_MNI152NLin6Asym_1mm.nii.gz"
    template_file = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
    
    if not os.path.exists(atlas_path):
        print(f"Warning: Atlas not found, returning empty dataframe")
        return pd.DataFrame()
    
    # Load atlas
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(int)
    
    # Parse network ROI map
    network_roi_map = parse_network_rois(template_file)
    networks = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis", "Subcortical"]
    
    network_lesion_data = []
    
    # Get all subject directories
    subject_dirs = [d for d in os.listdir(lesion_dir) if os.path.isdir(os.path.join(lesion_dir, d))]
    
    for subject_dir_name in subject_dirs:
        subject_dir = os.path.join(lesion_dir, subject_dir_name)
        all_regions_file = os.path.join(subject_dir, f"{subject_dir_name}_all_regions.npy")
        
        if not os.path.exists(all_regions_file):
            continue
        
        try:
            region_masks = np.load(all_regions_file, allow_pickle=True).item()
            lesion_data = region_masks.get('lesion')
            
            if lesion_data is None:
                continue
            
            # Resample lesion to atlas space if needed
            if lesion_data.shape != atlas_data.shape:
                zoom_factors = [a / l for a, l in zip(atlas_data.shape, lesion_data.shape)]
                lesion_data = zoom(lesion_data, zoom_factors, order=0)
                lesion_data = (lesion_data > 0.5).astype(np.float32)
            
            # Extract subject_id (remove 'M' prefix if present)
            subject_id = subject_dir_name.replace('M', '')
            
            row_data = {'subject_id': subject_id}
            
            # Calculate lesion percentage for each network
            for network in networks:
                roi_indices = network_roi_map.get(network, [])
                if not roi_indices:
                    row_data[f'{network}_lesion_pct'] = 0.0
                    continue
                
                # Create mask for this network's ROIs
                network_mask = np.zeros_like(atlas_data, dtype=bool)
                for roi_idx in roi_indices:
                    atlas_label = roi_idx + 1
                    network_mask |= (atlas_data == atlas_label)
                
                # Calculate total voxels in this network
                network_total_voxels = np.sum(network_mask)
                
                if network_total_voxels == 0:
                    row_data[f'{network}_lesion_pct'] = 0.0
                    continue
                
                # Calculate lesion voxels in this network
                lesion_in_network = np.sum(lesion_data & network_mask)
                
                # Calculate percentage
                lesion_pct = lesion_in_network / network_total_voxels
                row_data[f'{network}_lesion_pct'] = lesion_pct
            
            network_lesion_data.append(row_data)
            
        except Exception as e:
            print(f"  Warning: Failed to process {subject_dir_name}: {e}")
            continue
    
    network_lesion_df = pd.DataFrame(network_lesion_data)
    print(f"Calculated network lesion percentages for {len(network_lesion_df)} subjects")
    
    # Save to file
    output_file = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\disconnectome_mediation_results\network_lesion_percentages.csv"
    network_lesion_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
    
    return network_lesion_df

def calculate_ai_for_all_subjects():
    """Calculate AI asymmetry index for all subjects"""
    print("\nCalculating AI asymmetry indices...")
    
    sfc_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\sfc_txt_files"
    template_file = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
    
    # Get left-right hemisphere pairs
    pairs = []
    for i in range(25):
        left_idx = i + 25
        right_idx = i
        pairs.append((left_idx, right_idx))
    for i in range(200):
        left_idx = i + 50
        right_idx = i + 250
        pairs.append((left_idx, right_idx))
    
    # Get network ROI mapping
    network_roi_map = parse_network_rois(template_file)
    
    ai_data = []
    
    sfc_files = glob.glob(os.path.join(sfc_dir, "*sfc*.txt"))
    
    for sfc_file in sfc_files:
        # Extract subject_id
        filename = os.path.basename(sfc_file)
        match = re.search(r'M?(\d+)', filename)
        if not match:
            continue
        subject_id = match.group(1)
        
        # Load SFC data
        sfc_values = load_sfc_file(sfc_file)
        if sfc_values is None:
            continue
        
        # Calculate global AI
        global_ai = compute_ai(sfc_values, pairs)
        
        # Calculate dorsal and ventral AI
        dorsal_pairs = get_network_pairs(network_roi_map, 'DorsAttn')
        ventral_pairs = get_network_pairs(network_roi_map, 'SalVentAttn')
        dorsal_ai = compute_ai(sfc_values, dorsal_pairs)
        ventral_ai = compute_ai(sfc_values, ventral_pairs)
        
        # Calculate AI for 7 networks
        network_ais = {}
        for network in ['Cont', 'Default', 'DorsAttn', 'Limbic', 'SalVentAttn', 'SomMot', 'Vis']:
            net_pairs = get_network_pairs(network_roi_map, network)
            network_ais[f'{network}_ai'] = compute_ai(sfc_values, net_pairs)
        
        row = {
            'subject_id': subject_id,
            'global_ai': global_ai,
            'dorsal_ai': dorsal_ai,
            'ventral_ai': ventral_ai,
            **network_ais
        }
        ai_data.append(row)
    
    ai_df = pd.DataFrame(ai_data)
    print(f"Calculated AI for {len(ai_df)} subjects")
    return ai_df

def load_sfc_file(filepath):
    """Load SFC file"""
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
        if len(vals) == 450:
            return np.array(vals)
        return None
    except:
        return None

def compute_ai(sfc_values, pairs):
    """Calculate AI asymmetry index"""
    ai_values = []
    for left_idx, right_idx in pairs:
        left_val = sfc_values[left_idx]
        right_val = sfc_values[right_idx]
        if not np.isnan(left_val) and not np.isnan(right_val):
            denominator = abs((left_val + right_val) / 2)
            if denominator > 1e-10:
                ai = (left_val - right_val) / denominator
                ai_values.append(ai)
    if len(ai_values) == 0:
        return np.nan
    return np.mean(ai_values)

def parse_network_rois(template_file):
    """Parse network ROI mapping"""
    network_roi_map = {
        'Cont': [], 'Default': [], 'DorsAttn': [], 'Limbic': [],
        'SalVentAttn': [], 'SomMot': [], 'Vis': [], 'Subcortical': []
    }
    
    with open(template_file, 'r') as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        line = line.strip()
        network = 'Subcortical'
        if '7Networks' in line:
            match = re.search(r'7Networks_[LR]H_([A-Za-z]+)', line)
            if match:
                network = match.group(1)
        if network in network_roi_map:
            network_roi_map[network].append(idx)
    
    return network_roi_map

def get_network_pairs(network_roi_map, network):
    """Get left-right pairs for a network"""
    rois = network_roi_map.get(network, [])
    pairs = []
    # Only take left hemisphere ROIs (50-249) paired with right hemisphere ROIs (250-449)
    left_rois = [r for r in rois if 50 <= r < 250]
    for left_idx in left_rois:
        right_idx = left_idx + 200
        if right_idx < 450:
            pairs.append((left_idx, right_idx))
    return pairs

def calculate_partial_correlation(df, x_col, y_col, covariate_col):
    """Calculate partial correlation controlling for a covariate"""
    # Remove rows with missing values
    valid_data = df[[x_col, y_col, covariate_col]].dropna()
    
    if len(valid_data) < 3:
        return np.nan, np.nan, 0
    
    X = valid_data[x_col].values
    Y = valid_data[y_col].values
    Z = valid_data[covariate_col].values
    
    if HAS_PINGOUIN:
        # Use pingouin for partial correlation
        result = pg.partial_corr(data=valid_data, x=x_col, y=y_col, covar=covariate_col)
        return result['r'].values[0], result['p-val'].values[0], len(valid_data)
    else:
        # Manual calculation: regress out covariate from both X and Y
        # Residual of X ~ Z
        model_x = LinearRegression()
        model_x.fit(Z.reshape(-1, 1), X)
        X_resid = X - model_x.predict(Z.reshape(-1, 1))
        
        # Residual of Y ~ Z
        model_y = LinearRegression()
        model_y.fit(Z.reshape(-1, 1), Y)
        Y_resid = Y - model_y.predict(Z.reshape(-1, 1))
        
        # Correlation of residuals
        r, p = stats.pearsonr(X_resid, Y_resid)
        return r, p, len(valid_data)

def perform_correlation_analysis(df, x_vars, y_vars, analysis_name, control_tlv=False):
    """Perform correlation analysis with optional TLV control"""
    print(f"\n{'='*80}")
    print(f"{analysis_name}")
    if control_tlv:
        print("(Controlling for Total Lesion Volume)")
    print(f"{'='*80}")
    
    results = []
    
    for x_var in x_vars:
        x_name = x_var['name']
        x_label = x_var['label']
        
        if x_name not in df.columns:
            print(f"Warning: {x_name} not in data")
            continue
        
        for y_var in y_vars:
            y_name = y_var['name']
            y_label = y_var['label']
            
            if y_name not in df.columns:
                continue
            
            if control_tlv and 'lesion_volume' in df.columns:
                # Partial correlation controlling for TLV
                r, p, n = calculate_partial_correlation(df, x_name, y_name, 'lesion_volume')
                corr_type = 'partial'
            else:
                # Regular Pearson correlation
                X = df[x_name].values
                Y = df[y_name].values
                
                valid_mask = ~(np.isnan(X) | np.isnan(Y))
                X_valid = X[valid_mask]
                Y_valid = Y[valid_mask]
                
                if len(X_valid) < 3:
                    continue
                
                r, p = stats.pearsonr(X_valid, Y_valid)
                n = len(X_valid)
                corr_type = 'pearson'
            
            if np.isnan(r):
                continue
            
            results.append({
                'X_Variable': x_label,
                'Y_Variable': y_label,
                'r': r,
                'p_value': p,
                'n': n,
                'X_name': x_name,
                'Y_name': y_name,
                'correlation_type': corr_type
            })
            
            sig = '**' if p < 0.01 else '*' if p < 0.05 else ''
            corr_label = '(partial)' if corr_type == 'partial' else ''
            print(f"{x_label:35s} vs {y_label:20s}: r={r:7.4f}, p={p:.4f}, n={n} {sig} {corr_label}")
    
    return pd.DataFrame(results)

def plot_significant_correlations(df, corr_results, output_dir, prefix):
    """Plot significant correlations with |r| > 0.2 and save scatter data"""
    print(f"\nPlotting significant correlations for {prefix}...")
    
    # Filter for significant and |r| > 0.2
    sig_results = corr_results[(corr_results['p_value'] < 0.05) & (abs(corr_results['r']) > 0.2)]
    
    print(f"Found {len(sig_results)} significant correlations with |r| > 0.2")
    
    # Create subdirectory for scatter data
    scatter_data_dir = os.path.join(output_dir, 'scatter_data')
    os.makedirs(scatter_data_dir, exist_ok=True)
    
    for idx, row in sig_results.iterrows():
        x_name = row['X_name']
        y_name = row['Y_name']
        x_label = row['X_Variable']
        y_label = row['Y_Variable']
        r = row['r']
        p = row['p_value']
        n = row['n']
        
        # Extract data
        X = df[x_name].values
        Y = df[y_name].values
        
        valid_mask = ~(np.isnan(X) | np.isnan(Y))
        X_valid = X[valid_mask]
        Y_valid = Y[valid_mask]
        
        # Save scatter data to CSV with aphasia type if available
        scatter_data = {
            x_name: X_valid,
            y_name: Y_valid
        }
        
        # Add aphasia type if available in the dataframe
        if 'wab_type' in df.columns:
            wab_type_valid = df['wab_type'].values[valid_mask]
            scatter_data['wab_type'] = wab_type_valid
        
        scatter_df = pd.DataFrame(scatter_data)
        safe_x = x_name.replace('_', '-')
        safe_y = y_name.replace('_', '-')
        scatter_csv_file = os.path.join(scatter_data_dir, f'{prefix}_{safe_x}_vs_{safe_y}_data.csv')
        scatter_df.to_csv(scatter_csv_file, index=False)
        
        # Plot scatter
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(X_valid, Y_valid, c='#2874A6', s=80, alpha=0.6, 
                  edgecolors='black', linewidth=0.5)
        
        # Plot regression line
        z = np.polyfit(X_valid, Y_valid, 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(np.min(X_valid), np.max(X_valid), 100)
        ax.plot(x_line, p_fit(x_line), "r--", linewidth=2, alpha=0.8)
        
        # Add statistics
        corr_type = row.get('correlation_type', 'pearson')
        corr_label = '(Partial r)' if corr_type == 'partial' else '(Pearson r)'
        ax.text(0.05, 0.95, f'{corr_label}\nr = {r:.3f}\np = {p:.4f}\nn = {n}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
        ax.set_title(f'{x_label} vs {y_label}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f'{prefix}_{safe_x}_vs_{safe_y}.pdf')
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  Saved: {os.path.basename(output_file)} and {os.path.basename(scatter_csv_file)}")

def mediation_analysis(X, M, Y, n_bootstrap=5000):
    """
    Perform mediation analysis using bootstrap method.
    Tests: X -> M -> Y
    
    Returns:
        Dictionary with mediation results
    """
    # Standardize variables
    X = (X - np.mean(X)) / np.std(X)
    M = (M - np.mean(M)) / np.std(M)
    Y = (Y - np.mean(Y)) / np.std(Y)
    
    # Path c: Total effect (X -> Y)
    slope_c, intercept_c, r_c, p_c, se_c = stats.linregress(X, Y)
    total_effect = slope_c
    
    # Path a: X -> M
    slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(X, M)
    
    # Path b: M -> Y (controlling for X)
    model = LinearRegression()
    model.fit(np.column_stack([X, M]), Y)
    direct_effect = model.coef_[0]  # c' path
    b_path = model.coef_[1]  # b path
    
    # Indirect effect (mediation effect)
    indirect_effect = slope_a * b_path
    
    # Proportion mediated
    if total_effect != 0:
        proportion_mediated = indirect_effect / total_effect
    else:
        proportion_mediated = np.nan
    
    # Bootstrap confidence interval for indirect effect
    indirect_effects = []
    n_samples = len(X)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[indices]
        M_boot = M[indices]
        Y_boot = Y[indices]
        
        slope_a_boot, _, _, _, _ = stats.linregress(X_boot, M_boot)
        model_boot = LinearRegression()
        model_boot.fit(np.column_stack([X_boot, M_boot]), Y_boot)
        b_path_boot = model_boot.coef_[1]
        indirect_effects.append(slope_a_boot * b_path_boot)
    
    indirect_effects = np.array(indirect_effects)
    
    # Calculate p-value
    if indirect_effect > 0:
        p_value_indirect = np.mean(indirect_effects <= 0)
    else:
        p_value_indirect = np.mean(indirect_effects >= 0)
    p_value_indirect = 2 * min(p_value_indirect, 1 - p_value_indirect)
    
    # 95% confidence interval
    ci_lower = np.percentile(indirect_effects, 2.5)
    ci_upper = np.percentile(indirect_effects, 97.5)
    
    return {
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'indirect_effect': indirect_effect,
        'proportion_mediated': proportion_mediated,
        'p_value_indirect': p_value_indirect,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'path_a': slope_a,
        'path_b': b_path,
        'p_value_a': p_a,
    }

def perform_mediation_analysis(df, lesion_vars, ai_vars, clinical_vars, output_dir):
    """
    Perform mediation analysis: Lesion -> AI -> Clinical Score
    """
    print(f"\n{'='*80}")
    print("Mediation Analysis: Lesion -> AI -> Clinical Score")
    print(f"{'='*80}")
    
    mediation_results = []
    
    for lesion_var in lesion_vars:
        for ai_var in ai_vars:
            for clinical_var in clinical_vars:
                x_name = lesion_var['name']
                m_name = ai_var['name']
                y_name = clinical_var['name']
                
                if x_name not in df.columns or m_name not in df.columns or y_name not in df.columns:
                    continue
                
                # Extract data
                X = df[x_name].values
                M = df[m_name].values
                Y = df[y_name].values
                
                # Remove missing values
                valid_mask = ~(np.isnan(X) | np.isnan(M) | np.isnan(Y))
                X_valid = X[valid_mask]
                M_valid = M[valid_mask]
                Y_valid = Y[valid_mask]
                
                if len(X_valid) < 10:  # Need sufficient sample size
                    continue
                
                # Perform mediation analysis
                try:
                    result = mediation_analysis(X_valid, M_valid, Y_valid, n_bootstrap=5000)
                    
                    mediation_results.append({
                        'X_Variable': lesion_var['label'],
                        'M_Variable': ai_var['label'],
                        'Y_Variable': clinical_var['label'],
                        'X_name': x_name,
                        'M_name': m_name,
                        'Y_name': y_name,
                        'n': len(X_valid),
                        'total_effect': result['total_effect'],
                        'direct_effect': result['direct_effect'],
                        'indirect_effect': result['indirect_effect'],
                        'proportion_mediated': result['proportion_mediated'],
                        'p_value_indirect': result['p_value_indirect'],
                        'ci_lower': result['ci_lower'],
                        'ci_upper': result['ci_upper'],
                        'path_a': result['path_a'],
                        'path_b': result['path_b'],
                        'p_value_a': result['p_value_a']
                    })
                    
                    sig = '**' if result['p_value_indirect'] < 0.01 else '*' if result['p_value_indirect'] < 0.05 else ''
                    print(f"{lesion_var['label']:30s} -> {ai_var['label']:20s} -> {clinical_var['label']:15s}: "
                          f"indirect={result['indirect_effect']:7.4f}, p={result['p_value_indirect']:.4f} {sig}")
                    
                except Exception as e:
                    print(f"  Error in mediation: {lesion_var['label']} -> {ai_var['label']} -> {clinical_var['label']}: {e}")
                    continue
    
    mediation_df = pd.DataFrame(mediation_results)
    
    # Save results
    mediation_file = os.path.join(output_dir, '5_mediation_lesion_ai_clinical.csv')
    mediation_df.to_csv(mediation_file, index=False)
    print(f"\nMediation results saved to: {mediation_file}")
    
    # Plot significant mediations
    sig_mediations = mediation_df[mediation_df['p_value_indirect'] < 0.05]
    print(f"\nFound {len(sig_mediations)} significant mediations")
    
    for idx, row in sig_mediations.iterrows():
        plot_mediation_result(df, row, output_dir)
    
    return mediation_df

def plot_mediation_result(df, mediation_row, output_dir):
    """Plot mediation analysis result"""
    x_name = mediation_row['X_name']
    m_name = mediation_row['M_name']
    y_name = mediation_row['Y_name']
    
    X = df[x_name].values
    M = df[m_name].values
    Y = df[y_name].values
    
    valid_mask = ~(np.isnan(X) | np.isnan(M) | np.isnan(Y))
    X_valid = X[valid_mask]
    M_valid = M[valid_mask]
    Y_valid = Y[valid_mask]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: X -> Y (total effect)
    axes[0, 0].scatter(X_valid, Y_valid, alpha=0.6, s=60)
    z = np.polyfit(X_valid, Y_valid, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(X_valid, p(X_valid), 'r-', linewidth=2)
    axes[0, 0].set_xlabel(mediation_row['X_Variable'], fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel(mediation_row['Y_Variable'], fontsize=11, fontweight='bold')
    axes[0, 0].set_title(f"Total Effect (c path)\nβ = {mediation_row['total_effect']:.3f}", fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: X -> M (path a)
    axes[0, 1].scatter(X_valid, M_valid, alpha=0.6, s=60, color='green')
    z = np.polyfit(X_valid, M_valid, 1)
    p = np.poly1d(z)
    axes[0, 1].plot(X_valid, p(X_valid), 'g-', linewidth=2)
    axes[0, 1].set_xlabel(mediation_row['X_Variable'], fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel(mediation_row['M_Variable'], fontsize=11, fontweight='bold')
    axes[0, 1].set_title(f"Path a (X → M)\nβ = {mediation_row['path_a']:.3f}, p = {mediation_row['p_value_a']:.4f}", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: M -> Y (path b)
    axes[1, 0].scatter(M_valid, Y_valid, alpha=0.6, s=60, color='orange')
    z = np.polyfit(M_valid, Y_valid, 1)
    p = np.poly1d(z)
    axes[1, 0].plot(M_valid, p(M_valid), 'orange', linewidth=2)
    axes[1, 0].set_xlabel(mediation_row['M_Variable'], fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel(mediation_row['Y_Variable'], fontsize=11, fontweight='bold')
    axes[1, 0].set_title(f"Path b (M → Y | X)\nβ = {mediation_row['path_b']:.3f}", fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Summary
    axes[1, 1].axis('off')
    summary_text = f"""
Mediation Analysis Summary
{'='*40}

X: {mediation_row['X_Variable']}
M: {mediation_row['M_Variable']}
Y: {mediation_row['Y_Variable']}

Total Effect (c):      {mediation_row['total_effect']:.4f}
Direct Effect (c'):    {mediation_row['direct_effect']:.4f}
Indirect Effect (a×b): {mediation_row['indirect_effect']:.4f}

95% CI: [{mediation_row['ci_lower']:.4f}, {mediation_row['ci_upper']:.4f}]
p-value (bootstrap):   {mediation_row['p_value_indirect']:.4f}

Proportion Mediated:   {mediation_row['proportion_mediated']:.2%}
Sample Size:           {int(mediation_row['n'])}

{'[OK] Significant mediation' if mediation_row['p_value_indirect'] < 0.05 else '[X] Non-significant'}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    
    # Save figure
    safe_x = mediation_row['X_name'].replace('_', '-')
    safe_m = mediation_row['M_name'].replace('_', '-')
    safe_y = mediation_row['Y_name'].replace('_', '-')
    output_file = os.path.join(output_dir, f'5_mediation_{safe_x}_{safe_m}_{safe_y}.pdf')
    plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  Saved mediation plot: {os.path.basename(output_file)}")

def main():
    """Main function"""
    print("="*80)
    print("Comprehensive Correlation Analysis")
    print("="*80)
    
    # Create output directory
    output_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\correlation_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    disconn_df, network_lesion_df, aphasia_type_df = load_data()
    
    # 2. Calculate AI
    ai_df = calculate_ai_for_all_subjects()
    
    # 3. Merge data
    # Standardize subject_id
    disconn_df['subject_id'] = disconn_df['subject_id'].astype(str).str.strip()
    network_lesion_df['subject_id'] = network_lesion_df['subject_id'].astype(str).str.strip()
    ai_df['subject_id'] = ai_df['subject_id'].astype(str).str.strip()
    if aphasia_type_df is not None:
        aphasia_type_df['subject_id'] = aphasia_type_df['subject_id'].astype(str).str.strip()
    
    # Merge
    merged_df = disconn_df.merge(network_lesion_df, on='subject_id', how='inner', suffixes=('', '_net'))
    merged_df = merged_df.merge(ai_df, on='subject_id', how='inner', suffixes=('', '_ai'))
    
    # Merge aphasia type data if available
    if aphasia_type_df is not None:
        merged_df = merged_df.merge(aphasia_type_df, on='subject_id', how='left')
        print(f"\nMerged data: {len(merged_df)} subjects (with aphasia type)")
    else:
        print(f"\nMerged data: {len(merged_df)} subjects")
    
    # Define variable groups
    # Lesion-related variables (7 disconnection + 7 network lesion)
    lesion_vars = [
        {'name': 'af_disconnection', 'label': 'AF Disconnection'},
        {'name': 'ilf_disconnection', 'label': 'ILF Disconnection'},
        {'name': 'uf_disconnection', 'label': 'UF Disconnection'},
        {'name': 'af_ilf_disconnection', 'label': 'AF+ILF Disconnection'},
        {'name': 'af_uf_disconnection', 'label': 'AF+UF Disconnection'},
        {'name': 'ilf_uf_disconnection', 'label': 'ILF+UF Disconnection'},
        {'name': 'all_tracts_disconnection', 'label': 'All Tracts Disconnection'},
        {'name': 'Cont_lesion_pct', 'label': 'Cont Network Lesion'},
        {'name': 'Default_lesion_pct', 'label': 'Default Network Lesion'},
        {'name': 'DorsAttn_lesion_pct', 'label': 'DorsAttn Network Lesion'},
        {'name': 'Limbic_lesion_pct', 'label': 'Limbic Network Lesion'},
        {'name': 'SalVentAttn_lesion_pct', 'label': 'SalVentAttn Network Lesion'},
        {'name': 'SomMot_lesion_pct', 'label': 'SomMot Network Lesion'},
        {'name': 'Vis_lesion_pct', 'label': 'Vis Network Lesion'}
    ]
    
    # AI-related variables
    ai_vars = [
        {'name': 'global_ai', 'label': 'Global AI'},
        {'name': 'dorsal_ai', 'label': 'Dorsal AI'},
        {'name': 'ventral_ai', 'label': 'Ventral AI'},
        {'name': 'Cont_ai', 'label': 'Cont Network AI'},
        {'name': 'Default_ai', 'label': 'Default Network AI'},
        {'name': 'DorsAttn_ai', 'label': 'DorsAttn Network AI'},
        {'name': 'Limbic_ai', 'label': 'Limbic Network AI'},
        {'name': 'SalVentAttn_ai', 'label': 'SalVentAttn Network AI'},
        {'name': 'SomMot_ai', 'label': 'SomMot Network AI'},
        {'name': 'Vis_ai', 'label': 'Vis Network AI'}
    ]
    
    # SFC-related variables
    sfc_vars = [
        {'name': 'global_sfc', 'label': 'Global SFC'},
        {'name': 'dorsal_sfc', 'label': 'Dorsal SFC'},
        {'name': 'ventral_sfc', 'label': 'Ventral SFC'}
    ]
    
    # Clinical scores
    clinical_vars = [
        {'name': 'repetition_subscore', 'label': 'Repetition'},
        {'name': 'comprehension_subscore', 'label': 'Comprehension'},
        {'name': 'naming_subscore', 'label': 'Naming'},
        {'name': 'wab_r_aq', 'label': 'WAB AQ'}
    ]
    
    # 4. Perform 4 groups of correlation analysis
    # Analysis 1: Lesion vs Clinical
    corr1 = perform_correlation_analysis(merged_df, lesion_vars, clinical_vars, 
                                        "Analysis 1: Lesion Variables vs Clinical Scores",
                                        control_tlv=False)
    corr1.to_csv(os.path.join(output_dir, '1_lesion_vs_clinical.csv'), index=False)
    plot_significant_correlations(merged_df, corr1, output_dir, '1_lesion_clinical')
    
    # Analysis 2: Lesion vs AI (with partial correlation controlling for TLV)
    corr2 = perform_correlation_analysis(merged_df, lesion_vars, ai_vars,
                                        "Analysis 2: Lesion Variables vs AI Variables",
                                        control_tlv=True)
    corr2.to_csv(os.path.join(output_dir, '2_lesion_vs_ai_partial.csv'), index=False)
    plot_significant_correlations(merged_df, corr2, output_dir, '2_lesion_ai_partial')
    
    # Analysis 3: Lesion vs SFC
    corr3 = perform_correlation_analysis(merged_df, lesion_vars, sfc_vars,
                                        "Analysis 3: Lesion Variables vs SFC Variables",
                                        control_tlv=False)
    corr3.to_csv(os.path.join(output_dir, '3_lesion_vs_sfc.csv'), index=False)
    plot_significant_correlations(merged_df, corr3, output_dir, '3_lesion_sfc')
    
    # Analysis 4: AI vs Clinical
    corr4 = perform_correlation_analysis(merged_df, ai_vars, clinical_vars,
                                        "Analysis 4: AI Variables vs Clinical Scores",
                                        control_tlv=False)
    corr4.to_csv(os.path.join(output_dir, '4_ai_vs_clinical.csv'), index=False)
    plot_significant_correlations(merged_df, corr4, output_dir, '4_ai_clinical')
    
    # Analysis 5: Mediation Analysis (Lesion -> AI -> Clinical)
    mediation_df = perform_mediation_analysis(merged_df, lesion_vars, ai_vars, clinical_vars, output_dir)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved in: {output_dir}")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  - Analysis 1: Lesion vs Clinical ({len(corr1)} correlations)")
    print(f"  - Analysis 2: Lesion vs AI with TLV control ({len(corr2)} partial correlations)")
    print(f"  - Analysis 3: Lesion vs SFC ({len(corr3)} correlations)")
    print(f"  - Analysis 4: AI vs Clinical ({len(corr4)} correlations)")
    print(f"  - Analysis 5: Mediation Analysis ({len(mediation_df)} models tested)")
    print(f"\nScatter plot data saved in: {os.path.join(output_dir, 'scatter_data')}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

