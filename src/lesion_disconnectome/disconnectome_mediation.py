#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Disconnectome-based Mediation Analysis for Stroke Lesions

This script implements a mediation analysis framework to test the hypothesis that
structural disconnection (X) affects clinical outcomes (Y) through changes in
structure-function coupling (M).

References:
1. Foulon, C., et al. (2018). "Advanced lesion symptom mapping analyses and
   implementation as BCBtoolkit." GigaScience, 7(3), giy004.
2. Thiebaut de Schotten, M., et al. (2020). "The disconnectome symptoms discoverer:
   a map of healthy brain circuits supporting functional brain organization."
   Nature Communications, 11(1), 1-13.
3. Salvalaggio, A., et al. (2020). "Post-stroke deficit prediction from lesion and
   indirect structural and functional disconnection." Brain, 143(7), 2173-2188.

Author: Generated for SC-FC Coupling Analysis
Date: 2026-01-20
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from glob import glob
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# For mediation analysis
try:
    from scipy.stats import bootstrap
except ImportError:
    print("Warning: scipy.stats.bootstrap not available. Using alternative method.")


def load_tract_templates(tract_dir: str) -> Dict[str, np.ndarray]:
    """
    Load HCP-1065 tract probability templates.

    Args:
        tract_dir: Directory containing tract probability NIfTI files

    Returns:
        Dictionary mapping tract names to 3D probability arrays
    """
    tract_files = glob(os.path.join(tract_dir, "*.nii.gz"))
    if not tract_files:
        raise FileNotFoundError(f"No tract files found in {tract_dir}")

    tracts = {}
    print(f"Loading {len(tract_files)} tract templates...")

    for tract_file in tract_files:
        tract_name = os.path.basename(tract_file).replace('.nii.gz', '')
        img = nib.load(tract_file)
        tracts[tract_name] = img.get_fdata()

    print(f"Loaded {len(tracts)} tracts successfully.")
    return tracts


def load_lesion_mask(lesion_file: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load lesion mask from NIfTI file.

    Args:
        lesion_file: Path to lesion NIfTI file

    Returns:
        Tuple of (lesion_data, nifti_image)
    """
    if not os.path.exists(lesion_file):
        raise FileNotFoundError(f"Lesion file not found: {lesion_file}")

    img = nib.load(lesion_file)
    lesion_data = img.get_fdata()

    # Binarize lesion mask
    lesion_data = (lesion_data > 0).astype(np.float32)

    return lesion_data, img


def calculate_disconnectome(lesion_data: np.ndarray,
                           tracts: Dict[str, np.ndarray],
                           threshold: float = 0.1,
                           subject_id: str = None,
                           cache_dir: str = None) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Calculate structural disconnection scores for each tract with caching support.

    This implements the core disconnectome calculation: for each white matter tract,
    we compute the overlap between the lesion and the tract probability map.

    Args:
        lesion_data: Binary lesion mask (3D array)
        tracts: Dictionary of tract probability maps
        threshold: Minimum probability threshold for tract voxels (default: 0.1)
        subject_id: Subject identifier for caching
        cache_dir: Directory to store cached resampled lesions

    Returns:
        Tuple of (disconnection_scores_dict, resampled_lesion_data)
    """
    from scipy.ndimage import zoom

    disconnection_scores = {}

    # Get target shape from first tract
    target_shape = next(iter(tracts.values())).shape

    # Check cache first if subject_id and cache_dir provided
    resampled_lesion = None
    if subject_id and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{subject_id}_resampled_lesion.npy")

        if os.path.exists(cache_file):
            try:
                resampled_lesion = np.load(cache_file)
                if resampled_lesion.shape == target_shape:
                    lesion_data = resampled_lesion
                    print(f"  [Cache] Loaded resampled lesion from cache")
                else:
                    resampled_lesion = None
            except Exception as e:
                print(f"  [Warning] Failed to load cached lesion: {e}")
                resampled_lesion = None

    # Resample lesion data if needed and not loaded from cache
    if resampled_lesion is None and lesion_data.shape != target_shape:
        print(f"  Resampling lesion from {lesion_data.shape} to {target_shape}")
        zoom_factors = [t / l for t, l in zip(target_shape, lesion_data.shape)]
        lesion_data = zoom(lesion_data, zoom_factors, order=0)  # Nearest neighbor for binary mask
        lesion_data = (lesion_data > 0.5).astype(np.float32)  # Re-binarize

        # Save to cache
        if subject_id and cache_dir:
            try:
                np.save(cache_file, lesion_data)
                print(f"  [Cache] Saved resampled lesion to cache")
            except Exception as e:
                print(f"  [Warning] Failed to save cache: {e}")

    for tract_name, tract_data in tracts.items():
        # Threshold tract probability map
        tract_binary = (tract_data >= threshold).astype(np.float32)

        # Calculate overlap between lesion and tract
        overlap = np.sum(lesion_data * tract_binary)
        tract_volume = np.sum(tract_binary)

        if tract_volume > 0:
            # Disconnection score = proportion of tract affected by lesion
            disconnection_scores[tract_name] = overlap / tract_volume
        else:
            disconnection_scores[tract_name] = 0.0

    return disconnection_scores, lesion_data


def load_brain_template(template_file: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load brain parcellation template (e.g., Schaefer 400 + Tian Subcortex).

    Args:
        template_file: Path to template NIfTI file

    Returns:
        Tuple of (template_data, nifti_image)
    """
    if not os.path.exists(template_file):
        raise FileNotFoundError(f"Template file not found: {template_file}")

    img = nib.load(template_file)
    template_data = img.get_fdata()

    return template_data, img


def load_sfc_data(sfc_file: str, roi_number: int = 450) -> np.ndarray:
    """
    Load SFC values from text file.

    Args:
        sfc_file: Path to SFC text file (format: header + tab-separated values)
        roi_number: Expected number of ROIs

    Returns:
        1D array of SFC values
    """
    if not os.path.exists(sfc_file):
        raise FileNotFoundError(f"SFC file not found: {sfc_file}")

    sfc_values = []
    try:
        with open(sfc_file, 'r') as f:
            lines = f.readlines()
            # Skip header lines (first 3 lines: Subject, Method, ROI_Index header, separator)
            for line in lines[3:]:
                clean_line = line.strip()
                if not clean_line or clean_line.startswith('-'):
                    continue
                parts = clean_line.split('\t')
                if len(parts) >= 2:
                    try:
                        sfc_val = float(parts[1])
                        sfc_values.append(sfc_val)
                    except ValueError:
                        continue

        sfc_array = np.array(sfc_values, dtype=float)

        if len(sfc_array) != roi_number:
            print(f"Warning: Expected {roi_number} ROIs, got {len(sfc_array)}")

        return sfc_array
    except Exception as e:
        raise IOError(f"Failed to load SFC file {sfc_file}: {e}")


def extract_language_network_rois(template_info_file: str,
                                  networks: List[str] = None) -> List[int]:
    """
    Extract ROI indices belonging to language-related networks.

    Args:
        template_info_file: Path to template label file
        networks: List of network names (default: language-related networks)

    Returns:
        List of ROI indices (0-based)
    """
    # If template file doesn't exist, return all ROIs (0-449 for 450 ROIs)
    if not os.path.exists(template_info_file):
        print(f"Warning: Template info file not found: {template_info_file}")
        print("Using all ROIs for analysis")
        return list(range(450))  # Default to all 450 ROIs

    if networks is None:
        # Default language-related networks (Yeo 7-network parcellation)
        networks = ['DefaultMode', 'Ventral Attention', 'Dorsal Attention']

    # Load template info
    template_df = pd.read_csv(template_info_file, sep='\t')

    # Extract ROIs belonging to specified networks
    language_rois = []
    for network in networks:
        if 'network' in template_df.columns:
            network_rois = template_df[template_df['network'].str.contains(network, case=False, na=False)]
            language_rois.extend(network_rois.index.tolist())

    if not language_rois:
        print("Warning: No language network ROIs found, using all ROIs")
        return list(range(450))

    return sorted(list(set(language_rois)))


def extract_network_rois(template_info_file: str, network_name: str) -> List[int]:
    """
    Extract ROI indices for dorsal or ventral language pathways dynamically from template file.

    Based on dual-stream model of language processing:
    - Dorsal stream: Sound-to-Articulation (phonology to motor)
    - Ventral stream: Sound-to-Meaning (phonology to semantics)

    Args:
        template_info_file: Path to template label file
        network_name: 'DorsAttn' for dorsal stream or 'SalVentAttn' for ventral stream

    Returns:
        List of ROI indices (0-based)
    """
    if not os.path.exists(template_info_file):
        raise FileNotFoundError(f"Template info file not found: {template_info_file}. Cannot proceed without template file.")

    try:
        # Read the label file
        with open(template_info_file, 'r') as f:
            lines = f.readlines()

        network_rois = []

        if network_name == 'DorsAttn':
            # Dorsal stream components - dynamically match based on Yeo network labels
            target_patterns = ['DorsAttn', 'PUT-VA-lh', 'PUT-DA-lh', 'PUT-VP-lh', 'PUT-DP-lh',
                             'CAU-VA-lh', 'CAU-DA-lh', 'CAU-body-lh', 'CAU-tail-lh',
                             'SalVentAttn_FrOperIns', 'SalVentAttn_ParOper', 'SalVentAttn_TempOcc',
                             'DorsAttn_PrCv']
        elif network_name == 'SalVentAttn':
            # Ventral stream components - dynamically match based on Yeo network labels
            target_patterns = ['HIP-head-lh', 'HIP-body-lh', 'HIP-tail-lh',
                             'THA-VPm-lh', 'THA-VPl-lh', 'THA-VAi-lh', 'THA-VAs-lh',
                             'THA-DAm-lh', 'THA-DAl-lh', 'THA-DP-lh',
                             'Limbic_TempPole', 'Limbic_OFC',
                             'Default_Temp', 'Default_Par']
        else:
            raise ValueError(f"Unknown network name '{network_name}'. Must be 'DorsAttn' or 'SalVentAttn'.")

        for idx, line in enumerate(lines):
            line = line.strip()
            # Check if any target pattern is in this line
            for pattern in target_patterns:
                if (pattern in line and '_LH_' in line) or (pattern in line and '-lh' in line):
                    network_rois.append(idx)
                    break

        if not network_rois:
            raise ValueError(f"No ROIs found for {network_name} network. Check template file format.")

        print(f"[OK] Dynamically extracted {len(network_rois)} ROIs for {network_name} network")
        return network_rois

    except Exception as e:
        raise RuntimeError(f"Failed to extract {network_name} ROIs: {e}")


def calculate_global_disconnection(disconnection_scores: Dict[str, float]) -> float:
    """
    Calculate overall disconnection score across all tracts (global disconnection).

    Args:
        disconnection_scores: Dictionary of tract disconnection scores

    Returns:
        Mean disconnection score across all tracts
    """
    if not disconnection_scores:
        print("Warning: No disconnection scores available")
        return 0.0

    # Calculate mean disconnection across ALL tracts
    scores = list(disconnection_scores.values())

    return np.mean(scores)


def calculate_individual_tract_disconnection(disconnection_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate disconnection scores for individual white matter tracts (AF, UF, ILF).

    Args:
        disconnection_scores: Dictionary of tract disconnection scores

    Returns:
        Dictionary with separate scores for AF, ILF, UF, and all their combinations (7 total)
    """
    if not disconnection_scores:
        print("Warning: No disconnection scores available")
        return {
            'af_disconnection': 0.0,
            'ilf_disconnection': 0.0,
            'uf_disconnection': 0.0,
            'af_ilf_disconnection': 0.0,
            'af_uf_disconnection': 0.0,
            'ilf_uf_disconnection': 0.0,
            'all_tracts_disconnection': 0.0
        }

    # Define tract patterns for matching
    af_patterns = ['af', 'arcuate']
    ilf_patterns = ['ilf', 'inferior_longitudinal', 'inferior longitudinal']
    uf_patterns = ['uf', 'uncinate']

    # Find matching tracts for each type
    af_scores = []
    ilf_scores = []
    uf_scores = []

    for tract_key, score in disconnection_scores.items():
        tract_key_lower = tract_key.lower()

        # Check for AF
        if any(pattern in tract_key_lower for pattern in af_patterns):
            af_scores.append(score)

        # Check for ILF
        if any(pattern in tract_key_lower for pattern in ilf_patterns):
            ilf_scores.append(score)

        # Check for UF
        if any(pattern in tract_key_lower for pattern in uf_patterns):
            uf_scores.append(score)

    # Calculate individual tract scores
    af_disconnection = np.mean(af_scores) if af_scores else 0.0
    ilf_disconnection = np.mean(ilf_scores) if ilf_scores else 0.0
    uf_disconnection = np.mean(uf_scores) if uf_scores else 0.0

    # Calculate all 2-way combinations
    af_ilf_scores = af_scores + ilf_scores
    af_ilf_disconnection = np.mean(af_ilf_scores) if af_ilf_scores else 0.0

    af_uf_scores = af_scores + uf_scores
    af_uf_disconnection = np.mean(af_uf_scores) if af_uf_scores else 0.0

    ilf_uf_scores = ilf_scores + uf_scores
    ilf_uf_disconnection = np.mean(ilf_uf_scores) if ilf_uf_scores else 0.0

    # Calculate all 3 tracts combined
    all_scores = af_scores + ilf_scores + uf_scores
    all_tracts_disconnection = np.mean(all_scores) if all_scores else 0.0

    print(f"[OK] Found AF: {len(af_scores)}, ILF: {len(ilf_scores)}, UF: {len(uf_scores)} tracts")

    return {
        'af_disconnection': af_disconnection,
        'ilf_disconnection': ilf_disconnection,
        'uf_disconnection': uf_disconnection,
        'af_ilf_disconnection': af_ilf_disconnection,
        'af_uf_disconnection': af_uf_disconnection,
        'ilf_uf_disconnection': ilf_uf_disconnection,
        'all_tracts_disconnection': all_tracts_disconnection
    }


def calculate_language_network_sfc(sfc_values: np.ndarray,
                                   language_roi_indices: List[int]) -> float:
    """
    Calculate mean SFC for language network ROIs only.

    Args:
        sfc_values: Array of SFC values for all ROIs
        language_roi_indices: List of ROI indices belonging to language network (0-based)

    Returns:
        Mean SFC across language network ROIs
    """
    if len(sfc_values) == 0 or len(language_roi_indices) == 0:
        print("Warning: No SFC values or language ROI indices available")
        return np.nan

    # Extract SFC values for language network ROIs
    language_sfc_values = []
    for roi_idx in language_roi_indices:
        if roi_idx < len(sfc_values) and not np.isnan(sfc_values[roi_idx]):
            language_sfc_values.append(sfc_values[roi_idx])

    if len(language_sfc_values) == 0:
        return np.nan

    return np.mean(language_sfc_values)


def calculate_global_sfc(sfc_values: np.ndarray) -> float:
    """
    Calculate mean SFC across all ROIs (global SFC).

    Args:
        sfc_values: Array of SFC values for all ROIs

    Returns:
        Mean SFC across all ROIs
    """
    if len(sfc_values) == 0:
        print("Warning: No SFC values available")
        return np.nan

    # Calculate mean SFC across ALL ROIs
    return np.mean(sfc_values)


def mediation_analysis(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                      n_bootstrap: int = 5000) -> Dict[str, float]:
    """
    Perform mediation analysis using bootstrap method.

    Tests the hypothesis: X -> M -> Y
    Where:
        X = Independent variable (e.g., structural disconnection)
        M = Mediator (e.g., SFC)
        Y = Dependent variable (e.g., clinical score)

    Returns:
        Dictionary containing:
        - total_effect: Total effect of X on Y (c path)
        - direct_effect: Direct effect of X on Y controlling for M (c' path)
        - indirect_effect: Indirect effect through M (a*b path)
        - proportion_mediated: Proportion of total effect mediated
        - p_value_indirect: Bootstrap p-value for indirect effect
    """
    from scipy.stats import linregress

    # Standardize variables
    X = (X - np.mean(X)) / np.std(X)
    M = (M - np.mean(M)) / np.std(M)
    Y = (Y - np.mean(Y)) / np.std(Y)

    # Path c: Total effect (X -> Y)
    slope_c, intercept_c, r_c, p_c, se_c = linregress(X, Y)
    total_effect = slope_c

    # Path a: X -> M
    slope_a, intercept_a, r_a, p_a, se_a = linregress(X, M)

    # Path b: M -> Y (controlling for X)
    # Use multiple regression: Y ~ X + M
    from sklearn.linear_model import LinearRegression
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
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[indices]
        M_boot = M[indices]
        Y_boot = Y[indices]

        # Calculate indirect effect for bootstrap sample
        slope_a_boot, _, _, _, _ = linregress(X_boot, M_boot)
        model_boot = LinearRegression()
        model_boot.fit(np.column_stack([X_boot, M_boot]), Y_boot)
        b_path_boot = model_boot.coef_[1]
        indirect_effects.append(slope_a_boot * b_path_boot)

    indirect_effects = np.array(indirect_effects)

    # Calculate p-value (proportion of bootstrap samples with opposite sign)
    if indirect_effect > 0:
        p_value_indirect = np.mean(indirect_effects <= 0)
    else:
        p_value_indirect = np.mean(indirect_effects >= 0)

    # Two-tailed p-value
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


def plot_mediation_model(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                        results: Dict[str, float],
                        output_file: str,
                        x_label: str = "Disconnection",
                        m_label: str = "SFC",
                        y_label: str = "Clinical Score"):
    """
    Create visualization of mediation analysis results.

    Args:
        X, M, Y: Data arrays
        results: Dictionary from mediation_analysis()
        output_file: Path to save figure
        x_label, m_label, y_label: Axis labels
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: X -> Y (total effect)
    axes[0, 0].scatter(X, Y, alpha=0.6)
    axes[0, 0].plot(X, results['total_effect'] * X + np.mean(Y), 'r-', linewidth=2)
    axes[0, 0].set_xlabel(x_label)
    axes[0, 0].set_ylabel(y_label)
    axes[0, 0].set_title(f'Total Effect (c path)\nβ = {results["total_effect"]:.3f}')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: X -> M (path a)
    axes[0, 1].scatter(X, M, alpha=0.6, color='green')
    axes[0, 1].plot(X, results['path_a'] * X + np.mean(M), 'g-', linewidth=2)
    axes[0, 1].set_xlabel(x_label)
    axes[0, 1].set_ylabel(m_label)
    axes[0, 1].set_title(f'Path a (X → M)\nβ = {results["path_a"]:.3f}, p = {results["p_value_a"]:.4f}')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: M -> Y (path b, controlling for X)
    axes[1, 0].scatter(M, Y, alpha=0.6, color='orange')
    axes[1, 0].plot(M, results['path_b'] * M + np.mean(Y), 'orange', linewidth=2)
    axes[1, 0].set_xlabel(m_label)
    axes[1, 0].set_ylabel(y_label)
    axes[1, 0].set_title(f'Path b (M → Y | X)\nβ = {results["path_b"]:.3f}')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Mediation summary
    axes[1, 1].axis('off')
    summary_text = f"""
    Mediation Analysis Summary
    ═══════════════════════════

    Total Effect (c):        {results['total_effect']:.4f}
    Direct Effect (c'):      {results['direct_effect']:.4f}
    Indirect Effect (a×b):   {results['indirect_effect']:.4f}

    95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]
    p-value (bootstrap):     {results['p_value_indirect']:.4f}

    Proportion Mediated:     {results['proportion_mediated']:.2%}

    {'[OK] Significant mediation' if results['p_value_indirect'] < 0.05 else '[X] Non-significant mediation'}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mediation plot saved to: {output_file}")


def normalize_subject_identifier(value):
    """
    Normalize subject ID by extracting numeric part.
    Handles formats like: 'sub-M2748', 'M2748', '2748'
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value_str = str(value).strip()

    # Remove common prefixes
    for prefix in ['sub-M', 'sub-', 'M']:
        if value_str.startswith(prefix):
            value_str = value_str[len(prefix):]
            break

    # Extract digits only
    digits = ''.join(filter(str.isdigit, value_str))
    return digits or value_str


def load_fallback_wab_scores(participant_file_path: str) -> Dict[str, float]:
    """
    Load fallback WAB scores from participants_aphasia file.

    Args:
        participant_file_path: Path to participants CSV/TSV file

    Returns:
        Dictionary mapping subject numeric IDs to WAB AQ scores
    """
    fallback_scores = {}
    if not participant_file_path or not os.path.exists(participant_file_path):
        print("[Warning] No valid participants file path provided, cannot load fallback WAB scores")
        return fallback_scores

    try:
        # Try comma separator first, then tab
        try:
            participants_df = pd.read_csv(participant_file_path)
        except:
            participants_df = pd.read_csv(participant_file_path, sep='\t')

        print(f"[OK] Loaded participants file: {len(participants_df)} records")
    except Exception as e:
        print(f"[Warning] Failed to load participants file: {e}")
        return fallback_scores

    if 'participant_id' not in participants_df.columns or 'wab_aq' not in participants_df.columns:
        print("[Warning] Participants file missing participant_id or wab_aq columns")
        return fallback_scores

    for _, row in participants_df.iterrows():
        participant_id = str(row['participant_id'])
        wab_aq = row['wab_aq']
        if pd.isna(wab_aq):
            continue

        numeric_part = ''.join(filter(str.isdigit, participant_id))
        if numeric_part:
            fallback_scores[numeric_part] = wab_aq

    print(f"[OK] Loaded {len(fallback_scores)} fallback WAB AQ scores")
    return fallback_scores


def load_clinical_data_with_multiple_scores(clinical_file: str,
                                            score_columns: List[str] = None,
                                            fallback_file: str = None) -> Tuple[pd.DataFrame, str, List[str], Dict[str, float]]:
    """
    Load clinical data with multiple score columns and optional fallback scores.

    Args:
        clinical_file: Path to clinical CSV file
        score_columns: List of score column names to load
        fallback_file: Path to fallback participants file for additional WAB scores

    Returns:
        Tuple of (clinical_dataframe, subject_id_column, list_of_score_columns, fallback_scores_dict)
    """
    # Load fallback scores if provided
    fallback_scores = {}
    if fallback_file:
        fallback_scores = load_fallback_wab_scores(fallback_file)

    try:
        clinical_data = pd.read_csv(clinical_file)
        print(f"[OK] Loaded clinical data: {len(clinical_data)} records")
        print(f"   Available columns: {clinical_data.columns.tolist()}")
    except Exception as e:
        raise FileNotFoundError(f"Failed to load clinical file: {e}")

    # Find subject_id column
    subject_id_col = None
    for col in clinical_data.columns:
        if col.lower() == 'subject_id':  # Exact match first
            subject_id_col = col
            break

    if subject_id_col is None:
        # Fallback to partial match
        for col in clinical_data.columns:
            if 'subject_id' in col.lower() or 'participant_id' in col.lower():
                subject_id_col = col
                break

    if subject_id_col is None:
        raise ValueError("Cannot find subject_id column in clinical data")

    # Default score columns if not specified
    if score_columns is None:
        score_columns = [
            "repetition_subscore",
            "comprehension_subscore",
            "naming_subscore",
            "wab_r_aq"
        ]

    # Verify all score columns exist
    missing_cols = [col for col in score_columns if col not in clinical_data.columns]
    if missing_cols:
        print(f"[Warning] Missing score columns: {missing_cols}")
        score_columns = [col for col in score_columns if col in clinical_data.columns]

    if not score_columns:
        raise ValueError("No valid score columns found in clinical data")

    print(f"[OK] Using score columns: {score_columns}")

    # Check each score column validity
    for score_col in score_columns:
        score_data = clinical_data[score_col].dropna()
        print(f"[OK] {score_col}: {len(score_data)} valid values, "
              f"Range: {score_data.min():.1f} - {score_data.max():.1f}, "
              f"Mean: {score_data.mean():.1f}")

    return clinical_data, subject_id_col, score_columns, fallback_scores


def match_subject_clinical_scores(subject_id: str,
                                  clinical_data: pd.DataFrame,
                                  subject_id_col: str,
                                  score_cols: List[str],
                                  fallback_scores: Dict[str, float] = None) -> Optional[Dict[str, float]]:
    """
    Match subject ID with multiple clinical scores using flexible ID matching.
    Uses fallback scores if primary data is missing.

    Args:
        subject_id: Subject identifier (e.g., 'M2748', '2748')
        clinical_data: Clinical dataframe
        subject_id_col: Name of subject ID column
        score_cols: List of score column names
        fallback_scores: Dictionary of fallback WAB AQ scores

    Returns:
        Dictionary of {score_name: score_value} or None if not found
    """
    # Normalize the query subject ID
    normalized_query = normalize_subject_identifier(subject_id)

    for _, row in clinical_data.iterrows():
        clinical_subject_id = str(row[subject_id_col]).strip()
        normalized_clinical = normalize_subject_identifier(clinical_subject_id)

        if normalized_clinical == normalized_query:
            scores = {}
            for score_col in score_cols:
                score_value = row[score_col] if not pd.isna(row[score_col]) else None

                # Use fallback for wab_r_aq if missing and fallback available
                if score_value is None and score_col == 'wab_r_aq' and fallback_scores:
                    if normalized_query in fallback_scores:
                        score_value = fallback_scores[normalized_query]
                        print(f"    [Fallback] Using fallback WAB AQ for subject {subject_id}: {score_value}")

                scores[score_col] = score_value
            return scores

    # If not found in clinical data, try fallback for wab_r_aq only
    if fallback_scores and normalized_query in fallback_scores:
        scores = {}
        for score_col in score_cols:
            if score_col == 'wab_r_aq':
                scores[score_col] = fallback_scores[normalized_query]
                print(f"    [Fallback] Using fallback-only WAB AQ for subject {subject_id}: {scores[score_col]}")
            else:
                scores[score_col] = None
        return scores

    return None


def parse_network_from_template(template_info_file: str) -> Dict[str, List[int]]:
    """
    Parse template file to extract network information for each ROI.
    
    Args:
        template_info_file: Path to template label file
        
    Returns:
        Dictionary mapping network names to lists of ROI indices (0-based)
    """
    import re
    
    if not os.path.exists(template_info_file):
        print(f"Warning: Template file not found: {template_info_file}")
        return {}
    
    network_roi_map = {
        'Cont': [],
        'Default': [],
        'DorsAttn': [],
        'Limbic': [],
        'SalVentAttn': [],
        'SomMot': [],
        'Vis': [],
        'Subcortical': []
    }
    
    try:
        with open(template_info_file, 'r') as f:
            lines = f.readlines()
        
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Determine network from line content
            network = 'Subcortical'  # Default
            
            if '7Networks' in line:
                # Extract network name from pattern like "7Networks_LH_Cont_"
                match = re.search(r'7Networks_[LR]H_([A-Za-z]+)', line)
                if match:
                    network = match.group(1)
            
            # Add ROI index to corresponding network
            if network in network_roi_map:
                network_roi_map[network].append(idx)
        
        print(f"Parsed network ROI map:")
        for network, rois in network_roi_map.items():
            print(f"  {network}: {len(rois)} ROIs")
        
        return network_roi_map
        
    except Exception as e:
        print(f"Error parsing template file: {e}")
        return {}


def calculate_network_lesion_percentages(lesion_data: np.ndarray,
                                         atlas_img: nib.Nifti1Image,
                                         network_roi_map: Dict[str, List[int]],
                                         networks: List[str]) -> Dict[str, float]:
    """
    Calculate lesion percentage for each network.
    
    Args:
        lesion_data: Lesion mask array
        atlas_img: Atlas NIfTI image
        network_roi_map: Dictionary mapping network names to ROI indices
        networks: List of network names to calculate
        
    Returns:
        Dictionary mapping network names to lesion percentages
    """
    from scipy.ndimage import zoom
    
    # Load atlas data
    atlas_data = atlas_img.get_fdata().astype(int)
    
    # Resample lesion to atlas space if needed
    if lesion_data.shape != atlas_data.shape:
        zoom_factors = [a / l for a, l in zip(atlas_data.shape, lesion_data.shape)]
        lesion_data = zoom(lesion_data, zoom_factors, order=0)
        lesion_data = (lesion_data > 0.5).astype(np.float32)
    
    network_lesion_pcts = {}
    
    for network in networks:
        if network not in network_roi_map:
            network_lesion_pcts[network] = 0.0
            continue
        
        roi_indices = network_roi_map[network]
        if not roi_indices:
            network_lesion_pcts[network] = 0.0
            continue
        
        # Create mask for this network's ROIs
        network_mask = np.zeros_like(atlas_data, dtype=bool)
        for roi_idx in roi_indices:
            # Atlas labels are 1-based, roi_idx is 0-based
            atlas_label = roi_idx + 1
            network_mask |= (atlas_data == atlas_label)
        
        # Calculate total voxels in this network
        network_total_voxels = np.sum(network_mask)
        
        if network_total_voxels == 0:
            network_lesion_pcts[network] = 0.0
            continue
        
        # Calculate lesion voxels in this network
        lesion_in_network = np.sum(lesion_data & network_mask)
        
        # Calculate percentage
        lesion_pct = lesion_in_network / network_total_voxels
        network_lesion_pcts[network] = lesion_pct
    
    return network_lesion_pcts


def process_single_subject(subject_id: str,
                          lesion_dir: str,
                          sfc_dir: str,
                          tract_templates: Dict[str, np.ndarray],
                          template_info_file: str,
                          roi_number: int = 450,
                          dorsal_roi_indices: List[int] = None,
                          ventral_roi_indices: List[int] = None,
                          cache_dir: str = None) -> Dict[str, float]:
    """
    Process a single subject to extract disconnection and SFC measures.

    Args:
        subject_id: Subject identifier (e.g., '2748' or 'M2748')
        lesion_dir: Directory containing region masks (extracted_regions)
        sfc_dir: Directory containing SFC files
        tract_templates: Dictionary of tract probability maps
        template_info_file: Path to template label file
        roi_number: Number of ROIs in template
        dorsal_roi_indices: List of ROI indices for dorsal network (0-based)
        ventral_roi_indices: List of ROI indices for ventral network (0-based)
        cache_dir: Directory to store cached resampled lesions

    Returns:
        Dictionary with disconnection scores and network SFC values
    """
    # Normalize subject ID to match directory structure
    # Try both with and without 'M' prefix
    possible_ids = []
    if subject_id.startswith('M'):
        possible_ids.append(subject_id)
        possible_ids.append(subject_id[1:])  # Remove 'M'
    else:
        possible_ids.append(f'M{subject_id}')
        possible_ids.append(subject_id)

    # Find lesion data from all_regions.npy file
    lesion_data = None
    for test_id in possible_ids:
        subject_dir = os.path.join(lesion_dir, test_id)
        all_regions_file = os.path.join(subject_dir, f"{test_id}_all_regions.npy")

        if os.path.exists(all_regions_file):
            try:
                region_masks = np.load(all_regions_file, allow_pickle=True).item()
                lesion_data = region_masks.get('lesion')
                if lesion_data is not None:
                    break
            except Exception as e:
                print(f"Warning: Failed to load {all_regions_file}: {e}")
                continue

    if lesion_data is None:
        print(f"Warning: No lesion data found for {subject_id}")
        return None

    # Find SFC file
    sfc_pattern = os.path.join(sfc_dir, f"*{subject_id}*sfc*.txt")
    sfc_files = glob(sfc_pattern)

    if not sfc_files:
        print(f"Warning: No SFC file found for {subject_id}")
        return None

    sfc_file = sfc_files[0]

    # Load SFC data
    sfc_values = load_sfc_data(sfc_file, roi_number)

    # Calculate disconnection scores with caching
    disconnection_scores, resampled_lesion = calculate_disconnectome(
        lesion_data, tract_templates, subject_id=subject_id, cache_dir=cache_dir
    )
    global_disconnection = calculate_global_disconnection(disconnection_scores)

    # Calculate individual tract disconnection scores (AF, ILF, UF separately)
    tract_scores = calculate_individual_tract_disconnection(disconnection_scores)

    # Calculate global SFC
    global_sfc = calculate_global_sfc(sfc_values)

    # Calculate dorsal network SFC
    dorsal_sfc = np.nan
    if dorsal_roi_indices is not None and len(dorsal_roi_indices) > 0:
        dorsal_sfc = calculate_language_network_sfc(sfc_values, dorsal_roi_indices)

    # Calculate ventral network SFC
    ventral_sfc = np.nan
    if ventral_roi_indices is not None and len(ventral_roi_indices) > 0:
        ventral_sfc = calculate_language_network_sfc(sfc_values, ventral_roi_indices)

    result = {
        'subject_id': subject_id,
        'global_disconnection': global_disconnection,
        'af_disconnection': tract_scores['af_disconnection'],
        'ilf_disconnection': tract_scores['ilf_disconnection'],
        'uf_disconnection': tract_scores['uf_disconnection'],
        'af_ilf_disconnection': tract_scores['af_ilf_disconnection'],
        'af_uf_disconnection': tract_scores['af_uf_disconnection'],
        'ilf_uf_disconnection': tract_scores['ilf_uf_disconnection'],
        'all_tracts_disconnection': tract_scores['all_tracts_disconnection'],
        'global_sfc': global_sfc,
        'dorsal_sfc': dorsal_sfc,
        'ventral_sfc': ventral_sfc,
        'lesion_volume': np.sum(resampled_lesion)
    }

    return result


def main():
    """
    Main execution function for disconnectome-based mediation analysis.
    Performs mediation analysis for 4 clinical scores with tract-specific disconnection.
    """
    # ========== 配置参数 (直接修改这里的路径) ==========
    LESION_DIR = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\extracted_regions"
    SFC_DIR = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\batch_results_pca_450_t288\sfc_txt_files"
    TRACT_DIR = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\hcp1065_prob_coverage_nifti\prob"
    TEMPLATE_INFO = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
    CLINICAL_FILE = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\matched_subjects_data.csv"
    FALLBACK_FILE = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\aphasia_plots\participants_new.csv"
    OUTPUT_DIR = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\disconnectome_mediation_results"
    CACHE_DIR = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\lesion_cache"
    ROI_NUMBER = 450
    TRACT_THRESHOLD = 0.1
    N_BOOTSTRAP = 5000
    # ==================================================

    # Create output and cache directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 80)
    print("Disconnectome-based Mediation Analysis (4 Scores)")
    print("=" * 80)
    print(f"Lesion directory: {LESION_DIR}")
    print(f"SFC directory: {SFC_DIR}")
    print(f"Tract directory: {TRACT_DIR}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    # Define the 4 score columns
    score_columns = [
        "repetition_subscore",
        "comprehension_subscore",
        "naming_subscore",
        "wab_r_aq"
    ]

    # Step 1: Load tract templates
    print("\n[Step 1/7] Loading tract templates...")
    tract_templates = load_tract_templates(TRACT_DIR)

    # Step 2: Load clinical data with multiple scores
    print("\n[Step 2/7] Loading clinical data with multiple scores...")
    clinical_data, subject_id_col, score_cols, fallback_scores = load_clinical_data_with_multiple_scores(
        CLINICAL_FILE,
        score_columns,
        FALLBACK_FILE
    )
    print(f"Using columns: {subject_id_col} (subject ID)")
    print(f"Score columns: {score_cols}")

    # Step 3: Extract Dorsal and Ventral network ROI indices
    print("\n[Step 3/7] Extracting Dorsal and Ventral network ROI indices...")
    dorsal_roi_indices = extract_network_rois(TEMPLATE_INFO, 'DorsAttn')
    ventral_roi_indices = extract_network_rois(TEMPLATE_INFO, 'SalVentAttn')
    print(f"Dorsal network ROIs: {len(dorsal_roi_indices)} ROIs")
    print(f"Ventral network ROIs: {len(ventral_roi_indices)} ROIs")

    # Step 4: Extract subject IDs from SFC files
    print("\n[Step 4/7] Identifying subjects from SFC files...")
    sfc_files = glob(os.path.join(SFC_DIR, "*sfc*.txt"))
    print(f"Found {len(sfc_files)} SFC files")

    import re
    subject_ids = []
    for sfc_file in sfc_files:
        filename = os.path.basename(sfc_file)
        match = re.search(r'(\d+)', filename)
        if match:
            subject_ids.append(match.group(1))

    print(f"Extracted {len(subject_ids)} subject IDs from SFC files")

    # Step 5: Process all subjects
    print("\n[Step 5/7] Processing subjects...")
    results_list = []

    for idx, subject_id in enumerate(subject_ids):
        print(f"  Processing subject {subject_id} ({idx+1}/{len(subject_ids)})...")

        # Match clinical scores for all 4 score columns
        clinical_scores = match_subject_clinical_scores(
            subject_id, clinical_data, subject_id_col, score_cols, fallback_scores
        )

        if clinical_scores is None:
            print(f"    [Warning] No clinical scores found for subject {subject_id}, skipping")
            continue

        # Check if at least one score is valid
        valid_scores = {k: v for k, v in clinical_scores.items() if v is not None and not pd.isna(v)}
        if not valid_scores:
            print(f"    [Warning] All scores are missing for subject {subject_id}, skipping")
            continue

        result = process_single_subject(
            subject_id=subject_id,
            lesion_dir=LESION_DIR,
            sfc_dir=SFC_DIR,
            tract_templates=tract_templates,
            template_info_file=TEMPLATE_INFO,
            roi_number=ROI_NUMBER,
            dorsal_roi_indices=dorsal_roi_indices,
            ventral_roi_indices=ventral_roi_indices,
            cache_dir=CACHE_DIR
        )

        if result is not None:
            # Add all clinical scores to result
            for score_name, score_value in clinical_scores.items():
                result[score_name] = score_value
            results_list.append(result)
            print(f"    [OK] Processed successfully")

    # Create results dataframe
    results_df = pd.DataFrame(results_list)
    print(f"\nSuccessfully processed {len(results_df)} subjects")

    # Save intermediate results
    results_file = os.path.join(OUTPUT_DIR, 'subject_measures_all_scores.csv')
    results_df.to_csv(results_file, index=False)
    print(f"Subject measures saved to: {results_file}")

    # # Step 6: Perform comprehensive mediation analysis (3 mediators × 7 X variables × 4 Y variables = 84 models)
    # print("\n[Step 6/6] Performing comprehensive mediation analysis (84 models)...")
    # print("  3 Mediators: Dorsal SFC, Ventral SFC, Global SFC")
    # print("  7 X Variables: AF, ILF, UF, AF+ILF, AF+UF, ILF+UF, All Tracts")
    # print("  4 Y Variables: Repetition, Comprehension, Naming, WAB AQ")

    # all_mediation_results = []

    # # Define all combinations
    # mediator_vars = [
    #     {'name': 'dorsal_sfc', 'label': 'Dorsal SFC'},
    #     {'name': 'ventral_sfc', 'label': 'Ventral SFC'},
    #     {'name': 'global_sfc', 'label': 'Global SFC'}
    # ]

    # x_vars = [
    #     {'name': 'af_disconnection', 'label': 'AF'},
    #     {'name': 'ilf_disconnection', 'label': 'ILF'},
    #     {'name': 'uf_disconnection', 'label': 'UF'},
    #     {'name': 'af_ilf_disconnection', 'label': 'AF+ILF'},
    #     {'name': 'af_uf_disconnection', 'label': 'AF+UF'},
    #     {'name': 'ilf_uf_disconnection', 'label': 'ILF+UF'},
    #     {'name': 'all_tracts_disconnection', 'label': 'All Tracts'}
    # ]

    # y_vars = [
    #     {'name': 'repetition_subscore', 'label': 'Repetition'},
    #     {'name': 'comprehension_subscore', 'label': 'Comprehension'},
    #     {'name': 'naming_subscore', 'label': 'Naming'},
    #     {'name': 'wab_r_aq', 'label': 'WAB AQ'}
    # ]

    # total_models = len(mediator_vars) * len(x_vars) * len(y_vars)
    # current_model = 0

    # # Loop through all combinations
    # for mediator in mediator_vars:
    #     for x_var in x_vars:
    #         for y_var in y_vars:
    #             current_model += 1

    #             model_name = f"{y_var['label']}_{x_var['label']}_{mediator['label']}"
    #             model_name = model_name.replace(' ', '_').replace('+', '')

    #             print(f"\n  [{current_model}/{total_models}] X={x_var['label']} -> M={mediator['label']} -> Y={y_var['label']}")

    #             # Check if columns exist
    #             if y_var['name'] not in results_df.columns:
    #                 print(f"    [Skip] Y variable '{y_var['name']}' not found")
    #                 continue

    #             # Extract data
    #             X = results_df[x_var['name']].values
    #             M = results_df[mediator['name']].values
    #             Y = results_df[y_var['name']].values

    #             # Remove subjects with missing data
    #             valid_mask = ~(np.isnan(X) | np.isnan(M) | np.isnan(Y))
    #             X_valid = X[valid_mask]
    #             M_valid = M[valid_mask]
    #             Y_valid = Y[valid_mask]

    #             if len(X_valid) < 3:
    #                 print(f"    [Skip] Insufficient data (n={len(X_valid)})")
    #                 continue

    #             # Perform mediation analysis
    #             try:
    #                 mediation_results = mediation_analysis(X_valid, M_valid, Y_valid, n_bootstrap=N_BOOTSTRAP)
    #                 mediation_results['model_name'] = model_name
    #                 mediation_results['X_variable'] = x_var['name']
    #                 mediation_results['X_label'] = x_var['label']
    #                 mediation_results['M_variable'] = mediator['name']
    #                 mediation_results['M_label'] = mediator['label']
    #                 mediation_results['Y_variable'] = y_var['name']
    #                 mediation_results['Y_label'] = y_var['label']
    #                 mediation_results['n_subjects'] = len(X_valid)
    #                 all_mediation_results.append(mediation_results)

    #                 sig_marker = "[SIG]" if mediation_results['p_value_indirect'] < 0.05 else ""
    #                 print(f"    n={len(X_valid)}, indirect={mediation_results['indirect_effect']:.4f}, p={mediation_results['p_value_indirect']:.4f} {sig_marker}")
    #             except Exception as e:
    #                 print(f"    [Error] {str(e)}")
    #                 continue

    # # Save combined results to single CSV
    # if all_mediation_results:
    #     combined_mediation_df = pd.DataFrame(all_mediation_results)

    #     # Reorder columns for better readability
    #     column_order = [
    #         'model_name', 'X_variable', 'X_label', 'M_variable', 'M_label',
    #         'Y_variable', 'Y_label', 'n_subjects',
    #         'total_effect', 'direct_effect', 'indirect_effect',
    #         'proportion_mediated', 'p_value_indirect',
    #         'ci_lower', 'ci_upper', 'path_a', 'path_b', 'p_value_a'
    #     ]
    #     combined_mediation_df = combined_mediation_df[column_order]

    #     combined_file = os.path.join(OUTPUT_DIR, 'mediation_results_comprehensive_84models.csv')
    #     combined_mediation_df.to_csv(combined_file, index=False)
    #     print(f"\n{'='*80}")
    #     print(f"All {len(all_mediation_results)} mediation results saved to:")
    #     print(f"  {combined_file}")
    #     print(f"{'='*80}")

    # Step 7: Perform correlation analysis between all X/M variables and Y variables
    print("\n[Step 7/7] Performing correlation analysis...")
    print("  Independent variables (X + M): 10 variables (7 disconnection + 3 SFC)")
    print("  Dependent variables (Y): 4 clinical scores")
    print("=" * 80)

    # Define all independent variables (X + M)
    independent_vars = [
        {'name': 'af_disconnection', 'label': 'AF Disconnection'},
        {'name': 'ilf_disconnection', 'label': 'ILF Disconnection'},
        {'name': 'uf_disconnection', 'label': 'UF Disconnection'},
        {'name': 'af_ilf_disconnection', 'label': 'AF+ILF Disconnection'},
        {'name': 'af_uf_disconnection', 'label': 'AF+UF Disconnection'},
        {'name': 'ilf_uf_disconnection', 'label': 'ILF+UF Disconnection'},
        {'name': 'all_tracts_disconnection', 'label': 'All Tracts Disconnection'},
        {'name': 'dorsal_sfc', 'label': 'Dorsal SFC'},
        {'name': 'ventral_sfc', 'label': 'Ventral SFC'},
        {'name': 'global_sfc', 'label': 'Global SFC'}
    ]

    # Y variables (same as before)
    dependent_vars = [
        {'name': 'repetition_subscore', 'label': 'Repetition'},
        {'name': 'comprehension_subscore', 'label': 'Comprehension'},
        {'name': 'naming_subscore', 'label': 'Naming'},
        {'name': 'wab_r_aq', 'label': 'WAB AQ'}
    ]

    correlation_results = []

    for indep_var in independent_vars:
        for dep_var in dependent_vars:
            # Check if columns exist
            if indep_var['name'] not in results_df.columns or dep_var['name'] not in results_df.columns:
                continue

            # Extract data
            X = results_df[indep_var['name']].values
            Y = results_df[dep_var['name']].values

            # Remove missing values
            valid_mask = ~(np.isnan(X) | np.isnan(Y))
            X_valid = X[valid_mask]
            Y_valid = Y[valid_mask]

            if len(X_valid) < 3:
                continue

            # Calculate Pearson correlation
            r, p_value = stats.pearsonr(X_valid, Y_valid)

            correlation_results.append({
                'Independent_Variable': indep_var['label'],
                'Dependent_Variable': dep_var['label'],
                'r': r,
                'p_value': p_value,
                'n': len(X_valid)
            })

            # Print result
            sig_marker = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"  {indep_var['label']:30s} vs {dep_var['label']:15s}: r = {r:7.4f}, p = {p_value:.4f} {sig_marker}")

    # Save correlation results
    if correlation_results:
        correlation_df = pd.DataFrame(correlation_results)
        correlation_file = os.path.join(OUTPUT_DIR, 'correlation_analysis_results.csv')
        correlation_df.to_csv(correlation_file, index=False)
        print(f"\n{'='*80}")
        print(f"Correlation analysis results saved to:")
        print(f"  {correlation_file}")
        print(f"{'='*80}")

    # Additional correlation analysis: Disconnection variables vs SFC variables
    print("\n" + "=" * 80)
    print("Additional Correlation Analysis: Disconnection vs SFC")
    print("  7 Disconnection variables vs 3 SFC variables = 21 correlations")
    print("=" * 80)

    disconnection_vars = [
        {'name': 'af_disconnection', 'label': 'AF Disconnection'},
        {'name': 'ilf_disconnection', 'label': 'ILF Disconnection'},
        {'name': 'uf_disconnection', 'label': 'UF Disconnection'},
        {'name': 'af_ilf_disconnection', 'label': 'AF+ILF Disconnection'},
        {'name': 'af_uf_disconnection', 'label': 'AF+UF Disconnection'},
        {'name': 'ilf_uf_disconnection', 'label': 'ILF+UF Disconnection'},
        {'name': 'all_tracts_disconnection', 'label': 'All Tracts Disconnection'}
    ]

    sfc_vars = [
        {'name': 'dorsal_sfc', 'label': 'Dorsal SFC'},
        {'name': 'ventral_sfc', 'label': 'Ventral SFC'},
        {'name': 'global_sfc', 'label': 'Global SFC'}
    ]

    disconnection_sfc_correlations = []

    for disc_var in disconnection_vars:
        for sfc_var in sfc_vars:
            # Check if columns exist
            if disc_var['name'] not in results_df.columns or sfc_var['name'] not in results_df.columns:
                continue

            # Extract data
            X = results_df[disc_var['name']].values
            Y = results_df[sfc_var['name']].values

            # Remove missing values
            valid_mask = ~(np.isnan(X) | np.isnan(Y))
            X_valid = X[valid_mask]
            Y_valid = Y[valid_mask]

            if len(X_valid) < 3:
                continue

            # Calculate Pearson correlation
            r, p_value = stats.pearsonr(X_valid, Y_valid)

            disconnection_sfc_correlations.append({
                'Disconnection_Variable': disc_var['label'],
                'SFC_Variable': sfc_var['label'],
                'r': r,
                'p_value': p_value,
                'n': len(X_valid)
            })

            # Print result
            sig_marker = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"  {disc_var['label']:30s} vs {sfc_var['label']:15s}: r = {r:7.4f}, p = {p_value:.4f} {sig_marker}")

    # Save disconnection-SFC correlation results
    if disconnection_sfc_correlations:
        disc_sfc_corr_df = pd.DataFrame(disconnection_sfc_correlations)
        disc_sfc_corr_file = os.path.join(OUTPUT_DIR, 'correlation_disconnection_vs_sfc.csv')
        disc_sfc_corr_df.to_csv(disc_sfc_corr_file, index=False)
        print(f"\n{'='*80}")
        print(f"Disconnection-SFC correlation results saved to:")
        print(f"  {disc_sfc_corr_file}")
        print(f"{'='*80}")

    # Step 8: Calculate 7 network lesion percentages and correlate with SFC
    print("\n[Step 8/8] Calculating 7 network lesion percentages and correlating with SFC...")
    print("=" * 80)
    
    # Load atlas
    atlas_path = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_MNI152NLin6Asym_1mm.nii.gz"
    if not os.path.exists(atlas_path):
        print(f"Warning: Atlas file not found: {atlas_path}")
        print("Skipping network lesion analysis")
    else:
        atlas_img = nib.load(atlas_path)
        print(f"Loaded atlas: {atlas_path}")
        
        # Parse network information from template
        network_roi_map = parse_network_from_template(TEMPLATE_INFO)
        
        # Define 7 networks + Subcortical
        networks = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis", "Subcortical"]
        
        # Calculate network lesion percentages for each subject
        network_lesion_data = []
        
        for idx, subject_id in enumerate(subject_ids):
            print(f"  Processing subject {subject_id} ({idx+1}/{len(subject_ids)})...")
            
            # Load lesion data
            lesion_data = None
            possible_ids = []
            if subject_id.startswith('M'):
                possible_ids.append(subject_id)
                possible_ids.append(subject_id[1:])
            else:
                possible_ids.append(f'M{subject_id}')
                possible_ids.append(subject_id)
            
            for test_id in possible_ids:
                subject_dir = os.path.join(LESION_DIR, test_id)
                all_regions_file = os.path.join(subject_dir, f"{test_id}_all_regions.npy")
                
                if os.path.exists(all_regions_file):
                    try:
                        region_masks = np.load(all_regions_file, allow_pickle=True).item()
                        lesion_data = region_masks.get('lesion')
                        if lesion_data is not None:
                            break
                    except Exception as e:
                        print(f"    Warning: Failed to load {all_regions_file}: {e}")
                        continue
            
            if lesion_data is None:
                print(f"    Warning: No lesion data found for {subject_id}")
                continue
            
            # Get SFC values for this subject
            subject_result = next((r for r in results_list if r['subject_id'] == subject_id), None)
            if subject_result is None:
                print(f"    Warning: No SFC data found for {subject_id}")
                continue
            
            # Calculate lesion percentage for each network
            network_lesion_percentages = calculate_network_lesion_percentages(
                lesion_data, atlas_img, network_roi_map, networks
            )
            
            # Add to data
            row_data = {
                'subject_id': subject_id,
                'dorsal_sfc': subject_result['dorsal_sfc'],
                'ventral_sfc': subject_result['ventral_sfc'],
                'global_sfc': subject_result['global_sfc']
            }
            
            for network in networks:
                row_data[f'{network}_lesion_pct'] = network_lesion_percentages.get(network, 0.0)
            
            network_lesion_data.append(row_data)
        
        # Create DataFrame
        network_lesion_df = pd.DataFrame(network_lesion_data)
        
        # Save to CSV
        network_lesion_file = os.path.join(OUTPUT_DIR, 'network_lesion_percentages.csv')
        network_lesion_df.to_csv(network_lesion_file, index=False)
        print(f"\nNetwork lesion percentages saved to: {network_lesion_file}")
        
        # Calculate correlations between network lesion percentages and SFC
        print("\nCorrelations between Network Lesion Percentages and SFC:")
        print("=" * 80)
        
        sfc_vars = [
            {'name': 'dorsal_sfc', 'label': 'Dorsal SFC'},
            {'name': 'ventral_sfc', 'label': 'Ventral SFC'},
            {'name': 'global_sfc', 'label': 'Global SFC'}
        ]
        
        network_sfc_correlations = []
        
        for network in networks:
            lesion_col = f'{network}_lesion_pct'
            
            if lesion_col not in network_lesion_df.columns:
                continue
            
            for sfc_var in sfc_vars:
                if sfc_var['name'] not in network_lesion_df.columns:
                    continue
                
                # Extract data
                X = network_lesion_df[lesion_col].values
                Y = network_lesion_df[sfc_var['name']].values
                
                # Remove missing values
                valid_mask = ~(np.isnan(X) | np.isnan(Y))
                X_valid = X[valid_mask]
                Y_valid = Y[valid_mask]
                
                if len(X_valid) < 3:
                    continue
                
                # Calculate Pearson correlation
                r, p_value = stats.pearsonr(X_valid, Y_valid)
                
                network_sfc_correlations.append({
                    'Network': network,
                    'Lesion_Variable': lesion_col,
                    'SFC_Variable': sfc_var['label'],
                    'r': r,
                    'p_value': p_value,
                    'n': len(X_valid)
                })
                
                # Print result
                sig_marker = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  {network:20s} lesion vs {sfc_var['label']:15s}: r = {r:7.4f}, p = {p_value:.4f} {sig_marker}")
        
        # Save network-SFC correlation results
        if network_sfc_correlations:
            network_sfc_corr_df = pd.DataFrame(network_sfc_correlations)
            network_sfc_corr_file = os.path.join(OUTPUT_DIR, 'correlation_network_lesion_vs_sfc.csv')
            network_sfc_corr_df.to_csv(network_sfc_corr_file, index=False)
            print(f"\n{'='*80}")
            print(f"Network lesion-SFC correlation results saved to:")
            print(f"  {network_sfc_corr_file}")
            print(f"{'='*80}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

