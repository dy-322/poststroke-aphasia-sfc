from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

import compute_sfc_pca as scfc


DATA_ROOT = Path("data/hospital_validation")
INPUT_DIR = DATA_ROOT / "scfc_pca_inputs_450"
RESULT_DIR = DATA_ROOT / "sfc_pca_450_t288"
DISTANCE_MATRIX_FILE = Path("data/euclidean_distance_matrix_450.mat")

ROI_NUMBER = 450
TARGET_REAL_TIME = 288.0
REMOVE_FIRST_N = 10
N_COMPONENTS = 4
EXPLAINED_VARIANCE_THRESHOLD = 0.8


def compute_fc_matrix_numpy(timeseries_data, apply_fisher_z=True):
    """Compute ROI-by-ROI Pearson FC without relying on nilearn API details."""
    print("Computing FC matrix with numpy...")
    data = np.asarray(timeseries_data, dtype=float)
    data = data - np.nanmean(data, axis=0, keepdims=True)
    std = np.nanstd(data, axis=0, ddof=1, keepdims=True)
    std[std == 0] = np.nan
    data = data / std
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    fc = np.corrcoef(data, rowvar=False)

    if apply_fisher_z:
        fc = np.clip(fc, -0.999999, 0.999999)
        fc = np.arctanh(fc)
        print("Applied Fisher z-transform to FC matrix")

    print(fc.shape)
    return fc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCA-based SFC for the hospital validation cohort.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--distance-matrix-file", type=Path, default=DISTANCE_MATRIX_FILE)
    parser.add_argument("--roi-number", type=int, default=ROI_NUMBER)
    parser.add_argument("--target-real-time", type=float, default=TARGET_REAL_TIME)
    parser.add_argument("--remove-first-n", type=int, default=REMOVE_FIRST_N)
    parser.add_argument("--n-components", type=int, default=N_COMPONENTS)
    parser.add_argument("--explained-variance-threshold", type=float, default=EXPLAINED_VARIANCE_THRESHOLD)
    args = parser.parse_args()

    scfc.compute_fc_matrix = compute_fc_matrix_numpy

    scfc.roi_number = args.roi_number
    scfc.base_data_path = str(args.input_dir)
    scfc.result_dir = str(args.result_dir)
    scfc.sfc_txt_dir = str(args.result_dir / "sfc_txt_files")
    scfc.images_dir = str(args.result_dir / "images")

    args.result_dir.mkdir(parents=True, exist_ok=True)
    Path(scfc.sfc_txt_dir).mkdir(parents=True, exist_ok=True)
    Path(scfc.images_dir).mkdir(parents=True, exist_ok=True)

    distance_matrix = sio.loadmat(args.distance_matrix_file)["distance_matrix"]
    subjects_info = scfc.find_subject_files()
    if not subjects_info:
        raise RuntimeError(f"No valid subjects found in {args.input_dir}")

    results = []
    for idx, subject_info in enumerate(subjects_info, start=1):
        print(f"\nProcessing subject {idx}/{len(subjects_info)}")
        result = scfc.process_single_subject(
            subject_info,
            distance_matrix,
            args.target_real_time,
            n_components=args.n_components,
            explained_variance_threshold=args.explained_variance_threshold,
            remove_first_n=args.remove_first_n,
        )
        results.append(result)

    scfc.save_results(results)
    print(f"\nSFC txt files saved to: {scfc.sfc_txt_dir}")


if __name__ == "__main__":
    main()
