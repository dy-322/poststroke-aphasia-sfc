from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_DATA_ROOT = Path("data/hospital_validation/raw")


def copy_if_needed(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return "exists"
    shutil.copy2(src, dst)
    return "copied"


def prepare_inputs(data_root: Path, output_dir: Path) -> pd.DataFrame:
    manifest_path = data_root / "manifest_paired_subjects.csv"
    conn_dir = data_root / "connectioncsv"
    ts_dir = data_root / "timeseries"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not conn_dir.exists():
        raise FileNotFoundError(f"Connection directory not found: {conn_dir}")
    if not ts_dir.exists():
        raise FileNotFoundError(f"Timeseries directory not found: {ts_dir}")

    manifest = pd.read_csv(manifest_path)
    required_cols = {"subject", "connectioncsv_file", "timeseries_file"}
    missing_cols = sorted(required_cols - set(manifest.columns))
    if missing_cols:
        raise ValueError(f"Manifest missing columns: {missing_cols}")

    rows = []
    for row in manifest.itertuples(index=False):
        subject = str(row.subject)
        conn_src = conn_dir / str(row.connectioncsv_file)
        ts_src = ts_dir / str(row.timeseries_file)

        subject_dir = output_dir / subject
        conn_dst = subject_dir / "schaefer400_connectome.csv"
        ts_dst = subject_dir / str(row.timeseries_file)

        if not conn_src.exists() or not ts_src.exists():
            rows.append(
                {
                    "subject": subject,
                    "status": "missing_source",
                    "connection_source": str(conn_src),
                    "timeseries_source": str(ts_src),
                    "subject_dir": str(subject_dir),
                }
            )
            continue

        conn_status = copy_if_needed(conn_src, conn_dst)
        ts_status = copy_if_needed(ts_src, ts_dst)
        rows.append(
            {
                "subject": subject,
                "status": "ready",
                "connection_status": conn_status,
                "timeseries_status": ts_status,
                "connection_source": str(conn_src),
                "timeseries_source": str(ts_src),
                "subject_dir": str(subject_dir),
                "sc_file": str(conn_dst),
                "fc_file": str(ts_dst),
            }
        )

    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "prepared_subjects_manifest.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare hospital validation connectioncsv/timeseries files for compute_sfc_pca.py."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "scfc_pca_inputs_450",
    )
    args = parser.parse_args()

    summary = prepare_inputs(args.data_root, args.output_dir)
    ready = int((summary["status"] == "ready").sum())
    failed = int((summary["status"] != "ready").sum())
    print(f"Prepared subjects: {ready}")
    print(f"Failed subjects: {failed}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
