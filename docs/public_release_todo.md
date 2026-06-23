# Public Release TODO

This folder is a cleaned selection of the manuscript analysis code, not yet a fully portable release.

Before uploading to GitHub, complete the following checks.

## Required

- Replace remaining local absolute paths in plotting/statistical scripts with command-line arguments or a shared config loader.
- Remove or anonymize any clinical file names that imply private subject-level tables.
- Confirm that no raw imaging data, lesion masks, SFC output text files, generated figures, or manuscript source files are committed.
- Decide how to handle surface plotting:
  - vendor a lightweight `neurocat` helper, or
  - replace `neurocat.plotting.draw_and_save` calls with standard `nilearn`/`neuromaps` plotting.
- Confirm licensing for any atlas files if they are added later.
- Add a repository license after the author team decides the intended reuse terms.

## Already Done In This Draft

- Main non-ML and non-gene scripts are copied into a clean `src/` layout.
- Working script names were replaced with descriptive names.
- ML/foundation-model and gene/PLS scripts were excluded.
- Data and output patterns are blocked in `.gitignore`.
- Core SFC computation scripts use relative default paths and command-line arguments.
- A file mapping table is included in `docs/file_mapping.md`.

## Local Path Audit

The core SFC scripts have been cleaned first. Several figure/statistical scripts still contain paths such as:

- `D:\sc-fc_coupling\...`
- `D:\download\...`
- `batch_results_pca_450_t288`
- `single_subject_results_t288`
- `aphasia_plots\participants_*.txt`

These should be converted before public upload.
