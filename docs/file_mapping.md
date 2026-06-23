# File Mapping

This table maps the working-directory script names to the cleaned names used in this draft GitHub folder.

## SFC Computation

| Original file | Cleaned file |
| --- | --- |
| `batch_scfc_pca.py` | `src/compute_sfc/compute_sfc_pca.py` |
| `prepare_aphasia_scfc_pca_inputs.py` | `src/compute_sfc/prepare_hospital_sfc_inputs.py` |
| `run_aphasia_scfc_pca_450.py` | `src/compute_sfc/run_hospital_sfc_pca_450.py` |

## Group SFC Maps

| Original file | Cleaned file |
| --- | --- |
| `plot_sfc_browser.py` | `src/group_sfc_maps/summarize_sfc_by_group.py` |
| `plot_final.py` | `src/group_sfc_maps/plot_group_surface_maps.py` |
| `plot_all_network.py` | `src/group_sfc_maps/plot_network_sfc_boxplots.py` |
| `plot_aphasia_sfc_categories_plot_final_style.py` | `src/group_sfc_maps/plot_hospital_category_surfaces.py` |
| `plot_aphasia_sfc_network_boxplots.py` | `src/group_sfc_maps/plot_hospital_network_boxplots.py` |
| `aphasia_sfc_surface_utils.py` | `src/group_sfc_maps/surface_utils.py` |

## Group Statistics

| Original file | Cleaned file |
| --- | --- |
| `sig_brain_region.py` | `src/group_statistics/roi_group_comparison_cohens_d.py` |
| `sig_brain_spintest.py` | `src/group_statistics/spin_test_group_rois.py` |
| `plot_radar.py` | `src/group_statistics/plot_significant_network_radar.py` |
| `plot_box.py` | `src/group_statistics/plot_cohens_d_network_boxplots.py` |

## Language Correlations

| Original file | Cleaned file |
| --- | --- |
| `roi_sfc_wab_correlation.py` | `src/language_correlations/correlate_sfc_language_scores.py` |
| `plot_box_correlation.py` | `src/language_correlations/plot_language_network_correlations.py` |
| `spin_test_analysis.py` | `src/language_correlations/spin_test_language_rois.py` |
| `plot_correlation_scatters.py` | `src/language_correlations/plot_language_correlation_scatters.py` |

## Lesion, Disconnectome, and Mediation

| Original file | Cleaned file |
| --- | --- |
| `analyze_regional_sfc_zscore_final.py` | `src/lesion_disconnectome/analyze_lesion_zone_sfc_zscore.py` |
| `analyze_regional_sfc_by_aphasia.py` | `src/lesion_disconnectome/analyze_lesion_zone_sfc_by_aphasia.py` |
| `disconnectome_mediation_analysis.py` | `src/lesion_disconnectome/disconnectome_mediation.py` |
| `lesion_vs_sfc.py` | `src/lesion_disconnectome/plot_disconnection_sfc_scatter.py` |
| `correlation_analysis.py` | `src/lesion_disconnectome/analyze_lesion_clinical_correlations.py` |
| `lesion_clinical_visualization.py` | `src/lesion_disconnectome/plot_lesion_clinical_matrix.py` |
| `mediation_analysis.py` | `src/lesion_disconnectome/sfc_degree_mediation.py` |

## Asymmetry

| Original file | Cleaned file |
| --- | --- |
| `asymmetry_analysis.py` | `src/asymmetry/analyze_sfc_asymmetry.py` |
| `compute_ai_correlation.py` | `src/asymmetry/compute_left_right_ai_correlation.py` |

## Intentionally Excluded

- ML/foundation-model and GNNExplainer code.
- Gene-expression, PLS, and GO enrichment code.
- Raw data, subject-level clinical tables, lesion masks, generated SFC values, generated figures, manuscript PDFs, Illustrator files, and cached outputs.
