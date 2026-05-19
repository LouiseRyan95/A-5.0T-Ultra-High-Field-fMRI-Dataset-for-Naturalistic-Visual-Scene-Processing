# Script index

| Script | Purpose | Main inputs | Main outputs |
|---|---|---|---|
| `scripts/01_events_and_behavior/add_image_ids_to_events.py` | Convert stimulus file names to image IDs and encode blank conditions. | `*_events.tsv`, `images.csv` | Updated `*_events.tsv`, conversion summary, unmatched-condition report |
| `scripts/01_events_and_behavior/check_event_accuracy.py` | Compute run-wise behavioral accuracy from event files. | `*_events.tsv` | `events_check_summary.tsv`, `events_check_errors.tsv` |
| `scripts/02_quality_control/compute_framewise_displacement_summary.py` | Summarize FD after optional truncation to event-defined task duration. | confounds TSV, events TSV | long and wide FD CSV summaries |
| `scripts/02_quality_control/compute_cifti_tsnr.py` | Compute run-wise tSNR maps from CIFTI dtseries files. | `*.dtseries.nii` | `*_desc-tsnr.dscalar.nii`, summary CSV |
| `scripts/03_roi_mapping/create_visual_roi_map_hcp_mmp.py` | Build a 91k visual ROI map from an HCP-MMP dlabel file. | dlabel CIFTI, optional 91k template | ROI map `.mat`, ROI summary CSV, ROI ID table CSV |
| `scripts/04_glmsingle/build_glmsingle_bundle.py` | Build Python GLMsingle input bundles from CIFTI and events. | CIFTI dtseries, events TSV | per-subject bundle with data, design, condition IDs, and QC tables |
| `scripts/04_glmsingle/run_glmsingle.py` | Run Python GLMsingle on a prepared bundle. | GLMsingle bundle | GLMsingle output directory and diagnostic figures |
| `scripts/04_glmsingle/inspect_glmsingle_outputs.py` | Plot distributions of GLMsingle output fields. | `TYPED_FITHRF_GLMDENOISE_RR.npy` | histogram PNG/SVG and summary CSV |
| `scripts/04_glmsingle/extract_glmsingle_betas.py` | Export GLMsingle betas to CIFTI dscalar files. | GLMsingle output, trial table, CIFTI template, optional ROI map | raw-by-run and averaged beta dscalar files plus index CSVs |
| `scripts/05_reliability_noise_ceiling/compute_nsd_style_noise_ceiling.py` | Compute NSD-style ncsnr and noise ceiling maps from GLMsingle betas. | GLMsingle bundle, GLMsingle output, ROI map | voxel/grayordinate maps and ROI summaries |
| `scripts/06_stimulus_visualization/clip_tsne_visualization.py` | Extract CLIP image features and visualize them with PCA + t-SNE. | image directory | feature array, path table, 2D embedding CSV, PNG/SVG figure |

