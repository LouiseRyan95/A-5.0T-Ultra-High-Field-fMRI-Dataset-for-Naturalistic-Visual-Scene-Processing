# Code release for NVD

This repository contains analysis scripts for the manuscript:

**A 5.0T Ultra-High-Field fMRI Dataset for Naturalistic Visual Scene Processing**  
Gengchen Ye, Mo Wang, Chiyin Li, Yihao Peng, Yilin Qian, Yutao Wang, Xinyi Si, Shaoxin Xiang, Fanzhi Jiang, Lu Wang, and Ming Zhang.

The code was refactored from analysis scripts used for behavioral checks, quality control, CIFTI tSNR mapping, visual ROI construction, GLMsingle beta estimation, NSD-style noise ceiling estimation, and stimulus feature visualization. Hard-coded local paths were replaced with command-line arguments or template configuration files. Comments and docstrings are written in English for public release.

## Repository structure

```text
NVD_code_release/
├── configs/
│   └── analysis_params_template.yaml
├── scripts/
│   ├── 01_events_and_behavior/
│   │   ├── add_image_ids_to_events.py
│   │   └── check_event_accuracy.py
│   ├── 02_quality_control/
│   │   ├── compute_cifti_tsnr.py
│   │   └── compute_framewise_displacement_summary.py
│   ├── 03_roi_mapping/
│   │   └── create_visual_roi_map_hcp_mmp.py
│   ├── 04_glmsingle/
│   │   ├── build_glmsingle_bundle.py
│   │   ├── extract_glmsingle_betas.py
│   │   ├── inspect_glmsingle_outputs.py
│   │   └── run_glmsingle.py
│   ├── 05_reliability_noise_ceiling/
│   │   └── compute_nsd_style_noise_ceiling.py
│   └── 06_stimulus_visualization/
│       └── clip_tsne_visualization.py
├── docs/
│   └── script_index.md
├── requirements.txt
├── environment.yml
└── CITATION.cff
```

## Installation

A minimal Python environment can be created with conda:

```bash
conda env create -f environment.yml
conda activate nvd-code
```

For CLIP-based stimulus visualization, install the OpenAI CLIP package separately:

```bash
pip install git+https://github.com/openai/CLIP.git
```

For GLMsingle analyses, install the Python GLMsingle package following the official instructions for your operating system. The script `run_glmsingle.py` imports `glmsingle.glmsingle.GLM_single` and expects a working GLMsingle installation.

## Typical workflow

1. Convert event-file stimulus names to image IDs:

```bash
python scripts/01_events_and_behavior/add_image_ids_to_events.py \
  --events-dir /path/to/events \
  --images-csv /path/to/images.csv \
  --output-dir /path/to/events_with_image_id
```

2. Check behavioral accuracy from BIDS-style event files:

```bash
python scripts/01_events_and_behavior/check_event_accuracy.py \
  --events-dir /path/to/events_with_image_id \
  --output-dir /path/to/qc/behavior
```

3. Compute framewise-displacement summaries:

```bash
python scripts/02_quality_control/compute_framewise_displacement_summary.py \
  --confounds-dir /path/to/confounds \
  --events-dir /path/to/events_with_image_id \
  --output-dir /path/to/qc/fd \
  --tr 1.0 \
  --pad-sec 3.0
```

4. Compute run-wise CIFTI tSNR maps:

```bash
python scripts/02_quality_control/compute_cifti_tsnr.py \
  --input-dir /path/to/cifti \
  --output-dir /path/to/qc/tsnr
```

5. Build GLMsingle input bundles from CIFTI and event files:

```bash
python scripts/04_glmsingle/build_glmsingle_bundle.py \
  --cifti-dir /path/to/cifti \
  --events-dir /path/to/events_with_image_id \
  --output-dir /path/to/glmsingle_input_by_sub \
  --subjects 001 002 \
  --tr 1.0 \
  --stimdur 1.0 \
  --overwrite
```

6. Run GLMsingle:

```bash
python scripts/04_glmsingle/run_glmsingle.py \
  --bundle-dir /path/to/glmsingle_input_by_sub/sub-001_glmsingle_33runs_bundle \
  --output-dir /path/to/glmsingle_output/sub-001 \
  --figure-dir /path/to/glmsingle_figures/sub-001
```

7. Compute NSD-style noise ceiling from GLMsingle outputs:

```bash
python scripts/05_reliability_noise_ceiling/compute_nsd_style_noise_ceiling.py \
  --bundle-dir /path/to/glmsingle_input_by_sub/sub-001_glmsingle_33runs_bundle \
  --glmsingle-out-dir /path/to/glmsingle_output/sub-001 \
  --roimap-mat /path/to/roimap_91k_visual_rois.mat \
  --roi-table-csv /path/to/roimap_91k_visual_rois_id_table.csv
```

## Notes

The scripts assume BIDS-like file names such as:

```text
sub-001_ses-102_task-s1_space-fsLR_den-91k_bold.dtseries.nii
sub-001_ses-102_task-s1_events.tsv
sub-001_ses-102_task-s1_desc-confounds_timeseries.tsv
```

The GLMsingle bundle builder treats each unique image ID as one condition and excludes blank or baseline conditions by default. Edit command-line options if your event-file columns or condition coding differ.

## Citation

Please cite the NVD manuscript when using this code

