#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute temporal signal-to-noise ratio (tSNR) maps from CIFTI dtseries files.

For each input dtseries, tSNR is computed for each grayordinate as:

    tSNR = mean(time series) / standard_deviation(time series)

The output is a CIFTI dscalar file that preserves the input BrainModelAxis and
contains one scalar map named `tSNR`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.cifti2.cifti2_axes import ScalarAxis

SUB_RE = re.compile(r"(sub-[^_]+)")
SES_RE = re.compile(r"(ses-[^_]+)")
TASK_RE = re.compile(r"(task-[^_]+)")
RUN_RE = re.compile(r"(run-\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CIFTI tSNR maps.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing CIFTI dtseries files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for tSNR dscalar files.")
    parser.add_argument("--pattern", default="*_space-fsLR_den-91k_bold.dtseries.nii", help="Glob pattern for CIFTI dtseries files.")
    parser.add_argument("--ddof", type=int, default=0, help="Delta degrees of freedom for standard deviation.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tSNR files.")
    return parser.parse_args()


def grab(regex: re.Pattern[str], text: str, default: str = "NA") -> str:
    """Extract one BIDS-like entity from a filename."""
    match = regex.search(text)
    return match.group(1) if match else default


def find_series_and_brain_axes(img: nib.Cifti2Image) -> tuple[int, int]:
    """Return the time-axis and brain-model-axis indices for a CIFTI image."""
    series_axis = None
    brain_axis = None
    for axis_index in range(len(img.shape)):
        axis = img.header.get_axis(axis_index)
        axis_name = axis.__class__.__name__
        if axis_name == "SeriesAxis":
            series_axis = axis_index
        if axis_name == "BrainModelAxis":
            brain_axis = axis_index
    if series_axis is None or brain_axis is None:
        raise ValueError("Input CIFTI file must contain a SeriesAxis and a BrainModelAxis.")
    return series_axis, brain_axis


def compute_tsnr_dtseries(dtseries_path: Path, ddof: int = 0) -> tuple[nib.Cifti2Image, dict]:
    """Compute one tSNR dscalar CIFTI image and a summary dictionary."""
    img = nib.load(str(dtseries_path))
    if not isinstance(img, nib.Cifti2Image):
        raise TypeError(f"Expected a CIFTI2 image, got {type(img)} for {dtseries_path}")

    series_axis_index, brain_axis_index = find_series_and_brain_axes(img)
    data = img.get_fdata(dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D CIFTI dtseries, got shape {data.shape}: {dtseries_path}")

    if series_axis_index == 0:
        time_by_grayordinate = data
    else:
        time_by_grayordinate = data.T

    mean_ts = np.mean(time_by_grayordinate, axis=0)
    std_ts = np.std(time_by_grayordinate, axis=0, ddof=ddof)
    tsnr = np.zeros_like(mean_ts, dtype=np.float32)
    valid = std_ts > 0
    tsnr[valid] = mean_ts[valid] / std_ts[valid]

    brain_axis = img.header.get_axis(brain_axis_index)
    scalar_axis = ScalarAxis(["tSNR"])
    out_img = nib.Cifti2Image(
        tsnr[np.newaxis, :],
        header=nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis)),
    )
    out_img.update_headers()

    summary = {
        "file": dtseries_path.name,
        "n_timepoints": int(time_by_grayordinate.shape[0]),
        "n_grayordinates": int(time_by_grayordinate.shape[1]),
        "tsnr_mean_all": float(np.mean(tsnr)),
        "tsnr_median_all": float(np.median(tsnr)),
        "tsnr_p05_all": float(np.percentile(tsnr, 5)),
        "tsnr_p95_all": float(np.percentile(tsnr, 95)),
    }
    return out_img, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtseries_files = sorted(args.input_dir.rglob(args.pattern))
    if not dtseries_files:
        raise FileNotFoundError(f"No files found under {args.input_dir} with pattern {args.pattern}")

    rows: list[dict] = []
    for dtseries_path in dtseries_files:
        out_name = dtseries_path.name.replace("_bold.dtseries.nii", "_desc-tsnr.dscalar.nii")
        if out_name == dtseries_path.name:
            out_name = dtseries_path.name.replace(".dtseries.nii", "_desc-tsnr.dscalar.nii")
        out_path = args.output_dir / out_name

        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] Existing file: {out_path}")
            continue

        out_img, summary = compute_tsnr_dtseries(dtseries_path, ddof=args.ddof)
        nib.save(out_img, str(out_path))

        filename = dtseries_path.name
        summary.update(
            {
                "sub": grab(SUB_RE, filename),
                "ses": grab(SES_RE, filename),
                "task": grab(TASK_RE, filename),
                "run": grab(RUN_RE, filename),
                "out_tsnr": out_name,
            }
        )
        rows.append(summary)
        print(f"[DONE] tSNR saved: {out_path}")

    pd.DataFrame(rows).to_csv(args.output_dir / "tsnr_per_run_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[SUMMARY] {args.output_dir / 'tsnr_per_run_summary.csv'}")


if __name__ == "__main__":
    main()

