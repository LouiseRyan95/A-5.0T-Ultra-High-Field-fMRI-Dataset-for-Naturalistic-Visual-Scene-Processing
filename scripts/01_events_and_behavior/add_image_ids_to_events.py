#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert event-file stimulus names to image IDs.

The script reads an image table with columns `image_name` and `image_id`, maps
each event-file `stim_file` value to the corresponding image ID, and writes a
`condition` column. Blank trials are encoded separately so they can be excluded
from GLMsingle or other condition-wise analyses.

Default blank coding
--------------------
- trial_type == 0: condition = 20000
- trial_type == 1: condition = 30000

Non-blank values that cannot be matched to images.csv are preserved and reported
in `condition_unmatched_details.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add image_id-based condition labels to events.tsv files.")
    parser.add_argument("--events-dir", required=True, type=Path, help="Directory containing input *_events.tsv files.")
    parser.add_argument("--images-csv", required=True, type=Path, help="CSV file with image_name and image_id columns.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for converted event files and summaries.")
    parser.add_argument("--pattern", default="*_events.tsv", help="Glob pattern for event files.")
    parser.add_argument("--stim-file-column", default="stim_file", help="Column containing image file names.")
    parser.add_argument("--trial-type-column", default="trial_type", help="Column used to classify blank trials.")
    parser.add_argument("--condition-column", default="condition", help="Output condition column name.")
    parser.add_argument("--blank-id-nontarget", default="20000", help="Condition ID for blank/non-target trials.")
    parser.add_argument("--blank-id-target", default="30000", help="Condition ID for blank/target trials.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite input event files instead of writing to --output-dir.")
    return parser.parse_args()


def load_image_map(images_csv: Path) -> dict[str, object]:
    """Load a mapping from image_name to image_id and check duplicates."""
    images = pd.read_csv(images_csv)
    required = {"image_name", "image_id"}
    missing = required - set(images.columns)
    if missing:
        raise ValueError(f"{images_csv} is missing required columns: {sorted(missing)}")

    duplicated = images[images["image_name"].duplicated(keep=False)]
    if not duplicated.empty:
        raise ValueError(
            "Duplicated image_name values were found in images.csv. "
            "Remove duplicates before running this conversion."
        )
    return dict(zip(images["image_name"].astype(str), images["image_id"]))


def convert_one_file(path: Path, image_map: dict[str, object], args: argparse.Namespace) -> tuple[dict, pd.DataFrame | None]:
    """Convert one events.tsv file and return a summary row plus optional anomaly rows."""
    df = pd.read_csv(path, sep="\t")
    if args.stim_file_column not in df.columns:
        return {
            "file": path.name,
            "status": "skipped_missing_stim_file_column",
            "n_rows": len(df),
            "n_matched_image": 0,
            "n_blank": 0,
            "n_blank_to_nontarget_id": 0,
            "n_blank_to_target_id": 0,
            "n_blank_bad_trial_type": 0,
            "n_unmatched_nonblank": 0,
            "match_rate_excluding_blank": np.nan,
            "output_file": "",
        }, None

    if args.trial_type_column not in df.columns:
        raise ValueError(f"{path.name} does not contain the trial-type column: {args.trial_type_column}")

    stim = df[args.stim_file_column].astype(str)
    trial_type = pd.to_numeric(df[args.trial_type_column], errors="coerce")

    is_blank = stim.str.lower().eq("blank")
    is_matched_image = stim.isin(image_map)
    blank_target = is_blank & trial_type.eq(1)
    blank_nontarget = is_blank & trial_type.eq(0)
    blank_bad_trial_type = is_blank & ~(trial_type.eq(0) | trial_type.eq(1))
    unmatched_nonblank = (~is_blank) & (~is_matched_image)

    new_condition = stim.map(image_map)
    new_condition = new_condition.where(~blank_target, args.blank_id_target)
    new_condition = new_condition.where(~blank_nontarget, args.blank_id_nontarget)
    new_condition = new_condition.where(~blank_bad_trial_type, stim)
    new_condition = new_condition.where(~unmatched_nonblank, stim)
    df[args.condition_column] = new_condition

    anomaly_mask = unmatched_nonblank | blank_bad_trial_type
    anomaly_df = None
    if anomaly_mask.any():
        anomaly_df = df.loc[anomaly_mask].copy()
        anomaly_df.insert(0, "source_file", path.name)
        anomaly_df.insert(1, "row_index_0based", anomaly_df.index.astype(int))
        anomaly_df.insert(
            2,
            "problem_type",
            np.where(
                unmatched_nonblank.loc[anomaly_mask],
                "unmatched_nonblank_stim_file",
                "blank_bad_trial_type",
            ),
        )

    out_file = path if args.overwrite else args.output_dir / path.name
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, sep="\t", index=False)

    n_nonblank = int((~is_blank).sum())
    n_matched = int(is_matched_image.sum())
    summary = {
        "file": path.name,
        "status": "converted",
        "trial_type_column": args.trial_type_column,
        "n_rows": len(df),
        "n_matched_image": n_matched,
        "n_blank": int(is_blank.sum()),
        "n_blank_to_nontarget_id": int(blank_nontarget.sum()),
        "n_blank_to_target_id": int(blank_target.sum()),
        "n_blank_bad_trial_type": int(blank_bad_trial_type.sum()),
        "n_unmatched_nonblank": int(unmatched_nonblank.sum()),
        "match_rate_excluding_blank": float(n_matched / n_nonblank) if n_nonblank else np.nan,
        "output_file": str(out_file),
    }
    return summary, anomaly_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_map = load_image_map(args.images_csv)
    event_files = sorted(args.events_dir.glob(args.pattern))
    if not event_files:
        raise FileNotFoundError(f"No event files found under {args.events_dir} with pattern {args.pattern}")

    summary_rows: list[dict] = []
    anomaly_tables: list[pd.DataFrame] = []

    for path in event_files:
        summary, anomaly_df = convert_one_file(path, image_map, args)
        summary_rows.append(summary)
        if anomaly_df is not None:
            anomaly_tables.append(anomaly_df)
        print(
            f"[DONE] {path.name}: rows={summary['n_rows']} matched={summary['n_matched_image']} "
            f"blank_nontarget={summary['n_blank_to_nontarget_id']} "
            f"blank_target={summary['n_blank_to_target_id']} "
            f"unmatched={summary['n_unmatched_nonblank']}"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "condition_to_image_id_summary.csv", index=False, encoding="utf-8-sig")

    if anomaly_tables:
        anomalies = pd.concat(anomaly_tables, ignore_index=True)
    else:
        anomalies = pd.DataFrame(columns=["source_file", "row_index_0based", "problem_type"])
    anomalies.to_csv(args.output_dir / "condition_unmatched_details.csv", index=False, encoding="utf-8-sig")

    print(f"[SUMMARY] {args.output_dir / 'condition_to_image_id_summary.csv'}")
    print(f"[UNMATCHED] {args.output_dir / 'condition_unmatched_details.csv'}")


if __name__ == "__main__":
    main()

