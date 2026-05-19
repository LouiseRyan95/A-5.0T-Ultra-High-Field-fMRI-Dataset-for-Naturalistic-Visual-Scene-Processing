#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check behavioral accuracy from BIDS-style event files.

This script implements a simple target-detection rule used in the NVD task:
responses are expected only for target trials. By default, a target trial is
identified by trial_type == 1, and a valid response is defined as
response_time > 0.

Errors are classified as:
1. false_response: response on a non-target trial;
2. missed_response: no response on a target trial.

Outputs
-------
- events_check_summary.tsv: one row per event file.
- events_check_errors.tsv: one row per erroneous trial.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check behavioral accuracy from events.tsv files.")
    parser.add_argument("--events-dir", required=True, type=Path, help="Directory containing *_events.tsv files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to --events-dir.")
    parser.add_argument("--pattern", default="*_events.tsv", help="Glob pattern for event files.")
    parser.add_argument("--trial-type-column", default="trial_type", help="Column containing target/non-target labels.")
    parser.add_argument("--response-time-column", default="response_time", help="Column containing response times.")
    parser.add_argument("--stim-file-column", default="stim_file", help="Column containing stimulus names.")
    parser.add_argument("--target-trial-type", default="1", help="Value in trial_type_column that denotes a target trial.")
    parser.add_argument("--rt-eps", type=float, default=1e-12, help="Responses with RT > eps are treated as valid responses.")
    return parser.parse_args()


def to_numeric_series(values: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric values while coercing invalid values to NaN."""
    return pd.to_numeric(values, errors="coerce")


def normalize_label(value: object) -> str:
    """Normalize numeric-like labels so that 1 and 1.0 are treated identically."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return text


def required_columns(args: argparse.Namespace) -> list[str]:
    """Return columns required for the accuracy check."""
    return ["onset", "duration", args.trial_type_column, args.stim_file_column, args.response_time_column]


def check_one_file(path: Path, args: argparse.Namespace) -> tuple[dict, pd.DataFrame | None]:
    """Check one events.tsv file and return a summary row and an optional error table."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = [column for column in required_columns(args) if column not in df.columns]
    if missing:
        summary = {
            "file": path.name,
            "n_rows": len(df),
            "accuracy": np.nan,
            "n_errors": np.nan,
            "n_false_response": np.nan,
            "n_missed_response": np.nan,
            "note": "missing_columns=" + ",".join(missing),
        }
        return summary, None

    trial_type_norm = df[args.trial_type_column].map(normalize_label)
    response_time = to_numeric_series(df[args.response_time_column]).fillna(0.0)
    has_response = response_time > args.rt_eps
    is_target = trial_type_norm == normalize_label(args.target_trial_type)

    false_response = (~is_target) & has_response
    missed_response = is_target & (~has_response)
    any_error = false_response | missed_response

    n_rows = len(df)
    n_errors = int(any_error.sum())
    accuracy = (n_rows - n_errors) / n_rows if n_rows else np.nan

    summary = {
        "file": path.name,
        "n_rows": n_rows,
        "accuracy": round(float(accuracy), 6) if n_rows else np.nan,
        "n_errors": n_errors,
        "n_false_response": int(false_response.sum()),
        "n_missed_response": int(missed_response.sum()),
        "note": "",
    }

    if n_errors == 0:
        return summary, None

    keep_columns = ["onset", args.trial_type_column, args.stim_file_column, args.response_time_column]
    error_df = df.loc[any_error, keep_columns].copy()
    error_df.insert(0, "file", path.name)
    error_df.insert(1, "row_index_0based", error_df.index.astype(int))
    error_df["error_type"] = np.where(
        false_response[any_error],
        "false_response_non_target_with_rt",
        "missed_response_target_without_rt",
    )
    return summary, error_df


def write_outputs(summaries: Iterable[dict], errors: list[pd.DataFrame], output_dir: Path) -> None:
    """Write summary and error tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(list(summaries))
    summary_df.to_csv(output_dir / "events_check_summary.tsv", sep="\t", index=False)

    if errors:
        errors_df = pd.concat(errors, ignore_index=True).sort_values(["file", "row_index_0based"])
    else:
        errors_df = pd.DataFrame(
            columns=["file", "row_index_0based", "onset", "trial_type", "stim_file", "response_time", "error_type"]
        )
    errors_df.to_csv(output_dir / "events_check_errors.tsv", sep="\t", index=False)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.events_dir

    event_files = sorted(args.events_dir.glob(args.pattern))
    if not event_files:
        raise FileNotFoundError(f"No event files were found under {args.events_dir} with pattern {args.pattern}")

    summaries: list[dict] = []
    errors: list[pd.DataFrame] = []

    for path in event_files:
        summary, error_df = check_one_file(path, args)
        summaries.append(summary)
        if error_df is not None:
            errors.append(error_df)

    write_outputs(summaries, errors, output_dir)

    for row in summaries:
        print(
            f"{row['file']}: accuracy={row['accuracy']} errors={row['n_errors']} "
            f"false_response={row.get('n_false_response', '')} "
            f"missed_response={row.get('n_missed_response', '')} {row.get('note', '')}"
        )


if __name__ == "__main__":
    main()

