#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute framewise-displacement summaries from fMRIPrep confounds files.

For task runs, the script can restrict FD summaries to the event-defined task
period plus a short padding window. The default rule uses the last row of the
matching events.tsv file:

    used duration = last_onset + last_duration + pad_sec

Resting-state runs are summarized over the full confounds file.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

SUB_RE = re.compile(r"^(sub-[^_]+)_")
SES_RE = re.compile(r"_ses-([^_]+)_")
TASK_RE = re.compile(r"_task-([^_]+)_", re.IGNORECASE)
RUN_RE = re.compile(r"_run-(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize framewise displacement for fMRI runs.")
    parser.add_argument("--confounds-dir", required=True, type=Path, help="Directory containing fMRIPrep confounds TSV files.")
    parser.add_argument("--events-dir", required=True, type=Path, help="Directory containing matching events.tsv files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for FD summaries.")
    parser.add_argument("--pattern", default="*_desc-confounds_timeseries.tsv", help="Glob pattern for confounds files.")
    parser.add_argument("--tr", type=float, default=1.0, help="Repetition time in seconds.")
    parser.add_argument("--pad-sec", type=float, default=3.0, help="Padding after the last event duration for task runs.")
    parser.add_argument("--fd-column", default="framewise_displacement", help="FD column in confounds files.")
    parser.add_argument("--rest-label", default="rest", help="Task label treated as resting-state and not truncated.")
    return parser.parse_args()


def parse_sub(filename: str) -> str:
    match = SUB_RE.match(filename)
    return match.group(1) if match else "sub-UNKNOWN"


def parse_ses_group(filename: str) -> int:
    match = SES_RE.search(filename)
    if not match:
        return 0
    digits = re.search(r"\d+", match.group(1))
    if not digits:
        return 0
    ses = int(digits.group())
    return ses // 100 if ses >= 100 else ses


def parse_task_label(filename: str) -> str:
    match = TASK_RE.search(filename)
    return match.group(1).lower() if match else "unknown"


def parse_task_order(filename: str, rest_label: str = "rest") -> int:
    task = parse_task_label(filename)
    if task == rest_label.lower():
        return 0
    digits = re.search(r"\d+", task)
    return int(digits.group()) if digits else 9999


def parse_run(filename: str) -> int:
    match = RUN_RE.search(filename)
    return int(match.group(1)) if match else 0


def sort_key(path: Path, rest_label: str) -> tuple[int, int, int]:
    return (parse_ses_group(path.name), parse_task_order(path.name, rest_label), parse_run(path.name))


def confounds_to_events_path(confounds_file: Path, events_dir: Path) -> Path:
    """Return the expected matching events.tsv path for a confounds file."""
    suffix = "_desc-confounds_timeseries.tsv"
    if not confounds_file.name.endswith(suffix):
        raise ValueError(f"Unexpected confounds filename: {confounds_file.name}")
    prefix = confounds_file.name.replace(suffix, "")
    return events_dir / f"{prefix}_events.tsv"


def compute_event_defined_duration(events_file: Path, tr: float, pad_sec: float) -> tuple[float, float, int]:
    """Compute the number of TRs to retain from the final onset and duration."""
    events = pd.read_csv(events_file, sep="\t")
    if events.empty:
        raise ValueError(f"Events file is empty: {events_file}")
    if not {"onset", "duration"}.issubset(events.columns):
        raise ValueError(f"Events file is missing onset/duration columns: {events_file}")

    onset = pd.to_numeric(events["onset"], errors="coerce")
    duration = pd.to_numeric(events["duration"], errors="coerce")
    last_onset = float(onset.iloc[-1])
    last_duration = float(duration.iloc[-1])

    if not np.isfinite(last_onset) or not np.isfinite(last_duration):
        raise ValueError(f"Last onset/duration is not finite in: {events_file}")

    actual_end_sec = last_onset + last_duration
    actual_end_plus_pad_sec = actual_end_sec + pad_sec
    used_n_tr = int(math.ceil(actual_end_plus_pad_sec / tr))
    return actual_end_sec, actual_end_plus_pad_sec, used_n_tr


def summarize_one_confounds(path: Path, run_index: int, args: argparse.Namespace) -> dict:
    """Summarize one confounds file."""
    df = pd.read_csv(path, sep="\t")
    if args.fd_column not in df.columns:
        raise ValueError(f"Missing {args.fd_column} column in {path}")

    fd = pd.to_numeric(df[args.fd_column], errors="coerce").to_numpy(dtype=float)
    fd = np.nan_to_num(fd, nan=0.0, posinf=0.0, neginf=0.0)
    n_total = int(fd.size)

    task_label = parse_task_label(path.name)
    used_n_tr = n_total
    actual_end_sec = np.nan
    actual_end_plus_pad_sec = np.nan
    events_file_label = ""

    if task_label != args.rest_label.lower():
        events_path = confounds_to_events_path(path, args.events_dir)
        events_file_label = events_path.name
        if events_path.exists():
            try:
                actual_end_sec, actual_end_plus_pad_sec, estimated_n_tr = compute_event_defined_duration(
                    events_path, args.tr, args.pad_sec
                )
                used_n_tr = min(n_total, estimated_n_tr)
            except Exception as exc:
                events_file_label = f"{events_path.name} [ERROR: {exc}]"
                used_n_tr = n_total
        else:
            events_file_label = f"{events_path.name} [MISSING]"

    fd_used = fd[:used_n_tr] if used_n_tr > 0 else np.array([], dtype=float)

    return {
        "sub": parse_sub(path.name),
        "run_index": int(run_index),
        "ses_group": int(parse_ses_group(path.name)),
        "task_order": int(parse_task_order(path.name, args.rest_label)),
        "task_label": task_label,
        "run": int(parse_run(path.name)),
        "n_tr_total": n_total,
        "n_tr_used": int(used_n_tr),
        "actual_end_sec": float(actual_end_sec) if np.isfinite(actual_end_sec) else np.nan,
        "actual_end_plus_pad_sec": float(actual_end_plus_pad_sec) if np.isfinite(actual_end_plus_pad_sec) else np.nan,
        "tr_sec": float(args.tr),
        "pad_sec": float(args.pad_sec),
        "mean_fd": float(np.mean(fd_used)) if fd_used.size else 0.0,
        "median_fd": float(np.median(fd_used)) if fd_used.size else 0.0,
        "max_fd": float(np.max(fd_used)) if fd_used.size else 0.0,
        "n_fd_gt_0p2": int(np.sum(fd_used > 0.2)),
        "pct_fd_gt_0p2": float(np.mean(fd_used > 0.2) * 100.0) if fd_used.size else 0.0,
        "n_fd_gt_0p5": int(np.sum(fd_used > 0.5)),
        "pct_fd_gt_0p5": float(np.mean(fd_used > 0.5) * 100.0) if fd_used.size else 0.0,
        "confounds_file": path.name,
        "events_file": events_file_label,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    confounds_files = sorted(args.confounds_dir.glob(args.pattern))
    if not confounds_files:
        raise FileNotFoundError(f"No confounds files found under {args.confounds_dir} with pattern {args.pattern}")

    grouped: dict[str, list[Path]] = {}
    for path in confounds_files:
        grouped.setdefault(parse_sub(path.name), []).append(path)

    rows: list[dict] = []
    for sub in sorted(grouped):
        files = sorted(grouped[sub], key=lambda path: sort_key(path, args.rest_label))
        for run_index, path in enumerate(files, start=1):
            rows.append(summarize_one_confounds(path, run_index, args))

    per_run = pd.DataFrame(rows)
    per_run.to_csv(args.output_dir / "fd_per_run_long.csv", index=False, encoding="utf-8-sig")

    wide_mean = per_run.pivot(index="sub", columns="run_index", values="mean_fd")
    wide_mean.columns = [f"run{int(column):02d}_mean_fd" for column in wide_mean.columns]
    wide_mean.to_csv(args.output_dir / "fd_per_run_wide_mean.csv", index=True, encoding="utf-8-sig")

    run_count = per_run.groupby("sub")["run_index"].max().rename("n_runs_sorted").reset_index()
    run_count.to_csv(args.output_dir / "check_n_runs_per_subject.csv", index=False, encoding="utf-8-sig")

    print(f"[DONE] Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

