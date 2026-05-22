#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build Python GLMsingle input bundles from fsLR 91k CIFTI dtseries files.

For each subject, this script creates a bundle containing:

- data/run-XX_data.npy: grayordinates x timepoints, float32
- design/run-XX_design.npz: sparse design matrix, timepoints x conditions
- condition_ids.npy: condition labels matching the design columns
- run_table.csv: file index used by run_glmsingle.py
- run_summary.csv, event_summary.csv, trial_table.csv, condition_summary.csv
- meta.json

Each unique image ID is treated as one condition. Blank and baseline conditions
are excluded by default.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import sparse as sp


@dataclass
class BundleConfig:
    cifti_dir: Path
    events_dir: Path
    output_dir: Path
    tr: float = 1.0
    stimdur: float | None = 1.0
    expected_grayordinates: int | None = 91282
    expected_task_labels: tuple[str, ...] = tuple(f"s{i}" for i in range(1, 12))
    expected_repeat_indexes: tuple[int, ...] = (1, 2, 3)
    expected_presentations_per_image: int = 3
    run_order: str = "repeat_then_task"
    prefix_task_to_condition: bool = False
    condition_column_candidates: tuple[str, ...] = ("image_id", "condition", "stim_id", "stimulus", "stimulus_id")
    exclude_condition_ids: tuple[str, ...] = ("blank", "baseline", "fixation", "fix", "rest", "20000", "30000", "0")
    trial_type_image_values: tuple[str, ...] | None = None
    onset_binning: str = "round"
    require_expected_task_repeat_grid: bool = True
    require_each_image_presented_n_times: bool = True
    fail_on_out_of_range_onset: bool = True
    fail_on_duplicate_same_cell: bool = True
    fail_on_multi_condition_same_tr: bool = True
    allow_nonfinite_data_to_zero: bool = False
    overwrite_existing_bundle: bool = False
    stop_on_first_error: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GLMsingle input bundles from CIFTI dtseries and events.tsv files.")
    parser.add_argument("--cifti-dir", required=True, type=Path)
    parser.add_argument("--events-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subjects", nargs="*", default=None, help="Subject IDs without or with 'sub-' prefix. Defaults to all subjects.")
    parser.add_argument("--tr", type=float, default=1.0)
    parser.add_argument("--stimdur", type=float, default=1.0, help="Stimulus duration in seconds. Use --infer-stimdur to infer from events.")
    parser.add_argument("--infer-stimdur", action="store_true", help="Infer a single stimulus duration from non-blank event durations.")
    parser.add_argument("--expected-grayordinates", type=int, default=91282)
    parser.add_argument("--expected-task-labels", default=",".join(f"s{i}" for i in range(1, 12)))
    parser.add_argument("--expected-repeat-indexes", default="1,2,3")
    parser.add_argument("--expected-presentations-per-image", type=int, default=3)
    parser.add_argument("--run-order", choices=["repeat_then_task", "task_then_repeat"], default="repeat_then_task")
    parser.add_argument("--condition-columns", default="image_id,condition,stim_id,stimulus,stimulus_id")
    parser.add_argument("--exclude-condition-ids", default="blank,baseline,fixation,fix,rest,20000,30000,0")
    parser.add_argument("--trial-type-image-values", default=None, help="Comma-separated trial_type values to retain. Default: retain all non-blank conditions.")
    parser.add_argument("--prefix-task-to-condition", action="store_true")
    parser.add_argument("--onset-binning", choices=["round", "floor", "ceil"], default="round")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing bundles.")
    parser.add_argument("--allow-nonfinite-data-to-zero", action="store_true")
    parser.add_argument("--no-grid-validation", action="store_true")
    parser.add_argument("--no-repeat-validation", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def split_csv(text: str | None) -> tuple[str, ...] | None:
    if text is None:
        return None
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    return values or None


def make_config(args: argparse.Namespace) -> BundleConfig:
    return BundleConfig(
        cifti_dir=args.cifti_dir,
        events_dir=args.events_dir,
        output_dir=args.output_dir,
        tr=float(args.tr),
        stimdur=None if args.infer_stimdur else float(args.stimdur),
        expected_grayordinates=args.expected_grayordinates,
        expected_task_labels=tuple(split_csv(args.expected_task_labels) or ()),
        expected_repeat_indexes=tuple(int(x) for x in (split_csv(args.expected_repeat_indexes) or ())),
        expected_presentations_per_image=int(args.expected_presentations_per_image),
        run_order=args.run_order,
        prefix_task_to_condition=bool(args.prefix_task_to_condition),
        condition_column_candidates=tuple(split_csv(args.condition_columns) or ()),
        exclude_condition_ids=tuple(split_csv(args.exclude_condition_ids) or ()),
        trial_type_image_values=split_csv(args.trial_type_image_values),
        onset_binning=args.onset_binning,
        require_expected_task_repeat_grid=not args.no_grid_validation,
        require_each_image_presented_n_times=not args.no_repeat_validation,
        allow_nonfinite_data_to_zero=bool(args.allow_nonfinite_data_to_zero),
        overwrite_existing_bundle=bool(args.overwrite),
        stop_on_first_error=not args.continue_on_error,
    )


def normalize_id(value: Any) -> str | None:
    """Normalize condition, image, and trial-type identifiers."""
    if pd.isna(value):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return text


def condition_sort_key(value: Any) -> tuple:
    text = str(value)
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return (0, int(number), "")
    except Exception:
        pass
    parts = [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]
    return (1, 0, parts)


def task_index_from_label(task: str) -> int:
    match = re.search(r"\d+", str(task))
    if match is None:
        raise ValueError(f"Cannot parse task index from task label: {task}")
    return int(match.group())


def repeat_index_from_ses(ses: str) -> int:
    match = re.search(r"\d", str(ses))
    if match is None:
        raise ValueError(f"Cannot parse repeat index from session label: {ses}")
    return int(match.group())


def parse_run_name(path: Path) -> dict[str, Any]:
    pattern = re.compile(r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_task-(?P<task>[^_]+)")
    match = pattern.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse sub/ses/task from filename: {path.name}")
    sub = match.group("sub")
    ses = match.group("ses")
    task = match.group("task")
    return {
        "sub": sub,
        "ses": ses,
        "task": task,
        "task_index": task_index_from_label(task),
        "repeat_index": repeat_index_from_ses(ses),
    }


def find_matching_event_file(config: BundleConfig, sub: str, ses: str, task: str) -> Path:
    target = f"sub-{sub}_ses-{ses}_task-{task}_events.tsv"
    hits = sorted(config.events_dir.rglob(target))
    if not hits:
        raise FileNotFoundError(f"Missing events.tsv: {target}")
    if len(hits) > 1:
        raise RuntimeError("Multiple matched event files:\n" + "\n".join(str(path) for path in hits))
    return hits[0]


def choose_condition_column(events: pd.DataFrame, config: BundleConfig) -> str:
    for column in config.condition_column_candidates:
        if column in events.columns:
            return column
    raise ValueError(
        "No image/condition column was found. "
        f"Existing columns: {list(events.columns)}; expected one of {config.condition_column_candidates}"
    )


def read_events(event_path: Path, task_label: str, config: BundleConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    events = pd.read_csv(event_path, sep="\t")
    events.columns = [str(column).strip() for column in events.columns]
    if "onset" not in events.columns:
        raise ValueError(f"{event_path} has no onset column")

    condition_column = choose_condition_column(events, config)
    used = events.copy()

    if config.trial_type_image_values is not None:
        if "trial_type" not in used.columns:
            raise ValueError("trial_type filtering was requested, but no trial_type column exists")
        allowed = {normalize_id(value) for value in config.trial_type_image_values}
        used["_trial_type_norm"] = used["trial_type"].apply(normalize_id)
        used = used[used["_trial_type_norm"].isin(allowed)].copy()

    used["_condition_raw"] = used[condition_column].apply(normalize_id)
    exclude = {normalize_id(value).lower() for value in config.exclude_condition_ids if normalize_id(value) is not None}
    used = used[used["_condition_raw"].notna()].copy()
    used = used[~used["_condition_raw"].str.lower().isin(exclude)].copy()

    if config.prefix_task_to_condition:
        used["_condition"] = used["_condition_raw"].apply(lambda value: f"{task_label}_{value}")
    else:
        used["_condition"] = used["_condition_raw"]

    used["onset"] = pd.to_numeric(used["onset"], errors="coerce")
    if used["onset"].isna().any():
        raise ValueError(f"Non-numeric onset values were found in {event_path}")
    if "duration" in used.columns:
        used["duration"] = pd.to_numeric(used["duration"], errors="coerce")
    return events, used, condition_column


def infer_stimdur(filtered_events: list[pd.DataFrame], config: BundleConfig) -> float:
    if config.stimdur is not None:
        return float(config.stimdur)
    durations: list[float] = []
    for events in filtered_events:
        if "duration" not in events.columns:
            continue
        values = pd.to_numeric(events["duration"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        durations.extend(values.tolist())
    if not durations:
        raise ValueError("Could not infer stimulus duration from event files; set --stimdur manually.")
    unique = np.unique(np.round(np.asarray(durations, dtype=float), 6))
    if len(unique) != 1:
        raise ValueError(f"Multiple stimulus durations were detected: {unique}. Set --stimdur manually.")
    return float(unique[0])


def discover_runs(config: BundleConfig) -> pd.DataFrame:
    pattern = "*_space-fsLR_den-91k_bold.dtseries.nii"
    cifti_files = sorted(config.cifti_dir.rglob(pattern))
    if not cifti_files:
        raise FileNotFoundError(f"No CIFTI files found under {config.cifti_dir} with pattern {pattern}")
    rows = []
    for cifti_file in cifti_files:
        info = parse_run_name(cifti_file)
        events_file = find_matching_event_file(config, info["sub"], info["ses"], info["task"])
        rows.append({**info, "cifti_path": cifti_file, "events_path": events_file})
    return pd.DataFrame(rows)


def sort_runs(runs: pd.DataFrame, config: BundleConfig) -> pd.DataFrame:
    if config.run_order == "repeat_then_task":
        columns = ["repeat_index", "task_index", "ses", "task"]
    elif config.run_order == "task_then_repeat":
        columns = ["task_index", "repeat_index", "task", "ses"]
    else:
        raise ValueError("run_order must be repeat_then_task or task_then_repeat")
    return runs.sort_values(columns).reset_index(drop=True)


def validate_task_repeat_grid(runs: pd.DataFrame, sub: str, config: BundleConfig) -> dict[str, Any]:
    expected = pd.MultiIndex.from_product(
        [config.expected_task_labels, config.expected_repeat_indexes], names=["task", "repeat_index"]
    ).to_frame(index=False)
    counts = runs.groupby(["task", "repeat_index"]).size().reset_index(name="n")
    grid = expected.merge(counts, on=["task", "repeat_index"], how="left")
    grid["n"] = grid["n"].fillna(0).astype(int)

    missing = grid[grid["n"] == 0]
    duplicated = grid[grid["n"] > 1]
    unexpected_tasks = sorted(set(runs["task"]) - set(config.expected_task_labels), key=condition_sort_key)
    unexpected_repeats = sorted(set(runs["repeat_index"]) - set(config.expected_repeat_indexes))
    expected_n = len(config.expected_task_labels) * len(config.expected_repeat_indexes)

    problems = []
    if not missing.empty:
        problems.append("Missing task-repeat runs:\n" + missing.to_string(index=False))
    if not duplicated.empty:
        problems.append("Duplicated task-repeat runs:\n" + duplicated.to_string(index=False))
    if unexpected_tasks:
        problems.append(f"Unexpected task labels: {unexpected_tasks}")
    if unexpected_repeats:
        problems.append(f"Unexpected repeat indexes: {unexpected_repeats}")
    if len(runs) != expected_n:
        problems.append(f"Expected {expected_n} runs, found {len(runs)} runs")

    if problems and config.require_expected_task_repeat_grid:
        raise RuntimeError(f"Subject sub-{sub} failed task-repeat validation.\n\n" + "\n\n".join(problems))

    return {
        "expected_n_runs": int(expected_n),
        "found_n_runs": int(len(runs)),
        "n_missing_task_repeat": int(len(missing)),
        "n_duplicated_task_repeat": int(len(duplicated)),
        "unexpected_tasks": unexpected_tasks,
        "unexpected_repeats": unexpected_repeats,
    }


def load_cifti_units_by_time(cifti_path: Path, config: BundleConfig) -> np.ndarray:
    """Load CIFTI dtseries data as grayordinates x timepoints."""
    img = nib.load(str(cifti_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"CIFTI dtseries must be 2D, got shape {data.shape}: {cifti_path}")

    time_axis = None
    try:
        for axis_index in range(len(img.shape)):
            if img.header.get_axis(axis_index).__class__.__name__ == "SeriesAxis":
                time_axis = axis_index
                break
    except Exception:
        time_axis = None

    if time_axis is None:
        time_axis = 0 if data.shape[0] < data.shape[1] else 1
    if time_axis == 0:
        data = data.T

    if config.expected_grayordinates is not None and data.shape[0] != config.expected_grayordinates:
        if data.shape[1] == config.expected_grayordinates:
            data = data.T
    if config.expected_grayordinates is not None and data.shape[0] != config.expected_grayordinates:
        raise ValueError(
            f"Unexpected grayordinate count in {cifti_path.name}: expected {config.expected_grayordinates}, got {data.shape}"
        )

    n_bad = int((~np.isfinite(data)).sum())
    if n_bad:
        if config.allow_nonfinite_data_to_zero:
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            raise ValueError(f"Non-finite values found in {cifti_path.name}: {n_bad}")
    return np.ascontiguousarray(data, dtype=np.float32)


def onset_to_tr(onset_sec: float, tr: float, config: BundleConfig) -> int:
    onset_tr_float = onset_sec / tr
    if config.onset_binning == "round":
        return int(np.floor(onset_tr_float + 0.5))
    if config.onset_binning == "floor":
        return int(np.floor(onset_tr_float))
    if config.onset_binning == "ceil":
        return int(np.ceil(onset_tr_float))
    raise ValueError("onset_binning must be round, floor, or ceil")


def build_design_sparse(
    used_events: pd.DataFrame,
    n_timepoints: int,
    cond_to_col: dict[str, int],
    run_name: str,
    run_index_1based: int,
    config: BundleConfig,
) -> tuple[sp.csc_matrix, dict[str, Any], list[dict[str, Any]]]:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    trial_rows: list[dict[str, Any]] = []
    out_of_range: list[tuple] = []
    seen_cells: set[tuple[int, int]] = set()
    n_duplicate_same_cell = 0
    onset_errors_sec: list[float] = []

    for local_trial_index, (original_event_index, row) in enumerate(used_events.iterrows(), start=1):
        condition = row["_condition"]
        onset_sec = float(row["onset"])
        onset_tr_float = onset_sec / config.tr
        onset_tr = onset_to_tr(onset_sec, config.tr, config)
        onset_error_sec = onset_tr * config.tr - onset_sec
        onset_errors_sec.append(onset_error_sec)

        if onset_tr < 0 or onset_tr >= n_timepoints:
            out_of_range.append((original_event_index, onset_sec, onset_tr, n_timepoints, condition))
            continue

        col = cond_to_col[condition]
        cell = (onset_tr, col)
        if cell in seen_cells:
            n_duplicate_same_cell += 1
        seen_cells.add(cell)

        rows.append(onset_tr)
        cols.append(col)
        vals.append(1.0)
        trial_rows.append(
            {
                "run_index_1based": int(run_index_1based),
                "local_trial_index": int(local_trial_index),
                "orig_event_index": int(original_event_index),
                "condition_id": str(condition),
                "condition_col_0based": int(col),
                "condition_col_1based": int(col + 1),
                "onset_sec": float(onset_sec),
                "onset_tr_float": float(onset_tr_float),
                "onset_tr_0based": int(onset_tr),
                "onset_tr_1based": int(onset_tr + 1),
                "onset_error_sec": float(onset_error_sec),
                "onset_binning": config.onset_binning,
            }
        )

    if out_of_range and config.fail_on_out_of_range_onset:
        examples = "\n".join(str(item) for item in out_of_range[:10])
        raise ValueError(f"Out-of-range event onsets in {run_name}. First examples:\n{examples}")

    design = sp.coo_matrix(
        (np.asarray(vals, dtype=np.float64), (rows, cols)),
        shape=(n_timepoints, len(cond_to_col)),
        dtype=np.float64,
    ).tocsc()
    design.sum_duplicates()
    if design.nnz:
        design.data[:] = 1.0

    if n_duplicate_same_cell and config.fail_on_duplicate_same_cell:
        raise ValueError(f"Duplicate same onset-condition cells found in {run_name}: {n_duplicate_same_cell}")

    row_nnz = np.asarray((design != 0).sum(axis=1)).ravel()
    n_multi_condition_tr = int(np.sum(row_nnz > 1))
    if n_multi_condition_tr and config.fail_on_multi_condition_same_tr:
        rows_with_multiple = np.where(row_nnz > 1)[0][:10]
        raise ValueError(
            f"Multiple conditions occur at the same TR in {run_name}: {n_multi_condition_tr} TRs. "
            f"First 0-based TRs: {rows_with_multiple.tolist()}"
        )

    onset_errors = np.asarray(onset_errors_sec, dtype=float)
    qc = {
        "n_events_used": int(len(used_events)),
        "n_design_nonzero": int(design.nnz),
        "n_out_of_range_onsets": int(len(out_of_range)),
        "n_duplicate_same_cell": int(n_duplicate_same_cell),
        "n_multi_condition_tr": int(n_multi_condition_tr),
        "onset_binning": config.onset_binning,
        "mean_abs_onset_error_sec": float(np.mean(np.abs(onset_errors))) if len(onset_errors) else 0.0,
        "max_abs_onset_error_sec": float(np.max(np.abs(onset_errors))) if len(onset_errors) else 0.0,
    }
    return design, qc, trial_rows


def prepare_bundle_dir(bundle_dir: Path, config: BundleConfig) -> tuple[Path, Path]:
    if bundle_dir.exists():
        if config.overwrite_existing_bundle:
            shutil.rmtree(bundle_dir)
        else:
            raise FileExistsError(f"Bundle already exists: {bundle_dir}. Use --overwrite to rebuild it.")
    data_dir = bundle_dir / "data"
    design_dir = bundle_dir / "design"
    data_dir.mkdir(parents=True, exist_ok=True)
    design_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, design_dir


def save_json(path: Path, obj: Any) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return list(value)
        return value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=convert)


def relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def build_one_subject(sub: str, all_runs: pd.DataFrame, config: BundleConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    subject_runs = all_runs[all_runs["sub"] == sub].copy()
    if subject_runs.empty:
        raise RuntimeError(f"No runs found for sub-{sub}")
    subject_runs = sort_runs(subject_runs, config)
    grid_qc = validate_task_repeat_grid(subject_runs, sub, config)

    filtered_events: list[pd.DataFrame] = []
    event_summary_rows: list[dict[str, Any]] = []
    for run_i, row in subject_runs.reset_index(drop=True).iterrows():
        all_events, used_events, condition_column = read_events(row["events_path"], row["task"], config)
        filtered_events.append(used_events)
        event_summary_rows.append(
            {
                "run_index_1based": int(run_i + 1),
                "sub": str(sub),
                "ses": str(row["ses"]),
                "task": str(row["task"]),
                "repeat_index": int(row["repeat_index"]),
                "task_index": int(row["task_index"]),
                "condition_column_used": condition_column,
                "n_events_all": int(len(all_events)),
                "n_events_used_after_excluding_blank": int(len(used_events)),
                "n_unique_conditions_in_run": int(used_events["_condition"].nunique()),
                "events_file": str(row["events_path"]),
            }
        )

    condition_ids = sorted(set().union(*[set(events["_condition"].tolist()) for events in filtered_events]), key=condition_sort_key)
    if not condition_ids:
        raise RuntimeError(f"sub-{sub}: no valid image conditions after event filtering")
    cond_to_col = {condition: i for i, condition in enumerate(condition_ids)}
    stimdur = infer_stimdur(filtered_events, config)

    bundle_dir = config.output_dir / f"sub-{sub}_glmsingle_33runs_bundle"
    data_dir, design_dir = prepare_bundle_dir(bundle_dir, config)
    np.save(bundle_dir / "condition_ids.npy", np.asarray(condition_ids, dtype=object), allow_pickle=True)

    n_cond = len(condition_ids)
    total_presentations = np.zeros(n_cond, dtype=np.int64)
    n_runs_present = np.zeros(n_cond, dtype=np.int64)
    tasks_present = [set() for _ in range(n_cond)]
    repeats_present = [set() for _ in range(n_cond)]

    run_table_rows: list[dict[str, Any]] = []
    run_summary_rows: list[dict[str, Any]] = []
    all_trial_rows: list[dict[str, Any]] = []
    subject_runs_reset = subject_runs.reset_index(drop=True)

    print(f"[SUBJECT] sub-{sub}: {len(subject_runs_reset)} runs, {n_cond} conditions")
    for run_i, row in subject_runs_reset.iterrows():
        run_index = run_i + 1
        run_name = f"sub-{sub}_ses-{row['ses']}_task-{row['task']}"
        print(f"[LOAD] {run_index:02d}/{len(subject_runs_reset):02d} {run_name}")

        data = load_cifti_units_by_time(row["cifti_path"], config)
        design, design_qc, trial_rows = build_design_sparse(
            filtered_events[run_i], data.shape[1], cond_to_col, run_name, run_index, config
        )
        if design.shape[0] != data.shape[1]:
            raise RuntimeError(f"Design/data time mismatch in {run_name}: design={design.shape}, data={data.shape}")

        data_file = data_dir / f"run-{run_index:02d}_data.npy"
        design_file = design_dir / f"run-{run_index:02d}_design.npz"
        np.save(data_file, data.astype(np.float32, copy=False))
        sp.save_npz(design_file, design.tocsc(), compressed=True)

        per_condition = np.asarray(design.sum(axis=0)).ravel().astype(np.int64)
        present = per_condition > 0
        total_presentations += per_condition
        n_runs_present += present.astype(np.int64)
        for col in np.where(present)[0]:
            tasks_present[col].add(str(row["task"]))
            repeats_present[col].add(int(row["repeat_index"]))

        run_table_rows.append(
            {
                "run_index_1based": int(run_index),
                "sub": str(sub),
                "ses": str(row["ses"]),
                "task": str(row["task"]),
                "repeat_index": int(row["repeat_index"]),
                "task_index": int(row["task_index"]),
                "data_file": relpath(data_file, bundle_dir),
                "design_file": relpath(design_file, bundle_dir),
                "data_shape_0_grayordinates": int(data.shape[0]),
                "data_shape_1_timepoints": int(data.shape[1]),
                "design_shape_0_timepoints": int(design.shape[0]),
                "design_shape_1_conditions": int(design.shape[1]),
                "design_nnz": int(design.nnz),
            }
        )
        run_summary_rows.append(
            {
                "run_index_1based": int(run_index),
                "sub": str(sub),
                "ses": str(row["ses"]),
                "task": str(row["task"]),
                "repeat_index": int(row["repeat_index"]),
                "task_index": int(row["task_index"]),
                "n_grayordinates": int(data.shape[0]),
                "n_timepoints": int(data.shape[1]),
                "n_conditions_total": int(n_cond),
                "cifti_file": str(row["cifti_path"]),
                "events_file": str(row["events_path"]),
                **design_qc,
            }
        )
        for trial_row in trial_rows:
            trial_row.update(
                {
                    "sub": str(sub),
                    "ses": str(row["ses"]),
                    "task": str(row["task"]),
                    "repeat_index": int(row["repeat_index"]),
                    "task_index": int(row["task_index"]),
                }
            )
            all_trial_rows.append(trial_row)

        del data, design
        gc.collect()

    run_table = pd.DataFrame(run_table_rows)
    run_summary = pd.DataFrame(run_summary_rows)
    event_summary = pd.DataFrame(event_summary_rows)
    trial_table = pd.DataFrame(all_trial_rows)
    condition_summary = pd.DataFrame(
        {
            "condition_col_0based": np.arange(n_cond, dtype=int),
            "condition_col_1based": np.arange(1, n_cond + 1, dtype=int),
            "condition_id": condition_ids,
            "n_presentations_total": total_presentations,
            "n_runs_present": n_runs_present,
            "tasks_present": [",".join(sorted(values, key=condition_sort_key)) for values in tasks_present],
            "repeats_present": [",".join(str(value) for value in sorted(values)) for values in repeats_present],
        }
    )

    run_table.to_csv(bundle_dir / "run_table.csv", index=False, encoding="utf-8-sig")
    run_summary.to_csv(bundle_dir / "run_summary.csv", index=False, encoding="utf-8-sig")
    event_summary.to_csv(bundle_dir / "event_summary.csv", index=False, encoding="utf-8-sig")
    trial_table.to_csv(bundle_dir / "trial_table.csv", index=False, encoding="utf-8-sig")
    condition_summary.to_csv(bundle_dir / "condition_summary.csv", index=False, encoding="utf-8-sig")

    repeat_count_distribution = condition_summary["n_presentations_total"].value_counts().sort_index().to_dict()
    meta = {
        "sub": str(sub),
        "bundle_dir": str(bundle_dir),
        "n_runs": int(len(subject_runs_reset)),
        "expected_n_runs": int(len(config.expected_task_labels) * len(config.expected_repeat_indexes)),
        "n_conditions": int(n_cond),
        "tr": float(config.tr),
        "stimdur": float(stimdur),
        "expected_grayordinates": config.expected_grayordinates,
        "data_format": "data/run-XX_data.npy; grayordinates x timepoints, float32",
        "design_format": "design/run-XX_design.npz; sparse CSC matrix, timepoints x conditions, float64",
        "prefix_task_to_condition": bool(config.prefix_task_to_condition),
        "run_order": config.run_order,
        "onset_binning": config.onset_binning,
        "task_labels_expected": list(config.expected_task_labels),
        "repeat_indexes_expected": list(config.expected_repeat_indexes),
        "grid_qc": grid_qc,
        "repeat_count_distribution": repeat_count_distribution,
        "config": asdict(config),
    }
    save_json(bundle_dir / "meta.json", meta)

    bad_repeat = condition_summary[
        condition_summary["n_presentations_total"] != config.expected_presentations_per_image
    ]
    if config.require_each_image_presented_n_times and not bad_repeat.empty:
        raise RuntimeError(
            f"sub-{sub}: {len(bad_repeat)} conditions were not presented "
            f"{config.expected_presentations_per_image} times. See {bundle_dir / 'condition_summary.csv'}"
        )

    print(f"[DONE] sub-{sub}: {bundle_dir}")
    return meta


def load_subject_bundle(bundle_dir: str | Path, dense_design: bool = True, mmap_data: bool = False):
    """Load a bundle created by this script for Python GLMsingle."""
    bundle_dir = Path(bundle_dir)
    with open(bundle_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    run_table = pd.read_csv(bundle_dir / "run_table.csv").sort_values("run_index_1based").reset_index(drop=True)
    condition_ids = np.load(bundle_dir / "condition_ids.npy", allow_pickle=True)
    data = []
    design = []
    mmap_mode = "r" if mmap_data else None
    for _, row in run_table.iterrows():
        data_path = Path(str(row["data_file"]))
        design_path = Path(str(row["design_file"]))
        if not data_path.is_absolute():
            data_path = bundle_dir / data_path
        if not design_path.is_absolute():
            design_path = bundle_dir / design_path
        data.append(np.load(data_path, mmap_mode=mmap_mode))
        matrix = sp.load_npz(design_path).tocsc()
        design.append(matrix.toarray().astype(np.float32) if dense_design else matrix)
    return data, design, float(meta["tr"]), float(meta["stimdur"]), condition_ids, run_table, meta


def main() -> None:
    args = parse_args()
    config = make_config(args)
    all_runs = discover_runs(config)
    all_subjects = sorted(all_runs["sub"].unique(), key=condition_sort_key)
    if args.subjects is None:
        subjects = all_subjects
    else:
        subjects = [str(sub).replace("sub-", "") for sub in args.subjects]

    master_rows: list[dict[str, Any]] = []
    for sub in subjects:
        try:
            meta = build_one_subject(sub, all_runs, config)
            master_rows.append(
                {"sub": str(sub), "status": "ok", "n_runs": meta["n_runs"], "n_conditions": meta["n_conditions"], "bundle_dir": meta["bundle_dir"], "error": ""}
            )
        except Exception as exc:
            print("[ERROR]")
            print(traceback.format_exc())
            master_rows.append({"sub": str(sub), "status": "error", "n_runs": "", "n_conditions": "", "bundle_dir": "", "error": str(exc)})
            config.output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(master_rows).to_csv(config.output_dir / "master_build_summary.csv", index=False, encoding="utf-8-sig")
            if config.stop_on_first_error:
                raise

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(master_rows).to_csv(config.output_dir / "master_build_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[MASTER SUMMARY] {config.output_dir / 'master_build_summary.csv'}")


if __name__ == "__main__":
    main()

