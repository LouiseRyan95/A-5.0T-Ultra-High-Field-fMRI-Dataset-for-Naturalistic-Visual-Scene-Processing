#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute NSD-style ncsnr and noise ceiling from GLMsingle trial-wise betas.

Core steps
----------
1. Load trial-wise GLMsingle betas from TYPED_FITHRF_GLMDENOISE_RR.npy.
2. Reconstruct the GLMsingle trial table from the bundle design matrices.
3. Z-score each grayordinate's trial-wise betas within each scan session.
4. Use image conditions with exactly three presentations.
5. Estimate repeat noise variance, ncsnr, and noise ceiling.
6. Summarize voxel/grayordinate-wise maps within ROIs.

The main comparison metric is the median ncsnr across voxels/grayordinates
within a mask or ROI.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse as sp

DEFAULT_EXCLUDE = ("20000", "30000", "-1000", "blank", "Blank", "baseline", "Baseline", "fixation", "Fixation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute NSD-style ncsnr and noise ceiling.")
    parser.add_argument("--bundle-dir", required=True, type=Path, help="GLMsingle bundle directory.")
    parser.add_argument("--glmsingle-out-dir", required=True, type=Path, help="Directory containing TYPED_FITHRF_GLMDENOISE_RR.npy.")
    parser.add_argument("--roimap-mat", required=True, type=Path, help="MAT file containing roimap.")
    parser.add_argument("--roi-table-csv", required=True, type=Path, help="CSV with roi_id and roi_name columns.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: glmsingle-out-dir/noise_ceiling_nsd_style.")
    parser.add_argument("--n-repeats-required", type=int, default=3)
    parser.add_argument("--nc-avg-list", default="1,2,3", help="Comma-separated trial-average counts used for noise ceiling.")
    parser.add_argument("--exclude-condition-ids", default=",".join(DEFAULT_EXCLUDE))
    parser.add_argument("--chunklen", type=int, default=10000)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args()


def resolve_path(path_value: Any, base_dir: Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else base_dir / path


def normalize_id(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return text


def matlab_cellstr_to_list(value: Any) -> list[str]:
    array = np.asarray(value).squeeze()
    out = []
    for item in array:
        if isinstance(item, np.ndarray):
            item = item.squeeze()
            if item.size == 1:
                try:
                    out.append(str(item.item()))
                except Exception:
                    out.append(str(item))
            else:
                try:
                    out.append("".join(item.astype(str).ravel().tolist()))
                except Exception:
                    out.append(str(item))
        else:
            out.append(str(item))
    return out


def load_bundle_design_only(bundle_dir: Path):
    with open(bundle_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    run_table = pd.read_csv(bundle_dir / "run_table.csv").sort_values("run_index_1based").reset_index(drop=True)
    condition_ids = np.load(bundle_dir / "condition_ids.npy", allow_pickle=True)
    design = []
    for _, row in run_table.iterrows():
        design_file = resolve_path(row["design_file"], bundle_dir)
        design.append(sp.load_npz(design_file).tocsr())
    return design, condition_ids, run_table, meta


def infer_session_labels_for_runs(run_table: pd.DataFrame) -> np.ndarray:
    """Infer scan-session labels for within-session beta z-scoring."""
    for column in ["scan_session", "session", "session_id", "ses", "ses_id", "repeat_index"]:
        if column in run_table.columns:
            print(f"[INFO] Using run_table['{column}'] as the z-scoring session label.")
            return run_table[column].to_numpy().astype(str)
    print("[WARNING] No session-like column found; using run_index_1based as the session label.")
    return run_table["run_index_1based"].to_numpy().astype(str)


def build_trial_table_from_design(design: list[sp.csr_matrix], condition_ids: np.ndarray, run_table: pd.DataFrame) -> pd.DataFrame:
    """Build a trial table in the exact beta order expected by GLMsingle."""
    session_labels = infer_session_labels_for_runs(run_table)
    rows = []
    trial_index = 0
    for run_row_index, matrix in enumerate(design):
        matrix = matrix.tocsr()
        run_index = int(run_table.iloc[run_row_index]["run_index_1based"])
        session_label = str(session_labels[run_row_index])
        for time_index in range(matrix.shape[0]):
            cols = matrix[time_index].indices
            if len(cols) == 0:
                continue
            if len(cols) > 1:
                raise ValueError(f"Run {run_index}, time row {time_index}: more than one active condition: {cols[:10]}")
            col = int(cols[0])
            condition_id = condition_ids[col]
            rows.append(
                {
                    "trial_index_0based": int(trial_index),
                    "run_index_1based": run_index,
                    "run_row_0based": int(run_row_index),
                    "time_index": int(time_index),
                    "condition_col_0based": col,
                    "condition_id": condition_id,
                    "condition_id_norm": normalize_id(condition_id),
                    "session_label": session_label,
                }
            )
            trial_index += 1
    return pd.DataFrame(rows)


def select_exact_repeat_conditions(trial_table: pd.DataFrame, exclude_condition_ids: set[str], n_repeats_required: int):
    """Select conditions with exactly the requested number of repeated trials."""
    exclude_set = {normalize_id(value) for value in exclude_condition_ids}
    valid = trial_table.loc[~trial_table["condition_id_norm"].isin(exclude_set)].copy()
    count_table = (
        valid.groupby(["condition_col_0based", "condition_id_norm"], sort=True)
        .size()
        .reset_index(name="n_repeats")
    )
    print("[INFO] Repeat-count distribution before exact-repeat selection:")
    print(count_table["n_repeats"].value_counts().sort_index())

    exact = count_table.loc[count_table["n_repeats"] == n_repeats_required].copy()
    if exact.empty:
        raise RuntimeError(f"No condition has exactly {n_repeats_required} repeats.")

    rep_rows = []
    used_rows = []
    for _, row in exact.iterrows():
        col = int(row["condition_col_0based"])
        subset = valid.loc[valid["condition_col_0based"] == col].sort_values("trial_index_0based")
        indices = subset["trial_index_0based"].to_numpy(dtype=np.int64)
        if len(indices) != n_repeats_required:
            raise RuntimeError("Internal repeat-count mismatch.")
        rep_rows.append(indices)
        used_rows.append(
            {
                "condition_col_0based": col,
                "condition_id_norm": str(row["condition_id_norm"]),
                "n_repeats": int(len(indices)),
                **{f"trial_index_repeat{i + 1}_0based": int(indices[i]) for i in range(n_repeats_required)},
                **{f"session_repeat{i + 1}": str(subset.iloc[i]["session_label"]) for i in range(n_repeats_required)},
            }
        )
    rep_index_matrix = np.asarray(rep_rows, dtype=np.int64)
    used_conditions = pd.DataFrame(used_rows)
    print(f"[INFO] Conditions/images used: {len(used_conditions)}")
    return rep_index_matrix, used_conditions, count_table


def load_glmsingle_b3_betas(glmsingle_out_dir: Path) -> np.ndarray:
    path = glmsingle_out_dir / "TYPED_FITHRF_GLMDENOISE_RR.npy"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find GLMsingle b3 file: {path}")
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    if isinstance(obj, dict):
        if "betasmd" not in obj:
            raise KeyError(f"Expected key 'betasmd'. Available keys: {list(obj.keys())}")
        return obj["betasmd"]
    return obj


def coerce_betas_to_2d(betas: np.ndarray, n_gray_expected: int | None = None) -> np.ndarray:
    """Convert betas to grayordinates x trials."""
    arr = np.asarray(betas)
    if arr.ndim == 2:
        if n_gray_expected is not None:
            if arr.shape[0] == n_gray_expected:
                return arr
            if arr.shape[1] == n_gray_expected:
                return arr.T
        return arr
    return arr.reshape(-1, arr.shape[-1])


def load_roimap(roimap_mat: Path, roi_table_csv: Path):
    mat = sio.loadmat(roimap_mat)
    if "roimap" not in mat:
        raise KeyError(f"'roimap' not found in {roimap_mat}. Keys: {list(mat.keys())}")
    roimap = np.asarray(mat["roimap"]).squeeze().astype(np.int32)

    if roi_table_csv.exists():
        roi_table = pd.read_csv(roi_table_csv)
        roi_table.columns = [column.strip() for column in roi_table.columns]
        if not {"roi_id", "roi_name"}.issubset(roi_table.columns):
            raise KeyError("ROI table must contain roi_id and roi_name columns.")
        roi_table["roi_id"] = roi_table["roi_id"].astype(int)
        roi_table["roi_name"] = roi_table["roi_name"].astype(str)
    else:
        roi_ids = np.asarray(mat["roi_ids"]).squeeze().astype(int)
        roi_names = matlab_cellstr_to_list(mat["roi_names"])
        roi_table = pd.DataFrame({"roi_id": roi_ids, "roi_name": roi_names})
    return roimap, roi_table.sort_values("roi_id").reset_index(drop=True)


def compute_nsd_style_ncsnr_chunked(
    betas2d: np.ndarray,
    trial_table: pd.DataFrame,
    rep_index_matrix: np.ndarray,
    chunklen: int,
    nc_avg_list: tuple[int, ...],
    eps: float,
):
    """Compute NSD-style ncsnr in chunks over grayordinates."""
    n_gray, n_trials = betas2d.shape
    if trial_table.shape[0] != n_trials:
        raise ValueError(f"trial_table rows {trial_table.shape[0]} != beta trials {n_trials}")
    if rep_index_matrix.shape[1] != 3:
        raise ValueError(f"This NSD-style calculation expects exactly 3 repeats, got {rep_index_matrix.shape}")

    session_labels = trial_table["session_label"].astype(str).to_numpy()
    unique_sessions = np.unique(session_labels)
    print(f"[INFO] Number of z-score sessions: {len(unique_sessions)}")

    out = {
        "noise_sd": np.full(n_gray, np.nan, dtype=np.float32),
        "noise_var": np.full(n_gray, np.nan, dtype=np.float32),
        "signal_sd": np.full(n_gray, np.nan, dtype=np.float32),
        "signal_var": np.full(n_gray, np.nan, dtype=np.float32),
        "ncsnr": np.full(n_gray, np.nan, dtype=np.float32),
    }
    for n_avg in nc_avg_list:
        out[f"noiseceiling_avg{n_avg}_percent"] = np.full(n_gray, np.nan, dtype=np.float32)
        out[f"noiseceiling_avg{n_avg}_r"] = np.full(n_gray, np.nan, dtype=np.float32)

    for start in range(0, n_gray, chunklen):
        end = min(start + chunklen, n_gray)
        chunk_n = end - start
        betas = np.asarray(betas2d[start:end, :], dtype=np.float64)
        betas_z = np.full_like(betas, np.nan, dtype=np.float64)

        for session in unique_sessions:
            trial_idx = np.flatnonzero(session_labels == session)
            values = betas[:, trial_idx]
            mean = np.nanmean(values, axis=1, keepdims=True)
            std = np.nanstd(values, axis=1, ddof=0, keepdims=True)
            valid = np.isfinite(std[:, 0]) & (std[:, 0] > eps)
            if np.any(valid):
                betas_z[np.ix_(valid, trial_idx)] = (values[valid, :] - mean[valid, :]) / std[valid, :]

        vals3 = betas_z[:, rep_index_matrix]
        var_across_repeats = np.nanvar(vals3, axis=2, ddof=1)
        noise_var = np.nanmean(var_across_repeats, axis=1)
        noise_sd = np.sqrt(noise_var)
        signal_var = 1.0 - noise_var
        signal_var[~np.isfinite(signal_var)] = np.nan
        signal_var[signal_var < 0] = 0.0
        signal_sd = np.sqrt(signal_var)
        ncsnr = np.full(chunk_n, np.nan, dtype=np.float64)
        valid = np.isfinite(noise_sd) & (noise_sd > eps) & np.isfinite(signal_sd)
        ncsnr[valid] = signal_sd[valid] / noise_sd[valid]

        out["noise_sd"][start:end] = noise_sd.astype(np.float32)
        out["noise_var"][start:end] = noise_var.astype(np.float32)
        out["signal_sd"][start:end] = signal_sd.astype(np.float32)
        out["signal_var"][start:end] = signal_var.astype(np.float32)
        out["ncsnr"][start:end] = ncsnr.astype(np.float32)

        ncsnr2 = ncsnr ** 2
        for n_avg in nc_avg_list:
            nc_percent = 100.0 * ncsnr2 / (ncsnr2 + 1.0 / float(n_avg))
            out[f"noiseceiling_avg{n_avg}_percent"][start:end] = nc_percent.astype(np.float32)
            out[f"noiseceiling_avg{n_avg}_r"][start:end] = np.sqrt(nc_percent / 100.0).astype(np.float32)
        print(f"[NC] grayordinates {start:,}-{end:,}/{n_gray:,}")
    return out


def summarize_maps_by_roimap(roimap: np.ndarray, roi_table: pd.DataFrame, maps: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for _, roi in roi_table.iterrows():
        roi_id = int(roi["roi_id"])
        roi_name = str(roi["roi_name"])
        mask = roimap == roi_id
        row = {"roi_id": roi_id, "roi_name": roi_name, "n_grayordinates": int(mask.sum())}
        for name, array in maps.items():
            values = np.asarray(array)[mask]
            values = values[np.isfinite(values)]
            row[f"{name}_n_finite"] = int(len(values))
            if len(values):
                row[f"{name}_mean"] = float(np.mean(values))
                row[f"{name}_median"] = float(np.median(values))
                row[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                row[f"{name}_p05"] = float(np.percentile(values, 5))
                row[f"{name}_p25"] = float(np.percentile(values, 25))
                row[f"{name}_p75"] = float(np.percentile(values, 75))
                row[f"{name}_p95"] = float(np.percentile(values, 95))
            else:
                for suffix in ["mean", "median", "std", "p05", "p25", "p75", "p95"]:
                    row[f"{name}_{suffix}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_combined_visual_mask(roimap: np.ndarray, maps: dict[str, np.ndarray]) -> pd.DataFrame:
    mask = roimap > 0
    rows = []
    for metric in ["ncsnr", "noiseceiling_avg1_percent", "noiseceiling_avg1_r", "noiseceiling_avg3_percent", "noiseceiling_avg3_r"]:
        if metric not in maps:
            continue
        values = maps[metric][mask]
        values = values[np.isfinite(values)]
        rows.append(
            {
                "mask_name": "all_roimap_visual_rois",
                "metric": metric,
                "n_grayordinates": int(mask.sum()),
                "n_finite": int(len(values)),
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "p25": float(np.percentile(values, 25)) if len(values) else np.nan,
                "p75": float(np.percentile(values, 75)) if len(values) else np.nan,
                "p95": float(np.percentile(values, 95)) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.glmsingle_out_dir / "noise_ceiling_nsd_style")
    output_dir.mkdir(parents=True, exist_ok=True)
    nc_avg_list = tuple(int(value) for value in args.nc_avg_list.split(",") if value.strip())
    exclude_condition_ids = {value.strip() for value in args.exclude_condition_ids.split(",") if value.strip()}

    design, condition_ids, run_table, meta = load_bundle_design_only(args.bundle_dir)
    trial_table = build_trial_table_from_design(design, condition_ids, run_table)
    trial_table.to_csv(output_dir / "trial_table_for_nsd_style_ncsnr.csv", index=False, encoding="utf-8-sig")

    rep_index_matrix, used_conditions, count_table = select_exact_repeat_conditions(
        trial_table, exclude_condition_ids, args.n_repeats_required
    )
    used_conditions.to_csv(output_dir / "conditions_used_exactly3_for_nsd_style_ncsnr.csv", index=False, encoding="utf-8-sig")
    count_table.to_csv(output_dir / "condition_repeat_count_table.csv", index=False, encoding="utf-8-sig")

    roimap, roi_table = load_roimap(args.roimap_mat, args.roi_table_csv)
    betas = load_glmsingle_b3_betas(args.glmsingle_out_dir)
    betas2d = coerce_betas_to_2d(betas, n_gray_expected=len(roimap))
    if betas2d.shape[0] != len(roimap):
        raise ValueError(f"Beta grayordinate dimension {betas2d.shape[0]} does not match roimap length {len(roimap)}")
    if betas2d.shape[1] != len(trial_table):
        raise ValueError(f"Beta trial dimension {betas2d.shape[1]} does not match reconstructed trials {len(trial_table)}")

    del design
    gc.collect()

    maps = compute_nsd_style_ncsnr_chunked(betas2d, trial_table, rep_index_matrix, args.chunklen, nc_avg_list, args.eps)

    save_dict = {
        "ncsnr": maps["ncsnr"],
        "noise_sd": maps["noise_sd"],
        "noise_var": maps["noise_var"],
        "signal_sd": maps["signal_sd"],
        "signal_var": maps["signal_var"],
        "roimap": roimap,
        "roi_ids": roi_table["roi_id"].to_numpy(),
        "roi_names": roi_table["roi_name"].to_numpy(),
        "rep_index_matrix": rep_index_matrix,
        "used_condition_cols": used_conditions["condition_col_0based"].to_numpy(),
        "used_condition_ids": used_conditions["condition_id_norm"].to_numpy(),
    }
    for n_avg in nc_avg_list:
        save_dict[f"noiseceiling_avg{n_avg}_percent"] = maps[f"noiseceiling_avg{n_avg}_percent"]
        save_dict[f"noiseceiling_avg{n_avg}_r"] = maps[f"noiseceiling_avg{n_avg}_r"]
    np.savez_compressed(output_dir / "nsd_style_ncsnr_noiseceiling_91k.npz", **save_dict)

    mat_dict = save_dict.copy()
    mat_dict["roi_names"] = roi_table["roi_name"].astype(str).to_numpy(dtype=object)
    sio.savemat(output_dir / "nsd_style_ncsnr_noiseceiling_91k.mat", mat_dict)

    maps_for_summary = {key: value for key, value in maps.items() if key in save_dict or key.startswith("noiseceiling")}
    roi_summary = summarize_maps_by_roimap(roimap, roi_table, maps_for_summary)
    roi_summary.to_csv(output_dir / "roi_summary_nsd_style_ncsnr.csv", index=False, encoding="utf-8-sig")

    visual_summary = summarize_combined_visual_mask(roimap, maps)
    visual_summary.to_csv(output_dir / "all_visual_rois_summary_nsd_style_ncsnr.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] Outputs written to {output_dir}")


if __name__ == "__main__":
    main()

