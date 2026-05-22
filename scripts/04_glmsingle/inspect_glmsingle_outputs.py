#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inspect GLMsingle TYPED output fields and plot grayordinate-level histograms.

The script loads a GLMsingle `TYPED_FITHRF_GLMDENOISE_RR.npy` file, reduces each
requested field to one value per grayordinate when needed, and saves histogram
figures plus a summary table.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GLMsingle TYPED output distributions.")
    parser.add_argument("--typed-path", required=True, type=Path, help="Path to TYPED_FITHRF_GLMDENOISE_RR.npy")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for figures and summaries.")
    parser.add_argument("--fields", default="betasmd,R2,HRFindex,FRACvalue", help="Comma-separated fields to plot.")
    parser.add_argument("--expected-grayordinates", type=int, default=91282)
    parser.add_argument("--reduce-method", choices=["mean", "median"], default="mean")
    return parser.parse_args()


def load_typed(path: Path) -> dict:
    """Load a GLMsingle TYPED npy file."""
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dict-like GLMsingle TYPED object, got {type(obj)}")
    return obj


def infer_n_gray(typed: dict, expected: int, field: str = "betasmd") -> int:
    """Infer the grayordinate dimension from a GLMsingle output field."""
    array = np.squeeze(np.asarray(typed[field]))
    if array.ndim == 1:
        return int(array.shape[0])
    if array.ndim == 2:
        if expected in array.shape:
            return int(expected)
        return int(max(array.shape))
    raise ValueError(f"Cannot infer grayordinate count from {field} shape {array.shape}")


def to_grayordinate_vector(array: np.ndarray, n_gray: int, reduce_method: str) -> np.ndarray:
    """Convert a GLMsingle output array to one vector of length n_gray."""
    values = np.squeeze(np.asarray(array))
    if values.ndim == 1:
        if values.shape[0] != n_gray:
            raise ValueError(f"1D array length {values.shape[0]} does not match n_gray={n_gray}")
        return values
    if values.ndim == 2:
        if values.shape[0] == n_gray:
            condition_axis = 1
        elif values.shape[1] == n_gray:
            condition_axis = 0
        else:
            raise ValueError(f"Cannot identify grayordinate axis from shape {values.shape}")
        reducer = np.nanmean if reduce_method == "mean" else np.nanmedian
        return reducer(values, axis=condition_axis)
    raise ValueError(f"Unsupported array dimension: {values.ndim}, shape={values.shape}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    typed = load_typed(args.typed_path)
    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    n_gray = infer_n_gray(typed, args.expected_grayordinates, field="betasmd") if "betasmd" in typed else args.expected_grayordinates

    summary_rows = []
    n_cols = 2
    n_rows = math.ceil(len(fields) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows), squeeze=False)

    for i, field in enumerate(fields):
        ax = axes[i // n_cols][i % n_cols]
        if field not in typed:
            ax.set_title(f"{field} not found")
            ax.axis("off")
            summary_rows.append({"field": field, "status": "missing"})
            continue
        vector = to_grayordinate_vector(typed[field], n_gray=n_gray, reduce_method=args.reduce_method)
        finite = vector[np.isfinite(vector)]
        ax.hist(finite, bins=100)
        ax.set_title(field if field != "betasmd" else f"betasmd ({args.reduce_method} across conditions)")
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of grayordinates")
        summary_rows.append(
            {
                "field": field,
                "status": "ok",
                "n_grayordinates": int(n_gray),
                "n_finite": int(len(finite)),
                "mean": float(np.mean(finite)) if len(finite) else np.nan,
                "median": float(np.median(finite)) if len(finite) else np.nan,
                "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                "p05": float(np.percentile(finite, 5)) if len(finite) else np.nan,
                "p95": float(np.percentile(finite, 95)) if len(finite) else np.nan,
            }
        )

    for j in range(len(fields), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.tight_layout()
    fig.savefig(args.output_dir / "glmsingle_typed_field_histograms.png", dpi=300)
    fig.savefig(args.output_dir / "glmsingle_typed_field_histograms.svg")
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "glmsingle_typed_field_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

