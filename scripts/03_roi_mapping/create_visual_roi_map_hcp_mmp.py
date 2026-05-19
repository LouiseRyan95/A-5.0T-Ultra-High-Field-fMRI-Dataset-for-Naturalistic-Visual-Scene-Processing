#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a 91k visual ROI map from an HCP-MMP dlabel file.

The script matches HCP-MMP label names to a user-defined list of visual ROI
keywords, assigns each keyword a compact integer ID, and saves:

- roimap_91k_visual_rois.mat
- roimap_91k_visual_rois_summary.csv
- roimap_91k_visual_rois_id_table.csv

If the source dlabel does not have 91,282 grayordinates, provide a 91k CIFTI
template using --template-91k so the map can be aligned by surface vertex IDs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio

DEFAULT_KEYWORDS = ["V1", "V2", "V3", "V4", "V8", "VMV", "PIT", "FFC", "VVC", "LO"]
EXPECTED_91K = 91282


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a visual ROI map from an HCP-MMP dlabel CIFTI file.")
    parser.add_argument("--label-path", required=True, type=Path, help="HCP-MMP dlabel CIFTI file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--template-91k", type=Path, default=None, help="Optional 91k dtseries/dscalar/dlabel template.")
    parser.add_argument("--keywords", nargs="+", default=DEFAULT_KEYWORDS, help="ROI keyword prefixes to include.")
    parser.add_argument("--expected-grayordinates", type=int, default=EXPECTED_91K, help="Expected output grayordinate count.")
    return parser.parse_args()


def get_label_table_from_dlabel(img: nib.Cifti2Image) -> dict[int, str]:
    """Extract the integer-to-label-name table from a dlabel CIFTI image."""
    try:
        axis0 = img.header.get_axis(0)
        label_table = axis0.label[0]
        return {int(key): str(value[0]) for key, value in label_table.items()}
    except Exception:
        pass

    index_map = img.header.get_index_map(0)
    named_maps = list(index_map.named_maps)
    if not named_maps:
        raise RuntimeError("No label table was found in the dlabel CIFTI file.")

    label_table = named_maps[0].label_table
    return {int(label.key): str(label.label) for label in label_table.labels}


def clean_hcp_label_name(name: str) -> str:
    """Normalize HCP-MMP label names such as L_V1_ROI or R_LO1_ROI."""
    text = str(name).upper().strip().replace(" ", "_")
    text = re.sub(r"^(LEFT|RIGHT|L|R)[_\-\.]+", "", text)
    text = re.sub(r"[_\-\.]*ROI$", "", text)
    return text


def match_keyword(label_name: str, keywords: list[str]) -> str | None:
    """Match a cleaned HCP label to the first keyword prefix."""
    core = clean_hcp_label_name(label_name)
    for keyword in keywords:
        keyword_upper = keyword.upper()
        if core == keyword_upper or core.startswith(keyword_upper):
            return keyword
    return None


def align_to_template_91k(source_roimap: np.ndarray, source_img: nib.Cifti2Image, template_img: nib.Cifti2Image) -> np.ndarray:
    """Align a surface dlabel vector to the BrainModelAxis of a 91k CIFTI template."""
    source_axis = source_img.header.get_axis(1)
    target_axis = template_img.header.get_axis(1)
    target_roimap = np.zeros((target_axis.size,), dtype=source_roimap.dtype)

    target_structures = {name: (slc, bm) for name, slc, bm in target_axis.iter_structures()}

    for structure_name, source_slice, source_bm in source_axis.iter_structures():
        if structure_name not in target_structures:
            continue
        target_slice, target_bm = target_structures[structure_name]
        if not hasattr(source_bm, "vertex") or source_bm.vertex is None:
            continue
        if not hasattr(target_bm, "vertex") or target_bm.vertex is None:
            continue

        source_vertices = np.asarray(source_bm.vertex)
        target_vertices = np.asarray(target_bm.vertex)
        source_values = source_roimap[source_slice]
        target_lookup = {int(vertex): target_slice.start + i for i, vertex in enumerate(target_vertices)}

        for vertex, value in zip(source_vertices, source_values):
            target_index = target_lookup.get(int(vertex))
            if target_index is not None:
                target_roimap[target_index] = value

    return target_roimap


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    keyword_to_id = {keyword: i + 1 for i, keyword in enumerate(args.keywords)}
    img = nib.load(str(args.label_path))
    data = np.asarray(img.get_fdata()).squeeze().astype(np.int32)
    if data.ndim != 1:
        raise ValueError(f"Expected a 1D dlabel vector, got shape {data.shape}")

    label_table = get_label_table_from_dlabel(img)
    oldkey_to_newid: dict[int, int] = {}
    summary_rows: list[dict] = []

    for old_key, label_name in label_table.items():
        if old_key == 0:
            continue
        matched_keyword = match_keyword(label_name, args.keywords)
        if matched_keyword is None:
            continue
        new_id = keyword_to_id[matched_keyword]
        oldkey_to_newid[old_key] = new_id
        summary_rows.append(
            {
                "old_label_key": int(old_key),
                "old_label_name": label_name,
                "clean_name": clean_hcp_label_name(label_name),
                "target_keyword": matched_keyword,
                "new_roi_id": int(new_id),
                "n_grayordinates": int(np.sum(data == old_key)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise RuntimeError("No labels matched the requested ROI keywords.")

    roimap = np.zeros_like(data, dtype=np.int16)
    for old_key, new_id in oldkey_to_newid.items():
        roimap[data == old_key] = new_id

    if roimap.size != args.expected_grayordinates:
        if args.template_91k is None:
            raise ValueError(
                f"The source roimap length is {roimap.size}, not {args.expected_grayordinates}. "
                "Provide --template-91k to align the map to the target 91k space."
            )
        template_img = nib.load(str(args.template_91k))
        roimap = align_to_template_91k(roimap, img, template_img)

    if roimap.size != args.expected_grayordinates:
        raise ValueError(f"Final roimap length is {roimap.size}, expected {args.expected_grayordinates}")

    roimask = (roimap > 0).astype(np.uint8)
    roi_id_table = pd.DataFrame(
        {
            "roi_id": [keyword_to_id[keyword] for keyword in args.keywords],
            "roi_name": args.keywords,
        }
    )

    out_mat = args.output_dir / "roimap_91k_visual_rois.mat"
    out_summary = args.output_dir / "roimap_91k_visual_rois_summary.csv"
    out_id_table = args.output_dir / "roimap_91k_visual_rois_id_table.csv"

    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    roi_id_table.to_csv(out_id_table, index=False, encoding="utf-8-sig")
    sio.savemat(
        out_mat,
        {
            "roimap": roimap.astype(np.int16),
            "roimask": roimask.astype(np.uint8),
            "roi_ids": np.arange(1, len(args.keywords) + 1, dtype=np.int16),
            "roi_names": np.array(args.keywords, dtype=object).reshape(-1, 1),
            "source_dlabel": str(args.label_path),
        },
        do_compression=True,
        oned_as="row",
    )

    print(f"[DONE] ROI map saved: {out_mat}")
    print(f"[DONE] ROI summary saved: {out_summary}")
    print(f"[DONE] ROI ID table saved: {out_id_table}")
    print(f"[QC] target grayordinates: {int(roimask.sum())}")


if __name__ == "__main__":
    main()

