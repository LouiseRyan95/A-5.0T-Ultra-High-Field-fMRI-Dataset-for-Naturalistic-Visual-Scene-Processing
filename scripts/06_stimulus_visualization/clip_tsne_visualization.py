#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract CLIP image features and visualize image sets with PCA + t-SNE.

The script recursively reads images under an input directory, extracts features
using the OpenAI CLIP image encoder, optionally reduces features with PCA, runs
t-SNE, and writes both tables and figures. By default, images whose parent
directory is named `shared_1120` are plotted as one group and all other images
are plotted as another group.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "32")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from packaging.version import parse as version_parse
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class ImagePathDataset(Dataset):
    """Dataset that returns preprocessed images, paths, and binary group labels."""

    def __init__(self, image_dir: Path, preprocess, group_dir_name: str, extensions: tuple[str, ...]):
        self.paths = sorted({path for ext in extensions for path in image_dir.rglob(f"*{ext}")})
        if not self.paths:
            raise FileNotFoundError(f"No images were found in {image_dir} with extensions {extensions}")
        self.preprocess = preprocess
        self.group_dir_name = group_dir_name

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        parent = path.parent.name
        label = 1 if parent == self.group_dir_name else 0
        return self.preprocess(image), str(path), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP feature extraction and t-SNE visualization for stimulus images.")
    parser.add_argument("--image-dir", required=True, type=Path, help="Root image directory.")
    parser.add_argument("--output-prefix", required=True, type=Path, help="Output prefix without extension.")
    parser.add_argument("--group-dir-name", default="shared_1120", help="Parent directory name assigned to label=1.")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model name, e.g., ViT-B/32, ViT-L/14, RN50.")
    parser.add_argument("--feature-type", choices=["backbone", "proj"], default="backbone", help="Use visual backbone output or projected CLIP embedding.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--max-images", type=int, default=0, help="Use only the first N sorted images; 0 means all images.")
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=20.0)
    parser.add_argument("--tsne-iter", type=int, default=1500)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png,.webp", help="Comma-separated image extensions.")
    return parser.parse_args()


def l2_normalize(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize rows of a feature matrix."""
    norm = np.linalg.norm(array, axis=1, keepdims=True) + eps
    return array / norm


@torch.no_grad()
def extract_features(args: argparse.Namespace, device: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Extract CLIP image features for all images in the dataset."""
    try:
        import clip
    except ImportError as exc:
        raise ImportError(
            "OpenAI CLIP is required. Install it with: pip install git+https://github.com/openai/CLIP.git"
        ) from exc

    model, preprocess = clip.load(args.model, device=device, jit=False)
    model.eval()

    if device == "cuda" and args.use_fp16:
        model.half()
    else:
        model.float()

    extensions = tuple(ext.strip() for ext in args.extensions.split(",") if ext.strip())
    dataset = ImagePathDataset(args.image_dir, preprocess, args.group_dir_name, extensions)
    if args.max_images and args.max_images < len(dataset):
        dataset.paths = dataset.paths[: args.max_images]

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device == "cuda"), shuffle=False)
    target_dtype = next(model.parameters()).dtype
    features = []
    paths: list[str] = []
    labels = []

    for images, batch_paths, batch_labels in tqdm(dataloader, total=len(dataloader), desc=f"Extract {args.model} {args.feature_type}"):
        images = images.to(device, non_blocking=True).to(dtype=target_dtype)
        if args.feature_type == "proj":
            embedding = model.encode_image(images)
        else:
            embedding = model.visual(images)
        embedding = torch.nn.functional.normalize(embedding, dim=-1).cpu().numpy()
        features.append(embedding)
        paths.extend(list(batch_paths))
        labels.append(batch_labels.numpy())

    return np.concatenate(features, axis=0), paths, np.concatenate(labels, axis=0)


def make_tsne_kwargs(args: argparse.Namespace) -> dict:
    """Create scikit-learn-version-compatible TSNE keyword arguments."""
    kwargs = {
        "n_components": 2,
        "perplexity": args.perplexity,
        "init": "pca",
        "metric": args.metric,
        "random_state": args.random_state,
        "verbose": 1,
    }
    if version_parse(sklearn.__version__) >= version_parse("1.6"):
        kwargs["max_iter"] = args.tsne_iter
    else:
        kwargs["n_iter"] = args.tsne_iter
    if version_parse(sklearn.__version__) >= version_parse("1.2"):
        kwargs["learning_rate"] = "auto"
        kwargs["square_distances"] = True
    else:
        kwargs["learning_rate"] = 200
    signature = inspect.signature(TSNE.__init__)
    if "n_jobs" in signature.parameters:
        kwargs["n_jobs"] = 1
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def run_tsne(features: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Run PCA followed by t-SNE."""
    matrix = features.astype("float32")
    if args.pca_dim and args.pca_dim < matrix.shape[1]:
        matrix = PCA(n_components=args.pca_dim, random_state=args.random_state, svd_solver="auto").fit_transform(matrix)
    print(f"[t-SNE] N={matrix.shape[0]}, dim={matrix.shape[1]}, perplexity={args.perplexity}, metric={args.metric}")
    return TSNE(**make_tsne_kwargs(args)).fit_transform(matrix)


def save_outputs(features: np.ndarray, embedding_2d: np.ndarray, paths: list[str], labels: np.ndarray, args: argparse.Namespace) -> None:
    """Save features, path metadata, t-SNE coordinates, and figures."""
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{args.output_prefix}_features.npy", features.astype("float32"))

    with open(f"{args.output_prefix}_paths.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for path, label in zip(paths, labels):
            writer.writerow([path, int(label)])

    with open(f"{args.output_prefix}_2d.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label", "path"])
        for (x, y), label, path in zip(embedding_2d, labels, paths):
            writer.writerow([float(x), float(y), int(label), path])

    group_mask = labels == 1
    other_mask = ~group_mask
    plt.figure(figsize=(8, 8), dpi=160)
    plt.scatter(embedding_2d[other_mask, 0], embedding_2d[other_mask, 1], s=6, alpha=0.6, label="other_images")
    plt.scatter(embedding_2d[group_mask, 0], embedding_2d[group_mask, 1], s=12, alpha=0.8, label=args.group_dir_name)
    plt.legend(loc="best")
    plt.title(f"t-SNE of CLIP features: {args.group_dir_name} vs others")
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_2d.png", dpi=300)
    plt.savefig(f"{args.output_prefix}_2d.svg")
    plt.close()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")
    features, paths, labels = extract_features(args, device=device)
    features = l2_normalize(features)
    embedding_2d = run_tsne(features, args)
    save_outputs(features, embedding_2d, paths, labels, args)
    print(f"[DONE] Outputs written with prefix {args.output_prefix}")


if __name__ == "__main__":
    main()

