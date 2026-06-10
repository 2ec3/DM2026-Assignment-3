#!/usr/bin/env python3
"""
DM2026 Assignment 3 - one-file train/predict script.

Run from the repository root:
    python run_dm2026_assignment3.py

It will:
1) train a Gaussian Naive Bayes model on data/train/User_*/.csv
2) save the model to dm2026_gnb_model.npz
3) predict data/test/User_*/.csv
4) write submission.csv with columns: Id,Label

This script is designed to be robust and fast enough to run directly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


G_TO_MS2 = 9.81
EPS = 1e-6


def discover_csvs(root: Path) -> List[Path]:
    files = sorted(root.glob("User_*/*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {root}")
    return files


def load_and_engineer(path: Path) -> Tuple[np.ndarray, int, int]:
    """
    Returns:
        X: shape (T, F=12) float32
        label: int, or -1 if not present
        file_id: int if present, else stem parsed as int, or -1
    """
    df = pd.read_csv(path)

    # Required columns
    accel_cols = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
    for col in accel_cols:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in {path}")

    # Keep the exact scaling used in the baseline code
    df[accel_cols] = df[accel_cols] * G_TO_MS2

    # Temporal features: speed and position
    mean_x = df["mean_x"].to_numpy(dtype=np.float32)
    mean_y = df["mean_y"].to_numpy(dtype=np.float32)
    mean_z = df["mean_z"].to_numpy(dtype=np.float32)

    speed_x = np.cumsum(mean_x, dtype=np.float32)
    speed_y = np.cumsum(mean_y, dtype=np.float32)
    speed_z = np.cumsum(mean_z, dtype=np.float32)

    position_x = np.cumsum(speed_x, dtype=np.float32)
    position_y = np.cumsum(speed_y, dtype=np.float32)
    position_z = np.cumsum(speed_z, dtype=np.float32)

    X = np.column_stack(
        [
            mean_x,
            mean_y,
            mean_z,
            df["std_x"].to_numpy(dtype=np.float32),
            df["std_y"].to_numpy(dtype=np.float32),
            df["std_z"].to_numpy(dtype=np.float32),
            speed_x,
            speed_y,
            speed_z,
            position_x,
            position_y,
            position_z,
        ]
    ).astype(np.float32, copy=False)

    label = int(df["label"].iloc[0]) if "label" in df.columns else -1

    if "file_id" in df.columns:
        file_id = int(df["file_id"].iloc[0])
    else:
        # fallback: try to parse the filename stem
        try:
            file_id = int(path.stem)
        except Exception:
            file_id = -1

    return X, label, file_id


class GaussianNBTimeSeries:
    def __init__(self) -> None:
        self.labels_: np.ndarray | None = None
        self.label_to_index_: Dict[int, int] = {}
        self.counts_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None
        self.num_seconds_: int | None = None
        self.num_features_: int = 12

    def fit(self, train_files: List[Path]) -> "GaussianNBTimeSeries":
        if not train_files:
            raise ValueError("No training files provided")

        # First pass: discover shapes and labels
        first_X, first_label, _ = load_and_engineer(train_files[0])
        T, F = first_X.shape
        if F != 12:
            raise ValueError(f"Expected 12 engineered features, got {F}")

        labels = []
        # We need labels only; also make sure the time length is consistent
        all_rows = [(train_files[0], first_X, first_label)]
        labels.append(first_label)

        for path in train_files[1:]:
            X, y, _ = load_and_engineer(path)
            if X.shape[0] != T:
                raise ValueError(f"Inconsistent sequence length in {path}: {X.shape[0]} vs {T}")
            all_rows.append((path, X, y))
            labels.append(y)

        unique_labels = np.array(sorted(set(labels)), dtype=int)
        self.labels_ = unique_labels
        self.label_to_index_ = {int(lbl): i for i, lbl in enumerate(unique_labels)}
        C = len(unique_labels)

        self.counts_ = np.zeros(C, dtype=np.int64)
        self.means_ = np.zeros((C, F, T), dtype=np.float64)
        self.vars_ = np.zeros((C, F, T), dtype=np.float64)
        self.num_seconds_ = T

        # Accumulate sums and sumsq
        sums = np.zeros((C, F, T), dtype=np.float64)
        sumsq = np.zeros((C, F, T), dtype=np.float64)

        for _, X, y in all_rows:
            c = self.label_to_index_[int(y)]
            self.counts_[c] += 1
            X64 = X.astype(np.float64, copy=False).T  # (F, T)
            sums[c] += X64
            sumsq[c] += X64 * X64

        for c in range(C):
            n = max(int(self.counts_[c]), 1)
            self.means_[c] = sums[c] / n
            self.vars_[c] = sumsq[c] / n - self.means_[c] ** 2
            self.vars_[c] = np.maximum(self.vars_[c], EPS)

        return self

    def predict_one(self, X: np.ndarray) -> int:
        if self.labels_ is None or self.counts_ is None or self.means_ is None or self.vars_ is None:
            raise RuntimeError("Model is not fitted")

        if X.shape != (self.num_seconds_, self.num_features_):
            raise ValueError(f"Unexpected input shape {X.shape}; expected {(self.num_seconds_, self.num_features_)}")

        Xf = X.astype(np.float64, copy=False).T  # (F, T)
        total = float(self.counts_.sum())
        log_probs = np.empty(len(self.labels_), dtype=np.float64)

        for c in range(len(self.labels_)):
            prior = math.log(max(self.counts_[c], 1) / total)
            var = self.vars_[c]
            mean = self.means_[c]
            ll = -0.5 * (np.log(2.0 * math.pi * var) + ((Xf - mean) ** 2) / var).sum()
            log_probs[c] = prior + ll

        best = int(np.argmax(log_probs))
        return int(self.labels_[best])

    def predict(self, test_files: List[Path]) -> pd.DataFrame:
        rows = []
        for i, path in enumerate(test_files, start=1):
            X, _, file_id = load_and_engineer(path)
            pred = self.predict_one(X)
            rows.append({"Id": file_id, "Label": pred})
            if i % 500 == 0:
                print(f"Predicted {i}/{len(test_files)}")
        out = pd.DataFrame(rows).sort_values("Id").reset_index(drop=True)
        return out

    def save(self, path: Path) -> None:
        if self.labels_ is None or self.counts_ is None or self.means_ is None or self.vars_ is None:
            raise RuntimeError("Model is not fitted")

        meta = {
            "labels": self.labels_.tolist(),
            "counts": self.counts_.tolist(),
            "num_seconds": int(self.num_seconds_),
            "num_features": int(self.num_features_),
        }

        np.savez_compressed(
            path,
            means=self.means_,
            vars=self.vars_,
            meta=json.dumps(meta),
        )

    @classmethod
    def load(cls, path: Path) -> "GaussianNBTimeSeries":
        obj = cls()
        data = np.load(path, allow_pickle=False)
        meta = json.loads(data["meta"].item())
        obj.labels_ = np.array(meta["labels"], dtype=int)
        obj.label_to_index_ = {int(lbl): i for i, lbl in enumerate(obj.labels_)}
        obj.counts_ = np.array(meta["counts"], dtype=np.int64)
        obj.num_seconds_ = int(meta["num_seconds"])
        obj.num_features_ = int(meta["num_features"])
        obj.means_ = data["means"].astype(np.float64, copy=False)
        obj.vars_ = data["vars"].astype(np.float64, copy=False)
        return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, default=Path("dm2026_gnb_model.npz"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if model exists")
    args = parser.parse_args()

    train_root = args.data_root / "train"
    test_root = args.data_root / "test"

    if not train_root.exists():
        raise FileNotFoundError(f"Training folder not found: {train_root}")
    if not test_root.exists():
        raise FileNotFoundError(f"Test folder not found: {test_root}")

    test_files = discover_csvs(test_root)

    if args.model_path.exists() and not args.retrain:
        print(f"Loading model from {args.model_path}")
        model = GaussianNBTimeSeries.load(args.model_path)
    else:
        train_files = discover_csvs(train_root)
        print(f"Training on {len(train_files)} files...")
        model = GaussianNBTimeSeries().fit(train_files)
        model.save(args.model_path)
        print(f"Saved model to {args.model_path}")

    print(f"Predicting {len(test_files)} files...")
    submission = model.predict(test_files)
    submission.to_csv(args.output, index=False)
    print(f"Saved submission to {args.output}")
    print(submission.head())


if __name__ == "__main__":
    main()
