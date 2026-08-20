#!/usr/bin/env python3
"""
Phase 3 — Ablation Study Evaluator

Runs the currently available Phase-3 ablations on the SAME verified test set:

1. Temporal model vs spatial-only baseline
2. Mean vs median vs max video aggregation
3. Reports pooled and source-macro AUROC
4. Reports accuracy, precision, recall, F1 and PR-AUC
5. Saves a publication-ready CSV

Important:
- This script does NOT tune thresholds on the test set.
- Threshold-dependent metrics use the fixed 0.5 threshold.
- The progressive-unfreezing comparison is reported separately only if
  an epoch-05/frozen-stage checkpoint actually exists.
- 8-frame, tracking/no-tracking, and filtering/raw-detection ablations
  require corresponding datasets; they are not fabricated here.
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

from models.spatial_backbone import ConvNeXtSpatialBackbone
from models.temporal_transformer import TemporalTransformer
from models.temporal_model import SpatiotemporalDeepfakeModel


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3 deepfake-detection ablation evaluator"
    )

    parser.add_argument(
        "--checkpoint",
        default="./checkpoints_temporal/best_checkpoint.pth",
    )

    parser.add_argument(
        "--data-path",
        default="/home/aashish/deepfake_detection/data/shards/videos/test",
    )

    parser.add_argument(
        "--output-dir",
        default="./ablation_results",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--frames", type=int, default=16, choices=[8, 16, 32])

    return parser.parse_args()


# ============================================================
# TEST DATASET
# ============================================================

class AblationShardDataset(IterableDataset):
    """
    Worker-safe streaming dataset.

    Supports is_video stored either as:
        bool
    or:
        per-sample list/tensor
    """

    def __init__(self, shard_dir):
        self.shard_dir = shard_dir
        catalog_path = os.path.join(shard_dir, "catalog.csv")

        if not os.path.exists(catalog_path):
            raise FileNotFoundError(
                f"Catalog not found: {catalog_path}"
            )

        self.catalog = pd.read_csv(catalog_path)
        self.shard_files = self.catalog["shard_file"].tolist()

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            worker_shards = self.shard_files
        else:
            n = len(self.shard_files)
            per_worker = int(math.ceil(n / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, n)
            worker_shards = self.shard_files[start:end]

        for shard_file in worker_shards:
            shard_path = os.path.join(self.shard_dir, shard_file)

            try:
                shard = torch.load(
                    shard_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except Exception as exc:
                print(f"WARNING: failed to load {shard_file}: {exc}")
                continue

            required = {
                "tensors",
                "labels",
                "video_ids",
                "track_ids",
                "sources",
            }

            missing = required - set(shard.keys())
            if missing:
                print(
                    f"WARNING: {shard_file} missing keys: {missing}"
                )
                continue

            tensors = shard["tensors"]
            labels = shard["labels"]
            video_ids = shard["video_ids"]
            track_ids = shard["track_ids"]
            sources = shard["sources"]

            n_samples = len(tensors)

            assert len(labels) == n_samples
            assert len(video_ids) == n_samples
            assert len(track_ids) == n_samples
            assert len(sources) == n_samples

            if tensors.dtype != torch.float16:
                tensors = tensors.to(torch.float16)

            for i in range(n_samples):
                label = labels[i]
                if torch.is_tensor(label):
                    label = label.item()

                yield {
                    "tensor": tensors[i],
                    "label": int(label),
                    "video_id": video_ids[i],
                    "track_id": track_ids[i],
                    "source": sources[i],
                }


# ============================================================
# METRICS
# ============================================================

def binary_metrics(y_true, y_prob, threshold=0.5):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(np.int64)

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        auroc = np.nan
        pr_auc = np.nan

    return {
        "Accuracy": float(acc),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "AUROC": float(auroc),
        "PR-AUC": float(pr_auc),
    }


def aggregate_video_predictions(
    clip_df,
    probability_column,
    aggregation,
):
    """
    Aggregates clip probabilities by video_id.

    aggregation:
        mean
        median
        max
    """

    if aggregation == "mean":
        agg_fn = "mean"
    elif aggregation == "median":
        agg_fn = "median"
    elif aggregation == "max":
        agg_fn = "max"
    else:
        raise ValueError(aggregation)

    video_df = (
        clip_df
        .groupby("video_id")
        .agg(
            probability=(probability_column, agg_fn),
            label=("label", "first"),
            source=("source", "first"),
        )
        .reset_index()
    )

    return video_df


# ============================================================
# MODEL
# ============================================================

def build_model(checkpoint_path, frames, device):
    if frames != 16:
        raise ValueError(
            "The current TemporalTransformer checkpoint uses 16 frames. "
            "The 8/32-frame ablations require separately trained or "
            "compatible positional embeddings/checkpoints."
        )

    backbone = ConvNeXtSpatialBackbone(
        pretrained=False,
        num_classes=1,
    )

    transformer = TemporalTransformer(
        num_frames=16,
        embed_dim=768,
        depth=2,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
    )

    model = SpatiotemporalDeepfakeModel(
        backbone,
        transformer,
        embed_dim=768,
        num_classes=1,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        print(f"WARNING: {len(missing)} missing checkpoint keys.")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected checkpoint keys.")

    model.to(device)
    model.eval()

    return model


# ============================================================
# ONE-PASS INFERENCE
# ============================================================

@torch.inference_mode()
def collect_clip_predictions(model, loader, device):
    """
    One forward pass produces BOTH:

        temporal probability
        spatial-only probability

    Spatial-only:
        ConvNeXt features -> mean over 16 frames -> final model head

    Temporal:
        ConvNeXt features -> Temporal Transformer -> final model head
    """

    rows = []

    print("\nRunning one-pass spatial + temporal inference...")

    for batch in tqdm(loader):
        x = batch["tensor"].to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.float16,
        ):
            # Shared spatial feature extraction.
            features = model.backbone.extract_features(x)

            # ------------------------------------------------
            # Temporal model
            # ------------------------------------------------
            temporal_cls = model.temporal_transformer(features)

            temporal_logits = model.head(temporal_cls).squeeze(-1)

            # ------------------------------------------------
            # Spatial-only baseline
            # ------------------------------------------------
            spatial_features = features.mean(dim=1)

            spatial_logits = model.head(
                spatial_features
            ).squeeze(-1)

            temporal_probs = torch.sigmoid(
                temporal_logits
            )

            spatial_probs = torch.sigmoid(
                spatial_logits
            )

        temporal_probs = temporal_probs.float().cpu().numpy()
        spatial_probs = spatial_probs.float().cpu().numpy()

        labels = batch["label"].cpu().numpy()

        for i in range(len(labels)):
            rows.append(
                {
                    "video_id": batch["video_id"][i],
                    "track_id": batch["track_id"][i],
                    "source": batch["source"][i],
                    "label": int(labels[i]),
                    "temporal_probability": float(
                        temporal_probs[i]
                    ),
                    "spatial_probability": float(
                        spatial_probs[i]
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# ABLATION MATRIX
# ============================================================

def evaluate_configuration(
    clip_df,
    model_name,
    probability_column,
    aggregation,
):
    video_df = aggregate_video_predictions(
        clip_df,
        probability_column,
        aggregation,
    )

    metrics = binary_metrics(
        video_df["label"].values,
        video_df["probability"].values,
        threshold=0.5,
    )

    metrics.update(
        {
            "Model": model_name,
            "Aggregation": aggregation,
            "Videos": len(video_df),
            "Clips": len(clip_df),
        }
    )

    return metrics, video_df


def run_phase3(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 80)
    print("PHASE 3 — ABLATION STUDIES")
    print("=" * 80)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Test data  : {args.data_path}")
    print(f"Device     : {device}")
    print(f"Frames     : {args.frames}")
    print("=" * 80)

    model = build_model(
        args.checkpoint,
        args.frames,
        device,
    )

    dataset = AblationShardDataset(args.data_path)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    start = time.time()

    clip_df = collect_clip_predictions(
        model,
        loader,
        device,
    )

    inference_time = time.time() - start

    clip_df.to_csv(
        output_dir / "phase3_clip_predictions.csv",
        index=False,
    )

    print("\nInference complete.")
    print(f"Clips : {len(clip_df)}")
    print(
        f"Videos: {clip_df['video_id'].nunique()}"
    )
    print(
        f"Time  : {inference_time:.2f} sec"
    )

    # --------------------------------------------------------
    # Run all currently possible Phase-3 configurations.
    # --------------------------------------------------------
    results = []

    for model_name, prob_col in [
        ("Spatial-only", "spatial_probability"),
        ("Spatial + Temporal Transformer", "temporal_probability"),
    ]:
        for aggregation in [
            "mean",
            "median",
            "max",
        ]:
            metrics, video_df = evaluate_configuration(
                clip_df,
                model_name,
                prob_col,
                aggregation,
            )

            results.append(metrics)

            filename = (
                f"{model_name.lower().replace(' ', '_').replace('+', 'plus')}"
                f"_{aggregation}_video_predictions.csv"
            )

            video_df.to_csv(
                output_dir / filename,
                index=False,
            )

    results_df = pd.DataFrame(results)

    columns = [
        "Model",
        "Aggregation",
        "Videos",
        "Clips",
        "AUROC",
        "PR-AUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]

    results_df = results_df[columns]

    results_df.to_csv(
        output_dir / "phase3_ablation_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Source-macro AUROC
    # --------------------------------------------------------
    macro_rows = []

    for model_name, prob_col in [
        ("Spatial-only", "spatial_probability"),
        ("Spatial + Temporal Transformer", "temporal_probability"),
    ]:
        for aggregation in [
            "mean",
            "median",
            "max",
        ]:
            video_df = aggregate_video_predictions(
                clip_df,
                prob_col,
                aggregation,
            )

            source_aurocs = []

            for source, group in video_df.groupby(
                "source"
            ):
                labels = group["label"].values

                if len(np.unique(labels)) < 2:
                    continue

                from sklearn.metrics import roc_auc_score

                source_aurocs.append(
                    roc_auc_score(
                        labels,
                        group["probability"].values,
                    )
                )

            macro_auroc = (
                float(np.mean(source_aurocs))
                if source_aurocs
                else np.nan
            )

            macro_rows.append(
                {
                    "Model": model_name,
                    "Aggregation": aggregation,
                    "Macro_AUROC": macro_auroc,
                    "Valid_Sources": len(source_aurocs),
                }
            )

    macro_df = pd.DataFrame(macro_rows)

    macro_df.to_csv(
        output_dir / "phase3_macro_source_auroc.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final console table
    # --------------------------------------------------------
    print("\n" + "=" * 110)
    print("PHASE 3 ABLATION MATRIX")
    print("=" * 110)

    print(
        f"{'Model':<30} | "
        f"{'Pool':<8} | "
        f"{'AUROC':<8} | "
        f"{'PR-AUC':<8} | "
        f"{'Acc':<8} | "
        f"{'F1':<8}"
    )

    print("-" * 110)

    for _, row in results_df.iterrows():
        print(
            f"{row['Model']:<30} | "
            f"{row['Aggregation']:<8} | "
            f"{row['AUROC']:<8.4f} | "
            f"{row['PR-AUC']:<8.4f} | "
            f"{row['Accuracy']:<8.4f} | "
            f"{row['F1']:<8.4f}"
        )

    print("\n" + "=" * 80)
    print("MACRO SOURCE AUROC")
    print("=" * 80)

    for _, row in macro_df.iterrows():
        print(
            f"{row['Model']:<30} | "
            f"{row['Aggregation']:<8} | "
            f"Macro AUROC: {row['Macro_AUROC']:.4f} | "
            f"Sources: {int(row['Valid_Sources'])}"
        )

    # --------------------------------------------------------
    # Delta: temporal vs spatial using mean aggregation.
    # --------------------------------------------------------
    spatial_mean = results_df[
        (results_df["Model"] == "Spatial-only")
        & (results_df["Aggregation"] == "mean")
    ].iloc[0]

    temporal_mean = results_df[
        (
            results_df["Model"]
            == "Spatial + Temporal Transformer"
        )
        & (results_df["Aggregation"] == "mean")
    ].iloc[0]

    spatial_macro = macro_df[
        (macro_df["Model"] == "Spatial-only")
        & (macro_df["Aggregation"] == "mean")
    ].iloc[0]["Macro_AUROC"]

    temporal_macro = macro_df[
        (
            macro_df["Model"]
            == "Spatial + Temporal Transformer"
        )
        & (macro_df["Aggregation"] == "mean")
    ].iloc[0]["Macro_AUROC"]

    print("\n" + "=" * 80)
    print("TEMPORAL CONTRIBUTION")
    print("=" * 80)

    print(
        f"Mean pooled AUROC — Spatial-only : "
        f"{spatial_mean['AUROC']:.4f}"
    )
    print(
        f"Mean pooled AUROC — Temporal      : "
        f"{temporal_mean['AUROC']:.4f}"
    )
    print(
        f"Temporal AUROC Δ                  : "
        f"{temporal_mean['AUROC'] - spatial_mean['AUROC']:+.4f}"
    )

    print(
        f"Macro AUROC — Spatial-only        : "
        f"{spatial_macro:.4f}"
    )
    print(
        f"Macro AUROC — Temporal             : "
        f"{temporal_macro:.4f}"
    )
    print(
        f"Temporal Macro-AUROC Δ             : "
        f"{temporal_macro - spatial_macro:+.4f}"
    )

    print("\n" + "=" * 80)
    print("PHASE 3 STATUS")
    print("=" * 80)
    print(
        "✓ Spatial vs Temporal: READY"
    )
    print(
        "✓ Mean vs Median vs Max: READY"
    )
    print(
        "✓ Same test set and same checkpoint used for these comparisons"
    )
    print(
        "✓ No test-set threshold tuning"
    )
    print(
        "⚠ Progressive-unfreezing ablation requires an actual epoch-5 "
        "checkpoint."
    )
    print(
        "⚠ 8-frame ablation requires an 8-frame dataset/model checkpoint."
    )
    print(
        "⚠ Tracking/filtering ablations require corresponding stored "
        "test-set variants."
    )

    print("\nResults saved to:")
    print(
        f"  {output_dir / 'phase3_ablation_results.csv'}"
    )
    print(
        f"  {output_dir / 'phase3_macro_source_auroc.csv'}"
    )
    print(
        f"  {output_dir / 'phase3_clip_predictions.csv'}"
    )
    print("=" * 80)


if __name__ == "__main__":
    run_phase3(parse_args())
