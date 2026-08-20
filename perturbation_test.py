#!/usr/bin/env python3
"""
Phase 4.2 — Robustness & Perturbation Evaluator

Evaluates AUROC degradation under:
1. Clean baseline
2. Gaussian Blur (3x3, 5x5, 7x7)
3. Resolution Downscaling (160x160, 112x112, 80x80)
4. Lossy Compression Artifacts (JPEG Q=80, Q=50, Q=30)

Saves detailed degradation matrix to ./ablation_results/phase4_perturbations.csv
"""

import argparse
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision.transforms import functional as TVF
from tqdm import tqdm

from models.spatial_backbone import ConvNeXtSpatialBackbone
from models.temporal_transformer import TemporalTransformer
from models.temporal_model import SpatiotemporalDeepfakeModel


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4.2 Perturbation Evaluator")
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


# ============================================================
# DATASET
# ============================================================

class PerturbationShardDataset(IterableDataset):
    def __init__(self, shard_dir):
        self.shard_dir = shard_dir
        catalog_path = os.path.join(shard_dir, "catalog.csv")
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog not found: {catalog_path}")

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
                shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                print(f"WARNING: failed to load {shard_file}: {exc}")
                continue

            tensors = shard["tensors"]
            labels = shard["labels"]
            video_ids = shard["video_ids"]
            track_ids = shard["track_ids"]
            sources = shard["sources"]

            if tensors.dtype != torch.float32:
                tensors = tensors.float()

            for i in range(len(tensors)):
                label = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
                yield {
                    "tensor": tensors[i],
                    "label": int(label),
                    "video_id": video_ids[i],
                    "track_id": track_ids[i],
                    "source": sources[i],
                }


# ============================================================
# PERTURBATION TRANSFORMATIONS
# ============================================================

def apply_perturbation(tensor, pert_type, param):
    """
    tensor shape: [B, 16, 3, 224, 224]
    """
    if pert_type == "clean":
        return tensor

    B, T, C, H, W = tensor.shape
    x = tensor.view(B * T, C, H, W)

    if pert_type == "blur":
        # param: kernel size (3, 5, 7)
        k = int(param)
        x = TVF.gaussian_blur(x, kernel_size=[k, k])

    elif pert_type == "downscale":
        # param: target resolution (160, 112, 80)
        res = int(param)
        x_down = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
        x = F.interpolate(x_down, size=(224, 224), mode="bilinear", align_corners=False)

    elif pert_type == "jpeg":
        # param: quality factor (80, 50, 30)
        q = int(param)
        x_np = (x.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
        x_np = np.transpose(x_np, (0, 2, 3, 1))  # [N, H, W, C]

        corrupted = []
        for img in x_np:
            # OpenCV BGR conversion for JPEG compression
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            corrupted.append(rgb)

        corrupted_np = np.array(corrupted, dtype=np.float32) / 255.0
        x = torch.from_numpy(corrupted_np).permute(0, 3, 1, 2).to(tensor.device)

    return x.view(B, T, C, H, W)


# ============================================================
# METRICS & AGGREGATION
# ============================================================

def evaluate_predictions(clip_df):
    from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score

    video_df = (
        clip_df.groupby("video_id")
        .agg(
            probability=("probability", "mean"),
            label=("label", "first"),
            source=("source", "first"),
        )
        .reset_index()
    )

    y_true = video_df["label"].values
    y_prob = video_df["probability"].values
    y_pred = (y_prob >= 0.5).astype(np.int64)

    acc = accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    # Source-Macro AUROC
    source_aurocs = []
    for source, group in video_df.groupby("source"):
        lbls = group["label"].values
        if len(np.unique(lbls)) >= 2:
            source_aurocs.append(roc_auc_score(lbls, group["probability"].values))

    macro_auroc = float(np.mean(source_aurocs)) if source_aurocs else np.nan

    return {
        "Pooled_AUROC": float(auroc),
        "PR_AUC": float(pr_auc),
        "Accuracy": float(acc),
        "F1": float(f1),
        "Macro_AUROC": macro_auroc,
    }


# ============================================================
# INFERENCE PASS
# ============================================================

@torch.inference_mode()
def run_perturbation_eval(model, loader, device, pert_type, param):
    rows = []
    for batch in tqdm(loader, desc=f"{pert_type.upper()} ({param})"):
        x = batch["tensor"].to(device, non_blocking=True)
        x_pert = apply_perturbation(x, pert_type, param)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
            features = model.backbone.extract_features(x_pert)
            temporal_cls = model.temporal_transformer(features)
            logits = model.head(temporal_cls).squeeze(-1)
            probs = torch.sigmoid(logits).float().cpu().numpy()

        labels = batch["label"].cpu().numpy()
        for i in range(len(labels)):
            rows.append({
                "video_id": batch["video_id"][i],
                "source": batch["source"][i],
                "label": int(labels[i]),
                "probability": float(probs[i]),
            })

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("PHASE 4.2 — ROBUSTNESS & PERTURBATION EVALUATOR")
    print("=" * 80)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Test data  : {args.data_path}")
    print(f"Device     : {device}")
    print("=" * 80)

    # Load Model
    backbone = ConvNeXtSpatialBackbone(pretrained=False, num_classes=1)
    transformer = TemporalTransformer(
        num_frames=16, embed_dim=768, depth=2, num_heads=8, mlp_ratio=4, dropout=0.1
    )
    model = SpatiotemporalDeepfakeModel(backbone, transformer, embed_dim=768, num_classes=1)
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    dataset = PerturbationShardDataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    perturbation_suite = [
        ("clean", "None"),
        ("blur", 3),
        ("blur", 5),
        ("blur", 7),
        ("downscale", 160),
        ("downscale", 112),
        ("downscale", 80),
        ("jpeg", 80),
        ("jpeg", 50),
        ("jpeg", 30),
    ]

    results = []

    for pert_type, param in perturbation_suite:
        clip_df = run_perturbation_eval(model, loader, device, pert_type, param)
        metrics = evaluate_predictions(clip_df)
        metrics["Perturbation"] = pert_type.capitalize()
        metrics["Severity_Param"] = str(param)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    
    # Format table
    cols = ["Perturbation", "Severity_Param", "Macro_AUROC", "Pooled_AUROC", "PR_AUC", "Accuracy", "F1"]
    results_df = results_df[cols]

    csv_path = output_dir / "phase4_perturbations.csv"
    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print("PHASE 4.2 ROBUSTNESS MATRIX")
    print("=" * 90)
    print(f"{'Perturbation':<15} | {'Severity':<10} | {'Macro-AUROC':<12} | {'Pooled-AUROC':<12} | {'Accuracy':<10} | {'F1':<10}")
    print("-" * 90)
    for _, r in results_df.iterrows():
        print(
            f"{r['Perturbation']:<15} | "
            f"{r['Severity_Param']:<10} | "
            f"{r['Macro_AUROC']:<12.4f} | "
            f"{r['Pooled_AUROC']:<12.4f} | "
            f"{r['Accuracy']:<10.4f} | "
            f"{r['F1']:<10.4f}"
        )
    print("=" * 90)
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()