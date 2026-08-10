import os
import time
import math
import json
import datetime
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, 
    average_precision_score, log_loss, roc_curve, precision_recall_curve, 
    confusion_matrix
)

# IMPORT ARCHITECTURE
from models.spatial_backbone import ConvNeXtSpatialBackbone
from models.temporal_transformer import TemporalTransformer
from models.temporal_model import SpatiotemporalDeepfakeModel

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "checkpoint_path": "./checkpoints_temporal/best_checkpoint.pth",
    "test_shard_dir": "/home/aashish/deepfake_detection/data/shards/videos/test",
    "output_dir": "./evaluation_results",
    "batch_size": 64,
    "num_workers": 4,
    "aggregation_method": "mean",  # Options: 'mean', 'max', 'median'
    "threshold": 0.5,
    "expected_keys": {"tensors", "labels", "video_ids", "track_ids", "sources", "is_video"},
    "expected_dtype": torch.float16,
    "expected_frames": 16,
    "expected_spatial_shape": (3, 224, 224), # C, H, W
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Hardware Optimization
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ============================================================
# 1. EVALUATION DATASET WITH ERROR LOGGING
# ============================================================
def log_bad_shard(msg, output_dir):
    log_path = os.path.join(output_dir, "bad_shards.log")
    with open(log_path, "a") as f:
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

class EvaluationShardDataset(IterableDataset):
    def __init__(self, shard_dir, config):
        self.shard_dir = shard_dir
        self.config = config
        self.catalog_path = os.path.join(shard_dir, "catalog.csv")
        
        if not os.path.exists(self.catalog_path):
            raise FileNotFoundError(f"Catalog not found at {self.catalog_path}")
            
        self.catalog = pd.read_csv(self.catalog_path)
        self.shard_files = self.catalog['shard_file'].tolist()

    def __iter__(self):
        worker_info = get_worker_info()
        
        if worker_info is None:
            worker_shards = self.shard_files
        else:
            per_worker = int(math.ceil(len(self.shard_files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.shard_files))
            worker_shards = self.shard_files[start:end]

        for shard_file in worker_shards:
            shard_path = os.path.join(self.shard_dir, shard_file)
            
            try:
                # weights_only=False required for metadata (video_ids, etc.)
                shard_data = torch.load(shard_path, map_location='cpu', weights_only=False, mmap=True)
            except Exception as e:
                log_bad_shard(f"Error loading {shard_file}: {e}", self.config["output_dir"])
                continue
            
            if not self.config["expected_keys"].issubset(shard_data.keys()):
                log_bad_shard(f"Shard {shard_file} missing required keys.", self.config["output_dir"])
                continue
                
            tensors = shard_data["tensors"]
            
            if tensors.dtype != self.config["expected_dtype"]:
                tensors = tensors.to(self.config["expected_dtype"])
                
            # Strict Shape Verification
            if tensors.ndim != 5:
                log_bad_shard(f"Skipping {shard_file}: Expected 5 dimensions, got {tensors.ndim}", self.config["output_dir"])
                continue
            if tensors.shape[1] != self.config["expected_frames"]:
                log_bad_shard(f"Skipping {shard_file}: Expected {self.config['expected_frames']} frames, got {tensors.shape[1]}", self.config["output_dir"])
                continue
            if tensors.shape[2:] != self.config["expected_spatial_shape"]:
                log_bad_shard(f"Skipping {shard_file}: Invalid spatial shape {tensors.shape[2:]}", self.config["output_dir"])
                continue
                
            if torch.isnan(tensors).any() or torch.isinf(tensors).any():
                log_bad_shard(f"Skipping {shard_file}: Contains NaNs or Infs.", self.config["output_dir"])
                continue

            labels = shard_data["labels"]
            video_ids = shard_data["video_ids"]
            track_ids = shard_data["track_ids"]
            sources = shard_data["sources"]
            is_video = shard_data["is_video"]
            
            num_samples = len(tensors)
            assert len(labels) == num_samples
            assert len(video_ids) == num_samples
            assert len(track_ids) == num_samples
            assert len(sources) == num_samples
            
            for i in range(num_samples):
                label = labels[i]
                if torch.is_tensor(label):
                    label = label.item()
                    
                yield {
                    "tensor": tensors[i],
                    "label": label,
                    "video_id": video_ids[i],
                    "track_id": track_ids[i],
                    "source": sources[i],
                    "is_video": is_video
                }

# ============================================================
# 2. METRIC CALCULATIONS
# ============================================================
def compute_eer(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]

def calculate_full_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float('nan')
        
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float('nan')
        
    try:
        loss = log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15))
    except ValueError:
        loss = float('nan')
        
    try:
        eer, _ = compute_eer(y_true, y_prob)
    except Exception:
        eer = float('nan')

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUROC": auroc,
        "PR-AUC": pr_auc,
        "LogLoss": loss,
        "EER": eer,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }

# ============================================================
# 3. PUBLICATION ARTIFACTS
# ============================================================
def save_plots_and_data(y_true, y_prob, threshold, output_dir, name_prefix="video"):
    output_dir = Path(output_dir)
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(output_dir / f"{name_prefix}_roc_points.csv", index=False)
    
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic ({name_prefix.capitalize()}-Level)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"{name_prefix}_roc_curve.png", bbox_inches='tight')
    plt.close()

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    pd.DataFrame({'recall': recall, 'precision': precision}).to_csv(output_dir / f"{name_prefix}_pr_points.csv", index=False)
    
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({name_prefix.capitalize()}-Level)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"{name_prefix}_pr_curve.png", bbox_inches='tight')
    plt.close()

    # 3. Confusion Matrix
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=['Actual_Real', 'Actual_Fake'], columns=['Pred_Real', 'Pred_Fake'])
    cm_df.to_csv(output_dir / f"{name_prefix}_confusion_matrix.csv")
    
    plt.figure(figsize=(5, 4), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({name_prefix.capitalize()}-Level)')
    plt.savefig(output_dir / f"{name_prefix}_confusion_matrix.png", bbox_inches='tight')
    plt.close()

# ============================================================
# 4. MAIN EVALUATION ENGINE
# ============================================================
def run_evaluation(config):
    device = torch.device(config["device"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Metadata
    metadata = {
        "checkpoint_name": os.path.basename(config["checkpoint_path"]),
        "aggregation_method": config["aggregation_method"],
        "batch_size": config["batch_size"],
        "threshold": config["threshold"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__
    }
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Loading checkpoint: {config['checkpoint_path']}")
    checkpoint = torch.load(config["checkpoint_path"], map_location=device, weights_only=True)
    
    # Initialize architecture sub-modules
    backbone = ConvNeXtSpatialBackbone(
        pretrained=False,
        num_classes=1
    )
    
    transformer = TemporalTransformer(
        num_frames=config["expected_frames"],
        embed_dim=768,
        depth=2,
        num_heads=8
    )
    
    # Inject into main model
    model = SpatiotemporalDeepfakeModel(
        backbone,
        transformer
    )
    
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device)
    model.eval()

    dataset = EvaluationShardDataset(config["test_shard_dir"], config)
    dataloader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        num_workers=config["num_workers"], 
        pin_memory=True,
        persistent_workers=True if config["num_workers"] > 0 else False,
        shuffle=False
    )
    
    all_probs, all_labels = [], []
    all_video_ids, all_sources, all_track_ids = [], [], []
    
    start_time = time.time()
    
    print("Executing AMP/TF32 inference on test split...")
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            tensors = batch["tensor"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy()
            
            with torch.amp.autocast(device_type="cuda" if "cuda" in config["device"] else "cpu", dtype=torch.float16):
                logits = model(tensors)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_video_ids.extend(batch["video_id"])
            all_sources.extend(batch["source"])
            all_track_ids.extend(batch["track_id"])

    elapsed_time = time.time() - start_time
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Save Raw Clip Predictions CSV
    clip_df = pd.DataFrame({
        "video_id": all_video_ids,
        "track_id": all_track_ids,
        "source": all_sources,
        "label": all_labels,
        "clip_probability": all_probs
    })
    clip_df.to_csv(output_dir / "clip_predictions.csv", index=False)
    
    # Video Aggregation ('mean', 'max', or 'median')
    agg_func = config["aggregation_method"]
    video_df = clip_df.groupby("video_id").agg({
        "clip_probability": agg_func,
        "label": "first",
        "source": "first"
    }).reset_index().rename(columns={"clip_probability": "video_probability"})
    
    video_df.to_csv(output_dir / "video_predictions.csv", index=False)

    # Compute Global Metrics
    clip_metrics = calculate_full_metrics(all_labels, all_probs, config["threshold"])
    video_metrics = calculate_full_metrics(video_df["label"].values, video_df["video_probability"].values, config["threshold"])
    
    # Generate Plots and Raw Points
    save_plots_and_data(all_labels, all_probs, config["threshold"], output_dir, name_prefix="clip")
    save_plots_and_data(video_df["label"].values, video_df["video_probability"].values, config["threshold"], output_dir, name_prefix="video")

    # ============================================================
    # PRINT PUBLICATION REPORT
    # ============================================================
    print("\n" + "="*70)
    print("GLOBAL PERFORMANCE REPORT")
    print("="*70)
    video_title = f"Video-Level ({agg_func})"
    print(f"{'Metric':<15} | {'Clip-Level':<15} | {video_title:<20}")
    print("-" * 70)
    for m in ["AUROC", "PR-AUC", "EER", "Accuracy", "Precision", "Recall", "F1", "LogLoss"]:
        print(f"{m:<15} | {clip_metrics[m]:<15.4f} | {video_metrics[m]:<15.4f}")
    
    print("-" * 70)
    print("Video Confusion Matrix:")
    print(f"  TN: {video_metrics['TN']:<6} FP: {video_metrics['FP']:<6}")
    print(f"  FN: {video_metrics['FN']:<6} TP: {video_metrics['TP']:<6}")
    
    # Per-Source Metrics
    print("\n" + "="*110)
    print("PER-SOURCE VIDEO-LEVEL BREAKDOWN")
    print("="*110)
    print(f"{'Source':<18} | {'Vids':<5} | {'AUROC':<7} | {'PR-AUC':<7} | {'EER':<7} | {'Acc':<7} | {'Pre':<7} | {'Rec':<7} | {'F1':<7} | {'LogLoss':<7}")
    print("-" * 110)
    
    source_summary = []
    for source, group in video_df.groupby("source"):
        if len(group["label"].unique()) < 2:
            print(f"{source:<18} | {len(group):<5} | Single-class split (Skipped)")
            continue
        sm = calculate_full_metrics(group["label"].values, group["video_probability"].values, config["threshold"])
        print(f"{source:<18} | {len(group):<5} | {sm['AUROC']:<7.4f} | {sm['PR-AUC']:<7.4f} | {sm['EER']:<7.4f} | {sm['Accuracy']:<7.4f} | {sm['Precision']:<7.4f} | {sm['Recall']:<7.4f} | {sm['F1']:<7.4f} | {sm['LogLoss']:<7.4f}")
        sm["Source"] = source
        sm["Videos"] = len(group)
        source_summary.append(sm)
        
    pd.DataFrame(source_summary).to_csv(output_dir / "per_source_metrics.csv", index=False)

    # Throughput Report
    total_clips = len(all_probs)
    total_videos = len(video_df)
    clips_per_sec = total_clips / elapsed_time
    vids_per_sec = total_videos / elapsed_time
    
    print("\n" + "="*70)
    print("INFERENCE THROUGHPUT & BENCHMARK SUMMARY")
    print("="*70)
    print(f"Device               : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Total Inference Time : {elapsed_time:.2f} seconds")
    print(f"Total Clips Processed: {total_clips}")
    print(f"Total Videos Processed: {total_videos}")
    print(f"Throughput (Clips)   : {clips_per_sec:.2f} clips/sec")
    print(f"Throughput (Videos)  : {vids_per_sec:.2f} videos/sec")
    print(f"Artifacts Saved To   : {output_dir.resolve()}")
    print("="*70)

if __name__ == "__main__":
    run_evaluation(CONFIG)