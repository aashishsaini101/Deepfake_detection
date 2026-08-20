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

    "val_shard_dir":
        "/home/aashish/deepfake_detection/data/shards/videos/val",

    "test_shard_dir":
        "/home/aashish/deepfake_detection/data/shards/videos/test",

    "output_dir": "./evaluation_results",

    "batch_size": 64,
    "num_workers": 4,

    "aggregation_method": "mean",

    "threshold": 0.5,

    "expected_keys": {
        "tensors",
        "labels",
        "video_ids",
        "track_ids",
        "sources",
        "is_video"
    },

    "expected_dtype": torch.float16,
    "expected_frames": 16,
    "expected_spatial_shape": (3, 224, 224),

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

            # is_video may be stored as one boolean for the entire shard
            # rather than a list containing one value per sample.
            if torch.is_tensor(is_video):
                if is_video.ndim == 0:
                    is_video_values = [bool(is_video.item())] * num_samples
                else:
                    is_video_values = is_video.tolist()
            elif isinstance(is_video, (bool, np.bool_)):
                is_video_values = [bool(is_video)] * num_samples
            elif isinstance(is_video, (int, np.integer)):
                is_video_values = [bool(is_video)] * num_samples
            else:
                is_video_values = list(is_video)
                assert len(is_video_values) == num_samples
            
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
                    "is_video": is_video_values[i]
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

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")

    try:
        loss = log_loss(
            y_true,
            np.clip(y_prob, 1e-7, 1 - 1e-7)
        )
    except ValueError:
        loss = float("nan")

    try:
        eer, _ = compute_eer(y_true, y_prob)
    except Exception:
        eer = float("nan")

    # Always force a 2x2 confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUROC": auroc,
        "PR-AUC": pr_auc,
        "LogLoss": loss,
        "EER": eer,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }
    
# ============================================================
# 2B. VALIDATION THRESHOLD CALIBRATION
# ============================================================

def find_optimal_thresholds(y_true_video, y_probs_video):
    """
    Calculates Max-F1 and EER thresholds strictly from
    aggregated VIDEO-LEVEL validation predictions.
    """

    # --------------------------------------------------------
    # Max-F1 threshold
    # --------------------------------------------------------
    precisions, recalls, thresholds_pr = precision_recall_curve(
        y_true_video,
        y_probs_video
    )

    f1_scores = (
        2 * precisions * recalls
    ) / (
        precisions + recalls + 1e-8
    )

    optimal_idx = np.argmax(f1_scores)

    if optimal_idx < len(thresholds_pr):
        max_f1_thresh = thresholds_pr[optimal_idx]
    else:
        max_f1_thresh = 0.5

    # --------------------------------------------------------
    # EER threshold
    # --------------------------------------------------------
    fpr, tpr, thresholds_roc = roc_curve(
        y_true_video,
        y_probs_video
    )

    fnr = 1.0 - tpr

    eer_idx = np.nanargmin(
        np.absolute(fnr - fpr)
    )

    eer_thresh = thresholds_roc[eer_idx]

    return max_f1_thresh, eer_thresh


def evaluate_discrete_metrics(y_true, y_probs, threshold):
    """
    Calculates threshold-dependent metrics using a
    threshold determined from validation data.
    """

    y_pred = (y_probs >= threshold).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp

    acc = (tp + tn) / max(total, 1)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = (
        2 * precision * recall
    ) / (
        precision + recall + 1e-8
    )

    return {
        "Acc": acc,
        "Pre": precision,
        "Rec": recall,
        "F1": f1,
        "CM": (tn, fp, fn, tp)
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

def run_inference(config, model, shard_dir):
    """
    Runs inference and aggregates clip predictions into
    video-level predictions.
    """

    dataset = EvaluationShardDataset(
        shard_dir,
        config
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=(
            config["num_workers"] > 0
        ),
        shuffle=False
    )

    all_probs = []
    all_labels = []
    all_video_ids = []
    all_sources = []
    all_track_ids = []

    device = torch.device(config["device"])

    with torch.inference_mode():

        for batch in tqdm(dataloader):

            tensors = batch["tensor"].to(
                device,
                non_blocking=True
            )

            labels = batch["label"].cpu().numpy()

            with torch.amp.autocast(
                device_type="cuda"
                if "cuda" in config["device"]
                else "cpu",
                dtype=torch.float16
            ):

                logits = model(tensors)

                probs = (
                    torch.sigmoid(logits)
                    .cpu()
                    .numpy()
                    .flatten()
                )

            all_probs.extend(probs)
            all_labels.extend(labels)
            all_video_ids.extend(batch["video_id"])
            all_sources.extend(batch["source"])
            all_track_ids.extend(batch["track_id"])

    clip_df = pd.DataFrame({
        "video_id": all_video_ids,
        "track_id": all_track_ids,
        "source": all_sources,
        "label": all_labels,
        "clip_probability": all_probs
    })

    # --------------------------------------------------------
    # Aggregate clips → videos
    # --------------------------------------------------------

    agg_func = config["aggregation_method"]

    video_df = (
        clip_df
        .groupby("video_id")
        .agg({
            "clip_probability": agg_func,
            "label": "first",
            "source": "first"
        })
        .reset_index()
        .rename(
            columns={
                "clip_probability":
                "video_probability"
            }
        )
    )

    return (
        video_df["video_probability"].values,
        video_df["label"].values,
        video_df["source"].values,
        clip_df,
        video_df
    )


def run_evaluation(config):
    """
    Final publication evaluation protocol:

        Validation clips
            -> video aggregation
            -> Max-F1/EER threshold calibration

        Test clips
            -> video aggregation
            -> frozen validation thresholds
            -> pooled + per-source + macro metrics
    """

    device = torch.device(config["device"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if config["expected_frames"] != 16:
        raise ValueError(
            "This checkpoint/model is configured for 16-frame clips. "
            "Set expected_frames=16 for the current evaluation."
        )

    if config["aggregation_method"] not in {"mean", "median", "max"}:
        raise ValueError(
            "aggregation_method must be 'mean', 'median', or 'max'."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    print(f"Loading checkpoint: {config['checkpoint_path']}")

    checkpoint = torch.load(
        config["checkpoint_path"],
        map_location=device,
        weights_only=True,
    )

    backbone = ConvNeXtSpatialBackbone(
        pretrained=False,
        num_classes=1,
    )

    transformer = TemporalTransformer(
        num_frames=config["expected_frames"],
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

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        print(f"Warning: {len(missing)} missing checkpoint keys.")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected checkpoint keys.")

    model.to(device)
    model.eval()

    # ========================================================
    # PASS 1: VALIDATION CALIBRATION
    # ========================================================
    print("\n" + "=" * 70)
    print("PHASE 1: VIDEO-LEVEL VALIDATION CALIBRATION")
    print("=" * 70)

    val_start = time.time()

    (
        val_video_probs,
        val_video_labels,
        val_video_sources,
        val_clip_df,
        val_video_df,
    ) = run_inference(
        config,
        model,
        config["val_shard_dir"],
    )

    val_elapsed = time.time() - val_start

    max_f1_thresh, eer_thresh = find_optimal_thresholds(
        val_video_labels,
        val_video_probs,
    )

    print(f"Validation Videos       : {len(val_video_labels)}")
    print(f"Validation Time         : {val_elapsed:.2f} seconds")
    print(f"Locked Max-F1 Threshold : {max_f1_thresh:.6f}")
    print(f"Locked EER Threshold    : {eer_thresh:.6f}")

    val_clip_df.to_csv(
        output_dir / "validation_clip_predictions.csv",
        index=False,
    )
    val_video_df.to_csv(
        output_dir / "validation_video_predictions.csv",
        index=False,
    )

    # ========================================================
    # PASS 2: UNSEEN TEST EVALUATION
    # ========================================================
    print("\n" + "=" * 70)
    print("PHASE 2: UNSEEN TEST DATA EVALUATION")
    print("=" * 70)

    test_start = time.time()

    (
        test_video_probs,
        test_video_labels,
        test_sources,
        test_clip_df,
        test_video_df,
    ) = run_inference(
        config,
        model,
        config["test_shard_dir"],
    )

    elapsed_time = time.time() - test_start

    test_clip_df.to_csv(
        output_dir / "test_clip_predictions.csv",
        index=False,
    )
    test_video_df.to_csv(
        output_dir / "test_video_predictions.csv",
        index=False,
    )

    # ========================================================
    # TEST THRESHOLD-INDEPENDENT METRICS
    # ========================================================
    test_auroc = roc_auc_score(
        test_video_labels,
        test_video_probs,
    )

    test_prauc = average_precision_score(
        test_video_labels,
        test_video_probs,
    )

    safe_probs = np.clip(
        test_video_probs,
        1e-7,
        1 - 1e-7,
    )

    test_logloss = log_loss(
        test_video_labels,
        safe_probs,
    )

    test_eer, test_eer_threshold = compute_eer(
        test_video_labels,
        test_video_probs,
    )

    # ========================================================
    # THRESHOLD-DEPENDENT TEST METRICS
    # ========================================================
    metrics_05 = evaluate_discrete_metrics(
        test_video_labels,
        test_video_probs,
        0.5,
    )

    metrics_maxf1 = evaluate_discrete_metrics(
        test_video_labels,
        test_video_probs,
        max_f1_thresh,
    )

    metrics_eer = evaluate_discrete_metrics(
        test_video_labels,
        test_video_probs,
        eer_thresh,
    )

    # ========================================================
    # PER-SOURCE + MACRO AUROC
    # ========================================================
    print("\n" + "=" * 110)
    print("PER-SOURCE VIDEO-LEVEL BREAKDOWN")
    print("=" * 110)

    print(
        f"{'Source':<20} | {'Vids':<6} | {'AUROC':<8} | "
        f"{'PR-AUC':<8} | {'EER':<8} | {'Acc':<8} | {'F1':<8}"
    )
    print("-" * 110)

    source_summary = []
    source_aurocs = []

    for source, group in test_video_df.groupby("source"):
        labels = group["label"].values
        probs = group["video_probability"].values

        if len(np.unique(labels)) < 2:
            print(
                f"{source:<20} | {len(group):<6} | "
                f"Single-class split (Skipped)"
            )
            continue

        sm = calculate_full_metrics(
            labels,
            probs,
            max_f1_thresh,
        )

        source_aurocs.append(sm["AUROC"])

        print(
            f"{source:<20} | {len(group):<6} | "
            f"{sm['AUROC']:<8.4f} | {sm['PR-AUC']:<8.4f} | "
            f"{sm['EER']:<8.4f} | {sm['Accuracy']:<8.4f} | "
            f"{sm['F1']:<8.4f}"
        )

        sm["Source"] = source
        sm["Videos"] = int(len(group))
        source_summary.append(sm)

    macro_auroc = (
        float(np.mean(source_aurocs))
        if source_aurocs
        else float("nan")
    )

    print("-" * 110)
    print(f"Macro-Averaged Source AUROC: {macro_auroc:.4f}")

    pd.DataFrame(source_summary).to_csv(
        output_dir / "per_source_metrics.csv",
        index=False,
    )

    # ========================================================
    # PUBLICATION ARTIFACTS
    # ========================================================
    save_plots_and_data(
        test_clip_df["label"].values,
        test_clip_df["clip_probability"].values,
        0.5,
        output_dir,
        name_prefix="test_clip",
    )

    save_plots_and_data(
        test_video_labels,
        test_video_probs,
        0.5,
        output_dir,
        name_prefix="test_video_default",
    )

    save_plots_and_data(
        test_video_labels,
        test_video_probs,
        max_f1_thresh,
        output_dir,
        name_prefix="test_video_val_maxf1",
    )

    save_plots_and_data(
        test_video_labels,
        test_video_probs,
        eer_thresh,
        output_dir,
        name_prefix="test_video_val_eer",
    )

    # ========================================================
    # THRESHOLD COMPARISON CSV
    # ========================================================
    threshold_rows = []

    for name, threshold, metrics in [
        ("Default_0.5", 0.5, metrics_05),
        ("Validation_MaxF1", max_f1_thresh, metrics_maxf1),
        ("Validation_EER", eer_thresh, metrics_eer),
    ]:
        tn, fp, fn, tp = metrics["CM"]

        threshold_rows.append({
            "Threshold_Type": name,
            "Threshold": threshold,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Accuracy": metrics["Acc"],
            "Precision": metrics["Pre"],
            "Recall": metrics["Rec"],
            "F1": metrics["F1"],
        })

    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "threshold_comparison.csv",
        index=False,
    )

    # ========================================================
    # SAVE COMPLETE SUMMARY JSON
    # ========================================================
    summary = {
        "checkpoint_path": config["checkpoint_path"],
        "aggregation_method": config["aggregation_method"],
        "expected_frames": config["expected_frames"],
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "default_threshold": 0.5,
        "threshold_calibration": "validation_video_level_only",
        "validation_max_f1_threshold": float(max_f1_thresh),
        "validation_eer_threshold": float(eer_thresh),
        "test_intrinsic_eer": float(test_eer),
        "test_intrinsic_eer_threshold": float(test_eer_threshold),
        "test_video_auroc": float(test_auroc),
        "test_video_pr_auc": float(test_prauc),
        "test_video_logloss": float(test_logloss),
        "macro_source_auroc": float(macro_auroc),
        "test_default_accuracy": metrics_05["Acc"],
        "test_default_f1": metrics_05["F1"],
        "test_validation_maxf1_accuracy": metrics_maxf1["Acc"],
        "test_validation_maxf1_precision": metrics_maxf1["Pre"],
        "test_validation_maxf1_recall": metrics_maxf1["Rec"],
        "test_validation_maxf1_f1": metrics_maxf1["F1"],
        "test_validation_eer_accuracy": metrics_eer["Acc"],
        "test_validation_eer_precision": metrics_eer["Pre"],
        "test_validation_eer_recall": metrics_eer["Rec"],
        "test_validation_eer_f1": metrics_eer["F1"],
        "test_clips": int(len(test_clip_df)),
        "test_videos": int(len(test_video_df)),
        "inference_seconds": float(elapsed_time),
        "clips_per_second": float(
            len(test_clip_df) / max(elapsed_time, 1e-8)
        ),
        "videos_per_second": float(
            len(test_video_df) / max(elapsed_time, 1e-8)
        ),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "pytorch_version": torch.__version__,
        "date": datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "protocol": (
            "Validation calibration -> frozen thresholds -> "
            "unseen test evaluation"
        ),
    }

    with open(
        output_dir / "evaluation_summary.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=4)

    # ========================================================
    # FINAL PUBLICATION REPORT
    # ========================================================
    print("\n" + "=" * 80)
    print("FINAL EVALUATION PROTOCOL REPORT")
    print("=" * 80)

    print(f"Pooled Video-Level AUROC       : {test_auroc:.4f}")
    print(f"Macro Source-Level AUROC       : {macro_auroc:.4f}")
    print(f"Pooled Video-Level PR-AUC      : {test_prauc:.4f}")
    print(f"Pooled Video-Level EER         : {test_eer:.4f}")
    print(f"Pooled Video-Level LogLoss     : {test_logloss:.4f}")

    print("-" * 80)

    print(
        f"Threshold = 0.5000 (Default)  | "
        f"Acc: {metrics_05['Acc']:.4f} | "
        f"Pre: {metrics_05['Pre']:.4f} | "
        f"Rec: {metrics_05['Rec']:.4f} | "
        f"F1: {metrics_05['F1']:.4f}"
    )

    print(
        f"Threshold = {max_f1_thresh:.4f} (Validation Max-F1) | "
        f"Acc: {metrics_maxf1['Acc']:.4f} | "
        f"Pre: {metrics_maxf1['Pre']:.4f} | "
        f"Rec: {metrics_maxf1['Rec']:.4f} | "
        f"F1: {metrics_maxf1['F1']:.4f}"
    )

    print(
        f"Threshold = {eer_thresh:.4f} (Validation EER) | "
        f"Acc: {metrics_eer['Acc']:.4f} | "
        f"Pre: {metrics_eer['Pre']:.4f} | "
        f"Rec: {metrics_eer['Rec']:.4f} | "
        f"F1: {metrics_eer['F1']:.4f}"
    )

    print("-" * 80)

    tn, fp, fn, tp = metrics_maxf1["CM"]
    print("Confusion Matrix @ Validation Max-F1")
    print(f"  TN: {tn:<6} FP: {fp:<6}")
    print(f"  FN: {fn:<6} TP: {tp:<6}")

    print("-" * 80)

    print(f"Total Test Clips Processed : {len(test_clip_df)}")
    print(f"Total Test Videos Processed: {len(test_video_df)}")
    print(f"Total Inference Time       : {elapsed_time:.2f} seconds")
    print(
        f"Throughput (Clips)         : "
        f"{len(test_clip_df) / max(elapsed_time, 1e-8):.2f} clips/sec"
    )
    print(
        f"Throughput (Videos)        : "
        f"{len(test_video_df) / max(elapsed_time, 1e-8):.2f} videos/sec"
    )
    print(
        f"Artifacts Saved To         : "
        f"{output_dir.resolve()}"
    )

    print("=" * 80)


if __name__ == "__main__":
    run_evaluation(CONFIG)