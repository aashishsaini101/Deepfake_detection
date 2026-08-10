import os
import csv
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from tqdm import tqdm

from dataset_streaming import ShardIterableDataset
from models.spatial_backbone import ConvNeXtSpatialBackbone
from models.temporal_transformer import TemporalTransformer
from models.temporal_model import SpatiotemporalDeepfakeModel

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "video_train_dir": "/home/aashish/deepfake_detection/data/shards/videos/train",
    "video_val_dir": "/home/aashish/deepfake_detection/data/shards/videos/val",
    "checkpoint_dir": "./checkpoints_temporal",
    "spatial_ckpt": "./checkpoints_spatial/spatial_backbone_best.pth",
    "log_file": "temporal_training_log.csv",
    "epochs": 25,
    "patience": 5,
    "batch_size": 8,  # Reduced batch size for 5D video tensors
    "num_workers": 4,
    "lr_backbone": 1e-5,
    "lr_transformer": 1e-4,
    "lr_head": 5e-4,
    "weight_decay": 1e-2,
    "unfreeze_epoch": 6,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def get_parameter_groups(model):
    backbone_params = []
    transformer_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        elif "temporal_transformer" in name:
            transformer_params.append(param)
        elif "head" in name:
            head_params.append(param)
            
    return [
        {"params": backbone_params, "lr": CONFIG["lr_backbone"], "weight_decay": CONFIG["weight_decay"]},
        {"params": transformer_params, "lr": CONFIG["lr_transformer"], "weight_decay": CONFIG["weight_decay"]},
        {"params": head_params, "lr": CONFIG["lr_head"], "weight_decay": CONFIG["weight_decay"]},
    ]

def plot_learning_curves(log_path, out_dir):
    try:
        df = pd.read_csv(log_path)
        epochs = df["epoch"]
        
        plt.figure(figsize=(16, 10))
        
        # Loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, df["train_loss"], label="Train Loss", marker='o')
        plt.plot(epochs, df["val_loss"], label="Val Loss", marker='o')
        plt.title("Loss vs Epoch")
        plt.legend()
        plt.grid(True)
        
        # AUROC
        plt.subplot(2, 2, 2)
        plt.plot(epochs, df["clip_auroc"], label="Clip AUROC", marker='o')
        plt.plot(epochs, df["video_auroc"], label="Video AUROC", marker='o', linestyle='--')
        plt.title("AUROC vs Epoch")
        plt.legend()
        plt.grid(True)
        
        # Accuracy
        plt.subplot(2, 2, 3)
        plt.plot(epochs, df["clip_acc"], label="Clip Acc", marker='o')
        plt.plot(epochs, df["video_acc"], label="Video Acc", marker='o', linestyle='--')
        plt.title("Accuracy vs Epoch")
        plt.legend()
        plt.grid(True)
        
        # F1 Score
        plt.subplot(2, 2, 4)
        plt.plot(epochs, df["clip_f1"], label="Clip F1", marker='o')
        plt.plot(epochs, df["video_f1"], label="Video F1", marker='o', linestyle='--')
        plt.title("F1 Score vs Epoch")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not generate plots: {e}")

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        tensors = batch["tensor"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float()
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
            logits = model(tensors)
            loss = criterion(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / total_samples

@torch.inference_mode()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    all_preds = []
    all_targets = []
    all_video_ids = []
    
    pbar = tqdm(dataloader, desc="Validating")
    for batch in pbar:
        tensors = batch["tensor"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float()
        video_ids = batch["video_id"]
        
        with autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
            logits = model(tensors)
            loss = criterion(logits, labels)
            
        probs = torch.sigmoid(logits)
        
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(probs.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        all_video_ids.extend(video_ids)
        
    val_loss = total_loss / len(all_targets)
    
    # ==========================
    # 1. Clip-Level Metrics
    # ==========================
    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    binary_preds = (preds_arr >= 0.5).astype(int)
    
    clip_acc = accuracy_score(targets_arr, binary_preds)
    clip_prec, clip_rec, clip_f1, _ = precision_recall_fscore_support(targets_arr, binary_preds, average="binary", zero_division=0)
    clip_auroc = roc_auc_score(targets_arr, preds_arr) if len(np.unique(targets_arr)) > 1 else 0.5
    clip_prauc = average_precision_score(targets_arr, preds_arr) if len(np.unique(targets_arr)) > 1 else 0.5

    # ==========================
    # 2. Video-Level Metrics
    # ==========================
    vid_preds_dict = defaultdict(list)
    vid_targets_dict = {}
    
    for p, t, vid in zip(all_preds, all_targets, all_video_ids):
        vid_preds_dict[vid].append(p)
        vid_targets_dict[vid] = t
        
    video_preds = []
    video_targets = []
    for vid, preds in vid_preds_dict.items():
        video_preds.append(np.mean(preds))  # Average chunk probabilities per video
        video_targets.append(vid_targets_dict[vid])
        
    v_preds_arr = np.array(video_preds)
    v_targets_arr = np.array(video_targets)
    v_binary_preds = (v_preds_arr >= 0.5).astype(int)
    
    vid_acc = accuracy_score(v_targets_arr, v_binary_preds)
    vid_prec, vid_rec, vid_f1, _ = precision_recall_fscore_support(v_targets_arr, v_binary_preds, average="binary", zero_division=0)
    vid_auroc = roc_auc_score(v_targets_arr, v_preds_arr) if len(np.unique(v_targets_arr)) > 1 else 0.5
    vid_prauc = average_precision_score(v_targets_arr, v_preds_arr) if len(np.unique(v_targets_arr)) > 1 else 0.5

    return {
        "val_loss": val_loss,
        "clip_acc": clip_acc, "clip_prec": clip_prec, "clip_rec": clip_rec, "clip_f1": clip_f1, "clip_auroc": clip_auroc, "clip_prauc": clip_prauc,
        "vid_acc": vid_acc, "vid_prec": vid_prec, "vid_rec": vid_rec, "vid_f1": vid_f1, "vid_auroc": vid_auroc, "vid_prauc": vid_prauc,
    }

def main():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    Path(CONFIG["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    device = CONFIG["device"]
    print(f"Using device: {device}")

    # CSV Logging Setup
    log_path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["log_file"])
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", 
            "clip_acc", "clip_prec", "clip_rec", "clip_f1", "clip_auroc", "clip_prauc",
            "video_acc", "video_prec", "video_rec", "video_f1", "video_auroc", "video_prauc", "lr_transformer"
        ])

    # 1. Datasets & Loaders
    train_dataset = ShardIterableDataset(CONFIG["video_train_dir"], shuffle=True, buffer_size=1024)
    val_dataset = ShardIterableDataset(CONFIG["video_val_dir"], shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"],
        pin_memory=True, persistent_workers=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], num_workers=2,
        pin_memory=True, drop_last=False
    )

    # 2. Model Setup
    backbone = ConvNeXtSpatialBackbone(pretrained=False)
    transformer = TemporalTransformer(num_frames=16, embed_dim=768, depth=2)
    model = SpatiotemporalDeepfakeModel(backbone, transformer).to(device)
    
    # Load pretrained spatial weights
    model.load_spatial_weights(CONFIG["spatial_ckpt"])

    # 3. Optimizer, Scheduler, Loss, Scaler
    criterion = nn.BCEWithLogitsLoss()
    param_groups = get_parameter_groups(model)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scaler = GradScaler("cuda" if "cuda" in device else "cpu")
    early_stopping = EarlyStopping(patience=CONFIG["patience"])

    best_vid_auroc = 0.0

    # 4. Training Loop
    print("\n--- Starting Spatiotemporal Training ---")
    for epoch in range(1, CONFIG["epochs"] + 1):
        
        # Progressive Unfreezing Logic
        if epoch == 1:
            model.set_backbone_requires_grad(False)
            print("\n[Epoch 1-5] Spatial Backbone Frozen. Training Temporal Transformer & Head.")
        elif epoch == CONFIG["unfreeze_epoch"]:
            model.set_backbone_requires_grad(True)
            print(f"\n[Epoch {epoch}] Spatial Backbone Unfrozen. Joint Training Begun.")

        start_time = time.time()
        current_lr_tf = optimizer.param_groups[1]["lr"]
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        elapsed = time.time() - start_time
        
        # Log to CSV
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, metrics["val_loss"],
                metrics["clip_acc"], metrics["clip_prec"], metrics["clip_rec"], metrics["clip_f1"], metrics["clip_auroc"], metrics["clip_prauc"],
                metrics["vid_acc"], metrics["vid_prec"], metrics["vid_rec"], metrics["vid_f1"], metrics["vid_auroc"], metrics["vid_prauc"],
                current_lr_tf
            ])
            
        print(f"\nEpoch {epoch:02d}/{CONFIG['epochs']:02d} [{elapsed:.1f}s]")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f}")
        print(f"  [Clip]  Acc: {metrics['clip_acc']*100:.2f}% | F1: {metrics['clip_f1']:.4f} | AUROC: {metrics['clip_auroc']:.4f} | PR-AUC: {metrics['clip_prauc']:.4f}")
        print(f"  [Video] Acc: {metrics['vid_acc']*100:.2f}% | F1: {metrics['vid_f1']:.4f} | AUROC: {metrics['vid_auroc']:.4f} | PR-AUC: {metrics['vid_prauc']:.4f}")
        
        # Save Full State
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "vid_auroc": metrics["vid_auroc"]
        }
        
        # Save last checkpoint
        torch.save(state, os.path.join(CONFIG["checkpoint_dir"], "last_checkpoint.pth"))
        
        # Save best checkpoint
        if metrics["vid_auroc"] > best_vid_auroc:
            best_vid_auroc = metrics["vid_auroc"]
            torch.save(state, os.path.join(CONFIG["checkpoint_dir"], "best_checkpoint.pth"))
            print(f"  --> Saved new best checkpoint! (Video AUROC: {best_vid_auroc:.4f})")
            
        # Update plots
        plot_learning_curves(log_path, CONFIG["checkpoint_dir"])

        # Early Stopping check
        early_stopping(metrics["vid_auroc"])
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}. No improvement in Video AUROC for {CONFIG['patience']} epochs.")
            break

if __name__ == "__main__":
    main()