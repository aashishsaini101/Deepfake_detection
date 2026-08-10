import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

from dataset_streaming import ShardIterableDataset
from models.spatial_backbone import ConvNeXtSpatialBackbone

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "image_train_dir": "/home/aashish/deepfake_detection/data/shards/images/train",
    "image_val_dir": "/home/aashish/deepfake_detection/data/shards/images/val",
    "checkpoint_dir": "./checkpoints_spatial",
    "log_file": "training_log.csv",
    "model_name": "convnext_tiny",
    "epochs": 5,                     # Warm-start phase on image shards
    "batch_size": 32,
    "num_workers": 4,
    "backbone_lr": 1e-5,
    "head_lr": 5e-4,
    "weight_decay": 1e-2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def get_parameter_groups(model, backbone_lr, head_lr, weight_decay):
    """Splits parameters into backbone (pretrained) and head with distinct LRs."""
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]

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

        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

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
        
        with autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
            logits = model(tensors)
            loss = criterion(logits, labels)
            
        probs = torch.sigmoid(logits)
        
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(probs.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        all_video_ids.extend(batch["video_id"])
        
    val_loss = total_loss / len(all_targets)
    
    # Vectorized Metric Calculations
    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    binary_preds = (preds_arr >= 0.5).astype(int)
    
    acc = accuracy_score(targets_arr, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_arr, binary_preds, average="binary", zero_division=0
    )
    
    try:
        auroc = roc_auc_score(targets_arr, preds_arr)
    except ValueError:
        auroc = 0.5  # Fallback if single class present
        
    metrics = {
        "val_loss": val_loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc
    }
    
    return metrics

def main():
    # 0. Hardware Optimizations for Ampere RTX 3080 Ti
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    Path(CONFIG["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    device = CONFIG["device"]
    print(f"Using device: {device} (TF32 and cuDNN benchmark enabled)")

    # Initialize CSV Logging
    log_path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["log_file"])
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "acc", "precision", "recall", "f1", "auroc", "lr"])

    # 1. Dataset & DataLoaders
    train_dataset = ShardIterableDataset(CONFIG["image_train_dir"], shuffle=True, buffer_size=2048)
    val_dataset = ShardIterableDataset(CONFIG["image_val_dir"], shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True  # Consistent batch sizing for training
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    # 2. Model Initialization
    model = ConvNeXtSpatialBackbone(
        model_name=CONFIG["model_name"],
        pretrained=True,
        drop_path_rate=0.2,
        num_classes=1
    ).to(device)

    # 3. Loss, Optimizer, Scaler, Scheduler
    criterion = nn.BCEWithLogitsLoss()
    param_groups = get_parameter_groups(
        model, CONFIG["backbone_lr"], CONFIG["head_lr"], CONFIG["weight_decay"]
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scaler = GradScaler("cuda" if "cuda" in device else "cpu")

    best_auroc = 0.0

    # 4. Training Loop
    print("\n--- Starting Spatial Warm-Start Training ---")
    for epoch in range(1, CONFIG["epochs"] + 1):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        elapsed = time.time() - start_time
        
        # Log to CSV
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, val_metrics["val_loss"], val_metrics["acc"],
                val_metrics["precision"], val_metrics["recall"], val_metrics["f1"],
                val_metrics["auroc"], current_lr
            ])
        
        print(f"\nEpoch {epoch:02d}/{CONFIG['epochs']:02d} [{elapsed:.1f}s] | Current LR: {current_lr:.6f}")
        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Val Loss   : {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['acc']*100:.2f}% | Val AUROC: {val_metrics['auroc']:.4f}")
        print(f"  Precision  : {val_metrics['precision']:.4f} | Recall : {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
        
        # Save Best Checkpoint based on AUROC
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            save_path = os.path.join(CONFIG["checkpoint_dir"], "spatial_backbone_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auroc": val_metrics["auroc"],
                "val_acc": val_metrics["acc"],
                "val_f1": val_metrics["f1"]
            }, save_path)
            print(f"  Saved best checkpoint to {save_path} (AUROC: {best_auroc:.4f})")

if __name__ == "__main__":
    main()