import os
import torch
import warnings
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Mute PyTorch's excessive serialization warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Configuration
SOURCE_TENSORS_DIR = Path("/home/aashish/deepfake_detection/data/tensors")
METADATA_CSV = Path("/home/aashish/deepfake_detection/data/metadata.csv")
HDD_SHARDS_DIR = Path("/mnt/d/Dataset_Processed/shards")
CORRUPTED_LOG = Path("/home/aashish/deepfake_detection/data/corrupted_files.log")

SHARD_SIZE = 256
USE_FP16 = True  # Toggle for Float16 storage optimization

def compute_md5(file_path: Path) -> str:
    """Computes MD5 checksum for data integrity verification."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def pack_shard_bucket(df_bucket: pd.DataFrame, output_dir: Path, is_video: bool):
    """Validates tensors, enforces shape/type, and packs into shards using fast iteration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if df_bucket.empty:
        return
        
    num_samples = len(df_bucket)
    num_shards = (num_samples + SHARD_SIZE - 1) // SHARD_SIZE
    shard_catalog = []
    
    missing_count = 0
    corrupted_count = 0
    log_buffer = []
    expected_shape = (16, 3, 224, 224) if is_video else (1, 3, 224, 224)

    for shard_idx in tqdm(range(num_shards), desc=f"Packing {output_dir.name}"):
        batch_df = df_bucket.iloc[shard_idx * SHARD_SIZE : (shard_idx + 1) * SHARD_SIZE]
        tensors, labels, video_ids, track_ids, sources = [], [], [], [], []
        
        # 4x Faster Iteration using itertuples
        for row in batch_df.itertuples(index=False):
            tensor_path = SOURCE_TENSORS_DIR / row.file_name
            
            if not tensor_path.exists():
                log_buffer.append(f"{row.file_name}: Missing file\n")
                missing_count += 1
                continue
                
            try:
                # Setting weights_only=True is safer and silences the PyTorch warning
                data = torch.load(tensor_path, map_location="cpu", weights_only=True)
                tensor = data['data']
            except Exception as e:
                log_buffer.append(f"{row.file_name}: Read Error ({str(e)})\n")
                corrupted_count += 1
                continue
                
            # Strict Validation: Dimensions
            if tuple(tensor.shape) != expected_shape:
                log_buffer.append(f"{row.file_name}: Invalid Shape {tuple(tensor.shape)}, expected {expected_shape}\n")
                corrupted_count += 1
                continue
                
            # Strict Validation: NaNs/Infs
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                log_buffer.append(f"{row.file_name}: Contains NaN/Inf\n")
                corrupted_count += 1
                continue

            # Standardize dtype & precision
            tensor = tensor.float()
            if USE_FP16:
                tensor = tensor.half()
            
            tensors.append(tensor)
            labels.append(row.label)
            video_ids.append(str(row.video_id))
            track_ids.append(int(getattr(row, 'track_id', -1)))
            sources.append(str(row.source))
            
        if not tensors:
            continue

        stacked_tensors = torch.stack(tensors)
        shard_file_name = f"shard_{shard_idx:04d}.pt"
        shard_path = output_dir / shard_file_name
        
        save_dict = {
            "tensors": stacked_tensors,
            "labels": torch.tensor(labels, dtype=torch.long),
            "video_ids": video_ids,
            "track_ids": track_ids,
            "sources": sources,
            "is_video": is_video
        }
        
        torch.save(save_dict, shard_path)
        
        shard_catalog.append({
            "shard_file": shard_file_name,
            "sample_count": len(tensors),
            "real_count": labels.count(0),
            "fake_count": labels.count(1),
            "md5_checksum": compute_md5(shard_path),
            "dtype": str(stacked_tensors.dtype),
            "tensor_shape": str(stacked_tensors.shape)
        })
        
    pd.DataFrame(shard_catalog).to_csv(output_dir / "catalog.csv", index=False)
    
    # Flush logs to disk once per bucket
    if log_buffer:
        with open(CORRUPTED_LOG, "a") as f_log:
            f_log.writelines(log_buffer)
            
    if missing_count > 0 or corrupted_count > 0:
        print(f"\nSkipped {missing_count} missing and {corrupted_count} invalid tensors. Logged to {CORRUPTED_LOG.name}")

def execute_sharding():
    print("--- Step 1: Auditing Metadata against Source Directory ---")
    df = pd.read_csv(METADATA_CSV)
    print(f"Total metadata records: {len(df)}")
    
    # Fast check for file existence
    # Note: We still keep this so we can accurately build our base DataFrame
    df['exists'] = df['file_name'].apply(lambda x: (SOURCE_TENSORS_DIR / x).exists())
    df = df[df['exists']].drop(columns=['exists']).reset_index(drop=True)
    print(f"Verified tensors on SSD: {len(df)}")
    
    print("\n--- Step 2: Compound Stratified Split (Source + Label) ---")
    
    # Check for Label Consistency
    label_counts = df.groupby("video_id")["label"].nunique()
    bad_vids = label_counts[label_counts > 1]
    if len(bad_vids) > 0:
        raise RuntimeError(f"CRITICAL: {len(bad_vids)} video_ids contain conflicting labels. Review preprocessing.")
    
    # Group by video_id to get label and source
    vid_meta = df.groupby("video_id")[["label", "source"]].first()
    unique_vids = vid_meta.index.to_numpy(dtype=str)
    
    # Create compound stratification array (e.g., "CelebDF_1")
    compound_strat = (vid_meta["source"].astype(str) + "_" + vid_meta["label"].astype(str)).to_numpy()
    
    # Handle extremely rare subsets that break Stratified Split
    unique_strats, counts = np.unique(compound_strat, return_counts=True)
    rare_strats = unique_strats[counts < 2]
    if len(rare_strats) > 0:
        print(f"Warning: These source+label combinations have fewer than 2 videos and may skew splits: {rare_strats}")
    
    # 80% Train, 20% Temp (Val + Test)
    train_vids, temp_vids, _, temp_strat = train_test_split(
        unique_vids, compound_strat, test_size=0.20, stratify=compound_strat, random_state=42
    )
    
    # 10% Val, 10% Test
    val_vids, test_vids, _, _ = train_test_split(
        temp_vids, temp_strat, test_size=0.50, stratify=temp_strat, random_state=42
    )
    
    train_df = df[df['video_id'].isin(train_vids)]
    val_df = df[df['video_id'].isin(val_vids)]
    test_df = df[df['video_id'].isin(test_vids)]
    
    print(f"Train samples: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("\n--- Step 3: Packing Shards directly to HDD ---")
    splits = [('train', train_df), ('val', val_df), ('test', test_df)]
    
    for split_name, split_df in splits:
        img_df = split_df[split_df['is_video'] == False]
        vid_df = split_df[split_df['is_video'] == True]
        
        print(f"\n[SPLIT: {split_name.upper()}]")
        print(f"Packing {len(img_df)} static images...")
        pack_shard_bucket(img_df, HDD_SHARDS_DIR / "images" / split_name, is_video=False)
        
        print(f"Packing {len(vid_df)} video sequences...")
        pack_shard_bucket(vid_df, HDD_SHARDS_DIR / "videos" / split_name, is_video=True)

    print("\n--- Sharding Complete ---")
    print(f"Shards generated at: {HDD_SHARDS_DIR}")

if __name__ == "__main__":
    execute_sharding()