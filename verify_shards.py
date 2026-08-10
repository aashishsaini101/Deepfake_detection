import os
import torch
import pandas as pd
import hashlib
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "base_dir": "/home/aashish/deepfake_detection/data/shards/videos",
    "splits": ["train", "val", "test"],
    "expected_keys": {"tensors", "labels", "video_ids", "track_ids", "sources", "is_video"},
    "expected_dtype": torch.float16,
    "spatial_shape": (3, 224, 224) # Last 3 dims must match this
}

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def normalize_list(data):
    """Converts tensors to lists to prevent Pandas silent errors."""
    if torch.is_tensor(data):
        return data.cpu().tolist()
    return data

def verify_dataset(config):
    base_dir = Path(config["base_dir"])
    
    split_dfs = {}
    corrupted_files = []
    missing_files = []
    
    total_videos = 0
    total_samples = 0
    
    print(f"Starting rigorous dataset verification in {base_dir}...\n")
    
    for split in config["splits"]:
        split_dir = base_dir / split
        catalog_path = split_dir / "catalog.csv"
        
        if not catalog_path.exists():
            print(f"⚠️ Warning: No catalog found for split '{split}' at {catalog_path}")
            continue
            
        catalog_df = pd.read_csv(catalog_path)
        metadata_records = []
        
        print(f"[{split.upper()}] Validating shards...")
        for row in tqdm(catalog_df.itertuples(index=False), total=len(catalog_df)):
            shard_file = row.shard_file
            shard_path = split_dir / shard_file
            
            # 1. File Existence
            if not shard_path.exists():
                missing_files.append(shard_path.name)
                continue
                
            # 2. MD5 Verification
            actual_md5 = compute_md5(shard_path)
            if hasattr(row, 'md5_checksum') and actual_md5 != row.md5_checksum:
                corrupted_files.append(f"{shard_file} (MD5 mismatch)")
                continue

            # 3. Load Payload
            try:
                shard_data = torch.load(shard_path, map_location='cpu', weights_only=False)
            except Exception as e:
                corrupted_files.append(f"{shard_file} (Load failed: {e})")
                continue
                
            # 4. Key Verification
            missing_keys = config["expected_keys"] - set(shard_data.keys())
            if missing_keys:
                corrupted_files.append(f"{shard_file} (Missing keys: {missing_keys})")
                continue
                
            tensors = shard_data["tensors"]
            num_samples = len(tensors)
            
            # 5. Catalog Sample Count Verification
            if hasattr(row, 'sample_count') and num_samples != row.sample_count:
                corrupted_files.append(f"{shard_file} (Catalog count mismatch: {num_samples} vs {row.sample_count})")
                continue

            # 6. Tensor Integrity Verification
            if tensors.dtype != config["expected_dtype"]:
                corrupted_files.append(f"{shard_file} (Dtype mismatch: {tensors.dtype})")
                continue
            if tensors.shape[-3:] != config["spatial_shape"]:
                corrupted_files.append(f"{shard_file} (Shape mismatch: {tensors.shape})")
                continue
            if torch.isnan(tensors).any().item():
                corrupted_files.append(f"{shard_file} (Contains NaNs)")
                continue
                
            # 7. Normalization & Length Assertions
            labels = normalize_list(shard_data["labels"])
            video_ids = normalize_list(shard_data["video_ids"])
            track_ids = normalize_list(shard_data["track_ids"])
            sources = normalize_list(shard_data["sources"])
            
            assert len(labels) == num_samples, f"Label length mismatch in {shard_file}"
            assert len(video_ids) == num_samples, f"Video ID length mismatch in {shard_file}"
            assert len(track_ids) == num_samples, f"Track ID length mismatch in {shard_file}"
            assert len(sources) == num_samples, f"Source length mismatch in {shard_file}"
            
            # Construct DataFrame
            shard_df = pd.DataFrame({
                "split": [split] * num_samples,
                "video_id": video_ids,
                "label": labels,
                "source": sources,
                "track_id": track_ids
            })
            
            metadata_records.append(shard_df)
            
        if metadata_records:
            split_dfs[split] = pd.concat(metadata_records, ignore_index=True)
            
    if not split_dfs:
        raise ValueError("No valid data extracted. Verification aborted.")
        
    full_metadata = pd.concat(split_dfs.values(), ignore_index=True)
    
    # ==========================================
    # STATISTICS & LEAKAGE COMPUTATION
    # ==========================================
    video_sets = {split: set(df["video_id"]) for split, df in split_dfs.items()}
    
    train_val_leak = len(video_sets.get("train", set()) & video_sets.get("val", set()))
    train_test_leak = len(video_sets.get("train", set()) & video_sets.get("test", set()))
    val_test_leak = len(video_sets.get("val", set()) & video_sets.get("test", set()))
    
    total_leaks = train_val_leak + train_test_leak + val_test_leak
    
    # Label Consistency Check (Intra-split)
    inconsistent_labels = full_metadata.groupby("video_id")["label"].nunique()
    inconsistent_count = len(inconsistent_labels[inconsistent_labels > 1])
    
    # Distributions
    unique_vids = full_metadata.drop_duplicates(subset=["video_id"])
    source_counts = unique_vids["source"].value_counts(normalize=True) * 100
    class_counts = unique_vids["label"].value_counts(normalize=True) * 100
    
    clips_per_video = full_metadata.groupby("video_id").size()
    
    # ==========================================
    # PRINT PUBLICATION SUMMARY
    # ==========================================
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    
    print(f"{'Total Samples':<25}: {len(full_metadata)}")
    print(f"{'Total Videos':<25}: {len(unique_vids)}")
    print(f"{'Total Tracks':<25}: {full_metadata['track_id'].nunique()}")
    
    print("-" * 50)
    print("Class Balance (Unique Videos)")
    for lbl, pct in class_counts.items():
        name = "Fake (1)" if lbl == 1 else "Real (0)"
        print(f"  {name:<23}: {pct:.1f}%")
        
    print("-" * 50)
    print("Source Balance (Unique Videos)")
    for src, pct in source_counts.items():
        print(f"  {src:<23}: {pct:.1f}%")
        
    print("-" * 50)
    print(f"{'Average Clips / Video':<25}: {clips_per_video.mean():.2f}")
    print(f"{'Min Clips / Video':<25}: {clips_per_video.min()}")
    print(f"{'Max Clips / Video':<25}: {clips_per_video.max()}")
    
    print("-" * 50)
    print("Data Integrity & Validation")
    print(f"{'Missing Files':<25}: {len(missing_files)}")
    print(f"{'Corrupted/Failed Shards':<25}: {len(corrupted_files)}")
    print(f"{'Label Inconsistencies':<25}: {inconsistent_count}")
    
    print("-" * 50)
    print("Cross-Split Leakage Check")
    print(f"  Train ∩ Val   : {train_val_leak} {'❌' if train_val_leak else '✓'}")
    print(f"  Train ∩ Test  : {train_test_leak} {'❌' if train_test_leak else '✓'}")
    print(f"  Val ∩ Test    : {val_test_leak} {'❌' if val_test_leak else '✓'}")
    
    print("="*50)
    if total_leaks == 0 and len(corrupted_files) == 0 and inconsistent_count == 0:
        print("DATASET VERIFIED ✓")
    else:
        print("DATASET VERIFICATION FAILED ❌")
        if corrupted_files:
            print("\nCorrupted Files Log:")
            for cf in corrupted_files[:10]:
                print(f" - {cf}")
    print("="*50)
    
    return {
        "metadata": full_metadata,
        "num_videos": len(unique_vids),
        "num_samples": len(full_metadata),
        "class_distribution": class_counts.to_dict(),
        "source_distribution": source_counts.to_dict()
    }

if __name__ == "__main__":
    _ = verify_dataset(CONFIG)