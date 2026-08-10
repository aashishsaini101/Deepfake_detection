import math
import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

class ShardIterableDataset(IterableDataset):
    def __init__(self, shard_dir, shuffle=True, buffer_size=2048):
        """
        Args:
            shard_dir (str or Path): Directory containing shard_*.pt files.
            shuffle (bool): Whether to shuffle shards and maintain a random buffer.
            buffer_size (int): Number of samples to hold in RAM for randomization.
        """
        self.shard_paths = sorted(Path(shard_dir).glob("shard_*.pt"))
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        
        self.expected_keys = {
            "tensors", "labels", "video_ids", "track_ids", "sources"
        }

        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found in {shard_dir}")

    def __iter__(self):
        worker_info = get_worker_info()
        
        # 1. Distribute Shards Across Workers and Handle RNG Seeding
        if worker_info is None:
            worker_shards = self.shard_paths
            # Fallback RNG for single-process
            rng = random.Random()
        else:
            per_worker = int(math.ceil(len(self.shard_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.shard_paths))
            worker_shards = self.shard_paths[start:end]
            
            # Ensure deterministic shuffling across epochs for each worker
            worker_seed = torch.initial_seed() % (2**32)
            rng = random.Random(worker_seed)

        # 2. Shuffle Shard Order
        if self.shuffle:
            rng.shuffle(worker_shards)

        buffer = []

        # 3. Stream and Yield Data
        for shard_path in worker_shards:
            # mmap=True maps the file to memory without fully loading it into RAM
            shard_data = torch.load(
                shard_path, 
                map_location="cpu", 
                weights_only=True,
                mmap=True 
            )
            
            # Sanity Check
            if not self.expected_keys.issubset(shard_data.keys()):
                raise RuntimeError(f"Missing expected keys in {shard_path.name}")
            
            num_samples = shard_data["tensors"].shape[0]
            indices = list(range(num_samples))
            
            if self.shuffle:
                rng.shuffle(indices)

            for i in indices:
                sample = {
                    "tensor": shard_data["tensors"][i],
                    "label": shard_data["labels"][i],
                    "video_id": shard_data["video_ids"][i],
                    "track_id": shard_data["track_ids"][i],
                    "source": shard_data["sources"][i]
                }
                
                if self.shuffle and self.buffer_size > 1:
                    buffer.append(sample)
                    if len(buffer) >= self.buffer_size:
                        yield buffer.pop(rng.randrange(len(buffer)))
                else:
                    yield sample

        # Empty the remaining buffer at the end of the epoch
        while buffer:
            if self.shuffle:
                yield buffer.pop(rng.randrange(len(buffer)))
            else:
                yield buffer.pop(0)

# ============================================================
# QUICK TEST BLOCK
# ============================================================
if __name__ == "__main__":
    TRAIN_DIR = "/home/aashish/deepfake_detection/data/shards/videos/train"
    
    # Train dataset with shuffle
    train_dataset = ShardIterableDataset(TRAIN_DIR, shuffle=True, buffer_size=2048)
    
    loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,  # Keeps workers alive between epochs
        prefetch_factor=4         # Prepares 4 batches ahead per worker
    )

    print(f"Testing High-Throughput DataLoader on: {TRAIN_DIR}")
    
    # Simulating 2 epochs to verify persistent workers and mmap performance
    for epoch in range(2):
        print(f"\n--- Epoch {epoch + 1} ---")
        for batch_idx, batch in enumerate(loader):
            print(f"Batch {batch_idx + 1}")
            print(f" - Tensors: {batch['tensor'].shape} | dtype: {batch['tensor'].dtype}")
            print(f" - Labels:  {batch['label'].shape}")
            
            if batch_idx == 1:  # Stop after 2 batches per epoch
                break
                
    print("\nDataLoader test completed successfully.")