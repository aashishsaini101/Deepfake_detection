import hashlib
import random
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

SHARDS_ROOT = Path("/mnt/d/Dataset_Processed/shards")

EXPECTED_KEYS = {
    "tensors",
    "labels",
    "video_ids",
    "track_ids",
    "sources",
    "is_video",
}

# ============================================================

def md5(file_path):
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_folder(folder):

    print("=" * 80)
    print(f"Checking: {folder}")
    print("=" * 80)

    catalog_file = folder / "catalog.csv"

    if not catalog_file.exists():
        print("❌ catalog.csv missing")
        return False

    catalog = pd.read_csv(catalog_file)

    shard_files = sorted(folder.glob("shard_*.pt"))

    print(f"Catalog entries : {len(catalog)}")
    print(f"Shard files     : {len(shard_files)}")

    if len(shard_files) != len(catalog):
        print("❌ Catalog and shard count mismatch")
        return False

    total_samples = 0

    for shard_path in tqdm(shard_files):

        try:
            shard = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as e:
            print(f"\n❌ Cannot load {shard_path.name}")
            print(e)
            return False

        # --------------------------------------------------

        if set(shard.keys()) != EXPECTED_KEYS:
            print(f"\n❌ Missing keys in {shard_path.name}")
            print(shard.keys())
            return False

        tensors = shard["tensors"]
        labels = shard["labels"]

        video_ids = shard["video_ids"]
        track_ids = shard["track_ids"]
        sources = shard["sources"]

        is_video = shard["is_video"]

        # --------------------------------------------------
        # shape
        # --------------------------------------------------

        if is_video:

            expected = (16, 3, 224, 224)

            if tuple(tensors.shape[1:]) != expected:

                print(
                    f"\n❌ Shape mismatch in {shard_path.name}"
                )

                print(tensors.shape)

                return False

        else:

            expected = (1, 3, 224, 224)

            if tuple(tensors.shape[1:]) != expected:

                print(
                    f"\n❌ Shape mismatch in {shard_path.name}"
                )

                print(tensors.shape)

                return False

        # --------------------------------------------------
        # labels
        # --------------------------------------------------

        n = tensors.shape[0]

        if labels.shape[0] != n:

            print(f"\n❌ Label mismatch in {shard_path.name}")
            return False

        if len(video_ids) != n:
            print(f"\n❌ video_ids mismatch")
            return False

        if len(track_ids) != n:
            print(f"\n❌ track_ids mismatch")
            return False

        if len(sources) != n:
            print(f"\n❌ sources mismatch")
            return False

        # --------------------------------------------------
        # NaN
        # --------------------------------------------------

        if torch.isnan(tensors).any():
            print(f"\n❌ NaN detected")
            return False

        if torch.isinf(tensors).any():
            print(f"\n❌ Inf detected")
            return False

        # --------------------------------------------------
        # dtype
        # --------------------------------------------------

        if tensors.dtype not in (
            torch.float16,
            torch.float32,
        ):

            print(f"\n❌ Unexpected dtype")

            print(tensors.dtype)

            return False

        # --------------------------------------------------
        # MD5
        # --------------------------------------------------

        row = catalog[catalog.shard_file == shard_path.name]

        if len(row):

            if "md5_checksum" in row.columns:

                expected_md5 = row.iloc[0]["md5_checksum"]

                actual_md5 = md5(shard_path)

                if actual_md5 != expected_md5:

                    print(f"\n❌ MD5 mismatch")

                    print(shard_path.name)

                    return False

        total_samples += n

    print()

    print("✓ Folder verified successfully")

    print("Total samples:", total_samples)

    # ------------------------------------------------------
    # Random inspection
    # ------------------------------------------------------

    sample = random.choice(shard_files)

    shard = torch.load(
        sample,
        map_location="cpu",
        weights_only=True,
    )

    idx = random.randint(0, len(shard["labels"]) - 1)

    print("\nRandom Sample Inspection")
    print("------------------------")
    print("Shard      :", sample.name)
    print("Tensor     :", shard["tensors"][idx].shape)
    print("Label      :", shard["labels"][idx].item())
    print("Video ID   :", shard["video_ids"][idx])
    print("Track ID   :", shard["track_ids"][idx])
    print("Source     :", shard["sources"][idx])
    print("dtype      :", shard["tensors"].dtype)

    print()

    return True


def main():

    folders = [

        SHARDS_ROOT / "images" / "train",
        SHARDS_ROOT / "images" / "val",
        SHARDS_ROOT / "images" / "test",

        SHARDS_ROOT / "videos" / "train",
        SHARDS_ROOT / "videos" / "val",
        SHARDS_ROOT / "videos" / "test",

    ]

    success = True

    for folder in folders:

        if folder.exists():

            ok = verify_folder(folder)

            success &= ok

    print("=" * 80)

    if success:

        print("🎉 ALL SHARDS VERIFIED SUCCESSFULLY")
        print("It is now safe to delete the original tensor directory:")
        print()
        print("/home/aashish/deepfake_detection/data/tensors")

    else:

        print("❌ Verification failed.")
        print("Do NOT delete the original tensors.")

    print("=" * 80)


if __name__ == "__main__":
    main()