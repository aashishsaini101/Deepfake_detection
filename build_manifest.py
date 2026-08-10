import json
from pathlib import Path
from collections import defaultdict

# ==========================================================
# Configuration
# ==========================================================
DATA_DIR = Path("/mnt/d/Dataset")
MANIFEST_PATH = Path("/home/aashish/deepfake_detection/data/manifest.json")

# Skip fake DFD videos because they contain multiple identities
# without identity-level manipulation annotations.
EXCLUDE_DFD_FAKES = True

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ==========================================================
# Dataset Label Resolver
# ==========================================================
def determine_label_and_source(parts):
    lower = [p.lower() for p in parts]

    # ---------- AI_DEEPFAKE ----------
    if "ai_deepfake" in lower:
        if any("real" in p for p in lower):
            return 0, "AI_DEEPFAKE"
        return 1, "AI_DEEPFAKE"

    # ---------- Celeb-DF ----------
    if "celeb_df" in lower:
        if any("synthesis" in p for p in lower):
            return 1, "Celeb_DF"
        if any("real" in p for p in lower):
            return 0, "Celeb_DF"

    # ---------- DeeperForensics ----------
    if any("deeperforensics" in p for p in lower):
        if any("fake" in p for p in lower):
            return 1, "DeeperForensics"
        if any("real" in p for p in lower):
            return 0, "DeeperForensics"

    # ---------- DF-40 ----------
    if any("df-40" in p for p in lower):
        if any("fake" in p for p in lower):
            return 1, "DF-40"
        if any("real" in p for p in lower):
            return 0, "DF-40"

    # ---------- DFD ----------
    if any(p == "dfd" for p in lower):

        if any("manipulated" in p for p in lower):
            if EXCLUDE_DFD_FAKES:
                return -1, "DFD_Fake"
            return 1, "DFD"

        if any("original" in p for p in lower):
            return 0, "DFD"

    # ---------- DFDC ----------
    if any("dfdc" in p for p in lower):
        if any("fake" in p for p in lower):
            return 1, "DFDC"
        if any("real" in p for p in lower):
            return 0, "DFDC"

    # ---------- FaceForensics++ ----------
    if any("faceforensics" in p for p in lower):

        if any("original" in p for p in lower):
            return 0, "FaceForensics++"

        fake_keywords = [
            "deepfakes",
            "deepfakedetection",
            "faceswap",
            "face2face",
            "faceshifter",
            "neuraltextures",
        ]

        if any(any(k in p for k in fake_keywords) for p in lower):
            return 1, "FaceForensics++"

    # ---------- HiDF ----------
    if any("hidf" in p for p in lower):
        if any("fake" in p for p in lower):
            return 1, "HiDF"
        if any("real" in p for p in lower):
            return 0, "HiDF"

    # ---------- UADFV ----------
    if any("uadfv" in p for p in lower):
        if any("fake" in p for p in lower):
            return 1, "UADFV"
        if any("real" in p for p in lower):
            return 0, "UADFV"

    return -1, "Unknown"


# ==========================================================
# Manifest Builder
# ==========================================================
def build_manifest():

    if not DATA_DIR.exists():
        print(f"Dataset directory not found:\n{DATA_DIR}")
        return

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest = []

    stats = defaultdict(lambda: {"fake": 0, "real": 0})
    skipped = defaultdict(int)
    seen = set()

    print(f"\nScanning dataset:\n{DATA_DIR}\n")

    for file_path in DATA_DIR.rglob("*"):

        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()

        if ext not in VIDEO_EXTS and ext not in IMAGE_EXTS:
            continue

        resolved = file_path.resolve()

        if resolved in seen:
            continue

        seen.add(resolved)

        # DF-40 contains only images
        if any("df-40" in p.lower() for p in file_path.parts):

            if ext in VIDEO_EXTS:
                skipped["DF-40 video"] += 1
                continue

        label, source = determine_label_and_source(file_path.parts)

        if label == -1:
            skipped[source] += 1
            continue

        rel_path = file_path.relative_to(DATA_DIR).as_posix()

        manifest.append(
            {
                "path": rel_path,
                "label": label,
                "is_video": ext in VIDEO_EXTS,
                "source": source,
            }
        )

        if label == 1:
            stats[source]["fake"] += 1
        else:
            stats[source]["real"] += 1

    # Save manifest as a LIST (compatible with most loaders)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=4)

    print("=" * 70)
    print("Manifest Created")
    print("=" * 70)
    print(MANIFEST_PATH)
    print()

    print(f"{'Dataset':25} {'Fake':>10} {'Real':>10}")
    print("-" * 50)

    total_fake = 0
    total_real = 0

    for dataset in sorted(stats):

        fake = stats[dataset]["fake"]
        real = stats[dataset]["real"]

        total_fake += fake
        total_real += real

        print(f"{dataset:25} {fake:10d} {real:10d}")

    print("-" * 50)
    print(f"{'TOTAL':25} {total_fake:10d} {total_real:10d}")

    if skipped:
        print("\nSkipped Files")

        for reason, count in sorted(skipped.items()):
            print(f"{reason:25} : {count}")

    print("\nManifest entries:", len(manifest))

    if EXCLUDE_DFD_FAKES:
        print("\nDFD fake videos were intentionally excluded.")


if __name__ == "__main__":
    build_manifest()