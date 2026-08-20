#!/usr/bin/env python3
"""
Single-Video Real-Time Inference Script

Processes an un-sharded MP4 video using the exact preprocess_pipeline_v3 logic
and outputs clip-level + video-level deepfake probability predictions.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import torch

from insightface.app import FaceAnalysis
from tracker import LightweightFaceByteTracker

from models.spatial_backbone import ConvNeXtSpatialBackbone
from models.temporal_transformer import TemporalTransformer
from models.temporal_model import SpatiotemporalDeepfakeModel

# ============================================================
# PREPROCESSING CONSTANTS (Matching preprocess_pipeline_v3)
# ============================================================

TARGET_FPS = 5
SEQ_LENGTH = 16
QUALITY_THRESHOLD = 0.40

REF_PTS = np.array([
    [76.5892, 103.3926],
    [147.0636, 103.0028],
    [112.0504, 143.4732],
    [83.0986, 184.731],
    [141.4598, 184.4082]
], dtype=np.float32)


def align_face(img: np.ndarray, src_pts: np.ndarray) -> np.ndarray:
    if src_pts is None or len(src_pts) != 5:
        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    tform, _ = cv2.estimateAffinePartial2D(src_pts, REF_PTS)
    if tform is None:
        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    aligned_face = cv2.warpAffine(img, tform, (224, 224), flags=cv2.INTER_LANCZOS4)
    return aligned_face


def compute_quality_score(crop: np.ndarray, face_area: int, img_area: int) -> float:
    if crop is None or crop.size == 0 or img_area == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(lap_var / 500.0, 1.0)

    size_ratio = face_area / img_area
    size_score = min(size_ratio * 10, 1.0)

    return (0.6 * blur_score) + (0.4 * size_score)


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def load_spatiotemporal_model(checkpoint_path: str, device: torch.device):
    backbone = ConvNeXtSpatialBackbone(pretrained=False, num_classes=1)
    transformer = TemporalTransformer(
        num_frames=16,
        embed_dim=768,
        depth=2,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1
    )
    model = SpatiotemporalDeepfakeModel(
        backbone,
        transformer,
        embed_dim=768,
        num_classes=1
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


# ============================================================
# EXTRACT TENSOR CLIPS
# ============================================================

def extract_video_clips(video_path: str, app: FaceAnalysis):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(native_fps / TARGET_FPS))

    tracker = LightweightFaceByteTracker(high_thresh=0.6, low_thresh=0.1, match_thresh=0.8)

    track_buffers = {}
    completed_clips = []

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            h, w, _ = frame.shape
            faces = app.get(frame)

            detections = []
            landmarks_dict = {}

            for face in faces:
                x1, y1, x2, y2 = face.bbox
                score = face.det_score
                detections.append([x1, y1, x2, y2, score])
                bbox_key = (int(x1), int(y1), int(x2), int(y2))
                landmarks_dict[bbox_key] = face.kps

            active_tracks = tracker.update(detections)

            for track in active_tracks:
                tx1, ty1, tx2, ty2, score, track_id = track

                best_iou = 0
                best_lm = None
                for (dx1, dy1, dx2, dy2), lm in landmarks_dict.items():
                    ix1, iy1 = max(tx1, dx1), max(ty1, dy1)
                    ix2, iy2 = min(tx2, dx2), min(ty2, dy2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter > best_iou:
                        best_iou = inter
                        best_lm = lm

                bw, bh = tx2 - tx1, ty2 - ty1
                px1, py1 = max(0, int(tx1 - bw * 0.1)), max(0, int(ty1 - bh * 0.1))
                px2, py2 = min(w, int(tx2 + bw * 0.1)), min(h, int(ty2 + bh * 0.1))

                crop = frame[py1:py2, px1:px2]
                q_score = compute_quality_score(crop, bw * bh, w * h)

                if q_score >= QUALITY_THRESHOLD:
                    shifted_lm = best_lm - np.array([px1, py1]) if best_lm is not None else None
                    aligned_rgb = align_face(crop, shifted_lm)
                    aligned_rgb = cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2RGB)

                    if track_id not in track_buffers:
                        track_buffers[track_id] = []

                    track_buffers[track_id].append(aligned_rgb)

                    if len(track_buffers[track_id]) == SEQ_LENGTH:
                        # Convert to tensor: [16, 3, 224, 224], float32 in [0, 1]
                        clip_np = np.array(track_buffers[track_id])
                        clip_tensor = torch.from_numpy(clip_np).permute(0, 3, 1, 2).float() / 255.0
                        completed_clips.append((track_id, clip_tensor))

                        # Reset buffer for continuous processing
                        track_buffers[track_id] = []

        count += 1

    cap.release()
    return completed_clips


# ============================================================
# MAIN INFERENCE EXECUTION
# ============================================================

def run_single_video_inference():
    parser = argparse.ArgumentParser(description="Single-video deepfake inference")
    parser.add_argument(
        "--video",
        default="/home/aashish/deepfake_detection/testing_video/test_video (3).mp4",
        help="Path to input MP4 video file"
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints_temporal/best_checkpoint.pth",
        help="Path to trained model checkpoint"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("SINGLE-VIDEO REAL-TIME INFERENCE")
    print("=" * 80)
    print(f"Video Path : {args.video}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print("=" * 80)

    # 1. Initialize Face Detector
    print("\n[1/3] Initializing InsightFace Detector...")
    app = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 2. Extract and Preprocess Clips
    print("[2/3] Extracting aligned 16-frame clips from video...")
    clips = extract_video_clips(args.video, app)

    if not clips:
        print("\nERROR: No valid 16-frame face tracks meeting quality thresholds were extracted.")
        return

    print(f"✓ Extracted {len(clips)} clip(s) of length 16.")

    # 3. Load Model and Execute Inference
    print("[3/3] Running Spatiotemporal Model Inference...")
    model = load_spatiotemporal_model(args.checkpoint, device)

    clip_probs = []

    with torch.inference_mode():
        for i, (track_id, clip_tensor) in enumerate(clips):
            # Input shape: [1, 16, 3, 224, 224]
            x = clip_tensor.unsqueeze(0).to(device)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
                features = model.backbone.extract_features(x)
                temporal_cls = model.temporal_transformer(features)
                temporal_logits = model.head(temporal_cls).squeeze(-1)
                prob = torch.sigmoid(temporal_logits).item()

            clip_probs.append(prob)
            print(f"  Clip {i + 1:02d} (Track #{track_id}): Fake Probability = {prob:.4f} ({prob * 100:.2f}%)")

    # 4. Video-Level Aggregation
    mean_prob = float(np.mean(clip_probs))
    median_prob = float(np.median(clip_probs))
    max_prob = float(np.max(clip_probs))

    # Classification decision based on Phase 3 optimal pooling (Mean) with standard 0.5 threshold
    label_str = "DEEPFAKE (FAKE)" if mean_prob >= 0.5 else "REAL (GENUINE)"

    print("\n" + "=" * 80)
    print("VIDEO-LEVEL AGGREGATION RESULTS")
    print("=" * 80)
    print(f"Mean Aggregated Probability   : {mean_prob:.4f} ({mean_prob * 100:.2f}%)")
    print(f"Median Aggregated Probability : {median_prob:.4f} ({median_prob * 100:.2f}%)")
    print(f"Max Aggregated Probability    : {max_prob:.4f} ({max_prob * 100:.2f}%)")
    print("-" * 80)
    print(f"FINAL PREDICTION              : {label_str}")
    print("=" * 80)


if __name__ == "__main__":
    run_single_video_inference()