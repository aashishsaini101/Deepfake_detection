import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from insightface.app import FaceAnalysis
from tracker import LightweightFaceByteTracker

# Configurations
MANIFEST_PATH = Path("/home/aashish/deepfake_detection/data/manifest.json")
OUTPUT_DIR = Path("/home/aashish/deepfake_detection/data/tensors")
METADATA_CSV = Path("/home/aashish/deepfake_detection/data/metadata.csv")
TARGET_FPS = 5
SEQ_LENGTH = 16  # Fixed window size for the temporal transformer
QUALITY_THRESHOLD = 0.40

# Standard 5-point facial landmarks for 224x224 alignment
# Order: Left Eye, Right Eye, Nose, Left Mouth, Right Mouth
REF_PTS = np.array([
    [76.5892, 103.3926],
    [147.0636, 103.0028],
    [112.0504, 143.4732],
    [83.0986, 184.731],
    [141.4598, 184.4082]
], dtype=np.float32)

# Initialize GPU-native detector ONCE globally
# allowed_modules=['detection'] skips unnecessary recognition weights for speed.
# ctx_id=0 binds it to your primary GPU.
app = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def align_face(img: np.ndarray, src_pts: np.ndarray) -> np.ndarray:
    """Aligns face to a standard 224x224 template using an affine transform."""
    # InsightFace natively provides keypoints as a 5x2 array in the correct order.
    if src_pts is None or len(src_pts) != 5:
        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    tform, inliers = cv2.estimateAffinePartial2D(src_pts, REF_PTS)
    if tform is None:
        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
    aligned_face = cv2.warpAffine(img, tform, (224, 224), flags=cv2.INTER_LANCZOS4)
    return aligned_face

def compute_quality_score(crop: np.ndarray, face_area: int, img_area: int) -> float:
    """Computes a normalized quality score Q including resolution and blur."""
    if crop is None or crop.size == 0 or img_area == 0:
        return 0.0
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(lap_var / 500.0, 1.0)
    
    # Penalize extremely tiny faces
    size_ratio = face_area / img_area
    size_score = min(size_ratio * 10, 1.0) 
    
    return (0.6 * blur_score) + (0.4 * size_score)

def flush_clip_to_disk(clip_buffer, timestamps, frame_indices, video_id, track_id, label, source, clip_idx, metadata_list):
    """Saves a 16-frame tensor clip and logs it to metadata."""
    tensor_data = torch.from_numpy(np.array(clip_buffer)).permute(0, 3, 1, 2).float() / 255.0
    file_name = f"{video_id}_t{track_id}_c{clip_idx:03d}.pt"
    save_path = OUTPUT_DIR / file_name
    
    torch.save({
        "video_id": video_id,
        "track_id": track_id,
        "label": label,
        "source": source,
        "data": tensor_data # [16, 3, 224, 224]
    }, save_path)
    
    metadata_list.append({
        "file_name": file_name,
        "video_id": video_id,
        "track_id": track_id,
        "clip_index": clip_idx,
        "label": label,
        "source": source,
        "is_video": True,
        "start_time": timestamps[0],
        "end_time": timestamps[-1],
        "start_frame": frame_indices[0],
        "end_frame": frame_indices[-1]
    })

def process_video(file_path: str, label: int, source: str, sample_id: str, metadata_list: list):
    """Processes video with constant memory footprint and fixed window chunking."""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(native_fps / TARGET_FPS))
    
    tracker = LightweightFaceByteTracker(high_thresh=0.6, low_thresh=0.1, match_thresh=0.8)
    
    # Buffers to hold tracks up to SEQ_LENGTH
    track_buffers = {}
    track_clip_counts = {}
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            h, w, _ = frame.shape
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Detect (GPU Accelerated)
            faces = app.get(frame)
                
            detections = []
            landmarks_dict = {}
            
            for face in faces:
                x1, y1, x2, y2 = face.bbox
                score = face.det_score
                det_box = [x1, y1, x2, y2, score]
                detections.append(det_box)
                # Use a rounded tuple of the bbox as a key to associate landmarks to tracks
                bbox_key = (int(x1), int(y1), int(x2), int(y2))
                landmarks_dict[bbox_key] = face.kps  # 5x2 array
            
            # Track
            active_tracks = tracker.update(detections)
            
            for track in active_tracks:
                tx1, ty1, tx2, ty2, score, track_id = track
                
                # Match track back to original detection for precise landmarks
                best_iou = 0
                best_lm = None
                for (dx1, dy1, dx2, dy2), lm in landmarks_dict.items():
                    ix1, iy1 = max(tx1, dx1), max(ty1, dy1)
                    ix2, iy2 = min(tx2, dx2), min(ty2, dy2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter > best_iou:
                        best_iou = inter
                        best_lm = lm
                
                # Expand box slightly to capture full head/chin
                bw, bh = tx2 - tx1, ty2 - ty1
                px1, py1 = max(0, int(tx1 - bw*0.1)), max(0, int(ty1 - bh*0.1))
                px2, py2 = min(w, int(tx2 + bw*0.1)), min(h, int(ty2 + bh*0.1))
                
                crop = frame[py1:py2, px1:px2]
                face_area = bw * bh
                img_area = w * h
                
                q_score = compute_quality_score(crop, face_area, img_area)
                
                if q_score >= QUALITY_THRESHOLD:
                    # FIX: Shift landmarks to the local coordinate space of the crop before affine transform
                    shifted_lm = best_lm - np.array([px1, py1]) if best_lm is not None else None
                    aligned_rgb = align_face(crop, shifted_lm)
                    aligned_rgb = cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2RGB)
                    
                    if track_id not in track_buffers:
                        track_buffers[track_id] = {"frames": [], "times": [], "indices": []}
                        track_clip_counts[track_id] = 0
                        
                    track_buffers[track_id]["frames"].append(aligned_rgb)
                    track_buffers[track_id]["times"].append(timestamp)
                    track_buffers[track_id]["indices"].append(count)
                    
                    # Flush to disk when chunk reaches exactly SEQ_LENGTH
                    if len(track_buffers[track_id]["frames"]) == SEQ_LENGTH:
                        flush_clip_to_disk(
                            track_buffers[track_id]["frames"],
                            track_buffers[track_id]["times"],
                            track_buffers[track_id]["indices"],
                            sample_id, track_id, label, source,
                            track_clip_counts[track_id],
                            metadata_list
                        )
                        track_clip_counts[track_id] += 1
                        
                        # Clear buffer to maintain constant RAM usage
                        track_buffers[track_id] = {"frames": [], "times": [], "indices": []}
                        
        count += 1
    cap.release()

def process_image(file_path: str, label: int, source: str, sample_id: str, metadata_list: list):
    """Processes static images (Phase 1 pretraining data) using the same alignment pipeline."""
    frame = cv2.imread(file_path)
    if frame is None:
        return
        
    h, w, _ = frame.shape
    faces = app.get(frame)
    if not faces:
        return

    face_idx = 0
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        landmarks = face.kps
        
        bw, bh = x2 - x1, y2 - y1
        px1, py1 = max(0, int(x1 - bw*0.1)), max(0, int(y1 - bh*0.1))
        px2, py2 = min(w, int(x2 + bw*0.1)), min(h, int(y2 + bh*0.1))
        
        crop = frame[py1:py2, px1:px2]
        
        q_score = compute_quality_score(crop, bw * bh, w * h)
        if q_score >= QUALITY_THRESHOLD:
            # FIX: Shift landmarks to the local coordinate space of the crop
            shifted_lm = landmarks - np.array([px1, py1]) if landmarks is not None else None
            aligned_rgb = align_face(crop, shifted_lm)
            aligned_rgb = cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2RGB)
            
            # Save as a single-frame tensor for spatial pretraining
            tensor_data = torch.from_numpy(aligned_rgb).permute(2, 0, 1).float() / 255.0
            
            file_name = f"{sample_id}_f{face_idx:02d}.pt"
            save_path = OUTPUT_DIR / file_name
            
            torch.save({
                "video_id": sample_id,
                "label": label,
                "source": source,
                "data": tensor_data.unsqueeze(0) # [1, 3, 224, 224]
            }, save_path)
            
            metadata_list.append({
                "file_name": file_name,
                "video_id": sample_id,
                "track_id": face_idx,
                "clip_index": 0,
                "label": label,
                "source": source,
                "is_video": False,
                "start_time": 0.0,
                "end_time": 0.0,
                "start_frame": 0,
                "end_frame": 0
            })
            face_idx += 1

def execute():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define the root dataset directory to resolve relative paths
    DATA_ROOT = Path("/mnt/d/Dataset")
    
    if not MANIFEST_PATH.exists():
        print("Manifest missing.")
        return

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    metadata = []
    
    print(f"Processing {len(manifest)} items...")
    for idx, item in enumerate(tqdm(manifest)):
        sample_id = f"s_{idx:06d}"
        
        # Reconstruct absolute path
        abs_path = str(DATA_ROOT / item["path"])
        
        if item["is_video"]:
            process_video(abs_path, item["label"], item["source"], sample_id, metadata)
        else:
            process_image(abs_path, item["label"], item["source"], sample_id, metadata)
            
    df = pd.DataFrame(metadata)
    df.to_csv(METADATA_CSV, index=False)
    print(f"Preprocessing complete. Metadata saved to {METADATA_CSV}")

if __name__ == "__main__":
    execute()