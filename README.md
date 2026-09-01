# Spatiotemporal Deepfake Detection via Decoupled Spatial-Temporal Transformers
 
<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Model-ConvNeXt--Tiny-5C3EE8" alt="ConvNeXt-Tiny"/>
  <img src="https://img.shields.io/badge/Temporal-Transformer-FF6F00" alt="Temporal Transformer"/>
  <img src="https://img.shields.io/badge/Task-Deepfake%20Detection-8A2BE2" alt="Deepfake Detection"/>
</p>
<p align="center">
  <b>Identity-aware spatiotemporal deepfake detection for facial video sequences</b>
</p>
<p align="center">
  <i>Decouple spatial artifact extraction from temporal inconsistency modeling.</i>
</p>
---
 
## 📌 Overview
 
This repository contains the official implementation of **Spatiotemporal Deepfake Detection via Decoupled Spatial-Temporal Transformers**, a deepfake detection architecture designed to identify generative manipulations in facial video sequences by capturing both spatial artifacts (e.g., blending anomalies, texture inconsistencies) and temporal domain irregularities (e.g., flickering, unnatural motion).
 
The proposed architecture explicitly separates:
 
- **Spatial modeling** — detection of blending artifacts, texture inconsistencies, boundary artifacts, and other frame-level visual cues.
- **Temporal modeling** — detection of flickering, facial jitter, unnatural motion, temporal discontinuities, and inconsistent facial dynamics.
Instead of using an expensive 3D convolutional architecture, the model first extracts independent frame-level representations using **ConvNeXt-Tiny** and then models their temporal evolution using a lightweight **Transformer Encoder**.
 
The complete pipeline is identity-aware through **InsightFace + ByteTrack**, ensuring that temporal modeling is performed only across frames belonging to the same individual — preventing the entanglement of temporal features between different subjects in crowded scenes.
 
By decoupling spatial feature extraction from temporal sequence modeling, the architecture achieves state-of-the-art generalization across unseen generative models while maintaining a highly efficient computational footprint suitable for real-time video stream analysis.
 
## ⭐ Key Result
 
The proposed spatial-temporal architecture improves Macro-AUROC from:
 
**0.6293 → 0.9480**
 
compared with the spatial-only baseline — an absolute gain of **+31.87%**.
 
## 🚀 Key Features
 
- 🎯 Spatiotemporal deepfake detection
- 🧠 ConvNeXt-Tiny spatial feature extraction
- ⏱️ Lightweight 2-layer Temporal Transformer
- 👤 Identity-aware tracking with ByteTrack
- 👁️ InsightFace face detection and landmark alignment
- 🔄 16-frame temporal sequence modeling
- 📊 Mean-based video-level probability aggregation
- 🔓 Progressive backbone unfreezing
- ⚡ Efficient inference suitable for real-time analysis
- 🧪 Robustness evaluation under compression and spatial degradation
- 🔥 Grad-CAM-based explainability
- 📈 Extensive spatial-vs-temporal ablation studies
## 🏗️ Architecture
 
The complete pipeline is:
 
```
                           RAW VIDEO
                               │
                               ▼
                    ┌────────────────────┐
                    │  Frame Sampling    │
                    │      5 FPS         │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   InsightFace      │
                    │ Face Detection +   │
                    │ 5-Point Landmarks  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │     ByteTrack      │
                    │ Identity Tracking  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Quality Filtering  │
                    │      Q ≥ 0.40      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Face Alignment     │
                    │ 224 × 224 × 3      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ 16-Frame Clips     │
                    │      T = 16        │
                    └─────────┬──────────┘
                              │
                              ▼
              ┌────────────────────────────────┐
              │        SPATIAL ENCODER         │
              │                                │
              │        ConvNeXt-Tiny           │
              │                                │
              │  Frame-wise Feature Extraction │
              └───────────────┬────────────────┘
                              │
                              ▼
                    Z ∈ R^(B × 16 × 768)
                              │
                              ▼
              ┌────────────────────────────────┐
              │       TEMPORAL ENCODER         │
              │                                │
              │   [CLS] + Positional Encoding  │
              │                                │
              │   2-Layer Transformer Encoder  │
              │        8 Attention Heads       │
              └───────────────┬────────────────┘
                              │
                              ▼
                       768-D [CLS]
                              │
                              ▼
              ┌────────────────────────────────┐
              │      CLASSIFICATION HEAD       │
              │                                │
              │ LayerNorm → Dropout → Linear   │
              │             768 → 1            │
              └───────────────┬────────────────┘
                              │
                              ▼
                  Clip-Level Probability
                              │
                              ▼
                     Mean Aggregation
                              │
                              ▼
                 Video-Level Probability
```
 
## 🔬 Method
 
### 1. Preprocessing & Identity Tracking
 
**Frame Sampling**
 
Input videos are sampled uniformly at **5 frames per second (FPS)**. This provides a consistent temporal representation across videos with different native frame rates.
 
**Face Detection**
 
InsightFace is used to obtain:
- Face bounding boxes
- Five-point facial landmarks
**Identity Tracking**
 
ByteTrack maintains identity consistency across consecutive frames. This is particularly important for videos containing multiple people.
 
Without identity-aware tracking, a temporal sequence could contain:
 
```
Frame 1 → Person A
Frame 2 → Person A
Frame 3 → Person B
Frame 4 → Person A
```
 
The Temporal Transformer could incorrectly interpret this identity change as a temporal manipulation artifact.
 
With ByteTrack:
 
```
Frame 1  → Person A
Frame 2  → Person A
Frame 3  → Person A
...
Frame 16 → Person A
```
 
Thus, temporal attention is computed only over frames belonging to the same physical identity.
 
**Quality Filtering & Alignment**
 
Face detections are filtered using a quality threshold:
 
$$Q \geq 0.40$$
 
The quality score incorporates:
- Blur variance
- Relative face area
The remaining faces are aligned using an affine transformation based on the five facial landmarks and resized to:
 
$$224 \times 224 \times 3$$
 
**Temporal Clip Construction**
 
Identity-consistent tracks are divided into clips containing $T = 16$ frames. The resulting model input is:
 
$$X \in \mathbb{R}^{B \times 16 \times 3 \times 224 \times 224}$$
 
The processed clips are serialized into sharded PyTorch `.pt` payloads to reduce disk I/O and improve GPU utilization during training.
 
### 2. Spatial Feature Extraction
 
The spatial encoder uses a pretrained **ConvNeXt-Tiny** backbone. Each frame is processed independently, producing a 768-dimensional representation for every frame:
 
$$Z \in \mathbb{R}^{B \times 16 \times 768}$$
 
where:
- $B$ = batch size
- $16$ = number of frames
- $768$ = spatial feature dimension
The spatial encoder therefore focuses on frame-level visual evidence such as:
- Texture inconsistencies
- Blending boundaries
- Facial artifacts
- Fine-grained appearance anomalies
Temporal reasoning is deliberately delegated to the Transformer.
 
### 3. Temporal Transformer
 
**Sequence Construction**
 
A learnable classification token is prepended to the spatial sequence:
 
$$z_{CLS} \in \mathbb{R}^{768}$$
 
The sequence becomes $(B, 17, 768)$, where the additional token is the `[CLS]` token.
 
**Positional Encoding**
 
A learnable absolute positional embedding is added:
 
$$P \in \mathbb{R}^{1 \times 17 \times 768}$$
 
This allows the Transformer to preserve chronological information across the 16-frame sequence.
 
**Transformer Configuration**
 
| Component | Configuration |
|---|---|
| Transformer Layers | 2 |
| Attention Heads | 8 |
| Feature Dimension | 768 |
| MLP Ratio | 4 |
| Dropout | 0.1 |
| Sequence Length | 17 |
 
The Transformer learns temporal patterns such as:
- Flickering
- Facial jitter
- Unnatural motion
- Appearance discontinuities
- Temporal texture inconsistencies
- Frame-to-frame anomalies
The final `[CLS]` representation is used as the temporal representation, $z_{CLS} \in \mathbb{R}^{768}$.
 
### 4. Classification
 
The temporal representation is passed through:
 
```
768-D [CLS]
     │
     ▼
LayerNorm
     │
     ▼
Dropout (0.2)
     │
     ▼
Linear (768 → 1)
     │
     ▼
Clip Probability
```
 
The resulting scalar represents the probability that the input clip contains a manipulated facial sequence.
 
## 📊 Video-Level Aggregation
 
A video may generate multiple 16-frame clips. Instead of selecting only the most suspicious clip, the final prediction is computed using mean aggregation:
 
$$P_{video} = \frac{1}{N} \sum_{i=1}^{N} P_{clip}^{(i)}$$
 
where:
- $N$ = number of clips
- $P_{clip}^{(i)}$ = manipulation probability of clip $i$
Mean pooling provides a stable estimate of manipulation confidence across the complete video.
 
## 💡 Why Decouple Spatial and Temporal Modeling?
 
Traditional 3D-CNN architectures jointly learn spatial and temporal features. While effective, this can result in:
- Higher computational cost
- Larger parameter counts
- Increased optimization complexity
- Greater susceptibility to dataset-specific spatial artifacts
Our approach instead performs:
 
```
Frame
  │
  ▼
ConvNeXt-Tiny
  │
  ▼
Spatial Representation
  │
  ├── Frame 1
  ├── Frame 2
  ├── ...
  └── Frame 16
          │
          ▼
Temporal Transformer
          │
          ▼
Temporal Representation
          │
          ▼
Deepfake Probability
```
 
This explicitly separates *what a frame looks like* from *how that appearance changes over time*.
 
## 📈 Ablation Study
 
The Phase 3 ablation evaluates the contribution of explicit temporal modeling.
 
| Model | Macro-AUROC |
|---|---|
| Spatial-only baseline | 0.6293 |
| Spatial + Temporal Transformer | 0.9480 |
 
**Improvement:** The addition of temporal modeling results in an absolute improvement of **+31.87%**, demonstrating that temporal information is highly valuable for generalization to unseen generative models.
 
## 📊 Aggregation Ablation
 
Different video-level aggregation strategies were evaluated.
 
| Aggregation | Macro-AUROC |
|---|---|
| Max Pooling | 0.8936 |
| Median Pooling | 0.9138 |
| Mean Pooling | 0.9480 |
 
**Why Mean?** Max pooling can amplify isolated false positives. Median pooling can suppress subtle manipulation signals. Mean pooling integrates evidence across multiple clips and provides the strongest overall performance.
 
## 🔓 Progressive Unfreezing
 
The hybrid CNN + Transformer architecture is trained using a two-stage strategy. Initializing a randomly weighted Transformer on top of a pretrained CNN can lead to catastrophic forgetting, as large early gradients would otherwise destroy the CNN's spatial representations.
 
**Stage 1 — Temporal Adaptation (Epochs 1–5)**
 
```
ConvNeXt-Tiny        → Frozen
Temporal Transformer  → Trainable
Classification Head   → Trainable
```
 
The Transformer learns to interpret the pretrained spatial feature space without immediately modifying the CNN representation.
 
**Stage 2 — End-to-End Fine-Tuning (Epochs 6–25)**
 
```
ConvNeXt-Tiny        → Unfrozen
Temporal Transformer  → Trainable
Classification Head   → Trainable
```
 
Differential learning rates are used:
 
| Module | Learning Rate |
|---|---|
| ConvNeXt-Tiny | $1 \times 10^{-5}$ |
| Temporal Transformer | $1 \times 10^{-4}$ |
| Classification Head | $5 \times 10^{-4}$ |
 
The smaller backbone learning rate helps preserve useful pretrained spatial representations while allowing task-specific adaptation.
 
## ⚡ Computational Efficiency
 
The model was profiled on a single **NVIDIA GeForce RTX 3080 Ti** GPU, with batch size 1 and temporal length 16.
 
| Metric | Value |
|---|---|
| Total Parameters | 42.01 M |
| Spatial Backbone | 27.82 M |
| Temporal Transformer | 14.19 M |
| GFLOPs / 16-frame clip | 71.44 |
| GFLOPs / frame | 4.46 |
| Clip Latency | 15.45 ms |
| Effective Throughput | 1035.46 FPS |
 
The decoupled architecture provides temporal modeling without the computational overhead of a large 3D-CNN.
 
## 🧪 Robustness Evaluation
 
The model was evaluated under common video degradation scenarios.
 
**JPEG Compression**
 
Under heavy JPEG compression ($QF = 30$), the model achieves Macro-AUROC $= 0.9256$, indicating strong resilience to compression artifacts.
 
**Spatial Degradation**
 
The model is more sensitive to spatial smoothing and aggressive resolution reduction.
 
| Perturbation | Macro-AUROC |
|---|---|
| Original | 0.9480 |
| JPEG QF 30 | 0.9256 |
| $3 \times 3$ Gaussian Blur | 0.7022 |
| $112 \times 112$ Downscaling | 0.6168 |
 
These results indicate that the model benefits from high-frequency spatial information such as:
- Blending boundaries
- Fine-grained texture artifacts
- Facial manipulation boundaries
- Local appearance inconsistencies
Severe spatial degradation removes part of this evidence before it reaches the temporal module, starving the Temporal Transformer of the embeddings needed to detect sequence-level anomalies.
 
## 🔥 Explainability
 
The repository includes a Grad-CAM-based explainability pipeline that generates spatial heatmaps highlighting regions contributing to the model's predictions.
 
Potentially informative regions include:
- Facial boundaries
- Eyes
- Mouth
- Skin texture
- Blending regions
- High-frequency facial artifacts
Run:
 
```bash
python3 visualize_explainability.py \
    --checkpoint ./checkpoints_temporal/best_checkpoint.pth \
    --test-dir ./data/shards/videos/test
```
 
## 🛠️ Installation
 
Clone the repository:
 
```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```
 
Install dependencies:
 
```bash
pip install -r requirements.txt
```
 
Replace `<YOUR_REPOSITORY_URL>` and `<YOUR_REPOSITORY_NAME>` with your repository information.
 
## ▶️ Usage
 
### Phase 3 — Spatial vs. Temporal Ablation
 
To reproduce the zero-leakage, one-pass spatial-vs-temporal evaluation on the test set:
 
```bash
python3 ablation_phase3.py \
    --checkpoint ./checkpoints_temporal/best_checkpoint.pth \
    --data-path ./data/shards/videos/test \
    --frames 16
```
 
### Phase 4.2 — Robustness Testing
 
Evaluate performance under Gaussian blur, downscaling, and H.264/JPEG compression:
 
```bash
python3 perturbation_test.py \
    --checkpoint ./checkpoints_temporal/best_checkpoint.pth
```
 
### Phase 4.3 — Explainability
 
Generate Grad-CAM visualizations:
 
```bash
python3 visualize_explainability.py \
    --checkpoint ./checkpoints_temporal/best_checkpoint.pth
```
 
## 📂 Dataset Pipeline
 
The complete preprocessing pipeline is:
 
```
Raw Videos
    │
    ▼
5 FPS Sampling
    │
    ▼
InsightFace Detection
    │
    ▼
5-Point Landmark Extraction
    │
    ▼
ByteTrack Identity Association
    │
    ▼
Quality Filtering (Q ≥ 0.40)
    │
    ▼
Affine Alignment
    │
    ▼
224 × 224 Face Crops
    │
    ▼
16-Frame Temporal Chunking
    │
    ▼
Sharded PyTorch .pt Dataset
```
 
This preprocessing strategy ensures identity consistency and minimizes expensive preprocessing during model training.
 
## ⚙️ Model Configuration
 
| Component | Configuration |
|---|---|
| Input Sampling | 5 FPS |
| Face Detector | InsightFace |
| Identity Tracker | ByteTrack |
| Quality Threshold | $Q \geq 0.40$ |
| Face Resolution | $224 \times 224$ |
| Temporal Length | 16 frames |
| Spatial Backbone | ConvNeXt-Tiny |
| Spatial Feature Size | 768 |
| Temporal Encoder | Transformer |
| Transformer Layers | 2 |
| Attention Heads | 8 |
| MLP Ratio | 4 |
| Transformer Dropout | 0.1 |
| Classification Dropout | 0.2 |
| Classification Layer | 768 → 1 |
| Video Aggregation | Mean |
 
## 📁 Repository Structure
 
```
.
├── README.md
├── requirements.txt
│
├── ablation_phase3.py
├── perturbation_test.py
├── visualize_explainability.py
│
├── checkpoints_temporal/
│   └── best_checkpoint.pth
│
├── data/
│   └── shards/
│       └── videos/
│           └── test/
│
└── ...
```
 
### Recommended additions
 
For a complete research repository, the following structure can also be used:
 
```
.
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
├── models/
├── datasets/
├── preprocessing/
├── utils/
├── scripts/
│   ├── ablation_phase3.py
│   ├── perturbation_test.py
│   └── visualize_explainability.py
├── checkpoints/
├── data/
└── results/
```
 
## 🔄 Inference Pipeline
 
For a complete video, inference follows:
 
```
Video
 │
 ├── Sample at 5 FPS
 │
 ├── Detect faces
 │
 ├── Track identities
 │
 ├── Filter low-quality faces
 │
 ├── Align faces
 │
 ├── Create 16-frame clips
 │
 ├── ConvNeXt-Tiny
 │
 ├── Temporal Transformer
 │
 ├── Clip probability
 │
 └── Mean aggregation
          │
          ▼
   VIDEO PROBABILITY
```
 
Mathematically:
 
$$P_{video} = \frac{1}{N} \sum_{i=1}^{N} P_{clip}^{(i)}$$
 
where $P_{video}$ is the final manipulation probability.
 
## 📊 Summary of Results
 
**Main Ablation**
 
| Experiment | Macro-AUROC |
|---|---|
| Spatial-only | 0.6293 |
| Spatial + Temporal | 0.9480 |
| Absolute improvement | +31.87% |
 
**Aggregation**
 
| Method | Macro-AUROC |
|---|---|
| Max | 0.8936 |
| Median | 0.9138 |
| Mean | 0.9480 |
 
**Robustness**
 
| Condition | Macro-AUROC |
|---|---|
| Original | 0.9480 |
| JPEG QF 30 | 0.9256 |
| Gaussian Blur $3 \times 3$ | 0.7022 |
| Downscaled $112 \times 112$ | 0.6168 |
 
**Efficiency**
 
| Metric | Result |
|---|---|
| Parameters | 42.01 M |
| GFLOPs / clip | 71.44 |
| GFLOPs / frame | 4.46 |
| Clip latency | 15.45 ms |
| Effective throughput | 1035.46 FPS |
 
## 📚 Citation
 
If you use this work in your research, please cite:
 
```bibtex
@article{yourname2026spatiotemporal,
  title   = {Spatiotemporal Deepfake Detection via Decoupled Spatial-Temporal Transformers},
  author  = {Your Name and Coauthors},
  journal = {Your Journal or Conference},
  year    = {2026}
}
```
 
Replace the placeholder citation with the final bibliographic information of your paper.
 
## 🙏 Acknowledgements
 
This project builds upon several open-source technologies and research contributions, including:
 
- PyTorch
- ConvNeXt
- InsightFace
- ByteTrack
- Transformer architectures
- Grad-CAM
We thank the authors and maintainers of these projects for making their work publicly available.
 
## ⚖️ License
 
This project is released under the license specified in `LICENSE`.
 
If you have not selected a license yet, we recommend adding one before publicly distributing the repository.
 
## 📌 Notes
 
Large datasets and model checkpoints should generally not be committed directly to Git when they exceed GitHub's file-size limits.
 
Consider using:
- Git LFS
- Hugging Face Hub
- Cloud storage
- An appropriate research artifact repository
for large datasets and trained model weights.
 
## 🌟 Final Takeaway
 
The central idea of this work is simple:
 
> Extract spatial evidence independently, then explicitly model how that evidence evolves over time.
 
By combining **ConvNeXt-Tiny + Temporal Transformer + ByteTrack + Identity-Aware Preprocessing + Progressive Unfreezing**, the proposed architecture captures both spatial manipulation artifacts and temporal inconsistencies while maintaining a relatively lightweight computational footprint.
 
The experimental results show that explicit temporal modeling substantially improves deepfake detection performance over a spatial-only baseline, achieving a Macro-AUROC of **0.9480** while remaining suitable for efficient video analysis.
