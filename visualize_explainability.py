#!/usr/bin/env python3
"""
Phase 4.3 — Explainability & Visualization
Dynamic Grad-CAM + Temporal Attention

This version:
    - Automatically finds a valid spatial feature map
    - Avoids 1x1 classification/head features
    - Supports NCHW and NHWC feature layouts
    - Produces one Grad-CAM heatmap per temporal frame
    - Includes shape validation and diagnostic output
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from models.spatial_backbone import ConvNeXtSpatialBackbone
from models.temporal_transformer import TemporalTransformer
from models.temporal_model import SpatiotemporalDeepfakeModel


# ============================================================
# Dynamic Grad-CAM
# ============================================================

class DynamicGradCAM:
    """
    Robust Grad-CAM implementation for ConvNeXt-style
    spatial backbones.

    Automatically searches for the deepest valid layer
    producing a genuine spatial feature map.

    Valid target:
        [B, C, H, W] where H > 1 and W > 1

    Also supports:
        [B, H, W, C]
    """

    def __init__(self, model):

        self.model = model

        self.activations = {}
        self.gradients = {}
        self.hooks = []

        print("\n[Grad-CAM] Registering hooks...")

        # ----------------------------------------------------
        # Register hooks throughout the spatial backbone
        # ----------------------------------------------------

        for name, module in self.model.backbone.named_modules():

            if isinstance(
                module,
                (
                    torch.nn.Conv2d,
                    torch.nn.BatchNorm2d,
                    torch.nn.LayerNorm,
                    torch.nn.GELU,
                ),
            ):

                self.hooks.append(
                    module.register_forward_hook(
                        self.get_act_hook(name)
                    )
                )

                self.hooks.append(
                    module.register_full_backward_hook(
                        self.get_grad_hook(name)
                    )
                )

        print(
            f"[Grad-CAM] Registered {len(self.hooks)} hooks."
        )

    # ========================================================
    # Forward activation hook
    # ========================================================

    def get_act_hook(self, name):

        def hook(module, inputs, output):

            if torch.is_tensor(output):

                self.activations[name] = output

        return hook

    # ========================================================
    # Backward gradient hook
    # ========================================================

    def get_grad_hook(self, name):

        def hook(module, grad_input, grad_output):

            if (
                grad_output is not None
                and len(grad_output) > 0
                and grad_output[0] is not None
                and torch.is_tensor(grad_output[0])
            ):

                self.gradients[name] = grad_output[0]

        return hook

    # ========================================================
    # Remove hooks
    # ========================================================

    def remove_hooks(self):

        for hook in self.hooks:
            hook.remove()

        self.hooks = []

    # ========================================================
    # Find valid target layer
    # ========================================================

    def _find_target_layer(self):

        candidates = []

        for name, activation in self.activations.items():

            # Need both activation and gradient
            if name not in self.gradients:
                continue

            if not torch.is_tensor(activation):
                continue

            if activation.ndim != 4:
                continue

            shape = activation.shape

            # ------------------------------------------------
            # NCHW:
            #
            # [Batch, Channels, Height, Width]
            # ------------------------------------------------

            if shape[2] > 1 and shape[3] > 1:

                spatial_area = (
                    shape[2] * shape[3]
                )

                candidates.append(
                    {
                        "name": name,
                        "activation": activation,
                        "gradient": self.gradients[name],
                        "area": spatial_area,
                        "layout": "NCHW",
                    }
                )

            # ------------------------------------------------
            # NHWC:
            #
            # [Batch, Height, Width, Channels]
            # ------------------------------------------------

            elif shape[1] > 1 and shape[2] > 1:

                spatial_area = (
                    shape[1] * shape[2]
                )

                candidates.append(
                    {
                        "name": name,
                        "activation": activation,
                        "gradient": self.gradients[name],
                        "area": spatial_area,
                        "layout": "NHWC",
                    }
                )

        if not candidates:
            return None

        # ----------------------------------------------------
        # The activation dictionary preserves forward order.
        #
        # Therefore the final candidate is normally the
        # deepest valid spatial feature map.
        # ----------------------------------------------------

        return candidates[-1]

    # ========================================================
    # Generate Grad-CAM
    # ========================================================

    def generate_cam(self, input_tensor):

        print("\n[Grad-CAM] Starting forward pass...")

        self.model.zero_grad(set_to_none=True)

        self.activations = {}
        self.gradients = {}

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        features = self.model.backbone.extract_features(
            input_tensor
        )

        print(
            f"[Grad-CAM] Backbone feature shape: "
            f"{tuple(features.shape)}"
        )

        temporal_cls = self.model.temporal_transformer(
            features
        )

        print(
            f"[Grad-CAM] Temporal output shape: "
            f"{tuple(temporal_cls.shape)}"
        )

        logits = self.model.head(
            temporal_cls
        ).squeeze(-1)

        print(
            f"[Grad-CAM] Logit shape: "
            f"{tuple(logits.shape)}"
        )

        prob = torch.sigmoid(logits)

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        print("[Grad-CAM] Running backward pass...")

        logits.sum().backward()

        # ----------------------------------------------------
        # Find spatial feature map
        # ----------------------------------------------------

        target = self._find_target_layer()

        if target is None:

            print(
                "\n[Grad-CAM] ERROR: No valid spatial feature "
                "map was found."
            )

            print(
                "\nAvailable activation shapes:"
            )

            for name, activation in self.activations.items():

                if torch.is_tensor(activation):

                    print(
                        f"    {name}: "
                        f"{tuple(activation.shape)}"
                    )

            self.remove_hooks()

            raise ValueError(
                "Could not find a valid spatial feature "
                "map for Grad-CAM."
            )

        # ----------------------------------------------------
        # Extract target information
        # ----------------------------------------------------

        target_name = target["name"]
        acts = target["activation"]
        grads = target["gradient"]
        layout = target["layout"]

        print(
            f"\n[Grad-CAM] Successfully mapped features "
            f"from layer: {target_name}"
        )

        print(
            f"[Grad-CAM] Activation shape: "
            f"{tuple(acts.shape)}"
        )

        print(
            f"[Grad-CAM] Gradient shape: "
            f"{tuple(grads.shape)}"
        )

        print(
            f"[Grad-CAM] Feature layout: {layout}"
        )

        # ----------------------------------------------------
        # Convert NHWC → NCHW if necessary
        # ----------------------------------------------------

        if layout == "NHWC":

            acts = acts.permute(
                0,
                3,
                1,
                2
            )

            grads = grads.permute(
                0,
                3,
                1,
                2
            )

        print(
            f"[Grad-CAM] Converted activation shape: "
            f"{tuple(acts.shape)}"
        )

        # ----------------------------------------------------
        # Validate spatial dimensions
        # ----------------------------------------------------

        if acts.ndim != 4:

            self.remove_hooks()

            raise ValueError(
                "Grad-CAM activation is not 4-dimensional: "
                f"{tuple(acts.shape)}"
            )

        batch_size, channels, height, width = acts.shape

        if height <= 1 or width <= 1:

            self.remove_hooks()

            raise ValueError(
                "Selected feature map does not contain "
                f"meaningful spatial information: "
                f"{height}x{width}"
            )

        # ----------------------------------------------------
        # Gradient-weighted activation
        # ----------------------------------------------------

        # Global average pooling over spatial dimensions
        pooled_grads = grads.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted feature maps
        cam = (
            acts * pooled_grads
        ).sum(
            dim=1,
            keepdim=True
        )

        # Keep only positive evidence
        cam = F.relu(cam)

        # ----------------------------------------------------
        # Normalize each temporal frame independently
        # ----------------------------------------------------

        cam_min = cam.amin(
            dim=(2, 3),
            keepdim=True
        )

        cam_max = cam.amax(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        # ----------------------------------------------------
        # Remove channel dimension
        #
        # [B, 1, H, W]
        #
        # →
        #
        # [B, H, W]
        # ----------------------------------------------------

        cam = cam[:, 0, :, :]

        cam_masks = (
            cam.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = prob.mean().item()

        print(
            f"[Grad-CAM] Final CAM shape: "
            f"{cam_masks.shape}"
        )

        print(
            f"[Grad-CAM] Fake probability: "
            f"{probability:.6f}"
        )

        self.remove_hooks()

        return cam_masks, probability


# ============================================================
# Load sample clip
# ============================================================

def load_sample_clip(test_dir):

    catalog_path = os.path.join(
        test_dir,
        "catalog.csv"
    )

    if not os.path.exists(catalog_path):

        raise FileNotFoundError(
            f"Catalog not found:\n{catalog_path}"
        )

    print(
        f"[Dataset] Reading catalog: "
        f"{catalog_path}"
    )

    catalog = pd.read_csv(
        catalog_path
    )

    if len(catalog) == 0:

        raise ValueError(
            "catalog.csv is empty."
        )

    # --------------------------------------------------------
    # Use first shard
    # --------------------------------------------------------

    shard_file = catalog[
        "shard_file"
    ].iloc[0]

    shard_path = os.path.join(
        test_dir,
        shard_file
    )

    if not os.path.exists(shard_path):

        raise FileNotFoundError(
            f"Shard not found:\n{shard_path}"
        )

    print(
        f"[Dataset] Loading shard: "
        f"{shard_path}"
    )

    shard = torch.load(
        shard_path,
        map_location="cpu",
        weights_only=False
    )

    # --------------------------------------------------------
    # Search for fake sample
    # label = 1
    # --------------------------------------------------------

    for i, label in enumerate(
        shard["labels"]
    ):

        label_value = int(
            label.item()
            if torch.is_tensor(label)
            else label
        )

        if label_value == 1:

            tensor = (
                shard["tensors"][i]
                .unsqueeze(0)
            )

            video_id = (
                shard["video_ids"][i]
            )

            print(
                f"[Dataset] Selected FAKE sample: "
                f"{video_id}"
            )

            print(
                f"[Dataset] Tensor shape: "
                f"{tuple(tensor.shape)}"
            )

            return tensor, video_id

    # --------------------------------------------------------
    # If no fake sample exists, use first sample
    # --------------------------------------------------------

    print(
        "[Dataset] WARNING: No fake sample found."
    )

    tensor = (
        shard["tensors"][0]
        .unsqueeze(0)
    )

    video_id = (
        shard["video_ids"][0]
    )

    print(
        f"[Dataset] Using first sample: "
        f"{video_id}"
    )

    return tensor, video_id


# ============================================================
# Visualization
# ============================================================

def visualize_and_save(
    original_tensor,
    cam_masks,
    prob,
    video_id,
    output_dir
):

    # --------------------------------------------------------
    # Convert tensor:
    #
    # [1, T, C, H, W]
    #
    # →
    #
    # [T, H, W, C]
    # --------------------------------------------------------

    frames = (
        original_tensor
        .squeeze(0)
        .permute(
            0,
            2,
            3,
            1
        )
        .detach()
        .cpu()
        .numpy()
    )

    print(
        f"\n[Visualization] Frames shape: "
        f"{frames.shape}"
    )

    print(
        f"[Visualization] CAM shape: "
        f"{cam_masks.shape}"
    )

    # --------------------------------------------------------
    # Validate CAM
    # --------------------------------------------------------

    if not isinstance(
        cam_masks,
        np.ndarray
    ):

        cam_masks = np.asarray(
            cam_masks
        )

    if cam_masks.ndim != 3:

        raise ValueError(
            "Expected CAM shape [T,H,W], "
            f"but received {cam_masks.shape}"
        )

    # --------------------------------------------------------
    # Validate temporal dimensions
    # --------------------------------------------------------

    if len(cam_masks) != len(frames):

        raise ValueError(
            f"Number of CAMs ({len(cam_masks)}) "
            f"does not match number of frames "
            f"({len(frames)})."
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create 4x4 figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(16, 16)
    )

    fig.suptitle(
        "Spatiotemporal Grad-CAM Heatmaps\n"
        f"Video ID: {video_id} "
        f"(Fake Probability: "
        f"{prob * 100:.2f}%)",
        fontsize=16
    )

    # --------------------------------------------------------
    # Process each frame
    # --------------------------------------------------------

    for i, ax in enumerate(
        axes.flat
    ):

        if i >= len(frames):

            ax.axis("off")
            continue

        # ----------------------------------------------------
        # Original frame
        # ----------------------------------------------------

        frame = frames[i]

        # Handle normalized images
        frame = np.clip(
            frame,
            0.0,
            1.0
        )

        frame = (
            frame * 255.0
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # CAM
        # ----------------------------------------------------

        cam = cam_masks[i]

        cam = np.asarray(
            cam,
            dtype=np.float32
        )

        cam = np.nan_to_num(
            cam,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        cam = np.clip(
            cam,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Resize CAM to original frame dimensions
        # ----------------------------------------------------

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        heatmap = cv2.resize(
            cam,
            (
                frame_width,
                frame_height
            ),
            interpolation=cv2.INTER_LINEAR
        )

        heatmap = (
            heatmap * 255.0
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Apply color map
        # ----------------------------------------------------

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        # ----------------------------------------------------
        # Convert RGB → BGR
        # ----------------------------------------------------

        frame_bgr = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        superimposed = cv2.addWeighted(
            frame_bgr,
            0.60,
            heatmap,
            0.40,
            0
        )

        # ----------------------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------------------

        superimposed = cv2.cvtColor(
            superimposed,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        ax.imshow(
            superimposed
        )

        ax.set_title(
            f"Sequence Frame {i + 1}"
        )

        ax.axis("off")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        f"gradcam_{video_id}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"\n[Output] Visualization saved to:"
    )

    print(
        f"         {output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 4.3 — Dynamic Grad-CAM "
            "Explainability Visualization"
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "./checkpoints_temporal/"
            "best_checkpoint.pth"
        ),
        help="Path to trained model checkpoint."
    )

    parser.add_argument(
        "--test-dir",
        default=(
            "/home/aashish/deepfake_detection/"
            "data/shards/videos/test"
        ),
        help="Directory containing catalog.csv and shards."
    )

    parser.add_argument(
        "--output-dir",
        default="./ablation_results",
        help="Directory for Grad-CAM output."
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print(
        "PHASE 4.3 — EXPLAINABILITY "
        "& GRAD-CAM VISUALIZATION"
    )
    print("=" * 80)

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: "
            f"{torch.version.cuda}"
        )

    # ========================================================
    # Build model
    # ========================================================

    print("\n[Model] Building model...")

    backbone = ConvNeXtSpatialBackbone(
        pretrained=False,
        num_classes=1
    )

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

    # ========================================================
    # Load checkpoint
    # ========================================================

    print(
        f"[Model] Loading checkpoint:"
    )

    print(
        f"        {args.checkpoint}"
    )

    if not os.path.exists(
        args.checkpoint
    ):

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{args.checkpoint}"
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True
    )

    # --------------------------------------------------------
    # Handle different checkpoint formats
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    load_result = model.load_state_dict(
        state_dict,
        strict=False
    )

    print(
        f"[Model] Missing keys: "
        f"{len(load_result.missing_keys)}"
    )

    print(
        f"[Model] Unexpected keys: "
        f"{len(load_result.unexpected_keys)}"
    )

    model.to(device)

    model.eval()

    print(
        "[Model] Model loaded successfully."
    )

    # ========================================================
    # Initialize Grad-CAM
    # ========================================================

    grad_cam = DynamicGradCAM(
        model
    )

    # ========================================================
    # Load sample
    # ========================================================

    print(
        f"\n[Dataset] Loading sample from:"
    )

    print(
        f"          {args.test_dir}"
    )

    tensor, video_id = load_sample_clip(
        args.test_dir
    )

    tensor = (
        tensor
        .to(device)
        .float()
    )

    # ========================================================
    # Grad-CAM
    # ========================================================

    print(
        f"\n[Grad-CAM] Processing Video ID: "
        f"{video_id}"
    )

    cam_masks, prob = grad_cam.generate_cam(
        tensor
    )

    # ========================================================
    # Visualization
    # ========================================================

    visualize_and_save(
        tensor,
        cam_masks,
        prob,
        video_id,
        args.output_dir
    )

    print("\n" + "=" * 80)
    print(
        "PHASE 4.3 COMPLETED SUCCESSFULLY"
    )
    print("=" * 80)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()