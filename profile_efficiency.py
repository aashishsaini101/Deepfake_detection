#!/usr/bin/env python3
"""
Phase 4.1 — Computational Efficiency & Latency Profiler

Measures total parameters, estimated GFLOPs, inference latency (ms),
and throughput (FPS) for the Spatiotemporal Deepfake Detection Model.
"""

import time
import torch
import torch.nn as nn
from models.spatial_backbone import ConvNeXtSpatialBackbone
from models.temporal_transformer import TemporalTransformer
from models.temporal_model import SpatiotemporalDeepfakeModel

def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def profile_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("PHASE 4.1 — COMPUTATIONAL EFFICIENCY & HARDWARE PROFILING")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Initialize model components
    backbone = ConvNeXtSpatialBackbone(pretrained=False, num_classes=1)
    transformer = TemporalTransformer(
        num_frames=16, embed_dim=768, depth=2, num_heads=8, mlp_ratio=4, dropout=0.1
    )
    model = SpatiotemporalDeepfakeModel(backbone, transformer, embed_dim=768, num_classes=1)
    model.to(device)
    model.eval()

    # 1. Parameter Breakdown
    backbone_total, _ = count_parameters(model.backbone)
    transformer_total, _ = count_parameters(model.temporal_transformer)
    head_total, _ = count_parameters(model.head)
    total_params, _ = count_parameters(model)

    print("\n--- PARAMETER BREAKDOWN ---")
    print(f"Spatial Backbone (ConvNeXt) : {backbone_total / 1e6:.2f} M")
    print(f"Temporal Transformer        : {transformer_total / 1e6:.2f} M")
    print(f"Classification Head         : {head_total / 1e6:.2f} M")
    print(f"TOTAL MODEL PARAMETERS      : {total_params / 1e6:.2f} M")

    # 2. FLOPs Estimation via fvcore or thop if available
    print("\n--- COMPUTATIONAL COMPLEXITY ---")
    dummy_input = torch.randn(1, 16, 3, 224, 224, device=device)
    
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"GFLOPs per 16-frame clip    : {flops / 1e9:.2f} GFLOPs")
        print(f"GFLOPs per single frame     : {(flops / 1e9) / 16:.2f} GFLOPs/frame")
    except ImportError:
        print("Note: Install 'thop' (pip install thop) to print exact GFLOP counts.")

    # 3. Latency and Throughput (FPS) Benchmarking
    print("\n--- LATENCY & THROUGHPUT BENCHMARK (Batch Size = 1) ---")
    num_warmup = 20
    num_runs = 100

    # GPU Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Timing Execution
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    with torch.inference_mode():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))  # in ms
            else:
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

    mean_latency_clip = float(torch.tensor(latencies).mean().item())
    fps = (16.0 / mean_latency_clip) * 1000.0

    print(f"Clip Latency (16 frames)    : {mean_latency_clip:.2f} ms")
    print(f"Per-Frame Latency           : {mean_latency_clip / 16.0:.2f} ms")
    print(f"Inference Throughput        : {fps:.2f} FPS")
    print("=" * 80)

if __name__ == "__main__":
    profile_model()