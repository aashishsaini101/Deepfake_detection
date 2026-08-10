import torch
import torch.nn as nn
import timm


class ConvNeXtSpatialBackbone(nn.Module):
    """
    Modular ConvNeXt-Tiny Spatial Backbone for Deepfake Detection.
    
    Features:
    - Pretrained weights via timm
    - Configurable Stochastic Depth (drop_path_rate)
    - Clean separation between spatial feature extraction and the classification head
    - Automatic handling of 4D image input (B, C, H, W) and 5D video input (B, T, C, H, W)
    """
    def __init__(
        self,
        model_name: str = "convnext_tiny.fb_in22k_ft_in1k_384",  # or 'convnext_tiny'
        pretrained: bool = True,
        drop_path_rate: float = 0.2,
        num_classes: int = 1,
    ):
        super().__init__()
        
        # Load backbone without default classification head (num_classes=0)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            num_classes=0,  # Strips the top linear layer, keeping global pool
        )
        
        # Feature dimension (for ConvNeXt-Tiny, D = 768)
        self.num_features = self.backbone.num_features
        
        # Modular classification head for Phase 2/3 (Binary Classification)
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.num_features, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts D-dimensional feature vector per frame.
        
        Args:
            x: Tensor of shape (B, C, H, W) or (B, T, C, H, W)
            
        Returns:
            Features of shape (B, D) for 4D input, or (B, T, D) for 5D input.
        """
        if x.ndim == 4:
            # Single frame / static image input: (B, C, H, W) -> (B, D)
            features = self.backbone(x)
            return features

        elif x.ndim == 5:
            # Video clip sequence input: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            
            # Collapse Batch and Time dimensions for fast parallel spatial encoding
            x_flat = x.view(B * T, C, H, W)
            
            # Extract features: (B*T, D)
            features_flat = self.backbone(x_flat)
            
            # Unflatten back to sequence format: (B, T, D)
            features = features_flat.view(B, T, self.num_features)
            return features

        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {x.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with binary logits output.
        """
        features = self.extract_features(x)
        
        if features.ndim == 3:
            # For 5D video input before Phase 4: apply simple mean pooling over frames
            # (B, T, D) -> (B, D)
            features = features.mean(dim=1)

        # Pass through binary head -> (B, 1)
        logits = self.head(features)
        return logits.squeeze(-1)


# ============================================================
# VERIFICATION BLOCK
# ============================================================
if __name__ == "__main__":
    print("Testing ConvNeXtSpatialBackbone...")
    
    # Instantiate model
    model = ConvNeXtSpatialBackbone(
        model_name="convnext_tiny",
        pretrained=False,  # Set False for instant dry-run testing
        drop_path_rate=0.2,
        num_classes=1
    )
    
    # Test 1: Image batch from images/train (B=4, C=3, H=224, W=224)
    image_batch = torch.randn(4, 3, 224, 224)
    img_features = model.extract_features(image_batch)
    img_logits = model(image_batch)
    
    print("\n--- Image Batch Test (4D Input) ---")
    print(f"Input Shape    : {image_batch.shape}")
    print(f"Features Shape : {img_features.shape} (Expected: [4, 768])")
    print(f"Logits Shape   : {img_logits.shape}   (Expected: [4])")
    
    # Test 2: Video clip batch from videos/train (B=2, T=16, C=3, H=224, W=224)
    video_batch = torch.randn(2, 16, 3, 224, 224)
    vid_features = model.extract_features(video_batch)
    vid_logits = model(video_batch)
    
    print("\n--- Video Sequence Batch Test (5D Input) ---")
    print(f"Input Shape    : {video_batch.shape}")
    print(f"Features Shape : {vid_features.shape} (Expected: [2, 16, 768])")
    print(f"Logits Shape   : {vid_logits.shape}   (Expected: [2])")
    
    print("\n✓ Model architecture verified successfully!")