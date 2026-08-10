import torch
import torch.nn as nn

from models.spatial_backbone import ConvNeXtSpatialBackbone
from models.temporal_transformer import TemporalTransformer


class SpatiotemporalDeepfakeModel(nn.Module):
    """
    End-to-end model combining:

        ConvNeXt Spatial Backbone
                    +
        Temporal Transformer
                    +
        Binary Classification Head
    """

    def __init__(
        self,
        spatial_backbone: nn.Module,
        temporal_transformer: nn.Module,
        embed_dim: int = 768,
        num_classes: int = 1,
    ):
        super().__init__()

        self.backbone = spatial_backbone
        self.temporal_transformer = temporal_transformer

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes),
        )

    # -------------------------------------------------------
    # Load pretrained spatial checkpoint
    # -------------------------------------------------------
    def load_spatial_weights(self, checkpoint_path: str):
        """
        Load pretrained ConvNeXt weights.

        The old spatial classifier head is ignored automatically.
        """

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)

        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            state_dict,
            strict=False
        )

        print(f"\nLoaded spatial backbone weights from:")
        print(f"  {checkpoint_path}")

        if missing_keys:
            print("\nMissing keys:")
            for k in missing_keys:
                print(f"  {k}")

        if unexpected_keys:
            print("\nUnexpected keys:")
            for k in unexpected_keys:
                print(f"  {k}")

    # -------------------------------------------------------
    # Freeze / Unfreeze backbone
    # -------------------------------------------------------
    def set_backbone_requires_grad(self, requires_grad: bool):
        """
        Freeze or unfreeze the ConvNeXt backbone.
        """

        for param in self.backbone.parameters():
            param.requires_grad = requires_grad

    # -------------------------------------------------------
    # Feature extraction
    # -------------------------------------------------------
    def forward_features(self, x: torch.Tensor):
        """
        Returns temporal feature representation before classification.

        Input:
            (B,T,C,H,W)

        Output:
            (B,768)
        """

        spatial_features = self.backbone.extract_features(x)

        temporal_features = self.temporal_transformer(
            spatial_features
        )

        return temporal_features

    # -------------------------------------------------------
    # Forward
    # -------------------------------------------------------
    def forward(self, x: torch.Tensor):

        features = self.forward_features(x)

        logits = self.head(features)

        return logits.squeeze(-1)


# ============================================================
# VERIFICATION
# ============================================================

if __name__ == "__main__":

    print("Testing SpatiotemporalDeepfakeModel...")

    backbone = ConvNeXtSpatialBackbone(
        pretrained=False
    )

    transformer = TemporalTransformer(
        num_frames=16,
        embed_dim=768,
        depth=2,
        num_heads=8,
    )

    model = SpatiotemporalDeepfakeModel(
        backbone,
        transformer,
    )

    dummy = torch.randn(
        2,
        16,
        3,
        224,
        224
    )

    features = model.forward_features(dummy)
    logits = model(dummy)

    print(f"\nInput Shape    : {dummy.shape}")
    print(f"Feature Shape  : {features.shape} (Expected: [2,768])")
    print(f"Logits Shape   : {logits.shape}   (Expected: [2])")

    print("\n✓ Spatiotemporal architecture verified successfully!")