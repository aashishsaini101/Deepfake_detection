import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    """
    Temporal reasoning module that processes sequence of spatial features.
    Architecture aligns with TimeSformer/ViT temporal encoding.
    """
    def __init__(
        self,
        num_frames: int = 16,
        embed_dim: int = 768,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 1. Class Token & Positional Embeddings
        # +1 accounts for the CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames + 1, embed_dim))
        
        # Standard ViT initialization for tokens
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.pos_drop = nn.Dropout(p=dropout)
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Standard for ViT-style architectures
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatial features of shape (B, T, D) -> (B, 16, 768)
        Returns:
            CLS token feature of shape (B, D) -> (B, 768)
        """
        B, T, D = x.shape
        
        # Expand CLS token for the batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        
        # Concatenate CLS token to the frame features
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 17, 768)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Pass through Transformer
        x = self.transformer(x)
        
        # Extract the output corresponding to the CLS token (the first token)
        cls_output = x[:, 0, :]  # (B, 768)
        
        return cls_output