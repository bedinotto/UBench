"""
Thermal Facial Region Detection System - TransUNet
==================================================
CNN-Transformer hybrid architecture for thermal face segmentation.
Defines the TransUNet model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block for transformer"""

    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ResNetBlock(nn.Module):
    """ResNet bottleneck block"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CNNEncoder(nn.Module):
    """CNN encoder for feature extraction (ResNet-50 style)"""

    def __init__(self, in_channels=1):
        super().__init__()

        # Initial convolution - modified for single channel input
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet stages
        self.layer1 = self._make_layer(64, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = x  # 64 channels, H/2, W/2

        x = self.maxpool(x)
        x1 = self.layer1(x)  # 256 channels, H/4, W/4
        x2 = self.layer2(x1)  # 512 channels, H/8, W/8
        x3 = self.layer3(x2)  # 1024 channels, H/16, W/16

        return x0, x1, x2, x3


from codes.model_registry import register_model

@register_model("transunet")
class TransUNet(nn.Module):
    """TransUNet: Transformer-CNN Hybrid Architecture"""

    def __init__(self, img_size=256, in_channels=1, num_classes=10,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = 16

        # CNN Encoder
        self.cnn_encoder = CNNEncoder(in_channels)

        # Patch embedding from CNN features
        # Input from CNN: 1024 channels at 16x16 (H/16, W/16)
        self.patch_embed = nn.Conv2d(1024, embed_dim, kernel_size=1)

        # Positional embedding
        num_patches = (img_size // self.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer Encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Decoder - Cascaded Upsampler with skip connections
        self.decoder3 = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.decoder0 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)

        # Final output
        self.final = nn.Conv2d(64, num_classes, 1)

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape

        # CNN Encoder with skip connections
        x0, x1, x2, x3 = self.cnn_encoder(x)
        # x0: 64 channels, H/2, W/2
        # x1: 256 channels, H/4, W/4
        # x2: 512 channels, H/8, W/8
        # x3: 1024 channels, H/16, W/16

        # Patch embedding
        x_embed = self.patch_embed(x3)  # B, embed_dim, H/16, W/16
        B, C_emb, H_emb, W_emb = x_embed.shape

        # Reshape for transformer
        x_flat = x_embed.flatten(2).transpose(1, 2)  # B, N, embed_dim

        # Add positional embedding
        x_flat = x_flat + self.pos_embed

        # Transformer Encoder
        for blk in self.transformer_blocks:
            x_flat = blk(x_flat)

        x_flat = self.norm(x_flat)

        # Reshape back to spatial
        x_decoded = x_flat.transpose(1, 2).reshape(B, C_emb, H_emb, W_emb)

        # Decoder with skip connections
        # Stage 3
        d3 = self.decoder3(x_decoded)  # 512 channels, H/16, W/16
        d3 = self.up1(d3)  # 512 channels, H/8, W/8

        # Stage 2
        d2 = torch.cat([d3, x2], dim=1)  # 512+512 channels
        d2 = self.decoder2(d2)  # 256 channels, H/8, W/8
        d2 = self.up2(d2)  # 256 channels, H/4, W/4

        # Stage 1
        d1 = torch.cat([d2, x1], dim=1)  # 256+256 channels
        d1 = self.decoder1(d1)  # 128 channels, H/4, W/4
        d1 = self.up3(d1)  # 128 channels, H/2, W/2

        # Stage 0
        d0 = torch.cat([d1, x0], dim=1)  # 128+64 channels
        d0 = self.decoder0(d0)  # 64 channels, H/2, W/2
        d0 = self.up4(d0)  # 64 channels, H, W

        # Final output
        output = self.final(d0)

        return output


if __name__ == "__main__":  # pragma: no cover - offline package self-test (UB-21/T3.6)
    # Run as: python -m codes.transunet — builds with random weights (no
    # downloads) and prints the forward-pass output shape.
    import os

    os.environ.setdefault("UBENCH_PRETRAINED", "0")
    from codes.model_registry import create_model

    _m = create_model("transunet", in_channels=1, num_classes=10, img_size=256)
    print("transunet:", tuple(_m(torch.zeros(2, 1, 256, 256)).shape))
