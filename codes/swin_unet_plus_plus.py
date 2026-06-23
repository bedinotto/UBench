"""
Thermal Facial Region Detection System - Swin-UNet++
====================================================
Swin Transformer-based U-Net++ architecture for thermal face segmentation.
Defines the SwinUNetPlusPlus model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self attention"""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with shifted window attention"""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer for downsampling"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # Padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=256, patch_size=4, in_chans=1, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size //
                                   patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * \
            self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H // self.patch_size, W // self.patch_size


class NestedConvBlock(nn.Module):
    """Nested convolution block for dense skip connections"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


from .model_registry import register_model

@register_model("swin_unet_plus_plus")
class SwinUNetPlusPlus(nn.Module):
    """Swin-UNet++ architecture with nested dense skip connections"""

    def __init__(self, img_size=256, in_channels=1, num_classes=10,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_channels, embed_dim=embed_dim
        )

        # Swin Transformer encoder stages
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=layer_dim,
                    num_heads=num_heads[i_layer],
                    window_size=8,
                    shift_size=0 if (i % 2 == 0) else 8 // 2
                )
                for i in range(depths[i_layer])
            ])
            self.layers.append(layer)

            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(layer_dim)
                self.downsample_layers.append(downsample)

        # Projection layers to convert from transformer features to CNN features
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(int(embed_dim * 2 ** i), int(embed_dim * 2 ** i), 1)
            for i in range(self.num_layers)
        ])

        # Nested dense skip connections (UNet++ style)
        # x^0_0, x^1_0, x^2_0, x^3_0 are encoder outputs
        # x^0_1, x^1_1, x^2_1, x^3_1 are nested blocks
        self.conv0_1 = NestedConvBlock(embed_dim * 3, embed_dim)  # 288 -> 96
        self.conv1_1 = NestedConvBlock(
            embed_dim * 6, embed_dim * 2)  # 576 -> 192
        self.conv2_1 = NestedConvBlock(
            embed_dim * 12, embed_dim * 4)  # 1152 -> 384

        self.conv0_2 = NestedConvBlock(embed_dim * 4, embed_dim)  # 384 -> 96
        self.conv1_2 = NestedConvBlock(
            embed_dim * 8, embed_dim * 2)  # 768 -> 192

        self.conv0_3 = NestedConvBlock(embed_dim * 5, embed_dim)  # 480 -> 96

        # Backwards/upsample projection layers
        self.up1 = nn.ConvTranspose2d(
            embed_dim * 2, embed_dim * 2, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(
            embed_dim * 4, embed_dim * 4, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(
            embed_dim * 8, embed_dim * 8, 2, stride=2)

        # Final output layer
        self.final = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch embedding
        x, H_enc, W_enc = self.patch_embed(x)

        # Encoder - Swin Transformer stages
        enc_features = []
        for i_layer in range(self.num_layers):
            # Set spatial dimensions for blocks
            for blk in self.layers[i_layer]:
                blk.H, blk.W = H_enc, W_enc
                x = blk(x)

            # Convert to CNN format and store
            x_cnn = x.view(B, H_enc, W_enc, -1).permute(0, 3, 1, 2)
            x_cnn = self.proj_layers[i_layer](x_cnn)
            enc_features.append(x_cnn)

            # Downsample
            if i_layer < self.num_layers - 1:
                x = self.downsample_layers[i_layer](x, H_enc, W_enc)
                H_enc = H_enc // 2
                W_enc = W_enc // 2

        # Decoder with nested dense skip connections (UNet++ style)
        x0_0, x1_0, x2_0, x3_0 = enc_features

        # First column of nested blocks
        x2_1 = self.up3(x3_0)
        x2_1 = torch.cat([x2_1, x2_0], dim=1)
        x2_1 = self.conv2_1(x2_1)

        x1_1 = self.up2(x2_1)
        x1_1 = torch.cat([x1_1, x1_0], dim=1)
        x1_1 = self.conv1_1(x1_1)

        x0_1 = self.up1(x1_1)
        x0_1 = torch.cat([x0_1, x0_0], dim=1)
        x0_1 = self.conv0_1(x0_1)

        # Second column
        x1_2 = self.up2(x2_1)
        x1_2 = torch.cat([x1_2, x1_0, x1_1], dim=1)
        x1_2 = self.conv1_2(x1_2)

        x0_2 = self.up1(x1_2)
        x0_2 = torch.cat([x0_2, x0_0, x0_1], dim=1)
        x0_2 = self.conv0_2(x0_2)

        # Third column
        x0_3 = self.up1(x1_2)
        x0_3 = torch.cat([x0_3, x0_0, x0_1, x0_2], dim=1)
        x0_3 = self.conv0_3(x0_3)

        # Final upsampling to original resolution
        x_out = F.interpolate(x0_3, size=(
            H, W), mode='bilinear', align_corners=False)

        # Output
        output = self.final(x_out)

        return output
