import torch
import torch.nn as nn
from codes.unet_v2 import UNet
from codes.unified_training import CombinedLoss

device = torch.device('cuda')
model = UNet(in_channels=1, num_classes=10).to(device)

images = torch.rand(8, 1, 256, 256).to(device)

with torch.amp.autocast('cuda'):
    outputs = model(images)
    print("Outputs has nan:", torch.isnan(outputs).any().item())
    print("Outputs min:", outputs.min().item())
    print("Outputs max:", outputs.max().item())

