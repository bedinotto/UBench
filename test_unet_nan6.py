import torch
import torch.nn as nn
from codes.unet_v2 import UNet

device = torch.device('cuda')
model = UNet(in_channels=1, num_classes=10).to(device)

images = torch.rand(8, 1, 256, 256).to(device)

def check(name, tensor):
    if torch.isnan(tensor).any():
        print(f"{name} has NaN!")
        return True
    return False

with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    x = images
    
    # manual forward
    x1 = model.enc1(x)
    if check("enc1", x1): exit()

print("Reached end successfully with bfloat16")
