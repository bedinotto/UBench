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

with torch.amp.autocast('cuda'):
    x = images
    x = model.enc1(x)
    if check("enc1", x): exit()
    x = model.pool1(x)
    
    x = model.enc2(x)
    if check("enc2", x): exit()
    x = model.pool2(x)
    
    x = model.enc3(x)
    if check("enc3", x): exit()
    x = model.pool3(x)
    
    x = model.enc4(x)
    if check("enc4", x): exit()
    x = model.pool4(x)
    
    x = model.bottleneck(x)
    if check("bottleneck", x): exit()
    
    x1 = model.upconv4(x)
    if check("upconv4", x1): exit()

print("Reached end successfully")
