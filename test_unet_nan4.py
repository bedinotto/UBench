import torch
import torch.nn as nn
from codes.unet_v2 import DoubleConv

device = torch.device('cuda')
block = DoubleConv(1, 64).to(device)

images = torch.rand(8, 1, 256, 256).to(device)

def check(name, tensor):
    if torch.isnan(tensor).any():
        print(f"{name} has NaN!")
        return True
    return False

with torch.amp.autocast('cuda'):
    x = images
    
    # manual forward
    x1 = block.double_conv[0](x)
    if check("Conv1", x1): exit()
    
    x2 = block.double_conv[1](x1)
    if check("BN1", x2): exit()
    
    x3 = block.double_conv[2](x2)
    if check("ReLU1", x3): exit()
    
    x4 = block.double_conv[3](x3)
    if check("Conv2", x4): exit()
    
    x5 = block.double_conv[4](x4)
    if check("BN2", x5): exit()
    
print("Reached end successfully")
