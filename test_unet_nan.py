import torch
import torch.nn as nn
from codes.unet_v2 import UNet
from codes.unified_training import CombinedLoss

device = torch.device('cuda')
model = UNet(in_channels=1, num_classes=10).to(device)
criterion = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda')

# Dummy data
images = torch.rand(8, 1, 256, 256).to(device)
masks = torch.randint(0, 10, (8, 256, 256)).to(device)

for i in range(10):
    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        outputs = model(images)
        loss = criterion(outputs, masks)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"Step {i}: Loss = {loss.item()}")
