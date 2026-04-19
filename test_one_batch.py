import torch
import torch.nn as nn
import torch.optim as optim
from src.vision.vit.vit import VisionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(
    img_size=32,
    in_channels=3,
    patch_size=4,
    num_classes=10,
    emb_dim=128,
    depth=4,
    num_heads=4,
    mlp_ratio=2.0,
    dropout=0.0
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

inputs = torch.randn(8, 3, 32, 32).to(device)
labels = torch.randint(0, 10, (8,)).to(device)

print("Starting single batch overfitting...")
for i in range(50):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f"Iter {i}, Loss: {loss.item():.4f}")

