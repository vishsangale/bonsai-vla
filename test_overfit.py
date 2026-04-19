import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
import os
sys.path.append(os.path.abspath('.'))
import argparse
from src.vision.vit.vit import VisionTransformer, SimpleViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size, patch_size, in_channels, num_classes, emb_dim = 32, 4, 3, 10, 192
depth, num_heads, mlp_ratio, dropout = 12, 3, 4.0, 0.1
learning_rate, weight_decay = 3e-4, 1e-4

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="vit", choices=["vit", "simple_vit"])
args = parser.parse_args()

if args.model == "simple_vit":
    model = SimpleViT(
        img_size=img_size, in_channels=in_channels, patch_size=patch_size,
        num_classes=num_classes, emb_dim=emb_dim, depth=depth,
        num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=0.0
    ).to(device)
else:
    model = VisionTransformer(
        img_size=img_size, in_channels=in_channels, patch_size=patch_size,
        num_classes=num_classes, emb_dim=emb_dim, depth=depth,
        num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
    ).to(device)

print(f"Overfitting on single batch with {args.model}...")

transform_train = transforms.Compose([
    transforms.RandomCrop(img_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

single_batch = next(iter(trainloader))
inputs, labels = single_batch
inputs, labels = inputs.to(device), labels.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean() * 100
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Acc = {acc.item():.2f}%")
