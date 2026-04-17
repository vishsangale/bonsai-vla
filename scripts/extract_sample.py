import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

def save_sample():
    data_path = "./data"
    os.makedirs(data_path, exist_ok=True)
    
    # Simple transform to get raw image
    transform = transforms.Compose([transforms.ToTensor()])
    
    dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    img_tensor, label = dataset[4] # Index 4 is a clear deer image
    
    # Convert to PIL and save
    img_pil = transforms.ToPILImage()(img_tensor)
    # Resize to 512x512 for even better quality
    img_pil = img_pil.resize((512, 512), Image.Resampling.LANCZOS)
    
    os.makedirs("visualizations/assets", exist_ok=True)
    img_pil.save("visualizations/assets/cifar_deer.png")
    print(f"Saved sample image to visualizations/assets/cifar_deer.png")

if __name__ == "__main__":
    save_sample()
