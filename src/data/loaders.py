import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(cfg):
    """
    Returns train and test dataloaders based on the Hydra dataset configuration.
    """
    data_path = cfg.dataset.path
    batch_size = cfg.training.batch_size
    num_workers = cfg.training.num_workers
    img_size = cfg.dataset.image_size
    
    mean = list(cfg.dataset.mean)
    std = list(cfg.dataset.std)

    # Basic transforms
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if cfg.dataset.name.lower() == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: '{cfg.dataset.name}'.")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader
