import os
import sys
import torch
from torchvision.utils import save_image, make_grid
import hydra
from omegaconf import DictConfig

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.diffusion.unet import UNet
from src.vision.diffusion.diffusion import GaussianDiffusion
from src.data.loaders import get_dataloaders


@hydra.main(version_base=None, config_path="../configs", config_name="train_diffusion")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data and get one image
    trainloader, _ = get_dataloaders(cfg)
    images, _ = next(iter(trainloader))
    image = images[0:1].to(device) # Single image

    # Initialize model and diffusion
    model = UNet(
        dim=32, 
        channels=3,
        dim_mults=(1, 2, 4),
    ).to(device)
    
    diffusion = GaussianDiffusion(
        model=model,
        image_size=cfg.dataset.image_size,
        timesteps=1000, 
        beta_schedule='linear'
    ).to(device)

    # 1. Forward Process Visualization
    print("Visualizing Forward Process...")
    forward_steps = []
    # Show steps: 0, 100, 200, ..., 1000
    for t_val in range(0, 1001, 100):
        t = torch.tensor([min(t_val, 999)], device=device).long()
        noisy_image = diffusion.q_sample(image, t)
        forward_steps.append(noisy_image.cpu())
    
    forward_grid = torch.cat(forward_steps, dim=0)

    # 2. Reverse Process Visualization (using overfit checkpoint)
    # Note: The overfit checkpoint was trained with 100 timesteps. 
    # Let's re-initialize diffusion for 100 steps to match the checkpoint.
    diffusion_small = GaussianDiffusion(
        model=model,
        image_size=cfg.dataset.image_size,
        timesteps=100,
        beta_schedule='linear'
    ).to(device)

    # Use project root for checkpoint path since Hydra changes CWD
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    checkpoint_path = os.path.join(project_root, "checkpoints", "ddpm_overfit.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        diffusion_small.load_state_dict(torch.load(checkpoint_path, map_location=device))
        diffusion_small.eval()

        print("Visualizing Reverse Process...")
        # We'll collect samples every 10 steps
        reverse_steps = []
        
        # Start from noise
        img = torch.randn((1, 3, cfg.dataset.image_size, cfg.dataset.image_size), device=device)
        reverse_steps.append(img.cpu())

        for i in reversed(range(0, 100)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = diffusion_small.p_sample(img, t, i)
            if i % 10 == 0:
                reverse_steps.append(img.cpu())
        
        reverse_grid = torch.cat(reverse_steps, dim=0)
    else:
        print("Checkpoint not found, skipping reverse process visualization.")
        reverse_grid = None

    # De-normalize
    mean = torch.tensor(cfg.dataset.mean).view(1, 3, 1, 1)
    std = torch.tensor(cfg.dataset.std).view(1, 3, 1, 1)
    
    forward_grid = forward_grid * std + mean
    forward_grid = torch.clamp(forward_grid, 0, 1)
    save_image(make_grid(forward_grid, nrow=11), "forward_process.png")
    print("Saved forward_process.png")

    if reverse_grid is not None:
        reverse_grid = reverse_grid * std + mean
        reverse_grid = torch.clamp(reverse_grid, 0, 1)
        save_image(make_grid(reverse_grid, nrow=11), "reverse_process.png")
        print("Saved reverse_process.png")


if __name__ == "__main__":
    main()
