import os
import sys
import torch
import torch.optim as optim
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

    # Load data
    trainloader, _ = get_dataloaders(cfg)
    # Get a single batch
    batch = next(iter(trainloader))
    images, _ = batch
    images = images[:8].to(device) # Overfit on 8 images
    
    # Initialize model
    model = UNet(
        dim=32, # Smaller for fast overfit
        channels=3,
        dim_mults=(1, 2, 4),
    ).to(device)
    
    diffusion = GaussianDiffusion(
        model=model,
        image_size=cfg.dataset.image_size,
        timesteps=100, # Fewer timesteps for faster overfit test
        beta_schedule='linear'
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(diffusion.parameters(), lr=1e-3)
    
    print("Overfitting on a single batch...")
    for i in range(500):
        diffusion.train()
        t = torch.randint(0, 100, (images.shape[0],), device=device).long()
        
        optimizer.zero_grad()
        loss = diffusion.p_losses(images, t)
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Step {i}, loss: {loss.item():.6f}")

    # Save checkpoint
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.training.checkpoint_dir, "ddpm_overfit.pth")
    torch.save(diffusion.state_dict(), checkpoint_path)
    print(f"Saved overfit checkpoint to {checkpoint_path}")

    # Sample to verify
    print("Sampling from overfitted model...")
    diffusion.eval()
    samples = diffusion.sample(batch_size=8)
    final_samples = samples[-1]

    # De-normalize
    mean = torch.tensor(cfg.dataset.mean).view(1, 3, 1, 1)
    std = torch.tensor(cfg.dataset.std).view(1, 3, 1, 1)
    final_samples = final_samples * std + mean
    final_samples = torch.clamp(final_samples, 0, 1)

    # Save comparison
    output_path = "overfit_test.png"
    
    orig_images = images.cpu() * std + mean
    orig_images = torch.clamp(orig_images, 0, 1)
    
    comparison = torch.cat([orig_images, final_samples], dim=0)
    grid = make_grid(comparison, nrow=8)
    save_image(grid, output_path)
    print(f"Saved overfit comparison to {output_path}")


if __name__ == "__main__":
    main()
