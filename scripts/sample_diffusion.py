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


@hydra.main(version_base=None, config_path="../configs", config_name="train_diffusion")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model (must match training)
    model = UNet(
        dim=cfg.model.dim,
        init_dim=cfg.model.init_dim,
        out_dim=cfg.model.out_dim,
        dim_mults=cfg.model.dim_mults,
        channels=cfg.model.channels,
        resnet_block_groups=cfg.model.resnet_block_groups,
        attn_heads=cfg.model.attn_heads,
        attn_dim_head=cfg.model.attn_dim_head
    ).to(device)
    
    diffusion = GaussianDiffusion(
        model=model,
        image_size=cfg.dataset.image_size,
        timesteps=cfg.diffusion.timesteps,
        beta_schedule=cfg.diffusion.beta_schedule
    ).to(device)

    # Load checkpoint
    checkpoint_dir = cfg.training.checkpoint_dir
    # Find latest checkpoint or allow user to specify
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("ddpm_epoch") and f.endswith(".pth")]
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    # Sort checkpoints by epoch number correctly (natural sorting)
    def get_epoch(filename):
        try:
            return int(filename.split('_')[-1].split('.')[0])
        except (ValueError, IndexError):
            return -1
            
    checkpoints = sorted(checkpoints, key=get_epoch)
    latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    diffusion.load_state_dict(torch.load(checkpoint_path, map_location=device))
    diffusion.eval()

    # Sample
    print(f"Generating {cfg.training.num_samples} samples...")
    samples = diffusion.sample(batch_size=cfg.training.num_samples)
    final_samples = samples[-1]

    # De-normalize
    mean = torch.tensor(cfg.dataset.mean).view(1, 3, 1, 1)
    std = torch.tensor(cfg.dataset.std).view(1, 3, 1, 1)
    final_samples = final_samples * std + mean
    final_samples = torch.clamp(final_samples, 0, 1)

    # Save grid
    output_path = "generated_samples.png"
    grid = make_grid(final_samples, nrow=4)
    save_image(grid, output_path)
    print(f"Saved generated samples to {output_path}")


if __name__ == "__main__":
    main()
