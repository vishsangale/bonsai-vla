import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import hydra
from omegaconf import DictConfig

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.diffusion.unet import UNet
from src.vision.diffusion.diffusion import GaussianDiffusion
from src.data.loaders import get_dataloaders


@hydra.main(version_base=None, config_path="../configs", config_name="train_diffusion")
def main(cfg: DictConfig):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    trainloader, testloader = get_dataloaders(cfg)
    
    # Initialize model
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

    # Optimizer
    optimizer = optim.Adam(diffusion.parameters(), lr=cfg.training.learning_rate)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=cfg.training.log_dir)

    # Training Loop
    print("Starting training...")
    global_step = 0
    for epoch in range(cfg.training.epochs):
        diffusion.train()
        running_loss = 0.0
        
        for i, (images, _) in enumerate(trainloader):
            images = images.to(device)
            
            # Sample random timesteps for each image
            t = torch.randint(0, cfg.diffusion.timesteps, (images.shape[0],), device=device).long()
            
            optimizer.zero_grad()
            loss = diffusion.p_losses(images, t, loss_type=cfg.diffusion.loss_type)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            global_step += 1
            
            if i % cfg.training.log_interval == cfg.training.log_interval - 1:
                avg_loss = running_loss / cfg.training.log_interval
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {avg_loss:.4f}")
                writer.add_scalar("train/loss", avg_loss, global_step)
                running_loss = 0.0

        # Periodic sampling
        if (epoch + 1) % cfg.training.sample_interval == 0:
            print(f"Sampling images at epoch {epoch + 1}...")
            diffusion.eval()
            samples = diffusion.sample(batch_size=cfg.training.num_samples)
            # Take the final sample (the denoised image)
            final_samples = samples[-1]
            
            # De-normalize images (assuming normalization to [-1, 1] or mean/std)
            # For simplicity, we just clamp and scale if needed, but here we should ideally 
            # use the inverse of the transform used in loaders.py.
            # Assuming loaders.py used Normalize(mean, std), we should un-normalize.
            mean = torch.tensor(cfg.dataset.mean).view(1, 3, 1, 1)
            std = torch.tensor(cfg.dataset.std).view(1, 3, 1, 1)
            final_samples = final_samples * std + mean
            final_samples = torch.clamp(final_samples, 0, 1)

            grid = make_grid(final_samples, nrow=4)
            writer.add_image("samples", grid, epoch + 1)
            
            # Save checkpoint
            os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(cfg.training.checkpoint_dir, f"ddpm_epoch_{epoch + 1}.pth")
            torch.save(diffusion.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    writer.close()
    print("Finished Training")


if __name__ == "__main__":
    main()
