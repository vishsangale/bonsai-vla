import hydra
from omegaconf import DictConfig, OmegaConf
import torch

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    print("Starting training with config:")
    print(OmegaConf.to_yaml(cfg))
    
    # TODO: Instantiate model from cfg.model
    # TODO: Instantiate dataloader from cfg.dataset
    # TODO: Implement training loop
    
    print("Training complete.")

if __name__ == "__main__":
    main()
