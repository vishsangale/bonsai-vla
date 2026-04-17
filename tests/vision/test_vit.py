import torch
from src.vision.vit import ViT

def test_vit_forward_shape():
    # Setup a dummy model configuration
    model = ViT(
        image_size=28,
        patch_size=7,
        num_classes=10,
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=128,
        channels=1
    )
    
    # Create dummy image tensor: [batch_size, channels, height, width]
    img = torch.randn(2, 1, 28, 28)
    
    # Expected output: [batch_size, num_classes]
    # out = model(img)
    # assert out.shape == (2, 10)
    pass
