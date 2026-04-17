import torch
import unittest
import sys
import os

# Add the parent directory to sys.path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit import VisionTransformer

class TestVisionTransformer(unittest.TestCase):
    def test_forward_pass_cifar(self):
        # Configuration for CIFAR-10 like images
        img_size = 32
        patch_size = 4
        in_channels = 3
        num_classes = 10
        batch_size = 2
        
        model = VisionTransformer(
            img_size=img_size,
            in_channels=in_channels,
            patch_size=patch_size,
            num_classes=num_classes,
            emb_dim=64, # Small for testing
            depth=2,
            num_heads=2
        )
        
        # Dummy input batch
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        
        # Forward pass
        out = model(x)
        
        # Check output shape
        self.assertEqual(out.shape, (batch_size, num_classes))
        print("Test passed: Forward pass outputs correct shape.")

if __name__ == '__main__':
    unittest.main()
