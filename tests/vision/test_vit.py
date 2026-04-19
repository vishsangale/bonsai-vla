import torch
import unittest
import sys
import os

# Ensure the root directory is in the path for imports
sys.path.append(os.getcwd())
from src.vision.vit.vit import VisionTransformer, SimpleViT
from src.vision.vit.pos_encoding import sinusoidal_2d_pos_encoding

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

class TestSimpleViT(unittest.TestCase):
    def test_forward_pass_cifar(self):
        # Configuration for CIFAR-10 like images
        img_size = 32
        patch_size = 4
        in_channels = 3
        num_classes = 10
        batch_size = 2
        
        model = SimpleViT(
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
        print("SimpleViT Test passed: Forward pass outputs correct shape.")

    def test_no_cls_token(self):
        model = SimpleViT(emb_dim=64, num_heads=4)
        self.assertFalse(hasattr(model, "cls_token"), "SimpleViT should not have a cls_token")

    def test_gap_pooling(self):
        # GAP pooling should average over all patch tokens
        img_size = 32
        patch_size = 8 # 4x4 grid = 16 patches
        emb_dim = 64
        model = SimpleViT(img_size=img_size, patch_size=patch_size, emb_dim=emb_dim, num_heads=4)
        
        x = torch.randn(1, 3, img_size, img_size)
        # We can't easily check the internal mean without hooks, 
        # but we can check if it runs without error and gives correct output dim.
        out = model(x)
        self.assertEqual(out.shape, (1, 10))

    def test_pos_encoding_shape(self):
        grid_h, grid_w = 8, 8
        emb_dim = 128
        pos = sinusoidal_2d_pos_encoding(grid_h, grid_w, emb_dim)
        self.assertEqual(pos.shape, (grid_h * grid_w, emb_dim))

if __name__ == '__main__':
    unittest.main()
