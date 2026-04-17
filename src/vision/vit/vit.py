import torch
from torch import nn

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        # TODO: Implement PatchEmbedding, TransformerBlocks, and Classification Head
        pass

    def forward(self, img):
        # TODO: Implement forward pass
        pass
