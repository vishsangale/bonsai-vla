import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Turns a 2D input image into a 1D sequence learnable embedding vector.
    Uses a 2D convolution with stride=patch_size to extract non-overlapping patches.
    """
    def __init__(self, in_channels=3, patch_size=4, emb_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.proj(x) # shape: (B, emb_dim, H/patch_size, W/patch_size)
        x = x.flatten(2) # shape: (B, emb_dim, num_patches)
        x = x.transpose(1, 2) # shape: (B, num_patches, emb_dim)
        return x

class TransformerEncoderLayer(nn.Module):
    """
    A single Vision Transformer encoder block.
    """
    def __init__(self, emb_dim=192, num_heads=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(emb_dim)
        # Note: batch_first=True makes input shape (B, SeqLen, EmbDim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.ln_2 = nn.LayerNorm(emb_dim)
        hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 1. LayerNorm + Self-Attention + Residual
        x_ln1 = self.ln_1(x)
        attn_out, _ = self.attn(x_ln1, x_ln1, x_ln1)
        x = x + attn_out
        
        # 2. LayerNorm + MLP + Residual
        x_ln2 = self.ln_2(x)
        mlp_out = self.mlp(x_ln2)
        x = x + mlp_out
        
        return x

class VisionTransformer(nn.Module):
    """
    The complete Vision Transformer architecture.
    """
    def __init__(self, 
                 img_size=32, 
                 in_channels=3, 
                 patch_size=4, 
                 num_classes=10,
                 emb_dim=192, 
                 depth=12, 
                 num_heads=3, 
                 mlp_ratio=4.0, 
                 dropout=0.1):
        super().__init__()
        
        # Calculate number of patches
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, emb_dim=emb_dim)
        
        # Learnable [class] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        
        # Positional embedding (+1 for the cls token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(emb_dim=emb_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        
        # Final LayerNorm
        self.norm = nn.LayerNorm(emb_dim)
        
        # Classification head
        self.head = nn.Linear(emb_dim, num_classes)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize all Linear and LayerNorm modules first
        self.apply(self._init_module_weights)
        # Then explicitly init Parameters (not touched by apply)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Patch extraction and embedding
        x = self.patch_embed(x) # (B, num_patches, emb_dim)
        
        # 2. Add [class] token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, emb_dim)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 1 + num_patches, emb_dim)
        
        # 3. Add positional embeddings and dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 4. Transformer Encoder
        for block in self.blocks:
            x = block(x)
            
        # 5. Output processing
        x = self.norm(x)
        
        # Extract the state of the [class] token
        cls_token_final = x[:, 0]
        
        # 6. Classification Head
        out = self.head(cls_token_final)
        
        return out
