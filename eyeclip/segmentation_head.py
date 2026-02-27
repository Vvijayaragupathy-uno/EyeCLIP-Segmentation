import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    """
    Segmentation Head for CLIP-based models.
    Processes patch tokens through an MLP and upsamples to the original resolution.
    """
    def __init__(self, embed_dim=768, num_classes=3, patch_size=16, input_resolution=224):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.grid_size = input_resolution // patch_size
        
        # Projection layer to reduce dimensionality (optional but common)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: Patch tokens of shape (Batch, Num_Patches, Embed_Dim)
        Returns:
            Segmentation logits of shape (Batch, Num_Classes, H, W)
        """
        # x: (N, L, D) -> (N, L, num_classes)
        x = self.decoder(x)
        
        # Reshape to (N, grid, grid, num_classes)
        N, L, C = x.shape
        grid = int(L**0.5)
        x = x.reshape(N, grid, grid, C)
        
        # Permute to (N, C, grid, grid) for interpolation
        x = x.permute(0, 3, 1, 2)
        
        # Upsample to target resolution
        x = F.interpolate(x, size=(self.input_resolution, self.input_resolution), mode='bilinear', align_corners=False)
        
        return x
