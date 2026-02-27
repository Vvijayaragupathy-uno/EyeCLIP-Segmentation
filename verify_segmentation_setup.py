
import os
import torch

# Compatibility patch for older timm versions
try:
    import torch._six
except ImportError:
    import types
    import collections.abc
    torch._six = types.ModuleType("torch._six")
    torch._six.container_abcs = collections.abc
    import sys
    sys.modules["torch._six"] = torch._six

from torch.utils.data import DataLoader
from torchvision import transforms
import eyeclip
from eyeclip.segmentation_head import SegmentationHead
from util.segmentation_dataset import SegmentationDataset

def verify():
    device = "cpu"
    print(f"Verifying on {device}...")
    
    # Data paths
    data_root = "/Users/vijay/projects/eyeclip/EYE_SEG/clip_mlp/data/Ocular_Toxoplasmosis_Data_V3"
    csv_file = os.path.join(data_root, "dataset_labels.csv")
    img_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "masks")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    # Dataset
    dataset = SegmentationDataset(csv_file, img_dir, mask_dir, transform=transform)
    print(f"Dataset size: {len(dataset)}")
    
    # Get one item
    img, mask, cls_label, name = dataset[0]
    print(f"Item 0: {name}, Label: {cls_label}")
    print(f"Image shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Unique mask values: {torch.unique(mask)}")
    
    # Load model
    clip_model_type = "ViT-B/32" # Use B/32 for faster CPU test
    print(f"Loading model {clip_model_type}...")
    model, _ = eyeclip.load(clip_model_type, device=device, jit=False)
    model.float()
    
    visual = model.visual
    embed_dim = visual.positional_embedding.shape[-1]
    patch_size = visual.conv1.kernel_size[0]
    
    # Seg Head
    seg_head = SegmentationHead(embed_dim=embed_dim, num_classes=3, patch_size=patch_size)
    
    # Forward Pass
    with torch.no_grad():
        _, patches = model.visual(img.unsqueeze(0), return_patches=True)
        print(f"Patches shape: {patches.shape}")
        
        logits = seg_head(patches)
        print(f"Output logits shape: {logits.shape}")
        
    print("\nVerification Successful!")

if __name__ == "__main__":
    verify()
