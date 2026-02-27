
import os
import torch
import torch.nn as nn

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
from util.losses import CombinedLoss # This is for segmentation (Dice+Focal)
from tqdm import tqdm
import numpy as np

def train():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    epochs = 20
    lr = 1e-4
    
    # Loss Weights
    seg_weight = 1.0
    cls_weight = 0.5 # Classification is often easier, so we weight it slightly lower initially
    
    print(f"Using device: {device}")
    
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
    
    # Dataset & Loader
    print("Loading dataset...")
    dataset = SegmentationDataset(csv_file, img_dir, mask_dir, transform=transform)
    print(f"Found {len(dataset)} images with masks and labels.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    clip_model_type = "ViT-L/14" 
    print(f"Loading EyeCLIP backbone ({clip_model_type})...")
    model, _ = eyeclip.load(clip_model_type, device=device, jit=False)
    model.float()
    
    # Determine embed_dim from model
    visual = model.visual
    embed_dim = visual.positional_embedding.shape[-1]
    patch_size = visual.conv1.kernel_size[0]
    
    # 1. Segmentation Head
    seg_head = SegmentationHead(embed_dim=embed_dim, num_classes=3, patch_size=patch_size).to(device)
    
    # 2. Classification Head 
    # (We use the visual.output_dim which is usually the CLIP projection dimension)
    num_ftrs = visual.output_dim
    class_head = nn.Linear(num_ftrs, 3).to(device) # 3 classes: Healthy, Active, Inactive
    
    # Losses
    seg_criterion = CombinedLoss() # Dice + Focal
    cls_criterion = nn.CrossEntropyLoss()
    
    # Optimizer (optimize both heads)
    optimizer = torch.optim.AdamW(
        list(seg_head.parameters()) + list(class_head.parameters()), 
        lr=lr
    )
    
    # Training Loop
    model.eval() # Keep EyeCLIP backbone frozen as a feature extractor
    print("Starting Multi-Task Training (Classification + Segmentation)...")
    
    for epoch in range(epochs):
        seg_head.train()
        class_head.train()
        total_loss = 0
        total_cls_loss = 0
        total_seg_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, masks, cls_labels, names in pbar:
            imgs, masks, cls_labels = imgs.to(device), masks.to(device), cls_labels.to(device)
            
            # Forward (Shared Backbone)
            with torch.no_grad():
                # Get both CLS and Patch features
                cls_feat, patch_feat = model.visual(imgs.type(model.dtype), return_patches=True)
            
            # Head 1: Classification
            cls_logits = class_head(cls_feat.float())
            loss_cls = cls_criterion(cls_logits, cls_labels)
            
            # Head 2: Segmentation
            seg_logits = seg_head(patch_feat.float())
            loss_seg = seg_criterion(seg_logits, masks)
            
            # Combined Loss
            loss = (seg_weight * loss_seg) + (cls_weight * loss_cls)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_seg_loss += loss_seg.item()
            
            pbar.set_postfix({
                "T_loss": f"{loss.item():.3f}",
                "S_loss": f"{loss_seg.item():.3f}",
                "C_loss": f"{loss_cls.item():.3f}"
            })
            
        print(f"Epoch {epoch+1} Summary: Total={total_loss/len(dataloader):.4f}, Seg={total_seg_loss/len(dataloader):.4f}, Cls={total_cls_loss/len(dataloader):.4f}")
        
        # Save models
        torch.save(seg_head.state_dict(), f"seg_head_epoch_{epoch+1}.pth")
        torch.save(class_head.state_dict(), f"cls_head_epoch_{epoch+1}.pth")
        
    print("Multi-Task Training complete.")

if __name__ == "__main__":
    train()
