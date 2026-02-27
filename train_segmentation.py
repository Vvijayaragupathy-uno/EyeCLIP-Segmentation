
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import eyeclip
from eyeclip.segmentation_head import SegmentationHead
from util.segmentation_dataset import SegmentationDataset
from util.losses import CombinedLoss
from tqdm import tqdm

def train():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    epochs = 20
    lr = 1e-4
    
    print(f"Using device: {device}")
    
    # Data paths
    data_root = "/Users/vijay/projects/eyeclip/EYE_SEG/clip_mlp/data/Ocular_Toxoplasmosis_Data_V3"
    csv_file = os.path.join(data_root, "dataset_labels.csv")
    img_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "masks")
    
    # Transforms
    # Note: Using CLIP's recommended normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    # Dataset & Loader
    print("Loading dataset...")
    dataset = SegmentationDataset(csv_file, img_dir, mask_dir, transform=transform)
    print(f"Found {len(dataset)} images with masks.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    clip_model_type = "ViT-L/14" 
    print(f"Loading EyeCLIP backbone ({clip_model_type})...")
    model, _ = eyeclip.load(clip_model_type, device=device, jit=False)
    model.float()
    
    # Determine embed_dim and patch_size from model
    visual = model.visual
    embed_dim = visual.positional_embedding.shape[-1]
    patch_size = visual.conv1.kernel_size[0]
    print(f"Backbone properties: embed_dim={embed_dim}, patch_size={patch_size}")
    
    # Segmentation Head
    print("Initializing Segmentation Head...")
    seg_head = SegmentationHead(embed_dim=embed_dim, num_classes=3, patch_size=patch_size).to(device)
    
    # Losses & Optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(seg_head.parameters(), lr=lr)
    
    # Training Loop
    model.eval() # Keep backbone frozen
    print("Starting training...")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        seg_head.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, masks, cls_labels, names in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            
            # Forward
            with torch.no_grad():
                # Get patch features directly from the visual encoder
                # We modified VisionTransformer.forward to support return_patches
                _, patches = model.visual(imgs.type(model.dtype), return_patches=True)
            
            # patches is (N, 1%SAME%, D) - 196 for ViT-L/14
            logits = seg_head(patches.float()) # Ensure float for MLP
            
            loss = criterion(logits, masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Save the best head
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(seg_head.state_dict(), "segmentation_head_best.pth")
            print(f"New best model saved with loss: {best_loss:.4f}")
        
    print("Training complete.")

if __name__ == "__main__":
    train()
