
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
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import eyeclip
from eyeclip.segmentation_head import SegmentationHead
from util.segmentation_dataset import SegmentationDataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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

def visualize(num_samples=5):
    device = "cpu"
    print(f"Visualizing on {device}...")
    
    # Data paths
    data_root = "/Users/vijay/projects/eyeclip/EYE_SEG/clip_mlp/data/Ocular_Toxoplasmosis_Data_V3"
    csv_file = os.path.join(data_root, "dataset_labels.csv")
    img_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "masks")
    
    # Weights
    seg_weights = "seg_head_epoch_20.pth"
    cls_weights = "cls_head_epoch_20.pth"
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    # Dataset
    dataset = SegmentationDataset(csv_file, img_dir, mask_dir, transform=transform)
    
    # Load model
    clip_model_type = "ViT-L/14"
    model, _ = eyeclip.load(clip_model_type, device=device, jit=False)
    model.float()
    model.eval()
    
    visual = model.visual
    embed_dim = visual.positional_embedding.shape[-1]
    patch_size = visual.conv1.kernel_size[0]
    num_ftrs = visual.output_dim
    
    # Heads
    seg_head = SegmentationHead(embed_dim=embed_dim, num_classes=3, patch_size=patch_size)
    seg_head.load_state_dict(torch.load(seg_weights, map_location=device))
    seg_head.eval()
    
    class_head = nn.Linear(num_ftrs, 3)
    class_head.load_state_dict(torch.load(cls_weights, map_location=device))
    class_head.eval()
    
    label_map = {0: "Healthy", 1: "Inactive", 2: "Active"}
    
    # Output dir for plots
    os.makedirs("visualizations", exist_ok=True)
    
    for i in range(num_samples):
        # Pick random samples
        idx = np.random.randint(0, len(dataset))
        img_tensor, mask_tensor, cls_label, name = dataset[idx]
        
        with torch.no_grad():
            cls_feat, patch_feat = model.visual(img_tensor.unsqueeze(0), return_patches=True)
            
            # Predict Class
            cls_logits = class_head(cls_feat.float())
            pred_cls = torch.argmax(cls_logits, dim=1).item()
            
            # Predict Mask
            seg_logits = seg_head(patch_feat.float())
            pred_mask = torch.argmax(seg_logits, dim=1).squeeze(0).numpy()
            
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original Image (undo normalization for display)
        inv_normalize = transforms.Normalize(
            mean=[-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711],
            std=[1/0.26862954, 1/0.26130258, 1/0.27577711]
        )
        display_img = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
        display_img = np.clip(display_img, 0, 1)
        
        axes[0].imshow(display_img)
        axes[0].set_title(f"Image: {name}\nActual: {label_map[cls_label]}")
        axes[0].axis('off')
        
        # Ground Truth Mask
        axes[1].imshow(mask_tensor.numpy(), vmin=0, vmax=2, cmap='viridis')
        axes[1].set_title("Ground Truth Mask\n(0:BG, 1:Scar, 2:Inflam)")
        axes[1].axis('off')
        
        # Predicted Mask
        axes[2].imshow(pred_mask, vmin=0, vmax=2, cmap='viridis')
        axes[2].set_title(f"Predicted Mask\nPred Label: {label_map[pred_cls]}")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"visualizations/result_{i}_{name}.png")
        print(f"Saved visualization for {name} to visualizations/result_{i}_{name}.png")
        plt.close()

if __name__ == "__main__":
    visualize()
