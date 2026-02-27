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
from util.losses import CombinedLoss
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

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

def calculate_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan')) # If no ground truth or prediction for this class, ignore
        else:
            ious.append(intersection / union)
    return ious

def evaluate():
    device = "cpu" # Default to CPU for safe local evaluation, or "cuda" if available
    if torch.cuda.is_available(): device = "cuda"
    
    print(f"Evaluating on {device}...")
    
    # Data paths
    data_root = "/Users/vijay/projects/eyeclip/EYE_SEG/clip_mlp/data/Ocular_Toxoplasmosis_Data_V3"
    csv_file = os.path.join(data_root, "dataset_labels.csv")
    img_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "masks")
    
    # Model Setup
    clip_model_type = "ViT-L/14"
    model, _ = eyeclip.load(clip_model_type, device=device, jit=False)
    model.float()
    model.eval()
    
    visual = model.visual
    embed_dim = visual.positional_embedding.shape[-1]
    patch_size = visual.conv1.kernel_size[0]
    num_ftrs = visual.output_dim
    
    # Load Heads
    seg_head = SegmentationHead(embed_dim=embed_dim, num_classes=3, patch_size=patch_size).to(device)
    seg_head.load_state_dict(torch.load("seg_head_epoch_20.pth", map_location=device))
    seg_head.eval()
    
    class_head = nn.Linear(num_ftrs, 3).to(device)
    class_head.load_state_dict(torch.load("cls_head_epoch_20.pth", map_location=device))
    class_head.eval()
    
    # Dataset and Loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    dataset = SegmentationDataset(csv_file, img_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Metrics Accumulators
    all_cls_preds = []
    all_cls_targets = []
    all_ious = []
    
    print("Running inference on dataset...")
    with torch.no_grad():
        for imgs, masks, cls_labels, _ in tqdm(dataloader):
            imgs, masks, cls_labels = imgs.to(device), masks.to(device), cls_labels.to(device)
            
            # Forward
            cls_feat, patch_feat = model.visual(imgs.type(model.dtype), return_patches=True)
            
            # Classification
            cls_logits = class_head(cls_feat.float())
            cls_preds = torch.argmax(cls_logits, dim=1)
            all_cls_preds.extend(cls_preds.cpu().numpy())
            all_cls_targets.extend(cls_labels.cpu().numpy())
            
            # Segmentation
            seg_logits = seg_head(patch_feat.float())
            seg_preds = torch.argmax(seg_logits, dim=1)
            
            # Batch IoU
            for i in range(imgs.size(0)):
                batch_iou = calculate_iou(seg_preds[i], masks[i], num_classes=3)
                all_ious.append(batch_iou)
                
    # Calculate Final Classification Metrics
    accuracy = accuracy_score(all_cls_targets, all_cls_preds)
    f1 = f1_score(all_cls_targets, all_cls_preds, average='weighted')
    precision = precision_score(all_cls_targets, all_cls_preds, average='weighted')
    recall = recall_score(all_cls_targets, all_cls_preds, average='weighted')
    conf_mat = confusion_matrix(all_cls_targets, all_cls_preds)
    
    # Calculate Final Segmentation Metrics (Mean IoU per class)
    all_ious = np.array(all_ious)
    mean_ious = np.nanmean(all_ious, axis=0)
    miou = np.nanmean(mean_ious)
    
    # Report Results
    print("\n" + "="*40)
    print("      EYECLIP PERFORMANCE REPORT")
    print("="*40)
    print(f"Classification Metrics:")
    print(f" - Accuracy:  {accuracy:.4f}")
    print(f" - F1-Score:  {f1:.4f}")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall:    {recall:.4f}")
    print(f"\nConfusion Matrix:\n{conf_mat}")
    
    print("\n" + "-"*40)
    print(f"Segmentation Metrics (IoU):")
    label_names = ["Background", "Inactive (Scar)", "Active (Inflam)"]
    for i, name in enumerate(label_names):
        print(f" - {name}: {mean_ious[i]:.4f}")
    print(f"\n -> Mean IoU: {miou:.4f}")
    print("="*40)
    
    # Save results to file
    with open("performance_metrics.txt", "w") as f:
        f.write("EYECLIP PERFORMANCE REPORT\n")
        f.write("="*26 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"Mean IoU:  {miou:.4f}\n")
        f.write("\nIoU per Class:\n")
        for i, name in enumerate(label_names):
            f.write(f" - {name}: {mean_ious[i]:.4f}\n")

if __name__ == "__main__":
    evaluate()
