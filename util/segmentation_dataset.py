
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as T
import torch.nn.functional as F

class SegmentationDataset(Dataset):
    """
    Custom Dataset for EyeCLIP Segmentation.
    Handles matching images with their respective masks and disease labels.
    Categories: 0: Background, 1: Inactive Lesion (Scar), 2: Active Lesion (Inflammation)
    """
    def __init__(self, csv_file, img_dir, mask_dir, transform=None, mask_transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Filter only images that have AT LEAST ONE mask
        self.samples = self._find_masked_images()
        
    def _find_masked_images(self):
        masks = os.listdir(self.mask_dir)
        masked_samples = []
        
        for idx, row in self.labels_df.iterrows():
            img_name = row['Image_name']
            base = os.path.splitext(img_name)[0]
            
            # Find all masks related to this image
            related_masks = [m for m in masks if m.startswith(base)]
            if related_masks:
                masked_samples.append({
                    "image": img_name,
                    "label": row['Label'],
                    "masks": related_masks
                })
        return masked_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.img_dir, sample['image'])
        img = Image.open(img_path).convert("RGB")
        
        # Determine disease label (Healthy=0, Inactive=1, Active=2)
        label_map = {"healthy": 0, "inactive": 1, "active": 2}
        cls_label = label_map.get(sample['label'].lower(), 0)
        
        # Build 3-class mask
        # We start with the image dimensions (un-resized)
        raw_w, raw_h = img.size
        combined_mask = np.zeros((raw_h, raw_w), dtype=np.uint8)
        
        for m_name in sample['masks']:
            m_path = os.path.join(self.mask_dir, m_name)
            m_img = Image.open(m_path).convert("L")
            m_arr = np.array(m_img)
            
            # Threshold to handle JPEG artifacts
            binary_mask = (m_arr > 127).astype(np.uint8)
            
            if "-a" in m_name:
                # Active lesion area (Class 2)
                combined_mask[binary_mask == 1] = 2
            else:
                # General/Inactive lesion area (Class 1)
                # Only set if not already set to Active (to avoid overwriting -a with 1)
                mask_at_zero = (combined_mask == 0)
                combined_mask[np.logical_and(binary_mask == 1, mask_at_zero)] = 1
        
        mask = Image.fromarray(combined_mask)
        
        if self.transform:
            img = self.transform(img)
            
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Default resize for mask
            mask = mask.resize((224, 224), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).long()

        return img, mask, cls_label, sample['image']
