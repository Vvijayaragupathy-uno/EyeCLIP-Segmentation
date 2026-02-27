
import os
import pandas as pd

img_dir = "/Users/vijay/projects/eyeclip/EYE_SEG/clip_mlp/data/Ocular_Toxoplasmosis_Data_V3/images"
mask_dir = "/Users/vijay/projects/eyeclip/EYE_SEG/clip_mlp/data/Ocular_Toxoplasmosis_Data_V3/masks"
labels_file = "/Users/vijay/projects/eyeclip/EYE_SEG/clip_mlp/data/Ocular_Toxoplasmosis_Data_V3/dataset_labels.csv"

labels_df = pd.read_csv(labels_file)
masks = os.listdir(mask_dir)

# Check healthy images
healthy_imgs = labels_df[labels_df['Label'] == 'healthy']['Image_name'].tolist()
healthy_with_masks = []
for img in healthy_imgs:
    base = os.path.splitext(img)[0]
    if any(m.startswith(base) for m in masks):
        healthy_with_masks.append(img)

print(f"Healthy images: {len(healthy_imgs)}")
print(f"Healthy images with masks: {len(healthy_with_masks)}")

# Check active vs inactive mask counts
active_imgs = labels_df[labels_df['Label'] == 'active']['Image_name'].tolist()
inactive_imgs = labels_df[labels_df['Label'] == 'inactive']['Image_name'].tolist()

active_with_masks = [img for img in active_imgs if any(m.startswith(os.path.splitext(img)[0]) for m in masks)]
inactive_with_masks = [img for img in inactive_imgs if any(m.startswith(os.path.splitext(img)[0]) for m in masks)]

print(f"Active images: {len(active_imgs)}, with masks: {len(active_with_masks)}")
print(f"Inactive images: {len(inactive_imgs)}, with masks: {len(inactive_with_masks)}")

# Sample active masks
if active_with_masks:
    sample_active = active_with_masks[0]
    base = os.path.splitext(sample_active)[0]
    print(f"Sample Active ({sample_active}) masks: {[m for m in masks if m.startswith(base)]}")

if inactive_with_masks:
    sample_inactive = inactive_with_masks[0]
    base = os.path.splitext(sample_inactive)[0]
    print(f"Sample Inactive ({sample_inactive}) masks: {[m for m in masks if m.startswith(base)]}")
