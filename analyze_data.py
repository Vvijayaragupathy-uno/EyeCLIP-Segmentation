
import os
import pandas as pd

img_dir = "/Users/vijay/projects/eyeclip/EYE_SEG/clip_mlp/data/Ocular_Toxoplasmosis_Data_V3/images"
mask_dir = "/Users/vijay/projects/eyeclip/EYE_SEG/clip_mlp/data/Ocular_Toxoplasmosis_Data_V3/masks"
labels_file = "/Users/vijay/projects/eyeclip/EYE_SEG/clip_mlp/data/Ocular_Toxoplasmosis_Data_V3/dataset_labels.csv"

labels_df = pd.read_csv(labels_file)

images = os.listdir(img_dir)
masks = os.listdir(mask_dir)

print(f"Total images: {len(images)}")
print(f"Total masks: {len(masks)}")

mapping = []
for img in images:
    base = os.path.splitext(img)[0]
    ext = os.path.splitext(img)[1]
    
    related_masks = [m for m in masks if m.startswith(base)]
    if related_masks:
        mapping.append({
            "image": img,
            "label": labels_df[labels_df['Image_name'] == img]['Label'].values[0] if img in labels_df['Image_name'].values else "Unknown",
            "masks": related_masks
        })

mapping_df = pd.DataFrame(mapping)
print("\nMapping Sample (images with masks):")
print(mapping_df.head(20))

print("\nFrequency of mask suffixes:")
suffixes = []
for ms in mapping_df['masks']:
    for m in ms:
        # get what's between the base and the extension
        img_base = os.path.splitext(mapping_df[mapping_df['masks'].apply(lambda x: m in x)]['image'].values[0])[0]
        suffix = m.replace(img_base, "").replace(".jpg", "").replace(".JPG", "")
        suffixes.append(suffix)
print(pd.Series(suffixes).value_counts())
