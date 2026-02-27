
import numpy as np
from PIL import Image

masks_to_check = [
    "/Users/vijay/projects/eyeclip/EYE_SEG/clip_mlp/data/Ocular_Toxoplasmosis_Data_V3/masks/101-a.jpg",
    "/Users/vijay/projects/eyeclip/EYE_SEG/clip_mlp/data/Ocular_Toxoplasmosis_Data_V3/masks/101.jpg",
    "/Users/vijay/projects/eyeclip/EYE_SEG/clip_mlp/data/Ocular_Toxoplasmosis_Data_V3/masks/86.jpg"
]

for m_path in masks_to_check:
    img = Image.open(m_path)
    data = np.array(img)
    print(f"Mask: {m_path}")
    print(f"Min: {data.min()}, Max: {data.max()}, Mean: {data.mean():.2f}")
    print(f"Pixels == 0: {np.sum(data == 0)}")
    print(f"Pixels == 255: {np.sum(data == 255)}")
    print(f"Pixels < 128: {np.sum(data < 128)}")
    print(f"Pixels >= 128: {np.sum(data >= 128)}")
    print("-" * 20)
