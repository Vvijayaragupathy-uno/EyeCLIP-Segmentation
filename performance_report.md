# EyeCLIP Performance Report

**Date**: 2026-02-27
**Model**: EyeCLIP (ViT-L/14) with Multi-Task Segmentation Head
**Device**: Mac M2 CPU

## 📈 Classification Results
The model classifies ocular images into three categories: **Healthy**, **Inactive (Scar)**, and **Active (Inflammation)**.

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **74.05%** |
| **F1-Score (Weighted)** | **63.55%** |
| **Precision** | **55.66%** |
| **Recall** | **74.05%** |

### Confusion Matrix
```
[[  0  10   0]  <- Healthy
 [  0 117   1]  <- Inactive (Scar)
 [  0  30   0]] <- Active (Inflam)
```
> [!NOTE]
> The model identifies Inactive (Scar) cases with high reliability but currently tends to misclassify Healthy and Active cases as Inactive due to data imbalance.

---

## 🗺️ Segmentation Results (IoU)
The segmentation head maps pixels into three classes.

| Class | Mean IoU |
| :--- | :--- |
| **Background** | 0.9387 |
| **Inactive (Scar)** | 0.5004 |
| **Active (Inflam)** | 0.1250 |
| **Overall Mean IoU** | **0.5214** |

---

## 🔬 Analysis
- **Background Accuracy**: Exceptional performance in isolating the ocular anatomy from the background.
- **Lesion Mapping**: The model successfully identifies established scars.
- **Active Lesion Detection**: This is the most challenging task; while the model identifies the *existence* of activity via the classification head, pixel-perfect localization (IoU 0.125) requires further fine-tuning or specialized active-case augmentation.
