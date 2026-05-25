# Facial Expression Recognition Framework
### Comparative CNN Analysis with Explainable AI (Grad-CAM & SHAP)

> **Course:** Pattern Recognition — M.Sc. Programme  
> **Institution:** University of Europe for Applied Sciences  
> **Phase:** Phase 2 — Proposal, Code Development & Technical Implementation  
> **Due:** June 7, 2026  

---

## Project Overview

This project designs, implements, and evaluates a CNN-based facial expression recognition framework using the FER-2013 dataset. Three deep learning models are trained, compared, and interpreted using Explainable AI techniques.

| Model | Type | Purpose |
|-------|------|---------|
| Custom CNN | Designed from scratch | Baseline architecture |
| MobileNetV2 | Transfer Learning (ImageNet) | Lightweight comparison |
| ResNet50 | Transfer Learning (ImageNet) | Deep residual comparison |

**XAI Methods Applied:**
- **Grad-CAM** — Visualizes which image regions each model focuses on (correct & misclassified)
- **SHAP** — Estimates pixel-level contributions to model predictions

---

## Dataset

**FER-2013** (Facial Expression Recognition 2013)  
🔗 [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

| Property | Value |
|----------|-------|
| Total Images | ~35,887 |
| Image Size | 48 × 48 pixels (grayscale) |
| Classes | 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) |
| Format | CSV / folder structure |
| Split | Train: 28,709 / Test: 3,589 |

---

## Repository Structure

```
facial-expression-recognition-framework/
│
├── docs/
│   └── proposal.md                  # Phase 2 proposal document (all 12 sections)
│
├── notebooks/
│   └── fer2013_main.ipynb           # Main Kaggle notebook (all models + XAI)
│
├── src/
│   ├── config.py                    # Hyperparameters and paths
│   ├── preprocessing.py             # Data loading, augmentation, splits
│   ├── custom_cnn.py                # Custom CNN architecture
│   ├── transfer_learning.py         # MobileNetV2 & ResNet50
│   ├── train.py                     # Training loop (shared)
│   ├── evaluate.py                  # Metrics, confusion matrix, report
│   ├── gradcam.py                   # Grad-CAM implementation
│   └── shap_explain.py              # SHAP implementation
│
├── figures/
│   ├── Fig1_Dataset_Samples/        # Class grid visualization
│   ├── Fig2_Model_Workflow/         # Architecture diagram
│   ├── Fig3_Training_Curves/        # Loss & accuracy curves (all 3 models)
│   ├── Fig4_Confusion_Matrices/     # One per model
│   ├── Fig5_GradCAM_Correct/        # Correct prediction heatmaps
│   ├── Fig6_GradCAM_Misclassified/  # Misclassification heatmaps (important!)
│   └── Fig7_SHAP_Explanations/      # SHAP pixel-level explanations
│
├── models/
│   ├── custom_cnn_best.h5           # Saved best Custom CNN weights
│   ├── mobilenetv2_best.h5          # Saved best MobileNetV2 weights
│   └── resnet50_best.h5             # Saved best ResNet50 weights
│
├── outputs/
│   ├── tables/                      # Model comparison CSV tables
│   └── metrics/                     # Per-class metrics JSON
│
├── data/
│   └── README_dataset.md            # Dataset source info & download instructions
│
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Ignores model weights, large files
└── README.md                        # This file
```

---

## How to Run

### Option A: Kaggle Notebook (Recommended — GPU required)

1. Go to [Kaggle](https://www.kaggle.com) and open a new notebook
2. Add the FER-2013 dataset: `+ Add Data` → search "fer2013"
3. Upload or import `notebooks/fer2013_main.ipynb`
4. Enable GPU: `Settings` → `Accelerator` → `GPU P100`
5. Click `Run All`

### Option B: Local (Cursor / VS Code)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/facial-expression-recognition-framework.git
cd facial-expression-recognition-framework

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download FER-2013 dataset manually from Kaggle and place in data/
# See data/README_dataset.md for instructions

# 4. Run the notebook
jupyter notebook notebooks/fer2013_main.ipynb
```

---

## Key Results (To be updated after training)

| Model | Accuracy | F1-Score | Training Time | Parameters | Grad-CAM Quality | SHAP Interpretability |
|-------|----------|----------|---------------|------------|------------------|-----------------------|
| Custom CNN | — | — | — | — | — | — |
| MobileNetV2 | — | — | — | — | — | — |
| ResNet50 | — | — | — | — | — | — |

---

## Research Questions

| RQ | Question |
|----|----------|
| RQ1 | How accurately can CNN-based models perform facial expression recognition using the FER-2013 dataset? |
| RQ2 | Does transfer learning (MobileNetV2, ResNet50) improve classification performance over a custom CNN? |
| RQ3 | Which model provides the best trade-off between accuracy and computational cost? |
| RQ4 | Do Grad-CAM visualizations show that models focus on meaningful facial regions (eyes, mouth) during correct predictions? |
| RQ5 | Do Grad-CAM attention patterns differ between the custom CNN and transfer learning models? |
| RQ6 | What do SHAP explanations reveal about positive and negative visual evidence used by each model? |
| RQ7 | Can Grad-CAM and SHAP help diagnose why certain expressions (e.g., Sad vs. Angry) are misclassified? |

---

## Dependencies

```
tensorflow>=2.12.0
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
shap
grad-cam
Pillow
```

---

## Links

- 📓 **Kaggle Notebook:** *(link added after publication)*
- 📄 **Proposal Document:** [`docs/proposal.md`](docs/proposal.md)
- 🗃️ **Dataset:** [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

---

## Author

**[Yada Thadathanacharoen]**  
M.Sc. Student — University of Europe for Applied Sciences  
Pattern Recognition Course — Summer Semester 2026
