# Facial Expression Recognition Framework

A comparative CNN-based image classification project for recognizing seven facial expressions using the FER-2013 dataset. Built as part of the Pattern Recognition course at the University of Europe for Applied Sciences.

---

## Project Overview

This project designs, trains, and evaluates three deep learning models for facial expression recognition and explains their predictions using Grad-CAM and SHAP visualizations.

**Seven emotion classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

**Three models compared:**
- Custom CNN — designed from scratch for 48×48 grayscale images
- MobileNetV2 — transfer learning from ImageNet
- ResNet50 — transfer learning from ImageNet

---

## Dataset

**FER-2013** — Facial Expression Recognition 2013  
Source: https://www.kaggle.com/datasets/msambare/fer2013

| Split | Images |
|-------|--------|
| Training | 28,709 |
| Validation | 5,741 (80/20 split from train) |
| Test | 7,178 |
| **Total** | **35,887** |

**Class distribution (training set):**

| Emotion | Samples | % |
|---------|---------|---|
| Angry | 3,995 | 13.9% |
| Disgust | 436 | 1.5% |
| Fear | 4,097 | 14.3% |
| Happy | 7,215 | 25.1% |
| Neutral | 4,965 | 17.3% |
| Sad | 4,830 | 16.8% |
| Surprise | 3,171 | 11.0% |

> Note: Class imbalance is handled using balanced class weights during training.

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Custom CNN | 59.64% | 0.5453 | 0.6072 | 0.5494 | 24.1 min |
| MobileNetV2 | 50.64% | 0.4409 | 0.4733 | 0.4422 | 164.8 min |
| ResNet50 | ~40% | 0.35 | 0.40 | 0.35 | 118.0 min |

**Key finding:** The Custom CNN outperformed both transfer learning models. This is attributed to the domain gap between ImageNet (color photographs) and FER-2013 (48×48 grayscale faces).

---

## Repository Structure

```
facial-expression-recognition-framework/
│
├── notebook/
│   └── fer2013-facial-expression-recognition.ipynb
│
├── docs/
│   └── proposal.md
│
├── outputs/
│   ├── figures/
│   │   ├── Fig1_Class_Distribution.pdf
│   │   ├── Fig2_Sample_Images.pdf
│   │   ├── Fig3a_Training_Curves_Custom_CNN.pdf
│   │   ├── Fig3b_Training_Curves_MobileNetV2.pdf
│   │   ├── Fig3c_Training_Curves_ResNet50.pdf
│   │   ├── Fig4a_Confusion_Matrix_Custom_CNN.pdf
│   │   ├── Fig4b_Confusion_Matrix_MobileNetV2.pdf
│   │   ├── Fig4c_Confusion_Matrix_ResNet50.pdf
│   │   ├── Fig5_Model_Comparison.pdf
│   │   ├── Fig6a_GradCAM_Custom_CNN.pdf
│   │   ├── Fig6b_GradCAM_MobileNetV2.pdf
│   │   ├── Fig6c_GradCAM_ResNet50.pdf
│   │   ├── Fig7a_SHAP_Custom_CNN.pdf
│   │   ├── Fig7b_SHAP_MobileNetV2.pdf
│   │   ├── Fig7c_SHAP_ResNet50.pdf
│   │   ├── Fig8a_ROC_Custom_CNN.pdf
│   │   ├── Fig8b_ROC_MobileNetV2.pdf
│   │   └── Fig8c_ROC_ResNet50.pdf
│   └── tables/
│       ├── Table7_CustomCNN_PerClass_F1.csv
│       ├── Table7_MobileNetV2_PerClass_F1.csv
│       ├── Table7_ResNet50_PerClass_F1.csv
│       └── Table8_Model_Comparison.csv
│
└── README.md
```

---

## How to Run

### Option 1 — Google Colab (Recommended)

1. Open the notebook in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run Cell 0 to download the FER-2013 dataset using your Kaggle API key
4. Click Runtime → Run All
5. Wait approximately 6-8 hours for full training
6. Download `fer2013_outputs.zip` from the outputs folder

### Option 2 — Kaggle Notebook

1. Go to kaggle.com and open the notebook
2. Enable GPU accelerator in Settings
3. Add the FER-2013 dataset: https://www.kaggle.com/datasets/msambare/fer2013
4. Click Run All
5. Download outputs from the Output tab

---

## Requirements

```
tensorflow>=2.10.0
numpy
pandas
matplotlib
seaborn
scikit-learn
shap
Pillow
opencv-python
```

Install all dependencies:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn shap Pillow opencv-python
```

---

## Methodology

### Preprocessing
- Custom CNN: 48×48 grayscale images, normalized to [0, 1]
- Transfer models: 224×224 RGB images, normalized to [0, 1]
- Training augmentation: horizontal flip, rotation (±15°), zoom (10%), brightness adjustment
- Validation/test: normalization only

### Training Setup
- Optimizer: Adam
- Loss: Categorical crossentropy
- Custom CNN learning rate: 0.001
- Transfer learning rate: 0.0001
- Batch size: 32
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

### Transfer Learning Strategy
- Phase 1 — Feature extraction: base model frozen, only classification head trained
- Phase 2 — Fine-tuning: last 30 layers unfrozen, trained with lower learning rate (LR/10)

### Explainability
- **Grad-CAM**: Visualizes which facial regions activate each model's decision
- **SHAP**: Estimates pixel-level positive and negative contributions to predictions

---

## Research Questions

- RQ1: How accurately can a custom CNN classify facial expressions from FER-2013?
- RQ2: Does transfer learning improve classification performance over a custom CNN?
- RQ3: Which model provides the best balance between accuracy and computational cost?
- RQ4: Which facial regions does each model focus on according to Grad-CAM?
- RQ5: Do SHAP explanations reveal meaningful pixel-level evidence for predictions?
- RQ6: What are the main failure patterns and misclassification patterns across models?

---

## Course Information

**Course:** Pattern Recognition  
**Institution:** University of Europe for Applied Sciences  
**Phase:** Phase 2 — Proposal and Code Implementation  
**Due Date:** June 7, 2026
