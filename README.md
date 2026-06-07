# Facial Expression Recognition Framework

**Comparative Analysis of Custom CNN and Transfer Learning Models with Explainable AI**

> Course: Pattern Recognition
> Institution: University of Europe for Applied Sciences
> Student: Yada Thadathanacharoen
> Submission: Phase 2 - Proposal and Code Implementation

---

## Project Overview

This project builds and compares three deep learning models for seven-class facial expression recognition on the FER-2013 dataset:

- **Custom CNN** - a compact model designed specifically for 48×48 grayscale input
- **MobileNetV2** - lightweight transfer learning model pretrained on ImageNet
- **ResNet50** - deeper residual transfer learning model pretrained on ImageNet

Explainability is provided via **Grad-CAM** (gradient-based heatmaps) and **SHAP** (pixel-level contribution maps) to verify that predictions are grounded in meaningful facial features.

---

## Results Summary

| Model | Test Accuracy | Macro F1 | Training Time | Parameters |
|---|---|---|---|---|
| **Custom CNN** | **57.80%** | **0.5165** | **41.7 min** | 1,244,135 |
| MobileNetV2 | 49.71% | 0.4444 | 156.8 min | 2,588,743 |
| ResNet50 | 40.54% | 0.3392 | 156.9 min | 24,115,079 |

The Custom CNN outperformed both transfer learning models, demonstrating that a domain-specific compact model generalises better than large ImageNet-pretrained models on low-resolution grayscale FER tasks.

---

## Dataset

**FER-2013 (Facial Expression Recognition 2013)**

| Property | Value |
|---|---|
| Total Images | 35,887 |
| Training / Validation / Test | 22,967 / 5,742 / 7,178 |
| Image Size | 48 × 48 pixels |
| Image Type | Grayscale |
| Classes | 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise |

📥 **Download the dataset:** [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

> Note: The dataset is not included in this repository. Download it from Kaggle and place it at `/kaggle/input/datasets/msambare/fer2013/` before running the notebook.

---

## Repository Structure

```
facial-expression-recognition-framework/
│
├── README.md
├── fer2013_facial_expression_recognition.ipynb   # Main notebook (all code)
│
├── outputs/
│   ├── figures/                                  # All 18 output figures (PDF)
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
│   │
│   └── tables/                                   # CSV metric tables
│       ├── Table1_Dataset_Overview.csv
│       ├── Table2_Class_Distribution.csv
│       ├── Table7_CustomCNN_PerClass_F1.csv
│       ├── Table7_MobileNetV2_PerClass_F1.csv
│       ├── Table7_ResNet50_PerClass_F1.csv
│       └── Table8_Model_Comparison.csv
│
└── proposal/
    └── proposal_phase2_final.docx                # Phase 2 proposal document
```

---

## How to Run

### Option 1 - Run on Kaggle (Recommended)

The notebook is available publicly on Kaggle with the dataset already attached:

🔗 **[Open Kaggle Notebook](https://www.kaggle.com/code/yadaaaat/facial-expression-recognition-framework)**

1. Click **Copy & Edit** to fork the notebook to your account
2. Make sure the FER-2013 dataset is attached under **Add Data**
3. Enable GPU: **Settings → Accelerator → GPU T4 x2** or **P100**
4. Click **Run All**

### Option 2 - Run Locally

**Requirements**
```bash
pip install tensorflow keras scikit-learn matplotlib seaborn shap opencv-python pillow pandas numpy
```

**Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/yadatdtn-ctrl/facial-expression-recognition-framework.git
   cd facial-expression-recognition-framework
   ```

2. Download the FER-2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place it at:
   ```
   /kaggle/input/datasets/msambare/fer2013/
   ```
   Or update the `ROOT_DIR` path in **Cell 3** of the notebook to match your local path.

3. Open and run the notebook:
   ```bash
   jupyter notebook fer2013_facial_expression_recognition.ipynb
   ```

4. Run all cells in order (Cell 1 → Cell 32)

> **GPU strongly recommended.** Training all three models takes approximately 6 hours on CPU. On Kaggle GPU (P100) it takes ~6 hours total.

---

## Code Structure

The notebook is organized into 32 clearly commented cells:

| Cells | Content |
|---|---|
| 1–3 | Library installation, imports, configuration |
| 4–7 | Dataset loading, class distribution, sample images |
| 8–10 | Preprocessing pipeline, data augmentation, class weights |
| 11–17 | Custom CNN - build, train, evaluate, confusion matrix, classification report |
| 18–22 | MobileNetV2 - build, train (2-phase), evaluate, confusion matrix |
| 23–27 | ResNet50 - build, train (2-phase), evaluate, confusion matrix, ROC curves |
| 28–29 | Final comparison table and bar charts |
| 30–31 | Grad-CAM visualizations for all 3 models |
| 32 | SHAP explanations for all 3 models |

---

## Output Figures

All figures are saved as PDFs in `outputs/figures/`. Key outputs include:

- **Training curves** (Fig 3a–3c) - accuracy and loss per epoch
- **Confusion matrices** (Fig 4a–4c) - raw counts and normalized row %
- **Model comparison** (Fig 5) - classification metrics and training time
- **Grad-CAM heatmaps** (Fig 6a–6c) - correct vs incorrect prediction attention maps
- **SHAP explanations** (Fig 7a–7c) - pixel-level positive/negative evidence maps
- **ROC curves** (Fig 8a–8c) - per-class and macro/micro AUC

---

## References

1. Goodfellow et al. (2013). Challenges in representation learning. ICML 2013. https://arxiv.org/abs/1307.0414
2. He et al. (2016). Deep residual learning for image recognition. CVPR 2016. https://arxiv.org/abs/1512.03385
3. Sandler et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. CVPR 2018. https://arxiv.org/abs/1801.04381
4. Selvaraju et al. (2017). Grad-CAM: Visual explanations via gradient-based localization. ICCV 2017. https://arxiv.org/abs/1610.02391
5. Lundberg & Lee (2017). A unified approach to interpreting model predictions. NeurIPS 2017. https://arxiv.org/abs/1705.07874
6. Pramerdorfer & Kampel (2016). FER using CNNs: State of the art. arXiv. https://arxiv.org/abs/1612.02903
7. Li & Deng (2022). Deep facial expression recognition: A survey. IEEE TAFFC. https://doi.org/10.1109/TAFFC.2020.2981446
8. Tran et al. (2021). Deep-Emotion: FER using attentional CNNs. Sensors. https://doi.org/10.3390/s21113046
9. Tutuianu et al. (2024). Benchmarking deep FER. Engineering Applications of AI. https://doi.org/10.1016/j.engappai.2024.108896
10. Mandave & Patil (2025). MobileNetV2 vs EfficientNet-B0 for FER. Information Dynamics and Applications. https://doi.org/10.56578/ida040401

---

*University of Europe for Applied Sciences | Pattern Recognition | June 2026*
