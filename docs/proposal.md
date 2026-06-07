[proposal.md](https://github.com/user-attachments/files/28229783/proposal.md)
# Phase 2 Proposal Document

## Comparative Analysis of Custom CNN and Transfer Learning for Facial Expression Recognition

| | |
|---|---|
| **Course** | Pattern Recognition |
| **Institution** | University of Europe for Applied Sciences |
| **Student** | Yada Thadathanacharoen |
| **Submission** | Phase 2 - Proposal and Code Implementation |
| **Due Date** | June 7, 2026 |

---

## Section 1 - Project Title

**Facial Expression Recognition Framework: A Comparative Analysis of Custom CNN and Transfer Learning Models with Explainable AI**

---

## Section 2 - Background and Motivation

Facial expression recognition (FER) is an emotional processing problem used in healthcare systems (e.g., mental health screening), education (student engagement), transportation (driver safety), and human-computer interaction. Automated face detection from images can help support systems in responding to users' emotions and allow experts to identify facial expressions without the need for voice and speech. Convolutional neural networks (CNNs) are a widely used strategy in FER because they can learn directly from the hierarchical structure of image features from raw pixels [7]. However, a key question remains regarding this. For datasets like FER-2013 (48x48 monochrome images), should smaller CNN models be built for this problem, or should transfer learning be relied upon for larger models trained in high-resolution RGB image scenarios such as ImageNet? The accuracy reported in FER-2013 is far from perfect, largely due to the trade-offs between custom CNNs and transfer learning at such a low resolution, making them difficult to understand.

Transfer learning models, such as MobileNetV2 and ResNet50, can efficiently handle pre-trained features but suffer from domain mismatch issues: FER-2013 uses 48x48 pixel monochrome images, leading to noise and class imbalance problems, while these models predict the use of 224x224 pixel RGB images [3][2]. For 48x48 pixel black and white images, smaller CNN models can be specifically built, potentially offering better performance with fewer parameters and lower computational costs. Furthermore, the datasets present several challenges: Disgust has only 436 samples, while Happy has 7,215 samples. Labels are collected from people, and facial obscuration and positioning differ.

A major disadvantage of many depth models, aside from accuracy, is the difficulty in explaining them. Understanding the reasoning behind certain emotion predictions by a model is almost as important as the prediction itself in human-centered applications such as FER. SHAP can demonstrate relevance at the pixel level [5], while Grad-CAM can reveal which facial areas the model is using [4]. When used together, this helps in diagnosing misclassifications and verifying whether the model is truly focusing on important facial features.

This study considered custom CNNs, MobileNetV2, and ResNet50, all tested on the FER-2013 dataset [1] to understand how each model learns. We opted for SHAP and Grad-CAM for Explainable AI. The primary objective was to determine which method provides the best accuracy, cost-effectiveness, and ease of understanding. We view this final system as a tool to assist experts in making decisions, not to replace them.

---

## Section 3 - Problem Statement

Facial expression recognition from static images poses a difficult multi-class classification challenge. High intra-class variability exists (the same emotion can show up differently to individuals of varying ages, cultures, and behaviors) and significant interclass similarity (e.g., fear versus surprise, anger versus disgust), especially when working with low-resolution images. FER-2013 meets even more difficulties, since it consists of 48x48 pixel grayscale facial images, limiting the subtle detail that might help distinguish similar emotions [1].

The FER-2013 training dataset is highly imbalanced: the "happy" emotion has 7,215 samples, while the "disgusted" emotion includes just 436 samples, resulting in a ratio of 16.5 to 1 under standard training conditions. This imbalance causes the model to prioritize the majority class and perform poorly on minority emotions. Although several FER approaches utilizing CNNs have been proposed, direct comparisons between domain-specific custom CNNs and transfer learning models (MobileNetV2, ResNet50) on low-resolution monochrome images remain limited [3][2]. The combined applications of Grad-CAM and SHAP for diagnostics, both accurate and inaccurate predictions on FER-2013, have also not been extensively explored [4][5].

This project aims to address this gap by simultaneously training three CNN models (custom CNN, MobileNetV2, ResNet50) on the FER-2013 dataset under the same conditions, reporting both overall and per-class metrics, and using Grad-CAM and SHAP to examine the behavior of each model, particularly in minority mood. A key question is whether the compact custom CNN can outperform the larger pre-trained models in this specific low-resolution monochrome image task.

---

## Section 4 - Selected Dataset and Dataset Link

| Field | Details |
|---|---|
| Dataset Name | FER-2013 (Facial Expression Recognition 2013) |
| Source / Kaggle Link | https://www.kaggle.com/datasets/msambare/fer2013 |
| Original Source | Introduced at ICML 2013 (Goodfellow et al., 2013) |
| License | Publicly available for academic use |
| Format | Grayscale images in folder structure (train/test) |

FER-2013 is a performance measurement standard dataset for FER and is suitable for this project because it is publicly accessible, has a suitable size, and clearly defines training and testing data sets.

---

## Section 5 - Dataset Description

### 5.1 Overview

| Property | Value |
|---|---|
| Total Images | 35,887 |
| Training Images | 28,709 |
| Test Images | 7,178 |
| Image Size | 48 x 48 pixels |
| Image Type | Grayscale (1 channel) |
| Number of Classes | 7 emotions |

The dataset consists only of facial images, each identified by one of seven basic emotions and resized to 48x48 pixels. This size results in small and uniform images, but also makes the process more difficult, because much of the subtle facial detail is lost.

### 5.2 Emotion Classes and Training Distribution

| Emotion | Training Samples | Percentage |
|---|---|---|
| Angry | 3,995 | 13.9% |
| Disgust | 436 | 1.5% |
| Fear | 4,097 | 14.3% |
| Happy | 7,215 | 25.1% |
| Neutral | 4,965 | 17.3% |
| Sad | 4,830 | 16.8% |
| Surprise | 3,171 | 11.0% |
| **TOTAL** | **28,709** | **100%** |

The emotion "happy" was the most prevalent in the dataset, while the emotion "disgusted" was quite rare. This imbalance significantly impacts training and assessment and needs to be addressed clearly. The 16.5-to-1 ratio between Happy and Disgust suggests the model can achieve good overall accuracy, while hardly predicting Disgust. Without class weighting or other balancing techniques, minority emotions will remain poorly learned. In this project, class weights are calculated from the training dataset using `sklearn.utils.class_weight.compute_class_weight` and implemented during training for all three models.

### 5.3 Dataset Challenges

The FER-2013 dataset has several known challenges:

- Low spatial resolution (48x48 pixels), which makes it difficult to capture face information.
- Grayscale format, which removes color information that is useful to humans (for example, redness or paleness).
- Noisy labels, since annotations were collected via crowdsourcing rather than experts.
- Variations in pose and occlusion, such as a side view or a face partially covered by a hand and hair.
- High intra-class and inter-class variability, with similar facial patterns sometimes corresponding to different emotions.

As a result, even high-performance CNN models often achieve accuracy of only about 60% on the FER-2013 dataset, which is significantly lower than on more controlled laboratory datasets like CK+.

---

## Section 6 - Research Questions

| RQ | Research Question |
|---|---|
| RQ1 | How accurately can CNN-based models perform seven-class facial expression recognition on FER-2013? |
| RQ2 | Which model family - custom CNN or transfer learning (MobileNetV2, ResNet50) — offers the best balance between predictive performance and computational efficiency? |
| RQ3 | How do preprocessing, data augmentation, and class-imbalance handling influence robustness and per-class performance? |
| RQ4 | Do Grad-CAM explanations show that the trained models focus on meaningful facial regions such as eyes, mouth, and eyebrows? |
| RQ5 | What failure patterns and practical risks appear when deploying this framework in real human-centered settings? |

---

## Section 7 - Expected and Actual Results

This section summarizes the expectations based on the pre-training research and then briefly compares them to the results obtained from the actual experiments. At first, transfer learning models were expected to outperform the custom-CNNs. However, final experiments showed the opposite; the custom-built CNNs achieved the best test accuracy and macro-F1 scores on the FER-2013 dataset.

### 7.1 Expected Classification Performance (Before Experiments)

Based on prior work on FER-2013, the following performance ranges were expected:

| Model | Expected Accuracy | Expected Macro-F1 |
|---|---|---|
| Custom CNN (from scratch) | 63–70% | 0.55–0.65 |
| MobileNetV2 (transfer learning) | 65–72% | 0.65–0.70 |
| ResNet50 (transfer learning) | 65–73% | 0.62–0.70 |

These ranges reflect reported results for deep CNN and transfer learning techniques on FER-2013. It was assumed that MobileNetV2 and ResNet50, having more extensive pretrained features, would achieve greater accuracy than a smaller custom CNN.

### 7.2 Actual Classification Performance

After running the full experiments and evaluating on the FER-2013 test set, the results were:

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| **Custom CNN** | **57.80%** | **0.5233** | **0.5641** | **0.5165** |
| MobileNetV2 | 49.71% | 0.4432 | 0.4682 | 0.4444 |
| ResNet50 | 40.54% | 0.3586 | 0.4208 | 0.3392 |

Opposite to the original assumption, the custom CNN achieved the best overall results, while both transfer learning models performed poorly, indicating that the chosen fine-tuning strategy does not completely bridge the gap between ImageNet (high-resolution RGB) and FER-2013 (low-resolution grayscale).

### 7.3 Expected Per-Class Performance

| Emotion | Expected F1 | Reason |
|---|---|---|
| Happy | 0.85–0.92 | Largest class; visually distinct |
| Neutral | 0.60–0.75 | Well-represented; less confusion |
| Angry | 0.50–0.65 | High intra-class variation; confusable with Disgust |
| Sad | 0.50–0.65 | Overlaps with Neutral and Fear |
| Fear | 0.30–0.50 | Similar to Surprise and Sad |
| Disgust | 0.20–0.40 | Severely underrepresented; hard to learn |
| Surprise | 0.55–0.70 | Distinct, but overlaps with Fear |

### 7.3b Actual Per-Class F1-Score Results

| Emotion | Custom CNN F1 | MobileNetV2 F1 | ResNet50 F1 |
|---|---|---|---|
| Angry | 0.4955 | 0.3794 | 0.1535 |
| Disgust | 0.4080 | 0.3194 | 0.1504 |
| Fear | 0.2265 | 0.2318 | 0.1960 |
| Happy | 0.7978 | 0.7374 | 0.6339 |
| Neutral | 0.5572 | 0.4790 | 0.4234 |
| Sad | 0.3956 | 0.3504 | 0.2655 |
| Surprise | 0.7346 | 0.6132 | 0.5518 |

---

## Section 8 - Proposed Methodology

This project follows a simple but complete pipeline: starting with dataset preparation, preprocessing, model design, training, evaluation, and explanation analysis. The concept is to treat the three models as equally as possible, so that differences in results mainly come from the model architectures themselves. Before training, the dataset is checked for corrupted or unreadable files and the class distribution is plotted (Figure 1). Class weights are then computed from the training set to reduce the impact of the strong class imbalance.

### 8.1 Preprocessing Pipeline

All images are normalized by dividing pixel values by 255 to obtain values in the range [0, 1].

| Step | Custom CNN | MobileNetV2 & ResNet50 |
|---|---|---|
| Resize | 48x48 pixels | 224x224 pixels |
| Color format | Grayscale (1 channel) | Pseudo-RGB (3 channels) |
| Normalization | Divide by 255 → [0,1] | Divide by 255 → [0,1] |
| Channel conversion | - | Replicate grayscale channel ×3 |

### 8.2 Data Augmentation

Data augmentation is only applied to the training dataset to improve robustness and reduce overfitting.

| Augmentation | Value |
|---|---|
| Horizontal Flip | 50% probability |
| Random Rotation | ±15 degrees |
| Random Zoom | ±10% |
| Random Brightness | ±20% |

No random contrast, width shift, or height shift is used, and no augmentation is applied to the validation or test sets.

### 8.3 Train / Validation / Test Split

The predefined test split from FER-2013 is kept as the final test set. The training portion from Kaggle is split into new training and validation subsets using stratified sampling, preserving the original class distributions in both sets.

| Split | Images | Percentage |
|---|---|---|
| Training | 22,967 | 64% |
| Validation | 5,742 | 16% |
| Test | 7,178 | 20% |

### 8.4 Model Development

**Model 1 - Custom CNN (from scratch)**
A compact CNN tailored to 48×48 grayscale input. Four convolutional blocks, each with two Conv2D layers and Batch Normalization, followed by MaxPooling and Dropout. Global Average Pooling replaces Flatten. Full details in Section 9.

**Model 2 - MobileNetV2 (transfer learning)**
MobileNetV2 pretrained on ImageNet loaded with `include_top=False`. Custom head: `GlobalAveragePooling2D → Dropout(0.5) → Dense(256, ReLU) → BatchNormalization → Dropout(0.3) → Dense(7, Softmax)`. Base first frozen, then last 30 layers unfrozen for fine-tuning.

**Model 3 - ResNet50 (transfer learning)**
ResNet50 pretrained on ImageNet with the same custom head. Same two-phase training strategy as MobileNetV2.

### 8.6 Training Setup and Hyperparameters

All models trained on Kaggle Notebooks with GPU acceleration (NVIDIA Tesla P100).

| Hyperparameter | Custom CNN | MobileNetV2 | ResNet50 |
|---|---|---|---|
| Optimizer | Adam | Adam | Adam |
| Initial LR | 0.001 | 0.0001 | 0.0001 |
| Loss Function | Cat. Cross-Entropy | Cat. Cross-Entropy | Cat. Cross-Entropy |
| Batch Size | 32 | 32 | 32 |
| Epochs (max) | 30 | 20 | 20 |
| Class Weights | Yes | Yes | Yes |

Callbacks for all models:
- `EarlyStopping` (patience=5, monitor val_loss)
- `ModelCheckpoint` (save best by val_accuracy)
- `ReduceLROnPlateau` (factor=0.5, patience=3)

---

## Section 9 - CNN Model Design (Custom CNN)

### 9.1 Architecture Rationale

The custom CNN is specifically designed for 48×48 grayscale FER-2013 images, aiming to balance representational power and computational cost. It uses four convolutional blocks, each containing two Conv2D layers with equal filters (32, 64, 128, 256), followed by Batch Normalization and MaxPooling. A Dropout layer with rate 0.25 is applied after each pooling block, and Global Average Pooling is used instead of Flatten before a small dense head.

In the classifier head, Dropout(0.5) is applied before a Dense(256, ReLU) layer with Batch Normalization, followed by Dropout(0.3) and the final Dense(7, Softmax) output layer.

*See Appendix A for the full layer table.*

---

## Section 10 - Transfer Learning Models

### 10.1 Strategy

MobileNetV2 and ResNet50 are both loaded with ImageNet-pretrained weights and `include_top=False`. A shared custom head is attached: `GlobalAveragePooling2D → Dropout(0.5) → Dense(256, ReLU) → BatchNormalization → Dropout(0.3) → Dense(7, Softmax)`. Both follow a two-phase training strategy.

### 10.2 Model Comparison

| Property | MobileNetV2 | ResNet50 |
|---|---|---|
| Original Dataset | ImageNet (1.2M, 1000 classes) | ImageNet |
| Architecture Type | Inverted residuals, depthwise separable convs | Deep residual network with skip connections |
| Total Parameters | ~3.4M | ~25M |
| Input Size | 224×224×3 | 224×224×3 |
| Base Weights (Phase 1) | Frozen | Frozen |
| Fine-Tuning (Phase 2) | Last 30 layers unfrozen | Last 30 layers unfrozen |
| Optimizer | Adam (lr=0.0001) | Adam (lr=0.0001) |
| Epochs (max) | 20 | 20 |

### 10.3 Two-Phase Training Strategy

**Phase 1 - Feature Extraction**
- Base model layers are frozen (non-trainable)
- Only the custom head is trained
- Slightly higher learning rate to quickly adapt to FER-2013

**Phase 2 - Fine-Tuning**
- Base model set to trainable
- All layers except the last 30 are re-frozen
- Lower learning rate to avoid destroying pretrained features

### 10.4 Handling the Grayscale-to-RGB Mismatch

- Resize 48×48 grayscale images to 224×224
- Replicate the single grayscale channel three times → pseudo-RGB: [gray, gray, gray]
- Normalize pixel values to [0, 1]

### 10.5 Justification for Model Selection

| Model | Why Selected |
|---|---|
| MobileNetV2 | Lightweight transfer learning model for mobile/edge devices; efficient depthwise separable convolutions |
| ResNet50 | Deeper residual model with high capacity; common transfer learning baseline |

---

## Section 11 - Evaluation Metrics

### 11.1 Quantitative Metrics

| Metric | Method | Purpose |
|---|---|---|
| Accuracy | Correct / Total | Overall performance |
| Precision (macro) | `classification_report` | Quality of positive predictions |
| Recall (macro) | `classification_report` | Coverage of true positives |
| F1-Score (macro) | `classification_report` | Balanced measure with imbalance |
| Per-class F1 | F1 per emotion | Minority class performance |
| Confusion Matrix | `confusion_matrix` | Visual error patterns |
| Training Time | Wall-clock (Python `time`) | Computational cost |
| Parameter Count | `model.count_params()` | Model complexity |

### 11.2 Explainability Metrics

| Metric | Method | Purpose |
|---|---|---|
| Grad-CAM quality | Visual inspection of heatmaps | Check focus on meaningful facial regions |
| SHAP interpretability | Pixel-level contribution maps | Understand which regions support predictions |
| Misclassification diagnosis | Grad-CAM on incorrect predictions | Identify visual patterns in errors |

### 11.3 Final Model Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Parameters |
|---|---|---|---|---|---|---|
| **Custom CNN** | **57.80%** | **0.5233** | **0.5641** | **0.5165** | **41.68 min** | **1,244,135** |
| MobileNetV2 | 49.71% | 0.4432 | 0.4682 | 0.4444 | 156.76 min | 2,588,743 |
| ResNet50 | 40.54% | 0.3586 | 0.4208 | 0.3392 | 156.9 min | 24,115,079 |

---

## Section 12 - Figures, Tables, and Files

### 12.1 Figures

All figures are saved as PDFs in `outputs/figures/`.

| Figure | Description | Filename |
|---|---|---|
| Fig 1 | Class distribution bar chart | `Fig1_Class_Distribution.pdf` |
| Fig 2 | Sample images per class | `Fig2_Sample_Images.pdf` |
| Fig 3a | Training curves - Custom CNN | `Fig3a_Training_Curves_Custom_CNN.pdf` |
| Fig 3b | Training curves - MobileNetV2 | `Fig3b_Training_Curves_MobileNetV2.pdf` |
| Fig 3c | Training curves - ResNet50 | `Fig3c_Training_Curves_ResNet50.pdf` |
| Fig 4a | Confusion matrix - Custom CNN | `Fig4a_Confusion_Matrix_Custom_CNN.pdf` |
| Fig 4b | Confusion matrix - MobileNetV2 | `Fig4b_Confusion_Matrix_MobileNetV2.pdf` |
| Fig 4c | Confusion matrix - ResNet50 | `Fig4c_Confusion_Matrix_ResNet50.pdf` |
| Fig 5 | Model comparison bar chart | `Fig5_Model_Comparison.pdf` |
| Fig 6a | Grad-CAM - Custom CNN | `Fig6a_GradCAM_Custom_CNN.pdf` |
| Fig 6b | Grad-CAM - MobileNetV2 | `Fig6b_GradCAM_MobileNetV2.pdf` |
| Fig 6c | Grad-CAM - ResNet50 | `Fig6c_GradCAM_ResNet50.pdf` |
| Fig 7a | SHAP - Custom CNN | `Fig7a_SHAP_Custom_CNN.pdf` |
| Fig 7b | SHAP - MobileNetV2 | `Fig7b_SHAP_MobileNetV2.pdf` |
| Fig 7c | SHAP - ResNet50 | `Fig7c_SHAP_ResNet50.pdf` |
| Fig 8a | ROC curves - Custom CNN | `Fig8a_ROC_Custom_CNN.pdf` |
| Fig 8b | ROC curves - MobileNetV2 | `Fig8b_ROC_MobileNetV2.pdf` |
| Fig 8c | ROC curves - ResNet50 | `Fig8c_ROC_ResNet50.pdf` |

### 12.2 Tables

All tables are saved as CSVs in `outputs/tables/`.

| Table | Description | Filename |
|---|---|---|
| Table 1 | Dataset overview | Proposal Section 5 |
| Table 2 | Class distribution | Proposal Section 5 |
| Table 3 | Custom CNN architecture | Proposal Section 9 / Appendix A |
| Table 4 | Transfer learning comparison | Proposal Section 10 |
| Table 5 | Training hyperparameters | Proposal Section 8 |
| Table 6 | Model comparison metrics | `Table8_Model_Comparison.csv` |
| Table 7 | Per-class F1 - Custom CNN | `Table7_CustomCNN_PerClass_F1.csv` |
| Table 8 | Per-class F1 - MobileNetV2 | `Table7_MobileNetV2_PerClass_F1.csv` |
| Table 9 | Per-class F1 - ResNet50 | `Table7_ResNet50_PerClass_F1.csv` |

---

## References

### Foundational Papers

1. Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., & Bengio, Y. (2013). Challenges in representation learning: A report on three machine learning contests. *ICML 2013*. https://arxiv.org/abs/1307.0414
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778. https://arxiv.org/abs/1512.03385
3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*, 4510–4520. https://arxiv.org/abs/1801.04381
4. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*, 618–626. https://arxiv.org/abs/1610.02391
5. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30, 4765–4774. https://arxiv.org/abs/1705.07874

### FER-Specific Papers

6. Pramerdorfer, C., & Kampel, M. (2016). Facial expression recognition using convolutional neural networks: State of the art. *arXiv*. https://arxiv.org/abs/1612.02903
7. Li, S., & Deng, W. (2022). Deep facial expression recognition: A survey. *IEEE Transactions on Affective Computing*, 13(3), 1195–1215. https://doi.org/10.1109/TAFFC.2020.2981446
8. Tran, M., et al. (2021). Deep-Emotion: Facial expression recognition using attentional convolutional networks. *Sensors*, 21(11), 3046. https://doi.org/10.3390/s21113046
9. Tutuianu, G. I., Liu, Y., Alamaki, A., & Kauttonen, J. (2024). Benchmarking deep facial expression recognition. *Engineering Applications of Artificial Intelligence*, 133, 108896. https://doi.org/10.1016/j.engappai.2024.108896
10. Mandave, D. D., & Patil, L. V. (2025). A statistically rigorous comparison of MobileNetV2 and EfficientNet-B0 for facial expression recognition. *Information Dynamics and Applications*, 4(4). https://doi.org/10.56578/ida040401

---

## Appendix A - Custom CNN Architecture (Full Layer Table)

| Layer | Output Shape | Filters/Units | Activation | Notes |
|---|---|---|---|---|
| Input | 48×48×1 | - | - | - |
| Conv_1a + Conv_1b | 48×48×32 | 32 @ 3×3 | ReLU | Batch Norm after |
| MaxPool_1 + Dropout | 24×24×32 | 2×2 pool | - | rate=0.25 |
| Conv_2a + Conv_2b | 24×24×64 | 64 @ 3×3 | ReLU | Batch Norm after |
| MaxPool_2 + Dropout | 12×12×64 | 2×2 pool | - | rate=0.25 |
| Conv_3a + Conv_3b | 12×12×128 | 128 @ 3×3 | ReLU | Batch Norm after |
| MaxPool_3 + Dropout | 6×6×128 | 2×2 pool | - | rate=0.25 |
| Conv_4a + Conv_4b | 6×6×256 | 256 @ 3×3 | ReLU | Batch Norm after |
| MaxPool_4 + Dropout | 3×3×256 | 2×2 pool | - | rate=0.25 |
| GlobalAvgPool | 256 | - | - | Replaces Flatten |
| Dropout_5 + Dense_1 | 256 | 256 units | ReLU | rate=0.5, BatchNorm |
| Dropout_6 + Output | 7 | 7 units | Softmax | rate=0.3 |

---

*University of Europe for Applied Sciences | Pattern Recognition | June 2026*
