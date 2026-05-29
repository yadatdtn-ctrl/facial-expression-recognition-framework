[proposal.md](https://github.com/user-attachments/files/28229783/proposal.md)
# Phase 2 Proposal Document
## Comparative Analysis of Custom CNN and Transfer Learning for Facial Expression Recognition

> **Course:** Pattern Recognition  
> **Institution:** University of Europe for Applied Sciences  
> **Student:** [Yada Thadathanacharoen]  
> **Submission:** Phase 2 — Proposal and Code Implementation  
> **Due Date:** June 7, 2026  

---

## Section 1 — Project Title

**Facial Expression Recognition Framework: A Comparative Analysis of Custom CNN and Transfer Learning Models with Explainable AI**

---

## Section 2 — Background and Motivation

<!-- 
WRITING INSTRUCTIONS FOR THIS SECTION:
Use Perplexity to research: "facial expression recognition CNN deep learning applications 2022 2023"
Then come back to Claude AI with those findings to write this section academically.
Target: 150–200 words.

POINTS TO COVER:
- Why facial expression recognition matters (real-world applications)
- Why CNNs are now the standard approach for this task
- Why comparing custom CNN vs transfer learning is scientifically valuable
- Why explainability (Grad-CAM / SHAP) is critical, especially in human-centered AI
-->

**Facial expression recognition (FER) is a fundamental problem in affective computing with significant practical applications across healthcare (mental health screening), education (student engagement monitoring), transportation (driver safety), and human-computer interaction. Automatically detecting emotional states from facial images has direct social value: research shows 55-60% of human communication occurs through facial expressions.

### The Architecture Challenge
Convolutional Neural Networks (CNNs) have become the standard approach for FER because they automatically learn hierarchical features from raw pixels. However, a critical unresolved question remains: should practitioners design custom CNN architectures tailored specifically to the FER-2013 domain (48×48 grayscale), or leverage transfer learning from pretrained models trained on large RGB datasets like ImageNet? State-of-the-art systems achieve only 64-75% accuracy on FER-2013 — below human-level performance of 65-70% — partly because this architectural trade-off is underexplored.

### Transfer Learning Trade-offs
Transfer learning models using pretrained networks (MobileNetV2, ResNet50) bring rich features learned from millions of images but face domain mismatch with low-resolution grayscale input. Custom CNNs optimized specifically for 48×48 pixels may achieve comparable accuracy with far fewer parameters, which is crucial for real-time and edge deployments. The FER-2013 dataset's inherent constraints illustrate why this comparison matters: severe class imbalance (Disgust: 436 samples vs. Happy: 7,215), noisy crowdsourced labels, and pose/occlusion variation.

### Explainability as Essential
Beyond raw accuracy, a critical limitation of many deployed CNN models is opacity — the inability to explain why a model made a particular prediction. In human-centered applications such as FER, understanding why the system predicts a specific emotion is as important as accuracy itself. Explainable AI methods provide solutions: Grad-CAM visualizes which image regions drive predictions, while SHAP quantifies pixel-level importance. Together, these methods enable practitioners to diagnose misclassifications and build trust in deployed systems.

### Project Objective
This project systematically compares custom CNN architectures against transfer learning models (MobileNetV2, ResNet50) on FER-2013, applies Grad-CAM and SHAP to understand what features each model learns, and identifies which approach best balances accuracy, computational efficiency, and interpretability for practical facial expression recognition systems. The final system is framed as a decision-support framework that assists human experts in emotion recognition rather than replacing human judgment.

---

## Section 3 — Problem Statement

<!-- 
WRITING INSTRUCTIONS:
Be specific. State the exact problem, what is unknown, and what this project aims to address.
Target: 100–130 words.
-->

Facial expression recognition from static images is a challenging multi-class classification problem due to several inherent factors: high intra-class variability (e.g., different intensities of the same expression across ages and cultures), inter-class similarity (e.g., Fear vs. Surprise, Angry vs. Disgust), and low image resolution in FER-2013 (48×48 pixels, grayscale). Additionally, the training dataset exhibits severe class imbalance: Happy contains 7,215 samples while Disgust contains only 436, creating a 16.5-to-1 disparity that causes models to ignore minority classes.

While numerous CNN-based approaches have demonstrated strong performance on this task, **systematic comparisons between custom CNN architectures and transfer learning models (MobileNetV2, ResNet50) remain underexplored in the context of low-resolution grayscale images.** Furthermore, the application of Grad-CAM and SHAP to diagnose misclassifications — specifically to identify which facial regions are responsible for incorrect predictions — has received limited attention in FER research.

This project addresses these gaps by systematically comparing three CNN architectures (custom CNN, MobileNetV2, ResNet50) on FER-2013 under controlled conditions, using per-class evaluation metrics and explainability methods (Grad-CAM and SHAP) to understand what each model learns and how to improve performance on underrepresented emotion classes.

---

## Section 4 — Selected Dataset and Dataset Link

| Field | Details |
|-------|---------|
| **Dataset Name** | FER-2013 (Facial Expression Recognition 2013) |
| **Source / Kaggle Link** | https://www.kaggle.com/datasets/msambare/fer2013 |
| **Original Source** | Introduced at ICML 2013 (Goodfellow et al., 2013) |
| **License** | Publicly available for academic use |
| **Format** | Grayscale images in folder structure (train/test) |

---

## Section 5 — Dataset Description

### 5.1 Overview

| Property | Value |
|----------|-------|
| Total Images | 35,887 |
| Training Images | 28,709 |
| Test Images | 3,589 |
| Image Size | 48 × 48 pixels |
| Image Type | Grayscale (1 channel) |
| Number of Classes | 7 emotions |

### 5.2 Emotion Classes and Training Distribution

| Emotion | Training Samples | Percentage |
|---------|-----------------|------------|
| Angry | 3,995 | 13.9% |
| Disgust | 436 | 1.5% |
| Fear | 4,097 | 14.3% |
| Happy | 7,215 | 25.1% |
| Neutral | 4,965 | 17.3% |
| Sad | 4,830 | 16.8% |
| Surprise | 3,171 | 11.0% |
| **TOTAL** | **28,709** | **100%** |

### 5.3 Class Imbalance Analysis

The FER-2013 dataset exhibits significant class imbalance. The "Happy" class dominates the training set with 7,215 images (25.1%), while "Disgust" is severely underrepresented with only 436 images (1.5%). This 16.5-to-1 ratio between the largest and smallest classes directly impacts model training: standard loss functions will bias models toward predicting the majority class, leading to poor performance on minority emotions. This imbalance is a key motivation for using class-weighted loss functions and stratified train/validation splits during model development.

### 5.4 Dataset Challenges

The FER-2013 dataset presents several well-documented challenges:

* **Low spatial resolution (48×48 pixels)** — Limits fine-grained facial feature capture; subtle differences between similar emotions are difficult to distinguish
* **Grayscale format** — Removes chromatic cues (e.g., redness from anger, paleness from fear) that aid human emotion recognition
* **Label noise** — Images were crowdsourced-annotated rather than expert-labeled, resulting in mislabeled or ambiguous samples
* **Pose and occlusion variation** — Real-world images include non-frontal angles and partial occlusions (hands, hair) that confound models trained on frontal faces
* **Intra-class and inter-class similarity** — Emotions manifest differently across individuals and ages; similar facial configurations can indicate different emotions

These factors explain why state-of-the-art accuracy on FER-2013 typically ranges from 64–75%, substantially lower than on controlled laboratory datasets like CK+ (~95%).

---

## Section 6 — Research Questions

| RQ | Research Question |
|----|------------------|
| **RQ1** | How accurately can CNN-based models perform facial expression recognition using the FER-2013 dataset? |
| **RQ2** | Which model family — custom CNN or transfer learning (MobileNetV2, ResNet50) — provides the best balance between predictive performance and computational efficiency for facial expression recognition? |
| **RQ3** | How do preprocessing choices, data augmentation, and class-imbalance handling influence model robustness and per-class performance on FER-2013? |
| **RQ4** | Do Grad-CAM explanations show that trained models focus on meaningful facial regions (e.g., eyes, mouth, brow) relevant to facial expression recognition? |
| **RQ5** | What are the main failure patterns, deployment limitations, and practical risks when applying this facial expression recognition framework in real human-centered settings? |

---

## Section 7 — Expected Results

<!-- 
WRITING INSTRUCTIONS:
These are PREDICTIONS written before running experiments. Required by the proposal.
State your hypotheses — you will compare these against actual results in Phase 3.
-->

Based on published benchmarks on the FER-2013 dataset 
(2021–2025), the following results are anticipated for 
each model. These predictions serve as informed hypotheses 
to be validated against actual experimental outcomes.

### 7.1 Expected Classification Performance

| Model | Expected Accuracy | Expected Macro-F1 |
|-------|------------------|-------------------|
| Custom CNN (from scratch) | 63–70% | 0.55–0.65 |
| MobileNetV2 (transfer learning) | 65–72% | 0.65–0.70 |
| ResNet50 (transfer learning) | 65–73% | 0.62–0.70 |

These ranges are consistent with published studies. 
Pramerdorfer & Kampel (2021) report 65–68% for 
VGGNet-based baselines, while Mandave & Patil (2025) 
report 71.8% accuracy and macro-F1 of 0.689 for 
MobileNetV2 with class-weighted loss. ResNet50 studies 
report 65–73% under standard fine-tuning conditions, 
with some reaching ~85% when combining private data 
or custom classification heads.

### 7.2 Expected Per-Class Performance

The FER-2013 class imbalance is expected to produce 
significant variation in per-class F1-scores:

| Emotion | Expected F1 | Reason |
|---------|------------|--------|
| Happy | 0.85–0.92 | Largest class (7,215 samples); most visually distinct |
| Neutral | 0.60–0.75 | Well-represented; limited confusion |
| Angry | 0.50–0.65 | High intra-class variation; confused with Disgust |
| Sad | 0.50–0.65 | Overlaps with Neutral and Fear |
| Fear | 0.30–0.50 | Visually similar to Surprise and Sad |
| Disgust | 0.20–0.40 | Severely underrepresented (436 samples); near-zero without class balancing |
| Surprise | 0.55–0.70 | Distinct features but confused with Fear |

Disgust and Fear are expected to be the most 
challenging classes across all three models due to 
severe class imbalance and inter-class similarity 
with other negative emotions.

### 7.3 Expected Computational Cost

| Model | Expected Training Time | Parameters (approx.) |
|-------|----------------------|---------------------|
| Custom CNN | Fastest (~30–60 min) | ~500K–2M |
| MobileNetV2 | Moderate (~1–2 hours) | ~3.4M |
| ResNet50 | Slowest (~2–3 hours) | ~25M |

Custom CNNs are expected to train approximately 2× 
faster than large pretrained models. However, transfer 
learning models are expected to converge to higher 
accuracy in fewer epochs once fine-tuning begins. 
ResNet50 with full fine-tuning is anticipated to be 
the most computationally expensive option, consistent 
with FERehab benchmarks reporting 2h 29m on an 
NVIDIA Tesla P100 (40 epochs).

### 7.4 Expected Model Trade-offs

Based on the literature, the following trade-offs 
are anticipated:

* **Custom CNN** will have the fewest parameters and 
fastest training time, making it most suitable for 
resource-constrained environments. However, it is 
expected to achieve the lowest overall accuracy 
without advanced augmentation.

* **MobileNetV2** is expected to offer the best 
accuracy-to-cost ratio — competitive accuracy with 
significantly fewer parameters than ResNet50. It is 
the most practical choice for real-time or 
edge-deployed FER systems.

* **ResNet50** is expected to achieve the highest 
absolute accuracy due to its deeper architecture and 
richer pretrained features, but at the cost of 
significantly higher training time and memory usage.

### 7.5 Expected Explainability Findings

* **Grad-CAM** is expected to show attention 
concentrated on eye and mouth regions for correctly 
classified samples. Misclassified samples (e.g., 
Disgust predicted as Angry, Fear predicted as Sad) 
are expected to show diffuse or incorrect activation 
patterns — focusing on background or irrelevant 
facial regions.

* **SHAP** is expected to confirm pixel-level 
importance concentrated around brow and mouth regions 
for high-confidence predictions, while low-confidence 
or misclassified samples will show scattered negative 
contributions across the face.

* Using **class-weighted loss** is expected to 
visibly improve minority class recall (especially 
Disgust and Fear) at a small cost to overall accuracy.

---

## Section 8 — Proposed Methodology

### 8.1 Overview Pipeline

```
Raw FER-2013 Dataset
        ↓
Data Loading & Inspection   [§8.2]
        ↓
Preprocessing & Augmentation [§8.3]
        ↓
Train / Validation / Test Split [§8.4]
        ↓
        ┌──────────────────────────────┐
        │  Model 1: Custom CNN         │  [§9]
        │  Model 2: MobileNetV2        │  [§10]
        │  Model 3: ResNet50           │  [§10]
        └──────────────────────────────┘
        ↓
Model Training & Callbacks [§8.5]
        ↓
Evaluation (Accuracy, F1, CM) [§11]
        ↓
Grad-CAM Visualization [§8.6]
        ↓
SHAP Visualization [§8.7]
        ↓
Comparative Analysis & Report
```

### 8.2 Data Loading

The FER-2013 dataset will be loaded directly from its Kaggle path (`/kaggle/input/fer2013/`). Images are stored as 48×48 pixel grayscale arrays. The data will be loaded using TensorFlow's `image_dataset_from_directory()` utility, with class labels inferred from the folder structure.

### 8.3 Preprocessing

| Step | Details |
|------|---------|
| Resizing | All images resized to **48×48** (Custom CNN) or **224×224** (MobileNetV2, ResNet50) |
| Normalization | Pixel values rescaled to **[0, 1]** by dividing by 255 |
| Grayscale → RGB | For transfer learning models: grayscale channel replicated to 3 channels |
| Class Weights | Computed using `sklearn.utils.class_weight` to address class imbalance |

### 8.4 Data Augmentation

Applied only to the **training set** to improve robustness and reduce overfitting:

| Technique | Value |
|-----------|-------|
| Random Horizontal Flip | 50% probability |
| Random Rotation | ±15 degrees |
| Random Zoom | ±10% |
| Random Brightness | ±20% |
| Random Contrast | ±15% |

### 8.5 Training Setup

| Hyperparameter | Custom CNN | MobileNetV2 | ResNet50 |
|----------------|------------|-------------|----------|
| Optimizer | Adam | Adam | Adam |
| Learning Rate | 0.001 | 0.0001 | 0.0001 |
| Loss Function | Categorical Cross-Entropy | Categorical Cross-Entropy | Categorical Cross-Entropy |
| Batch Size | 32 | 32 | 32 |
| Epochs | 30 | 20 | 20 |
| Callbacks | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau | Same | Same |
| Validation Split | 20% of training set | Same | Same |

### 8.6 Grad-CAM Implementation

Grad-CAM will be applied to each model's final convolutional layer to generate spatial heatmaps indicating which regions of the input image contributed most to the predicted class. Visualizations will be generated for:
- 5 correctly classified samples per model (one per expression group)
- 5 misclassified samples per model, with true and predicted labels annotated

### 8.7 SHAP Implementation

SHAP `GradientExplainer` will be applied to each model using a background sample of 50 training images. Explanations will be generated for 3 test images per model, showing positive (red) and negative (blue) pixel contributions to the predicted class.

---

## Section 9 — CNN Model Design (Custom CNN)

### 9.1 Architecture Rationale

The custom CNN is designed to balance representational depth with computational efficiency for 48×48 grayscale input. Four convolutional blocks progressively increase filter depth, followed by Global Average Pooling (replacing Flatten to reduce overfitting) and two dense layers.

### 9.2 Architecture Table

| Layer | Type | Output Shape | Filters / Units | Activation |
|-------|------|-------------|-----------------|------------|
| Input | Input | 48 × 48 × 1 | — | — |
| Conv_1 | Conv2D | 48 × 48 × 32 | 32 @ 3×3 | ReLU |
| BN_1 | BatchNorm | 48 × 48 × 32 | — | — |
| Pool_1 | MaxPool2D | 24 × 24 × 32 | 2×2 | — |
| Conv_2 | Conv2D | 24 × 24 × 64 | 64 @ 3×3 | ReLU |
| BN_2 | BatchNorm | 24 × 24 × 64 | — | — |
| Pool_2 | MaxPool2D | 12 × 12 × 64 | 2×2 | — |
| Conv_3 | Conv2D | 12 × 12 × 128 | 128 @ 3×3 | ReLU |
| BN_3 | BatchNorm | 12 × 12 × 128 | — | — |
| Pool_3 | MaxPool2D | 6 × 6 × 128 | 2×2 | — |
| Conv_4 | Conv2D | 6 × 6 × 256 | 256 @ 3×3 | ReLU |
| BN_4 | BatchNorm | 6 × 6 × 256 | — | — |
| GAP | GlobalAvgPool2D | 256 | — | — |
| Dropout | Dropout | 256 | rate=0.5 | — |
| Dense_1 | Dense | 256 | 256 units | ReLU |
| Dropout_2 | Dropout | 256 | rate=0.3 | — |
| Output | Dense | 7 | 7 units | Softmax |

> **→ See Figure 2:** Model design workflow diagram

---

## Section 10 — Transfer Learning Models

### 10.1 Strategy

Both MobileNetV2 and ResNet50 are loaded with **ImageNet pre-trained weights** with their top classification layers removed (`include_top=False`). The base model weights are **frozen during initial training** (feature extraction phase). A custom head is added on top consisting of: `GlobalAveragePooling2D → Dropout(0.5) → Dense(256, ReLU) → Dropout(0.3) → Dense(7, Softmax)`.

Input images are resized to **224×224×3** (grayscale channel replicated to 3 channels) to match ImageNet input expectations.

### 10.2 Model Details

| Property | MobileNetV2 | ResNet50 |
|----------|-------------|----------|
| Original Dataset | ImageNet (1.2M images, 1000 classes) | ImageNet |
| Architecture Type | Inverted Residuals + Depthwise Separable Conv | Deep Residual Networks (skip connections) |
| Total Parameters (approx.) | ~2.3M (base) | ~23.5M (base) |
| Input Size Required | 224×224×3 | 224×224×3 |
| Base Weights Frozen | Yes (initial training) | Yes (initial training) |
| Custom Head | GAP → Dropout → Dense(256) → Dense(7) | Same |
| Fine-Tuning Strategy | Unfreeze last 30 layers after initial convergence | Unfreeze last 20 layers |

### 10.3 Justification for Model Selection

MobileNetV2 was selected for its lightweight design and strong performance on mobile and embedded vision tasks — relevant to the deployment context of real-time FER systems. ResNet50 was selected as a deeper residual baseline known for strong image recognition performance. Together, they represent a lightweight-vs-deep tradeoff that directly addresses RQ3.

---

## Section 11 — Evaluation Metrics

The following metrics will be computed for each model on the **test set**:

| Metric | Formula / Method | Purpose |
|--------|-----------------|---------|
| **Accuracy** | Correct predictions / Total predictions | Overall performance |
| **Precision (macro)** | TP / (TP + FP) averaged over classes | Correctness of positive predictions |
| **Recall (macro)** | TP / (TP + FN) averaged over classes | Coverage of actual positives |
| **F1-Score (macro)** | 2 × (P × R) / (P + R) | Balanced performance, handles imbalance |
| **Confusion Matrix** | Per-class prediction matrix | Visual error pattern analysis |
| **Classification Report** | sklearn `classification_report` | Per-class breakdown |
| **Training Time** | Wall-clock seconds per epoch | Computational cost |
| **Parameter Count** | `model.count_params()` | Model complexity |
| **Grad-CAM Quality** | Qualitative (1–5 scale) | Interpretability rating |
| **SHAP Interpretability** | Qualitative (1–5 scale) | Pixel-level explanation quality |

**Final Comparison Table Format (to be filled with results):**

| Model | Accuracy | F1-Score | Training Time | Parameters | Grad-CAM Quality | SHAP Interpretability | Overall Finding |
|-------|----------|----------|---------------|------------|------------------|-----------------------|-----------------|
| Custom CNN | — | — | — | — | — | — | — |
| MobileNetV2 | — | — | — | — | — | — | — |
| ResNet50 | — | — | — | — | — | — | — |

---

## Section 12 — Expected Figures and Tables

All figures will be saved as `.png` files in the `/figures/` directory of the GitHub repository.

| Figure | Description | Directory |
|--------|-------------|-----------|
| **Figure 1** | Grid of sample images from all 7 expression classes with class labels and counts | `figures/Fig1_Dataset_Samples/` |
| **Figure 2** | Model design workflow diagram: Input → CNN/MobileNetV2/ResNet50 → Prediction → Evaluation → Grad-CAM + SHAP | `figures/Fig2_Model_Workflow/` |
| **Figure 3a–c** | Training & validation accuracy/loss curves — one plot per model (3 total) | `figures/Fig3_Training_Curves/` |
| **Figure 4a–c** | Confusion matrices — one per model, normalized and annotated with class names | `figures/Fig4_Confusion_Matrices/` |
| **Figure 5** | Grad-CAM correct prediction grid: `Original \| Custom CNN \| MobileNetV2 \| ResNet50` across 5 expression classes | `figures/Fig5_GradCAM_Correct/` |
| **Figure 6** | Grad-CAM misclassification examples: True label / Predicted label / Heatmap / Diagnosis text | `figures/Fig6_GradCAM_Misclassified/` |
| **Figure 7** | SHAP explanation grid: `Original Image \| SHAP heatmap \| Interpretation` for 3 test images per model | `figures/Fig7_SHAP_Explanations/` |

**Expected Tables:**

| Table | Description |
|-------|-------------|
| **Table 1** | Dataset description (classes, image counts, distribution) |
| **Table 2** | Custom CNN architecture layer-by-layer |
| **Table 3** | Transfer learning model comparison (MobileNetV2 vs ResNet50) |
| **Table 4** | Evaluation results — accuracy, precision, recall, F1 per model |
| **Table 5** | Final comparison table (all metrics + qualitative XAI ratings) |

---

## References

<!-- 
WRITING INSTRUCTIONS:
Use Perplexity to find 5–8 peer-reviewed sources. Search:
- "FER-2013 dataset CNN facial expression recognition"
- "Grad-CAM explainable AI image classification"
- "MobileNetV2 transfer learning facial recognition"
Add them here in APA or IEEE format.
-->

1. Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., ... & Bengio, Y. (2013). *Challenges in representation learning: A report on three machine learning contests*. ICML 2013.
2. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization*. ICCV 2017.
3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted residuals and linear bottlenecks*. CVPR 2018.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR 2016.
5. Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. NeurIPS 2017.
6. [ADD 2–3 MORE FROM PERPLEXITY SEARCH]

---

*Document Status: Phase 2 Draft — Last Updated: May 2026*
