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
| Test Images | 7,178 |
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
VGGNet-based baselines, while Mandave & Patil (2025) report competitive results comparing MobileNetV2 and EfficientNet-B0 on FER-2013, with EfficientNet-B0 outperforming MobileNetV2 across all metrics. ResNet50 studies 
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

This project follows a structured end-to-end pipeline 
covering dataset preparation, preprocessing, model 
development, training, evaluation, and explainability 
analysis. The methodology is designed to ensure 
reproducibility and fair comparison across all three 
models.

### 8.1 Dataset Selection and Preparation

The FER-2013 dataset is selected as the primary 
benchmark due to its public availability, academic-use 
permissions, and documented class labels. The dataset 
is loaded directly from Kaggle 
(https://www.kaggle.com/datasets/msambare/fer2013) 
using its folder structure (train/test split). 

Prior to training, the dataset is inspected for:
* Corrupted or unreadable image files
* Duplicate images across train and test sets
* Class distribution summary (Figure 1)

Class weights are computed using 
`sklearn.utils.class_weight.compute_class_weight` 
to address the 16.5-to-1 imbalance between Happy 
(7,215) and Disgust (436) classes.

### 8.2 Preprocessing Pipeline

All images undergo the following preprocessing steps 
before being fed into any model:

| Step | Custom CNN | MobileNetV2 & ResNet50 |
|------|-----------|----------------------|
| Resize | 48×48 pixels | 224×224 pixels |
| Color format | Grayscale (1 channel) | Pseudo-RGB (3 channels) |
| Normalization | Divide by 255 → [0,1] | Divide by 255 → [0,1] |
| Channel conversion | — | Replicate grayscale channel ×3 |

Transfer learning models (MobileNetV2, ResNet50) 
require 224×224 RGB input to match their ImageNet 
pretraining expectations. Grayscale images are 
converted to pseudo-RGB by replicating the single 
channel three times.

### 8.3 Data Augmentation

Data augmentation is applied **only to the training 
set** to improve model robustness and reduce 
overfitting. The following transformations are applied 
randomly during training:

| Augmentation | Value |
|-------------|-------|
| Horizontal Flip | 50% probability |
| Random Rotation | ±15 degrees |
| Random Zoom | ±10% |
| Random Brightness | ±20% |
| Random Contrast | ±15% |

No augmentation is applied to validation or test sets 
to ensure unbiased evaluation.

### 8.4 Train / Validation / Test Split

The FER-2013 dataset provides a predefined train/test 
split. The training set is further divided to create 
a validation set:

| Split | Images | Percentage |
|-------|--------|-----------|
| Training | 22,967 | 64% |
| Validation | 5,742 | 16% |
| Test | 7,178 | 20% |

Stratified splitting is used to preserve class 
distribution across all three splits.

### 8.5 Model Development

Three models are developed and compared under 
identical training conditions:

**Model 1 — Custom CNN (from scratch)**
A custom CNN architecture designed specifically for 
48×48 grayscale input. Four convolutional blocks with 
progressively increasing filter depth (32 → 64 → 128 
→ 256), each followed by Batch Normalization and 
MaxPooling. Global Average Pooling replaces Flatten 
to reduce overfitting. Full architecture details are 
provided in Section 9.

**Model 2 — MobileNetV2 (Transfer Learning)**
MobileNetV2 pretrained on ImageNet loaded with 
`include_top=False`. Base weights frozen during 
initial training. Custom classification head added: 
`GlobalAveragePooling2D → Dropout(0.5) → Dense(256, 
ReLU) → Dropout(0.3) → Dense(7, Softmax)`.

**Model 3 — ResNet50 (Transfer Learning)**
ResNet50 pretrained on ImageNet loaded with 
`include_top=False`. Same custom head as MobileNetV2. 
Fine-tuning applied to last 20 layers after initial 
convergence. Full transfer learning details in 
Section 10.

### 8.6 Training Setup and Hyperparameters

All models are trained on Kaggle Notebooks using 
GPU acceleration (NVIDIA Tesla P100).

| Hyperparameter | Custom CNN | MobileNetV2 | ResNet50 |
|----------------|-----------|-------------|----------|
| Optimizer | Adam | Adam | Adam |
| Learning Rate | 0.001 | 0.0001 | 0.0001 |
| Loss Function | Categorical Cross-Entropy | Categorical Cross-Entropy | Categorical Cross-Entropy |
| Batch Size | 32 | 32 | 32 |
| Epochs | 30 | 20 | 20 |
| Class Weights | Yes | Yes | Yes |

The following callbacks are applied during training:
* **EarlyStopping** — stops training if validation 
loss does not improve for 5 consecutive epochs
* **ModelCheckpoint** — saves the best model weights 
based on validation accuracy
* **ReduceLROnPlateau** — reduces learning rate by 
factor 0.5 if validation loss plateaus for 3 epochs

### 8.7 Evaluation Strategy

Each model is evaluated on the held-out test set 
using the following metrics:

| Metric | Method |
|--------|--------|
| Accuracy | Overall correct predictions / total |
| Precision (macro) | sklearn `classification_report` |
| Recall (macro) | sklearn `classification_report` |
| F1-Score (macro) | sklearn `classification_report` |
| Confusion Matrix | Per-class prediction heatmap |
| Training Time | Wall-clock seconds per epoch |
| Parameter Count | `model.count_params()` |

Per-class metrics (precision, recall, F1) are 
reported for all 7 emotion classes to expose 
minority-class performance that overall accuracy 
would otherwise mask.

### 8.8 Explainability Analysis (Grad-CAM and SHAP)

**Grad-CAM** is applied to the final convolutional 
layer of each model to generate spatial heatmaps 
showing which image regions contributed most to 
each prediction. Visualizations are generated for:
* 5 correctly classified samples per model 
  (Figure 5)
* 5 misclassified samples per model with true 
  and predicted labels annotated (Figure 6)

**SHAP** (`GradientExplainer`) is applied using a 
background sample of 50 training images. Pixel-level 
positive (red) and negative (blue) contributions are 
visualized for 3 test images per model (Figure 7).

Together, Grad-CAM and SHAP address RQ4 and RQ5 by 
revealing whether models focus on semantically 
meaningful facial regions and diagnosing the visual 
causes of misclassifications.

### 8.9 Deployment Considerations and Ethics

The final system is presented as a 
**decision-support framework** that assists human 
experts rather than replacing human judgment. 
Key deployment considerations include:

* **Computational constraints** — MobileNetV2 is 
  the most suitable model for edge or mobile 
  deployment due to its lightweight architecture
* **Class imbalance risks** — Without proper 
  class weighting, models will systematically 
  underperform on minority emotions (Disgust, Fear)
* **Ethical concerns** — Automated emotion 
  recognition carries risks of misuse in 
  surveillance or profiling contexts; the system 
  should only be deployed with informed consent 
  and human oversight
* **Limitations** — FER-2013's crowdsourced labels 
  and low resolution limit real-world generalization; 
  future work should validate on higher-resolution, 
  expert-annotated datasets
  
---

## Section 9 — CNN Model Design (Custom CNN)

### 9.1 Architecture Rationale

The custom CNN is designed specifically for 48×48 
grayscale input, balancing representational depth 
with computational efficiency. Four convolutional 
blocks progressively increase filter depth (32 → 64 
→ 128 → 256), allowing the network to learn 
increasingly abstract facial features. Global Average 
Pooling replaces the traditional Flatten layer to 
reduce the risk of overfitting on the relatively 
small FER-2013 training set.

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
| Dropout_1 | Dropout | 256 | rate = 0.5 | — |
| Dense_1 | Dense | 256 | 256 units | ReLU |
| Dropout_2 | Dropout | 256 | rate = 0.3 | — |
| Output | Dense | 7 | 7 units | Softmax |

### 9.3 Design Decisions

| Decision | Justification |
|----------|--------------|
| **4 Convolutional Blocks** | Sufficient depth for 48×48 images without excessive parameters |
| **Filter doubling (32→256)** | Progressive feature abstraction from edges to complex expressions |
| **Batch Normalization** | Stabilizes training, reduces sensitivity to learning rate |
| **MaxPooling 2×2** | Reduces spatial dimensions while preserving dominant features |
| **Global Average Pooling** | Replaces Flatten to reduce overfitting and parameter count |
| **Dropout (0.5 + 0.3)** | Regularization to prevent overfitting on limited training data |
| **Softmax output** | Multi-class classification across 7 emotion categories |
| **ReLU activation** | Standard non-linear activation; avoids vanishing gradient |

---

## Section 10 — Transfer Learning Models

### 10.1 Strategy

Both MobileNetV2 and ResNet50 are loaded with 
ImageNet pre-trained weights with their top 
classification layers removed (`include_top=False`). 
The base model weights are frozen during initial 
training (feature extraction phase). A custom 
classification head is added on top:

`GlobalAveragePooling2D → Dropout(0.5) → Dense(256, ReLU) → Dropout(0.3) → Dense(7, Softmax)`

Input images are resized to **224×224×3** (grayscale 
channel replicated to 3 channels) to match ImageNet 
input expectations.

### 10.2 Model Comparison

| Property | MobileNetV2 | ResNet50 |
|----------|-------------|----------|
| **Original Dataset** | ImageNet (1.2M images, 1000 classes) | ImageNet |
| **Architecture Type** | Inverted Residuals + Depthwise Separable Convolutions | Deep Residual Networks (skip connections) |
| **Total Parameters (approx.)** | ~3.4M | ~25M |
| **Input Size Required** | 224×224×3 | 224×224×3 |
| **Base Weights** | Frozen (initial training) | Frozen (initial training) |
| **Fine-Tuning** | Unfreeze last 30 layers after convergence | Unfreeze last 30 layers after convergence |
| **Custom Head** | GAP → Dropout(0.5) → Dense(256) → Dense(7) | Same |
| **Optimizer** | Adam (lr = 0.0001) | Adam (lr = 0.0001) |
| **Epochs** | 20 | 20 |

### 10.3 Two-Phase Training Strategy

Both transfer learning models follow a 
**two-phase training strategy**:

**Phase 1 — Feature Extraction (Epochs 1–10):**
* Base model weights are **frozen**
* Only the custom classification head is trained
* Higher learning rate acceptable (0.001)
* Purpose: adapt the new head to FER-2013 features

**Phase 2 — Fine-Tuning (Epochs 11–20):**
* Last 30 layers (MobileNetV2) or last 20 layers 
  (ResNet50) are **unfrozen**
* Lower learning rate used (0.0001) to avoid 
  destroying pretrained weights
* Purpose: fine-tune higher-level features 
  specifically for facial expression patterns

### 10.4 Justification for Model Selection

| Model | Why Selected |
|-------|-------------|
| **MobileNetV2** | Lightweight architecture (~3.4M params) designed for mobile and edge deployment; best accuracy-to-cost ratio; directly relevant to real-time FER applications |
| **ResNet50** | Deep residual architecture (~25M params) with skip connections that prevent vanishing gradients; strong benchmark performance; represents high-capacity transfer learning |

Together, MobileNetV2 and ResNet50 represent a 
**lightweight vs. deep** trade-off that directly 
addresses RQ2 and RQ3 — which model family provides 
the best balance between performance and 
computational efficiency?

### 10.5 Handling the Grayscale-to-RGB Mismatch

A key challenge when applying ImageNet-pretrained 
models to FER-2013 is the domain mismatch between 
large RGB images (224×224×3) and small grayscale 
images (48×48×1). This project addresses this by:

* Resizing all images from 48×48 to 224×224 using 
  bilinear interpolation
* Replicating the single grayscale channel three 
  times to create pseudo-RGB input: 
  `[gray, gray, gray]`
* This approach is consistent with published FER 
  transfer learning studies (Tutuianu et al., 2024; Mandave & Patil, 2025)
  
---

## Section 11 — Evaluation Metrics

All three models are evaluated on the held-out test 
set using the same metrics to ensure fair and 
reproducible comparison.

### 11.1 Quantitative Metrics

| Metric | Formula / Method | Purpose |
|--------|-----------------|---------|
| **Accuracy** | Correct predictions / Total predictions | Overall performance baseline |
| **Precision (macro)** | TP / (TP + FP) averaged over all classes | Correctness of positive predictions |
| **Recall (macro)** | TP / (TP + FN) averaged over all classes | Coverage of actual positives |
| **F1-Score (macro)** | 2 × (P × R) / (P + R) averaged over classes | Balanced metric — handles class imbalance |
| **Per-class F1** | F1 computed individually per emotion | Exposes minority class failures |
| **Confusion Matrix** | Per-class prediction heatmap | Visual error pattern analysis |
| **Training Time** | Wall-clock seconds per epoch | Computational cost comparison |
| **Parameter Count** | `model.count_params()` | Model complexity comparison |

### 11.2 Why Macro-Averaged Metrics

Macro-averaging computes each metric independently 
per class and then takes the unweighted average. 
This treats all 7 emotion classes equally regardless 
of sample size — meaning Disgust (436 samples) 
contributes equally to the final score as Happy 
(7,215 samples). This directly addresses the 
*accuracy illusion* problem described in Section 3, 
where overall accuracy masks poor performance on 
minority classes.

### 11.3 Explainability Metrics

| Metric | Method | Purpose |
|--------|--------|---------|
| **Grad-CAM Quality** | Qualitative visual assessment (1–5 scale) | Does the model focus on meaningful facial regions? |
| **SHAP Interpretability** | Qualitative pixel-level analysis | Are positive/negative contributions semantically meaningful? |
| **Misclassification Diagnosis** | Grad-CAM on incorrect predictions | Identifies which facial regions cause errors |

### 11.4 Final Model Comparison Table

Results will be reported in the following format 
after training:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Parameters | Grad-CAM Quality |
|-------|----------|-----------|--------|----------|---------------|------------|-----------------|
| Custom CNN | 58.58% | 0.5259 | 0.5872 | 0.5364 | 27.3 min | ~1.2M | High |
| MobileNetV2 | 50.49% | 0.4434 | 0.4873 | 0.4440 | 153.5 min | ~3.4M | Medium |
| ResNet50 | 40.78% | 0.3782 | 0.4110 | 0.3530 | 138.9 min | ~25M | Medium |

### 11.5 Implementation Tools

| Tool | Purpose |
|------|---------|
| `sklearn.metrics.classification_report` | Precision, Recall, F1 per class |
| `sklearn.metrics.confusion_matrix` | Confusion matrix generation |
| `matplotlib` / `seaborn` | Visualization of curves and matrices |
| `tf-keras-vis` / `grad-cam` library | Grad-CAM heatmap generation |
| `shap.GradientExplainer` | SHAP pixel-level explanations |
| `model.count_params()` | Parameter count |
| `time` module | Training time measurement |

---

## Section 12 — Expected Figures and Tables

All figures will be saved as `.pdf` files in the 
`/figures/` directory of the GitHub repository 
and generated during model training on Kaggle.

### 12.1 Expected Figures

| Figure | Description | Filename |
|--------|-------------|----------|
| **Figure 1** | Class distribution bar chart | `outputs/figures/Fig1_Class_Distribution.pdf` |
| **Figure 2** | Sample images per class | `outputs/figures/Fig2_Sample_Images.pdf` |
| **Figure 3a** | Training curves — Custom CNN | `outputs/figures/Fig3a_Training_Curves_Custom_CNN.pdf` |
| **Figure 3b** | Training curves — MobileNetV2 | `outputs/figures/Fig3b_Training_Curves_MobileNetV2.pdf` |
| **Figure 3c** | Training curves — ResNet50 | `outputs/figures/Fig3c_Training_Curves_ResNet50.pdf` |
| **Figure 4a** | Confusion matrix — Custom CNN | `outputs/figures/Fig4a_Confusion_Matrix_Custom_CNN.pdf` |
| **Figure 4b** | Confusion matrix — MobileNetV2 | `outputs/figures/Fig4b_Confusion_Matrix_MobileNetV2.pdf` |
| **Figure 4c** | Confusion matrix — ResNet50 | `outputs/figures/Fig4c_Confusion_Matrix_ResNet50.pdf` |
| **Figure 5** | Model comparison chart | `outputs/figures/Fig5_Model_Comparison.pdf` |
| **Figure 6a** | Grad-CAM — Custom CNN | `outputs/figures/Fig6a_GradCAM_Custom_CNN.pdf` |
| **Figure 6b** | Grad-CAM — MobileNetV2 | `outputs/figures/Fig6b_GradCAM_MobileNetV2.pdf` |
| **Figure 6c** | Grad-CAM — ResNet50 | `outputs/figures/Fig6c_GradCAM_ResNet50.pdf` |
| **Figure 7a** | SHAP — Custom CNN | `outputs/figures/Fig7a_SHAP_Custom_CNN.pdf` |
| **Figure 7b** | SHAP — MobileNetV2 | `outputs/figures/Fig7b_SHAP_MobileNetV2.pdf` |
| **Figure 7c** | SHAP — ResNet50 | `outputs/figures/Fig7c_SHAP_ResNet50.pdf` |
| **Figure 8a** | ROC curves — Custom CNN | `outputs/figures/Fig8a_ROC_Custom_CNN.pdf` |
| **Figure 8b** | ROC curves — MobileNetV2 | `outputs/figures/Fig8b_ROC_MobileNetV2.pdf` |
| **Figure 8c** | ROC curves — ResNet50 | `outputs/figures/Fig8c_ROC_ResNet50.pdf` |

### 12.2 Expected Tables

| Table | Description | Filename |
|-------|-------------|----------|
| **Table 1** | Dataset overview | In proposal Section 5 |
| **Table 2** | Class distribution | In proposal Section 5 |
| **Table 3** | Custom CNN architecture | In proposal Section 9 |
| **Table 4** | Transfer learning comparison | In proposal Section 10 |
| **Table 5** | Training hyperparameters | In proposal Section 8 |
| **Table 6** | Model comparison results | `outputs/tables/Table8_Model_Comparison.csv` |
| **Table 7** | Per-class F1 — Custom CNN | `outputs/tables/Table7_CustomCNN_PerClass_F1.csv` |
| **Table 8** | Per-class F1 — MobileNetV2 | `outputs/tables/Table7_MobileNetV2_PerClass_F1.csv` |
| **Table 9** | Per-class F1 — ResNet50 | `outputs/tables/Table7_ResNet50_PerClass_F1.csv` |

### 12.3 Figure Priority

Not all figures carry equal weight. The following 
are considered most critical for this project:

| Priority | Figure | Why Critical |
|----------|--------|-------------|
| 🔴 High | Figure 6 (Grad-CAM Misclassified) | Directly addresses RQ4, RQ5, and error analysis requirement |
| 🔴 High | Figure 4 (Confusion Matrices) | Exposes per-class failures and class imbalance impact |
| 🔴 High | Figure 3 (Training Curves) | Shows model convergence and overfitting/underfitting |
| 🟡 Medium | Figure 5 (Grad-CAM Correct) | Validates that models focus on meaningful facial regions |
| 🟡 Medium | Figure 7 (SHAP) | Provides pixel-level explanation depth |
| 🟢 Lower | Figure 1 (Class Distribution) | Dataset visualization — quick to generate |
| 🟢 Lower | Figure 2 (Workflow Diagram) | Methodology visualization |

---

## References

### Foundational Papers

1. Goodfellow, I. J., Erhan, D., Carrier, P. L., 
   Courville, A., Mirza, M., Hamner, B., & Bengio, Y. 
   (2013). Challenges in representation learning: 
   A report on three machine learning contests. 
   *International Conference on Machine Learning 
   (ICML 2013)*. 
   https://arxiv.org/abs/1307.0414

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). 
   Deep residual learning for image recognition. 
   *Proceedings of the IEEE Conference on Computer 
   Vision and Pattern Recognition (CVPR)*, 770–778. 
   https://arxiv.org/abs/1512.03385

3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., 
   & Chen, L. C. (2018). MobileNetV2: Inverted 
   residuals and linear bottlenecks. *Proceedings 
   of the IEEE Conference on Computer Vision and 
   Pattern Recognition (CVPR)*, 4510–4520. 
   https://arxiv.org/abs/1801.04381

4. Selvaraju, R. R., Cogswell, M., Das, A., 
   Vedantam, R., Parikh, D., & Batra, D. (2017). 
   Grad-CAM: Visual explanations from deep networks 
   via gradient-based localization. *Proceedings of 
   the IEEE International Conference on Computer 
   Vision (ICCV)*, 618–626. 
   https://arxiv.org/abs/1610.02391

5. Lundberg, S. M., & Lee, S. I. (2017). A unified 
   approach to interpreting model predictions. 
   *Advances in Neural Information Processing 
   Systems (NeurIPS)*, 30, 4765–4774. 
   https://arxiv.org/abs/1705.07874

### FER-Specific Papers

6. Pramerdorfer, C., & Kampel, M. (2016). Facial 
   expression recognition using convolutional 
   neural networks: State of the art. *arXiv 
   preprint*. 
   https://arxiv.org/abs/1612.02903

7. Li, S., & Deng, W. (2022). Deep facial expression 
   recognition: A survey. *IEEE Transactions on 
   Affective Computing*, 13(3), 1195–1215. 
   https://doi.org/10.1109/TAFFC.2020.2981446

8. Tran, M., et al. (2021). Deep-Emotion: Facial 
   expression recognition using attentional 
   convolutional network. *Sensors (MDPI)*, 
   21(11), 3046. 
   https://doi.org/10.3390/s21113046

9. Tutuianu, G. I., Liu, Y., Alamäki, A., & 
   Kauttonen, J. (2024). Benchmarking deep facial 
   expression recognition: An extensive protocol 
   with balanced dataset in the wild. *Engineering 
   Applications of Artificial Intelligence*, 
   133, 108896. 
   https://doi.org/10.1016/j.engappai.2024.108896

10. Mandave, D. D., & Patil, L. V. (2025). A 
    statistically rigorous comparison of MobileNetV2 
    and EfficientNet-B0 for facial expression 
    recognition on the FER2013 benchmark. 
    *Information Dynamics and Applications*, 4(4). 
    https://doi.org/10.56578/ida040401

---

*Document Status: Phase 2 Draft — Last Updated: May 2026*
