[proposal.md](https://github.com/user-attachments/files/28229783/proposal.md)
# Phase 2 Proposal Document
## Comparative Analysis of Custom CNN and Transfer Learning for Facial Expression Recognition

> **Course:** Pattern Recognition  
> **Institution:** University of Europe for Applied Sciences  
> **Student:** Yada Thadathanacharoen  
> **Submission:** Phase 2 — Proposal and Code Implementation  
> **Due Date:** June 7, 2026  

---

## Section 1 — Project Title

**Facial Expression Recognition Framework: A Comparative Analysis of Custom CNN and Transfer Learning Models with Explainable AI**

---

## Section 2 — Background and Motivation

Facial expression recognition (FER) is an important problem in affective computing, with practical applications in healthcare (e.g., mental health screening), education (student engagement monitoring), transportation (driver safety), and human–computer interaction. Automatically detecting basic emotions from facial images can support systems that adapt to a user’s emotional state or help clinicians and teachers interpret subtle non‑verbal cues.

Convolutional Neural Networks (CNNs) have become the standard approach for FER because they learn hierarchical features directly from raw pixels, avoiding the need for hand‑crafted features. However, a key design question remains: should practitioners build a compact CNN tailored specifically to the FER‑2013 domain (48×48 grayscale faces), or should they rely on transfer learning from large pretrained models trained on high‑resolution RGB datasets such as ImageNet? Reported accuracies on FER‑2013 remain noticeably below performance on more controlled datasets, which suggests that this architectural trade‑off is still not fully understood.

Transfer learning models such as MobileNetV2 and ResNet50 bring powerful generic features learned from millions of color images, but they must be adapted to low‑resolution grayscale input. A custom CNN, in contrast, can be designed from scratch for 48×48 grayscale faces, potentially achieving competitive performance with far fewer parameters and lower computational cost. The FER‑2013 dataset further complicates this choice because of its strong class imbalance (Disgust: 436 samples vs. Happy: 7,215), noisy crowdsourced labels, and variation in pose and occlusion.

Beyond raw accuracy, a major limitation of many deep learning systems is their opacity. In human‑centered applications like FER, it is important to understand why a model predicts a specific emotion for a given face. Explainable AI methods such as Grad‑CAM, which visualizes discriminative image regions, and SHAP, which estimates pixel‑level importance, can help reveal whether a model is focusing on meaningful facial regions (eyes, mouth, eyebrows) or on artefacts and background. This project therefore combines model comparison with explainability to build a more transparent FER framework.

The aim of this project is to systematically compare a custom CNN against two transfer learning models (MobileNetV2, ResNet50) on FER‑2013, to analyze how their performance and computational cost differ, and to use Grad‑CAM and SHAP to interpret their behavior. The final system is framed as a decision‑support tool that assists human experts in emotion recognition rather than replacing human judgment.

---

## Section 3 — Problem Statement

Facial expression recognition from static images is a challenging multi‑class classification task for several reasons. There is high intra‑class variability: the same emotion can appear differently across individuals, ages, and cultures, and with different intensities. There is also strong inter‑class similarity, for example between Fear and Surprise or between Angry and Disgust, which can look similar in low‑resolution images. The FER‑2013 dataset adds further difficulty, because it contains only 48×48 grayscale faces, which reduces fine‑grained visual information.

The training portion of FER‑2013 exhibits severe class imbalance: Happy contains 7,215 samples while Disgust contains only 436, creating a 16.5‑to‑1 disparity. This imbalance can cause models trained with standard loss functions to focus mainly on majority classes and ignore minority emotions. While many CNN‑based methods have been proposed for FER‑2013, systematic comparisons between a domain‑specific custom CNN and particular transfer learning architectures (MobileNetV2 and ResNet50) on low‑resolution grayscale faces are still limited. In addition, the combined use of Grad‑CAM and SHAP to diagnose misclassifications and to understand which facial regions drive model decisions has not been widely explored for this dataset.

This project addresses these gaps by comparing three CNN architectures (custom CNN, MobileNetV2, ResNet50) on FER‑2013 under controlled training conditions, reporting both overall and per‑class metrics, and applying Grad‑CAM and SHAP to analyze correct and incorrect predictions. A key question is whether a compact, domain‑matched CNN can outperform heavier ImageNet‑pretrained models on this specific grayscale, low‑resolution FER task.

---

## Section 4 — Selected Dataset and Dataset Link

| Field | Details |
|-------|---------|
| **Dataset Name** | FER‑2013 (Facial Expression Recognition 2013) |
| **Source / Kaggle Link** | https://www.kaggle.com/datasets/msambare/fer2013 |
| **Original Source** | Introduced at ICML 2013 (Goodfellow et al., 2013) |
| **License** | Publicly available for academic use |
| **Format** | Grayscale images in folder structure (train/test) |

FER‑2013 is widely used in FER research and is suitable for an academic project because it is freely available, reasonably large, and already split into training and test sets.

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

The dataset contains only facial images, each labeled with one of seven basic emotion categories. All images are resized and stored at 48×48 pixels, which significantly constrains the amount of detail available to the model.

### 5.2 Emotion Classes and Training Distribution

| Emotion | Training Samples | Percentage |
|---------|-----------------|------------|
| Angry   | 3,995 | 13.9% |
| Disgust | 436   | 1.5% |
| Fear    | 4,097 | 14.3% |
| Happy   | 7,215 | 25.1% |
| Neutral | 4,965 | 17.3% |
| Sad     | 4,830 | 16.8% |
| Surprise| 3,171 | 11.0% |
| **TOTAL** | **28,709** | **100%** |

The Happy class dominates the training set, while Disgust is severely underrepresented. This skewed distribution has a direct impact on model performance and motivates the use of class‑weighted loss and careful evaluation.

### 5.3 Class Imbalance Analysis

The 16.5‑to‑1 ratio between Happy and Disgust makes it difficult for a model to learn reliable patterns for minority classes. If all samples are treated equally in the loss function, the model can reach high overall accuracy simply by focusing on the majority emotions while misclassifying most minority examples. To counter this, class weights are computed from the training set frequencies and applied in the loss function, and macro‑averaged metrics are used during evaluation to give each class equal importance.

### 5.4 Dataset Challenges

The FER‑2013 dataset presents several well‑known challenges:

- **Low spatial resolution (48×48 pixels)** — limits the model’s ability to capture fine‑grained facial detail, especially for subtle emotion differences.
- **Grayscale format** — removes color cues (such as redness or pallor) that may help humans distinguish some emotional states.
- **Label noise** — labels were collected via crowdsourcing, so some images are mislabeled or ambiguous.
- **Pose and occlusion variation** — some faces are non‑frontal or partially occluded by hair, hands, or objects.
- **Intra‑class and inter‑class similarity** — expressions vary widely between individuals, and similar facial configurations can correspond to different emotions.

Because of these factors, reported accuracies on FER‑2013 (often around the mid‑60% range) are lower than on more controlled datasets such as CK+, where accuracy can exceed 90%.

---

## Section 6 — Research Questions

| RQ | Research Question |
|----|------------------|
| **RQ1** | How accurately can CNN‑based models perform seven‑class facial expression recognition using the FER‑2013 dataset? |
| **RQ2** | Which model family—custom CNN or transfer learning (MobileNetV2, ResNet50)—provides the best balance between predictive performance and computational efficiency? |
| **RQ3** | How do preprocessing choices, data augmentation, and class‑imbalance handling influence model robustness and per‑class performance on FER‑2013? |
| **RQ4** | Do Grad‑CAM explanations show that the trained models focus on meaningful facial regions (e.g., eyes, mouth, brow) when making predictions? |
| **RQ5** | What are the main failure patterns, deployment limitations, and practical risks when applying this facial expression recognition framework in real human‑centered settings? |

These questions guide the design of the models, the evaluation metrics, and the explainability analysis.

---

## Section 7 — Expected Results

This section describes the expectations that motivated the study and then briefly compares them with the actual experimental results. The original expectations were based on published FER‑2013 benchmarks and typical transfer learning practice, and they assumed that transfer learning models would outperform a custom CNN. After training the models, it turned out that the overall performance levels were lower than these optimistic predictions and that the custom CNN actually achieved the best results on FER‑2013.

### 7.1 Expected Classification Performance

Based on prior work on FER‑2013, the initial hypothesis was that all three models would reach moderately high accuracy, with transfer learning slightly ahead of the custom CNN:

| Model | Expected Accuracy | Expected Macro‑F1 |
|-------|------------------|-------------------|
| Custom CNN (from scratch)   | 63–70% | 0.55–0.65 |
| MobileNetV2 (transfer learning) | 65–72% | 0.65–0.70 |
| ResNet50 (transfer learning)   | 65–73% | 0.62–0.70 |

These ranges are consistent with reported studies using deep CNNs and transfer learning on FER‑2013. It was assumed that the larger pretrained models, once fine‑tuned, would reach the highest accuracy.

In practice, the final experiments did not reach these ranges. On the held‑out FER‑2013 test set, the custom CNN achieved **58.58%** accuracy with macro‑F1 of **0.5364**, MobileNetV2 achieved **50.49%** accuracy with macro‑F1 of **0.4440**, and ResNet50 achieved **40.78%** accuracy with macro‑F1 of **0.3530**. These results show that, under the chosen configuration, the custom CNN generalized better than both transfer learning models.

### 7.2 Expected Per‑Class Performance

The class imbalance in FER‑2013 suggests that performance would vary significantly across emotions. Before training, the following pattern was expected:

| Emotion | Expected F1 | Reason |
|---------|------------|--------|
| Happy   | 0.85–0.92 | Largest class; visually distinct |
| Neutral | 0.60–0.75 | Well‑represented; limited confusion |
| Angry   | 0.50–0.65 | High intra‑class variation; confused with Disgust |
| Sad     | 0.50–0.65 | Overlaps with Neutral and Fear |
| Fear    | 0.30–0.50 | Visually similar to Surprise and Sad |
| Disgust | 0.20–0.40 | Severely underrepresented; hard to learn |
| Surprise| 0.55–0.70 | Distinct features but confusable with Fear |

As expected, the final per‑class F1‑scores for all three models show that minority classes, especially Disgust and Fear, remain the most difficult, even with class‑weighted loss. The custom CNN generally performs better on several emotions, but no model fully overcomes the dataset’s imbalance and noise.

### 7.3 Expected Computational Cost

Before running experiments, the anticipated computational trade‑offs were:

| Model      | Expected Training Time | Parameters (approx.) |
|-----------|------------------------|----------------------|
| Custom CNN | Fastest (~30–60 min)  | ~0.5M–2M |
| MobileNetV2 | Moderate (~1–2 hours) | ~3.4M |
| ResNet50    | Slowest (~2–3 hours)  | ~25M |

It was assumed that the custom CNN would be fastest to train, but that transfer learning models would reach higher accuracy per epoch once fine‑tuning began.

The actual training confirmed the cost pattern but not the accuracy assumption. The custom CNN (about 1.2M parameters) trained in roughly **27.3 minutes**, MobileNetV2 (about 3.4M parameters) in **153.5 minutes**, and ResNet50 (about 25M parameters) in **138.9 minutes** on the chosen GPU environment. Despite being the lightest model, the custom CNN achieved the best test accuracy and macro‑F1.

### 7.4 Expected Model Trade‑offs

The initial expectation from the literature was:

- The **custom CNN** would have the fewest parameters and fastest training time, but the lowest overall accuracy.
- **MobileNetV2** would offer the best accuracy‑to‑cost ratio.
- **ResNet50** would achieve the highest absolute accuracy, at the expense of more training time and memory.

After experiments, only part of this expectation held. The custom CNN did have the smallest parameter count and the shortest training time, but it also achieved the highest test accuracy and macro‑F1. Both MobileNetV2 and ResNet50 suffered from a domain gap between ImageNet (color, 224×224) and FER‑2013 (48×48 grayscale), which limited the benefit of transfer learning under the configurations used in this project.

### 7.5 Expected Explainability Findings

From an explainability perspective, it was expected that:

- Grad‑CAM heatmaps for correctly classified samples would highlight key facial regions such as the eyes, mouth, and eyebrows.
- Misclassified samples (for example, Disgust predicted as Angry, or Fear predicted as Sad) would show more diffuse or misplaced activation, sometimes focusing on background or occlusions.
- SHAP visualizations would indicate concentrated positive contributions around relevant facial areas for confident predictions and scattered negative contributions for uncertain or incorrect predictions.

The qualitative Grad‑CAM and SHAP results are broadly consistent with these expectations. Correct predictions often show focused attention on central facial features, while some errors reveal attention leaking to irrelevant regions, which helps explain recurring misclassification patterns.

---

## Section 8 — Proposed Methodology

This project follows a structured end‑to‑end pipeline including dataset preparation, preprocessing, model development, training, evaluation, and explainability analysis. The main goal of the methodology is to allow a fair comparison between the custom CNN and the two transfer learning models.

### 8.1 Dataset Selection and Preparation

The FER‑2013 dataset is selected as the primary benchmark due to its public availability, academic‑use permissions, and established use in FER research. The dataset is loaded from Kaggle using its existing train/test folder structure. Before training, the data are checked for corrupted or unreadable images, duplicate images, and class distribution statistics.

The original training split is further divided into a new training and validation set using stratified splitting to preserve class distributions:

| Split      | Images | Percentage |
|-----------|--------|-----------|
| Training   | 22,967 | 64% |
| Validation | 5,742  | 16% |
| Test       | 7,178  | 20% |

Class weights are computed using `sklearn.utils.class_weight.compute_class_weight` to address the imbalance between Happy (7,215 samples) and Disgust (436 samples).

### 8.2 Preprocessing Pipeline

All images are normalized by dividing pixel values by 255, resulting in values in the range [0, 1]. The preprocessing differs slightly between the custom CNN and the transfer learning models:

| Step             | Custom CNN            | MobileNetV2 & ResNet50       |
|------------------|-----------------------|------------------------------|
| Resize           | 48×48 pixels          | 224×224 pixels               |
| Color format     | Grayscale (1 channel) | Pseudo‑RGB (3 channels)      |
| Normalization    | Divide by 255 → [0,1] | Divide by 255 → [0,1]        |
| Channel conversion | —                   | Replicate grayscale channel ×3 |

For MobileNetV2 and ResNet50, the single grayscale channel is replicated three times to create pseudo‑RGB images compatible with ImageNet‑pretrained weights.

### 8.3 Data Augmentation

Data augmentation is applied only to the training set to improve robustness and reduce overfitting. The following random transformations are used:

| Augmentation      | Value             |
|-------------------|-------------------|
| Horizontal Flip   | 50% probability   |
| Random Rotation   | ±15 degrees       |
| Random Zoom       | ±10%              |
| Random Brightness | ±20%              |
| Random Contrast   | ±15%              |

No augmentation is applied to the validation or test sets.

### 8.4 Train / Validation / Test Split

As noted above, the original training set is split into training and validation subsets using stratified sampling. This ensures that each split reflects the overall class distribution and provides a realistic evaluation during model development.

### 8.5 Model Development

Three CNN‑based models are developed and compared under similar conditions:

**Model 1 — Custom CNN (from scratch)**  
A compact CNN designed for 48×48 grayscale input. It has four convolutional blocks with increasing filters (32 → 64 → 128 → 256), each followed by Batch Normalization and MaxPooling. Global Average Pooling (instead of Flatten) reduces the risk of overfitting, followed by a dense layer with 256 units and a final dense layer with 7 softmax outputs. Full architecture details are given in Section 9.

**Model 2 — MobileNetV2 (Transfer Learning)**  
MobileNetV2 is loaded with ImageNet weights using `include_top=False`. The base is initially frozen, and a custom classification head is added: `GlobalAveragePooling2D → Dropout(0.5) → Dense(256, ReLU) → Dropout(0.3) → Dense(7, Softmax)`.

**Model 3 — ResNet50 (Transfer Learning)**  
ResNet50 is also loaded with ImageNet weights and `include_top=False`, using the same custom classification head as MobileNetV2. After an initial training phase with frozen base, a subset of the deeper layers is unfrozen and fine‑tuned.

### 8.6 Training Setup and Hyperparameters

All models are trained on Kaggle Notebooks with GPU acceleration (NVIDIA Tesla P100). The main hyperparameters are:

| Hyperparameter | Custom CNN | MobileNetV2 | ResNet50 |
|----------------|-----------|-------------|----------|
| Optimizer      | Adam      | Adam        | Adam     |
| Learning Rate  | 0.001     | 0.0001      | 0.0001   |
| Loss Function  | Categorical Cross‑Entropy | Categorical Cross‑Entropy | Categorical Cross‑Entropy |
| Batch Size     | 32        | 32          | 32       |
| Epochs         | 30        | 20          | 20       |
| Class Weights  | Yes       | Yes         | Yes      |

The following callbacks are used:

- **EarlyStopping**: stops training if validation loss does not improve for 5 consecutive epochs.
- **ModelCheckpoint**: saves the best model weights based on validation accuracy.
- **ReduceLROnPlateau**: reduces the learning rate by a factor of 0.5 if validation loss plateaus for 3 epochs.

### 8.7 Evaluation Strategy

Each model is evaluated on the held‑out test set using the following metrics:

| Metric            | Method                          |
|-------------------|---------------------------------|
| Accuracy          | Overall correct / total         |
| Precision (macro) | `sklearn.classification_report` |
| Recall (macro)    | `sklearn.classification_report` |
| F1‑Score (macro)  | `sklearn.classification_report` |
| Confusion Matrix  | `sklearn.confusion_matrix`      |
| Training Time     | Wall‑clock time per epoch       |
| Parameter Count   | `model.count_params()`          |

Per‑class precision, recall, and F1‑scores are reported for all seven emotions, so that performance on minority classes is visible and not hidden behind a single accuracy number.

### 8.8 Explainability Analysis (Grad‑CAM and SHAP)

Grad‑CAM is applied to the final convolutional layer of each model to generate class‑specific heatmaps for selected test samples. For each model, Grad‑CAM visualizations are produced for several correctly classified images and several misclassified images, with true and predicted labels annotated. This helps reveal whether the model focuses on meaningful facial regions when predicting a given emotion.

SHAP (`GradientExplainer`) is applied with a small background set of training images to compute pixel‑level importance values for selected test faces. Positive and negative contributions are visualized as overlays, providing a more detailed picture of how local features influence the final prediction. Together, Grad‑CAM and SHAP address RQ4 and RQ5 by assessing interpretability and diagnosing typical error patterns.

### 8.9 Deployment Considerations and Ethics

The final system is presented as a decision‑support framework that could assist human experts, rather than as an autonomous decision maker. Several deployment aspects are considered:

- **Computational constraints**: MobileNetV2 is the most suitable model for resource‑constrained or edge deployment due to its small parameter count, even though in this project the custom CNN was most accurate.
- **Class imbalance risks**: Misclassification of minority emotions can lead to biased or misleading outputs if the model is used in sensitive contexts.
- **Ethical concerns**: Automated emotion recognition carries risks of misuse in surveillance or profiling. Any real deployment should involve informed consent, clear communication of limitations, and human oversight.
- **Dataset limitations**: FER‑2013’s low resolution and crowdsourced labels limit generalization. Future work should validate models on higher‑resolution, expert‑annotated datasets.

---

## Section 9 — CNN Model Design (Custom CNN)

### 9.1 Architecture Rationale

The custom CNN is designed specifically for 48×48 grayscale input, aiming to balance representational power and computational efficiency. Four convolutional blocks gradually increase the number of filters (32, 64, 128, 256) to capture low‑level edges and higher‑level expression features. Batch Normalization stabilizes training, MaxPooling reduces spatial dimensions, and Global Average Pooling replaces Flatten to reduce overfitting on the relatively small training set.

### 9.2 Architecture Table

| Layer      | Type         | Output Shape  | Filters / Units | Activation |
|-----------|--------------|---------------|-----------------|------------|
| Input     | Input        | 48 × 48 × 1   | —               | —          |
| Conv_1    | Conv2D       | 48 × 48 × 32  | 32 @ 3×3        | ReLU       |
| BN_1      | BatchNorm    | 48 × 48 × 32  | —               | —          |
| Pool_1    | MaxPool2D    | 24 × 24 × 32  | 2×2             | —          |
| Conv_2    | Conv2D       | 24 × 24 × 64  | 64 @ 3×3        | ReLU       |
| BN_2      | BatchNorm    | 24 × 24 × 64  | —               | —          |
| Pool_2    | MaxPool2D    | 12 × 12 × 64  | 2×2             | —          |
| Conv_3    | Conv2D       | 12 × 12 × 128 | 128 @ 3×3       | ReLU       |
| BN_3      | BatchNorm    | 12 × 12 × 128 | —               | —          |
| Pool_3    | MaxPool2D    | 6 × 6 × 128   | 2×2             | —          |
| Conv_4    | Conv2D       | 6 × 6 × 256   | 256 @ 3×3       | ReLU       |
| BN_4      | BatchNorm    | 6 × 6 × 256   | —               | —          |
| GAP       | GlobalAvgPool2D | 256       | —               | —          |
| Dropout_1 | Dropout      | 256           | rate = 0.5      | —          |
| Dense_1   | Dense        | 256           | 256 units       | ReLU       |
| Dropout_2 | Dropout      | 256           | rate = 0.3      | —          |
| Output    | Dense        | 7             | 7 units         | Softmax    |

### 9.3 Design Decisions

| Decision                     | Justification |
|-----------------------------|--------------|
| 4 convolutional blocks      | Sufficient depth for 48×48 images without excessive parameters |
| Filter doubling (32→256)    | Progressive feature abstraction from edges to complex facial patterns |
| Batch Normalization         | Stabilizes training and helps with convergence |
| MaxPooling (2×2)            | Reduces spatial resolution while keeping dominant features |
| Global Average Pooling      | Reduces overfitting and parameter count compared to Flatten |
| Dropout (0.5 and 0.3)       | Regularization to prevent overfitting on limited data |
| Softmax output layer        | Supports multi‑class classification across 7 emotions |
| ReLU activations            | Standard non‑linear activation that avoids vanishing gradients |

---

## Section 10 — Transfer Learning Models

### 10.1 Strategy

Both MobileNetV2 and ResNet50 are initialized with ImageNet‑pretrained weights and loaded with `include_top=False` to remove their original classification heads. The base weights are frozen during an initial feature extraction phase, and a custom head is added on top:

`GlobalAveragePooling2D → Dropout(0.5) → Dense(256, ReLU) → Dropout(0.3) → Dense(7, Softmax)`

Input images are resized to 224×224 and converted from single‑channel grayscale to pseudo‑RGB by replicating the channel three times.

### 10.2 Model Comparison

| Property               | MobileNetV2          | ResNet50               |
|------------------------|----------------------|------------------------|
| Original dataset       | ImageNet (1.2M, 1000 classes) | ImageNet |
| Architecture type      | Inverted residuals + depthwise separable convolutions | Deep residual network (skip connections) |
| Total parameters (approx.) | ~3.4M          | ~25M                   |
| Required input size    | 224×224×3           | 224×224×3              |
| Base weights           | Frozen initially    | Frozen initially       |
| Fine‑tuning            | Unfreeze last 30 layers | Unfreeze last 20+ layers |
| Custom head            | GAP → Dropout → Dense(256) → Dense(7) | Same |
| Optimizer              | Adam (lr = 0.0001)  | Adam (lr = 0.0001)     |
| Epochs                 | 20                  | 20                     |

### 10.3 Two‑Phase Training Strategy

Both transfer models use a two‑phase training schedule:

**Phase 1 — Feature Extraction (Epochs 1–10)**  
- Base model frozen  
- Only custom head trained  
- Higher learning rate (0.001)  
- Goal: adapt new head to FER‑2013 features

**Phase 2 — Fine‑Tuning (Epochs 11–20)**  
- Selected deeper layers of the base are unfrozen  
- Lower learning rate (0.0001)  
- Goal: adapt high‑level features to facial expression patterns

### 10.4 Justification for Model Selection

| Model       | Why Selected |
|------------|--------------|
| MobileNetV2 | Lightweight architecture (~3.4M parameters) designed for mobile and edge deployment; widely used in practical applications where memory and speed are constrained. |
| ResNet50    | Deeper residual architecture (~25M parameters) with strong benchmark performance on many vision tasks; represents a high‑capacity transfer learning baseline. |

Together, MobileNetV2 and ResNet50 represent a trade‑off between lightweight and deep architectures, helping to answer which type of transfer learning model, if any, is preferable to a custom CNN on FER‑2013.

### 10.5 Handling the Grayscale‑to‑RGB Mismatch

Applying ImageNet‑pretrained models to FER‑2013 requires addressing the mismatch between large RGB images (224×224×3) and small grayscale images (48×48×1). In this project:

- All images are resized from 48×48 to 224×224 using bilinear interpolation.
- The single grayscale channel is replicated three times to create pseudo‑RGB input `[gray, gray, gray]`.

This simple strategy is consistent with many transfer learning studies on FER‑2013, although it does not fully resolve the domain gap between the two datasets.

---

## Section 11 — Evaluation Metrics

All three models are evaluated on the same test set using a consistent set of metrics to ensure a fair comparison.

### 11.1 Quantitative Metrics

| Metric           | Formula / Method                  | Purpose |
|------------------|-----------------------------------|---------|
| Accuracy         | Correct predictions / Total       | Overall performance baseline |
| Precision (macro)| TP / (TP + FP), averaged over classes | Correctness of positive predictions |
| Recall (macro)   | TP / (TP + FN), averaged over classes | Coverage of actual positives |
| F1‑Score (macro) | 2 × (P × R) / (P + R), averaged   | Balanced metric under class imbalance |
| Per‑class F1     | F1 per emotion                    | Highlights which emotions are hardest |
| Confusion Matrix | `confusion_matrix`                | Visual error pattern analysis |
| Training Time    | Wall‑clock seconds per epoch      | Measures computational cost |
| Parameter Count  | `model.count_params()`            | Compares model complexity |

### 11.2 Why Macro‑Averaged Metrics

Macro‑averaging computes each metric independently per class and then takes the unweighted average across classes. This means that each of the seven emotions contributes equally to the final macro‑precision, macro‑recall, and macro‑F1 scores, regardless of its frequency in the dataset. This approach helps avoid the “accuracy illusion”, where high overall accuracy hides very poor performance on minority emotions such as Disgust.

### 11.3 Explainability Metrics

Explainability is assessed qualitatively but in a structured way:

| Metric                  | Method                     | Purpose |
|-------------------------|----------------------------|---------|
| Grad‑CAM focus quality  | Visual inspection (1–5 scale) | Checks if attention is on meaningful facial regions |
| SHAP interpretability   | Pixel‑level contribution maps | Evaluates whether positive/negative contributions align with key features |
| Misclassification diagnosis | Grad‑CAM on incorrect predictions | Identifies which regions contribute to typical errors |

### 11.4 Final Model Comparison Table

The main quantitative results of the three models on the FER‑2013 test set are summarized as follows:

| Model       | Accuracy | Precision (macro) | Recall (macro) | F1‑Score (macro) | Training Time | Parameters | Grad‑CAM Quality |
|------------|----------|-------------------|----------------|------------------|---------------|-----------|------------------|
| Custom CNN | 58.58%   | 0.5259            | 0.5872         | 0.5364           | 27.3 min      | ~1.2M    | High             |
| MobileNetV2| 50.49%   | 0.4434            | 0.4873         | 0.4440           | 153.5 min     | ~3.4M    | Medium           |
| ResNet50   | 40.78%   | 0.3782            | 0.4110         | 0.3530           | 138.9 min     | ~25M     | Medium           |

Overall, the custom CNN achieved the best balance between performance and efficiency on FER‑2013, despite being the smallest model. The two transfer learning models did not transfer as well as initially expected, which is likely due to the domain gap between ImageNet and FER‑2013.

### 11.5 Implementation Tools

| Tool                                | Purpose |
|-------------------------------------|---------|
| `sklearn.metrics.classification_report` | Computes precision, recall, and F1 per class |
| `sklearn.metrics.confusion_matrix`  | Generates confusion matrices |
| `matplotlib` / `seaborn`           | Visualizes curves and matrices |
| `tf-keras-vis` or Grad‑CAM library | Generates Grad‑CAM heatmaps |
| `shap.GradientExplainer`           | Computes SHAP explanations |
| `model.count_params()`             | Counts model parameters |
| `time` module                      | Measures training time |

---

## Section 12 — Expected Figures and Tables

All figures will be saved as `.pdf` files in the `outputs/figures/` directory of the project repository and generated during model training and evaluation.

### 12.1 Expected Figures

| Figure      | Description                       | Filename                                           |
|-------------|-----------------------------------|---------------------------------------------------|
| Figure 1    | Class distribution bar chart      | `outputs/figures/Fig1_Class_Distribution.pdf`     |
| Figure 2    | Sample images per class           | `outputs/figures/Fig2_Sample_Images.pdf`          |
| Figure 3a   | Training curves — Custom CNN      | `outputs/figures/Fig3a_Training_Curves_Custom_CNN.pdf` |
| Figure 3b   | Training curves — MobileNetV2     | `outputs/figures/Fig3b_Training_Curves_MobileNetV2.pdf` |
| Figure 3c   | Training curves — ResNet50        | `outputs/figures/Fig3c_Training_Curves_ResNet50.pdf` |
| Figure 4a   | Confusion matrix — Custom CNN     | `outputs/figures/Fig4a_Confusion_Matrix_Custom_CNN.pdf` |
| Figure 4b   | Confusion matrix — MobileNetV2    | `outputs/figures/Fig4b_Confusion_Matrix_MobileNetV2.pdf` |
| Figure 4c   | Confusion matrix — ResNet50       | `outputs/figures/Fig4c_Confusion_Matrix_ResNet50.pdf` |
| Figure 5    | Model comparison chart            | `outputs/figures/Fig5_Model_Comparison.pdf`       |
| Figure 6a   | Grad‑CAM — Custom CNN             | `outputs/figures/Fig6a_GradCAM_Custom_CNN.pdf`    |
| Figure 6b   | Grad‑CAM — MobileNetV2            | `outputs/figures/Fig6b_GradCAM_MobileNetV2.pdf`   |
| Figure 6c   | Grad‑CAM — ResNet50               | `outputs/figures/Fig6c_GradCAM_ResNet50.pdf`      |
| Figure 7a   | SHAP — Custom CNN                 | `outputs/figures/Fig7a_SHAP_Custom_CNN.pdf`       |
| Figure 7b   | SHAP — MobileNetV2                | `outputs/figures/Fig7b_SHAP_MobileNetV2.pdf`      |
| Figure 7c   | SHAP — ResNet50                   | `outputs/figures/Fig7c_SHAP_ResNet50.pdf`         |
| Figure 8a   | ROC curves — Custom CNN           | `outputs/figures/Fig8a_ROC_Custom_CNN.pdf`        |
| Figure 8b   | ROC curves — MobileNetV2          | `outputs/figures/Fig8b_ROC_MobileNetV2.pdf`       |
| Figure 8c   | ROC curves — ResNet50             | `outputs/figures/Fig8c_ROC_ResNet50.pdf`          |

### 12.2 Expected Tables

| Table    | Description                     | Location / Filename                                     |
|----------|---------------------------------|--------------------------------------------------------|
| Table 1  | Dataset overview                | Proposal Section 5                                     |
| Table 2  | Class distribution              | Proposal Section 5                                     |
| Table 3  | Custom CNN architecture         | Proposal Section 9                                     |
| Table 4  | Transfer learning comparison    | Proposal Section 10                                    |
| Table 5  | Training hyperparameters        | Proposal Section 8                                     |
| Table 6  | Model comparison results        | `outputs/tables/Table8_Model_Comparison.csv`          |
| Table 7  | Per‑class F1 — Custom CNN       | `outputs/tables/Table7_CustomCNN_PerClass_F1.csv`     |
| Table 8  | Per‑class F1 — MobileNetV2      | `outputs/tables/Table7_MobileNetV2_PerClass_F1.csv`   |
| Table 9  | Per‑class F1 — ResNet50         | `outputs/tables/Table7_ResNet50_PerClass_F1.csv`      |

### 12.3 Figure Priority

Not all figures have equal importance. For this project, the most critical are:

| Priority | Figure          | Why Critical |
|----------|-----------------|-------------|
| High     | Figure 6 (Grad‑CAM on misclassified samples) | Directly addresses RQ4 and RQ5 and supports error analysis |
| High     | Figure 4 (Confusion matrices)                | Exposes per‑class failures and the impact of class imbalance |
| High     | Figure 3 (Training curves)                   | Shows model convergence, overfitting, or underfitting behavior |
| Medium   | Figure 5 (Grad‑CAM on correct predictions)   | Confirms that models focus on meaningful regions when they are correct |
| Medium   | Figure 7 (SHAP visualizations)               | Provides deeper, pixel‑level insight into model decisions |
| Lower    | Figure 1 (Class distribution)                | Basic dataset visualization |
| Lower    | Figure 2 (Workflow diagram)                  | Summarizes the methodology visually |

---

## References

### Foundational Papers

1. Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., & Bengio, Y. (2013). Challenges in representation learning: A report on three machine learning contests. *International Conference on Machine Learning (ICML 2013).* https://arxiv.org/abs/1307.0414  

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),* 770–778. https://arxiv.org/abs/1512.03385  

3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),* 4510–4520. https://arxiv.org/abs/1801.04381  

4. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad‑CAM: Visual explanations from deep networks via gradient‑based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV),* 618–626. https://arxiv.org/abs/1610.02391  

5. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS),* 30, 4765–4774. https://arxiv.org/abs/1705.07874  

### FER‑Specific Papers

6. Pramerdorfer, C., & Kampel, M. (2016). Facial expression recognition using convolutional neural networks: State of the art. *arXiv preprint.* https://arxiv.org/abs/1612.02903  

7. Li, S., & Deng, W. (2022). Deep facial expression recognition: A survey. *IEEE Transactions on Affective Computing,* 13(3), 1195–1215. https://doi.org/10.1109/TAFFC.2020.2981446  

8. Tran, M., et al. (2021). Deep‑Emotion: Facial expression recognition using attentional convolutional network. *Sensors,* 21(11), 3046. https://doi.org/10.3390/s21113046  

9. Tutuianu, G. I., Liu, Y., Alamäki, A., & Kauttonen, J. (2024). Benchmarking deep facial expression recognition: An extensive protocol with balanced dataset in the wild. *Engineering Applications of Artificial Intelligence,* 133, 108896. https://doi.org/10.1016/j.engappai.2024.108896  

10. Mandave, D. D., & Patil, L. V. (2025). A statistically rigorous comparison of MobileNetV2 and EfficientNet‑B0 for facial expression recognition on the FER‑2013 benchmark. *Information Dynamics and Applications,* 4(4). https://doi.org/10.56578/ida040401  

---

*Document Status: Phase 2 — Final Proposal Draft (June 2026)*
