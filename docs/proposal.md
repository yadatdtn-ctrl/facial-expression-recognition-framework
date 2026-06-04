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

Facial expression recognition (FER) is an emotional processing problem used in healthcare systems (eg., mental health screening), education (student engagement), transportation (driver safety), and human-computer interaction. Automatic face detection from images can help support systems in responding to users' emotions and allow experts to identify expressions without sound and speech.

Convolutional Neural Networks (CNNs) are a widely used strategy in FER because they can learn directly from the hierarchical structure of image features from raw pixels. However, a key question remains: with datasets like FER-2013 (48x48 grayscale), is it better to build a smaller CNN model for this problem, or should we rely on transfer learning for larger models trained in high-resolution RGB image scenarios, such as those on ImageNet? Reported accuracy in FER-2013 is far from perfect, primarily due to the trade-off between custom CNNs and transfer learning at such a low resolution, which is difficult to understand.

Transfer learning models, such as MobileNetV2 and ResNet50, can handle pre-trained features powerfully, but suffer from domain mismatch issues: FER-2013 uses 48x48 grayscale, which encounters noise and class imbalance problems, but they anticipate 224x224 RGB input. For 48x48 grayscale, smaller CNN models can be specifically built and may achieve better performance with fewer parameters and lower computational costs. The dataset has challenges: Disgust only has 436 examples, whereas Happy has 7,215; labels are crowdsourced, and the occlusions and facial locations differ.

A major drawback of many deep models, aside from accuracy, is their difficulty in explanation. Understanding the reasons behind a model's forecast of a certain emotion is nearly as important as the prediction itself in human-centered applications such as FER. SHAP can display pixel-level relevance, while Grad-CAM can reveal which facial regions a model uses. When combined, they aid in diagnosing misclassifications and determining whether the models truly concentrate on important facial features.

This study looked at a custom CNN, MobileNetV2, and ResNet50, all tested on the FER-2013 dataset. To comprehend how each model learned, we employed SHAP and Grad-CAM. The primary objective was to determine which approach provided the best combination of accuracy, affordability, and ease of comprehension. We see this final system as a tool to help human experts make decisions, not to replace them.

---

## Section 3 — Problem Statement

Facial expression recognition from static images is a challenging multi‑class classification problem. There is high intra‑class variability (the same emotion looks different across people, ages, cultures, and intensities) and strong inter‑class similarity (e.g., Fear vs. Surprise, Angry vs. Disgust), especially when working with low‑resolution images. FER‑2013 adds further difficulty because it contains 48×48 grayscale faces, which limits fine‑grained detail that could help separate similar emotions.

The training split of FER‑2013 is severely imbalanced: Happy has 7,215 samples while Disgust has only 436, a 16.5‑to‑1 ratio. Under standard training this pushes models to favor majority classes and to ignore minority ones. Although many CNN‑based FER methods exist, direct comparisons between a domain‑specific custom CNN and transfer learning models (MobileNetV2, ResNet50) on low‑resolution grayscale images are still limited. The combined use of Grad‑CAM and SHAP to analyze both correct and incorrect predictions on FER‑2013 is also not very common.

This project addresses these gaps by training three CNN‑based models (custom CNN, MobileNetV2, ResNet50) on FER‑2013 under the same conditions, reporting both global and per‑class metrics, and using Grad‑CAM and SHAP to inspect how each model behaves, especially on minority emotions. A key question is whether a compact custom CNN can actually outperform heavier pretrained models on this specific grayscale, low‑resolution task.

---

## Section 4 — Selected Dataset and Dataset Link

| Field | Details |
|-------|---------|
| **Dataset Name** | FER‑2013 (Facial Expression Recognition 2013) |
| **Source / Kaggle Link** | https://www.kaggle.com/datasets/msambare/fer2013 |
| **Original Source** | Introduced at ICML 2013 (Goodfellow et al., 2013) |
| **License** | Publicly available for academic use |
| **Format** | Grayscale images in folder structure (train/test) |

FER‑2013 is a standard benchmark for FER and is suitable for this project because it is publicly available, has a reasonable size, and already includes a clear train/test split.

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

The dataset contains only faces, each labeled with one of seven basic emotions and resized to 48×48 pixels. This size keeps the images small and uniform but also makes the task harder because much of the subtle facial detail is lost.

### 5.2 Emotion Classes and Training Distribution

| Emotion  | Training Samples | Percentage |
|----------|-----------------|------------|
| Angry    | 3,995 | 13.9% |
| Disgust  | 436   | 1.5%  |
| Fear     | 4,097 | 14.3% |
| Happy    | 7,215 | 25.1% |
| Neutral  | 4,965 | 17.3% |
| Sad      | 4,830 | 16.8% |
| Surprise | 3,171 | 11.0% |
| **TOTAL**| **28,709** | **100%** |

Happy clearly dominates the dataset while Disgust is extremely rare. This imbalance has a strong impact on training and evaluation and needs to be handled explicitly.

### 5.3 Class Imbalance Analysis

The 16.5‑to‑1 ratio between Happy and Disgust means that a model can achieve a decent overall accuracy while almost never predicting Disgust. Without class weighting or other imbalance strategies, minority emotions remain poorly learned. In this project, class weights are computed from the training set using `sklearn.utils.class_weight.compute_class_weight` and applied during training for all three models.

### 5.4 Dataset Challenges

The FER‑2013 dataset has several known challenges:

- Low spatial resolution (48×48 pixels), which makes it hard to capture subtle facial cues.  
- Grayscale format, which removes color information that would help humans (for example, redness or paleness).  
- Noisy labels, since annotations were collected via crowdsourcing rather than experts.  
- Variations in pose and occlusion, such as side views or faces partly covered by hands and hair.  
- High intra‑class and inter‑class variability, with similar facial patterns sometimes corresponding to different emotions.

As a result, even strong CNN models usually reach only mid‑60% accuracy on FER‑2013, which is noticeably lower than on more controlled laboratory datasets like CK+.

---

## Section 6 — Research Questions

| RQ   | Research Question |
|------|------------------|
| **RQ1** | How accurately can CNN‑based models perform seven‑class facial expression recognition on FER‑2013? |
| **RQ2** | Which model family—custom CNN or transfer learning (MobileNetV2, ResNet50)—offers the best balance between predictive performance and computational efficiency? |
| **RQ3** | How do preprocessing, data augmentation, and class‑imbalance handling influence robustness and per‑class performance? |
| **RQ4** | Do Grad‑CAM explanations show that the trained models focus on meaningful facial regions such as eyes, mouth, and eyebrows? |
| **RQ5** | What failure patterns and practical risks appear when deploying this framework in real human‑centered settings? |

These questions guide the modeling choices, evaluation metrics, and explainability analyses in later sections.

---

## Section 7 — Expected and Actual Results

This section first summarizes the expectations formed from the literature before training, and then briefly compares them with the actual results from the experiments. Initially, transfer learning models were expected to outperform the custom CNN. However, the final experiments showed the opposite: the custom CNN achieved the best test accuracy and macro‑F1 on FER‑2013.

### 7.1 Expected Classification Performance (Before Experiments)

Based on prior work on FER‑2013, the following performance ranges were expected:

| Model                            | Expected Accuracy | Expected Macro‑F1 |
|----------------------------------|-------------------|-------------------|
| Custom CNN (from scratch)        | 63–70%            | 0.55–0.65         |
| MobileNetV2 (transfer learning)  | 65–72%            | 0.65–0.70         |
| ResNet50 (transfer learning)     | 65–73%            | 0.62–0.70         |

These ranges reflect reported results for deep CNN and transfer learning methods on FER‑2013. It was assumed that MobileNetV2 and ResNet50, with richer pretrained features, would reach higher accuracy than a smaller custom CNN.

### 7.2 Actual Classification Performance

After running the full experiments and evaluating on the FER‑2013 test set, the results were:

| Model        | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------------|---------------|-----------------|-------------|----------|
| Custom CNN  | 58.58%        | 0.5259          | 0.5872      | 0.5364   |
| MobileNetV2 | 50.49%        | 0.4434          | 0.4873      | 0.4440   |
| ResNet50    | 40.78%        | 0.3782          | 0.4110      | 0.3530   |

Contrary to the initial hypothesis, the custom CNN achieved the best overall performance, while both transfer learning models performed worse, suggesting that the domain gap between ImageNet (high‑resolution RGB) and FER‑2013 (low‑resolution grayscale) is not fully bridged by the chosen fine‑tuning strategy.

### 7.3 Expected Per‑Class Performance

The class imbalance suggested that per‑class performance would vary considerably. Before training, the following pattern was expected:

| Emotion  | Expected F1 | Reason |
|----------|------------|--------|
| Happy    | 0.85–0.92  | Largest class; visually distinct |
| Neutral  | 0.60–0.75  | Well‑represented; less confusion |
| Angry    | 0.50–0.65  | High intra‑class variation; confusable with Disgust |
| Sad      | 0.50–0.65  | Overlaps with Neutral and Fear |
| Fear     | 0.30–0.50  | Similar to Surprise and Sad |
| Disgust  | 0.20–0.40  | Severely underrepresented; hard to learn |
| Surprise | 0.55–0.70  | Distinct, but overlaps with Fear |

In the actual results, Disgust and Fear did indeed remain challenging across all models, and Happy and Surprise were among the stronger classes, but the absolute F1 values were lower than these optimistic predictions. The custom CNN still showed more balanced performance than the transfer learning models.

### 7.4 Expected and Actual Computational Cost

Before training, the expected computational trade‑offs were:

| Model       | Expected Training Time | Parameters (approx.) |
|------------|------------------------|----------------------|
| Custom CNN | Fastest (~30–60 min)   | ~0.5M–2M             |
| MobileNetV2| Moderate (~1–2 hours)  | ~3.4M                |
| ResNet50   | Slowest (~2–3 hours)   | ~25M                 |

The experiments confirmed this pattern. On the chosen GPU environment, the custom CNN (about 1.2M parameters) trained in ~27.3 minutes, MobileNetV2 (about 3.4M parameters) in ~153.5 minutes, and ResNet50 (about 25M parameters) in ~138.9 minutes. Despite being the lightest and fastest, the custom CNN achieved the best performance, challenging the assumption that larger pretrained models would be superior on FER‑2013.

### 7.5 Expected Explainability Findings

It was expected that Grad‑CAM would highlight key facial regions (eyes, mouth, eyebrows) for correct predictions, and that misclassifications would have more diffuse or misplaced activations, sometimes even focusing on the background. SHAP was expected to show positive contributions concentrated around meaningful facial features for confident predictions, with scattered contributions for uncertain or incorrect ones.

The qualitative Grad‑CAM and SHAP results broadly matched these expectations. Correct predictions often showed focused attention on central facial regions, while some errors showed attention drifting away from the key facial areas, which helped explain recurring confusions such as Disgust vs. Angry or Fear vs. Surprise.

---

## Section 8 — Proposed Methodology

This project follows a simple but complete pipeline: dataset preparation, preprocessing, model design, training, evaluation, and explainability analysis. The idea is to treat all three models as fairly as possible so that differences in results mainly come from the architectures themselves.

### 8.1 Dataset Selection and Preparation

The FER‑2013 dataset from Kaggle is used as the only dataset. The original folder structure provides a training and test split, and the training portion is further divided into a new training and validation set using stratified splitting:

| Split      | Images | Percentage |
|-----------|--------|-----------|
| Training   | 22,967 | 64% |
| Validation | 5,742  | 16% |
| Test       | 7,178  | 20% |

Before training, the dataset is checked for corrupted or unreadable files and the class distribution is plotted (Figure 1). Class weights are then computed from the training set to reduce the impact of the strong class imbalance.

### 8.2 Preprocessing Pipeline

All images are normalized by dividing pixel values by 255 to obtain values in the range [0, 1]. The main preprocessing steps for each model family are:

| Step             | Custom CNN            | MobileNetV2 & ResNet50         |
|------------------|-----------------------|--------------------------------|
| Resize           | 48×48 pixels          | 224×224 pixels                 |
| Color format     | Grayscale (1 channel) | Pseudo‑RGB (3 channels)        |
| Normalization    | Divide by 255 → [0,1] | Divide by 255 → [0,1]          |
| Channel conversion | —                   | Replicate grayscale channel ×3 |

For transfer learning models, the single grayscale channel is replicated three times to form a 3‑channel pseudo‑RGB image suitable for ImageNet‑pretrained backbones.

### 8.3 Data Augmentation

Data augmentation is applied only to the training set to improve robustness and reduce overfitting. The actual `ImageDataGenerator` configuration uses:

| Augmentation      | Value           |
|-------------------|-----------------|
| Horizontal Flip   | 50% probability |
| Random Rotation   | ±15 degrees     |
| Random Zoom       | ±10%            |
| Random Brightness | ±20%            |

No random contrast, width shift, or height shift is used, and no augmentation is applied to validation or test sets. This keeps evaluation unbiased while still encouraging the models to generalize across small variations.

### 8.4 Train / Validation / Test Split

The predefined test split from FER‑2013 is kept as the final test set. The training portion from Kaggle is split into new training and validation subsets using stratified sampling, preserving the original class distributions in both sets. This makes validation performance more representative of real test performance.

### 8.5 Model Development

Three CNN‑based models are developed and trained under as similar conditions as possible:

**Model 1 — Custom CNN (from scratch)**  
A compact CNN tailored to 48×48 grayscale input. It has four convolutional blocks, each with two convolutional layers and Batch Normalization, followed by MaxPooling and Dropout. Global Average Pooling replaces Flatten, and a small dense head produces the final 7‑class output. Full details are in Section 9.

**Model 2 — MobileNetV2 (transfer learning)**  
MobileNetV2 pretrained on ImageNet is loaded with `include_top=False`. A custom head (`GlobalAveragePooling2D → Dropout(0.5) → Dense(256, ReLU) → BatchNormalization → Dropout(0.3) → Dense(7, Softmax)`) is added. The base is first frozen for feature extraction, then the last 30 layers are unfrozen for fine‑tuning.

**Model 3 — ResNet50 (transfer learning)**  
ResNet50 pretrained on ImageNet is also loaded with `include_top=False` and the same custom head. The training strategy mirrors MobileNetV2: first train only the head, then unfreeze and fine‑tune the last 30 layers of the base network.

### 8.6 Training Setup and Hyperparameters

All models are trained on Kaggle Notebooks with GPU acceleration (NVIDIA Tesla P100). The main hyperparameters are:

| Hyperparameter | Custom CNN | MobileNetV2 | ResNet50 |
|----------------|-----------|-------------|----------|
| Optimizer      | Adam      | Adam        | Adam     |
| Initial LR     | 0.001     | 0.0001      | 0.0001   |
| Loss Function  | Categorical Cross‑Entropy | Categorical Cross‑Entropy | Categorical Cross‑Entropy |
| Batch Size     | 32        | 32          | 32       |
| Epochs (max)   | 30        | 20          | 20       |
| Class Weights  | Yes       | Yes         | Yes      |

Callbacks used for all models:

- EarlyStopping (patience = 5, monitoring validation loss).  
- ModelCheckpoint (saving best weights by validation accuracy).  
- ReduceLROnPlateau (factor 0.5, patience = 3 on validation loss).

### 8.7 Evaluation Strategy

Each model is evaluated on the held‑out test set using the same metrics:

| Metric           | Method / Tool                         |
|------------------|----------------------------------------|
| Accuracy         | `model.evaluate` on test generator     |
| Precision (macro)| `sklearn.metrics.classification_report` |
| Recall (macro)   | `sklearn.metrics.classification_report` |
| F1‑Score (macro) | `sklearn.metrics.classification_report` |
| Per‑class metrics| `classification_report` per emotion    |
| Confusion Matrix | `sklearn.metrics.confusion_matrix`     |
| Training Time    | Wall‑clock (Python `time` module)      |
| Parameter Count  | `model.count_params()`                 |

Macro‑averaged metrics are used so that each emotion contributes equally, regardless of class size.

### 8.8 Explainability Analysis (Grad‑CAM and SHAP)

Grad‑CAM is computed for the final convolutional layer of each model to produce heatmaps over the input faces, and SHAP (GradientExplainer) is used to estimate pixel‑level contributions. In practice:

- Grad‑CAM is applied to several correctly classified and misclassified test images for each model, with the resulting heatmaps saved as PDF files (Figures 6a–6c).  
- SHAP is applied to a small set of test images with a background of 50 training images, generating visualizations that show positive (red) and negative (blue) contributions to each prediction (Figures 7a–7c).  

Together, these methods help answer whether the models focus on meaningful facial regions and how their attention patterns change when they make mistakes.

### 8.9 Deployment Considerations and Ethics

The final system is presented as a decision‑support tool. It is not intended to replace human judgement but to assist experts in contexts such as education or healthcare. Some key points:

- Computationally, the custom CNN is fastest and lightest, while MobileNetV2 provides a good compromise; ResNet50 is the heaviest and slowest.  
- Class imbalance means that certain emotions (especially Disgust and Fear) remain less reliable, so decisions based purely on these predictions would be risky.  
- Ethical issues include privacy, consent, and the risk of misuse in surveillance or profiling. Real‑world deployment should involve clear communication with users and human oversight.  

---

## Section 9 — CNN Model Design (Custom CNN)

### 9.1 Architecture Rationale

The custom CNN is designed specifically for 48×48 grayscale FER‑2013 images, aiming to balance representational power and computational cost. It uses four convolutional blocks, and each block contains two Conv2D layers with the same number of filters (32, 64, 128, 256), followed by Batch Normalization and MaxPooling. This “two‑conv‑per‑block” pattern lets the network learn richer features at each scale. A Dropout layer with rate 0.25 is applied after each pooling block for early regularization, and Global Average Pooling is used instead of Flatten before a small dense head.

In the classifier head, Dropout(0.5) is applied before a Dense(256, ReLU) layer with Batch Normalization, followed by Dropout(0.3) and a final Dense(7, Softmax) output layer. This setup matches the actual Keras implementation in the notebook.

### 9.2 Architecture Table

| Layer      | Type              | Output Shape   | Filters / Units | Activation |
|-----------|-------------------|----------------|-----------------|------------|
| Input     | Input             | 48 × 48 × 1    | —               | —          |
| Conv_1a   | Conv2D            | 48 × 48 × 32   | 32 @ 3×3        | ReLU       |
| Conv_1b   | Conv2D            | 48 × 48 × 32   | 32 @ 3×3        | ReLU       |
| BN_1      | BatchNorm         | 48 × 48 × 32   | —               | —          |
| Pool_1    | MaxPool2D         | 24 × 24 × 32   | 2×2             | —          |
| Dropout_1 | Dropout           | 24 × 24 × 32   | rate = 0.25     | —          |
| Conv_2a   | Conv2D            | 24 × 24 × 64   | 64 @ 3×3        | ReLU       |
| Conv_2b   | Conv2D            | 24 × 24 × 64   | 64 @ 3×3        | ReLU       |
| BN_2      | BatchNorm         | 24 × 24 × 64   | —               | —          |
| Pool_2    | MaxPool2D         | 12 × 12 × 64   | 2×2             | —          |
| Dropout_2 | Dropout           | 12 × 12 × 64   | rate = 0.25     | —          |
| Conv_3a   | Conv2D            | 12 × 12 × 128  | 128 @ 3×3       | ReLU       |
| Conv_3b   | Conv2D            | 12 × 12 × 128  | 128 @ 3×3       | ReLU       |
| BN_3      | BatchNorm         | 12 × 12 × 128  | —               | —          |
| Pool_3    | MaxPool2D         | 6 × 6 × 128    | 2×2             | —          |
| Dropout_3 | Dropout           | 6 × 6 × 128    | rate = 0.25     | —          |
| Conv_4a   | Conv2D            | 6 × 6 × 256    | 256 @ 3×3       | ReLU       |
| Conv_4b   | Conv2D            | 6 × 6 × 256    | 256 @ 3×3       | ReLU       |
| BN_4      | BatchNorm         | 6 × 6 × 256    | —               | —          |
| Pool_4    | MaxPool2D         | 3 × 3 × 256    | 2×2             | —          |
| Dropout_4 | Dropout           | 3 × 3 × 256    | rate = 0.25     | —          |
| GAP       | GlobalAvgPool2D   | 256            | —               | —          |
| Dropout_5 | Dropout           | 256            | rate = 0.5      | —          |
| Dense_1   | Dense             | 256            | 256 units       | ReLU       |
| BN_5      | BatchNorm         | 256            | —               | —          |
| Dropout_6 | Dropout           | 256            | rate = 0.3      | —          |
| Output    | Dense             | 7              | 7 units         | Softmax    |

### 9.3 Design Decisions

| Decision                               | Justification |
|----------------------------------------|--------------|
| 4 convolutional blocks with 2 convs each | Enough depth for 48×48 images while keeping parameter count manageable |
| Filter doubling (32→256) across blocks | Progressive feature abstraction from low‑level edges to more complex facial patterns |
| Batch Normalization after conv layers  | Stabilizes training and allows slightly higher learning rates |
| MaxPooling 2×2 after each block        | Reduces spatial dimensions and helps control overfitting |
| Dropout(0.25) after each block         | Early regularization to reduce overfitting in deeper parts of the network |
| Global Average Pooling instead of Flatten | Reduces parameters and improves generalization on a limited dataset |
| Dropout(0.5 and 0.3) in classifier head | Further regularization before and after the dense layer |
| Softmax output over 7 units            | Supports seven‑class emotion classification |

---

## Section 10 — Transfer Learning Models

### 10.1 Strategy

MobileNetV2 and ResNet50 are both loaded with ImageNet‑pretrained weights and `include_top=False`. A shared custom classification head is attached:  
`GlobalAveragePooling2D → Dropout(0.5) → Dense(256, ReLU) → BatchNormalization → Dropout(0.3) → Dense(7, Softmax)`.  
During training, both models follow a two‑phase strategy: first, train only the new head with the base frozen; second, unfreeze and fine‑tune the last 30 layers of the base network.

### 10.2 Model Comparison

| Property               | MobileNetV2                | ResNet50                    |
|------------------------|----------------------------|-----------------------------|
| Original Dataset       | ImageNet (1.2M, 1000 classes) | ImageNet                |
| Architecture Type      | Inverted residuals, depthwise separable convs | Deep residual network with skip connections |
| Total Parameters (approx.) | ~3.4M                  | ~25M                        |
| Input Size Required    | 224×224×3                  | 224×224×3                   |
| Base Weights (Phase 1) | Frozen                     | Frozen                      |
| Fine‑Tuning (Phase 2)  | Last 30 layers unfrozen    | Last 30 layers unfrozen     |
| Custom Head            | GAP → Dropout(0.5) → Dense(256) → BN → Dropout(0.3) → Dense(7) | Same      |
| Optimizer              | Adam (lr 0.0001)           | Adam (lr 0.0001)            |
| Epochs (max)           | 20                         | 20                          |

### 10.3 Two‑Phase Training Strategy

Both transfer learning models use the same two‑phase schedule:

**Phase 1 — Feature Extraction**  
- Base model layers are frozen (non‑trainable).  
- Only the custom head is trained.  
- Slightly higher learning rate is used to quickly adapt the head to FER‑2013.  

**Phase 2 — Fine‑Tuning**  
- Base model is set to trainable.  
- All layers except the last 30 are re‑frozen, leaving the final 30 layers of the backbone and the full head trainable.  
- A lower learning rate is used to avoid destroying pretrained features.  

### 10.4 Handling the Grayscale‑to‑RGB Mismatch

To adapt 48×48 grayscale images to ImageNet‑pretrained backbones, the following steps are used:

- Resize 48×48 grayscale images to 224×224.  
- Replicate the single grayscale channel three times to form pseudo‑RGB: `[gray, gray, gray]`.  
- Normalize pixel values to [0, 1] as for the custom CNN.

### 10.5 Justification for Model Selection

| Model       | Why Selected |
|------------|-------------|
| MobileNetV2 | Represents a lightweight transfer learning model aimed at mobile and edge devices, with relatively few parameters and efficient depthwise separable convolutions. |
| ResNet50    | Represents a deeper residual model with high capacity and strong ImageNet performance, making it a common baseline for transfer learning. |

Together with the custom CNN, these models cover a spectrum from small domain‑specific to large pretrained architectures, which directly addresses RQ2 and RQ3.

---

## Section 11 — Evaluation Metrics

### 11.1 Quantitative Metrics

All three models are evaluated on the same FER‑2013 test set using:

| Metric            | Formula / Method                  | Purpose                         |
|-------------------|-----------------------------------|---------------------------------|
| Accuracy          | Correct / Total                   | Overall performance             |
| Precision (macro) | Macro‑average from `classification_report` | Quality of positive predictions |
| Recall (macro)    | Macro‑average from `classification_report` | Coverage of true positives      |
| F1‑Score (macro)  | Macro‑average from `classification_report` | Balanced measure with imbalance |
| Per‑class F1      | F1 per emotion                    | Minority class performance      |
| Confusion Matrix  | `confusion_matrix`                | Visual error patterns           |
| Training Time     | Wall‑clock seconds per epoch      | Computational cost              |
| Parameter Count   | `model.count_params()`            | Model complexity                |

Macro‑averaged metrics treat all seven emotions equally, regardless of how many samples each class has, which is important because Disgust is heavily underrepresented.

### 11.2 Explainability Metrics

Explainability is evaluated qualitatively:

| Metric                 | Method                         | Purpose |
|------------------------|--------------------------------|---------|
| Grad‑CAM quality       | Visual inspection of heatmaps  | Check if focus is on meaningful facial regions |
| SHAP interpretability  | Pixel‑level contribution maps  | Understand which regions support predictions |
| Misclassification diagnosis | Grad‑CAM on incorrect predictions | Identify visual patterns in errors |

These qualitative assessments support RQ4 and RQ5 and complement the quantitative metrics.

### 11.3 Final Model Comparison Table

The main quantitative results across models are summarized as:

| Model        | Accuracy | Precision | Recall | F1‑Score | Training Time | Parameters | Grad‑CAM Quality |
|-------------|----------|-----------|--------|----------|---------------|------------|------------------|
| Custom CNN  | 58.58%   | 0.5259    | 0.5872 | 0.5364   | 27.3 min      | ~1.2M      | High             |
| MobileNetV2 | 50.49%   | 0.4434    | 0.4873 | 0.4440   | 153.5 min     | ~3.4M      | Medium           |
| ResNet50    | 40.78%   | 0.3782    | 0.4110 | 0.3530   | 138.9 min     | ~25M       | Medium           |

This table shows that the custom CNN is not only the fastest and smallest, but also achieves the best test performance and the most focused Grad‑CAM heatmaps.

### 11.4 Implementation Tools

| Tool / Library      | Purpose                        |
|---------------------|-------------------------------|
| TensorFlow / Keras  | Model implementation and training |
| `ImageDataGenerator`| Preprocessing and augmentation |
| `sklearn.metrics`   | Accuracy, classification report, confusion matrix |
| `matplotlib`, `seaborn` | Visualization of curves, confusion matrices |
| `tf-keras-vis` / custom Grad‑CAM code | Grad‑CAM heatmaps |
| `shap.GradientExplainer` | SHAP explanations |
| Python `time`       | Training time measurement     |

---

## Section 12 — Figures, Tables, and Files

### 12.1 Figures

All figures are saved as PDF files in the `outputs/figures/` directory during notebook execution.

| Figure    | Description                               | Filename                                      |
|----------|-------------------------------------------|-----------------------------------------------|
| Figure 1 | Class distribution bar chart              | `outputs/figures/Fig1_Class_Distribution.pdf` |
| Figure 2 | Sample images per class                   | `outputs/figures/Fig2_Sample_Images.pdf`      |
| Figure 3a| Training curves — Custom CNN              | `outputs/figures/Fig3a_TrainingCurves_CustomCNN.pdf` |
| Figure 3b| Training curves — MobileNetV2             | `outputs/figures/Fig3b_TrainingCurves_MobileNetV2.pdf` |
| Figure 3c| Training curves — ResNet50                | `outputs/figures/Fig3c_TrainingCurves_ResNet50.pdf` |
| Figure 4a| Confusion matrix — Custom CNN             | `outputs/figures/Fig4a_ConfusionMatrix_CustomCNN.pdf` |
| Figure 4b| Confusion matrix — MobileNetV2            | `outputs/figures/Fig4b_ConfusionMatrix_MobileNetV2.pdf` |
| Figure 4c| Confusion matrix — ResNet50               | `outputs/figures/Fig4c_ConfusionMatrix_ResNet50.pdf` |
| Figure 5 | Model comparison bar chart                | `outputs/figures/Fig5_Model_Comparison.pdf`   |
| Figure 6a| Grad‑CAM — Custom CNN                     | `outputs/figures/Fig6a_GradCAM_CustomCNN.pdf` |
| Figure 6b| Grad‑CAM — MobileNetV2                    | `outputs/figures/Fig6b_GradCAM_MobileNetV2.pdf` |
| Figure 6c| Grad‑CAM — ResNet50                       | `outputs/figures/Fig6c_GradCAM_ResNet50.pdf`  |
| Figure 7a| SHAP — Custom CNN                         | `outputs/figures/Fig7a_SHAP_CustomCNN.pdf`    |
| Figure 7b| SHAP — MobileNetV2                        | `outputs/figures/Fig7b_SHAP_MobileNetV2.pdf`  |
| Figure 7c| SHAP — ResNet50                           | `outputs/figures/Fig7c_SHAP_ResNet50.pdf`     |

### 12.2 Tables

Some tables appear directly in this proposal, and others are exported as CSV files from the notebook.

| Table   | Description                       | Location / Filename                                |
|--------|-----------------------------------|----------------------------------------------------|
| Table 1| Dataset overview                  | Proposal Section 5                                 |
| Table 2| Class distribution                | Proposal Section 5                                 |
| Table 3| Custom CNN architecture           | Proposal Section 9                                 |
| Table 4| Transfer learning comparison      | Proposal Section 10                                |
| Table 5| Training hyperparameters          | Proposal Section 8                                 |
| Table 6| Model comparison metrics          | `outputs/tables/Table8_Model_Comparison.csv`       |
| Table 7| Per‑class F1 — Custom CNN         | `outputs/tables/Table7_CustomCNN_PerClass_F1.csv`  |
| Table 8| Per‑class F1 — MobileNetV2        | `outputs/tables/Table7_MobileNetV2_PerClass_F1.csv`|
| Table 9| Per‑class F1 — ResNet50           | `outputs/tables/Table7_ResNet50_PerClass_F1.csv`   |

---

## References

### Foundational Papers

1. Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., & Bengio, Y. (2013). Challenges in representation learning: A report on three machine learning contests. *International Conference on Machine Learning (ICML 2013)*. https://arxiv.org/abs/1307.0414  

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778. https://arxiv.org/abs/1512.03385  

3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.‑C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4510–4520. https://arxiv.org/abs/1801.04381  

4. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad‑CAM: Visual explanations from deep networks via gradient‑based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618–626. https://arxiv.org/abs/1610.02391  

5. Lundberg, S. M., & Lee, S.‑I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774. https://arxiv.org/abs/1705.07874  

### FER‑Specific Papers

6. Pramerdorfer, C., & Kampel, M. (2016). Facial expression recognition using convolutional neural networks: State of the art. *arXiv preprint*. https://arxiv.org/abs/1612.02903  

7. Li, S., & Deng, W. (2022). Deep facial expression recognition: A survey. *IEEE Transactions on Affective Computing*, 13(3), 1195–1215. https://doi.org/10.1109/TAFFC.2020.2981446  

8. Tran, M., et al. (2021). Deep‑Emotion: Facial expression recognition using attentional convolutional network. *Sensors*, 21(11), 3046. https://doi.org/10.3390/s21113046  

9. Tutuianu, G. I., Liu, Y., Alamäki, A., & Kauttonen, J. (2024). Benchmarking deep facial expression recognition: An extensive protocol with balanced dataset in the wild. *Engineering Applications of Artificial Intelligence*, 133, 108896. https://doi.org/10.1016/j.engappai.2024.108896  

10. Mandave, D. D., & Patil, L. V. (2025). A statistically rigorous comparison of MobileNetV2 and EfficientNet‑B0 for facial expression recognition on the FER2013 benchmark. *Information Dynamics and Applications*, 4(4). https://doi.org/10.56578/ida040401  

---

*Document Status: Phase 2 — Final Proposal 
