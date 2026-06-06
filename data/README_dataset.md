# Dataset Information

## FER-2013 — Facial Expression Recognition 2013

| Field | Details |
|-------|---------|
| **Dataset Name** | FER-2013 |
| **Kaggle Link** | https://www.kaggle.com/datasets/msambare/fer2013 |
| **Original Source** | Goodfellow et al., ICML 2013 |
| **License** | Publicly available for academic use |
| **Format** | Grayscale JPG images in folder structure |

---

## Dataset Statistics

| Split | Images |
|-------|--------|
| Training | 28,709 |
| Test | 7,178 |
| **Total** | **35,887** |

---

## Class Distribution (Training Set)

| Emotion | Samples | Percentage |
|---------|---------|------------|
| Angry | 3,995 | 13.9% |
| Disgust | 436 | 1.5% |
| Fear | 4,097 | 14.3% |
| Happy | 7,215 | 25.1% |
| Neutral | 4,965 | 17.3% |
| Sad | 4,830 | 16.8% |
| Surprise | 3,171 | 11.0% |
| **TOTAL** | **28,709** | **100%** |

---

## Image Properties

- **Size:** 48 × 48 pixels
- **Color:** Grayscale (1 channel)
- **Format:** JPG
- **Classes:** 7 emotion categories

---

## How to Download

### Via Kaggle API
```bash
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d fer2013
```

### Via Kaggle Website
1. Go to https://www.kaggle.com/datasets/msambare/fer2013
2. Click **"Download"**
3. Extract the ZIP file

---

## Folder Structure After Download

```
fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```
---

## Citation

**Dataset accessed from:**
https://www.kaggle.com/datasets/msambare/fer2013

**Original dataset paper:**
Goodfellow, I. J., Erhan, D., Carrier, P. L., 
Courville, A., Mirza, M., Hamner, B., & Bengio, Y. (2013). 
Challenges in representation learning: A report on three 
machine learning contests. ICML 2013.
https://arxiv.org/abs/1307.0414
