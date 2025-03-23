# OCTMNIST Classification

This project focuses on classifying retinal diseases using the OCTMNIST dataset, a collection of grayscale optical coherence tomography (OCT) images. The aim is to implement and improve deep learning models for accurate medical image classification.

---

## 📊 Dataset Overview

- **Source**: OCTMNIST (part of MedMNIST)
- **Image Dimensions**: 28×28 pixels (grayscale)
- **Classes**: 4 retinal disease categories (labeled 0–3)
- **Size**:
  - Training: 97,477 images
  - Validation: 10,832 images
  - Test: 1,000 images (250 per class)

---

## 📌 Key Statistics

- **Mean Pixel Intensity**: Computed across dataset
- **Standard Deviation**: For each class
- **Class Distribution**:
  - Train: [33,484, 10,213, 7,754, 46,026]
  - Validation: [3,721, 1,135, 862, 5,114]
  - Test: [250, 250, 250, 250]
- **No missing values**

---

## 📷 Visualizations

- Sample image grid from each class
- Histogram of pixel intensities
- Class distribution plot
- Standard deviation per class

---

## 🧠 Models Implemented

### 🔹 Initial Model
Basic CNN model with fewer layers, served as a baseline.

### 🔹 Improved CNN Architecture

- **Conv1**: 32 filters, 3x3 kernel + BatchNorm + ReLU + MaxPooling
- **Conv2**: 64 filters, 3x3 kernel + BatchNorm + ReLU + MaxPooling
- **Conv3**: 128 filters, 3x3 kernel + BatchNorm + ReLU + MaxPooling
- **FC1**: 256 neurons + ReLU + Dropout (p=0.5)
- **FC2**: Output layer with 4 neurons (Softmax)
- **Loss**: CrossEntropyLoss (best-performing)
- **Accuracy**: 
  - Initial: 91.00%
  - Improved: **93.64%**

---

## ⚙️ Training Details

- **Optimizer**: AdamW with weight decay
- **Scheduler**: OneCycleLR
- **Regularization Techniques**:
  - **Dropout (0.5)**: Prevent overfitting
  - **Batch Normalization**: Improves training stability
  - **Early Stopping**: `patience=2` to halt overfitting
- **Data Augmentation**:
  - Random Rotation (±10°)
  - Random Horizontal Flip (p=0.5)

---

## 🔍 Loss Function Comparison

| Loss Function       | Notes |
|---------------------|-------|
| CrossEntropyLoss    | ✅ Best performance, stable training |
| FocalLoss           | Slower convergence, sensitive to imbalance |
| LabelSmoothingLoss  | Reduced overconfidence, but lower precision |

---

## 📈 Results Summary

| Model           | Accuracy |
|----------------|----------|
| Initial CNN    | 91.00%   |
| Improved CNN   | 93.64%   |

---

## 🧪 Future Improvements

- Use larger image sizes (56×56) for higher detail
- Explore transfer learning (e.g., ResNet, EfficientNet)
- Try focal loss with class weighting
- Apply advanced augmentation (CutMix, MixUp)

---

## 📚 References

- [MedMNIST Dataset](https://medmnist.com/)
- Kingma & Ba (2015), *Adam Optimizer*
- Paszke et al. (2019), *PyTorch Library*
- He et al. (2016), *Deep Residual Learning*
- [Focal Loss Paper](https://arxiv.org/pdf/1708.02002v2)
- [Label Smoothing Paper](https://paperswithcode.com/paper/when-does-label-smoothing-help)

---

## 🩺 Disclaimer

This project is for academic and research purposes only. Do not use it for clinical diagnosis or treatment.

