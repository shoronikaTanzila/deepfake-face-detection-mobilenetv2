# 🔍 Deepfake & AI Face Detection via MobileNetV2 (RGB, ELA & SRM)

An end-to-end computer vision and image forensics framework designed to classify **Real vs. Synthetic/Fake (Deepfake)** faces using **MobileNetV2** along with forensic feature extraction methods including **Error Level Analysis (ELA)** and **Spatial Rich Model (SRM)**.

---

## 📌 Project Overview
With the rise of generative AI and deepfake technologies, identifying manipulated faces is crucial for digital media security. This project implements a lightweight and robust classification pipeline that analyzes:
1. **Standard Color Features (RGB)** for visual pattern recognition.
2. **Error Level Analysis (ELA)** to uncover compression discrepancies and post-processing artifacts.
3. **Spatial Rich Model (SRM)** filtering to expose high-frequency noise variations and pixel tamper fingerprints.

---

## 🔬 Feature Extraction & Methodologies

### 1. Multi-Representation Image Processing
- **Standard RGB (224x224)**: Preserves spatial facial structures and color distributions.
- **Error Level Analysis (ELA)**: Re-saves the image at 90% JPEG quality and calculates the difference extrema to detect digital tampering.
- **Spatial Rich Model (SRM)**: Applies high-pass edge/noise residual convolution kernels:
  $$\begin{bmatrix} -1 & 2 & -1 \\ 2 & -4 & 2 \\ -1 & 2 & -1 \end{bmatrix}$$
  to extract hidden frequency noise left behind by GANs and diffusion generators.

### 2. Deep Learning Architecture
- **Backbone**: **MobileNetV2** (pre-trained on ImageNet) utilized for efficient, low-latency feature extraction.
- **Classification Head**: Global Average Pooling followed by Dropout and Dense layers with Sigmoid activation for binary classification (`Real` vs `Fake`).
- **Framework**: TensorFlow / Keras with OpenCV and PIL.

---

## 📊 Dataset Structure
The dataset consists of binary-labeled facial images partitioned into train, validation, and test subsets:

```text
real_vs_fake/
├── train/
│   ├── fake/
│   └── real/
├── valid/
│   ├── fake/
│   └── real/
└── test/
    ├── fake/
    └── real/
