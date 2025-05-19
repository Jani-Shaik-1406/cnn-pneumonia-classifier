
# 🩺 From Pixels to Diagnosis: AI in Pneumonia Detection

This project showcases a deep learning-based approach to detect pneumonia from chest X-ray images using transfer learning with EfficientNetB3. It aims to automate and improve the speed and reliability of pneumonia diagnosis, especially in under-resourced regions.

---

## 📌 Project Links

- 🔗 [📂 Code (Google Colab)](https://colab.research.google.com/drive/1_iKq5BDUn905GB22eNnUFaRb-MBSEYgM?usp=sharing)

---

## 🩻 Problem Statement

Manual interpretation of chest X-rays for pneumonia is time-consuming, error-prone, and resource-intensive. This project aims to:

- Automate pneumonia diagnosis from chest X-rays
- Support healthcare in remote areas
- Minimize diagnostic delays and errors

---

## 📊 Dataset Overview

- 📁 Source: [Kaggle – Chest X-ray (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- 👨‍⚕️ 5,856 labeled chest X-rays (Normal vs Pneumonia)
- ✅ Preprocessed: Resized to 224x224, normalized, augmented
- ⚖️ Addressed imbalance with class weighting and oversampling

---

## 🧠 Model Architecture

- 🔍 Base: **EfficientNetB3** (ImageNet weights)
- ➕ Layers: Global Avg Pooling → Dropout (30%) → Dense (Sigmoid)
- 🎯 Loss: Binary Crossentropy with L2 Regularization
- 🚀 Optimizer: Adam + Learning Rate Scheduling

---

## 🛠️ Hyperparameter Tuning

- ✅ Tuned: Learning Rate, Dropout Rate, L2 Regularization
- 🔁 Method: Keras Tuner with Random Search
- ⚙️ Tuning Data: Small subsets (100 train / 50 validation)
- ⏱️ Training: EarlyStopping & ModelCheckpoint

---

## 📈 Training Strategy

- 👨‍🔬 Epochs: Max 8 (early stopping enabled)
- 📦 Batch Size: 32
- 🎛️ Augmentation: Applied only to training data
- 🧠 Callbacks: EarlyStopping & Checkpointing
- ⚙️ Regularization: Dropout & L2 to prevent overfitting

---

## 📊 Model Evaluation

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 83.33%    |
| Precision  | 79.4%     |
| Recall     | 99%       |

- ✅ High Recall ensures nearly all pneumonia cases are detected
- ⚠️ Moderate Precision reflects some false positives (acceptable in medical settings)
- 🔍 Confusion matrix confirms low false negatives and acceptable false positives

---

## ✅ Achievements

- 📦 Built an end-to-end deep learning pipeline for medical diagnosis
- 💻 Fine-tuned EfficientNetB3 for high recall and generalization
- 🌍 Suitable for deployment in low-resource or high-volume clinical settings

---

## 🚀 Future Scope

- 🔁 Expand dataset for better generalization across demographics
- 📱 Deploy lightweight models (MobileNet, TinyML) on edge devices
- 🌐 Develop web/mobile apps with explainable AI (saliency maps)
- 🛠️ Explore model pruning, quantization for real-time usage

---

## 🔍 References

- 📊 Dataset: [Kaggle - Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- 📚 EfficientNet Paper: [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- 🧪 Frameworks: [TensorFlow](https://www.tensorflow.org/api_docs) | [Keras](https://keras.io/)
- 💻 Full Code: [Google Colab](https://colab.research.google.com/drive/1_iKq5BDUn905GB22eNnUFaRb-MBSEYgM?usp=sharing)

---

## 👤 Author

**Jani Shariff Shaik**  
*MS Applied Statistics & Data Science, University of Texas at Arlington*  
Email: shaikjanishariff@gmail.com

---

## 📦 Repository Structure

```
├── README.md
├── code.ipynb  ← [Google Colab Link]
├── chest_xray/ ← [Training, Testing & Validation folders]
└── model_weights/ ← [Best checkpoint]
```
