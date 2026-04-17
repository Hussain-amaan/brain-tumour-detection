---

# 🧠 Brain Tumor Classification using Deep Learning



---

## 📘 Overview

This project focuses on building a deep learning model to automatically classify **brain MRI scans** into two categories: **Tumor** and **Healthy**. Early and accurate tumor detection is critical in medical diagnostics, and this project demonstrates the application of Convolutional Neural Networks (CNNs) for this task using a publicly available dataset.

---

## 📂 Dataset Description

* **Source**: [Kaggle - Brain Tumor Dataset by Preet Viradiya](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)
* **Total Images**: 3,762 MRI scans
* **Classes**:

  * `yes` (Tumor present)
  * `no` (No tumor)
* **Format**: JPEG images, grayscale/color, varying dimensions
* **Directory Structure**:

  ```
  Brain Tumor Data Set/
  ├── yes/
  └── no/
  ```

---

## 🧠 Model Used: Custom CNN

A custom Convolutional Neural Network (CNN) was built and trained from scratch to classify the MRI scans.

### 📐 Architecture

* 4 × \[Conv2D + ReLU + MaxPooling + Dropout]
* Flatten layer
* Dense(128) + ReLU
* Dropout(0.5)
* Output layer: Dense(1) with Sigmoid activation

### 🔧 Training Configuration

* **Input shape**: 256 × 256 × 3
* **Loss function**: Binary Crossentropy
* **Optimizer**: Adam (lr = 0.0001)
* **Epochs**: 20
* **Data Split**:

  * 70% Training
  * 15% Validation
  * 15% Test

---

## 📊 Evaluation Results

After training, the model achieved the following performance on the test set:

### 📉 Confusion Matrix

```
               Predicted
              |  No  | Yes |
Actual | No   | 168  | 41  |
       | Yes  | 22   | 230 |
```

### 🧮 Metrics

| Metric    | Tumor (`yes`) | Healthy (`no`) | Overall |
| --------- | ------------- | -------------- | ------- |
| Precision | 0.85          | 0.88           |         |
| Recall    | 0.91          | 0.80           |         |
| F1-score  | 0.88          | 0.84           |         |
| Accuracy  | —             | —              | **86%** |

✅ The model demonstrates high recall for tumor cases, meaning it effectively identifies most positive cases — which is vital for medical screening tools.

---

## 🧪 Prediction on New MRI Image

A helper function allows classification of new brain MRI images:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_brain_tumor(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    print(f"Prediction Score: {pred:.4f}")
    return "Tumor" if pred > 0.5 else "Healthy"
```

---

## 💾 Model Saving & Loading

```python
# Save model
model.save("best_brain_tumor_model.h5")

# Load model
from tensorflow.keras.models import load_model
model = load_model("best_brain_tumor_model.h5")
```

---

## 🛠 Tech Stack

* **Language**: Python 3
* **Libraries**: TensorFlow / Keras, NumPy, Matplotlib
* **Platform**: Google Colab

---

## 🔍 Optional Improvements

* ✅ Integrate **Grad-CAM** for visual explanation
* ✅ Add **Streamlit or Gradio** frontend for web demo
* 🧠 Extend to **multi-class** classification (e.g., tumor type)
* 💡 Use **ensemble models** for boosting accuracy further

---

## 📌 How to Use (Quickstart)

1. Clone the repo or open in Google Colab
2. Upload the dataset or link via Kaggle API
3. Train the model using the provided code
4. Predict new images with `predict_brain_tumor()`
5. (Optional) Save and deploy the model

---



## ⭐️ Acknowledgements

* Kaggle dataset by [Preet Viradiya](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)
* TensorFlow community and Keras documentation

---


