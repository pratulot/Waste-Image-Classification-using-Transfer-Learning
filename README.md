# Waste Image Classification using Transfer Learning

## 📘 Overview
This project builds an image-classification model to categorize waste items (e.g., **plastic**, **metal**, **glass**, **paper**) using **transfer learning** with **ResNet50**.  
It was originally developed as part of the CS665 Deep Learning Final Project and demonstrates an end-to-end deep-learning pipeline — from a simple baseline model to a fine-tuned ResNet50 classifier.

---

## 🎯 Project Goals
- **Primary Goal:** Accurately classify waste images into predefined categories to support better sorting and recycling.
- **Secondary Goals:**
  - Show clear performance improvement from a deliberately weak baseline (“dumb” model) to a fine-tuned transfer-learning model.
  - Apply standard ML practices — data augmentation, callbacks, evaluation metrics.
  - Present a clean, reproducible workflow suitable for GitHub portfolios.

---

## 🧩 Dataset
- **Source:** https://archive.ics.uci.edu/dataset/908/realwaste
- **Structure:** One folder per class (compatible with `ImageDataGenerator.flow_from_directory`)

data/
└── RealWaste/
├── Plastic/
├── Metal/
├── Glass/
├── Paper/
└── TextileTrash/

- **Loading:** via `ImageDataGenerator` with training/validation split.
- **Note:** The dataset path in the notebook can be changed to match your local setup.

---

## ⚙️ Approach
### 1. Baseline (“Dumb”) Model
- Architecture: Flatten → Dense(softmax)
- Training: 1 epoch, high learning rate
- **Validation Accuracy:** **16.72 %**
- Purpose: Create a weak starting point to highlight transfer-learning gains.

### 2. Transfer Learning — ResNet50
- Base: `ResNet50(weights="imagenet", include_top=False)`
- Custom Head: `GlobalAveragePooling2D → Dense(1024, relu) → Dense(num_classes, softmax)`
- Optimizer: `Adam`, Loss: `categorical_crossentropy`
- Augmentation: rotation, shift, shear, zoom, horizontal flip
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### 3. Fine-Tuning
- Unfrozen ~33 layers of ResNet50 (excluding BatchNorm)
- Lowered learning rate (1e-5) for stable convergence
- **Final Validation Accuracy:** **30.42 %**  
- **Validation Loss:** 1.91  
- **Improvement:** +13.7 percentage points vs. baseline

---

## 📊 Evaluation
- **Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Quantitative Findings:**
- Accuracy improved from **16.7 % → 30.4 %**
- Loss decreased from **~10.9 M → 1.9**
- **Qualitative Findings:**
- Predictions visualized with true vs. predicted labels
- Model learned some patterns but underfit on minority classes

---

## 🧠 Key Learnings & Achievements
- Built and evaluated multiple model stages showing measurable improvement.
- Applied **transfer learning** and **fine-tuning** effectively.
- Demonstrated end-to-end ML workflow (data → model → evaluation).
- Produced clear metrics and visual analyses.
- Organized code in a **reproducible, GitHub-ready** structure.

---

## 📂 Recommended Repo Structure

```
├── CS665_Final_Project.ipynb
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🏁 Results Summary

| Model                 | Validation Accuracy | Validation Loss | Notes                     |
| :-------------------- | :-----------------: | :-------------: | :------------------------ |
| Dumb Baseline         |       16.72 %       |    10,890,099   | Expected poor learner     |
| ResNet50 (TL)         |         27 %        |       1.96      | Initial transfer learning |
| ResNet50 (Fine-tuned) |     **30.42 %**     |     **1.91**    | Significant improvement   |


---

## 📜 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it.

---

## 🙌 Acknowledgements

* **TensorFlow / Keras** — Model development
* **scikit-learn** — Evaluation metrics
* **Matplotlib** — Visualization
