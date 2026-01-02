#  Image Classification (Naive Bayes, Decision Tree, MLP)

This project compares three machine learning models for **image classification**:
- **Naive Bayes** (baseline using raw pixel values)
- **Decision Tree** (using HOG features)
- **Feedforward Neural Network (MLP)** (using HOG + PCA)

The goal is to study how feature engineering and model complexity affect performance on natural scene images. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

## Dataset

This project uses a subset of the **Intel Image Classification Dataset** (Kaggle), originally published by **Puneet Chawla**.  
The full dataset contains 6 natural scene categories (buildings, forest, glacier, mountain, sea, street). In this study, we selected **three classes**: **glacier**, **forest**, and **mountain**. :contentReference[oaicite:2]{index=2}

### Notes about the dataset folder
- The `datasets/` folder is usually **too big** to upload to GitHub.
- Recommended: keep the dataset locally, and upload only a small sample (optional).

### Download (Kaggle)
Use this dataset page:
```text
Intel Image Classification Dataset (Kaggle) - by Puneet Chawla
https://www.kaggle.com/datasets/puneet6060/intel-image-classification
