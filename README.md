# Image Classification: Naive Bayes vs Decision Tree vs MLP

**ENCS3340 – Artificial Intelligence | Project #2**

This repository compares three machine learning models for image classification:
- **Gaussian Naive Bayes**
- **Decision Tree**
- **Feedforward Neural Network (MLPClassifier)**

---

## Dataset

This project is based on the **Intel Image Classification Dataset** (Kaggle), created by **Puneet Chawla**.

- Dataset: Natural scene images
- Typical categories include: buildings, forest, glacier, mountain, sea, street
- In our experiments, we focus on selected classes depending on the project setup.

**Download from Kaggle:**
- Intel Image Classification Dataset (Kaggle)

> Note: The full dataset is not included in this GitHub repo because it is large.

---

## What’s Inside

- Data loading + preprocessing
- Feature extraction (if used in your code)
- Training and testing for:
  - Naive Bayes
  - Decision Tree
  - MLP
- Evaluation using accuracy (and any other metrics you implemented)

---

## How to Run

```bash
# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows

# Install requirements
pip install numpy scikit-learn opencv-python scikit-image matplotlib

# Dataset setup (place the Intel dataset locally)


# Run the program
python main.py
