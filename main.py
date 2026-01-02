import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
"""
Rand Saleh 1221124
Roa Makhtoob 1221636
"""
def load_and_preprocess_images(dataset_path, image_size=(32, 32), use_hog=False):
    """
    Loads all images from the dataset folder, resizes them, and extracts features.
    If use_hog is True, it applies HOG feature extraction.
    Otherwise, it flattens the resized image into a 1D array.
    """
    X, y = [], []
    for class_name in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_folder):
            continue
        for img_name in os.listdir(class_folder):
            try:
                img_path = os.path.join(class_folder, img_name)
                img = imread(img_path)
                img_resized = resize(img, image_size, anti_aliasing=True)

                if use_hog:
                    img_gray = rgb2gray(img_resized)
                    features = hog(
                        img_gray,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        orientations=9,
                        block_norm='L2-Hys',
                        feature_vector=True
                    )
                    X.append(features)
                else:
                    X.append(img_resized.flatten())

                y.append(class_name)
            except Exception as e:
                print(f"Couldn't load {img_path}: {e}")
    print(f"Total images loaded: {len(X)} | HOG used: {use_hog}")
    return np.array(X), np.array(y)

def encode_split_and_scale(X, y, test_size=0.2, scale=True, apply_pca=False, pca_components=100):
    """
    Encodes labels, splits data, and optionally scales and applies PCA.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if apply_pca:
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test, label_encoder

def train_naive_bayes(X_train, X_test, y_train, y_test, label_encoder):
    model = GaussianNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("=== Naive Bayes Results ===")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

def train_decision_tree(X_train, X_test, y_train, y_test, label_encoder):
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=3,  # Shallow tree for visualization
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("=== Decision Tree Results (HOG Features) ===")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # Visualize top few levels of the tree
    plt.figure(figsize=(24, 12))
    plot_tree(model, class_names=label_encoder.classes_, filled=True, max_depth=3)
    plt.title("Decision Tree Visualization (HOG Features, Depth = 3)") 
    plt.show()

def train_mlp_classifier(X_train, X_test, y_train, y_test, label_encoder):
    model = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu',
                          solver='adam', max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("=== MLP Classifier Results ===")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

def run_image_classification_project(dataset_path):
    # Naive Bayes Report (Raw Pixel Values)
    print("Preparing data for Naive Bayes")
    X_nb, y_nb = load_and_preprocess_images(dataset_path, use_hog=False)
    X_train_nb, X_test_nb, y_train_nb, y_test_nb, label_encoder_nb = encode_split_and_scale(
        X_nb, y_nb, scale=False, apply_pca=False)
    print("\nTraining Naive Bayes model...")
    train_naive_bayes(X_train_nb, X_test_nb, y_train_nb, y_test_nb, label_encoder_nb)

    # Decision Tree Report (HOG Features)
    print("\nPreparing data for Decision Tree (HOG features)...")
    X_dt, y_dt = load_and_preprocess_images(dataset_path, use_hog=True)
    X_train_dt, X_test_dt, y_train_dt, y_test_dt, label_encoder_dt = encode_split_and_scale(
        X_dt, y_dt, scale=True, apply_pca=False)
    print("\nTraining Decision Tree model...")
    train_decision_tree(X_train_dt, X_test_dt, y_train_dt, y_test_dt, label_encoder_dt)

    # MLP Report (HOG + PCA)
    print("\nPreparing data for MLP (HOG + PCA)...")
    X_mlp, y_mlp = X_dt, y_dt  # Reusing HOG features
    X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp, label_encoder_mlp = encode_split_and_scale(
        X_mlp, y_mlp, scale=True, apply_pca=True, pca_components=100)
    print("\nTraining MLP Classifier...")
    train_mlp_classifier(X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp, label_encoder_mlp)

# Run the full pipeline
run_image_classification_project(r"C:\Users\Lenovo\Desktop\AI\Dataset")