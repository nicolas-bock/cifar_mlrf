from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import typer
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from loguru import logger
from skimage import exposure
from skimage.color import rgb2gray
from skimage.feature import hog, corner_harris, corner_peaks, corner_subpix

from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap

from MLRF.dataset import load_data
import MLRF.model_utils as md
import MLRF.features as ft
from MLRF.config import FIGURES_DIR, PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()

def plot_color_histo_features(image, save_path=None):
    """
    Plot the color histogram features of an image.

    Args:
        image: The image to plot the color histogram features.
        save_path: The path to save the plot.
    """
    gray_image = rgb2gray(image)

    _, axis = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 3]})
    
    # Grascale Image
    axis[1][0].imshow(gray_image, cmap='gray')
    axis[1][1].set_title('Histogram')
    axis[1][0].set_title('Grayscale Image')
    axis[1][0].axis('off')
    hist = exposure.histogram(gray_image)
    axis[1][1].plot(hist[0])

    # Color image
    if image.ndim == 3:
        axis[0][0].imshow(image, cmap='gray')
        axis[0][1].set_title('Histogram')
        axis[0][0].set_title('Original Image')
        axis[0][0].axis('off')
        rgbcolors = ['red', 'green', 'blue']
        for i, mycolor in enumerate(rgbcolors):
            axis[0][1].plot(exposure.histogram(image[...,i])[0], color=mycolor)

        contrast_image = ft.contrast((image * 255).astype(np.uint8))

        axis[2][0].imshow(contrast_image, cmap='gray')
        axis[2][1].set_title('Histogram')
        axis[2][0].set_title('Contrasted Image')
        axis[2][0].axis('off')
        rgbcolors = ['red', 'green', 'blue']
        for i, mycolor in enumerate(rgbcolors):
            axis[2][1].plot(exposure.histogram(contrast_image[...,i])[0], color=mycolor)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_hog_features(image, save_path=None):
    """
    Plot the HOG features of an image.

    Args:
        image: The image to plot the HOG features.
        save_path: The path to save the plot.
    """
    # Convertir l'image en niveaux de gris
    gray_image = rgb2gray(image)
    contrast_image = ft.contrast((image * 255).astype(np.uint8))
    gray_contrast = rgb2gray(contrast_image)

    # Extraire les caractéristiques HOG de l'image
    normal_hog = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    contrast_hog = hog(gray_contrast, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')

    # Normaliser l'image HOG
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10)) # L'image est déjà normalisée au préalable
    
    # Afficher l'image originale et l'image HOG
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    axs[0][0].imshow(image)
    axs[0][0].set_title('Original Image')
    axs[0][0].axis('off')

    axs[0][1].imshow(normal_hog[1])
    axs[0][1].set_title('HOG Image')
    axs[0][1].axis('off')

    axs[0][2].imshow(normal_hog[1], cmap='gray')
    axs[0][2].set_title('HOG Image with grayscale')
    axs[0][2].axis('off')

    axs[1][0].imshow(contrast_image)
    axs[1][0].set_title('Contrasted Image')
    axs[1][0].axis('off')

    axs[1][1].imshow(contrast_hog[1])
    axs[1][1].set_title('HOG Contrasted Image')
    axs[1][1].axis('off')

    axs[1][2].imshow(contrast_hog[1], cmap='gray')
    axs[1][2].set_title('HOG Contrasted Image with grayscale')
    axs[1][2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_sift_features(image, save_path=None):
    """
    Plots the original image, grayscale image, keypoints on grayscale image, and final image with keypoints overlaid.
    
    Args:
        image: The input image as a numpy array.
        save_path: The path to save the plot.
    """
    image_8bit = (image * 255).astype(np.uint8)
    gray_image = (rgb2gray(image) * 255).astype(np.uint8)
    image_contrast = (ft.contrast((image * 255).astype(np.uint8)))

    # Initialisation du détecteur de points clés SIFT
    sift = cv2.SIFT_create()
    
    # Détection des points clés et calcul des descripteurs
    key_desc_original = sift.detectAndCompute(image_8bit, None)
    key_desc_gray = sift.detectAndCompute(gray_image, None)
    key_desc_contrast = sift.detectAndCompute(image_contrast, None)

    # Afficher les points clés sur l'image originale
    image_with_keypoints = cv2.drawKeypoints(image_8bit, key_desc_original[0], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Afficher les points clés sur l'image en niveaux de gris
    gray_image_with_keypoints = cv2.drawKeypoints(gray_image, key_desc_gray[0], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Afficher les points clés sur l'image contrastée
    contrast_image_with_keypoints = cv2.drawKeypoints(image_contrast, key_desc_contrast[0], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Afficher les images: originale, en niveaux de gris, en niveaux de gris avec les points clés, et originale avec les points clés
    fig, axs = plt.subplots(2, 3, figsize=(10, 8))
    
    axs[0][0].imshow(image)
    axs[0][0].set_title('Original Image')
    axs[0][0].axis('off')

    axs[0][1].imshow(gray_image, cmap='gray')
    axs[0][1].set_title('Grayscale Image')
    axs[0][1].axis('off')

    axs[0][2].imshow(image_contrast)
    axs[0][2].set_title('Contrasted Image')
    axs[0][2].axis('off')
        
    axs[1][0].imshow(image_with_keypoints)
    axs[1][0].set_title('Image with SIFT Keypoints')
    axs[1][0].axis('off')
                   
    axs[1][1].imshow(gray_image_with_keypoints)
    axs[1][1].set_title('Grayscale Image with SIFT Keypoints')
    axs[1][1].axis('off')
                   
    axs[1][2].imshow(contrast_image_with_keypoints)
    axs[1][2].set_title('Contrasted Image with SIFT Keypoints')
    axs[1][2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_corner_detect_features(image, save_path=None):
    """
    Plot the corner detection features of an image.

    Args:
        image: The image to plot the corner detection features.
        save_path: The path to save the plot.
    """
    
    gray_image = rgb2gray(image)
    image = (image * 255).astype(np.uint8)

    coords = corner_peaks(corner_harris(gray_image), min_distance=2, threshold_rel=0.01)
    coords_subpix = corner_subpix(gray_image, coords, window_size=4)

    fig, axis = plt.subplots(2, 2, figsize=(6, 6))
    axis[0][0].axis('off')
    axis[0][0].imshow(image)
    axis[0][0].plot(
        coords[:, 1], coords[:, 0], color='cyan', marker='o', linestyle='None', markersize=6
    )
    axis[0][0].plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)

    axis[0][1].axis('off')
    axis[0][1].imshow(gray_image, cmap='gray')
    axis[0][1].plot(
        coords[:, 1], coords[:, 0], color='cyan', marker='o', linestyle='None', markersize=6
    )
    axis[0][1].plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
    
    # ----------- Contrasted Image ------------
    contrasted_image = ft.contrast(image)
    gray_contrasted_image = rgb2gray(contrasted_image)

    coords2 = corner_peaks(corner_harris(gray_contrasted_image), min_distance=2, threshold_rel=0.01)
    coords_subpix2 = corner_subpix(gray_contrasted_image, coords2, window_size=4)

    axis[1][0].axis('off')
    axis[1][0].imshow(contrasted_image)
    axis[1][0].plot(
        coords2[:, 1], coords2[:, 0], color='cyan', marker='o', linestyle='None', markersize=6
    )
    axis[1][0].plot(coords_subpix2[:, 1], coords_subpix2[:, 0], '+r', markersize=15)

    axis[1][1].axis('off')
    axis[1][1].imshow(gray_contrasted_image, cmap='gray')
    axis[1][1].plot(
        coords2[:, 1], coords2[:, 0], color='cyan', marker='o', linestyle='None', markersize=6
    )
    axis[1][1].plot(coords_subpix2[:, 1], coords_subpix2[:, 0], '+r', markersize=15)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(predictions_df, save_path=None):
    """
    Plot the correlation matrix of model predictions.

    Args:
        predictions_df: The dataframe containing the model predictions.
        save_path: The path to save the plot.
    """
    correlation_matrix = predictions_df.corr()
    plt.figure(figsize=(10, 7))
    plt.title('Correlation Matrix of Model Predictions')
    sns.heatmap(correlation_matrix, annot=True, cmap='plasma')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(predictions, y_test, label_names, name, save_path=None):
    cmat = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 7))
    plt.title(f'Confusion Matrix for {name}')
    plt.xticks(rotation=45)
    sns.heatmap(cmat, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_test_bin, preds_bin, label_names, name, save_path=None):
    """
    Plot the precision-recall curve for the model predictions.
    
    Args:
        y_test_bin: The binarized true labels.
        preds_bin: The binarized model predictions.
        label_names: The names of the labels.
        name: The name of the model.
        save_path: The path to save the plot.
    """
    plt.figure()
    for i, label in enumerate(label_names):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], preds_bin[:, i])
        plt.plot(recall, precision, lw=2, label=f'{label}')
    plt.xticks(rotation=45)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve for {name}')
    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_test_bin, preds_bin, label_names, name, save_path=None):
    """
    Plot the ROC curve for the model predictions.

    Args:
        y_test_bin: The binarized true labels.
        preds_bin: The binarized model predictions.
        label_names: The names of the labels.
        name: The name of the model.
        save_path: The path to save the plot.
    """
    plt.figure()
    for i, label in enumerate(label_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], preds_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xticks(rotation=45)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve for {name}')
    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_2D_decision_boundaries(train_data, models, save_path=None):
    """
    Plot the decision boundaries of the classifiers in 2D.

    Args:
        train_data: The training data.
        models: The dictionary of models to compare.
        save_path: The path to save the plot.
    """
    figure = plt.figure(figsize=(15, 5))
    i = 1

    X, y = train_data[b'data'], train_data[b'labels']
    X = X.reshape(-1, 32*32*3) / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    x_min, x_max = X_train_pca[:, 0].min() - 0.5, X_train_pca[:, 0].max() + 0.5
    y_min, y_max = X_train_pca[:, 1].min() - 0.5, X_train_pca[:, 1].max() + 0.5

    cm = plt.cm.hsv
    cm_bright = ListedColormap(['#FF0000', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#FF00FF', '#FF7F00', '#7FFF00', '#7F00FF', '#FF007F'])
    ax = plt.subplot(1, len(models) + 1, i)
    ax.set_title("Input data")

    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    for name, model in models.items():
        ax = plt.subplot(1, len(models) + 1, i)
        md.pipeline.set_params(classifier=model)
        md.pipeline.fit(X_train_pca, y_train)
        score = md.pipeline.score(X_test_pca, y_test)
        DecisionBoundaryDisplay.from_estimator(
            md.pipeline, X_train_pca, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.6)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_3D_decision_boundaries(train_data, models, save_path=None):
    """
    Plot the decision boundaries of the classifiers in 3D.

    Args:
        train_data: The training data.
        models: The dictionary of models to compare.
        save_path: The path to save the plot.
    """
    figure = plt.figure(figsize=(27, 9))
    i = 1

    X, y = train_data[b'data'], train_data[b'labels']
    X = X.reshape(-1, 32*32*3) / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # PCA with 3 components for 3D plotting
    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Set up 3D plot
    ax = figure.add_subplot(1, len(models) + 1, i, projection='3d')
    ax.set_title("Input data")

    # Colors for 10 classes
    cm_bright = ListedColormap(['#FF0000', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#FF00FF', '#FF7F00', '#7FFF00', '#7F00FF', '#FF007F'])

    # Plot training and test points
    scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], c=y_train, cmap=cm_bright, edgecolors="k")
    scatter = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], X_test_pca[:, 2], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.6)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    i += 1

    for name, model in models.items():
        ax = figure.add_subplot(1, len(models) + 1, i, projection='3d')

        md.pipeline.set_params(classifier=model)
        md.pipeline.fit(X_train_pca, y_train)
        score = md.pipeline.score(X_test_pca, y_test)

        # Create a mesh grid
        x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
        y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
        z_min, z_max = X_train_pca[:, 2].min() - 1, X_train_pca[:, 2].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        # Predict on the mesh grid
        Z = np.array([md.pipeline.predict(np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, z)]) for z in np.arange(z_min, z_max, 0.1)])
        
        # Plot decision boundary
        for j, z in enumerate(np.arange(z_min, z_max, 0.1)):
            zz = np.full(xx.shape, z)
            ax.plot_surface(xx, yy, zz, facecolors=cm_bright(Z[j].reshape(xx.shape)), alpha=0.3, rstride=100, cstride=100)

        # Plot training and test points
        scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], c=y_train, cmap=cm_bright, edgecolors="k")
        scatter = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], X_test_pca[:, 2], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.6)

        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.set_zlabel('PCA3')
        ax.set_title(name)
        ax.text2D(0.05, 0.95, ("%.2f" % score).lstrip("0"), transform=ax.transAxes, size=15, horizontalalignment="right")
        i += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    data_path: Path = PROCESSED_DATA_DIR / "processed_dataset.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    figures_path: Path = FIGURES_DIR,
    # -----------------------------------------
):
    
    predictions_df = pd.read_csv(predictions_path)
    data = load_data(data_path)
    label_names = [label.decode('utf-8') for label in data[b'label_names']]

    logger.info(f"Plotting feature extraction examples...")
    X_train = np.array(data[b'data']).reshape(-1, 32, 32, 3) / 255.0
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_train = np.array([ft.to_image(img) for img in X_train])
    y_test = np.array(data[b'test_labels'])

    sample_image = X_train[4]        
    plot_color_histo_features(sample_image, figures_path / "color_histo_features.png")
    plot_hog_features(sample_image, figures_path / "hog_features.png")
    plot_sift_features(sample_image, figures_path / "sift_features.png")
    plot_corner_detect_features(sample_image, figures_path / "corner_detect_features.png")

    logger.info(f"Plotting evaluation metrics...")
    train_data = load_data(PROCESSED_DATA_DIR / "train_data.pkl")
    test_data = load_data(PROCESSED_DATA_DIR / "test_data.pkl")

    X_train = train_data[b'data']
    y_test = test_data[b'labels']

    n_classes = len(label_names)
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    plot_correlation_matrix(predictions_df, figures_path / "correlation_matrix.png")

    for name, _ in md.models.items():
        predictions = predictions_df[name]
        plot_confusion_matrix(predictions, y_test, label_names, name, figures_path / name / f"{name}_confusion_matrix.png")

        preds_bin = label_binarize(predictions, classes=np.arange(n_classes))
        plot_precision_recall_curve(y_test_bin, preds_bin, label_names, name, figures_path / name / f"{name}_precision_recall_curve.png")
        
        plot_roc_curve(y_test_bin, preds_bin, label_names, name, figures_path / name / f"{name}_roc_curve.png")

    # plot_2D_decision_boundaries(train_data, md.models, figures_path / "2D_classifier_comparison.png")
    # plot_3D_decision_boundaries(train_data, md.models, figures_path / "3D_classifier_comparison.png")

    logger.success(f"Figures saved to {figures_path}")
    # -----------------------------------------


if __name__ == "__main__":
    app()
