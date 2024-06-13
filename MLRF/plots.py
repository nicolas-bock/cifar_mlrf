from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import typer
from loguru import logger
from tqdm import tqdm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

from MLRF.dataset import load_data
import MLRF.model_utils as md
from MLRF.config import FIGURES_DIR, PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    data_path: Path = PROCESSED_DATA_DIR / "processed_dataset.pkl",
    model_path: Path = MODELS_DIR,
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    figures_path: Path = FIGURES_DIR,
    # -----------------------------------------
):
    
    predictions_df = pd.read_csv(predictions_path)
    data = load_data(data_path)
    label_names = [label.decode('utf-8') for label in data[b'label_names']]
    
    logger.info(f"Plotting evaluation metrics...")
    # Correlation des prédictions
    correlation_matrix = predictions_df.corr()
    plt.figure(figsize=(10, 7))
    plt.title('Correlation Matrix of Model Predictions')
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.savefig(figures_path / 'predictions_correlation_matrix.png')
    plt.close()

    y_test = np.array(data[b'test_labels'])
    n_classes = len(label_names)
    y_test_bin = md.label_binarize(y_test, classes=np.arange(n_classes))

    # Matrice de confusion
    for name, _ in md.models.items():
        predictions = predictions_df[name]
        cm = md.confusion_matrix(y_test, predictions)
        plt.figure(figsize=(10, 7))
        plt.title(f'Confusion Matrix for {name}')
        plt.xticks(rotation=45)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(figures_path / f'{name}_confusion_matrix.png')
        plt.close()

        # Courbes de précision/rappel et ROC
        preds_bin = md.label_binarize(predictions, classes=np.arange(n_classes))
        
        # Precision-Recall curve
        plt.figure()
        for i, label in enumerate(label_names):
            precision, recall, _ = md.precision_recall_curve(y_test_bin[:, i], preds_bin[:, i])
            plt.plot(recall, precision, lw=2, label=f'{label}')
        plt.xticks(rotation=45)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall curve for {name}')
        plt.legend(loc='best')
        plt.savefig(figures_path / f'{name}_precision_recall_curve.png')
        plt.close()
        
        # ROC curve
        plt.figure()
        for i, label in enumerate(label_names):
            fpr, tpr, _ = md.roc_curve(y_test_bin[:, i], preds_bin[:, i])
            roc_auc = md.auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xticks(rotation=45)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve for {name}')
        plt.legend(loc='best')
        plt.savefig(figures_path / f'{name}_roc_curve.png')
        plt.close()

    # -----------------------------------------
    # Comparaison des classifieurs
    X_train = np.array(data[b'data']).reshape(-1, 32, 32, 3) / 255.0
    y_train = np.array(data[b'labels'])
    X_test = np.array(data[b'test_data']).reshape(-1, 32, 32, 3) / 255.0
    y_test = np.array(data[b'test_labels'])

    # Flattening the image data for classifiers
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    figure = plt.figure(figsize=(27, 9))
    i = 1
   
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(1, len(md.models) + 1, i)
    ax.set_title('Input data')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    for name, _ in md.models.items():
        model = md.joblib.load(model_path / f'{name}_model.pkl')
        score = model.score(X_test, y_test)

        ax = plt.subplot(1, len(md.models) + 1, i)

        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(f'{name} (score = {score:.2f})')
        ax.text(x_max - 0.3, y_min + 0.3, ("%.2f" % score).lstrip("0"),
                size=15, horizontalalignment='right')
        i += 1

    plt.tight_layout()
    plt.savefig(figures_path / 'classifier_comparison.png')
    plt.close()
    logger.success(f"Figures saved to {figures_path}")
    # -----------------------------------------


if __name__ == "__main__":
    app()
