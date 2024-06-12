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
    # X_train = np.array(data[b'data']).reshape(-1, 32, 32, 3) / 255.0
    # y_train = np.array(data[b'labels'])
    # X_test = np.array(data[b'test_data']).reshape(-1, 32, 32, 3) / 255.0
    # y_test = np.array(data[b'test_labels'])

    # X_train_2d = X_train[:, :2]
    # X_test_2d = X_test[:, :2]

    # figure = plt.figure(figsize=(27, 9))
    # i = 1
    # cm = plt.cm.RdBu
    # cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    # # Just plot the input data
    # ax = plt.subplot(1, len(predictions_df.columns) + 1, i)
    # ax.set_title("Input data")
    # ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")
    # ax.set_xlim(X_test_2d[:, 0].min() - .5, X_test_2d[:, 0].max() + .5)
    # ax.set_ylim(X_test_2d[:, 1].min() - .5, X_test_2d[:, 1].max() + .5)
    # ax.xticks(rotation=45)
    # ax.set_xticks(())
    # ax.set_yticks(())
    # i += 1

    # for name, model in md.models.items():
    #     ax = plt.subplot(1, len(predictions_df.columns) + 1, i)
        
    #     model = make_pipeline(StandardScaler(), model)
    #     model.fit(X_train_2d, y_train)
    #     score = model.score(X_test_2d, y_test)
    #     DecisionBoundaryDisplay.from_estimator(
    #         model, X_train_2d, cmap=cm, alpha=0.8, ax=ax, eps=0.5
    #     )
        
    #     ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    #     ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")
    #     ax.set_xlim(X_test_2d[:, 0].min() - .5, X_test_2d[:, 0].max() + .5)
    #     ax.set_ylim(X_test_2d[:, 1].min() - .5, X_test_2d[:, 1].max() + .5)
    #     ax.xticks(rotation=45)
    #     ax.set_xticks(())
    #     ax.set_yticks(())
    #     ax.set_title(name)
    #     ax.text(
    #         X_test_2d[:, 0].max() - .3,
    #         X_test_2d[:, 1].min() + .3,
    #         ("%.2f" % score).lstrip("0"),
    #         size=15,
    #         horizontalalignment="right",
    #     )
    #     i += 1

    # plt.tight_layout()
    # plt.savefig('reports/figures/classifier_comparison.png')
    # plt.close()
    logger.success(f"Figures saved to {figures_path}")
    # -----------------------------------------


if __name__ == "__main__":
    app()
