from pathlib import Path
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd

from MLRF.dataset import load_data
import MLRF.model_utils as md
from MLRF.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    # ---- DEFAULT PATHS ----
    features_path: Path = PROCESSED_DATA_DIR / "processed_dataset.pkl",
    model_path: Path = MODELS_DIR,
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    
    data = load_data(features_path)

    X_test = np.array(data[b'test_data']).reshape(-1, 32, 32, 3) / 255.0
    y_test = np.array(data[b'test_labels'])
    X_test = X_test.reshape(X_test.shape[0], -1)

    predictions = {'True Labels': y_test}

    # Evaluate the models
    for name, _ in md.models.items():
        model = md.load_model(f'{name}_model.pkl', model_path)
        logger.info(f"Evaluating {name} model...")
        predictions[name] = model.predict(X_test)
        accuracy = md.accuracy_score(y_test, predictions[name])
        # accuracy = md.pipeline.score(X_test, y_test)
        logger.success(f"{name} model accuracy: {accuracy:.2f}")
        scores = md.cross_val_score(model, X_test, y_test, cv=5)
        logger.success(f"{name} model cross-validation accuracy: {scores.mean():.2f}")

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to {predictions_path}")
    # -----------------------------------------


if __name__ == "__main__":
    app()
