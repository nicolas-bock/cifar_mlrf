from pathlib import Path
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from MLRF.dataset import load_data, save_data
import MLRF.model_utils as md
from MLRF.config import MODELS_DIR, PROCESSED_DATA_DIR
from MLRF.features import to_image


app = typer.Typer()

@app.command()
def main(
    # ---- DEFAULT PATHS ----
    data_path: Path = PROCESSED_DATA_DIR / "processed_dataset.pkl",
    model_path: Path = MODELS_DIR,
    # -----------------------------------------
):

    data = load_data(data_path)

    X_train = np.array(data[b'data']).reshape(-1, 32, 32, 3) / 255.0
    y_train = np.array(data[b'labels'])
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_train = np.array([to_image(img) for img in X_train])

    train_dict = {
        b'data': X_train,
        b'labels': y_train,
    }
    save_data(train_dict, PROCESSED_DATA_DIR / "train_data.pkl")

    X_validation = np.array(data[b'validation_data']).reshape(-1, 32, 32, 3) / 255.0
    y_validation = np.array(data[b'validation_labels'])
    X_validation = X_validation.reshape(X_validation.shape[0], -1)
    X_validation = np.array([to_image(img) for img in X_validation])

    validation_dict = {
        b'data': X_validation,
        b'labels': y_validation,
    }
    save_data(validation_dict, PROCESSED_DATA_DIR / "validation_data.pkl")

    # Train the models
    for name, model in md.models.items():
        logger.info(f"Training {name} model...")
        md.pipeline.set_params(classifier=model)
        md.pipeline.fit(X_train, y_train)
        logger.success(f"{name} model trained.")
        y_pred = md.pipeline.predict(X_validation)
        accuracy = md.accuracy_score(y_validation, y_pred)
        logger.success(f"{name} model accuracy: {accuracy:.2f}")
        md.save_model(model, f"{name}_model.pkl", model_path)
        logger.success(f"{name} model saved.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
