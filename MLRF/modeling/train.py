from pathlib import Path
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from MLRF.dataset import load_data
import MLRF.model_utils as md
from MLRF.config import MODELS_DIR, PROCESSED_DATA_DIR

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

    # Train the models
    for name, model in md.models.items():
        logger.info(f"Training {name} model...")
        model.fit(X_train, y_train)
        logger.success(f"{name} model trained.")
        md.save_model(model, f"{name}_model.pkl", model_path)
        logger.success(f"{name} model saved.")

    # ---- Example of tqdm uses ----
    # logger.info("Training some model...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
