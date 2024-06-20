from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import typer
from loguru import logger
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from skimage.feature import hog

from MLRF.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from MLRF.features import extract_hog_features

app = typer.Typer()

def load_data(file_path, encoding=None):
    """
    Load data from a pickle file

    Args:
        file_path: str
    """
    with open(file_path, 'rb') as f:
        if (encoding is not None):
            data = pickle.load(f, encoding=encoding)
        else:
            data = pickle.load(f)
    return data

def save_data(data, file):
    """
    Save data to a pickle file

    Args:
        data: any
        file: str
    """
    with open(file, 'wb') as f:
        pickle.dump(data, f)

@app.command()
def main(
    # ---- DEFAULT PATHS ----
    input_path: Path = RAW_DATA_DIR / "cifar-10-batches-py",
    output_path: Path = PROCESSED_DATA_DIR / "processed_dataset.pkl",
    # ----------------------------------------------
):

    logger.info("Processing dataset...")

    # Charger les 10 labels/features depuis batches.meta
    meta_dict = load_data(input_path / "batches.meta", encoding='bytes')
    label_names = meta_dict[b'label_names']
    logger.info(f"Labels/Features: {label_names}")
    
    combined_dict = {
        b'data': [],
        b'labels': [],
        b'label_names': label_names,
        b'hog_data': [],
        b'validation_data': [],
        b'validation_labels': [],
        b'validation_hog_data': [],
        b'test_data': [],
        b'test_labels': [],
        b'test_hog_data': [],
        b'sample_images': {label: [] for label in label_names}
    }

    for i in range(5):
        batch_dict = load_data(input_path / f"data_batch_{i+1}", encoding='bytes')
        combined_dict[b'data'].extend(batch_dict[b'data'])
        combined_dict[b'labels'].extend(batch_dict[b'labels'])
        
    for _ in tqdm(range(0, 50000), ncols=100, desc="Processing dataset"):
        continue

    # Diviser les données en données d'entraînement et de validation
    combined_dict[b'data'], combined_dict[b'validation_data'], combined_dict[b'labels'], combined_dict[b'validation_labels'] = train_test_split(
        combined_dict[b'data'], combined_dict[b'labels'], test_size=0.2, random_state=42
    )

    # Charger les données de test depuis test_batch
    test_batch_dict = load_data(input_path / "test_batch", encoding='bytes')
    combined_dict[b'test_data'].extend(test_batch_dict[b'data'])
    combined_dict[b'test_labels'].extend(test_batch_dict[b'labels'])

    # Sélectionner 10 images aléatoires pour chaque classe
    for label_index, label in enumerate(label_names):
        label_indices = [i for i, lbl in enumerate(combined_dict[b'labels']) if lbl == label_index]
        sample_indices = random.sample(label_indices, 10)
        for idx in sample_indices:
            combined_dict[b'sample_images'][label].append(combined_dict[b'data'][idx])

    logger.success("Processing dataset complete.")

    logger.info("Saving processed dataset...")
    save_data(combined_dict, output_path)
    logger.success(f"Processed dataset saved at {output_path}")
    # -----------------------------------------

if __name__ == "__main__":
    app()
