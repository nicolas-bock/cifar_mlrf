from pathlib import Path
import matplotlib.pyplot as plt
from skimage import exposure
import numpy as np
import pickle
import random
import typer
from loguru import logger
from tqdm import tqdm

from skimage.feature import hog

from MLRF.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def load_data(file_path, encoding=None):
    with open(file_path, 'rb') as f:
        if (encoding is not None):
            data = pickle.load(f, encoding=encoding)
        else:
            data = pickle.load(f)
    return data

def extract_hog_features(images, save_path=None):
    hog_features = []
    
    for image in images:
        feature, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hog_features.append(feature)
    
    if save_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Original Image')
        ax2.imshow(hog_image_rescaled, cmap='gray')
        # ax2.imshow(hog_image, cmap='gray') # -> without exposure rescale intensity
        ax2.axis('off')
        ax2.set_title('HOG Image')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    return np.array(hog_features)

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

    # ================ ADDED HOG FEATURES ================
    combined_dict[b'data'] = np.array(combined_dict[b'data'])
    combined_dict[b'data'] = combined_dict[b'data'].reshape(-1, 32, 32, 3)

    # Extraire les HOG features pour les donées d'entraînement
    logger.info("Extracting HOG features for training data...")
    combined_dict[b'hog_data'] = extract_hog_features(combined_dict[b'data'], 'reports/figures/hog_example.png')
    logger.info("HOG features for training data extracted.")
    # ================ ADDED HOG FEATURES ================

    # Charger les données de test depuis test_batch
    test_batch_dict = load_data(input_path / "test_batch", encoding='bytes')
    combined_dict[b'test_data'].extend(test_batch_dict[b'data'])
    combined_dict[b'test_labels'].extend(test_batch_dict[b'labels'])

    # ================ ADDED HOG FEATURES ================
    combined_dict[b'test_data'] = np.array(combined_dict[b'test_data'])
    combined_dict[b'test_data'] = combined_dict[b'test_data'].reshape(-1, 32, 32, 3)

    # Extraire les HOG features pour les donées de test
    logger.info("Extracting HOG features for test data...")
    combined_dict[b'test_hog_data'] = extract_hog_features(combined_dict[b'test_data'])
    logger.info("HOG features for test data extracted.")
    # ================ ADDED HOG FEATURES ================

    # Sélectionner 10 images aléatoires pour chaque classe
    for label_index, label in enumerate(label_names):
        label_indices = [i for i, lbl in enumerate(combined_dict[b'labels']) if lbl == label_index]
        sample_indices = random.sample(label_indices, 10)
        for idx in sample_indices:
            combined_dict[b'sample_images'][label].append(combined_dict[b'data'][idx])

    logger.success("Processing dataset complete.")
    print(combined_dict.keys())

    logger.info("Saving processed dataset...")
    with open(output_path, 'wb') as f:
        pickle.dump(combined_dict, f)
    logger.success(f"Processed dataset saved at {output_path}")
    # -----------------------------------------

if __name__ == "__main__":
    app()
