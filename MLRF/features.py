from pathlib import Path
from skimage import exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np

import typer
from loguru import logger
from tqdm import tqdm

from MLRF.config import PROCESSED_DATA_DIR

app = typer.Typer()

def to_image(img_flat):
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))
    return img

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

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
    input_path: Path = PROCESSED_DATA_DIR / "processed_data",
    output_path: Path = PROCESSED_DATA_DIR / "features",
    # -----------------------------------------
):
    # -----------------------------------------
    return
    # -----------------------------------------


if __name__ == "__main__":
    app()
