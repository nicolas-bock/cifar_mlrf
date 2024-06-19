from pathlib import Path
from skimage import exposure
from skimage.feature import hog, local_binary_pattern
from scipy.ndimage import gaussian_filter, gaussian_laplace, sobel
import cv2
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

def extract_hog_features(images):
    hog_features = []

    for image in images:
        gray_image = rgb2gray(image)
        hog_image_rescaled = exposure.rescale_intensity(gray_image, in_range=(0, 10))
        feature, hog_image = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=None, multichannel=False, block_norm='L2-Hys')
        # feature, hog_image = hog(hog_image_rescaled, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=None, multichannel=False, block_norm='L2-Hys')
        hog_features.append(feature)

    return np.array(hog_features)

def extract_sift_features(images: np.ndarray):
    sift = cv2.SIFT_create()
    sift_features = []

    for image in images:
        gray_image = rgb2gray(image)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is None:
            sift_features.append(np.zeros((128,)))
        else:
            sift_features.append(descriptors.flatten())
    
    return np.array(sift_features)

def detect_keypoints(image, num_intervals=4, sigma=1.6, threshold=0.04):
    image = image.astype('float32')
    octaves = []
    for octave in range(4):
        octave_images = []
        for interval in range(num_intervals + 3):
            k = 2 ** (interval / num_intervals)
            sigma_ = sigma * k
            octave_images.append(gaussian_filter(image, sigma_))
        octaves.append(octave_images)
        image = octave_images[-3][::2, ::2]
    
    keypoints = []
    for octave in octaves:
        dog_images = [octave[i] - octave[i + 1] for i in range(len(octave) - 1)]
        for i in range(1, len(dog_images) - 1):
            dog_prev, dog_curr, dog_next = dog_images[i - 1], dog_images[i], dog_images[i + 1]
            keypoints += detect_local_extrema(dog_prev, dog_curr, dog_next, threshold)
    
    return keypoints

def detect_local_extrema(dog_prev, dog_curr, dog_next, threshold):
    keypoints = []
    rows, cols = dog_curr.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            patch = dog_curr[r - 1:r + 2, c - 1:c + 2]
            if is_extrema(patch, dog_prev[r, c], dog_next[r, c], threshold):
                keypoints.append((r, c))
    return keypoints

def is_extrema(patch, prev_pixel, next_pixel, threshold):
    center_pixel = patch[1, 1]
    if abs(center_pixel) < threshold:
        return False
    if center_pixel > 0:
        return center_pixel == np.max(patch) and center_pixel > prev_pixel and center_pixel > next_pixel
    else:
        return center_pixel == np.min(patch) and center_pixel < prev_pixel and center_pixel < next_pixel

def compute_orientation(image, keypoints):
    orientations = []
    for kp in keypoints:
        r, c = kp
        patch = image[r-8:r+8, c-8:c+8]
        magnitude = np.hypot(sobel(patch, axis=0), sobel(patch, axis=1))
        orientation = np.arctan2(sobel(patch, axis=1), sobel(patch, axis=0))
        hist, _ = np.histogram(orientation, bins=36, range=(-np.pi, np.pi), weights=magnitude)
        dominant_orientation = np.argmax(hist)
        orientations.append(dominant_orientation)
    return orientations

def compute_descriptors(image, keypoints, orientations):
    descriptors = []
    for kp, ori in zip(keypoints, orientations):
        r, c = kp
        patch = image[r-8:r+8, c-8:c+8]
        rotated_patch = rotate_patch(patch, ori)
        descriptor = local_binary_pattern(rotated_patch, P=8, R=1, method='uniform')
        descriptors.append(descriptor.flatten())
    return np.array(descriptors)

def rotate_patch(patch, orientation):
    center = (patch.shape[0] // 2, patch.shape[1] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, np.degrees(orientation), 1)
    rotated_patch = cv2.warpAffine(patch, rot_matrix, (patch.shape[1], patch.shape[0]))
    return rotated_patch

def extract_sift_features(images):
    sift_features = []
    for image in images:
        gray_image = rgb2gray(image)
        keypoints = detect_keypoints(gray_image)
        orientations = compute_orientation(gray_image, keypoints)
        descriptors = compute_descriptors(gray_image, keypoints, orientations)
        sift_features.append(descriptors.flatten())
    return np.array(sift_features)

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
