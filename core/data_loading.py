import numpy as np
from sklearn.datasets import fetch_olivetti_faces

from config import RANDOM_STATE, TRAIN_SAMPLES, IMAGE_SHAPE, DATASET_CACHE_DIR

def load_olivetti_data(
        shuffle=True,
        random_state=RANDOM_STATE):
    """Loads the Olivetti faces dataset."""
    dataset = fetch_olivetti_faces(
        shuffle=shuffle,
        random_state=random_state,
        data_home=DATASET_CACHE_DIR
    )
    faces = dataset.data
    targets = dataset.target
    n_samples, n_features = faces.shape
    h, w = IMAGE_SHAPE

    print(f"Dataset loaded: {n_samples} samples, {n_features} features each.")
    print(f"Image dimensions: {h}x{w}")
    print(f"Number of unique individuals: {len(np.unique(targets))}")
    return faces, targets, (h, w)

def split_data(faces, targets, train_samples=TRAIN_SAMPLES):
    """Splits data into training and testing sets based on a fixed number of training samples."""
    train_faces = faces[:train_samples, :]
    test_faces = faces[train_samples:, :]
    train_targets = targets[:train_samples]
    test_targets = targets[train_samples:]
    print(f"Data split: {len(train_faces)} training samples, {len(test_faces)} testing samples.")
    return train_faces, test_faces, train_targets, test_targets
