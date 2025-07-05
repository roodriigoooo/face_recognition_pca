"""
config for the project
"""

# -*- coding: utf-8 -*-
"""
Centralized configuration for the face recognition project.
"""

# --- PCA & Model Parameters ---
N_COMPONENTS = 150  # Default number of principal components to keep.
WHITEN = True       # Whether to whiten the data, decorrelating the output.
RANDOM_STATE = 101  # Seed for reproducibility of random operations.

# --- Dataset Parameters ---
TRAIN_SAMPLES = 350 # Number of samples for the training set.
IMAGE_SHAPE = (64, 64) # Height and width of the face images.

# --- Analysis Parameters ---
# The index of a test sample to use for single-match visualization.
SAMPLE_IDX_TO_VISUALIZE = 17
# Number of components for the residual analysis PCA.
N_RESIDUAL_COMPONENTS = 10

# --- File System ---
PLOTS_DIR = "pca_visualizations" # directory to save all generated plots.
DATASET_CACHE_DIR = './olivetti_faces_cache' # where to cache the downloaded dataset.