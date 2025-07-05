import numpy as np
from sklearn.decomposition import PCA

from config import WHITEN, RANDOM_STATE, N_COMPONENTS


def train_pca_model(train_faces,
                    n_components=N_COMPONENTS,
                    random_state=RANDOM_STATE,
                    whiten=WHITEN):
    """Trains a PCA model on the training faces."""
    print(f"\nTraining PCA model...")
    if isinstance(n_components, float) and 0 < n_components < 1:
        print(f"Using n_components to explain {n_components * 100:.2f}% of variance.")
    else:
        print(f"Using fixed n_components={n_components}.")

    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver='auto',
        random_state=random_state
    )
    pca.fit(train_faces)

    actual_n_components = pca.n_components_
    explained_variance_total = np.sum(pca.explained_variance_ratio_)

    print(f"PCA model trained.")
    print(f"Number of components selected: {actual_n_components}")
    print(f"Total explained variance by {actual_n_components} components: {explained_variance_total:.4f}")
    return pca


def get_eigenfaces(pca_model, h, w):
    """Extracts eigenfaces from the PCA model."""
    return pca_model.components_.reshape((pca_model.n_components_, h, w))
