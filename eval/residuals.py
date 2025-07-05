import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from config import RANDOM_STATE, N_RESIDUAL_COMPONENTS
from utils.plotting import plot_gallery, plot_cumulative_explained_variance

def analyze_misclassification_residuals(
        main_pca_model,
        original_test_faces,
        compressed_test_faces_by_main_pca,
        y_true, y_pred, h, w,
        n_residual_components=N_RESIDUAL_COMPONENTS,
        random_state=RANDOM_STATE):
    """
    Analyzes the residuals of misclassified instances using PCA.

    Args:
        main_pca_model (PCA): The main PCA model used for recognition.
        original_test_faces (np.array): The original pixel data of the test faces.
        compressed_test_faces_by_main_pca (np.array): Test faces transformed by main_pca_model.
        y_true (list or np.array): True labels for the test set.
        y_pred (list or np.array): Predicted labels for the test set.
        h (int): Height of the images.
        w (int): Width of the images.
        n_residual_components (int): Number of PCA components for residual analysis.
        random_state (int): Random state for PCA on residuals.
    """
    print("\n--- Analyzing Residuals of Misclassified Instances ---")
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]

    if not misclassified_indices:
        print("No misclassifications found. Skipping residual analysis.")
        return

    print(f"Found {len(misclassified_indices)} misclassified instances.")

    original_misclassified_faces = original_test_faces[misclassified_indices]
    compressed_misclassified_faces = compressed_test_faces_by_main_pca[misclassified_indices]

    # Reconstruct the misclassified faces using the main PCA model
    reconstructed_misclassified_faces = main_pca_model.inverse_transform(compressed_misclassified_faces)

    # Calculate residual images
    residual_images = original_misclassified_faces - reconstructed_misclassified_faces
    # Ensure n_residual_components is not more than number of samples or features
    actual_n_residual_components = min(n_residual_components,
                                       residual_images.shape[0],
                                       residual_images.shape[1] - 1 if residual_images.shape[1] > 1 else 1)

    if actual_n_residual_components <= 0:
        print("Not enough residual samples or features to perform PCA on residuals. Skipping.")
        return

    print(f"Performing PCA on {len(residual_images)} residual images with {actual_n_residual_components} components...")
    pca_on_residuals = PCA(n_components=actual_n_residual_components, whiten=True, random_state=random_state)
    pca_on_residuals.fit(residual_images)

    eigen_residuals = pca_on_residuals.components_.reshape((actual_n_residual_components, h, w))
    eigen_residual_titles = [f"Eigen-Residual {i + 1}" for i in range(actual_n_residual_components)]

    num_eigen_residuals_to_plot = min(12, actual_n_residual_components)

    plot_gallery(eigen_residuals[:num_eigen_residuals_to_plot],
                 eigen_residual_titles[:num_eigen_residuals_to_plot],
                 h, w,
                 filename=f"eigen_residuals_top{num_eigen_residuals_to_plot}.png",
                 n_row=max(1, (num_eigen_residuals_to_plot + 3) // 4),
                 n_col=4)
    plt.suptitle(f"Top {num_eigen_residuals_to_plot} Eigen-Residuals (from {len(residual_images)} misclassifications)",
                 fontsize=14)
    plt.show()

    plot_cumulative_explained_variance(pca_on_residuals, filename="cumulative_variance_residuals_pca.png")
    plt.suptitle(f"Cumulative Explained Variance for PCA on Residuals ({pca_on_residuals.n_components_} components)")
    plt.show()

    print("Residual analysis complete.")