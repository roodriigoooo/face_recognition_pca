import numpy as np
import os
from sklearn.metrics import classification_report

# --- Local Imports ---
import config
from core.data_loading import load_olivetti_data, split_data
from core.model import train_pca_model, get_eigenfaces
from eval.metrics import evaluate_recognition_system, find_best_match
from eval.residuals import analyze_misclassification_residuals
from utils.plotting import (
    plot_gallery,
    plot_cumulative_explained_variance,
    plot_accuracy_vs_components,
    plot_confusion_matrix,
    plot_example_matches,
    plot_single_match_result
)


def find_optimal_components(train_faces, test_faces, train_targets, test_targets):
    """Analyzes accuracy across a range of n_components to find the best value."""
    print("\n--- Analyzing Accuracy vs. Number of PCA Components ---")
    if len(test_faces) == 0:
        print("Test set is empty. Skipping component analysis.")
        return config.N_COMPONENTS

    max_components = min(train_faces.shape)
    component_range = sorted(list(set(
        list(range(10, min(100, max_components), 10)) +
        list(range(100, min(251, max_components), 25))
    )))

    accuracies, variances = [], []
    for n_c in component_range:
        pca_eval = train_pca_model(train_faces, n_components=n_c)
        train_compressed = pca_eval.transform(train_faces)
        test_compressed = pca_eval.transform(test_faces)

        acc, _, _, _ = evaluate_recognition_system(
            test_compressed, train_compressed, test_targets, train_targets
        )
        accuracies.append(acc)
        variances.append(np.sum(pca_eval.explained_variance_ratio_))
        print(f"  Accuracy for {n_c} components: {acc * 100:.2f}%")

    if not accuracies:
        print("Could not determine optimal components. Using default.")
        return config.N_COMPONENTS

    plot_accuracy_vs_components(
        component_range, accuracies, variances, "accuracy_vs_n_components.png"
    )

    best_acc_idx = np.argmax(accuracies)
    optimal_n = component_range[best_acc_idx]
    print(f"\nOptimal number of components found: {optimal_n} (Accuracy: {accuracies[best_acc_idx] * 100:.2f}%)")
    return optimal_n


def main():
    """Main function to run the face recognition pipeline."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    print(f"Plots will be saved to '{config.PLOTS_DIR}/' directory.")

    # 1. Load and prepare data
    faces, targets, (h, w) = load_olivetti_data()
    train_faces, test_faces, train_targets, test_targets = split_data(faces, targets)

    # 2. Find the optimal number of PCA components
    optimal_n_components = find_optimal_components(
        train_faces, test_faces, train_targets, test_targets
    )

    # 3. Train final model and perform detailed evaluation
    print(f"\n--- Performing Detailed Evaluation with {optimal_n_components} Components ---")
    pca = train_pca_model(train_faces, n_components=optimal_n_components)

    # 4. Visualize Eigenfaces and Variance
    eigenfaces = get_eigenfaces(pca, h, w)
    plot_gallery(
        eigenfaces[:12],
        [f"Eigenface {i + 1}" for i in range(12)],
        h, w, f"eigenfaces_pca_{pca.n_components_}c.png"
    )
    plot_cumulative_explained_variance(pca, f"cumulative_variance_pca_{pca.n_components_}c.png")

    # 5. Evaluate on the test set
    if len(test_faces) > 0:
        train_compressed = pca.transform(train_faces)
        test_compressed = pca.transform(test_faces)

        accuracy, y_true, y_pred, matched_indices = evaluate_recognition_system(
            test_compressed, train_compressed, test_targets, train_targets
        )

        print(f"\nOverall accuracy on the test set: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

        # 6. Visualize results
        plot_confusion_matrix(y_true, y_pred,
                              class_names=[f"P{i}" for i in np.unique(targets)],
                              filename=f"confusion_matrix_pca_{pca.n_components_}comps.png")
        plot_example_matches(
            test_faces, train_faces, y_true, y_pred, matched_indices,
            train_targets, h, w, f"example_matches_pca_{pca.n_components_}c.png"
        )

        # 7. Perform residual analysis on misclassifications
        analyze_misclassification_residuals(pca, test_faces, test_compressed, y_true, y_pred, h, w)
    else:
        print("\nTest set is empty. Skipping final evaluation and residual analysis.")

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()