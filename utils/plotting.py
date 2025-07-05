import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import PLOTS_DIR

def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_gallery(images, titles, h, w, filename, n_row=3, n_col=4, cmap=plt.cm.gray):
    """Helper function to plot a gallery of portraits and save it."""
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        if i < len(images):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=cmap, interpolation='nearest')
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())
        else:
            # If there are fewer images than subplot slots, hide the empty ones
            fig = plt.gcf()  # gcf gets the current figure
            ax = fig.add_subplot(n_row, n_col, i + 1)
            ax.axis('off')
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    print(f"Saved gallery plot to {save_path}")


def plot_cumulative_explained_variance(pca_model, filename):
    """Plots the cumulative explained variance by PCA components and saves it."""
    fig = plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA: Cumulative Explained Variance vs. Number of Components')
    plt.grid(True)

    # Highlight common thresholds
    if len(cumulative_variance) > 0:
        for val_thresh in [0.90, 0.95, 0.99]:
            components_for_thresh = np.where(cumulative_variance >= val_thresh)[0]
            if len(components_for_thresh) > 0:
                num_comp = components_for_thresh[0] + 1
                plt.axhline(y=val_thresh, color='r', linestyle='--',
                            label=f'{val_thresh * 100:.0f}% EV (at {num_comp} comps)')
                plt.axvline(x=num_comp, color='r', linestyle=':', alpha=0.7)

    plt.legend(loc='center right')
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    print(f"Saved cumulative explained variance plot to {save_path}")

def plot_single_match_result(test_image, matched_train_image, test_title, match_title, h, w, filename):
    """Plots a single test image alongside its best match from the training set and saves it."""
    fig = plt.figure(figsize=(8, 4))

    # Plot Test Image
    plt.subplot(1, 2, 1)
    plt.imshow(test_image.reshape(h, w), cmap=plt.cm.gray, interpolation='nearest')
    plt.title(test_title, size=12)
    plt.axis('off')

    # Plot Matched Train Image
    plt.subplot(1, 2, 2)
    plt.imshow(matched_train_image.reshape(h, w), cmap=plt.cm.gray, interpolation='nearest')
    plt.title(match_title, size=12)
    plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    print(f"Saved single match plot to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, filename):
    """Plots the confusion matrix and saves it."""
    cm = confusion_matrix(y_true, y_pred,
                          labels=np.unique(np.concatenate((y_true, y_pred))))  # Use all unique labels for CM
    fig = plt.figure(figsize=(12, 10))

    # Ensure class_names match the labels used in confusion_matrix
    # If class_names cover all possible people, but some aren't in y_true, CM will be smaller.
    unique_labels_in_data = np.unique(np.concatenate((y_true, y_pred)))
    # Filter class_names to only those present in the data, or create them if not enough provided
    if len(class_names) > len(unique_labels_in_data):
        relevant_class_names = [cn for i, cn in enumerate(class_names) if i in unique_labels_in_data]
        if len(relevant_class_names) != len(unique_labels_in_data):  # fallback if mapping is complex
            relevant_class_names = [f"P{i}" for i in unique_labels_in_data]
    else:
        relevant_class_names = [f"P{i}" for i in unique_labels_in_data]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=relevant_class_names, yticklabels=relevant_class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")


def plot_example_matches(test_images_original, train_images_original,
                         y_true, y_pred, matched_train_indices,
                         train_labels_original,  # Labels for the training set
                         h, w, filename, n_examples=8):
    """Plots a few example matches, indicating correct/incorrect predictions and showing the matched image, then saves it."""

    num_available_samples = len(test_images_original)
    actual_n_examples = min(n_examples, num_available_samples)

    if actual_n_examples == 0:
        print("No examples to plot in plot_example_matches.")
        return

    plt.figure(figsize=(2.2 * actual_n_examples, 6.5))  # Adjusted for two rows and more spacing
    plt.suptitle("Example Test Matches (True vs. Predicted)", fontsize=16)

    # Ensure indices are within bounds of y_true, y_pred, and matched_train_indices
    if num_available_samples > 0:
        indices_to_plot = np.random.choice(num_available_samples, actual_n_examples, replace=False)
    else:
        indices_to_plot = []

    for plot_col_idx, original_test_idx in enumerate(indices_to_plot):
        test_img = test_images_original[original_test_idx]
        true_label = y_true[original_test_idx]
        pred_label = y_pred[original_test_idx]

        # Plot Test Image (Top Row)
        plt.subplot(2, actual_n_examples, plot_col_idx + 1)  # subplot indices are 1-based
        plt.imshow(test_img.reshape(h, w), cmap=plt.cm.gray, interpolation='nearest')
        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(f"Test Sample (Idx: {original_test_idx})\nTrue ID: P{true_label}\nPred ID: P{pred_label}",
                  color=title_color, fontsize=9)
        plt.axis('off')

        # Plot Matched Training Image (Bottom Row)
        # matched_train_indices[original_test_idx] gives the index in train_images_original
        matched_train_image_idx = matched_train_indices[original_test_idx]
        matched_img = train_images_original[matched_train_image_idx]
        # The actual label of the image shown as the match
        actual_label_of_matched_img = train_labels_original[matched_train_image_idx]

        plt.subplot(2, actual_n_examples, plot_col_idx + actual_n_examples + 1)
        plt.imshow(matched_img.reshape(h, w), cmap=plt.cm.gray, interpolation='nearest')
        plt.title(
            f"Best Match from Train\n(Train Idx: {matched_train_image_idx})\nActual ID: P{actual_label_of_matched_img}",
            fontsize=9)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93], h_pad=3.0)  # Adjust layout for suptitle and padding
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    print(f"Saved example matches plot to {save_path}")

def plot_accuracy_vs_components(components, accuracies, variances, filename):
    """Plots recognition accuracy and variance against the number of components."""
    _ensure_plots_dir()
    fig, ax1 = plt.subplots(figsize=(12, 7))
    # Accuracy plot
    ax1.set_xlabel('Number of PCA Components')
    ax1.set_ylabel('Recognition Accuracy', color='tab:blue')
    ax1.plot(components, accuracies, marker='o', linestyle='-', color='tab:blue', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle=':', alpha=0.7)
    # Variance plot
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative Explained Variance', color='tab:red')
    ax2.plot(components, variances, marker='x', linestyle='--', color='tab:red', label='Explained Variance')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    # Finalize
    plt.title('Accuracy & Variance vs. Number of PCA Components')
    fig.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved accuracy vs n_components plot to {save_path}")
