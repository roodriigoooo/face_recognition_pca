
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# --- Configuration ---
N_COMPONENTS = 150  # Number of principal components to keep, can be tuned
IMAGE_SHAPE = (64, 64) # Height and width of the images
RANDOM_STATE = 101
TRAIN_SAMPLES = 350 # Number of samples for training, as in the original script
PLOTS_DIR = "pca_visualizations" # Directory to save plots

# --- Helper Functions ---
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
            fig = plt.gcf() # gcf gets the current figure
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
                            label=f'{val_thresh*100:.0f}% EV (at {num_comp} comps)')
                plt.axvline(x=num_comp, color='r', linestyle=':', alpha=0.7)
    
    plt.legend(loc='center right')
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    print(f"Saved cumulative explained variance plot to {save_path}")


# --- Core Logic Functions ---
def load_olivetti_data(shuffle=True, random_state=RANDOM_STATE):
    """Loads the Olivetti faces dataset."""
    dataset = fetch_olivetti_faces(shuffle=shuffle, random_state=random_state, data_home='./olivetti_faces_cache')
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

def train_pca_model(train_faces, n_components=N_COMPONENTS, whiten=True, random_state=RANDOM_STATE):
    """Trains a PCA model on the training faces."""
    print(f"\nTraining PCA model...")
    if isinstance(n_components, float) and 0 < n_components < 1:
        print(f"Using n_components to explain {n_components*100:.2f}% of variance.")
    else:
        print(f"Using fixed n_components={n_components}.")
        
    pca = PCA(n_components=n_components, whiten=whiten, svd_solver='auto', random_state=random_state)
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

# --- Main Execution ---
def main():
    """Main function to run the enhanced face recognition pipeline."""
    
    # Create directory for saving plots if it doesn't exist
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Plots will be saved to '{PLOTS_DIR}/' directory.")

    faces, targets, (h, w) = load_olivetti_data()
    train_faces, test_faces, train_targets, test_targets = split_data(faces, targets)

    # --- Analysis: Accuracy vs. Number of PCA Components ---
    print("\n--- Analyzing Accuracy vs. Number of PCA Components ---")
    max_possible_components = min(train_faces.shape[0], train_faces.shape[1] -1) # Max components for PCA

    # Ensure steps are reasonable and don't exceed max_possible_components
    component_steps = []
    if max_possible_components >= 10:
        component_steps.extend(np.arange(10, min(100, max_possible_components + 1), 10))
    if max_possible_components >= 100:
        component_steps.extend(np.arange(100, min(max_possible_components + 1, 251), 25)) # Max 250 or dataset max
    
    # Ensure all values are unique and sorted, and within bounds
    component_range = sorted(list(set(cs for cs in component_steps if cs <= max_possible_components and cs > 0)))

    if not component_range and max_possible_components > 0: # If list is empty but could have values
        component_range = [min(10, max_possible_components)]
    elif not component_range and max_possible_components <= 0:
        print("Not enough features/samples to run PCA component analysis. Skipping.")
        component_range = []


    accuracies = []
    explained_variances_at_n = []

    if len(test_faces) == 0:
        print("Test set is empty. Skipping accuracy analysis and detailed evaluation.")
        OPTIMAL_N_COMPONENTS = N_COMPONENTS # Fallback
    else:
        for n_c in component_range:
            print(f"Testing with n_components = {n_c}")
            pca_eval = train_pca_model(train_faces, n_components=n_c, whiten=True, random_state=RANDOM_STATE)
            
            compressed_train_eval = pca_eval.transform(train_faces)
            compressed_test_eval = pca_eval.transform(test_faces)
            
            acc, _, _, _ = evaluate_recognition_system(
                compressed_test_eval,
                compressed_train_eval,
                test_targets,
                train_targets
            )
            accuracies.append(acc)
            explained_variances_at_n.append(np.sum(pca_eval.explained_variance_ratio_))
            print(f"  Accuracy: {acc*100:.2f}%, Explained Variance: {explained_variances_at_n[-1]*100:.2f}%")

        # Plot Accuracy vs. Number of Components
        if accuracies:
            plt.figure(figsize=(12, 7))
            ax1 = plt.gca()
            color = 'tab:blue'
            ax1.set_xlabel('Number of PCA Components')
            ax1.set_ylabel('Recognition Accuracy', color=color)
            ax1.plot(component_range, accuracies, marker='o', linestyle='-', color=color, label='Accuracy')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, linestyle=':', alpha=0.7)

            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Cumulative Explained Variance', color=color)
            ax2.plot(component_range, explained_variances_at_n, marker='x', linestyle='--', color=color, label='Explained Variance')
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title('Recognition Accuracy & Explained Variance vs. Number of PCA Components')
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='center right')
            
            save_path = os.path.join(PLOTS_DIR, "accuracy_vs_n_components.png")
            plt.savefig(save_path)
            print(f"Saved accuracy vs n_components plot to {save_path}")
            plt.show()

            best_acc_idx = np.argmax(accuracies)
            OPTIMAL_N_COMPONENTS = component_range[best_acc_idx]
            print(f"\nOptimal number of components based on accuracy: {OPTIMAL_N_COMPONENTS} (Accuracy: {accuracies[best_acc_idx]*100:.2f}%)")
        else:
            print("No accuracy results to plot. Defaulting n_components.")
            OPTIMAL_N_COMPONENTS = N_COMPONENTS if max_possible_components > N_COMPONENTS else max_possible_components
            if OPTIMAL_N_COMPONENTS <= 0: OPTIMAL_N_COMPONENTS = min(10, train_faces.shape[1]-1 if train_faces.shape[1]>1 else 1)


    # --- Detailed Evaluation with Chosen/Optimal N_COMPONENTS ---
    if OPTIMAL_N_COMPONENTS <= 0 : # Ensure OPTIMAL_N_COMPONENTS is at least 1
        print(f"Warning: Optimal components is {OPTIMAL_N_COMPONENTS}. Setting to 1 for PCA.")
        OPTIMAL_N_COMPONENTS = 1
        
    print(f"\n--- Performing Detailed Evaluation with {OPTIMAL_N_COMPONENTS} PCA Components ---")
    pca = train_pca_model(train_faces, n_components=OPTIMAL_N_COMPONENTS, whiten=True, random_state=RANDOM_STATE)

    plot_cumulative_explained_variance(pca, filename=f"cumulative_variance_pca_{pca.n_components_}comps.png")
    plt.suptitle(f"Cumulative Explained Variance (PCA: {pca.n_components_} components)")
    # savefig is now inside plot_cumulative_explained_variance
    plt.show()


    eigenfaces = get_eigenfaces(pca, h, w)
    eigenface_titles = [f"Eigenface {i+1}" for i in range(eigenfaces.shape[0])]
    num_eigenfaces_to_plot = min(12, pca.n_components_)
    if pca.n_components_ > 0: # Ensure there are eigenfaces to plot
        plot_gallery(eigenfaces[:num_eigenfaces_to_plot],
                     eigenface_titles[:num_eigenfaces_to_plot],
                     h, w,
                     filename=f"eigenfaces_pca_{pca.n_components_}comps.png",
                     n_row=max(1, (num_eigenfaces_to_plot + 3) // 4),
                     n_col=4)
        plt.suptitle(f"Top {num_eigenfaces_to_plot} Eigenfaces (PCA: {pca.n_components_} components)")
        plt.show()
    else:
        print("No eigenfaces to plot as n_components is 0 or less.")


    print("\nTransforming data for detailed evaluation...")
    compressed_train_faces = pca.transform(train_faces)
    
    if len(test_faces) > 0:
        compressed_test_faces = pca.transform(test_faces)
        print(f"Compressed train_faces shape: {compressed_train_faces.shape}")
        print(f"Compressed test_faces shape: {compressed_test_faces.shape}")

        # --- Single Sample Matching Visualization (using the chosen PCA model) ---
        sample_idx_to_test = 17
        if sample_idx_to_test < len(test_faces) :
            test_sample_image = test_faces[sample_idx_to_test]
            compressed_test_sample = compressed_test_faces[sample_idx_to_test]
            true_test_label = test_targets[sample_idx_to_test]

            best_match_idx, best_match_error = find_best_match(compressed_test_sample, compressed_train_faces)
            
            print(f"\n--- Single Sample Matching Result (Test Sample {sample_idx_to_test}, PCA: {pca.n_components_} comps) ---")
            print(f"True person ID: {true_test_label}")
            print(f"Best match (train idx {best_match_idx}, ID: {train_targets[best_match_idx]}), Error: {best_match_error:.2f}")

            plot_single_match_result(
                test_sample_image, train_faces[best_match_idx],
                f"Test Sample {sample_idx_to_test} (ID: {true_test_label})",
                f"Best Match (ID: {train_targets[best_match_idx]})",
                h, w,
                filename=f"single_match_test{sample_idx_to_test}_pca_{pca.n_components_}comps.png"
            )
            plt.suptitle(f"Face Matching: Test Sample {sample_idx_to_test} (PCA: {pca.n_components_} comps)", fontsize=14)
            # savefig is now inside plot_single_match_result
            plt.show()
        else:
            print(f"\nSkipping single sample visualization: Test sample index {sample_idx_to_test} out of bounds ({len(test_faces)} samples).")

        # --- Systematic Evaluation ---
        print("\n--- Systematic Evaluation of Recognition System (PCA: {pca.n_components_} comps) ---")
        accuracy, y_true, y_pred, matched_train_indices = evaluate_recognition_system(
            compressed_test_faces,
            compressed_train_faces,
            test_targets,
            train_targets
        )
        print(f"Overall accuracy on the test set: {accuracy * 100:.2f}%")

        unique_labels_report = np.unique(np.concatenate((y_true, y_pred)))
        target_names_report = [f"Person {i}" for i in unique_labels_report]
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=unique_labels_report, target_names=target_names_report, zero_division=0))

        plot_confusion_matrix(y_true, y_pred,
                              class_names=[f"P{i}" for i in np.unique(targets)],
                              filename=f"confusion_matrix_pca_{pca.n_components_}comps.png")
        plt.suptitle(f"Confusion Matrix (PCA: {pca.n_components_} comps)", fontsize=16)
        plt.show()

        plot_example_matches(
            test_faces, train_faces,
            y_true, y_pred, matched_train_indices,
            train_targets, h, w, n_examples=8,
            filename=f"example_matches_pca_{pca.n_components_}comps.png"
        )
        plt.suptitle(f"Example Matches (PCA: {pca.n_components_} comps)", fontsize=16)
        plt.show()
        print("\nSystematic evaluation and detailed visualizations complete.")

        # --- Perform Residual Analysis on Misclassifications ---
        analyze_misclassification_residuals(
            pca, # trained with OPTIMAL_N_COMPONENTS
            test_faces, # original test faces
            compressed_test_faces, # test faces transformed by the PCA
            y_true, # true labels
            y_pred, # predicted labels
            h, w,
            n_residual_components=10 # components for PCA on residuals
        )
    else:
        print("Test set is empty. Cannot perform detailed evaluation or residual analysis.")


# --- Evaluation Logic ---
def evaluate_recognition_system(compressed_test_set, compressed_train_set, test_labels, train_labels):
    """
    Evaluates the face recognition system on the test set.

    Args:
        compressed_test_set (np.array): PCA-transformed test faces.
        compressed_train_set (np.array): PCA-transformed train faces.
        test_labels (np.array): True labels for the test faces.
        train_labels (np.array): True labels for the train faces.

    Returns:
        tuple: (accuracy, true_labels_list, predicted_labels_list, matched_train_indices_list)
    """
    predictions = []
    matched_train_indices_list = []
    correct_predictions = 0

    if len(test_labels) == 0: # Handle empty test set
        return 0.0, [], [], []

    for i in range(len(compressed_test_set)):
        test_sample_compressed = compressed_test_set[i]
        true_label = test_labels[i]
        
        best_match_idx, _ = find_best_match(test_sample_compressed, compressed_train_set)
        predicted_label = train_labels[best_match_idx]
        
        predictions.append(predicted_label)
        matched_train_indices_list.append(best_match_idx)
        
        if predicted_label == true_label:
            correct_predictions += 1
            
    accuracy = correct_predictions / len(test_labels)
    return accuracy, list(test_labels), predictions, matched_train_indices_list


# --- Face Matching Logic ---
def find_best_match(compressed_test_sample, compressed_train_set):
    """
    Finds the best match for a compressed test sample in the compressed training set.
    Uses squared Euclidean distance.
    
    Args:
        compressed_test_sample (np.array): The PCA-transformed features of a single test face.
        compressed_train_set (np.array): The PCA-transformed features of all training faces.
        
    Returns:
        tuple: (index_of_best_match, error_of_best_match)
    """
    # Calculate squared Euclidean distances
    squared_errors = np.sum((compressed_train_set - compressed_test_sample)**2, axis=1)
    best_match_idx = np.argmin(squared_errors)
    min_error = squared_errors[best_match_idx]
    return best_match_idx, min_error

# --- Result Visualization Functions ---
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
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(np.concatenate((y_true, y_pred)))) # Use all unique labels for CM
    fig = plt.figure(figsize=(12, 10))
    
    # Ensure class_names match the labels used in confusion_matrix
    # If class_names cover all possible people, but some aren't in y_true, CM will be smaller.
    unique_labels_in_data = np.unique(np.concatenate((y_true, y_pred)))
    # Filter class_names to only those present in the data, or create them if not enough provided
    if len(class_names) > len(unique_labels_in_data):
        relevant_class_names = [cn for i, cn in enumerate(class_names) if i in unique_labels_in_data]
        if len(relevant_class_names) != len(unique_labels_in_data): # fallback if mapping is complex
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
                         train_labels_original, # Labels for the training set
                         h, w, filename, n_examples=8):
    """Plots a few example matches, indicating correct/incorrect predictions and showing the matched image, then saves it."""
    
    num_available_samples = len(test_images_original)
    actual_n_examples = min(n_examples, num_available_samples)

    if actual_n_examples == 0:
        print("No examples to plot in plot_example_matches.")
        return

    plt.figure(figsize=(2.2 * actual_n_examples, 6.5)) # Adjusted for two rows and more spacing
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
        plt.subplot(2, actual_n_examples, plot_col_idx + 1) # subplot indices are 1-based
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
        plt.imshow(matched_img.reshape(h,w), cmap=plt.cm.gray, interpolation='nearest')
        plt.title(f"Best Match from Train\n(Train Idx: {matched_train_image_idx})\nActual ID: P{actual_label_of_matched_img}",
                    fontsize=9)
        plt.axis('off')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.93], h_pad=3.0) # Adjust layout for suptitle and padding
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    print(f"Saved example matches plot to {save_path}")

# --- Residual Analysis Functions ---
def analyze_misclassification_residuals(main_pca_model, original_test_faces, compressed_test_faces_by_main_pca,
                                      y_true, y_pred, h, w,
                                      n_residual_components=10, random_state=RANDOM_STATE):
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
                                       residual_images.shape[1] -1 if residual_images.shape[1]>1 else 1)
    
    if actual_n_residual_components <=0:
        print("Not enough residual samples or features to perform PCA on residuals. Skipping.")
        return

    print(f"Performing PCA on {len(residual_images)} residual images with {actual_n_residual_components} components...")
    pca_on_residuals = PCA(n_components=actual_n_residual_components, whiten=True, random_state=random_state)
    pca_on_residuals.fit(residual_images)

    eigen_residuals = pca_on_residuals.components_.reshape((actual_n_residual_components, h, w))
    eigen_residual_titles = [f"Eigen-Residual {i+1}" for i in range(actual_n_residual_components)]
    
    num_eigen_residuals_to_plot = min(12, actual_n_residual_components)

    plot_gallery(eigen_residuals[:num_eigen_residuals_to_plot],
                 eigen_residual_titles[:num_eigen_residuals_to_plot],
                 h, w,
                 filename=f"eigen_residuals_top{num_eigen_residuals_to_plot}.png",
                 n_row=max(1, (num_eigen_residuals_to_plot + 3) // 4),
                 n_col=4)
    plt.suptitle(f"Top {num_eigen_residuals_to_plot} Eigen-Residuals (from {len(residual_images)} misclassifications)", fontsize=14)
    plt.show()

    plot_cumulative_explained_variance(pca_on_residuals, filename="cumulative_variance_residuals_pca.png")
    plt.suptitle(f"Cumulative Explained Variance for PCA on Residuals ({pca_on_residuals.n_components_} components)")
    plt.show()

    print("Residual analysis complete.")


if __name__ == "__main__":
    main()