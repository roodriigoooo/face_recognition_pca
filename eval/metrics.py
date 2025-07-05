import numpy as np

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

    if len(test_labels) == 0:  # Handle empty test set
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
    squared_errors = np.sum((compressed_train_set - compressed_test_sample) ** 2, axis=1)
    best_match_idx = np.argmin(squared_errors)
    min_error = squared_errors[best_match_idx]
    return best_match_idx, min_error
