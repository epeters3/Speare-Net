import pickle
import numpy as np
import torch

def load(labels_path="labels.pkl"):
    """
    Loads the pickled labels and returns the one-hot
    encoded pytorch input data.
    """
    # Load the labels
    with open(labels_path, 'rb') as rf:
        labels = pickle.load(rf)
    num_labels = np.max(labels) + 1
    n = np.size(labels)

    # Build a return the onehots
    one_hots = np.zeros((n, num_labels))
    one_hots[np.arange(n), labels] = 1
    return torch.from_numpy(one_hots)