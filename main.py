import pickle
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from parse_text import make_codex, make_labels

def load(text_path="t8.shakespeare.txt", codex_path="codex.json", labels_path="labels.pkl"):
    """
    Loads the pickled labels and returns the one-hot
    encoded pytorch input data.
    """

    # Resolve all needed prerequisite files

    if not os.path.isfile(text_path):
        raise Exception("No file was found at '{}'".format(text_path))
    
    if not os.path.isfile(codex_path):
        print("Making codex at '{}'...".format(codex_path))
        make_codex(text_path, codex_path)
    
    if not os.path.isfile(labels_path):
        print("Making labels at '{}'...".format(labels_path))
        make_labels(codex_path, text_path, labels_path)

    # Load the labels
    with open(labels_path, 'rb') as rf:
        print("Loading labels from '{}'...".format(labels_path))
        labels = pickle.load(rf)
    num_labels = np.max(labels) + 1
    n = np.size(labels)

    # Build and return the onehots
    one_hots = np.zeros((n, num_labels))
    one_hots[np.arange(n), labels] = 1
    return torch.from_numpy(one_hots)

class SpeareNet(nn.Module):
    """A char-net"""

    def __init__(self, in_size, h_size=120):
        super(SpeareNet, self).__init__()
        self.a_lstm = nn.LSTM(in_size, h_size, num_layers=2)
        self.b_dense = nn.Linear(h_size, h_size)
        self.c_dense = nn.Linear(h_size, in_size)
    
    def forward(self, x):
        h1 = F.relu(self.a_lstm(x))
        h2 = F.relu(self.b_dense(h1))
        return F.sigmoid(self.c_dense(h2))

def train(data, *, epochs=100, checkpoint_every=10, seq_size=50):
    print(f"data.dtype: {data.dtype}")
    n, c = data.size()
    model = SpeareNet(c)
    params = model.parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params)

    num_iters = n - seq_size
    best_loss = math.inf
    for epoch_i in range(1, epochs+1):
        target_i, outputs = None, None
        sum_loss = 0.0
        for data_i in range(num_iters):

            target_i = data_i + seq_size
            model.zero_grad()

            for i in range(data_i, target_i):
                # Forward pass through this character sequence
                # of length `seq_size`.
                outputs = model(data[i, :].view(1, 1, -1))

            # Backpropagate through time. We are trying to
            # predict the character right after the input sequence,
            # the one-hot encoded character vector at `target_i`.
            loss = criterion(outputs, data[target_i, :])
            sum_loss += loss
            loss.backward()
            optimizer.step()

        print(f"[{epoch_i}] ==> Avg. Loss: {sum_loss / num_iters}")

if __name__ == "__main__":
    data = load()
    train(data)