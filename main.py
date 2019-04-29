import pickle
import os

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

    def __init__(self):
        super(SpeareNet, self).__init__()
        self.a_lstm = nn.LSTM(50, 120, num_layers=2)
        self.b_dense = nn.Linear(120, 120)
        self.c_dense = nn.Linear(120, 50)
    
    def forward(self, x):
        h1 = F.relu(self.a_lstm(x))
        h2 = F.relu(self.b_dense(h1))
        return F.sigmoid(self.c_dense(h2))

def train(data, *, epochs=100, checkpoint_every=10):
    model = SpeareNet()
    params = model.parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params)

    for epoch_i in range(1, epochs+1):
        # Prep
        for data_i in range(data.size()[0]):
            model.zero_grad()
            inputs = None # TODO inputs
            target = None # TODO targets

            # Forward pass
            outputs = model(inputs)

            # Backpropagate
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    data = load()
    train(data)