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
    torch_one_hots = torch.from_numpy(one_hots).float()

    if torch.cuda.is_available():
        return torch_one_hots.cuda()
    return torch_one_hots

class SpeareNet(nn.Module):
    """A char-net"""

    def __init__(self, in_size, h_size=200):
        super(SpeareNet, self).__init__()
        self.a_gru = nn.GRU(in_size, h_size, num_layers=2)
        self.b_dense = nn.Linear(h_size, in_size)
    
    def forward(self, x):
        _, h1 = self.a_gru(x)
        return F.softmax(self.b_dense(h1), dim=2)


def train(data, *, epochs=100, report_every=100, seq_size=50, save_dir="checkpoints"):
    n, c = data.size()
    model = SpeareNet(c)
    if torch.cuda.is_available():
        model = model.cuda()
    params = model.parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, 1e-4)

    num_iters = n - seq_size
    sum_loss = 0.0
    for epoch_i in range(1, epochs+1):
        iters_done = 0
        for data_i in range(num_iters):

            target_i = data_i + seq_size
            model.zero_grad()

            # Forward pass through this character sequence
            # of length `seq_size`.
            inputs = data[data_i:target_i, :].view(seq_size, 1, -1)
            outputs = model(inputs)

            # Backpropagate through time. We are trying to
            # predict the character right after the input sequence,
            # the one-hot encoded character vector at `target_i`.
            target = data[target_i, :].view(1, -1).long()
            _, target_label = target.max(dim=1)
            outputs = outputs.view(1, -1)

            loss = criterion(outputs, target_label)
            sum_loss += loss
            loss.backward()
            optimizer.step()
            iters_done += 1

            if data_i % report_every == 0:
                # print(f"outputs={outputs}\ntarget_label={target_label}")
                avg_loss = sum_loss / report_every
                sum_loss = 0.0
                print(f"loss @[{round(iters_done/num_iters*100, 4)}%] = {avg_loss}")

        # Checkpoint the model after each epoch
        torch.save(model.state_dict(), f"{save_dir}/epoch_{epoch_i}.pt")


if __name__ == "__main__":
    data = load()
    train(data, seq_size=20, report_every=500)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
