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
        h1, _ = self.a_gru(x)
        return F.softmax(self.b_dense(h1), dim=2)


def train(data, *, epochs=100, report_every=100, checkpoint_every=10, seq_size=50, save_dir="checkpoints"):
    n, c = data.size()
    model = SpeareNet(c)
    if torch.cuda.is_available():
        model = model.cuda()
    params = model.parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, 1e-4)

    num_iters = n - seq_size
    sum_loss = 0.0
    num_correct = 0.0
    for epoch_i in range(1, epochs+1):
        iters_done = 0
        for data_i in range(num_iters):

            target_i = data_i + seq_size
            model.zero_grad()

            # Forward pass through this character sequence
            outputs = None
            for seq_i in range(data_i, target_i):
                inputs = data[seq_i, :].view(1, 1, -1)
                outputs = model(inputs)

            # Backpropagate through time. We are trying to
            # predict the character right after the input sequence,
            # the one-hot encoded character vector at `target_i`.
            target = data[target_i, :].view(1, -1).long()
            _, target_label = target.max(dim=1)
            outputs = outputs.view(1, -1)
            _, output_label = outputs.max(dim=1)
            if output_label == target_label:
                num_correct += 1

            loss = criterion(outputs, target_label)
            loss.backward()
            optimizer.step()
            sum_loss += loss
            iters_done += 1

            if iters_done % report_every == 0:
                # print(f"outputs.size={outputs.size()} target.size={target.size()}\noutputs={outputs}\ntarget={target}\ntarget_label={target_label}")
                avg_loss = sum_loss / report_every
                avg_accuracy = num_correct / report_every
                sum_loss = 0.0
                num_correct = 0.0
                print(
                    f"[{round(iters_done/num_iters*100, 4)}% of epoch {epoch_i}]:"
                    f"\n\tavg_loss = {avg_loss}"
                    f"\n\tavg_accuracy = {avg_accuracy}"
                )

        # Checkpoint the model after `checkpoint_every` epochs
        if epoch_i % checkpoint_every == 0:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            torch.save(model.state_dict(), f"{save_dir}/epoch_{epoch_i}.pt")


if __name__ == "__main__":
    data = load(text_path="sixpence_small.txt", )
    train(data, epochs=400, seq_size=20, report_every=170)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
