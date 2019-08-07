import pickle
import os
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from parse_text import make_codex, make_labels, make_sequences, make_simplified_corpus


def strip_path_and_ext(file_path):
    file_ext_regex = r"\.[^\.]+$"
    corpus_file_name = file_path.split("/")[-1]
    return re.sub(file_ext_regex, "", corpus_file_name)


def load(
    corpus_path="data/t8.shakespeare.txt", labels_path="data/labels.pkl", seq_len=20
):
    """
    Loads the pickled labels and returns the one-hot
    encoded pytorch input data.
    """

    # Resolve all needed prerequisite files

    corpus_name = strip_path_and_ext(corpus_path)
    build_path = f"model_data/{corpus_name}"
    simplified_corpus_path = f"{build_path}/simple-corpus.txt"
    sequences_path = f"{build_path}/sequences.txt"
    codex_path = f"{build_path}/codex.json"
    labels_path = f"{build_path}/labels.pkl"

    if not os.path.isfile(corpus_path):
        raise Exception(f"No file was found at '{corpus_path}'")

    if not os.path.isdir(build_path):
        os.makedirs(build_path)

    if not os.path.isfile(simplified_corpus_path):
        print(f"Making simplified corpus at '{simplified_corpus_path}'...")
        make_simplified_corpus(corpus_path, simplified_corpus_path)

    if not os.path.isfile(codex_path):
        print(f"Making codex at '{codex_path}'...")
        make_codex(simplified_corpus_path, codex_path)

    if not os.path.isfile(sequences_path):
        print(f"Making sequences at '{sequences_path}'...")
        make_sequences(simplified_corpus_path, sequences_path)

    if not os.path.isfile(labels_path):
        print(f"Making labels at '{labels_path}'...")
        make_labels(codex_path, sequences_path, labels_path, seq_len)

    # Load the labels
    with open(labels_path, "rb") as rf:
        print(f"Loading labels from '{labels_path}'...")
        labels = pickle.load(rf)

    num_labels = np.max(labels) + 1
    num_seqs = np.shape(labels)[0]

    # Build and return the onehots
    one_hots = torch.zeros(num_seqs, seq_len, num_labels)
    for i in range(num_seqs):
        for j in range(seq_len):
            char_label = labels[i][j]
            one_hots[i][j][char_label] = 1

    if torch.cuda.is_available():
        return one_hots.cuda()
    return one_hots, build_path


class SpeareNet(nn.Module):
    """A char-net"""

    def __init__(self, in_size, h_size=200):
        super(SpeareNet, self).__init__()
        self.in_size = in_size
        self.h_size = h_size
        self.gru = nn.GRU(in_size + h_size, h_size, num_layers=2)
        self.dense = nn.Linear(h_size, in_size)

    def forward(self, x, hidden):
        h1, _ = self.gru(torch.cat((x, hidden)))
        return F.softmax(self.dense(h1), dim=2)

    def get_init_h(self):
        return torch.zeros(self.h_size)


def train(data, build_path, *, epochs=100, report_every=100, checkpoint_every=10):
    num_seqs, seq_len, input_len = data.size()
    model = SpeareNet(input_len)
    if torch.cuda.is_available():
        model = model.cuda()
    params = model.parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, 1e-4)

    sum_loss = 0.0
    num_correct = 0.0
    for epoch_i in range(1, epochs + 1):
        iters_done = 0
        for seq_i in np.random.permutation(num_seqs):

            model.zero_grad()

            # Forward pass through this character sequence
            outputs = model.get_init_h()
            for char_i in range(seq_len - 1):
                inputs = data[seq_i, char_i, :].view(1, 1, -1)
                outputs = model(inputs, outputs)

            # Backpropagate through time. We are trying to
            # predict the character right after the input sequence,
            # the one-hot encoded character vector at `target_i`.
            target = data[seq_i, -1, :].view(1, -1).long()
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
                    f"[{round(iters_done/num_seqs*100, 4)}% of epoch {epoch_i}]:"
                    f"\n\tavg_loss = {avg_loss}"
                    f"\n\tavg_accuracy = {avg_accuracy}"
                )

        # Checkpoint the model after `checkpoint_every` epochs
        if epoch_i % checkpoint_every == 0:
            torch.save(model.state_dict(), f"{build_path}/epoch_{epoch_i}.pt")


if __name__ == "__main__":
    data, build_path = load(corpus_path="datasets/sixpence.txt")
    train(data, build_path, epochs=400, report_every=170)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
