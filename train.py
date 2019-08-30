import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from model import SpeareNet
from dataset import CharNetDataset, batches


def train(
    dataset: CharNetDataset,
    *,
    epochs=100,
    report_every=100,
    checkpoint_every=10,
    grad_clip=4,
    batch_size=8,
    h_size=100,
):
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    num_batches = len(dataloader)
    _, cols = dataset.size()
    model = SpeareNet(cols, h_size)
    if torch.cuda.is_available():
        model = model.cuda()
    params = model.parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, 1e-5)

    sum_loss = 0.0
    num_correct = 0.0

    print(f"Training {num_batches} batches for {epochs} epochs...")

    for epoch_i in range(1, epochs + 1):
        batches_done = 0

        for X, y in batches(dataloader):
            # print(f"dims of X: {X.size()}")
            # print(f"dims of y: {y.size()}")

            model.zero_grad()
            outputs = model(X)
            outputs = outputs.squeeze(0)
            # print(f"dims of outputs: {outputs.size()}")

            # Backpropagate through time. We are trying to
            # predict the character right after the input sequence,
            # the one-hot encoded character vector at `target_i`.
            _, target_labels = y.max(dim=1)
            # print(f"dims of target_labels: {target_labels.size()}")
            # print(f"target_labels: {target_labels}")
            _, output_labels = outputs.max(dim=1)
            # print(f"dims of output_labels: {output_labels.size()}")
            # print(f"output_labels: {output_labels}")

            num_correct += torch.sum(output_labels == target_labels)

            loss = criterion(outputs, target_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            sum_loss += loss
            batches_done += 1

            if batches_done % report_every == 0:
                target_chars = [dataset.num_to_token[x] for x in target_labels.cpu().numpy()]
                output_chars = [dataset.num_to_token[x] for x in output_labels.cpu().numpy()]
                avg_loss = sum_loss / report_every
                avg_accuracy = num_correct // (report_every * batch_size)
                sum_loss = 0.0
                num_correct = 0.0
                print(
                    f"[{round(batches_done/num_batches*100, 4)}% of epoch {epoch_i}]:"
                    f"\n\tavg_loss = {avg_loss}"
                    f"\n\tavg_accuracy = {avg_accuracy}"
                    f"\n\toutput_chars={output_chars}"
                    f"\n\ttarget_chars={target_chars}"
                )

        # Checkpoint the model after `checkpoint_every` epochs
        if epoch_i % checkpoint_every == 0:
            torch.save(model.state_dict(), f"{dataset.build_path}/epoch_{epoch_i}.pt")
