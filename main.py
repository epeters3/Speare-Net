import torch

from dataset import CharNetDataset
from train import train


if __name__ == "__main__":
    dataset = CharNetDataset("datasets/sixpence.txt", seq_len=25)
    train(dataset, epochs=400, report_every=50, h_size=32)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
