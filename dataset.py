import os

from torch.utils.data import Dataset, DataLoader

from parse_text import load_corpus


class CharNetDataset(Dataset):
    def __init__(self, corpus_path, *, seq_len):
        self.seq_len = seq_len
        one_hots, token_to_num, num_to_token, build_path = load_corpus(corpus_path)
        self.one_hots = one_hots
        self.token_to_num = token_to_num
        self.num_to_token = num_to_token
        self.build_path = build_path

        if not os.path.isdir(self.build_path):
            os.makedirs(self.build_path)

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, i):
        return self.one_hots[i : i + self.seq_len]

    def size(self):
        _, *rest = self.one_hots.size()
        return (self.num_seqs, *rest)

    @property
    def num_seqs(self):
        # This is the number of unique sequences that can be built.
        return self.one_hots.size(0) - self.seq_len + 1


def batches(dataloader: DataLoader):
    for batch in dataloader:
        # The incoming dims are: [batch_i, seq_i, char_onehot_i],
        # so we need to permute the first two
        batch = batch.permute(1, 0, 2)
        # Now the dims are [seq_i, batch_i, char_onehot_i]
        # Separate X and y
        X = batch[:-1, :, :]
        y = batch[-1, :, :]
        yield X, y
