import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeareNet(nn.Module):
    """A char-net"""

    def __init__(self, in_size, h_size):
        super(SpeareNet, self).__init__()
        self.in_size = in_size
        self.h_size = h_size
        # batch_first – If True, then the input and output tensors are
        # provided as (batch, seq, feature). Default: False
        self.lstm = nn.LSTM(in_size, h_size)
        self.dense = nn.Linear(h_size, in_size)

    def forward(self, x):
        # "The first value returned by LSTM is all of the hidden states
        # throughout the sequence. The second is just the most recent
        # hidden state."
        # Source:
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        # "Pytorch’s LSTM expects all of its inputs to be 3D tensors."
        # "The first axis is the sequence itself, the second indexes
        # instances in the mini-batch, and the third indexes elements
        # of the input."
        output, (hidden_state, cell_state) = self.lstm(x)
        affined = self.dense(output[-1, :, :].unsqueeze(0))
        return F.softmax(affined, 2)

    def get_init_h(self, batch_size, seq_len):
        return torch.zeros(batch_size, seq_len - 1, self.h_size).requires_grad_()
