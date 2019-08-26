import json
import numpy as np
import pickle
import re
import os
from typing import List

import torch


def simplify_corpus(corpus: str, remove: List[str], to_lower: bool = True):
    for token in remove:
        corpus = corpus.replace(token, "")
    if to_lower:
        corpus = corpus.lower()
    return corpus


def strip_path_and_ext(file_path):
    file_ext_regex = r"\.[^\.]+$"
    corpus_file_name = file_path.split("/")[-1]
    return re.sub(file_ext_regex, "", corpus_file_name)


def load_corpus(corpus_path):

    corpus_name = strip_path_and_ext(corpus_path)
    build_path = f"model_data/{corpus_name}"
    # pkl_path = f"{build_path}/dataset.pkl"

    if not os.path.isfile(corpus_path):
        raise Exception(f"No file was found at '{corpus_path}'")

    # Load the corpus
    with open(corpus_path, "r") as rf:
        print(f"Loading data from '{corpus_path}'...")
        corpus = rf.read()

    corpus = simplify_corpus(corpus, [])
    n = len(corpus)
    tokens = set(corpus)
    token_to_num = {token: i for i, token in enumerate(tokens)}
    num_to_token = {i: token for i, token in enumerate(tokens)}

    # Build the onehots. Each row is a onehot
    # encoding of a char in the corpus.
    one_hots = torch.zeros(n, len(tokens))
    for i, token in enumerate(corpus):
        one_hots[i][token_to_num[token]] = 1

    if torch.cuda.is_available():
        return one_hots.cuda()
    return one_hots, token_to_num, num_to_token, build_path
