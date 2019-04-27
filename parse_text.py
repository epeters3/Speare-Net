import json
import numpy as np
import pickle

def make_codex(text_path, write_path="codex.json"):
    """
    Parses the file at `text_path`, finding all unique
    characters in the file, and encoding them in a vector.
    Each char's code will be the index of its
    position in `codex`, the array that is written to
    `write_path` as JSON. 
    """
    codex = []
    char_cnt = 0
    token_cnt = 0
    print("Encoding document: {}".format(text_path))
    with open(text_path, 'r') as rf:
        for line in rf:
            for char in line:
                char_cnt += 1
                if char not in codex:
                    token_cnt += 1
                    codex.append(char)

    with open(write_path, 'w') as wf:
        print("Writing codex to document: {}".format(write_path))
        json.dump({
            "char_cnt": char_cnt,
            "token_cnt": token_cnt,
            "codex": codex
        }, wf)

def make_labels(codex_path, text_path, labels_path):
    """
    Uses the codex at `codex_path` (stored as JSON), and
    the text at `text_path`, and builds a numpy array
    that holds the encoded integers for each char in
    the document. Can be used to make a one-hot matrix.
    """
    with open(codex_path, 'r') as rmf:
        print("Reading codex data: {}".format(codex_path))
        text_meta = json.load(rmf)
    char_cnt = text_meta['char_cnt']
    token_cnt = text_meta['token_cnt']
    codex = text_meta['codex']

    labels = np.zeros((char_cnt), dtype=np.int32)
    
    # Populate the labels
    char_idx = 0
    with open(text_path, 'r') as rtf:
        print("Reading text file: {}".format(text_path))
        for line in rtf:
            for char in line:
                # Use the integer encoding of this char
                # as its label.
                labels[char_idx] = codex.index(char)
                char_idx += 1
    
    # Write out the labels
    with open(labels_path, 'wb') as wf:
        print(labels[0:40])
        print("Writing labels data to: {}".format(labels_path))
        pickle.dump(labels, wf)

if __name__ == "__main__":
    text_path = "t8.shakespeare.txt"
    codex_path = "codex.json"
    # make_codex(text_path, codex_path)
    make_labels(codex_path, text_path, "labels.pkl")