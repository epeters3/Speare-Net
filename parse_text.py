import json
import numpy as np
import pickle


def make_simplified_corpus(
    text_path="data/t8.shakespeare.txt",
    write_path="data/simple-corpus.txt",
    remove=["\n"],
    to_lower=True,
):
    """
    Eliminate all tokens in `remove` from the file at
    `text_path`, also optionally converting the file to lower
    case too.
    """
    print(f"Simplifying '{text_path}'...")
    with open(text_path, "r") as rf:
        with open(write_path, "w") as wf:
            for corpus_line in rf.readlines():
                for token in remove:
                    corpus_line = corpus_line.replace(token, "")
                # Add a space at the end in case the next line
                # doesn't begin with a space.
                corpus_line += " "
                if to_lower:
                    corpus_line = corpus_line.lower()
                wf.write(corpus_line)
    print(f"Results written to '{write_path}'")


def make_sequences(
    text_path="data/simple-corpus.txt", write_path="data/sequences.txt", seq_len=20
):
    print(f"Constructing sequences of length {seq_len} from '{text_path}'")
    current_seq = ""
    with open(text_path, "r") as rf:
        with open(write_path, "w") as wf:
            for line in rf:
                for char in line:
                    # Append this character to the end of `current_seq`
                    current_seq += char
                    if len(current_seq) > seq_len:
                        # Trim to keep just the last `seq_len`
                        # characters of `current_seq`.
                        current_seq = current_seq[-seq_len:]
                    if len(current_seq) == seq_len:
                        wf.write(f"{current_seq}\n")


def make_codex(text_path="data/simple-corpus.txt", write_path="data/codex.json"):
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
    print(f"Encoding document: '{text_path}'...")
    with open(text_path, "r") as rf:
        for line in rf:
            for char in line:
                char_cnt += 1
                if char not in codex:
                    token_cnt += 1
                    codex.append(char)

    with open(write_path, "w") as wf:
        print(f"Writing codex to document: '{write_path}'...")
        json.dump({"char_cnt": char_cnt, "token_cnt": token_cnt, "codex": codex}, wf)


def make_labels(
    codex_path="data/codex.json",
    sequences_path="data/t8.shakespeare.txt",
    labels_path="data/labels.pkl",
    seq_len=20,
):
    """
    Uses the codex at `codex_path` (stored as JSON), and
    the text at `sequences_path`, and builds a numpy matrix
    that holds the encoded integers for each sequence in
    the document.
    """
    with open(codex_path, "r") as rmf:
        print(f"Reading codex data: '{codex_path}'...")
        text_meta = json.load(rmf)
    char_cnt = text_meta["char_cnt"]
    token_cnt = text_meta["token_cnt"]
    codex = text_meta["codex"]

    labels = np.zeros((char_cnt, seq_len), dtype=np.int32)

    # Populate the labels
    seq_idx = 0
    with open(sequences_path, "r") as rtf:
        print(f"Reading text file: '{sequences_path}'...")
        for line in rtf:
            # Don't consider the last char of line here
            # since it is always '\n'
            for char_idx, char in enumerate(line[:-1]):
                # Use the integer encoding of this char
                # as its label.
                labels[seq_idx, char_idx] = codex.index(char)
            seq_idx += 1

    # Write out the labels
    with open(labels_path, "wb") as wf:
        print(f"Writing labels data to: '{labels_path}'...")
        pickle.dump(labels, wf)
