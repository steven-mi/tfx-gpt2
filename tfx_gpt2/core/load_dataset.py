import os
import logging
import numpy as np
import pandas as pd
import tqdm


def load_dataset(enc, path, encoding=None, end_token=None):
    paths = []
    for (dirpath, _, fnames) in os.walk(path):
        for fname in fnames:
            paths.append(os.path.join(dirpath, fname))

    token_chunks = []
    for path in paths:
        logging.info("loading file from {}".format("path"))
        if path.endswith('.npz'):
            # Pre-encoded
            logging.info("found .npz, loading pre encoded dataset")
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:
            # big merged text
            with open(path) as file:
                logging.info("found merged text")
                raw_text = file.read()
                logging.info("merged text has len {}".format(len(raw_text)))
                raw_text = raw_text.split(end_token)
                logging.info("and {} text documents".format(len(raw_text)))
                for text in raw_text:
                    if text:
                        tokens = np.stack(enc.encode(text))
                        token_chunks.append(tokens)
    logging.info("found {} tokens".format("token_chunks"))
    return token_chunks


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks, seed=None):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])
        self.rs = np.random.RandomState(seed=seed)

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(
            length)
        while True:
            index = self.rs.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]
