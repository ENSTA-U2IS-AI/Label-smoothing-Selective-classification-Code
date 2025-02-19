"""This file is a filtered copy of torchtext.vocab.

Torchtext is deprecated and hence not imported as a dependency. Copyright to the owners of the torchtext.
"""

import gzip
import logging
import os
import re
import tarfile
import zipfile
from urllib.request import urlretrieve

import torch
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def reporthook(t):
    """https://github.com/tqdm/tqdm."""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """b: int, optional
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def _infer_shape(f):
    num_lines, vector_dim = 0, None
    for line in f:
        if vector_dim is None:
            row = line.rstrip().split(b" ")
            vector = row[1:]
            # Assuming word, [vector] format
            if len(vector) > 2:
                # The header present in some (w2v) formats contains two elements.
                vector_dim = len(vector)
                num_lines += 1  # First element read
        else:
            num_lines += 1
    f.seek(0)
    return num_lines, vector_dim


class Vectors:
    def __init__(self, name, cache=None, url=None, unk_init=None, max_vectors=None) -> None:
        """Args:
        name: name of the file that contains the vectors
        cache: directory for cached vectors
        url: url for download if vectors not found in cache
        unk_init (callback): by default, initialize out-of-vocabulary word vectors
            to zero vectors; can be any function that takes in a Tensor and returns a Tensor of the same size
        max_vectors (int): this can be used to limit the number of
            pre-trained vectors loaded.
            Most pre-trained vector sets are sorted
            in the descending order of word frequency.
            Thus, in situations where the entire set doesn't fit in memory,
            or is not needed for another reason, passing `max_vectors`
            can limit the size of the loaded set.

        """
        cache = ".vector_cache" if cache is None else cache
        self.itos = None
        self.stoi = None
        self.vectors = None
        self.dim = None
        self.unk_init = torch.Tensor.zero_ if unk_init is None else unk_init
        self.cache(name, cache, url=url, max_vectors=max_vectors)

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        return self.unk_init(torch.Tensor(self.dim))

    def __contains__(self, token):
        return token in self.stoi

    def cache(self, name, cache, url=None, max_vectors=None):
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
        if os.path.isfile(name):
            path = name
            file_suffix = f"_{max_vectors}.pt" if max_vectors else ".pt"
            path_pt = os.path.join(cache, os.path.basename(name)) + file_suffix
        else:
            path = os.path.join(cache, name)
            file_suffix = f"_{max_vectors}.pt" if max_vectors else ".pt"
            path_pt = path + file_suffix

        if not os.path.isfile(path_pt):
            if not os.path.isfile(path) and url:
                logger.info(f"Downloading vectors from {url}")
                if not os.path.exists(cache):
                    os.makedirs(cache)
                dest = os.path.join(cache, os.path.basename(url))
                if not os.path.isfile(dest):
                    with tqdm(unit="B", unit_scale=True, miniters=1, desc=dest) as t:
                        try:
                            urlretrieve(url, dest, reporthook=reporthook(t))
                        except KeyboardInterrupt:  # remove the partial zip file
                            os.remove(dest)
                            raise
                logger.info(f"Extracting vectors into {cache}")
                ext = os.path.splitext(dest)[1][1:]
                if ext == "zip":
                    with zipfile.ZipFile(dest, "r") as zf:
                        zf.extractall(cache)
                elif ext == "gz" and dest.endswith(".tar.gz"):
                    with tarfile.open(dest, "r:gz") as tar:
                        tar.extractall(path=cache)
            if not os.path.isfile(path):
                raise RuntimeError(f"no vectors found at {path}")

            logger.info(f"Loading vectors from {path}")
            ext = os.path.splitext(path)[1][1:]
            open_file = gzip.open if ext == "gz" else open

            vectors_loaded = 0
            with open_file(path, "rb") as f:
                num_lines, dim = _infer_shape(f)
                if not max_vectors or max_vectors > num_lines:
                    max_vectors = num_lines

                itos, vectors, dim = [], torch.zeros((max_vectors, dim)), None

                for line in tqdm(f, total=max_vectors):
                    # Explicitly splitting on " " is important, so we don't
                    # get rid of Unicode non-breaking spaces in the vectors.
                    entries = line.rstrip().split(b" ")

                    word, entries = entries[0], entries[1:]
                    if dim is None and len(entries) > 1:
                        dim = len(entries)
                    elif len(entries) == 1:
                        logger.warning(f"Skipping token {word} with 1-dimensional " f"vector {entries}; likely a header")
                        continue
                    elif dim != len(entries):
                        raise RuntimeError(
                            f"Vector for token {word} has {len(entries)} dimensions, but previously "
                            f"read vectors have {dim} dimensions. All vectors must have "
                            "the same number of dimensions."
                        )

                    try:
                        if isinstance(word, bytes):
                            word = word.decode("utf-8")
                    except UnicodeDecodeError:
                        logger.info(f"Skipping non-UTF8 token {word!r}")
                        continue

                    vectors[vectors_loaded] = torch.tensor([float(x) for x in entries])
                    vectors_loaded += 1
                    itos.append(word)

                    if vectors_loaded == max_vectors:
                        break

            self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).view(-1, dim)
            self.dim = dim
            logger.info(f"Saving vectors to {path_pt}")
            if not os.path.exists(cache):
                os.makedirs(cache)
            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:
            logger.info(f"Loading vectors from {path_pt}")
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt, weights_only=True)

    def __len__(self):
        """Returns the number of vectors in the object."""
        return len(self.vectors)

    def get_vecs_by_tokens(self, tokens, lower_case_backup=False):
        """Look up embedding vectors of tokens.

        Args:
            tokens: a token or a list of tokens. if `tokens` is a string,
                returns a 1-D tensor of shape `self.dim`; if `tokens` is a
                list of strings, returns a 2-D tensor of shape=(len(tokens),
                self.dim).
            lower_case_backup : Whether to look up the token in the lower case.
                If False, each token in the original case will be looked up;
                if True, each token in the original case will be looked up first,
                if not found in the keys of the property `stoi`, the token in the
                lower case will be looked up. Default: False.

        Examples:
            >>> examples = ['chip', 'baby', 'Beautiful']
            >>> vec = text.vocab.GloVe(name='6B', dim=50)
            >>> ret = vec.get_vecs_by_tokens(examples, lower_case_backup=True)

        """
        to_reduce = False

        if not isinstance(tokens, list):
            tokens = [tokens]
            to_reduce = True

        if not lower_case_backup:
            indices = [self[token] for token in tokens]
        else:
            indices = [self[token] if token in self.stoi else self[token.lower()] for token in tokens]

        vecs = torch.stack(indices)
        return vecs[0] if to_reduce else vecs


class GloVe(Vectors):
    url = {
        "42B": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
        "840B": "http://nlp.stanford.edu/data/glove.840B.300d.zip",
        "twitter.27B": "http://nlp.stanford.edu/data/glove.twitter.27B.zip",
        "6B": "http://nlp.stanford.edu/data/glove.6B.zip",
    }

    def __init__(self, name="840B", dim=300, **kwargs) -> None:
        url = self.url[name]
        name = f"glove.{name}.{dim!s}d.txt"
        super().__init__(name, url=url, **kwargs)


_patterns = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]

_replacements = [" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "]

_patterns_dict = [(re.compile(p), r) for p, r in zip(_patterns, _replacements, strict=False)]


def basic_english_normalize(line):
    r"""Basic normalization for a line of text.
    Normalization includes
    - lowercasing
    - complete some basic text normalization for English words as follows:
        add spaces before and after '\''
        remove '\"',
        add spaces before and after '.'
        replace '<br \/>'with single space
        add spaces before and after ','
        add spaces before and after '('
        add spaces before and after ')'
        add spaces before and after '!'
        add spaces before and after '?'
        replace ';' with single space
        replace ':' with single space
        replace multiple spaces with single space.

    Returns a list of tokens after splitting on whitespace.
    """
    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line


def truncate(inputs, max_seq_len: int):
    """Truncate inputs sequence or batch.

    :param inputs: Inputs sequence or batch to be truncated
    :type inputs: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    :param max_seq_len: Maximum length beyond which inputs is discarded
    :type max_seq_len: int
    :return: Truncated sequence
    :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    """
    if torch.jit.isinstance(inputs, list[int]) or torch.jit.isinstance(inputs, list[str]):
        return inputs[:max_seq_len]
    if torch.jit.isinstance(inputs, list[list[int]]):
        output: list[list[int]] = []
        for ids in inputs:
            output.append(ids[:max_seq_len])
        return output
    if torch.jit.isinstance(inputs, list[list[str]]):
        output: list[list[str]] = []
        for ids in inputs:
            output.append(ids[:max_seq_len])
        return output
    raise TypeError("Inputs type not supported")


class Truncate(nn.Module):
    r"""Truncate inputs sequence.

    :param max_seq_len: The maximum allowable length for inputs sequence
    :type max_seq_len: int
    """

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, inputs):
        """:param inputs: Inputs sequence or batch of sequence to be truncated
        :type inputs: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        :return: Truncated sequence
        :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """
        return truncate(inputs, self.max_seq_len)
