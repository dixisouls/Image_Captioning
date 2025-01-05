import nltk
import pickle
from collections import Counter
from config import Config
import pandas as pd
import os
from tqdm import tqdm


class Vocabulary:
    """
    A class to represent a vocabulary for text processing.

    Attributes:
        word2idx (dict): A dictionary mapping words to their indices.
        idx2word (dict): A dictionary mapping indices to their corresponding words.
        idx (int): The current index for the next new word.
        config (Config): Configuration object containing various parameters.
    """

    def __init__(self, config):
        """
        Initialize the Vocabulary object.

        Args:
            config (Config): Configuration object containing various parameters.
        """
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.config = config

    def add_word(self, word):
        """
        Add a word to the vocabulary.

        Args:
            word (str): The word to be added.
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        """
        Get the index of a word. If the word is not in the vocabulary, return the index of '<unk>'.

        Args:
            word (str): The word to look up.

        Returns:
            int: The index of the word.
        """
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        """
        Get the number of words in the vocabulary.

        Returns:
            int: The number of words in the vocabulary.
        """
        return len(self.word2idx)


def build_vocab(csv_file, threshold):
    """
    Build a vocabulary from a COCO annotations file.

    Args:
        csv_file (str): Path to the COCO annotations file.
        threshold (int): Minimum frequency of words to be included in the vocabulary.

    Returns:
        Vocabulary: The built vocabulary object.
    """
    captions_df = pd.read_csv(csv_file, delimiter="|")
    counter = Counter()

    progress = tqdm(
        enumerate(captions_df.iterrows()),
        total=len(captions_df),
        desc="Building Vocabulary",
    )
    for i, (index, row) in progress:
        caption = str(row[" comment"]).strip()
        tokens = nltk.word_tokenize(caption.lower())
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary(Config)
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")

    for word in words:
        vocab.add_word(word)
    return vocab


def get_vocab():
    """
    Get the vocabulary, loading it from a file if it exists, or building it if it does not.

    Returns:
        Vocabulary: The vocabulary object.
    """
    config = Config()
    vocab_path = config.VOCAB_FILE

    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    else:
        print("Vocabulary not found. Creating vocabulary.")
        vocab = build_vocab(config.CAPTIONS_FILE, config.VOCAB_THRESHOLD)
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary created and saved at {vocab_path}")
    return vocab
