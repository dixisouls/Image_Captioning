import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from config import Config
from collections import Counter
from nltk.tokenize import word_tokenize
from utils import save_object, load_object, setup_logger
import nltk
from tqdm import tqdm

nltk.download("punkt")

logger = setup_logger(__name__)


class Vocabulary:
    """Manages the vocabulary for the image captioning model"""

    def __init_(
        self, freq_threshold=Config.MIN_WORD_FREQ, vocab_size=Config.VOCABULARY_SIZE
    ):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        """Builds the vocabulary from a list of sentences"""

        frequencies = Counter()
        idx = 4

        for sentence in tqdm(sentence_list, desc="Building vocabulary"):
            for word in word_tokenize(sentence.lower()):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

        # select top n words based on frequency
        top_words = frequencies.most_common(self.vocab_size)
        for word, _ in top_words:
            if word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        """Convert text string to list of numerical indices"""
        tokenized_text = word_tokenize(text.lower())

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]


class FlickerDataset(Dataset):
    """Flicker30k dataset"""

    def __init__(self, image_dir, caption_file, vocab, transform=None, mode="train"):
        """
        Args:
        image_dir: Path to the directory with images
        caption_file: Path to the file with captions
        vocab: Vocabulary object
        transform: Image transformation
        mode: Train, validation or test mode
        """
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.vocab = vocab

        df = pd.read_csv(caption_file, sep="|", engine="python")
        df.columns = ["image", "comment_number", "caption"]
        df["caption"] = df["caption"].str.strip()
        df = df.drop("comment_number", axis=1)
        df = df.groupby("image")["caption"].apply(list).reset_index(name="captions")

        # split data into train, validation and test
        train_split = int(len(df) * Config.TRAIN_SPLIT)
        val_split = int(len(df) * (Config.TRAIN_SPLIT + Config.VAL_SPLIT))

        if self.mode == "train":
            self.data = df[:train_split]
        elif self.mode == "val":
            self.data = df[train_split:val_split]
        elif self.mode == "test":
            self.data = df[val_split:]
        else:
            raise ValueError("Invalid mode. Use train, val or test")

        self.image_names = self.data["image"].tolist()
        self.captions = self.data["captions"].tolist()

        logger.info(f"Loaded {len(self.image_names)} samples for {self.mode} mode")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        caption_list = self.captions[idx]

        # load and preprocess image
        img_path = os.path.join(self.image_dir, image_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"Image not found: {img_path}")
            return None, None
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)

        # numericalize and pad captions
        numericalized_captions = []
        for caption in caption_list:
            caption_vec = [self.vocab.stoi["<start>"]]
            caption_vec += self.vocab.numericalize(caption)
            caption_vec.append(self.vocab.stoi["<end>"])
            numericalized_captions.append(torch.tensor(caption_vec))

        return image, numericalized_captions


def pad_sequences(batch):
    # seperate images and captions
    images, captions = zip(*batch)

    # Filter out none values(due to image loading errors)
    valid_data = [
        (img, cap)
        for img, cap in zip(images, captions)
        if img is not None and cap is not None
    ]
    if not valid_data:
        return None, None

    images, captions = zip(*valid_data)

    # pad captions to maximum length in the batch
    padded_captions = torch.nn.utils.rnn.pad_sequence(
        captions[0], batch_first=True, padding_value=0
    )

    # stack images
    images = torch.stack(images, dim=0)

    return images, padded_captions


def get_data_loaders(mode, vocab, batch_size, transform):
    """Returns the data loader for the specific mode"""
    image_dir = Config.IMAGE_DIR
    caption_file = Config.CAPTIONS_FILE

    dataset = FlickerDataset(image_dir, caption_file, vocab, transform, mode)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        collate_fn=pad_sequences,
    )

    return data_loader


def get_datasets(vocab):
    """Returns train, test and val datasets"""

    image_dir = Config.IMAGE_DIR
    caption_file = Config.CAPTIONS_FILE
    transform = transforms.Compose(
        [
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = FlickerDataset(image_dir, caption_file, vocab, transform, "train")
    val_dataset = FlickerDataset(image_dir, caption_file, vocab, transform, "val")
    test_dataset = FlickerDataset(image_dir, caption_file, vocab, transform, "test")

    return train_dataset, val_dataset, test_dataset


def preprocess_data():
    """Preprocesses Flicker30k dataset"""

    logger.info("Preprocessing data...")

    # load captions
    captions_file = Config.CAPTIONS_FILE
    df = pd.read_csv(captions_file, sep="|", engine="python")
    df.columns = ["image", "comment_number", "caption"]
    df["caption"] = df["caption"].str.strip()
    df = df.drop("comment_number", axis=1)
    df = df.groupby("image")["caption"].apply(list).reset_index(name="captions")
    all_captions = [
        caption for sublist in df["captions"].tolist() for caption in sublist
    ]

    # build vocabulary
    vocab = Vocabulary()
    vocab.build_vocabulary(all_captions)

    # save vocabulary
    vocab_path = os.path.join(Config.PREPROCESSED_DATA_DIR, "vocab.pkl")
    save_object(vocab, vocab_path)
    logger.info(f"Vocabulary saved to {vocab_path}")

    # get datasets
    train_dataset, val_dataset, test_dataset = get_datasets(vocab)

    # save datasets
    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    for mode, dataset in datasets.items():
        dataset_path = os.path.join(Config.PREPROCESSED_DATA_DIR, f"{mode}_dataset.pkl")
        save_object(dataset, dataset_path)
        logger.info(f"{mode.capitalize()} dataset saved to {dataset_path}")

    logger.info("Data preprocessing complete")

    return train_dataset, val_dataset, test_dataset, vocab
