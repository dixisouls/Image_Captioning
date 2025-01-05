import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from config import Config
from vocabulary import get_vocab
import nltk


class Flicker32kDataset(Dataset):
    """
    A custom Dataset class for the Flicker32k dataset.

    Attributes:
        image_dir (str): Directory where the images are stored.
        vocab (Vocabulary): Vocabulary object for text processing.
        transform (callable, optional): Optional transform to be applied on a sample.
        img_captions (list): List of tuples containing image names and their corresponding captions.
    """

    def __init__(self, image_dir, caption_file, vocab, transform=None):
        """
        Initialize the Flicker32kDataset object.

        Args:
            image_dir (str): Directory where the images are stored.
            caption_file (str): Path to the file containing image captions.
            vocab (Vocabulary): Vocabulary object for text processing.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform

        captions_df = pd.read_csv(caption_file, delimiter="|")
        captions_df.columns = [col.strip() for col in captions_df.columns]
        captions_df["comment"] = captions_df["comment"].str.strip()

        self.img_captions = []
        for index, row in captions_df.iterrows():
            self.img_captions.append((row["image_name"], row["comment"]))

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.img_captions)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, target) where image is the transformed image and target is the tokenized caption.
        """
        img_name, caption = self.img_captions[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption_vec = []
        caption_vec.append(self.vocab("<start>"))
        caption_vec.extend([self.vocab(token) for token in tokens])
        caption_vec.append(self.vocab("<end>"))
        target = torch.LongTensor(caption_vec)

        return img, target


def collate_fn(data):
    """
    Custom collate function to merge a list of samples to form a mini-batch.

    Args:
        data (list): List of tuples (image, caption).

    Returns:
        tuple: (images, targets, lengths) where images is a tensor of images, targets is a tensor of tokenized captions, and lengths is a list of caption lengths.
    """
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_data_loader(config):
    """
    Get the data loader for the Flicker32k dataset.

    Args:
        config (Config): Configuration object containing various parameters.

    Returns:
        tuple: (data_loader, dataset) where data_loader is the DataLoader object and dataset is the Flicker32kDataset object.
    """
    vocab = get_vocab()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    dataset = Flicker32kDataset(
        config.IMAGE_DIR, config.CAPTIONS_FILE, vocab, transform
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return data_loader, dataset
