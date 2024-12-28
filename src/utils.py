import logging
import pickle
import torch
from config import Config


def setup_logger(name):
    """Setup logger"""

    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(Config.LOG_FILE)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_object(obj, path):
    """Save object to an pickle file"""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_object(path):
    """Load object from a pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)


def to_device(data, device):
    """Move tensor(s) to device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
