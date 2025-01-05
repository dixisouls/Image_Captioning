import torch


class Config:
    """
    Configuration class for setting up various parameters for the project.

    Attributes:
        DATA_DIR (str): Directory where the data is stored.
        IMAGE_DIR (str): Directory where the images are stored.
        CAPTIONS_FILE (str): File path for the captions file.
        DEVICE (torch.device): Device to be used for computation (CPU or CUDA).
        EMBEDDING_SIZE (int): Size of the embedding layer.
        HIDDEN_SIZE (int): Size of the hidden layer in the model.
        NUM_LAYERS (int): Number of layers in the model.
        BATCH_SIZE (int): Batch size for training.
        LEARNING_RATE (float): Learning rate for the optimizer.
        NUM_EPOCHS (int): Number of epochs for training.
        VOCAB_THRESHOLD (int): Minimum frequency of words to be included in the vocabulary.
        LOG_FILE (str): File path for the log file.
    """

    DATA_DIR = "data/"
    IMAGE_DIR = "data/images/"
    CAPTIONS_FILE = "data/results.csv"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model parameters
    EMBEDDING_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 1
    CHECKPOINT_DIR = "checkpoints/"

    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10

    # Vocabulary parameters
    VOCAB_THRESHOLD = 5

    # Logging parameters
    LOG_DIR = "logs/"
    LOG_FILE = f"{LOG_DIR}/training.log"
