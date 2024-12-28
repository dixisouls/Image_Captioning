import logging


class Config:
    """Configuration class for the application"""

    # Data
    DATA_DIR = "data/flicker30k"
    PREPROCESSED_DATA_DIR = "data/preprocessed"
    IMAGE_DIR = f"{DATA_DIR}/images"
    CAPTIONS_FILE = f"{DATA_DIR}/results.csv"

    # Model
    MODEL_DIR = "models"
    ENCODER_MODEL = "vgg19"
    DECODER_MODEL = "lstm"
    ATTENTION_TYPE = "bahdanau"

    # Training
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    TEACHER_FORCING_RATIO = 0.5

    # Voctabulary
    VOCABULARY_SIZE = 5000
    MIN_WORD_FREQ = 5

    # Embedding
    EMBEDDING_DIM = 512
    HIDDEN_DIM = 512

    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FILE = "logs/training.log"

    # Inference
    BEAM_SIZE = 5
    MAX_CAPTION_LEN = 50

    # Image preprocessing
    IMAGE_SIZE = (224, 224)

    # Train, Validation and Test split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
