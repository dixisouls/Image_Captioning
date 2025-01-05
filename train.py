import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from config import Config
from data import get_data_loader
from model import EncoderCNN, DecoderRNN
from tqdm import tqdm
from utils import setup_logger
import os
import warnings

warnings.filterwarnings("ignore")


def train():
    """
    Train the image captioning model.

    This function sets up the configuration, logger, data loader, and models.
    It then trains the encoder and decoder models using the specified number of epochs.

    The training process includes:
    - Loading the data
    - Initializing the encoder and decoder models
    - Defining the loss function and optimizer
    - Training the models with forward and backward passes
    - Logging the training progress
    - Saving the models after each epoch
    """

    os.remove("logs/training.log")

    config = Config()
    logger = setup_logger(config.LOG_FILE)

    # Load data
    logger.info("Loading data...")
    data_loader, dataset = get_data_loader(config)
    vocab = dataset.vocab
    logger.info("Data Loaded")
    logger.info(f"Vocabulary size: {len(vocab)}")

    # Initialize the encoder and decoder
    logger.info("Initializing models...")
    encoder = EncoderCNN(config.EMBEDDING_SIZE).to(config.DEVICE)
    decoder = DecoderRNN(
        config.EMBEDDING_SIZE, config.HIDDEN_SIZE, len(vocab), config.NUM_LAYERS
    ).to(config.DEVICE)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    params = (
        list(decoder.parameters())
        + list(encoder.linear.parameters())
        + list(encoder.bn.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=config.LEARNING_RATE)

    logger.info("Training models...")
    # Train the models
    total_step = len(data_loader)
    for epoch in range(config.NUM_EPOCHS):
        epoch_loss = 0
        progress = tqdm(
            enumerate(data_loader),
            desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}",
            total=total_step,
        )
        for i, (images, captions, lengths) in progress:
            images = images.to(config.DEVICE)
            captions = captions.to(config.DEVICE)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward pass
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            progress.set_postfix(loss=loss.item(), perplexity=torch.exp(loss).item())

        avg_loss = epoch_loss / total_step
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        logger.info(
            f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}] completed. "
            f"Average Loss: {avg_loss:.4f}, Average Perplexity: {avg_perplexity:.4f}"
        )

        # Save the models
        torch.save(
            decoder.state_dict(), f"{config.CHECKPOINT_DIR}/decoder-{epoch+1}.ckpt"
        )
        torch.save(
            encoder.state_dict(), f"{config.CHECKPOINT_DIR}/encoder-{epoch+1}.ckpt"
        )

    logger.info("Training complete.")


if __name__ == "__main__":
    train()
