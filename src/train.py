import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import get_data_loaders, get_datasets, preprocess_data, Vocabulary
from models import CaptioningModel
from utils import setup_logger, save_object, load_object, DeviceDataLoader, to_device
from config import Config
from tqdm import tqdm
import os
import time

logger = setup_logger(__name__)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, vocab):
    """Trains the model for one epoch"""
    model.train()
    total_loss = 0.0
    start_Time = time.time()

    data_loader_iter = iter(train_loader)
    num_batches = len(train_loader)

    with tqdm(
        total=num_batches, desc=f"Epoch {epoch + 1}/{Config.EPOCHS}", unit="batch"
    ) as progress_bar:
        for batch_idx in range(num_batches):
            try:
                data = next(data_loader_iter)
            except StopIteration:
                data_loader_iter = iter(train_loader)
                data = next(data_loader_iter)

            if data == None or data[0] == None or data[1] == None:
                continue

            images, captions = data
            images = images.to(device)

            optimizer.zero_grad()

            for i in range(captions.shape[1]):
                caption = captions[:, i, :].to(device)
                caption_lengths = (
                    (caption != vocab.stoi["<pad>"]).sum(dim=1).unsqueeze(1).to(device)
                )

                # forward pass
                outputs, caps_sorted, decode_lengths, alphas, sort_ind = model(
                    images, caption, caption_lengths
                )

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack padded sequence
                outputs = nn.utils.rnn.pack_padded_sequence(
                    outputs, decode_lengths, batch_first=True
                ).data
                targets = nn.utils.rnn.pack_padded_sequence(
                    targets, decode_lengths, batch_first=True
                ).data

                # calculate loss
                loss = criterion(outputs, targets)

                # backpropagation
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            progress_bar.update(1)

    avg_loss = total_loss / num_batches
    elapsed_time = time.time() - start_Time

    logger.info(
        f"Epoch [{epoch + 1}/{Config.EPOCHS}]",
        f"Loss: {avg_loss:.4f}",
        f"Time: {elapsed_time:.2f}s",
    )

    return avg_loss


def validate_epoch(model, val_loader, criterion, device, epoch, vocab):
    """Validates the model for one epoch"""

    model.eval()
    total_loss = 0.0
    start_time = time.time()

    data_loader_iter = iter(val_loader)
    num_batches = len(val_loader)

    with torch.no_grad():
        with tqdm(
            total=num_batches,
            desc=f"Validation Epoch {epoch+1}/{Config.EPOCHS}",
            unit="batch",
        ) as progress_bar:
            for batch_idx in range(num_batches):
                try:
                    data = next(data_loader_iter)
                except StopIteration:
                    data_loader_iter = iter(val_loader)
                    data = next(data_loader_iter)

                if data == None or data[0] == None or data[1] == None:
                    continue

                images, captions = data
                images = images.to(device)

                for i in range(captions.shape[1]):
                    caption = captions[:, i, :].to(device)
                    caption_lengths = (
                        (caption != vocab.stoi["<pad>"])
                        .sum(dim=1)
                        .unsqueeze(1)
                        .to(device)
                    )

                    # forward pass
                    outputs, caps_sorted, decode_lengths, alphas, sort_ind = model(
                        images, caption, caption_lengths
                    )

                    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                    targets = caps_sorted[:, 1:]

                    # Remove timesteps that we didn't decode at, or are pads
                    # pack padded sequence
                    outputs = nn.utils.rnn.pack_padded_sequence(
                        outputs, decode_lengths, batch_first=True
                    ).data
                    targets = nn.utils.rnn.pack_padded_sequence(
                        targets, decode_lengths, batch_first=True
                    ).data

                    # calculate loss
                    loss = criterion(outputs, targets)

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
                progress_bar.update(1)

    avg_loss = total_loss / num_batches
    elapsed_time = time.time() - start_time

    logger.info(
        f"Validation Epoch [{epoch + 1}/{Config.EPOCHS}]",
        f"Loss: {avg_loss:.4f}",
        f"Time: {elapsed_time:.2f}s",
    )

    return avg_loss


def train(model, train_loader, val_loader, criterion, optimizer, device, vocab):
    """Trains the model"""
    best_val_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, vocab
        )
        val_loss = validate_epoch(model, val_loader, criterion, device, epoch, vocab)

        # save the model if validation loss is decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(
                Config.MODEL_DIR, f"captioning_model_epoch_{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "vocab": vocab,
                },
                model_path,
            )
            logger.info(f"Saved model checkpoint to {model_path}")


def main():
    """Main function to run the training process"""

    # create directories if they dont exist
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.PREPROCESSED_DATA_DIR, exist_ok=True)

    # check for cuda availability
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Training on CPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load or preprocess the data
    train_dataset, val_dataset, test_dataset, vocab = preprocess_data()

    # create data loaders
    train_loader = get_data_loaders(
        "train",
        vocab,
        Config.BATCH_SIZE,
        transforms.Compose(
            [
                transforms.Resize(Config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    )
    val_loader = get_data_loaders(
        "val",
        vocab,
        Config.BATCH_SIZE,
        transforms.Compose(
            [
                transforms.Resize(Config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    )

    # initialize the model
    model = CaptioningModel(
        embed_size=Config.EMBEDDING_DIM,
        attention_dim=Config.HIDDEN_DIM,
        decoder_dim=Config.HIDDEN_DIM,
        encoder_dim=Config.EMBEDDING_DIM,
        vocab_size=len(vocab),
        dropout=0.5,
    ).to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # train the model
    train(model, train_loader, val_loader, criterion, optimizer, device, vocab)


if __name__ == "__main__":
    main()
