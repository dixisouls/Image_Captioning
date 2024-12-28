import torch
from torch.utils.data import DataLoader
from dataset import get_data_loaders, get_datasets, Vocabulary
from models import CaptioningModel
from config import Config
from utils import setup_logger, load_object
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from tqdm import tqdm
import os
from torchvision import transforms

logger = setup_logger(__name__)


def evaluate(model, test_loader, device, vocab):
    """Evaluate the model on the test set using the bleu score"""

    model.eval()
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(
            tqdm(test_loader, desc="Evaluating")
        ):
            images = images.to(device)

            # generate captions for the batch
            for i in range(images.shape[0]):
                img = images[i].unsqueeze(0)
                predicted_caption = model.generate_caption(
                    img, vocab, Config.MAX_CAPTION_LEN
                )
                all_predictions.append(predicted_caption)

                # get corresponding reference captions
                references = [vocab.numericalize(caption) for caption in captions[i]]
                references = [
                    [
                        vocab.itos[token]
                        for token in ref
                        if token
                        not in {
                            vocab.stoi["<pad>"],
                            vocab.stoi["<start>"],
                            vocab.stoi["<end>"],
                        }
                    ]
                    for ref in references
                ]
                all_references.append(references)

        # calculate bleu score
        bleu1 = corpus_bleu(all_references, all_predictions, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(all_references, all_predictions, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(
            all_references, all_predictions, weights=(0.33, 0.33, 0.33, 0)
        )
        bleu4 = corpus_bleu(
            all_references, all_predictions, weights=(0.25, 0.25, 0.25, 0.25)
        )

        logger.info(f"BLEU-1: {bleu1}")
        logger.info(f"BLEU-2: {bleu2}")
        logger.info(f"BLEU-3: {bleu3}")
        logger.info(f"BLEU-4: {bleu4}")


def main():
    """Main function for evaluation"""

    # create directories if they dont exist
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.PREPROCESSED_DATA_DIR, exist_ok=True)

    # check for cuda availability
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Evaluating on CPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the vocabulary
    vocab_path = os.path.join(Config.PREPROCESSED_DATA_DIR, "vocab.pkl")
    vocab = load_object(vocab_path)

    # load dataset
    _, _, test_dataset = get_datasets(vocab)

    # create data loader
    test_loader = get_data_loaders(
        "test",
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

    # load the trained model
    model_checkpoint = int(input("Enter the epoch number of the model to evaluate: "))
    model_path = os.path.join(
        Config.MODEL_DIR, f"captioning_model_epoch_{model_checkpoint}.pth"
    )
    checkpoint = torch.load(model_path, map_location=device)
    model = CaptioningModel(
        embed_size=Config.EMBEDDING_DIM,
        attention_dim=Config.HIDDEN_DIM,
        decoder_dim=Config.HIDDEN_DIM,
        encoder_dim=Config.EMBEDDING_DIM,
        vocab_size=len(vocab),
        dropout=0.5,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # evaluate the model
    evaluate(model, test_loader, device, vocab)


if __name__ == "__main__":
    main()
