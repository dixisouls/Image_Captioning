import torch
from torchvision import transforms
from PIL import Image
from config import Config
from models import CaptioningModel
from utils import load_object, setup_logger
import os

logger = setup_logger(__name__)


def generate_caption_for_image(image_path, model, vocab, device):
    """Generate caption for a single image"""
    model.eval()

    # preprocess the image
    transform = transforms.Compose(
        [
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        logger.error(f"Image not found at {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error opening image: {e}")
        return None

    image = transform(image).unsqueeze(0).to(device)

    # generate caption
    with torch.no_grad():
        caption = model.generate_caption(image, vocab, Config.MAX_CAPTION_LEN)

    # convert tokens to words
    caption_words = [
        word for word in caption if word not in {"<start>", "<end>", "<pad>"}
    ]
    caption_text = " ".join(caption_words)


def main():
    """Main function to run the inference"""

    # check for cuda availability
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Inference will run on CPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the vocabulary
    vocab_path = os.path.join(Config.PREPROCESSED_DATA_DIR, "vocab.pkl")
    vocab = load_object(vocab_path)

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

    image_path = input("Enter the path to the image: ")

    # generate and print caption
    caption = generate_caption_for_image(image_path, model, vocab, device)
    if caption:
        print(f"Generated Caption: {caption}")


if __name__ == "__main__":
    main()
