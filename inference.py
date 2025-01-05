import torch
from torchvision import transforms
from PIL import Image
from config import Config
from model import EncoderCNN, DecoderRNN
from vocabulary import get_vocab
import re
import argparse
import warnings

warnings.filterwarnings("ignore")


def load_image(image_path, transform=None):
    """
    Load an image and apply the specified transformations.

    Args:
        image_path (str): Path to the image file.
        transform (callable, optional): Optional transform to be applied on the image.

    Returns:
        torch.Tensor: Transformed image tensor.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224), Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def generate_caption(image_path):
    """
    Generate a caption for the given image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Generated caption for the image.
    """
    config = Config()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    vocab = get_vocab()

    encoder = EncoderCNN(config.EMBEDDING_SIZE).eval().to(config.DEVICE)
    decoder = DecoderRNN(
        config.EMBEDDING_SIZE, config.HIDDEN_SIZE, len(vocab), config.NUM_LAYERS
    ).to(config.DEVICE)
    encoder.load_state_dict(torch.load(f"{config.CHECKPOINT_DIR}/encoder-8.ckpt", map_location=config.DEVICE))
    decoder.load_state_dict(torch.load(f"{config.CHECKPOINT_DIR}/decoder-8.ckpt", map_location=config.DEVICE))

    # Prepare image
    image = load_image(image_path, transform)
    image_tensor = image.to(config.DEVICE)

    # Generate caption
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == "<end>":
            break
    sentence = " ".join(sampled_caption)

    return re.sub(r'<start>\s*|\s*<end>', '', sentence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for images.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()
    caption = generate_caption(args.image_path)
    print(caption)
