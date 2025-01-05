import torch
from torch.utils.data import random_split
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from data import get_data_loader
from model import EncoderCNN, DecoderRNN
from config import Config
import warnings

warnings.filterwarnings("ignore")


def evaluate():
    """
    Evaluate the performance of the trained image captioning model using BLEU scores.

    This function loads the dataset, splits it into training and evaluation sets,
    loads the pre-trained models, and evaluates the models on the evaluation set.
    It calculates BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores for the generated captions
    compared to the ground truth captions.
    """
    # Configuration object
    config = Config()

    # Load the dataset and data loader
    data_loader, dataset = get_data_loader(config)

    # Split dataset: use 20% for evaluation
    num_eval = int(0.2 * len(dataset))
    num_train = len(dataset) - num_eval
    train_set, eval_set = random_split(dataset, [num_train, num_eval])

    eval_loader = torch.utils.data.DataLoader(
        eval_set, batch_size=1, shuffle=True, collate_fn=data_loader.collate_fn
    )

    # Load vocabulary from dataset
    vocab = dataset.vocab

    # Load pre-trained models
    encoder = EncoderCNN(config.EMBEDDING_SIZE).eval().to(config.DEVICE)
    decoder = DecoderRNN(
        config.EMBEDDING_SIZE, config.HIDDEN_SIZE, len(vocab), config.NUM_LAYERS
    ).to(config.DEVICE)

    encoder.load_state_dict(
        torch.load(
            f"{config.CHECKPOINT_DIR}/encoder-8.ckpt", map_location=config.DEVICE
        )
    )
    decoder.load_state_dict(
        torch.load(
            f"{config.CHECKPOINT_DIR}/decoder-8.ckpt", map_location=config.DEVICE
        )
    )

    # Evaluation loop
    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    total_samples = 0

    progress = tqdm(eval_loader, desc="Evaluating", total=num_eval)
    smoothing_function = SmoothingFunction().method1

    for images, captions, lengths in progress:
        images = images.to(config.DEVICE)

        # Generate caption using the trained model
        features = encoder(images)
        sampled_ids = decoder.sample(features)
        sampled_ids = sampled_ids[0].cpu().numpy()

        # Map generated word IDs to words
        generated_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            if word == "<end>":
                break
            generated_caption.append(word)

        # Map ground truth word IDs to words
        ground_truth_caption = []
        for word_id in captions[0].cpu().numpy():
            word = vocab.idx2word[word_id]
            if word == "<end>":
                break
            ground_truth_caption.append(word)

        # Calculate BLEU scores
        bleu1 += sentence_bleu(
            [ground_truth_caption],
            generated_caption,
            weights=(1, 0, 0, 0),
            smoothing_function=smoothing_function,
        )
        bleu2 += sentence_bleu(
            [ground_truth_caption],
            generated_caption,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothing_function,
        )
        bleu3 += sentence_bleu(
            [ground_truth_caption],
            generated_caption,
            weights=(0.33, 0.33, 0.33, 0),
            smoothing_function=smoothing_function,
        )
        bleu4 += sentence_bleu(
            [ground_truth_caption],
            generated_caption,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing_function,
        )

        total_samples += 1

        # Update progress bar
        progress.set_postfix(
            bleu1=bleu1 / total_samples,
            bleu2=bleu2 / total_samples,
            bleu3=bleu3 / total_samples,
            blue4=bleu4 / total_samples,
        )

    # Print final BLEU scores
    print(f"BLEU-1: {bleu1 / total_samples:.4f}")
    print(f"BLEU-2: {bleu2 / total_samples:.4f}")
    print(f"BLEU-3: {bleu3 / total_samples:.4f}")
    print(f"BLEU-4: {bleu4 / total_samples:.4f}")


if __name__ == "__main__":
    evaluate()
