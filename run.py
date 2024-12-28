import argparse
from src.train import main as train
from src.evaluate import main as evaluate
from src.inference import main as inference
from src.utils import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Image Captioning")
    parser.add_argument(
        "mode",
        choices=["train", "evaluate", "inference"],
        help="Mode to run the script",
    )

    args = parser.parse_args()

    if args.mode == "train":
        logger.info("Starting training...")
        train()
    elif args.mode == "evaluate":
        logger.info("Starting evaluation...")
        evaluate()
    elif args.mode == "inference":
        logger.info("Starting inference...")
        inference()


if __name__ == "__main__":
    main()
