import logging


def setup_logger(log_file):
    """
    Set up a logger that logs messages to both the console and a file.

    Args:
        log_file (str): The file path for the log file.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
