# Image Captioning with Deep Learning

An end-to-end **image captioning system** that utilizes a **Convolutional Neural Network (CNN)** and a **Recurrent Neural Network (RNN)** to generate textual descriptions for images. This project is built with **PyTorch** and employs **pretrained VGG19** for feature extraction and **LSTM-based RNN** for image caption generation.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The goal of this project is to build a neural network capable of describing images in natural language. The system trains on the **Flicker32k dataset** and uses the VGG19 model for image feature extraction and an LSTM for generating captions.

### Key Workflow:
1. **Image feature extraction** with a pretrained CNN (VGG19).
2. **Caption generation** using an LSTM-based RNN that learns from image features and preprocessed captions.
3. **Vocabulary generation** with threshold-based word filtering.

---

## Features

- **Pre-trained VGG19** for image feature extraction.
- **Custom LSTM decoder** to generate captions.
- **Build and store vocabulary** dynamically from dataset captions.
- Support for configurable **training parameters** like batch size, learning rate, and epochs.
- **Inference mode** to generate captions for unseen images.

---

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.11.0 or higher
- Other dependencies listed in `requirements.txt`

Clone the repository:

```bash
git clone https://github.com/dixisouls/Image_Captioning.git
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

To train the image captioning model, run:

```bash
python train.py
```

Training configuration is handled in the `config.py` file. After training:
- Model checkpoints (encoder and decoder) will be saved in the `checkpoints/` directory.
- Training logs will be stored in the `logs/training.log` file.

### Inference

To generate a caption for an image:

```bash
python inference.py <path_to_image>
```

Example:

```bash
python inference.py data/images/test_image.jpg
```

The generated caption will be printed in the console.

---

## Project Structure
```
project/
├── config.py           # Configuration for dataset and model parameters
├── train.py            # Training pipeline for the model
├── inference.py        # Inference script to generate captions
├── vocabulary.py       # Vocabulary builder and loader
├── data.py             # Custom dataset loader for Flicker32k
├── model.py            # Encoder and Decoder model definition
├── utils.py            # Logging utility
├── checkpoints/        # Directory for saving model checkpoints
├── logs/               # Directory for training logs
├── data/               # Directory for dataset and vocabulary
│   ├── vocab.pkl       # Saved vocabulary pickle file
│   └── results.csv     # Captions file
└── README.md           # Project README file
```


---

## Configuration

You can customize various parameters in `config.py`:

- **Dataset and paths**
  - `DATA_DIR`: Base data directory.
  - `IMAGE_DIR`: Directory for dataset images.
  - `CAPTIONS_FILE`: File path for captions.

- **Model parameters**
  - `EMBEDDING_SIZE`: Size of the embedding layer.
  - `HIDDEN_SIZE`: Size of the LSTM hidden layer.
  - `NUM_LAYERS`: Number of LSTM layers.

- **Training parameters**
  - `BATCH_SIZE`: Batch size for training.
  - `LEARNING_RATE`: Learning rate for optimization.
  - `NUM_EPOCHS`: Number of training epochs.

- **Vocabulary**
  - `VOCAB_THRESHOLD`: Minimum word frequency to include in the vocabulary.

---

## Results

After training the model for 10 epochs:
- **Generated captions** demonstrate the ability to produce meaningful descriptions for images.
- Checkpoints for encoder and decoder models are saved in the `checkpoints/` directory.

You can visualize results by running inference on test images.

---

