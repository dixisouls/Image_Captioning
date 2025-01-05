import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    Convolutional Neural Network (CNN) encoder that uses a pre-trained VGG19 model to extract image features
    and then projects them into a specified embedding size.

    Attributes:
        vgg19 (nn.Sequential): Pre-trained VGG19 model without the final classification layer.
        linear (nn.Linear): Linear layer to project the extracted features into the embedding size.
        bn (nn.BatchNorm1d): Batch normalization layer for the projected features.
    """

    def __init__(self, embed_size):
        """
        Initialize the EncoderCNN.

        Args:
            embed_size (int): The size of the embedding vector.
        """
        super(EncoderCNN, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        modules = list(vgg19.children())[:-1]
        self.vgg19 = nn.Sequential(*modules)
        self.linear = nn.Linear(vgg19.classifier[0].in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """
        Forward pass through the encoder.

        Args:
            images (torch.Tensor): Batch of input images.

        Returns:
            torch.Tensor: Batch of image features projected into the embedding size.
        """
        with torch.no_grad():
            features = self.vgg19(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    """
    Recurrent Neural Network (RNN) decoder that generates captions from image features using an LSTM.

    Attributes:
        embed (nn.Embedding): Embedding layer for the input captions.
        lstm (nn.LSTM): LSTM layer for sequence modeling.
        linear (nn.Linear): Linear layer to project LSTM outputs to vocabulary size.
    """

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """
        Initialize the DecoderRNN.

        Args:
            embed_size (int): The size of the embedding vector.
            hidden_size (int): The size of the hidden state in the LSTM.
            vocab_size (int): The size of the vocabulary.
            num_layers (int): The number of LSTM layers.
        """
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        """
        Forward pass through the decoder.

        Args:
            features (torch.Tensor): Batch of image features from the encoder.
            captions (torch.Tensor): Batch of input captions.
            lengths (list[int]): List of lengths of each caption in the batch.

        Returns:
            torch.Tensor: Batch of predicted word scores for each time step.
        """
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True
        )
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None, max_len=20):
        """
        Generate captions for given image features using greedy search.

        Args:
            features (torch.Tensor): Batch of image features from the encoder.
            states (tuple, optional): Initial states for the LSTM.
            max_len (int, optional): Maximum length of the generated captions.

        Returns:
            torch.Tensor: Batch of generated caption word indices.
        """
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids
