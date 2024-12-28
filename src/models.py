import torch
import torch.nn as nn
import torchvision.models as models
from config import Config
from utils import setup_logger

logger = setup_logger(__name__)


class EncoderCNN(nn.Module):
    """Encoder model used to train teh VGG19 model"""

    def __init__(self, embed_size):
        """
        :param embed_size: Size of the embedding vector
        """

        super(EncoderCNN, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)

        # replace the last fully connected layer to output embed size
        num_ftrs = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = nn.Linear(num_ftrs, embed_size)

        # freze all layers except the last one
        for param in self.vgg19.parameters():
            param.requires_grad = False
        for param in self.vgg19.classifier[6].parameters():
            param.requires_grad = True

    def forward(self, images):
        """Forward pass of the encoder"""
        features = self.vgg19(images)
        return features


class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism"""

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: Dimension of the encoder output
        :param decoder_dim: Dimension of the hidden state
        :param attention_dim: Dimension of the attention layer
        """

        super(BahdanauAttention, self).__init__()
        self.encoder_attn = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim)
        self.full_attn = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, hidden):
        """
        :param encoder_out: Encoder output
        :param hidden: Decoder hidden state
        """

        att1 = self.encoder_attn(encoder_out)
        att2 = self.decoder_attn(hidden)
        att = self.full_attn(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):
    """Decoder model with attention mechanism"""

    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        encoder_dim,
        vocab_size,
        dropout=0.5,
    ):
        """
        :param attention_dim: Dimension of the attention layer
        :param embed_dim: Dimension of the embedding layer
        :param decoder_dim: Dimension of the decoder hidden state
        :param encoder_dim: Dimension of the encoder output
        :param vocab_size: Size of the vocabulary
        :param dropout: Dropout probability
        """
        super(DecoderRNN, self).__init__()
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        :param encoder_out: output of the encoder
        :param encoded_captions: Encoded captions
        :param caption_lengths: Caption lengths
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(
            dim=0, descending=True
        )
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # embedding
        embeddings = self.embedding(encoded_captions)

        # initliaze LSTM state
        h, c = self.init_hidden_state(encoder_out)

        # we wont decode at the <end> position, since we have finished generating as soon as we generate <end>
        # so, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(
            encoder_out.device
        )
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(
            encoder_out.device
        )

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat(
                    [embeddings[:batch_size_t, t, :], attention_weighted_encoding],
                    dim=1,
                ),
                (h[:batch_size_t], c[:batch_size_t]),
            )
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


class CaptioningModel(nn.Module):
    """Combine encoder and decoder models"""

    def __init__(
        self,
        embed_size,
        attention_dim,
        decoder_dim,
        encoder_dim,
        vocab_size,
        dropout=0.5,
    ):
        """
        :param embed_size: Size of the embedding vector
        :param attention_dim: Dimension of the attention layer
        :param decoder_dim: Dimension of the hidden state
        :param encoder_dim: Dimension of the encoder output
        :param vocab_size: Size of the vocabulary
        :param dropout: Dropout probability
        """
        super(CaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(
            attention_dim, embed_size, decoder_dim, encoder_dim, vocab_size, dropout
        )

    def forward(self, images, captions, lengths):
        """Forward pass of the combined model"""
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs

    def generate_caption(self, image, vocab, max_length=50):
        """Generate a caption for the given image"""

        result_caption = []
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.fc(hiddens.squeeze(0))
                predicted = output.argmax(1)

                if vocab.itos[predicted.item()] == "<end>":
                    break

                result_caption.append(predicted.item())
                x = self.decoder.embedding(predicted).unsqueeze(0)

        caption = [vocab.itos[idx] for idx in result_caption]
        return caption
