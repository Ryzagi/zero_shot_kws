import torch
from torch import nn

from zerospot.models.model import BcResNetModel
from zerospot.models.text_encoder import TextEncoder


class ZeroShotModel(nn.Module):
    def __init__(self, num_embeddings, hidden_dim, num_layers):
        super(ZeroShotModel, self).__init__()
        self.cnn_model = BcResNetModel(n_class=None)
        self.text_encoder = TextEncoder(num_embeddings, hidden_dim, num_layers)
        self.criterion = nn.BCEWithLogitsLossLoss()

    def forward(self, text_indexes, text_lengths, spectrogram_features):
        encoder_output, hidden = self.text_encoder(text_indexes, text_lengths)
        cnn_model_output = self.cnn_model(spectrogram_features)
        encoder_output = torch.mean(encoder_output, dim=1)
        output = torch.bmm(encoder_output.unsqueeze(1), cnn_model_output.unsqueeze(2))
        return output.squeeze()


if __name__ == '__main__':
    model = ZeroShotModel(4, 32, 1)
    text_indexes = torch.LongTensor([[1, 2, 3], [1, 2, 0]])
    text_lengths = torch.LongTensor([3, 2])
    spectrogram_features = torch.randn(2, 1, 40, 128)
    pred = model(text_indexes, text_lengths, spectrogram_features)
