from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch

class TextEncoder(nn.Module):
    def __init__(self, num_embeddings, hidden_dim, num_layers):
        super(TextEncoder, self).__init__()

        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings, self.hidden_dim )
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers)

    def forward(self, input_tokens, input_len):
        embedded = self.embedding(input_tokens)
        packed_embedded = pack_padded_sequence(embedded, input_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.gru(packed_embedded)
        unpacked_seq, unpacked_lens = pad_packed_sequence(packed_outputs, batch_first=True)
        return unpacked_seq, hidden

