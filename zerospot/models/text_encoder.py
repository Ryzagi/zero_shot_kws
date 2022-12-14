from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextEncoder(nn.Module):
    def __init__(self, num_embeddings, hidden_dim, num_layers):
        super(TextEncoder, self).__init__()

        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings, self.hidden_dim )
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers)

    def forward(self, input, input_len):
        embedded = self.embedding(input)
        packed_embedded = pack_padded_sequence(embedded, input_len, batch_first=True)
        packed_outputs, hidden = self.gru(packed_embedded)
        return packed_outputs, hidden

