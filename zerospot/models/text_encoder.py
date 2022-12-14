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

    def forward(self, input, input_len):

        embedded = self.embedding(input)
        #embedded = embedded.transpose(1, 2)
        packed_embedded = pack_padded_sequence(embedded, input_len, batch_first=True)
        #print(packed_embedded)
        packed_outputs, hidden = self.gru(packed_embedded)
        unpacked_seq, unpacked_lens = pad_packed_sequence(packed_outputs, batch_first=True)
        #outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        #hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return unpacked_seq, hidden

