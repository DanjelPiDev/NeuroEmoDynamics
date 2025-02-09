import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_input):
        embedded = self.embedding(text_input)
        outputs, (hidden, cell) = self.lstm(embedded)

        hidden = hidden[-1]
        modulation = self.fc(hidden)
        return modulation
