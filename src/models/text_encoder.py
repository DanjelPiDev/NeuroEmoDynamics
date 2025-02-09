import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1,
                 bidirectional=True, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(lstm_out_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, text_input):
        embedded = self.embedding(text_input)
        outputs, (hidden, cell) = self.lstm(embedded)
        pooled = self.pool(outputs.transpose(1, 2)).squeeze(-1)
        modulation = self.fc(pooled)
        modulation = self.activation(modulation)
        return modulation
