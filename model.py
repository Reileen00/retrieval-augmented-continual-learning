import torch
import torch.nn as nn

class ContinualTransformer(nn.Module):
    def __init__(self, vocab=1000, dim=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        enc = nn.TransformerEncoderLayer(dim, 4)
        self.tr = nn.TransformerEncoder(enc, 4)
        self.fc = nn.Linear(dim, vocab)

    def forward(self, x):
        x = self.emb(x)
        x = self.tr(x)
        return self.fc(x)
