import faiss
import torch
import numpy as np

class Memory:
    def __init__(self, dim=256):
        self.index = faiss.IndexFlatL2(dim)
        self.x, self.y = [], []

    def add(self, emb, x, y):
        self.index.add(emb)
        self.x.append(x)
        self.y.append(y)

    def sample(self, emb, k=32):
        if len(self.x) == 0:
            return [], []
        D,I = self.index.search(emb, k)
        return [self.x[i] for i in I[0]], [self.y[i] for i in I[0]]
