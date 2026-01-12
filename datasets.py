import torch
import random

def generate_task(vocab_size=1000, seq_len=10, samples=1000):
    X = torch.randint(0, vocab_size, (samples, seq_len))
    Y = X.clone()   # identity task
    return list(zip(X,Y))

def get_tasks(n=5):
    return [generate_task() for _ in range(n)]
