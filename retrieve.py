import torch

def embed(model, x):
    with torch.no_grad():
        return model.emb(x).mean(1).cpu().numpy()
