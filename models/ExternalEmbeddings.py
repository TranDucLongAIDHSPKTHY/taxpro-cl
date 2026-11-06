"""User and item tables trained outside this repository, evaluated with its own protocol.

The tables are final representations, so scoring is a dot product with no propagation.
Scores pass through the sigmoid every model of this repository uses, so the rankings are
built exactly as theirs, including the ties of saturated scores.
"""
import torch
from torch import nn


class ExternalEmbeddings(nn.Module):
    def __init__(self, config, dataset, device):
        super().__init__()
        size = int(config["embedding_size"])
        self.user_embedding = nn.Embedding(dataset.num_users, size)
        self.item_embedding = nn.Embedding(dataset.num_items, size)
        self.to(device)

    def load_tables(self, users, items):
        with torch.no_grad():
            self.user_embedding.weight.copy_(users)
            self.item_embedding.weight.copy_(items)

    def get_rating_for_test(self, user):
        users = self.user_embedding(user.long())
        return torch.sigmoid(users @ self.item_embedding.weight.transpose(0, 1))


class Trainer:
    def __init__(self, args, config, dataset, device, logger):
        self.model = ExternalEmbeddings(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    def train(self):
        raise RuntimeError("External tables are trained by tools/experiments/run_comparators.py")
