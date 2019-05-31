from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        # create user embeddings
        self.user_factors = nn.Embedding(n_users, n_factors, sparse=True)
        # create item embeddings
        self.item_factors = nn.Embedding(n_items, n_factors, sparse=True)

    def forward(self, user, item):
        # matrix multiplication
        return (self.user_factors(user) * self.item_factors(item)).sum(1)
