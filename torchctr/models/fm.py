import torch
from torch import nn
import torch.nn.functional as F


class FactorizationMachine(nn.Module):
    def __init__(self, input_dim=-1, embedding_dim=-1):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        if input_dim > 0 and embedding_dim > 0:
            self.bias = torch.randn(1, 1, dtype=torch.float32)
            self.weights = torch.randn(input_dim, 1)
            self.embedding = torch.randn(input_dim, embedding_dim)
            self.bias = nn.Parameter(self.bias, requires_grad=True)
            self.weights = nn.Parameter(self.weights, requires_grad=True)
            self.embedding = nn.Parameter(self.embedding, requires_grad=True)
            nn.init.xavier_uniform_(self.weights)
            nn.init.xavier_uniform_(self.embedding)

    def first_order(self, batch_size, index, values, bias, weights):
        # type: (int, Tensor, Tensor, Tensor, Tensor) -> Tensor
        size = batch_size
        srcs = weights.view(1, -1).mul(values.view(1, -1)).view(-1)
        output = torch.zeros(size, dtype=torch.float32)
        output.scatter_add_(0, index, srcs)
        first = output + bias
        return first

    def second_order(self, batch_size, index, values, embeddings):
        # type: (int, Tensor, Tensor, Tensor) -> Tensor
        k = embeddings.size(1)
        b = batch_size

        # t1: [k, n]
        t1 = embeddings.mul(values.view(-1, 1)).transpose_(0, 1)
        # t1: [k, b]
        t1_ = torch.zeros(k, b, dtype=torch.float32)

        for i in range(k):
            t1_[i].scatter_add_(0, index, t1[i])

        # t1: [k, b]
        t1 = t1_.pow(2)

        # t2: [k, n]
        t2 = embeddings.pow(2).mul(values.pow(2).view(-1, 1)).transpose_(0, 1)
        # t2: [k, b]
        t2_ = torch.zeros(k, b, dtype=torch.float32)
        for i in range(k):
            t2_[i].scatter_add_(0, index, t2[i])

        # t2: [k, b]
        t2 = t2_

        second = t1.sub(t2).transpose_(0, 1).sum(1).mul(0.5)
        return second

    def forward_(self, batch_size, index, feats, values, bias, weights, embeddings):
        # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
        first = self.first_order(batch_size, index, values, bias, weights)
        second = self.second_order(batch_size, index, values, embeddings)
        return torch.sigmoid(first + second)

    def forward(self, batch_size, index, feats, values):
        # type: (int, Tensor, Tensor, Tensor) -> Tensor
        batch_first = F.embedding(feats, self.weights)
        batch_second = F.embedding(feats, self.embedding)
        return self.forward_(batch_size, index, feats, values, self.bias, batch_first, batch_second)
