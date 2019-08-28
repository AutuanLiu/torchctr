import torch
import torch.nn.functional as F

from torch import Tensor, nn
from typing import List


class DeepFM(nn.Module):
    def __init__(self, input_dim=-1, n_fields=-1, embedding_dim=-1, fc_dims=[]):
        super().__init__()
        self.input_dim = input_dim
        self.n_fields = n_fields
        self.embedding_dim = embedding_dim
        self.mats = []
        if input_dim > 0 and embedding_dim > 0 and n_fields > 0 and fc_dims:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1))
            self.weights = torch.nn.Parameter(torch.zeros(input_dim, 1))
            self.embedding = torch.nn.Parameter(torch.zeros(input_dim, embedding_dim))
            torch.nn.init.xavier_uniform_(self.weights)
            torch.nn.init.xavier_uniform_(self.embedding)
            dim = n_fields * embedding_dim    # DNN input dim
            # DNN FC layers
            for (index, fc_dim) in enumerate(fc_dims):
                self.mats.append(torch.nn.Parameter(torch.randn(dim, fc_dim)))    # weight
                self.mats.append(torch.nn.Parameter(torch.randn(1, 1)))    # bias
                torch.nn.init.xavier_uniform_(self.mats[index * 2])
                dim = fc_dim

    def first_order(self, batch_size, index, values, bias, weights):
        # type: (int, Tensor, Tensor, Tensor, Tensor) -> Tensor
        srcs = weights.view(1, -1).mul(values.view(1, -1)).view(-1)
        output = torch.zeros(batch_size, dtype=torch.float32)
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

    def higher_order(self, batch_size, embeddings, mats):
        # type: (int, Tensor, List[Tensor]) -> Tensor
        # activate function: relu
        output = embeddings.view(batch_size, -1)

        for i in range(int(len(mats) / 2)):
            output = torch.relu(output.matmul(mats[i * 2]) + mats[i * 2 + 1])

        return output.view(-1)

    def forward_(self, batch_size, index, feats, values, bias, weights, embeddings, mats):
        # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[Tensor]) -> Tensor

        first = self.first_order(batch_size, index, values, bias, weights)
        second = self.second_order(batch_size, index, values, embeddings)
        higher = self.higher_order(batch_size, embeddings, mats)

        return torch.sigmoid(first + second + higher)

    def forward(self, batch_size, index, feats, values):
        # type: (int, Tensor, Tensor, Tensor) -> Tensor
        batch_first = F.embedding(feats, self.weights)
        batch_second = F.embedding(feats, self.embedding)
        return self.forward_(batch_size, index, feats, values, self.bias, batch_first, batch_second, self.mats)
