import torch
from torch import nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    def __init__(self, input_dim=-1):
        super().__init__()
        self.input_dim = input_dim
        assert input_dim > 0, "input_dim must be greater than 0."
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float32), requires_grad=True)
        self.weights = torch.nn.Parameter(torch.randn(input_dim, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weights)

    def forward_(self, batch_size, index, feats, values, bias, weight):
        # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
        index = index.view(-1)
        values = values.view(1, -1)
        srcs = weight.view(1, -1).mul(values).view(-1)
        output = torch.zeros(batch_size, dtype=torch.float32)
        output.scatter_add_(0, index, srcs)
        output = output + bias
        return torch.sigmoid(output)

    def forward(self, batch_size, index, feats, values):
        # index: sample id, feats: feature id, values: feature value
        # type: (int, Tensor, Tensor, Tensor) -> Tensor
        weight = F.embedding(feats, self.weights)
        bias = self.bias
        return self.forward_(batch_size, index, feats, values, bias, weight)
