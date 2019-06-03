import torch
from torch import nn
from .datasets import DataInput, defaults


class EmbeddingLayer(nn.Module):
    """Embedding layer: convert sparse data to dense data.
    
    Args:
        emb_szs (dict): {feature: embedding size}.
        emb_drop (float): drop out.
        x (DataInput): instance of DataInput, which includes sparse, sequence, dense data.
    
    Returns:
        torch.Tensor: dense data.
    """

    def __init__(self, x, emb_szs=None, emb_drop=None, mode='mean'):
        super().__init__()
        layers = []
        if x.sparse_data:
            if emb_szs:
                emb_szs = [emb_szs[f] for f in x.sparse_data.features]
            else:
                emb_szs = [self.emb_sz_rule(t) for t in x.sparse_data.nunique]
            self.sparse_embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in zip(x.sparse_data.nunique, emb_szs)])
        if x.sequence_data:
            if emb_szs:
                emb_szs = [emb_szs[f] for f in x.sequence_data.features]
            else:
                emb_szs = [self.emb_sz_rule(t) for t in x.sequence_data.nunique]
            self.sequence_embeds = nn.ModuleList(
                [nn.EmbeddingBag(ni, nf, mode=mode) for ni, nf in zip(x.sequence_data.nunique, emb_szs)])
        self.drop = nn.Dropout(emb_drop)

    def emb_sz_rule(self, nunique: int) -> int:
        return min(600, round(1.6 * nunique**0.56))

    def forward(self, x):
        out = []
        if x.sparse_data:
            data = torch.LongTensor(x.sparse_data.data, defaults.device)
            sparse_out = [e(data[:, i]) for i, e in enumerate(self.sparse_embeds)]
            sparse_out = torch.cat(sparse_out, 1)
            sparse_out = self.emb_drop(sparse_out)
            out.append(sparse_out)
        if x.sequence_data:
            data = [torch.LongTensor(t, defaults.device) for t in x.sequence_data.data]
            offset = [torch.LongTensor(t, defaults.device) for t in x.sequence_data.bag_offsets]
            sequence_out = [e(data[i], offset[i]) for i, e in enumerate(self.sequence_embeds)]
            sequence_out = torch.cat(sequence_out, 1)
            sequence_out = self.emb_drop(sequence_out)
            out.append(sequence_out)
        if x.dense_data:
            dense_data = torch.as_tensor(x.dense_data.data, defaults.device)
            out.append(dense_data)
        return torch.cat(out, 1)
