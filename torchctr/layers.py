import torch
from torch import nn
from .datasets import DataInput, defaults, dropout_mask, emb_sz_rule, totensor
from typing import Optional
import torch.nn.functional as F


class EmbeddingDropout(nn.Module):
    "Apply dropout with probabily `embed_p` to an embedding layer `emb`."

    def __init__(self, emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p

    def forward(self, words: torch.LongTensor, scale: Optional[float] = None) -> torch.Tensor:
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        if scale: masked_embed.mul_(scale)
        return F.embedding(words, masked_embed, self.emb.padding_idx, self.emb.max_norm, self.emb.norm_type,
                           self.emb.scale_grad_by_freq, self.emb.sparse)


class EmbeddingLayer(nn.Module):
    """Embedding layer: convert sparse data to dense data.
    
    Args:
        emb_szs (dict): {feature: embedding size}.
        emb_drop (float): drop out. only support for sparse data now.
        x (DataInput): instance of DataInput, which includes sparse, sequence, dense data.
    
    Returns:
        torch.Tensor: dense data.
    """

    def __init__(self, x, emb_szs_dict=None, emb_drop=0, mode='mean'):
        super().__init__()
        assert mode in ['sum', 'mean'], "mode must in {'sum', 'mean'}"
        layers = []
        self.mode = mode
        if x.sparse_data:
            nuniques = x.sparse_data.nunique
            if emb_szs_dict:
                emb_szs = [emb_szs_dict[f] for f in x.sparse_data.features]
            else:
                emb_szs = [emb_sz_rule(t) for t in nuniques]
            self.sparse_embeds = nn.ModuleList(
                [EmbeddingDropout(nn.Embedding(ni, nf), emb_drop) for ni, nf in zip(nuniques, emb_szs)])
            del nuniques, emb_szs
        if x.sequence_data:
            nuniques = x.sequence_data.nunique
            if emb_szs_dict:
                emb_szs = [emb_szs_dict[f] for f in x.sequence_data.features]
            else:
                emb_szs = [emb_sz_rule(t) for t in nuniques]
            self.sequence_embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in zip(nuniques, emb_szs)])
            del nuniques, emb_szs
        self.drop = emb_drop

    def forward(self, x):
        out = []
        if x.sparse_data:
            data = totensor(x.sparse_data.data).long()
            sparse_out = [e(data[:, i]) for i, e in enumerate(self.sparse_embeds)]
            sparse_out = torch.cat(sparse_out, 1)
            out.append(sparse_out)
        if x.sequence_data:
            nuniques = x.sequence_data.nunique
            data = totensor(x.sequence_data.data).float()
            data = data.split(nuniques, dim=1)

            sequence_out = [
                data[i] @ e.weight if self.mode == 'sum' else data[i] @ e.weight / data[i].sum(dim=1).view(-1, 1)
                for i, e in enumerate(self.sequence_embeds)
            ]
            sequence_out = torch.cat(sequence_out, 1)
            out.append(sequence_out)
        if x.dense_data:
            dense_data = totensor(x.dense_data.data).float()
            out.append(dense_data)
        return torch.cat(out, 1)
