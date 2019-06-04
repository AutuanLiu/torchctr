import os

import torch, gc
import numpy as np

from .utils import DataInput, DataMeta


class RecommendDataset:
    """only support for sparse, sequence and dense data"""

    def __init__(self, input, target):
        self.data = input
        self.sparse = input.sparse_data if input.sparse_data else None
        self.sequence = input.sequence_data if input.sequence_data else None
        self.dense = input.dense_data if input.dense_data else None
        self.target = target
        self.lens = len(self.target)
        if self.sequence:
            data, offsets = [], np.zeros((self.lens, len(self.sequence.bag_offsets)), dtype=int)
            for i in range(self.lens):
                tmp = []
                for x, y in zip(self.sequence.data, self.sequence.bag_offsets):
                    if i == self.lens - 1:
                        t = x[y[-1]:]
                        t = [t] if isinstance(t, int) else t
                        tmp.append(t)
                    else:
                        t = x[y[i]:y[i + 1]]
                        t = [t] if isinstance(t, int) else t
                        tmp.append(t)
                data.append(tmp)
            self.offsets = offsets
        gc.collect()

    def __getitem__(self, index):
        sparse, dense, sequence = None, None, None
        if self.sparse:
            sparse = DataMeta(self.sparse.data[index], None, self.sparse.features, self.sparse.nunique, None)
        if self.dense:
            dense = DataMeta(self.dense.data[index], None, self.dense.features, self.dense.nunique, None)
        if self.sequence:
            tmp = self.sequence_data[index]
            sequence = DataMeta(data, None, self.sequence.features, self.sequence.nunique,
                                self.offsets[index])
        data = sparse, dense, sequence
        gc.collect()
        return DataInput(*data), self.target[index]

    def __len__(self):
        return self.lens
