import os

import torch, gc

from .utils import DataInput, DataMeta


class RecommendDataset:
    """only support for sparse, sequence and dense data"""

    def __init__(self, input, target):
        self.data = input
        self.sparse = self.data.sparse_data if self.data.sparse_data else None
        self.sequence = self.data.sequence_data if self.data.sequence_data else None
        self.dense = self.data.dense_data if self.data.dense_data else None
        self.target = target
        self.lens = len(self.target)
        if self.sequence:
            data, offsets = [], [[0] * len(self.sequence.bag_offsets)]
            for i in range(self.lens):
                for x, y in zip(self.sequence.data, self.sequence.bag_offsets):
                    tmp = []
                    if i == self.lens - 1:
                        tmp.append(x[y[i:][0]])
                    else:
                        tmp.append(x[y[i:(i + 1)][0]])
                data.append(tmp)
            self.sequence_data = data
            self.offsets = offsets
        gc.collect()

    def __getitem__(self, index):
        sparse, dense, sequence = None, None, None
        if self.sparse:
            sparse = DataMeta(self.sparse.data[index], None, self.sparse.features, self.sparse.nunique, None)
        if self.dense:
            dense = DataMeta(self.dense.data[index], None, self.dense.features, self.dense.nunique, None)
        if self.sequence:
            sequence = DataMeta(self.sequence_data[index], None, self.sequence.features, self.sequence.nunique,
                                self.offsets)
        data = sparse, dense, sequence
        gc.collect()
        return DataInput(*data), self.target[index]

    def __len__(self):
        return self.lens
