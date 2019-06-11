from .utils import DataInput, DataMeta, totensor


class RecommendDataset:
    """only support for sparse, sequence and dense data"""

    def __init__(self, input, target):
        self.data = input
        self.sparse = input.sparse_data if input.sparse_data else None
        self.sequence = input.sequence_data if input.sequence_data else None
        self.dense = input.dense_data if input.dense_data else None
        self.target = totensor(target)
        self.lens = len(self.target)

    def __getitem__(self, index):
        sparse = DataMeta(self.sparse.data[index], self.sparse.features, self.sparse.nunique) if self.sparse else None
        dense = DataMeta(self.dense.data[index], self.dense.features, self.dense.nunique) if self.dense else None
        sequence = DataMeta(self.sequence.data[index], self.sequence.features,
                            self.sequence.nunique) if self.sequence else None
        return DataInput(sparse, dense, sequence), self.target[index]

    def __len__(self):
        return self.lens
