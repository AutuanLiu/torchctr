from .utils import DataInput, DataMeta, totensor


class RecommendDataset:
    """only support for sparse, sequence and dense data"""

    def __init__(self, input, target):
        self.data = input
        self.sparse = totensor(input.sparse_data.data) if input.sparse_data else None
        self.sequence = totensor(input.sequence_data.data) if input.sequence_data else None
        self.dense = totensor(input.dense_data.data) if input.dense_data else None
        self.target = target
        self.lens = len(self.target)

    def __getitem__(self, index):
        sparse = DataMeta(self.sparse[index], self.sparse.features, self.sparse.nunique) if self.sparse else None
        dense = DataMeta(self.dense[index], self.dense.features, self.dense.nunique) if self.dense else None
        sequence = DataMeta(self.sequence[index], self.sequence.features,
                            self.sequence.nunique) if self.sequence else None
        return DataInput(sparse, dense, sequence), self.target[index]

    def __len__(self):
        return self.lens
