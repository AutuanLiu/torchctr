import os

import torch, gc
import numpy as np

from .utils import DataInput, DataMeta


class RecommendDataset:
    """only support for sparse, sequence and dense data"""

    def __init__(self, input, target):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.lens
