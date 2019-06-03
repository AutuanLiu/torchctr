from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from .utils import DataMeta


def sparse_feature_encoding(data: pd.DataFrame, features_names: Union[str, List[str]]):
    """Encoding for sparse features."""

    for feat in features_names:
        data[feat] = LabelEncoder().fit_transform(data[feat])
    data_meta = DataMeta(data[features_names].values, data[features_names].shape, features_names)
    return data_meta


def sequence_feature_encoding(data: pd.DataFrame, features_names: Union[str, List[str]], sep: str = ','):
    """Encoding for sequence features."""

    data_value, padding_offsets = [], []
    for feature in features_names:
        vocab = set.union(*[set(str(x).strip().split(sep=sep)) for x in data[feature]])
        vec = CountVectorizer(vocabulary=vocab)
        # multi_hot = vec.transform(data[feature])
        # to index
        ret, offsets, offset = [], [], 0
        for row in data[feature]:
            offsets.append(offset)
            row = row.split(sep) if isinstance(row, str) else str(row).split(sep)
            ret.extend(map(lambda word: vec.vocabulary_[word], row))
            offset += len(row)
        data_value.append(np.array(ret))
        padding_offsets.append(np.array(offsets))
    padding_offsets = np.array(padding_offsets).T
    data_meta = DataMeta(np.array(data_value), padding_offsets.shape, features_names, padding_offsets)
    return data_meta


def dense_feature_scale(data: pd.DataFrame, features_names: Union[str, List[str]], scaler_instance=None):
    """Scaling for sparse features."""

    scaler = scaler_instance if scaler_instance else StandardScaler()
    data[features_names] = scaler.fit_transform(data[features_names])
    data_meta = DataMeta(data[features_names].values, data[features_names].shape, features_names)
    return data_meta
