from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from .utils import DataMeta, defaults


def sparse_feature_encoding(data: pd.DataFrame, features_names: Union[str, List[str]]):
    """Encoding for sparse features."""

    nuniques = []
    for feat in features_names:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        nuniques.append(len(lbe.classes_))
    data_meta = DataMeta(data[features_names].values, data[features_names].shape, features_names, nuniques)
    return data_meta


def sequence_feature_encoding(data: pd.DataFrame, features_names: Union[str, List[str]], sep: str = ','):
    """Encoding for sequence features."""

    data_value = bags_offsets = nuniques = []
    for feature in features_names:
        vocab = set.union(*[set(str(x).strip().split(sep=sep)) for x in data[feature]])
        vec = CountVectorizer(vocabulary=vocab)
        nuniques.append(len(vocab))
        # multi_hot = vec.transform(data[feature])
        # to index
        ret, offsets, offset = [], [], 0
        for row in data[feature]:
            offsets.append(offset)
            row = row.split(sep) if isinstance(row, str) else str(row).split(sep)
            ret.extend(map(lambda word: vec.vocabulary_[word], row))
            offset += len(row)
        data_value.append(ret)
        bags_offsets.append(offsets)
    data_meta = DataMeta(data_value, None, features_names, nuniques, bags_offsets)
    return data_meta


def dense_feature_scale(data: pd.DataFrame, features_names: Union[str, List[str]], scaler_instance=None):
    """Scaling for sparse features."""

    scaler = scaler_instance if scaler_instance else StandardScaler()
    scaler = scaler.fit(data[features_names])
    data[features_names] = scaler.transform(data[features_names])
    data_meta = DataMeta(data[features_names].values, data[features_names].shape, features_names)
    return data_meta, scaler


def fillna(data: pd.DataFrame, features_names: Union[str, List[str]], fill_v, **kwargs):
    """Fill Nan with fill_v."""

    data[features_names] = data[features_names].fillna(fill_v, **kwargs)
    return data
