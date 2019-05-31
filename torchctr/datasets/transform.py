from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def sparse_feature_encoding(data: pd.DataFrame, features_names: Union[str, List[str]]):
    """Encoding for sparse features in place."""

    for feat in features_names:
        data[feat] = LabelEncoder().fit_transform(data[feat])
    return data[features_names].values


def sequence_feature_encoding(data: pd.DataFrame, features_names: Union[str, List[str]], sep: str =','):
    """Encoding for sequence features in place."""
    
    ret = {}
    for feature in features_names:
        vocab = set.union(*[set(str(x).strip().split(sep=sep)) for x in data[feature]])
        vec = CountVectorizer(vocabulary=vocab)
        multi_hot = vec.transform(data[feature])
        # to index
        ret[feature] = multi_hot.toarray() * np.arange(len(vocab))
    return ret


def dense_feature_scale(data: pd.DataFrame, features_names: Union[str, List[str]], scaler_instance=None):
    """Scaling for sparse features in place."""

    scaler = scaler_instance if scaler_instance else StandardScaler()
    data[features_names] = scaler.fit_transform(data[features_names])
    return data[features_names].values
