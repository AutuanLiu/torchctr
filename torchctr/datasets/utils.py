import gzip
import os
import tarfile
import zipfile
from collections import namedtuple
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import random_split

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

# data meta and init
DataMeta = namedtuple('DataMeta', ['data', 'shape', 'features', 'bag_offsets'])
DataMeta.__new__.__defaults__ = (None, ) * len(DataMeta._fields)


def extract_file(from_path, to_path, remove_finished=False):
    """https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py"""

    if from_path.endswith(".zip"):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    elif from_path.endswith(".tar"):
        with tarfile.open(from_path, 'r:') as tar:
            tar.extractall(path=to_path)
    elif from_path.endswith(".tar.gz"):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif from_path.endswith(".gz") and not from_path.endswith(".tar.gz"):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    else:
        raise ValueError("Extraction of {from_path} not supported")

    if remove_finished:
        os.unlink(from_path)


def train_test_split(dataset, test_rate):
    """Split dataset into two subdataset(train/test)."""

    test_size = round(len(dataset) * test_rate)
    train_size = len(dataset) - test_size
    return random_split(dataset, [train_size, test_size])


def read_data(filename, **kwargs):
    """read data from files.

    Args:
        filename (str or Path): file name.
    """

    if not isinstance(filename, Path):
        filename = Path(filename)
    return pd.read_csv(filename, **kwargs)
