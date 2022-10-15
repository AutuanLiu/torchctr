import gzip
import os
import tarfile
import zipfile
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache

import pandas as pd
import torch
from torch.utils.data import random_split


def num_cpus() -> int:
    "Get number of cpus"

    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


# simple name space
defaults = SimpleNamespace(cpus=min(16, num_cpus()),
                           device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


def extract_file(from_path, to_path, remove_finished=False):
    """https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py"""

    if from_path.endswith(".zip"):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    elif from_path.endswith(".tar"):
        with tarfile.open(from_path, 'r:') as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=to_path)
    elif from_path.endswith(".tar.gz"):
        with tarfile.open(from_path, 'r:gz') as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=to_path)
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
    return pd.read_csv(filename, engine='python', **kwargs)


def emb_sz_rule(dim: int) -> int:
    return min(600, round(1.6 * dim**0.56))


def totensor(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, device=defaults.device)


def dropout_mask(x: torch.Tensor, sz: Collection[int], p: float):
    "Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element."

    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)
