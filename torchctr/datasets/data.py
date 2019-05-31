import os
import tarfile
import zipfile

import torch
import torch.utils.data as data


class RecommendDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self):
        return ""


def extract_zip(from_path, to_path, remove_finished=False):
    if from_path.endswith(".zip"):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    elif from_path.endswith(".tar.gz"):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.unlink(from_path)
