import os
from pathlib import Path

from torchvision.datasets.utils import download_url, makedir_exist_ok

from .utils import extract_file


def get_movielens(root, version='ml_20m'):
    """Download the MovieLens data if it doesn't exist."""

    urls = {
        'ml-latest': 'http://files.grouplens.org/datasets/movielens/ml-latest.zip',
        'ml-100k': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
        'ml-1m': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'ml-10m': 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
        'ml-20m': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
    }

    assert version in urls.keys(), f"version must be one of {set(urls.keys())}"
    raw_folder = os.path.join(root, version, 'raw')
    processed_folder = os.path.join(root, version, 'processed')
    makedir_exist_ok(raw_folder)
    makedir_exist_ok(processed_folder)

    # download files and extract
    filename = urls[version].rpartition('/')[2]
    print('Downloading...')
    download_url(urls[version], root=raw_folder, filename=filename, md5=None)
    print('Extracting...')
    extract_file(os.path.join(raw_folder, filename), processed_folder)
    print('Done!')
    return Path(os.path.join(processed_folder, version))
