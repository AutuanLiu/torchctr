import os

from torchvision.datasets.utils import download_url, makedir_exist_ok

from .utils import extract_file


def get_criteo(root):
    """Download the Criteo data if it doesn't exist."""

    url = 'https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz'

    raw_folder = os.path.join(root, 'criteo', 'raw')
    processed_folder = os.path.join(root, 'criteo', 'processed')
    makedir_exist_ok(raw_folder)
    makedir_exist_ok(processed_folder)

    # download files and extract
    filename = url.rpartition('/')[2]
    print('Downloading...')
    download_url(url, root=raw_folder, filename=filename, md5=None)
    print('Extracting...')
    extract_file(os.path.join(raw_folder, filename), processed_folder)
    print('Done!')
