import tensorflow as tf
import os
import tarfile
import sys
from config import FLAGS
from six.moves import urllib

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def download_data():
    """download CIFAR-10 data"""
    dest_directory = FLAGS.data_dir  # /tmp/cifar10_data

    # if the path not exist, create the path
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    filename = DATA_URL.split('/')[-1]  # cifar-10-binary.tar.gz
    filepath = os.path.join(dest_directory, filename)  # /tmp/cifar10_data/cifar-10-binary.tar.gz
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

if __name__ == "__main__":
    download_data()

