import os
import random
import sys
import numpy as np
import scipy.io as sio
from scipy import sparse


def load_data(config):
    data_name = config['dataset']
    X_list = []
    Y_list = []
    main_dir = sys.path[0]
    file_path = os.path.join(main_dir, 'data', data_name + '.mat')

    if data_name in ['Scene-15']:
        mat = sio.loadmat(file_path)
        X = mat['X'][0]
        for view in [0, 1, 2]:
            X_list.append(X[view].astype('float32'))
            Y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['LandUse-21']:
        mat = sio.loadmat(file_path)
        train_x = [sparse.csr_matrix(mat['X'][0, 0]).A,
                   sparse.csr_matrix(mat['X'][0, 1]).A,
                   sparse.csr_matrix(mat['X'][0, 2]).A]
        index = random.sample(range(train_x[0].shape[0]), 2100)  # 30000
        for view in [0, 1, 2]:
            x = train_x[view][index]
            y = np.squeeze(mat['Y']).astype('int')[index]
            X_list.append(x)
            Y_list.append(y)

    elif data_name in ['handwritten']:
        mat = sio.loadmat(file_path)
        X = mat['X'][0]
        for view in [0, 1, 2, 3, 4, 5]:
            X_list.append(X[view].astype('float32'))
            Y_list.append(np.squeeze(mat['truth']))

    elif data_name in ['MSRC_v1']:
        mat = sio.loadmat(file_path)
        for view in ['msr1', 'msr2', 'msr3', 'msr4', 'msr5']:
            X_list.append(mat[view])
            Y_list.append(np.squeeze(mat['truth']))

    elif data_name in ['ORL_mtv']:
        mat = sio.loadmat(file_path)
        X = mat['X'][0]
        for view in [0, 1, 2]:
            X_list.append(X[view].astype('float32').T)
            Y_list.append(np.squeeze(mat['gt']))

    elif data_name in ['NoisyMNIST']:
        data = sio.loadmat(file_path)
        # train = DataSet_NoisyMNIST(data['X1'], data['X2'], data['trainLabel'])
        tune = DataSet_NoisyMNIST(data['XV1'], data['XV2'], data['tuneLabel'])
        test = DataSet_NoisyMNIST(data['XTe1'], data['XTe2'], data['testLabel'])
        X_list.append(np.concatenate([tune.images1, test.images1], axis=0))
        X_list.append(np.concatenate([tune.images2, test.images2], axis=0))
        Y_list.append(np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))
        Y_list.append(np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))

    else:
        raise Exception('Undefined data_name in Datasets')

    X_list = [X_list[config['view'][0]], X_list[config['view'][1]]]
    Y_list = [Y_list[config['view'][0]], Y_list[config['view'][1]]]
    return X_list, Y_list


class DataSet_NoisyMNIST(object):

    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False, dtype=np.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into `[0, 1]`.
        """
        t = 2
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                    'images1.shape: %s labels.shape: %s' % (images1.shape, labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                    'images2.shape: %s labels.shape: %s' % (images2.shape, labels.shape))
            self._num_examples = images1.shape[0] // t

            if dtype == np.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                # print("type conversion view 1")
                images1 = images1.astype(np.float32)

            if dtype == np.float32 and images2.dtype != np.float32:
                # print("type conversion view 2")
                images2 = images2.astype(np.float32)

        self._images1 = images1[::t]
        self._images2 = images2[::t]
        self._labels = labels[::t]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_image for _ in range(batch_size)], [fake_label for _
                                                                                                      in range(
                    batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]


def normalize(x):
    """ Normalize """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
