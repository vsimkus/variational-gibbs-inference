
import os
import os.path

import numpy as np
import scipy.io as sio
import torch.utils.data as data


class ToyDataset(data.Dataset):
    """
    A toy dataset wrapper for data generated from a Factor Analysis model with
    the parameters declared below.
    Dataset observables are 6-dimensional generated from 2-dimensional latents.
    """
    # Parameters of the FA model that generated this dataset
    params = {
        1: {
            'c': np.array([3., -1., 0., 2., -1., 0]),
            'F': np.array([[-5.,  -2.],
                           [0.,  0.],
                           [3., -1.],
                           [0.,  0.],
                           [0.,  0.],
                           [0.,  0.]]),
            'Psi': np.array([1.4794, 2.0988, 1.7666, 1.3357, 0.9839, 1.1122]),
            'dataset_filename': 'toy_set.mat'
        },
        2: {
            'c': np.array([3., -1., 0., 2., -1., 0]),
            'F': np.array([[-5.,  -2.],
                           [4.,  0.],
                           [3., -1.],
                           [-3., -3.],
                           [1.,  5.],
                           [-1.,  2.]]),
            'Psi': np.array([1.4794, 2.0988, 1.7666, 1.3357, 0.9839, 1.1122]),
            'dataset_filename': 'toy_set2.mat'
        },
        3: {
            'c': np.array([3., -1., 0., 2., -1., 0]),
            'F': np.array([[-5.,  -2.],
                           [4.,  0.],
                           [3., -1.],
                           [-3., -3.],
                           [1.,  5.],
                           [-1.,  2.]]),
            'Psi': np.array([50.4794, 30.0988, 6.7666, 17.3357, 40.9839,
                             25.1122]),
            'dataset_filename': 'toy_set3.mat',
            'test_dataset_filename': 'toy_set3_test.mat'
        }
    }

    def __init__(self, root, version=1, filename=None, test=False):
        """
        Args:
            root (string): root directory that contains the `toy_set.mat`
                dataset.
        """
        self.root = os.path.expanduser(root)
        self.version = version
        self.filename = filename

        self.meta_data = ToyDataset.params[self.version]

        # Filename
        if filename is not None:
            self.filename = filename
        elif test:
            self.filename = self.meta_data['test_dataset_filename']
            print(f'Using test set. {self.filename}')
        else:
            self.filename = self.meta_data['dataset_filename']
        self.filename = os.path.join(self.root, self.filename)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        # Load Toy dataset
        data_file = sio.loadmat(self.filename)
        self.data = data_file['dataset']
        self.data = self.data.astype(np.float32)
        self.index = np.arange(0, len(self.data), dtype=np.long)

    def save_data(self, filepath):
        # Prepare dict
        data = {
            "dataset": self.data,
            "F": self.meta_data['F'],
            "Psi": self.meta_data['Psi'],
            "c": self.meta_data['c'],
            "dataset_filename": self.meta_data['dataset_filename']
        }

        # Save and return samples
        sio.savemat(file_name=filepath, mdict=data)
        return data

    def _check_exists(self):
        """
        Checks if dataset file is present.
        """
        return self.filename

    def __getitem__(self, index):
        """
        Args:
            index (int): index of sample
        Returns:
            image (float): dataset sample of length 6
            index (int): the index of the sample returned, useful for
                mutability of the dataset
        """

        return self.data[index], self.index[index]

    def __setitem__(self, index, x):
        """
        Allows to mutate the dataset.
        Missingness mask is immutable.
        Args:
            index (tuple): index(or slice) of sample and variables to set
                e.g. (10:12, 0:2) will set the samples 10 and 11 and their
                variables 0 and 1
            x (list/array): values to set in place specified by index
        """
        self.data[index] = x

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def generate(num_samples, version):
        F = ToyDataset.params[version]['F']
        Psi = ToyDataset.params[version]['Psi']
        c = ToyDataset.params[version]['c']
        # Compute the covariance matrix of observables
        cov = np.matmul(F, F.T) + np.diag(Psi)

        # Generate samples
        return np.random.multivariate_normal(mean=c, cov=cov, size=num_samples)

    @staticmethod
    def generate_and_save(num_samples, location, version):
        # Generate samples
        data = {
            "dataset": ToyDataset.generate(num_samples, version=version),
            "F": ToyDataset.params[version]['F'],
            "Psi": ToyDataset.params[version]['Psi'],
            "c": ToyDataset.params[version]['c']
        }

        # Save and return samples
        sio.savemat(file_name=location, mdict=data)
        return data
