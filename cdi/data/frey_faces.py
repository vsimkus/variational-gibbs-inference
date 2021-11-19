import os
import os.path

import numpy as np
import scipy.io as sio
import scipy.special as sspec
import torch.utils.data as data


class FreyFacesDataset(data.Dataset):
    """
    Dataset wrapper for Frey Faces dataset (available at
    https://cs.nyu.edu/~roweis/data/frey_rawface.mat).
    """
    dataset_filename = 'frey_rawface.mat'
    fa_dataset_filename = 'frey_generated.mat'
    fa_large_dataset_filename = 'frey_generated_large.mat'
    fcvae_dataset_filename = 'fcvae_frey_generated.mat'
    fcvae_large_dataset_filename = 'fcvae_frey_generated_large.mat'

    # Used to prevent inf and -inf values with logit transformation
    logit_margin = 0.15

    def __init__(self, root, preprocess=True, generated=None, filename=None,
                 truncate=False, test=False):
        """
        Args:
            root (string): root directory that contains the `frey_rawface.mat`
                dataset.
            preprocess (boolean): Whether to transform the data using a logit
                transform, or not.
            generated (str): Whether to load FA-Frey, FC-VAE, or original (None).
            truncate (boolean): Whether to truncate the Frey faces images to smaller dimensionality.
        """
        self.generated = generated
        self.root = os.path.expanduser(root)
        self.filename = filename

        if self.filename is None:
            if self.generated is None or self.generated =='original':
                self.filename = self.dataset_filename
            elif self.generated == 'FA':
                self.filename = self.fa_dataset_filename
            elif self.generated == 'FA-large':
                self.filename = self.fa_large_dataset_filename
            elif self.generated == 'FC-VAE':
                self.filename = self.fcvae_dataset_filename
            elif self.generated == 'FC-VAE-large':
                self.filename = self.fcvae_large_dataset_filename

            if test:
                self.filename = '{}_test{}'.format(*os.path.splitext(self.filename))
                print(f'Using test set. {self.filename}')
        else:
            print(f'Using set from {self.filename}')

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        # Load Frey Faces dataset
        if self.generated not in ('FA', 'FA-large', 'FC-VAE', 'FC-VAE-large'):
            self.data = sio.loadmat(os.path.join(self.root, self.filename))['ff'].T
            self.data = self.data.astype('float32') / 255.
        else:
            self.meta_data = sio.loadmat(os.path.join(self.root, self.filename))
            self.data = self.meta_data['ff'].T

        if truncate:
            # Make data 100-dimensional from 560-dims, and capture roughly the lips area
            self.data = self.data[:, 20*16:20*21]

        if preprocess:
            self.data = self.preprocess(self.data)

        self.index = np.arange(0, len(self.data), dtype=np.long)

    def preprocess(self, data):
        """
        Unconstrain data, such that Gaussian tails can correspond to valid values.
        """
        return sspec.logit(data*(1 - 2*self.logit_margin) + self.logit_margin)

    def postprocess(self, data):
        """
        Transform data back to [0, 1] range.
        """
        # expit == sigmoid
        return ((sspec.expit(data) - self.logit_margin) / (1. - 2*self.logit_margin))

    def _check_exists(self):
        """
        Checks if dataset file is present.
        """
        return os.path.exists(os.path.join(self.root, self.filename))

    def save_data(self, filepath):
        if self.generated in ('FA', 'FA-large'):
            # Prepare dict
            data = {
                "ff": self.data.T,
                "F": self.meta_data['F'],
                "Psi": self.meta_data['Psi'],
                "c": self.meta_data['c']
            }
        else:
            data = {
                "ff": (self.data.T * 255).astype('uint8')
            }

        # Save and return samples
        sio.savemat(file_name=filepath, mdict=data)

        return data

    def __getitem__(self, index):
        """
        Args:
            index (int): index of sample
        Returns:
            image (float): dataset sample of length 560
            index (int): the index of the sample returned, useful for
                mutability of the dataset
        """

        return self.data[index], self.index[index]

    def __setitem__(self, index, x):
        """
        Allows to mutate the dataset.
        Args:
            index (tuple): index(or slice) of sample and variables to set
                e.g. (10:12, 0:2) will set the samples 10 and 11 and their
                variables 0 and 1
            x (list/array): values to set in place specified by index
        """
        self.data[index] = x

    def __len__(self):
        return self.data.shape[0]
