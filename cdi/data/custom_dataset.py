
import numpy as np
import torch.utils.data as data


class CustomDataset(data.Dataset):
    def __init__(self, data):
        self.data = data
        self.index = np.arange(0, len(self.data), dtype=np.long)

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
