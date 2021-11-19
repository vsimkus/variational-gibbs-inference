
import numpy as np
import torch
from torch.utils.data import Dataset


class FullyMissingDataFilter(Dataset):
    """
    Removes samples that are fully missing.
    """
    def __init__(self, dataset):
        self.dataset = dataset

        X, M, _ = self.dataset[:]
        if isinstance(M, np.ndarray):
            self.not_fully_missing_idx = \
                np.argwhere(np.sum(M, axis=1) != 0).squeeze()
        elif isinstance(M, torch.Tensor):
            self.not_fully_missing_idx = \
                torch.nonzero(torch.sum(M, dim=1) != 0, as_tuple=False).squeeze()
        else:
            raise TypeError('Wrong imput type!')

    def __getitem__(self, idx):
        dataset_idx = self.not_fully_missing_idx[idx]

        return self.dataset[dataset_idx]

    def __setitem__(self, key, value):
        """
        Allows mutation of the underlying dataset.
        """
        self.dataset[key] = value

    def __len__(self):
        return len(self.not_fully_missing_idx)
