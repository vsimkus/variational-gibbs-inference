import numpy as np
from torch.utils.data import Sampler


class RegressionSampler(Sampler):
    r"""Sampler for training regression functions. Omits samples that are
    missing(/or observed) at the target dimension.
    Arguments:
        data_source (Dataset): dataset to sample indices for.
        dim (int): dimension of data_source samples that is treated as the
            targets, and which are filtered to only observed samples.
        shuffle (bool, default=False): Shuffle indices or not.
        omit_missing (bool, default=True): If true omits missing, otherwise
            observed.
    """

    def __init__(self, data_source, dim, shuffle=False, omit_missing=True):
        self.data_source = data_source
        self.dim = dim
        self.shuffle = shuffle
        self.omit_missing = omit_missing

        # Create a vector of observed/missing sample indices
        observed = np.zeros(len(data_source), dtype=np.bool_)
        observed = data_source[:][1][:, dim]
        self.observed = np.argwhere(observed == 1).flatten()
        self.missing = np.argwhere(observed == 0).flatten()

        # Count of observed samples
        if self.omit_missing:
            self.length = self.observed.size
        else:
            self.length = self.missing.size

    def __iter__(self):
        if self.omit_missing:
            indices = np.copy(self.observed)
        else:
            indices = np.copy(self.missing)

        if self.shuffle:
            np.random.shuffle(indices)

        # Iterate through (randomised) indices
        for i in np.nditer(indices):
            yield i

    def __len__(self):
        return self.length
