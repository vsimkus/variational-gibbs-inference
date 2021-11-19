import torch
from torch.utils.data import Dataset


class BlockMissingnessData(Dataset):
    """
    Block-miss mask.
    """
    def __init__(self, dataset, fraction_incomplete, width, top_left, bottom_right, inverse=False, target_idx=0):
        super().__init__()
        self.dataset = dataset
        self.target_idx = target_idx
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.fraction_incomplete = fraction_incomplete

        assert top_left[0] <= bottom_right[0] and top_left[1] <= bottom_right[1]

        X = dataset[0]
        if isinstance(X, tuple):
            X = X[0]

        self.miss_mask = torch.ones_like(torch.tensor(X), dtype=torch.bool).reshape(-1, width)
        for i in range(top_left[1], bottom_right[1]):
            for j in range(top_left[0], bottom_right[0]):
                self.miss_mask[i, j] = 0

        self.miss_mask = self.miss_mask.reshape(1, -1)
        if inverse:
            self.miss_mask = ~self.miss_mask

        # Leave some complete so that we have some information to initialise the images from.
        self.which_comp = torch.bernoulli(torch.ones(len(dataset))*(1-fraction_incomplete)).bool()

    def __getitem__(self, idx):
        data = self.dataset[idx]
        which_comp = self.which_comp[idx]
        if isinstance(data, tuple):
            # Insert missingness mask after the target_idx tensor to which it corresponds
            miss_mask = self.miss_mask.repeat(len(data[self.target_idx]), 1)
            miss_mask[which_comp] = 1.
            data = (data[:self.target_idx+1]
                    + (miss_mask,)
                    + data[self.target_idx+1:])
        else:
            miss_mask = self.miss_mask.repeat(len(data), 1)
            miss_mask[which_comp] = 1.
            data = (data, miss_mask)
        return data

    def __setitem__(self, key, value):
        """
        Allows mutation of the underlying dataset.
        The missingness mask is immutable.
        """
        self.dataset[key] = value

    def __len__(self):
        return len(self.dataset)
