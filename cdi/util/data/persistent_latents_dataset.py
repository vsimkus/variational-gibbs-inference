import numpy as np
from torch.utils.data import Dataset


class LatentsDataset(Dataset):
    """
    Add another variable for the latents.
    Initialises the latents to standard normal
    """
    def __init__(self, dataset, latent_dim, target_idx=0, index_idx=2, miss_idx=1):
        super().__init__()
        self.dataset = dataset
        self.index_idx = index_idx
        self.miss_idx = miss_idx
        self.target_idx = target_idx
        self.latent_dim = latent_dim

        I = dataset.get_all_data()[self.index_idx]
        self.latents = np.random.randn(len(I), latent_dim).astype(np.float32)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        latents = self.latents[data[self.index_idx]]

        X = np.concatenate([data[self.target_idx], latents], axis=-1)
        zeros = np.zeros((X.shape[0], self.latent_dim), dtype=data[self.miss_idx].dtype)
        M = np.concatenate([data[self.miss_idx], zeros], axis=-1)

        # Insert the latents in target_idx
        data = data[:self.target_idx] + (X, ) + data[self.target_idx+1:]
        data = data[:self.miss_idx] + (M, ) + data[self.miss_idx+1:]

        return data

    def __setitem__(self, key, value):
        """
        Allows mutation of the underlying dataset.
        """
        X, Z = value[:, :value.shape[-1]-self.latent_dim], value[:, value.shape[-1]-self.latent_dim:]
        self.dataset[key] = X
        self.latents[key] = Z

    def __len__(self):
        return len(self.dataset)


class LatentsDataset2(Dataset):
    """
    Add another variable for the latents.
    Initialises the latents to standard normal
    """
    def __init__(self, dataset, latent_dim, target_idx=1, index_idx=2):
        super().__init__()
        self.dataset = dataset
        self.index_idx = index_idx
        self.target_idx = target_idx
        self.latent_dim = latent_dim

        I = dataset.get_all_data()[self.index_idx]
        self.latents = np.random.randn(len(I), latent_dim).astype(np.float32)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        latents = self.latents[data[self.index_idx]]

        # Insert the latents in target_idx
        data = data[:self.target_idx] + (latents, ) + data[self.target_idx:]

        return data

    def __setitem__(self, key, value):
        """
        Allows mutation of the underlying dataset.
        """
        self.dataset[key] = value

    def set_latents(self, key, value):
        self.latents[key] = value

    def __len__(self):
        return len(self.dataset)
