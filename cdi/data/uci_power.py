import numpy as np
import os

from matplotlib import pyplot as plt
from torch.utils.data import Dataset

"""
Adapted from https://github.com/bayesiains/nsf/
"""


def load_power(root):
    def load_data(root):
        file = os.path.join(root, 'power', 'data.npy')
        return np.load(file)

    def load_data_split_with_noise(root):
        rng = np.random.RandomState(42)

        data = load_data(root)
        rng.shuffle(data)
        N = data.shape[0]

        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)
        ############################
        # Add noise
        ############################
        # global_intensity_noise = 0.1*rng.rand(N, 1)
        voltage_noise = 0.01 * rng.rand(N, 1)
        # grp_noise = 0.001*rng.rand(N, 1)
        gap_noise = 0.001 * rng.rand(N, 1)
        sm_noise = rng.rand(N, 3)
        time_noise = np.zeros((N, 1))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data += noise

        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    def load_data_normalised(root):
        data_train, data_validate, data_test = load_data_split_with_noise(root)
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu) / s
        data_validate = (data_validate - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_validate, data_test

    return load_data_normalised(root)


def save_splits(root):
    train, val, test = load_power(root)
    splits = (
        ('train', train),
        ('val', val),
        ('test', test)
    )
    for split in splits:
        name, data = split
        file = os.path.join(root, 'power', '{}.npy'.format(name))
        np.save(file, data)


def print_shape_info(root):
    train, val, test = load_power(root)
    print(train.shape, val.shape, test.shape)


class PowerDataset(Dataset):
    def __init__(self, root, split='train', frac=None):
        path = os.path.join(root, 'power', '{}.npy'.format(split))
        self.data = np.load(path).astype(np.float32)
        self.n, self.dim = self.data.shape
        if frac is not None:
            self.n = int(frac * self.n)

        self.index = np.arange(0, len(self.data), dtype=np.long)

    def __getitem__(self, index):
        return self.data[index], self.index[index]

    def __setitem__(self, index, x):
        self.data[index] = x

    def __len__(self):
        return self.n


def main():
    dataset = PowerDataset(root='./data', split='train')
    print(type(dataset.data))
    print(dataset.data.shape)
    print(dataset.data.min(), dataset.data.max())
    # plt.hist(dataset.data.reshape(-1), bins=250)
    # plt.show()

    # fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True, sharey=True)
    # axes = axes.reshape(-1)
    # for i, dimension in enumerate(dataset.data.T):
    #     axes[i].hist(dimension, bins=250)
    # fig.tight_layout()
    # plt.show()

    # import seaborn as sns
    # import pandas as pd

    # sns.pairplot(pd.DataFrame(dataset.data),
    #              plot_kws={'s': 6},
    #              diag_kws={'bins': 25})
    # plt.tight_layout()
    # plt.draw()

    corrmat = np.corrcoef(dataset.data, rowvar=False)
    abs_corrmat = np.abs(corrmat)
    triang_indices = np.triu_indices_from(corrmat, k=1)
    print('Median abs correlation: ', np.median(abs_corrmat[triang_indices]))
    print('Mean abs correlation: ', np.mean(abs_corrmat[triang_indices]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(abs_corrmat, interpolation='nearest')
    fig.colorbar(cax)
    plt.draw()

    import scipy.stats as ss

    def multivariate_kendalltau(X):
        kendal_taus = []
        kendal_taus_p = []
        for d1 in range(X.shape[-1]):
            for d2 in range(d1+1, X.shape[-1]):
                t = ss.kendalltau(X[:, d1], X[:, d2])
                kendal_taus.append(t[0])
                kendal_taus_p.append(t[1])
        idx = np.triu_indices(X.shape[-1], k=1)

        taus = np.zeros((X.shape[-1],)*2)
        taus[idx] = np.array(kendal_taus)

        p = np.zeros((X.shape[-1],)*2)
        p[idx] = np.array(kendal_taus_p)

        return taus, p

    taus, taus_p = multivariate_kendalltau(dataset.data)
    abs_taus = np.abs(taus)
    print('Median abs Kendal-tau: ', np.median(abs_taus[triang_indices]))
    print('Mean abs Kendal-tau: ', np.mean(abs_taus[triang_indices]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(abs_taus, interpolation='nearest')
    fig.colorbar(cax)
    plt.show()


if __name__ == '__main__':
    main()
