import numpy as np
import os
from numpy.lib.function_base import corrcoef
import pandas as pd

from matplotlib import pyplot as plt
from torch.utils.data import Dataset


def save_splits(root, dataset):
    samples = np.load(os.path.join(root, f'{dataset}_gen_data.npz'))['samples']
    splits = (
        ('train', samples[:int(8e5)]),
        ('val', samples[int(8e5):int(9e5)]),
        ('test', samples[int(9e5):])
    )
    for split in splits:
        name, data = split
        file = os.path.join(root, f'gen_{dataset}', '{}.npy'.format(name))
        np.save(file, data)


class GenDataset(Dataset):
    def __init__(self, root, data='gas', split='train', frac=None):
        path = os.path.join(root, f'gen_{data}', '{}.npy'.format(split))
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
    dataset = GenDataset(root='./data', data='gas', split='train')
    print(type(dataset.data))

    print("mean", dataset.data.mean(axis=0))
    print("median", np.median(dataset.data, axis=0))
    print(dataset.data.shape)
    print("min", dataset.data.min(axis=0))
    print("max", dataset.data.max(axis=0))
    print(np.where(dataset.data == dataset.data.max()))
    # fig, axs = plt.subplots(3, 3, figsize=(10, 7), sharex=True, sharey=True)
    # axs = axs.reshape(-1)
    # for i, dimension in enumerate(dataset.data.T):
    #     axs[i].hist(dimension, bins=100)
    # fig.tight_layout()
    # plt.show()

    import seaborn as sns
    import pandas as pd

    sns.pairplot(pd.DataFrame(dataset.data),
                 plot_kws={'s': 6},
                 diag_kws={'bins': 25})
    plt.tight_layout()
    plt.draw()

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
