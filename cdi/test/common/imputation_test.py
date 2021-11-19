import numpy as np
import torch
from pytest import fixture
from torch.utils.data import TensorDataset

from cdi.common.imputation import impute_with_empirical_distribution_sample, \
                                  impute_with_mean
from cdi.util.data.data_augmentation_dataset import DataAugmentation


@fixture
def dataset():
    # Original data
    X = torch.tensor([[1, 2, 3, 4, 5],
                      [9, 8, 7, 6, 5],
                      [1, 0, 4, 0, 6],
                      [1, 1, 1, 5, 5],
                      [9, 0, 8, 2, 7],
                      [3, 2, 5, 4, 3]], dtype=torch.float)
    M = torch.tensor([[1, 0, 0, 1, 1],
                      [1, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1],
                      [1, 1, 1, 1, 0],
                      [1, 0, 1, 1, 1],
                      [1, 1, 1, 1, 1]], dtype=torch.bool)
    I = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)

    dataset = TensorDataset(X, M, I)
    dataset = DataAugmentation(dataset, num_copies=3)
    return dataset


def test_impute_with_mean(dataset):
    # Method call
    impute_with_mean(dataset)

    dim0_mean = sum([1, 9, 1, 9, 3])/5
    dim1_mean = sum([8, 1, 2])/3
    dim2_mean = sum([7, 4, 1, 8, 5])/5
    dim3_mean = sum([4, 6, 9, 5, 2, 4])/6
    dim4_mean = sum([5, 5, 6, 7, 3])/5

    x = dataset[0][0]
    assert (np.all(x[:, 1] == dim1_mean)
            and np.all(x[:, 2] == dim2_mean)),\
        'Mean imputation incorrect!'

    x = dataset[1][0]
    assert np.all(x == np.array([9, 8, 7, 6, 5])),\
        'Mean imputation should not affect complete samples!'

    x = dataset[2][0]
    assert (np.all(x[:, 0] == dim0_mean)
            and np.all(x[:, 1] == dim1_mean)),\
        'Mean imputation incorrect!'

    x = dataset[3][0]
    assert (np.all(x[:, 4] == dim4_mean)),\
        'Mean imputation incorrect!'

    x = dataset[4][0]
    assert (np.all(x[:, 1] == dim1_mean)),\
        'Mean imputation incorrect!'

    x = dataset[5][0]
    assert np.all(x == np.array([3, 2, 5, 4, 3])),\
        'Mean imputation should not affect complete samples!'


def test_impute_with_empirical_distribution_sample(dataset):
    # Method call
    impute_with_empirical_distribution_sample(dataset)

    dim0_emp = np.array([1, 9, 1, 9, 3])
    dim1_emp = np.array([8, 1, 2])
    dim2_emp = np.array([7, 4, 1, 8, 5])
    dim3_emp = np.array([4, 6, 9, 5, 2, 4])
    dim4_emp = np.array([5, 5, 6, 7, 3])

    x = dataset[0][0]
    assert (np.all(np.isin(x[:, 1], dim1_emp))
            and np.all(np.isin(x[:, 2], dim2_emp))),\
        'Emp. sample imputation incorrect!'

    x = dataset[1][0]
    assert np.all(x == np.array([9, 8, 7, 6, 5])),\
        'Imputation should not affect complete samples!'

    x = dataset[2][0]
    assert (np.all(np.isin(x[:, 0], dim0_emp))
            and np.all(np.isin(x[:, 1], dim1_emp))),\
        'Emp. sample imputation incorrect!'

    x = dataset[3][0]
    assert (np.all(np.isin(x[:, 4], dim4_emp))),\
        'Emp. sample imputation incorrect!'

    x = dataset[4][0]
    assert (np.all(np.isin(x[:, 1], dim1_emp))),\
        'Emp. sample imputation incorrect!'

    x = dataset[5][0]
    assert np.all(x == np.array([3, 2, 5, 4, 3])),\
        'Imputation should not affect complete samples!'

    # This test might sporadically fail if all
    # imputations happen to be the same
    assert not (np.all(dataset[0][0][:, 1] == dataset[0][0][:, 1][0])
                and np.all(dataset[0][0][:, 2] == dataset[0][0][:, 2][0])
                and np.all(dataset[2][0][:, 0] == dataset[2][0][:, 0][0])
                and np.all(dataset[2][0][:, 1] == dataset[2][0][:, 1][0])
                and np.all(dataset[3][0][:, 4] == dataset[3][0][:, 4][0])
                and np.all(dataset[4][0][:, 1] == dataset[4][0][:, 1][0])),\
        'Emp. sample imputation should not impute same values!'
