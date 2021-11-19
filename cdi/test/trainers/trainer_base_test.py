import copy
import unittest.mock as mock

import jsonargparse
import numpy as np
import torch
from pytest import fixture
from torch.utils.data import TensorDataset, random_split

from cdi.trainers.trainer_base import TrainerBase
from cdi.util.arg_utils import convert_namespace


@fixture
def trainer_base():
    args = ['--model_seed', '20190508',
            '--data_seed', '20200325', '20200406', '20200407',
            '--data.num_imputed_copies', '3',
            '--data.dataset_root', 'data',
            '--data.batch_size', '100',
            '--data.dataset', 'toy_set3',
            '--data.total_miss', '0.3',
            '--data.miss_type', 'MCAR',
            '--data.pre_imputation', 'empirical_distribution_samples',
            '--data.filter_fully_missing', 'False']

    argparser = jsonargparse.ArgumentParser()
    argparser = TrainerBase.add_model_args(argparser)
    args = argparser.parse_args(args)

    args = convert_namespace(args)

    return TrainerBase(args)


def test_prepare_data_is_deterministic_within_data_seed(trainer_base):
    # Mock
    original_split_dataset = TrainerBase.split_dataset
    original_data_seeds = copy.deepcopy(trainer_base.hparams.data_seeds)

    def patched_return(dataset):
        train, val = original_split_dataset(dataset)
        # Only keep a random subset of the training data to improve
        # the test speed, since imputation of whole dataset
        # can take a while.
        train = random_split(train, [500, len(train) - 500])[0]
        return train, val

    with mock.patch("cdi.trainers.trainer_base.TrainerBase.split_dataset",
                    side_effect=patched_return):
        # Method call
        trainer_base.setup(stage='fit')
        X1, M1, I1 = trainer_base.train_dataset[:][:3]

        # Keep data seed, change model_seed
        trainer_base.hparams.model_seed = 111
        trainer_base.setup(stage='fit')
        X2, M2, I2 = trainer_base.train_dataset[:][:3]

        assert (X1.shape == X2.shape
                and (np.all(X1 == X2)
                     and np.all(M1 == M2)
                     and np.all(I1 == I2))),\
            'Does not return the same data!'

        # Change data generation seed and prepare a different dataset
        trainer_base.hparams.data_seeds[0] = 111
        trainer_base.setup(stage='fit')
        X3, M3, I3 = trainer_base.train_dataset[:][:3]

        # A bit awkward and not perfect test
        # Check that the observed data is the same
        # But that the missingness masks are different
        X1_amputed = X1*M1
        X3_amputed = X3*M3
        M_different = []
        for i in range(X1_amputed.shape[0]):
            x1 = X1_amputed[i, :]
            m1 = M1[i, :]
            X3_temp = X3_amputed * m1

            x1 = x1*M3
            np.all(x1 == X3_temp, axis=0)

            idx = np.argwhere(np.all(x1 == X3_temp, axis=1))
            assert len(idx) != 0,\
                ('The observed data should be the same!')
            M_different.append(np.any(np.all(M3[idx, :] != m1, axis=0)))

        # Some missingness can be the same (e.g. for fully-observed
        # samples for by chance), but not all!
        assert np.sum(M_different) != 0,\
            ('The missingness masks should be different!')

        # Change data split seed and prepare a different dataset
        trainer_base.hparams.data_seeds = copy.deepcopy(original_data_seeds)
        trainer_base.hparams.data_seeds[1] = 111
        trainer_base.setup(stage='fit')
        X4, M4, I4 = trainer_base.train_dataset[:][:3]

        assert (X1.shape != X4.shape
                or not (np.all(X1 == X4)
                        or np.all(M1 == M4))),\
            'Should not return the same data!'

        # Change data initialization seed and prepare a different dataset
        trainer_base.hparams.data_seeds = copy.deepcopy(original_data_seeds)
        trainer_base.hparams.data_seeds[2] = 111
        trainer_base.setup(stage='fit')
        X5, M5, I5 = trainer_base.train_dataset[:][:3]

        assert (X1.shape == X5.shape
                and np.all(X1*M1 == X5*M5)
                and np.all(M1 == M5)),\
            'Should not return the same data!'


def test_prepare_data_augments_incomplete_data(trainer_base):
    with mock.patch('cdi.trainers.trainer_base.TrainerBase.load_dataset') as tb_ld,\
         mock.patch('cdi.trainers.trainer_base.MissingDataProvider') as tb_dm,\
         mock.patch('cdi.trainers.trainer_base.TrainerBase.split_dataset') as tb_sd:
        # Mock data
        X = torch.tensor([[1, 2, 3, 4, 5],
                          [9, 8, 7, 6, 5],
                          [1, 0, 4, 0, 6],
                          [1, 1, 1, 5, 5],
                          [9, 0, 8, 2, 7]], dtype=torch.float)
        M = torch.tensor([[1, 0, 0, 1, 1],
                          [1, 1, 1, 1, 1],
                          [0, 0, 1, 1, 1],
                          [1, 1, 1, 1, 0],
                          [1, 0, 1, 1, 1]], dtype=torch.bool)
        I = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

        # Mock
        tb_ld.return_value = TensorDataset(X, I)
        tb_dm.return_value = TensorDataset(X, M, I)
        # Don't split for now.
        tb_sd.return_value = (tb_dm.return_value, None)

        # Method call
        trainer_base.setup(stage='fit')
        train_dataset = trainer_base.train_dataset

        X, M, I = train_dataset[:][:3]
        incomp_idx = np.argwhere(M.sum(axis=1) != X.shape[-1])

        # Check that incomplete samples are copied N times
        for i in incomp_idx:
            x, m = X[i, :], M[i, :]
            count = 0
            for xi, mi in zip(X, M):
                if (np.all(mi == m)
                        and np.all(x*m == xi*mi)):
                    count += 1

            assert count == trainer_base.hparams.data.num_imputed_copies,\
                (f'There should be {trainer_base.hparams.data.num_imputed_copies}'
                 ' copies for each incomplete sample!')

        # Check that complete samples are not augmented
        comp_idx = np.argwhere(M.sum(axis=0) == X.shape[-1])
        for i in comp_idx:
            x, m = X[i, :], M[i, :]
            count = 0
            for xi, mi in zip(X, M):
                if (np.all(mi == m)
                    and np.all(x*m == xi*mi)
                        and not np.all(x*~m == xi*~mi)):
                    count += 1

            assert count == 1,\
                ('There should be only one copy for each complete sample!')


def test_missing_mask_determinism(trainer_base):
    # No copies for now
    trainer_base.hparams.data.num_imputed_copies = 1
    trainer_base.hparams.data.pre_imputation = 'true_values'

    def patched_split_return(dataset):
        # Don't split
        return dataset, None

    # of returned samples
    with mock.patch('cdi.trainers.trainer_base.TrainerBase.load_dataset') as tb_ld,\
        mock.patch('cdi.trainers.trainer_base.TrainerBase.split_dataset',
                   side_effect=patched_split_return) as tb_sd:
        # Mock data
        X = torch.tensor([[1, 2, 3, 4, 5],
                          [9, 8, 7, 6, 5],
                          [1, 0, 4, 0, 6],
                          [1, 1, 1, 5, 5],
                          [9, 0, 8, 2, 7]], dtype=torch.float)
        I = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

        # Mock
        tb_ld.return_value = TensorDataset(X, I)

        # Method call
        trainer_base.hparams.data_seeds[0] = 1
        trainer_base.setup(stage='fit')
        train_dataset = trainer_base.train_dataset

        X1, M1, I1 = train_dataset[:][:3]

        # Try same seed again
        trainer_base.hparams.data_seeds[0] = 1
        trainer_base.setup(stage='fit')
        train_dataset = trainer_base.train_dataset

        X2, M2, I2 = train_dataset[:][:3]

        # Try different seed
        trainer_base.hparams.data_seeds[0] = 3
        trainer_base.setup(stage='fit')
        train_dataset = trainer_base.train_dataset

        X3, M3, I3 = train_dataset[:][:3]

        assert np.all(X1 == X2) and np.all(M1 == M2),\
            'Same seed returned different data!'

        assert (np.all(X1 == X3) and not np.all(M1 == M3)),\
            'Different seed didn\'t return different data!'


def test_removes_fully_observed_samples(trainer_base):
    trainer_base.hparams.data.filter_fully_missing = True
    trainer_base.hparams.data.pre_imputation = 'true_values'
    trainer_base.hparams.data.num_imputed_copies = 1

    with mock.patch('cdi.trainers.trainer_base.TrainerBase.load_dataset') as tb_ld,\
            mock.patch('cdi.trainers.trainer_base.MissingDataProvider') as tb_dm,\
            mock.patch('cdi.trainers.trainer_base.TrainerBase.split_dataset') as tb_sd:
        # Mock data
        X = torch.tensor([[1, 2, 3, 4, 5],
                          [9, 8, 7, 6, 5],
                          [1, 0, 4, 0, 6],
                          [1, 1, 1, 5, 5],
                          [9, 0, 8, 2, 7]], dtype=torch.float)
        M = torch.tensor([[1, 0, 0, 1, 1],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1],
                          [1, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0]], dtype=torch.bool)
        I = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

        # Mock
        tb_ld.return_value = TensorDataset(X, I)
        tb_dm.return_value = TensorDataset(X, M, I)
        # Don't split for now.
        tb_sd.return_value = (tb_dm.return_value, None)

        # Method call
        trainer_base.setup(stage='fit')
        X1, M1, I1 = trainer_base.train_dataset[:][:3]

        assert len(X1) == 3 and len(M1) == 3 and len(I1) == 3,\
            'The dataset should be smaller!'

        X = X.numpy()
        M = M.numpy()
        assert (np.all(X[0] == X1[0])
                and np.all(X[2] == X1[1])
                and np.all(X[3] == X1[2])
                and np.all(M[0] == M1[0])
                and np.all(M[2] == M1[1])
                and np.all(M[3] == M1[2])),\
            'The filtered dataset should contain not fully-missing samples!'
