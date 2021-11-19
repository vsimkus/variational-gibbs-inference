import math
import unittest.mock as mock

import jsonargparse
import torch
import torch.testing as tt
from pytest import fixture

from cdi.trainers.cdi import CDI, UpdateComponentsEnum
from cdi.util.arg_utils import parse_bool, convert_namespace
from cdi.util.test_utils import TensorMatcher
from cdi.util.utils import EpochScheduler


@fixture
def cdi_trainer():
    args = ['--fa_model.input_dim', '6',
            '--fa_model.latent_dim', '2',
            '--model_optim.learning_rate', '0.3',
            '--model_optim.weight_decay_coeff', '0.0',
            '--cdi.update_components', 'MISSING',
            '--cdi.update_comp_schedule', '0', '10', '20',
            '--cdi.update_comp_schedule_values', '-1', '1', '-1',
            '--cdi.num_samples', '3',
            '--cdi.imputation_delay', '0',
            '--cdi.num_imp_steps_schedule', '0',
            '--cdi.num_imp_steps_schedule_values', '1',
            '--cdi.imputation_comp_schedule', '0', '5', '25',
            '--cdi.imputation_comp_schedule_values', '-1', '1', '-1',
            '--cdi.entropy_coeff', '1.0',
            '--cdi.imp_acceptance_check_schedule', '0',
            '--cdi.imp_acceptance_check_schedule_values', 'False',
            '--data.num_imputed_copies', '2',
            '--data.miss_type', 'MCAR',
            '--data.total_miss', '0.5',
            '--data.batch_size', '2048',
            '--data.filter_fully_missing', 'False',
            '--method', 'analytical']

    argparser = jsonargparse.ArgumentParser()
    argparser.add_argument('--method',
                           type=str, required=True,
                           choices=['complete-case', 'variational',
                                    'analytical', 'expectation-maximisation',
                                    'mc-expectation-maximisation',
                                    'variational-inference',
                                    'var-pretraining', 'variational-em'])
    argparser = CDI.add_model_args(argparser, args)
    args = argparser.parse_args(args)

    args = convert_namespace(args)

    cdi_trainer = CDI(args)
    cdi_trainer.on_epoch_start()
    return cdi_trainer


def test_select_fraction_components_per_example_from_M_one(cdi_trainer):
    # Input data
    M = torch.tensor([[0, 1, 1, 0, 1],
                      [1, 0, 0, 1, 1],
                      [0, 0, 0, 0, 1],
                      [1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1]], dtype=bool)

    # Call method
    M_selected = cdi_trainer.select_fraction_components_per_example_from_M(M, l=-1)

    assert M_selected.dtype == torch.bool, \
        'Does not return a boolean mask.'

    assert (~M_selected).sum() == 4, \
        'Does not select one per row.'

    # Check that the selected elements are from the original M
    tt.assert_allclose((M | M_selected).to(torch.uint8),
                       M_selected.to(torch.uint8))


def test_select_fraction_components_per_example_one(cdi_trainer):
    # Input data
    X = torch.randn(10, 5)

    # Call method
    M_selected = cdi_trainer.select_fraction_components_per_example(X, l=-1)

    assert M_selected.dtype == torch.bool, \
        'Does not return a boolean mask.'

    assert torch.all((~M_selected).sum(dim=1) == 1), \
        'Does not select one per row.'


def test_select_fraction_components_per_example_from_M_fraction(cdi_trainer):
    # Input data
    M = torch.tensor([[0, 1, 1, 0, 1],
                      [1, 0, 0, 1, 1],
                      [0, 0, 0, 0, 1],
                      [1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1]], dtype=bool)

    # Call method
    M_selected = cdi_trainer.select_fraction_components_per_example_from_M(M, l=0.5)

    assert M_selected.dtype == torch.bool, \
        'Does not return a boolean mask.'

    assert torch.all((~M_selected).sum(dim=1) == torch.tensor([1, 1, 2, 1, 0])), \
        'Does not select fraction per row.'

    # Check that the selected elements are from the original M
    tt.assert_allclose((M | M_selected).to(torch.uint8),
                       M_selected.to(torch.uint8))


def test_select_fraction_components_per_example_fraction(cdi_trainer):
    # Input data
    X = torch.randn(10, 5)

    # Call method
    M_selected = cdi_trainer.select_fraction_components_per_example(X, l=0.5)

    assert M_selected.dtype == torch.bool, \
        'Does not return a boolean mask.'

    assert torch.all((~M_selected).sum(dim=1) == 3), \
        'Does not select fraction per row.'


def test_select_fraction_components_per_example_from_M_all(cdi_trainer):
    # Input data
    M = torch.tensor([[0, 1, 1, 0, 1],
                      [1, 0, 0, 1, 1],
                      [0, 0, 0, 0, 1],
                      [1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1]], dtype=bool)

    # Call method
    M_selected = cdi_trainer.select_fraction_components_per_example_from_M(M, l=1)

    assert M_selected.dtype == torch.bool, \
        'Does not return a boolean mask.'

    assert torch.all(M == M_selected), \
        'Does not select all per row.'

    # Check that the selected elements are from the original M
    tt.assert_allclose((M | M_selected).to(torch.uint8),
                       M_selected.to(torch.uint8))


def test_select_fraction_components_per_example_all(cdi_trainer):
    # Input data
    X = torch.randn(10, 5)

    # Call method
    M_selected = cdi_trainer.select_fraction_components_per_example(X, l=1)

    assert M_selected.dtype == torch.bool, \
        'Does not return a boolean mask.'

    assert torch.all(~M_selected), \
        'Does not select all per row.'


def test_complete_all_dims(cdi_trainer):
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [11, 0, 0, 0],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)

    # Mock data
    X_samples = torch.tensor([[[10, 0, 0, 0],
                               [0, 1, 2, 3],
                               [0, 0, 0, 0]],
                              [[8, 0, 0, 0],
                               [0, 3, 2, 1],
                               [0, 0, 0, 0]],
                              [[1, 0, 0, 0],
                               [0, 4, 4, 4],
                               [0, 0, 0, 0]]],
                             dtype=torch.float)

    # True outputs
    X_comp_true = torch.tensor([[10, 1, -1, 20],
                                [11, 1, 0, 0],
                                [11, 0, 2, 0],
                                [11, 0, 0, 3],
                                [8, 1, -1, 20],
                                [11, 3, 0, 0],
                                [11, 0, 2, 0],
                                [11, 0, 0, 1],
                                [1, 1, -1, 20],
                                [11, 4, 0, 0],
                                [11, 0, 4, 0],
                                [11, 0, 0, 4]],
                               dtype=torch.float)

    x_idx_true = torch.tensor([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                              dtype=torch.long)

    # Call method
    X_completed, x_idx = cdi_trainer.complete_all_dims(X, M, X_samples)

    # Check that completion is correct
    tt.assert_allclose(X_completed, X_comp_true)

    # Check that reference indices of completed samples are corect
    tt.assert_allclose(x_idx, x_idx_true)


def run_training_step_update_all(cdi_trainer):
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [0, 1, -1, 20],  # Augmentation of incomplete sample
                      [11, 0, 0, 0],
                      [11, 0, 0, 0],  # Augmentation of incomplete sample
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)
    I = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    OI = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    incomp_mask = torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)

    # Mock data
    X_samples = torch.tensor([[[10, 0, 0, 0],
                               [10, 0, 0, 0],
                               [0, 1, 2, 3],
                               [0, 1, 2, 3],
                               [0, 0, 0, 0]],
                              [[8, 0, 0, 0],
                               [8, 0, 0, 0],
                               [0, 3, 2, 1],
                               [0, 3, 2, 1],
                               [0, 0, 0, 0]],
                              [[1, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 4, 4, 4],
                               [0, 4, 4, 4],
                               [0, 0, 0, 0]]],
                             dtype=torch.float)
    entropy = torch.tensor(
                    [[1, float('nan'), float('nan'), float('nan')],
                     [2, float('nan'), float('nan'), float('nan')],
                     [float('nan'), -1, 5, -4],
                     [float('nan'), -2, 6, -2],
                     [float('nan'), float('nan'), float('nan'), float('nan')]],
                    dtype=torch.float)

    # True outputs
    X_comp_true = torch.tensor([[10.,  1., -1., 20.],
                                [10.,  1., -1., 20.],
                                [11.,  1.,  0.,  0.],
                                [11.,  0.,  2.,  0.],
                                [11.,  0.,  0.,  3.],
                                [11.,  1.,  0.,  0.],
                                [11.,  0.,  2.,  0.],
                                [11.,  0.,  0.,  3.],
                                [ 8.,  1., -1., 20.],
                                [ 8.,  1., -1., 20.],
                                [11.,  3.,  0.,  0.],
                                [11.,  0.,  2.,  0.],
                                [11.,  0.,  0.,  1.],
                                [11.,  3.,  0.,  0.],
                                [11.,  0.,  2.,  0.],
                                [11.,  0.,  0.,  1.],
                                [ 1.,  1., -1., 20.],
                                [ 1.,  1., -1., 20.],
                                [11.,  4.,  0.,  0.],
                                [11.,  0.,  4.,  0.],
                                [11.,  0.,  0.,  4.],
                                [11.,  4.,  0.,  0.],
                                [11.,  0.,  4.,  0.],
                                [11.,  0.,  0.,  4.]], dtype=torch.float)
    idx_ref_true = torch.tensor([0, 1, 2, 2, 2, 3, 3, 3, 0, 1, 2, 2, 2, 3,
                                 3, 3, 0, 1, 2, 2, 2, 3, 3, 3],
                                dtype=torch.long)

    # Mock calls
    cdi_trainer.sample_missing_values = mock.MagicMock(
        return_value=(X_samples, entropy)
    )

    forward_out = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                               dtype=torch.float)
    def forward_call(X, M):
        if torch.all(torch.eq(X, X_comp_true)):
            return (forward_out, None)
        elif torch.all(torch.eq(X, torch.tensor(
                                            [[1, 2, 3, 4]],
                                            dtype=torch.float))):
            return torch.tensor([-1], dtype=torch.float), None
        else:
            print('neither!')

    cdi_trainer.forward = mock.MagicMock(side_effect=forward_call)

    # Call method
    output = cdi_trainer.training_step((X, M, I, OI, incomp_mask), 0)

    true_log_lik = torch.zeros(X.shape[0])
    true_log_lik = true_log_lik.index_add(0, idx_ref_true, forward_out)

    num_mis = (~M).sum(axis=1)
    num_mis[num_mis == 0] = 1  # Avoid division by zero
    true_log_lik /= num_mis

    x_aug_idx = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    true_log_lik2 = torch.zeros(3)
    true_log_lik = true_log_lik2.index_add(0, x_aug_idx, true_log_lik)

    true_log_lik = true_log_lik.sum()

    # True outputs
    # # First sample had one missing value
    # true_log_lik = (sum([1, 2, 3])
    #                 # Second sample (augmented copy of the first one same mis)
    #                 + sum([4, 5, 6])
    #                 # Third sample had 3 missing values
    #                 + sum([7, 8, 9, 10, 11, 12, 13, 14, 15])/3
    #                 # Fourth sample (augmented copy of the third one same mis)
    #                 + sum([16, 17, 18, 19, 20, 21, 22, 23, 24])/3)
    # Divide by the number of (expectation) samples K
    true_log_lik /= X_samples.shape[0]
    # Since the incomplete samples have 2 copies each (num_imputed_copies)
    # we divide their likelihood by 2
    true_log_lik /= cdi_trainer.hparams.data.num_imputed_copies
    # Add the log-lik of the fully-observed third sample
    true_log_lik += -1
    # Divide by the number of *true unaugmented* samples to get average
    num_true_samples = (~incomp_mask).sum() \
        + torch.true_divide(incomp_mask.sum(), cdi_trainer.hparams.data.num_imputed_copies)
    true_log_lik /= num_true_samples.item()

    true_entropy = entropy.clone()
    true_entropy[torch.isnan(true_entropy)] = 0.
    true_entropy = true_entropy.sum(dim=1)
    # Divide sum for each sample by the number of missing values
    true_entropy[0] = true_entropy[0] / 1
    true_entropy[1] = true_entropy[1] / 1
    true_entropy[2] = true_entropy[2] / 3
    true_entropy[3] = true_entropy[3] / 3
    true_entropy = (true_entropy[0] + true_entropy[1]
                    + true_entropy[2] + true_entropy[3]).item()
    # Divide by the number of augmentations to scale down the entropies
    true_entropy /= cdi_trainer.hparams.data.num_imputed_copies
    # Divide by the number of *true unaugmented* samples to get batch average
    true_entropy /= num_true_samples.item()

    cdi_trainer.sample_missing_values.assert_called_once_with(
        (X, TensorMatcher(M.type_as(X)), I), M_selected=M,
        K=cdi_trainer.hparams.cdi.num_samples, sample_all=True
    )

    assert math.isclose(output['progress_bar']['train_log_lik'], true_log_lik,
                        rel_tol=0.00001),\
        'Log likelihood not correct!'
    assert math.isclose(output['progress_bar']['train_entropy'], true_entropy,
                        rel_tol=0.00001),\
        'Entropy not correct!'
    assert math.isclose(output['loss'].item(),
                        (-true_log_lik - true_entropy),
                        rel_tol=0.00001),\
        'Loss is not correct!'


def test_training_step_update_all(cdi_trainer):
    # update_components == UpdateComponentsEnum.MISSING
    cdi_trainer.hparams.cdi.imputation_delay = 10000
    cdi_trainer.current_epoch = 10
    run_training_step_update_all(cdi_trainer)

    cdi_trainer.current_epoch = 19
    run_training_step_update_all(cdi_trainer)

    # cdi_trainer.hparams.data.num_imputed_copies = 1
    # run_training_step_update_all(cdi_trainer)

    # cdi_trainer.hparams.data.num_imputed_copies = 5
    # run_training_step_update_all(cdi_trainer)

    # update_components == UpdateComponentsEnum.ALL
    cdi_trainer.hparams.update_components = UpdateComponentsEnum.ALL

    cdi_trainer.hparams.cdi.imputation_delay = 10000
    cdi_trainer.current_epoch = 10
    run_training_step_update_all(cdi_trainer)

    cdi_trainer.current_epoch = 19
    run_training_step_update_all(cdi_trainer)

    # cdi_trainer.hparams.data.num_imputed_copies = 1
    # run_training_step_update_all(cdi_trainer)

    # cdi_trainer.hparams.data.num_imputed_copies = 5
    # run_training_step_update_all(cdi_trainer)


def run_training_step_update_one(cdi_trainer):
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [0, 1, -1, 20],
                      [11, 0, 0, 0],
                      [11, 0, 0, 0],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)
    I = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    OI = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    incomp_mask = torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)

    # Mock data
    M_selected = torch.tensor([[0, 1, 1, 1],
                               [0, 1, 1, 1],
                               [1, 1, 1, 0],
                               [1, 1, 0, 1],
                               [1, 1, 1, 1]],
                              dtype=torch.bool)

    X_samples = torch.tensor([[[10, 0, 0, 0],
                               [11, 0, 0, 0],
                               [0, 0, 0, 3],
                               [0, 0, 1, 0],
                               [0, 0, 0, 0]],
                              [[8, 0, 0, 0],
                               [9, 0, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 2, 0],
                               [0, 0, 0, 0]],
                              [[1, 0, 0, 0],
                               [3, 0, 0, 0],
                               [0, 0, 0, 4],
                               [0, 0, 3, 0],
                               [0, 0, 0, 0]]],
                             dtype=torch.float)
    entropy = torch.tensor(
                    [[1, float('nan'), float('nan'), float('nan')],
                     [1, float('nan'), float('nan'), float('nan')],
                     [float('nan'), float('nan'), float('nan'), -4],
                     [float('nan'), float('nan'), -2, float('nan')],
                     [float('nan'), float('nan'), float('nan'), float('nan')]],
                    dtype=torch.float)

    # True outputs
    X_comp_true = torch.tensor([[10, 1, -1, 20],
                                [11, 1, -1, 20],
                                [11, 0, 0, 3],
                                [11, 0, 1, 0],
                                [8, 1, -1, 20],
                                [9, 1, -1, 20],
                                [11, 0, 0, 1],
                                [11, 0, 2, 0],
                                [1, 1, -1, 20],
                                [3, 1, -1, 20],
                                [11, 0, 0, 4],
                                [11, 0, 3, 0]],
                               dtype=torch.float)

    # Mock calls
    cdi_trainer.sample_missing_values = mock.MagicMock(
        return_value=(X_samples, entropy)
    )
    cdi_trainer.select_fraction_components_per_example_from_M = mock.MagicMock(
        return_value=M_selected
    )

    def forward_call(X, M):
        if torch.all(torch.eq(X, X_comp_true)):
            return (torch.tensor([1, 2, 3, 4,
                                  6, 7, 8, 9,
                                  11, 12, 13, 14],
                                 dtype=torch.float)
                    , None)
        elif torch.all(torch.eq(X, torch.tensor(
                                    [[1, 2, 3, 4]],
                                    dtype=torch.float))):
            return torch.tensor([5], dtype=torch.float), None
        else:
            print('neither!')

    cdi_trainer.forward = mock.MagicMock(side_effect=forward_call)

    # Call method
    output = cdi_trainer.training_step((X, M, I, OI, incomp_mask), 0)

    # True outputs
    # First sample had one missing value
    true_log_lik = (sum([1, 6, 11])
                    # Second sample (copy of first) has same mis
                    + sum([2, 7, 12])
                    # Third sample had one missing values
                    + sum([3, 8, 13])
                    # Fourth sample (copy of third) has same mis
                    + sum([4, 9, 14]))
    # Divide by the number of (expectation) samples K
    true_log_lik /= X_samples.shape[0]
    # Since the incomplete samples have 2 copies each (num_imputed_copies)
    # we divide their likelihood by 2
    true_log_lik /= cdi_trainer.hparams.data.num_imputed_copies
    # Add the log-lik of the fully-observed third sample
    true_log_lik += 5
    # Divide by the number of *true unaugmented* samples to get average
    num_true_samples = (~incomp_mask).sum() \
        + torch.true_divide(incomp_mask.sum(), cdi_trainer.hparams.data.num_imputed_copies)
    true_log_lik /= num_true_samples.item()

    true_entropy = entropy.clone()
    true_entropy[torch.isnan(true_entropy)] = 0.
    true_entropy = true_entropy.sum(dim=1)
    true_entropy[0] = true_entropy[0] / 1
    true_entropy[1] = true_entropy[1] / 1
    true_entropy[2] = true_entropy[2] / 1
    true_entropy[3] = true_entropy[3] / 1
    true_entropy = (true_entropy[0] + true_entropy[1]
                    + true_entropy[2] + true_entropy[3]).item()
    # Divide by the number of incomplete samples to scale down the entropies
    true_entropy /= cdi_trainer.hparams.data.num_imputed_copies
    # Divide by the number of *true unaugmented* samples to get batch average
    true_entropy /= num_true_samples.item()

    cdi_trainer.sample_missing_values.assert_called_once_with(
        (X, TensorMatcher(M.type_as(X)), I), M_selected=M_selected,
        K=cdi_trainer.hparams.cdi.num_samples, sample_all=False
    )

    assert math.isclose(output['progress_bar']['train_log_lik'], true_log_lik,
                        rel_tol=0.00001),\
        'Log likelihood not correct!'
    assert math.isclose(output['progress_bar']['train_entropy'], true_entropy,
                        rel_tol=0.00001),\
        'Entropy not correct!'
    assert math.isclose(output['loss'].item(),
                        (-true_log_lik - true_entropy),
                        rel_tol=0.00001),\
        'Loss is not correct!'


def test_training_step_update_one(cdi_trainer):
    # update_components == UpdateComponentsEnum.MISSING
    cdi_trainer.hparams.cdi.imputation_delay = 10000
    cdi_trainer.current_epoch = 0
    run_training_step_update_one(cdi_trainer)

    cdi_trainer.current_epoch = 9
    run_training_step_update_one(cdi_trainer)

    cdi_trainer.current_epoch = 20
    run_training_step_update_one(cdi_trainer)

    cdi_trainer.current_epoch = 100
    run_training_step_update_one(cdi_trainer)

    # cdi_trainer.hparams.data.num_imputed_copies = 1
    # run_training_step_update_one(cdi_trainer)

    # cdi_trainer.hparams.data.num_imputed_copies = 5
    # run_training_step_update_one(cdi_trainer)

    # update_components == UpdateComponentsEnum.ALL
    cdi_trainer.hparams.update_components = UpdateComponentsEnum.ALL

    cdi_trainer.hparams.cdi.imputation_delay = 10000
    cdi_trainer.current_epoch = 0
    run_training_step_update_one(cdi_trainer)

    cdi_trainer.current_epoch = 9
    run_training_step_update_one(cdi_trainer)

    cdi_trainer.current_epoch = 20
    run_training_step_update_one(cdi_trainer)

    cdi_trainer.current_epoch = 100
    run_training_step_update_one(cdi_trainer)

    # cdi_trainer.hparams.data.num_imputed_copies = 1
    # run_training_step_update_one(cdi_trainer)

    # cdi_trainer.hparams.data.num_imputed_copies = 5
    # run_training_step_update_one(cdi_trainer)


def run_impute_batch_impute_all(cdi_trainer):
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [11, 0, 0, 0],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)
    I = torch.tensor([0, 1, 2], dtype=torch.long)

    # Mock data
    X_samples = torch.tensor([[10, 0, 0, 0],
                              [0, 1, 2, 3],
                              [0, 0, 0, 0]],
                             dtype=torch.float)

    # True outputs
    X_imputed_true = torch.tensor([[10, 1, -1, 20],
                                   [11, 1, 2, 3],
                                   [1, 2, 3, 4]],
                                  dtype=torch.float)

    # Mocks
    cdi_trainer.sample_imputation_values = mock.MagicMock(
        return_value=X_samples
    )
    cdi_trainer.train_dataset = mock.MagicMock()
    cdi_trainer.train_dataset.__setitem__ = mock.MagicMock()

    # Call method
    cdi_trainer.impute_batch((X, M, I), stage='train')

    cdi_trainer.sample_imputation_values.assert_called_once_with(
        (X, M, I), M_selected=M, sample_all=True
    )

    # Check that the batch is correctly imputed
    tt.assert_allclose(X, X_imputed_true)

    # Check that the dataset is imputed!
    cdi_trainer.train_dataset.__setitem__.assert_called_once_with(I, X)


def test_impute_batch_impute_all(cdi_trainer):
    cdi_trainer.current_epoch = 5
    run_impute_batch_impute_all(cdi_trainer)

    cdi_trainer.current_epoch = 24
    run_impute_batch_impute_all(cdi_trainer)


def run_impute_batch_impute_one(cdi_trainer):
    l = cdi_trainer.update_comp_schedule.get_value(cdi_trainer.current_epoch)
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [11, 0, 0, 0],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)
    I = torch.tensor([0, 1, 2], dtype=torch.long)

    # Mock data
    X_samples = torch.tensor([[10, 0, 0, 0],
                              [0, 0, 0, 3],
                              [0, 0, 0, 0]],
                             dtype=torch.float)
    M_selected = torch.tensor([[0, 1, 1, 1],
                               [1, 1, 1, 0],
                               [1, 1, 1, 1]],
                              dtype=torch.bool)

    # True outputs
    X_imputed_true = torch.tensor([[10, 1, -1, 20],
                                   [11, 0, 0, 3],
                                   [1, 2, 3, 4]],
                                  dtype=torch.float)

    # Mocks
    cdi_trainer.sample_imputation_values = mock.MagicMock(
        return_value=X_samples
    )
    cdi_trainer.select_fraction_components_per_example_from_M = mock.MagicMock(
        return_value=M_selected
    )
    cdi_trainer.train_dataset = mock.MagicMock()
    cdi_trainer.train_dataset.__setitem__ = mock.MagicMock()

    # Call method
    cdi_trainer.impute_batch((X, M, I), stage='train')

    cdi_trainer.select_fraction_components_per_example_from_M.assert_called_once_with(M, l)
    cdi_trainer.sample_imputation_values.assert_called_once_with(
        (X, M, I), M_selected=M_selected, sample_all=False
    )

    # Check that the batch is correctly imputed
    tt.assert_allclose(X, X_imputed_true)

    # Check that the dataset is imputed!
    cdi_trainer.train_dataset.__setitem__.assert_called_once_with(I, X)


def test_impute_batch_impute_one(cdi_trainer):
    cdi_trainer.current_epoch = 0
    run_impute_batch_impute_one(cdi_trainer)

    cdi_trainer.current_epoch = 4
    run_impute_batch_impute_one(cdi_trainer)

    cdi_trainer.current_epoch = 25
    run_impute_batch_impute_one(cdi_trainer)

    cdi_trainer.current_epoch = 50
    run_impute_batch_impute_one(cdi_trainer)


def test_impute_batch_acceptance_check(cdi_trainer):
    l = cdi_trainer.update_comp_schedule.get_value(cdi_trainer.current_epoch)
    cdi_trainer.hparams.cdi.imp_acceptance_check_schedule_values[0] = 'True'
    cdi_trainer.imp_acceptance_check_schedule = EpochScheduler(
                    cdi_trainer,
                    cdi_trainer.hparams.cdi.imp_acceptance_check_schedule,
                    [parse_bool(v) for v
                     in cdi_trainer.hparams.cdi.imp_acceptance_check_schedule_values])
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [0, 1, -1, 20],
                      [11, 0, 0, 0],
                      [11, 0, 0, 0],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)
    I = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    # Mock data
    X_samples = torch.tensor([[10, 0, 0, 0],
                              [9, 0, 0, 0],
                              [0, 0, 0, 10000],
                              [0, 0, 0, 3],
                              [0, 0, 0, 0]],
                             dtype=torch.float)
    M_selected = torch.tensor([[0, 1, 1, 1],
                               [0, 1, 1, 1],
                               [1, 1, 1, 0],
                               [1, 1, 1, 0],
                               [1, 1, 1, 1]],
                              dtype=torch.bool)

    # True outputs
    X_imputed_true = torch.tensor([[10, 1, -1, 20],
                                   [9, 1, -1, 20],
                                   [11, 0, 0, 10000],
                                   [11, 0, 0, 3],
                                   [1, 2, 3, 4]],
                                  dtype=torch.float)

    # Mocks
    cdi_trainer.sample_imputation_values = mock.MagicMock(
        return_value=X_samples
    )
    cdi_trainer.select_fraction_components_per_example_from_M = mock.MagicMock(
        return_value=M_selected
    )
    cdi_trainer.train_dataset = mock.MagicMock()
    cdi_trainer.train_dataset.__setitem__ = mock.MagicMock()

    cdi_trainer.forward = mock.MagicMock(
        return_value=torch.tensor([3, 4, -1000, 1, 5], dtype=torch.float)
    )

    # Call method
    logs = cdi_trainer.impute_batch((X, M, I), stage='train')

    cdi_trainer.select_fraction_components_per_example_from_M.assert_called_once_with(M, l)
    cdi_trainer.sample_imputation_values.assert_called_once_with(
        (X, M, I), M_selected=M_selected, sample_all=False
    )

    # Check that the batch is correctly imputed
    tt.assert_allclose(X[[0, 1, 3, 4], :], X_imputed_true[[0, 1, 3, 4], :])

    # Check that the third sample is rejected, since it is a clear outlier
    tt.assert_allclose(X[2, :], torch.tensor([11, 0, 0, 0], dtype=torch.float))

    assert logs['imp_rejected'] == 1,\
        'Logs should reflect the number of rejected samples!'
