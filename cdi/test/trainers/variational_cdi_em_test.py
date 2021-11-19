import unittest.mock as mock

import jsonargparse
import torch
from pytest import fixture

from cdi.trainers.variational_cdi_em import VarCDIEM
from cdi.util.arg_utils import convert_namespace

@fixture
def cdiem_trainer():
    args = ['--fa_model.input_dim', '4',
            '--fa_model.latent_dim', '2',
            '--variational_model', 'individual',
            '--var_model.input_dim', '3',
            '--var_model.hidden_dims', '10', '4',
            '--var_model.activation', 'lrelu',
            '--var_model.input_missing_vectors', 'False',
            '--var_model.num_univariate_q', '4',
            '--var_optim.learning_rate', '0.3',
            '--var_optim.weight_decay_coeff', '0.0',
            '--cdi.update_components', 'MISSING',
            '--cdi.update_comp_schedule', '0', '10',
            '--cdi.update_comp_schedule_values', '1', '-1',
            '--cdi.num_samples', '1',
            '--cdi.imputation_delay', '0',
            '--cdi.num_imp_steps_schedule', '0',
            '--cdi.num_imp_steps_schedule_values', '1',
            '--cdi.imputation_comp_schedule', '0', '5', '25',
            '--cdi.imputation_comp_schedule_values', '-1', '1', '-1',
            '--cdi.entropy_coeff', '1.0',
            '--cdi.imp_acceptance_check_schedule', '0',
            '--cdi.imp_acceptance_check_schedule_values', 'False',
            '--model_optim.learning_rate', '1e-3',
            '--model_optim.weight_decay_coeff', '0.0',
            '--var_optim.optimiser', 'amsgrad',
            '--var_optim.learning_rate', '1e-3',
            '--var_optim.weight_decay_coeff', '0.0',
            '--var_optim.epsilon', '1e-8',
            '--data.num_imputed_copies', '3',
            '--data.miss_type', 'MCAR',
            '--data.total_miss', '0.5',
            '--data.batch_size', '2048',
            '--data.filter_fully_missing', 'False']

    argparser = jsonargparse.ArgumentParser()
    argparser = VarCDIEM.add_model_args(argparser, args)
    args = argparser.parse_args(args)

    args = convert_namespace(args)

    vcdi = VarCDIEM(args)
    vcdi.configure_optimizers()
    return vcdi


def test_select_one_chain(cdiem_trainer):
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [0, 1, -1, 20],
                      [0, 1, -1, 20],
                      [11, 0, 0, 0],
                      [11, 0, 0, 0],
                      [11, 0, 0, 0],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [0, 1, 1, 1],
                      [0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)
    I = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    OI = torch.tensor([0, 0, 0, 1, 1, 1, 2], dtype=torch.long)
    incomp_mask = torch.tensor([1, 1, 1, 1, 1, 1, 0], dtype=torch.bool)

    one_chain_batch = cdiem_trainer.select_one_chain((X, M, I, OI, incomp_mask))

    X_out, M_out, I_out, OI_out, incomp_mask_out = one_chain_batch

    assert OI_out.shape[0] == 3,\
        'Number of samples selected is incorrect.'

    assert torch.all(torch.unique(OI_out, sorted=True) == torch.tensor([0, 1, 2])),\
        'Origianal index is incorrect.'

    assert torch.all(incomp_mask_out == torch.tensor([1, 1, 0], dtype=torch.bool)),\
        'Incomp_mask is incorrect.'


def test_maximise_model_step(cdiem_trainer):
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [0, 1, -1, 20],
                      [0, 1, -1, 20],
                      [11, 0, 0, 0],
                      [11, 0, 0, 0],
                      [11, 0, 0, 0],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [0, 1, 1, 1],
                      [0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)
    I = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    OI = torch.tensor([0, 0, 0, 1, 1, 1, 2], dtype=torch.long)
    incomp_mask = torch.tensor([1, 1, 1, 1, 1, 1, 0], dtype=torch.bool)

    cdiem_trainer.forward = mock.MagicMock(
                            return_value=(torch.tensor([1, 2, 3, 6, 7, 8, 11],
                                                       dtype=torch.float),
                                          None))

    output = cdiem_trainer.maximise_model_step((X, M, I, OI, incomp_mask))
    loss = output['loss']

    true_loss = -torch.tensor((sum([1, 2, 3]) / 3 + sum([6, 7, 8]) / 3 + 11) / 3)

    assert loss == true_loss,\
        'Loss doesn\'t match.'


def test_update_var_model_step(cdiem_trainer):
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [0, 1, -1, 20],
                      [0, 1, -1, 20],
                      [11, 0, 0, 0],
                      [11, 0, 0, 0],
                      [11, 0, 0, 0],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [0, 1, 1, 1],
                      [0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)
    I = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    OI = torch.tensor([0, 0, 0, 1, 1, 1, 2], dtype=torch.long)
    incomp_mask = torch.tensor([1, 1, 1, 1, 1, 1, 0], dtype=torch.bool)

    def forward_call(X, M):
        assert X.shape[0] == 12,\
            'Tensor shape incorrect.'
        return (torch.tensor([1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                             dtype=torch.float),
                None)

    cdiem_trainer.forward = mock.MagicMock(side_effect=forward_call)

    def sample_missing_values_call(batch, **kwargs):
        assert X.shape[0] == 7,\
            'Tensor shape incorrect.'
        return (torch.randn_like(X).unsqueeze(0),
                torch.tensor([[1, 0, 0, 0],
                              [2, 0, 0, 0],
                              [3, 0, 0, 0],
                              [0, 4, 5, 6],
                              [0, 7, 8, 9],
                              [0, 10, 11, 12],
                              [0, 0, 0, 0]],
                             dtype=torch.float))

    cdiem_trainer.sample_missing_values = mock.MagicMock(side_effect=sample_missing_values_call)

    output = cdiem_trainer.update_step((X, M, I, OI, incomp_mask),
                                       only_var=True)
    loss = output['loss']

    true_loss = -torch.tensor(((sum([1, 2, 3]) / 3
                               + (sum([6, 9, 12]) / 3
                                  + sum([7, 10, 14]) / 3
                                  + sum([8, 11, 13]) / 3) / 3)
                               / 2))

    true_entropy = torch.tensor((sum([1, 2, 3]) / 3
                                 + (sum([4, 5, 6]) / 3
                                    + sum([7, 8, 9]) / 3
                                    + sum([10, 11, 12]) / 3) / 3)
                                / 2)
    true_loss -= true_entropy

    assert loss == true_loss,\
        'Loss doesn\'t match.'
