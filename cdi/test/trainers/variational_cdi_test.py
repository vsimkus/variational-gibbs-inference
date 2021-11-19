import torch
import jsonargparse
from pytest import fixture

from cdi.trainers.variational_cdi import VarCDI
from cdi.util.arg_utils import convert_namespace


@fixture
def cdi_trainer_individual():
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
            '--model_optim.learning_rate', '1e-3',
            '--model_optim.weight_decay_coeff', '0.0',
            '--var_optim.optimiser', 'amsgrad',
            '--var_optim.learning_rate', '1e-3',
            '--var_optim.weight_decay_coeff', '0.0',
            '--var_optim.epsilon', '1e-8',
            '--data.num_imputed_copies', '2',
            '--data.miss_type', 'MCAR',
            '--data.total_miss', '0.5',
            '--data.batch_size', '2048',
            '--data.filter_fully_missing', 'False']

    argparser = jsonargparse.ArgumentParser()
    argparser = VarCDI.add_model_args(argparser, args)
    args = argparser.parse_args(args)

    args = convert_namespace(args)

    vcdi = VarCDI(args)
    vcdi.configure_optimizers()
    return vcdi


@fixture
def cdi_trainer_shared():
    args = ['--fa_model.input_dim', '4',
            '--fa_model.latent_dim', '2',
            '--variational_model', 'shared',
            '--var_model.input_dim', '4',
            '--var_model.prenet_layer_dims', '10',
            '--var_model.outnet_layer_dims', '4',
            '--var_model.activation', 'lrelu',
            '--var_model.input_missing_vectors', 'missing',
            '--var_model.set_missing_zero', 'False',
            '--var_optim.learning_rate', '0.3',
            '--var_optim.weight_decay_coeff', '0.0',
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
            '--model_optim.learning_rate', '1e-3',
            '--model_optim.weight_decay_coeff', '0.0',
            '--var_optim.optimiser', 'amsgrad',
            '--var_optim.learning_rate', '1e-3',
            '--var_optim.weight_decay_coeff', '0.0',
            '--var_optim.epsilon', '1e-8',
            '--data.num_imputed_copies', '2',
            '--data.miss_type', 'MCAR',
            '--data.total_miss', '0.5',
            '--data.batch_size', '2048',
            '--data.filter_fully_missing', 'False']

    argparser = jsonargparse.ArgumentParser()
    argparser = VarCDI.add_model_args(argparser, args)
    args = argparser.parse_args(args)

    args = convert_namespace(args)

    vcdi = VarCDI(args)
    vcdi.configure_optimizers()
    return vcdi


@fixture
def cdi_trainer_shared3():
    args = ['--fa_model.input_dim', '4',
            '--fa_model.latent_dim', '2',
            '--variational_model', 'shared3',
            '--var_model.input_dim', '4',
            '--var_model.prenet_layer_dims', '10',
            '--var_model.outnet_layer_dims', '4',
            '--var_model.activation', 'lrelu',
            '--var_model.input_missing_vectors', 'missing',
            '--var_model.set_missing_zero', 'False',
            '--var_optim.learning_rate', '0.3',
            '--var_optim.weight_decay_coeff', '0.0',
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
            '--model_optim.learning_rate', '1e-3',
            '--model_optim.weight_decay_coeff', '0.0',
            '--var_optim.optimiser', 'amsgrad',
            '--var_optim.learning_rate', '1e-3',
            '--var_optim.weight_decay_coeff', '0.0',
            '--var_optim.epsilon', '1e-8',
            '--data.num_imputed_copies', '2',
            '--data.miss_type', 'MCAR',
            '--data.total_miss', '0.5',
            '--data.batch_size', '2048',
            '--data.filter_fully_missing', 'False']

    argparser = jsonargparse.ArgumentParser()
    argparser = VarCDI.add_model_args(argparser, args)
    args = argparser.parse_args(args)

    args = convert_namespace(args)

    vcdi = VarCDI(args)
    vcdi.configure_optimizers()
    return vcdi


def run_sample_missing_values_outputs(cdi_trainer):
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [11, 0, 0, 0],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)
    I = torch.tensor([1, 2, 3], dtype=torch.long)

    # Method call
    x_samples, entropy = cdi_trainer.sample_missing_values(
                            (X, M.type_as(X), I),
                            M_selected=M,
                            K=3,
                            sample_all=True)

    x_samples_populated = x_samples != 0
    entropy_populated = entropy != float('-inf')

    assert torch.all(~(x_samples_populated ^ ~M)),\
        'Samples should only be returned for missing values.'

    assert torch.all(~(entropy_populated ^ ~M)),\
        'Entropy should only be returned for missing values.'


def test_sample_missing_values_outputs(cdi_trainer_individual,
                                       cdi_trainer_shared,
                                       cdi_trainer_shared3):
    run_sample_missing_values_outputs(cdi_trainer_individual)
    run_sample_missing_values_outputs(cdi_trainer_shared)
    run_sample_missing_values_outputs(cdi_trainer_shared3)
