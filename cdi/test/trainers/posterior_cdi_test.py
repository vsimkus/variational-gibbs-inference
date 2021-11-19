import torch
import jsonargparse
from pytest import fixture

from cdi.trainers.posterior_cdi import PosteriorCDI


@fixture
def cdi_trainer_gibbs():
    args = ['--fa_model.input_dim', '4',
            '--fa_model.latent_dim', '2',
            '--cdi.update_components', 'MISSING',
            '--cdi.update_comp_schedule', '0', '10', '20',
            '--cdi.update_comp_schedule_values', '-1', '1', '-1',
            '--cdi.num_samples', '3',
            '--cdi.imputation_delay', '0',
            '--cdi.num_imputation_steps', '1',
            '--cdi.imputation_comp_schedule', '0', '5', '25',
            '--cdi.imputation_comp_schedule_values', '-1', '1', '-1',
            '--cdi.imp_acceptance_check_schedule', '0',
            '--cdi.imp_acceptance_check_schedule_values', 'False',
            '--cdi.entropy_coeff', '1.0']

    argparser = jsonargparse.ArgumentParser()
    argparser = PosteriorCDI.add_model_args(argparser)
    args = argparser.parse_args(args)

    return PosteriorCDI(args)


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
