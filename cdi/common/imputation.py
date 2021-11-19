import sys
from collections import defaultdict

import numpy as np
import torch

import cdi.trainers.trainer_base as tb
import cdi.trainers.variational_cdi as vcdi
from cdi.models.regression import UnivariateRegression
from cdi.trainers.regression_fn_trainer \
    import evaluate_imputation_loss as regression_evaluate_imputation_loss
from cdi.trainers.regression_fn_trainer import \
    (impute_dimension_with_regression_predictions,
     load_models,
     send_model_to_device)
from cdi.util.data.data_augmentation_dataset import collate_augmented_samples
from cdi.util.regression_sampler import RegressionSampler
from cdi.util.utils import construct_experiment_name


# TODO: refactor this to work on other datasets too.

def accumulate_feature_observations(dataset):
    """
    Collects all *observed* values for each dimension into a dictionary.
    """
    # Find all observed variables in each dimension
    observations = defaultdict(list)
    X, M, _ = dataset.unaugmented_data()
    for x, m in zip(X, M):
        for d in range(0, x.shape[-1]):
            # Check if the dimension is observed
            if m[d] == 1:
                observations[d].append(x[d])

    return observations


def compute_empirical_mean_and_std_of_observations(dataset):
    """
    Computes the empirical mean and standard deviation of *observed*
    values only.
    """
    observations = accumulate_feature_observations(dataset)

    # Find empirical means of observable dimensions
    means = np.zeros(dataset[0][0].shape[-1])
    stds = np.zeros(dataset[0][0].shape[-1])
    for d, obs in observations.items():
        means[d] = np.mean(obs)
        stds[d] = np.std(obs, ddof=1)

    return means, stds


def impute_with_mean(dataset):
    """
    Computes the empirical mean of each dimension, using only observed values.
    Then, replaces the missing values with the empirical mean for that
    dimension.
    """
    # Compute observed mean for each dimension
    # Use observed unaugmented data
    X, M = dataset.unaugmented_data()[:2]
    count_observed = M.sum(axis=0)
    sum_observed = (X * M).sum(axis=0)
    means = sum_observed / count_observed

    # Replace missing values with empirical means
    X, M, I = dataset.get_all_data()[:3]
    M_not = ~M
    X[M_not] = (means * M_not)[M_not]
    dataset[I] = X

    return means


def impute_with_empirical_distribution_sample(dataset):
    """
    Computes the empirical distribution of each dimension, using only observed
    values. Then, replaces the missing values with random sample from the
    empirical distribution.
    """
    X_emp, M_emp = dataset.unaugmented_data()[:][:2]
    X, M, I = dataset.get_all_data()[:3]

    M_not = ~M
    count_missing_per_dim = (M_not).sum(axis=0)
    # Impute each dimension
    for i in range(M.shape[-1]):
        if count_missing_per_dim[i] == 0:
            continue
        samples = np.random.choice(X_emp[:, i][M_emp[:, i]],
                                   size=count_missing_per_dim[i])
        X[M_not[:, i], i] = samples

    # Set imputed data
    dataset[I] = X


def impute_with_regression_predictions(dataset, regression_args, device,
                                       repeat=1, root_dir='.', exp_group='',
                                       evaluate_on_observed=False,
                                       eval_postprocess_fn=None,
                                       disable_print=True):
    """
    Impute the missing values using trained regression models.
    Args:
        dataset (iterable): dataset to be imputed
        regression_args (dict): args used for training the regression model
            (layer dimensionality, etc.)
        device (torch.device): device to perform computations on
            (e.g. cuda, or cpu)
        repeat (int): number of times to perform the imputation.
            If >1 then the order of imputation is shuffled
    """
    # Initialise dataset by completing it
    trainer_base = tb.TrainerBase(regression_args)
    trainer_base.initialise_dataset(regression_args, dataset)

    # Prepare regression models
    print('Loading regression models.')
    models = []
    for i in range(regression_args.num_regression_fns):
        models.append(UnivariateRegression(regression_args))

    # Create Path for the experiment
    experiment_name = construct_experiment_name(regression_args)

    # Load model parameters
    load_models(models,
                experiment_name,
                root_dir=root_dir,
                exp_group=exp_group,
                disable_print=disable_print)

    # Send models to relevant device
    for i, model in enumerate(models):
        models[i] = send_model_to_device(model, device)

    # Impute each missing variable with deterministic imputation from
    # regression function
    print('Imputing with regression predictions {} times.'.format(repeat))
    for i in range(repeat):
        dims = np.arange(dataset[0][0].shape[-1])
        # If we repeat imputation more than 1 time, then shuffle the
        # imputation order.
        if repeat > 1:
            np.random.shuffle(dims)

        observed_loss = []
        for dim in dims:
            if evaluate_on_observed:
                with torch.no_grad():
                    observed_data = torch.utils.data.DataLoader(
                                    dataset,
                                    batch_size=len(dataset),
                                    collate_fn=collate_augmented_samples,
                                    num_workers=0,
                                    sampler=RegressionSampler(
                                                dataset,
                                                dim=dim,
                                                shuffle=False,
                                                omit_missing=True))
                    # There will only be one pass through this dataloader,
                    # since the batchsize is the entire data
                    for X, M, I, OI, incomp_mask in observed_data:
                        # Targets are the d-th dimension
                        Y_d = X[:, dim]

                        # Remove d-th feature from X
                        feature_indices = np.arange(X.shape[-1])
                        feature_indices = feature_indices[feature_indices != dim]
                        X_d = X[:, feature_indices]

                        M_d = None
                        if regression_args.input_missing_vectors:
                            M_d = M[:, feature_indices]

                        # run a validation iter
                        metrics = regression_evaluate_imputation_loss(
                                            X=X_d, Y=Y_d,
                                            model=models[dim],
                                            device=device,
                                            M=M_d,
                                            postprocess_fn=eval_postprocess_fn)
                        observed_loss.append(metrics['loss'])

            # Impute missing values at d-th dimension.
            impute_dimension_with_regression_predictions(
                dataset=dataset,
                model=models[dim],
                dim=dim,
                batch_size=regression_args.data.batch_size,
                device=device,
                input_missing_vectors=regression_args.input_missing_vectors)
        return observed_loss


def impute_with_variational_means(dataset, checkpoint, device,
                                  repeat=1,
                                  switch_imputations=None):
    """
    Impute the missing values using variational distribution means.
    Args:
        dataset (iterable): dataset to be imputed
        checkpoint (str): Path to the checkpoint.
        device (torch.device): device to perform computations on
            (e.g. cuda, or cpu)
        repeat (int): number of times to perform the imputation.
            If >1 then the order of imputation is shuffled
    """
    # TODO
    # Prepare variational model
    print('Loading variational model.')
    var_cdi = vcdi.VarCDI.load_from_checkpoint(checkpoint)
    variational_args = var_cdi.args

    # Pre-imputation method
    if variational_args.imputation == 'mean':
        print('Performing mean imputation before variational mean imputation.')
        impute_with_mean(dataset)
    elif variational_args.imputation == 'empirical_distribution_samples':
        print('Imputation with samples from empirical distribution before variational mean imputation.')
        impute_with_empirical_distribution_sample(dataset)
    else:
        print('No such imputation method for imputing before variational mean imputation!')
        sys.exit()

    # variational_args.cdi.sample_all = False

    # We want to impute with variational means
    variational_args.cdi.sample_imputation = False

    print('Imputing with variational mean predictions {} times.'
          .format(repeat))
    sample_all = variational_args.sample_all
    for i in range(repeat):
        if sample_all:
            if switch_imputations is not None and i == switch_imputations:
                sample_all = not sample_all

            # Create dataloaders
            data = torch.utils.data.DataLoader(
                                dataset,
                                batch_size=variational_args.data.batch_size,
                                collate_fn=collate_augmented_samples,
                                num_workers=0,
                                shuffle=False)

            with torch.no_grad():
                for X, M, I in data:
                    M_selected = M

                    M = M.type_as(X)
                    X = X.to(device=device)
                    M = M.to(device=device)

                    X_imp = var_cdi.sample_imputation_values(
                                                X, M,
                                                M_selected=M_selected,
                                                sample_all=sample_all)
                    X_imp[M_selected] = X[M_selected]
                    dataset[I] = X_imp.cpu()
        else:
            dims = np.arange(dataset[0][0].shape[-1])
            # If we repeat imputation more than 1 time, then shuffle
            # the imputation order.
            if repeat > 1:
                np.random.shuffle(dims)

            for dim in dims:
                # Create dataloader
                data = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=len(dataset),
                            collate_fn=collate_augmented_samples,
                            num_workers=0,
                            sampler=RegressionSampler(
                                dataset,
                                dim=dim,
                                shuffle=False,
                                # Iterate over missing-values at
                                # d-th dimension only
                                omit_missing=False))

                with torch.no_grad():
                    for X, M, I in data:
                        M = M.type_as(X)
                        X = X.to(device=device)
                        M = M.to(device=device)

                        # Impute only missing variables of current dimension
                        observed_indices = np.arange(X.shape[-1])
                        observed_indices = observed_indices[observed_indices != dim]
                        M_selected = M.clone().byte()
                        M_selected[:, observed_indices] = 1

                        X_imp = var_cdi.sample_imputation_values(
                                                    X, M,
                                                    M_selected=M_selected,
                                                    sample_all=sample_all)
                        if len(X_imp.shape) == 1:
                            X_imp[M_selected.squeeze()] = X[M_selected]
                        else:
                            X_imp[M_selected] = X[M_selected]
                        dataset[I] = X_imp.cpu()


def impute_with_posterior_mean(dataset, posterior_fn, F, cov, mu):
    X, M, I = dataset[:]
    X = torch.tensor(X)
    M = torch.tensor(M, dtype=torch.float32)
    _, _, mu_x, _ = posterior_fn(X, M, F, cov, mu)

    dataset[I] = (X*M + mu_x*(1 - M)).detach().numpy()
