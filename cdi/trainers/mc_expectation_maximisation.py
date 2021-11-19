import argparse
from collections import OrderedDict
from enum import Enum

import numpy as np
import torch
import torch.distributions as distr

import cdi.trainers.complete_mle as cm
from cdi.common.posterior_computation import exact_inference


class PosteriorType(Enum):
    INDEPENDENT = 0
    JOINT = 1
    VARIATIONAL = 2

    def from_string(astring):
        try:
            return PosteriorType[astring.upper()]
        except KeyError:
            raise argparse.ArgumentError()


class MCEM(cm.CompleteMLE):
    """
    Monte Carlo Expectation-Maximisation estimation for FA model.
    """
    def __init__(self, hparams):
        super(MCEM, self).__init__(hparams)

        self.posterior_type = PosteriorType.from_string(
                                self.hparams.mcem.posterior_type)

        if self.num_imputed_copies_scheduler is not None:
            num_imputed_copies = self.num_imputed_copies_scheduler.get_value()
        elif isinstance(self.hparams.data.num_imputed_copies, list):
            num_imputed_copies = self.hparams.data.num_imputed_copies[0]
        else:  # BC
            num_imputed_copies = self.hparams.data.num_imputed_copies

        assert num_imputed_copies == 1,\
            'MCEM shouldn\'t use augmented data!'

    @staticmethod
    def add_model_args(parent_parser):
        parser = super(MCEM, MCEM).add_model_args(parent_parser)

        # MCEM args
        parser._posterior_type = parser.add_argument(
                            '--mcem.posterior_type',
                            type=PosteriorType.from_string, required=True,
                            choices=['independent', 'joint', 'variational'],
                            help=('The type of posterior used.'))
        parser.add_argument('--mcem.num_samples',
                            type=int, required=True,
                            help=('The number of samples from the '
                                  'posterior distribution to '
                                  'approximate the expectation.'))

        return parser

    def sample_univariate_gaussian(self, mean, var, K):
        """
        Sample univariate Gaussian distribution using reparameterisation trick.
        Args:
            mean (float): N Gaussian means
            var (float): N Gaussian variances
            K (int): number of samples for each Gaussian
        Returns:
            x_d (float of shape (K, N, D)): K samples from N*D
                univariate Gaussian distribution
        """
        std = var.pow(0.5)

        # Sample by reparameterisation
        normal = distr.Normal(loc=mean, scale=std)
        x_d = normal.rsample(sample_shape=(K,))

        return x_d

    def sample_joint_gaussian(self, mean, cov, K):
        """
        Sample joint Gaussian distribution using reparameterisation trick.
        Args:
            mean (float): (N, D) Gaussian means
            cov (float): (N, D, D) Gaussian covariances
            K (int): number of samples for each Gaussian
        Returns:
            x_d (float of shape (K, N, D)): K samples from N*D joint Gaussian
        """
        # Sample by reparameterisation
        normal = distr.MultivariateNormal(loc=mean, covariance_matrix=cov)
        x_d = normal.rsample(sample_shape=(K,))

        return x_d

    def sample_missing_values(self, batch, K):
        """
        Sample K examples of the missing values for each x, indicated by M.
        Used for approximating the expectation in ELBO.
        Args:
            batch:
                X (N, D): input batch
                M (N, D): missing data input batch binary mask,
                    1 - observed, 0 - missing.
                I (N): indices of the data
            K (int): number of samples for each missing value
        Returns:
            x_samples (K, N, D): K posterior samples
            entropy (N): analytic entropy, -inf for observed values
        """
        X, M, _ = batch
        # Get posterior distribution of x
        _, _, mean_x, cov_x = exact_inference(
                                        X, M.type_as(X),
                                        F=self.fa_model.factor_loadings,
                                        cov=torch.exp(self.fa_model.log_cov),
                                        mean=self.fa_model.mean)

        # Sample K values for each missing value and detach graph,
        # since we don't want to train the model through this channel
        if self.posterior_type == PosteriorType.INDEPENDENT:
            # Set variance and mean of observed variables to 0
            # And select only the diagonal of the covariance matrix
            # Since we treat the posterior as independent.
            var = torch.diagonal(cov_x, dim1=-2, dim2=-1)
            var[M] = 0.
            # mean_x[M] = 0.

            # Sample from conditional univariate distributions
            x_samples = self.sample_univariate_gaussian(mean_x, var, K)

            # Entropy of the Gaussian distribution computed analytically
            entropy = 1/2 * torch.log(2*np.e*np.pi*var)
            entropy[~M] = 0.
            entropy = entropy.sum(dim=-1)
        elif self.posterior_type == PosteriorType.JOINT:
            # Sample from conditional joint distribution
            x_samples = self.sample_joint_gaussian(mean_x, cov_x, K)
            x_samples[:, M] = 0.

            # TODO: entropy for joint conditional (for logging only)
            # Compute once batched determinant is implemented in PyTorch
            # https://github.com/pytorch/pytorch/issues/7500
            # entropy = (1/2 * torch.log(torch.batched_det(cov_x))
            # + (M_selected==0).sum(dim=1)/2
            # * (1 + torch.log(2*np.pi))).detach()
            # For now just log zeros
            entropy = torch.zeros(X.shape[0])
        else:
            raise ValueError('Invalid posterior type!')

        # Detaching computation graph since parameters are "fixed"
        # when evaluating the expectation, so we don't update through
        # this channel.
        return x_samples.detach(), entropy.detach()

    def training_step(self, batch, batch_idx):
        """
        One iteration of MLE update using MCEM algorithm.
        batch:
            X (N, D): observable variables
            M (N, D): binary missingness mask.
            I (N,): indices of the X samples in the dataset
                (can used for imputation where necessary)
        """
        X, M, I = batch[:3]

        # Sample #num_samples values for each missing value
        X_samples, entropy = self.sample_missing_values(
                                            (X, M, I),
                                            K=self.hparams.mcem.num_samples)

        # Set observed values to the actual values
        X_samples[:, M] = X[M]
        X_samples_shape = X_samples.shape
        X_samples = X_samples.reshape(-1, X_samples.shape[-1])
        M_samples = (M.expand(self.hparams.mcem.num_samples, -1, -1))
        M_samples = M_samples.reshape(-1, X_samples.shape[-1])

        # Evaluate log-probability of data
        fa_log_prob, mis_log_prob = self.forward(X_samples, M_samples)
        if mis_log_prob is not None:
            log_probs = fa_log_prob + mis_log_prob
        else:
            log_probs = fa_log_prob

        # Take average over the K samples approximating the expectation
        log_probs = log_probs.reshape(X_samples_shape[0],
                                      X.shape[0]).mean(dim=0)

        # Compute average log-likelihood
        log_prob = log_probs.mean()
        entropy = entropy.mean()

        loss = -log_prob - entropy

        pbar = {
            'train_log_lik': log_prob.item(),
            'train_entropy': entropy.item(),
        }
        output = OrderedDict({
            'loss': loss,
            'progress_bar': pbar,
        })
        pbar['train_fa_log_prob'] = fa_log_prob.mean().item()
        if mis_log_prob is not None:
            pbar['train_mis_log_prob'] = mis_log_prob.mean().item()
        return output
