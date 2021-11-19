import argparse
from collections import OrderedDict
from enum import Enum

import torch

import cdi.trainers.complete_mle as cm
from cdi.common.posterior_computation import (exact_inference,
                                              compute_xm_z_covariance)


class PosteriorType(Enum):
    INDEPENDENT = 0
    JOINT = 1

    def from_string(astring):
        try:
            return PosteriorType[astring.upper()]
        except KeyError:
            raise argparse.ArgumentError()


class EM(cm.CompleteMLE):
    """
    Expectation-Maximisation estimation for FA model.
    """
    def __init__(self, hparams):
        super().__init__(hparams)

        self.posterior_type = PosteriorType.from_string(
                                self.hparams.em.posterior_type)

        if self.num_imputed_copies_scheduler is not None:
            num_imputed_copies = self.num_imputed_copies_scheduler.get_value()
        elif isinstance(self.hparams.data.num_imputed_copies, list):
            num_imputed_copies = self.hparams.data.num_imputed_copies[0]
        else:  # BC
            num_imputed_copies = self.hparams.data.num_imputed_copies

        assert num_imputed_copies == 1,\
            'EM shouldn\'t use augmented data!'

    @staticmethod
    def add_model_args(parent_parser):
        parser = super(EM, EM).add_model_args(parent_parser)

        # EM args
        parser._posterior_type = parser.add_argument(
                            '--em.posterior_type',
                            type=PosteriorType.from_string, required=True,
                            choices=['independent', 'joint'],
                            help=('The type of posterior used.'))

        return parser

    # Setup

    def configure_optimizers(self):
        return None

    def setup(self, stage):
        out = super().setup(stage)

        # Make sure that the batch size is larger than the data
        if stage == 'fit':
            assert len(self.train_dataset) <= self.hparams.data.batch_size, \
                'For EM the batch size must be >= to the size of the dataset!'
            # assert len(self.val_dataset) <= self.hparams.data.batch_size, \
            #     'For EM the batch size must be >= to the size of the dataset!'

        return out

    # Training

    def backward(self, use_amp, loss, optimizer, optimizer_idx):
        # Do nothing on backward for EM (or could set the updated values here
        # instead of the training loop)
        pass

    def compute_updated_parameters_for_complete_data(self, batch):
        X = batch[0]

        # Model params
        F = self.fa_model.factor_loadings
        Psi = torch.exp(self.fa_model.log_cov)
        mean = self.fa_model.mean

        # Compute mean
        mean_up = mean = X.mean(dim=0)

        # Efficiently invert diagonal matrix
        Psi_inv = (1/Psi)

        # Dif
        d = (X-mean)

        latent_dim = F.shape[1]

        cov_z = torch.inverse(torch.eye(latent_dim, device=X.device) + F.T @ (Psi_inv.unsqueeze(-1) * F))
        mean_z = (cov_z @ F.T * Psi_inv).unsqueeze(0) @ d.unsqueeze(-1)

        H = cov_z + torch.mean(mean_z @ mean_z.transpose(-1, -2), dim=0)
        A = torch.mean(d.unsqueeze(-1) @ mean_z.transpose(-1, -2), dim=0)
        V = (d.unsqueeze(-1) @ d.unsqueeze(-2)).mean(dim=0)

        F_up = A @ torch.inverse(H)
        Psi_up = V - 2 * (F_up @ A.T) + F_up @ H @ F_up.T
        # Select the diagonal only
        Psi_up = torch.diagonal(Psi_up)
        log_cov_up = torch.log(Psi_up)

        return mean_up, F_up, log_cov_up

    def compute_updated_parameters_with_missing(self, batch):
        X, M = batch[:2]
        M_not = ~M

        # Model params
        F = self.fa_model.factor_loadings
        Psi = torch.exp(self.fa_model.log_cov)
        mean = self.fa_model.mean

        # Get posterior distribution of x and z
        mean_z, cov_z, mean_x, cov_x = exact_inference(
                                        X, M.type_as(X),
                                        F=F,
                                        cov=Psi,
                                        mean=mean)
        # Set variances-covariances and means of observed rows/cols to 0
        cov_x = cov_x * (M_not.unsqueeze(-1) * M_not.unsqueeze(1))
        mean_x = mean_x * M_not

        # Impute missing values with posterior means
        X = X.clone()
        X[M_not] = mean_x[M_not]

        if self.posterior_type == PosteriorType.INDEPENDENT:
            # If we want conditionally-independent imputations (conditional on x_o) then set the
            # off-diagonal covariance elements to zero
            # TODO: Use the fact that cov_x is diagonal to make some of the below computations more efficient
            cov_x = torch.diag_embed(torch.diagonal(cov_x, dim1=1, dim2=2))

            # We must then recompute cov_z and cov_xm_z, which are both affected by the assumption

            # Compute common
            Q = F.T @ torch.inverse(F @ F.T + torch.diag(Psi))

            # Compute posterior of z covariance for fully-observed data
            sigma_z = torch.inverse(torch.eye(F.shape[-1], device=X.device) + F.T @ torch.diag(1/Psi) @ F)
            # Recompute the posterior covariance of z using the assumption on x_m
            cov_z = sigma_z + Q.unsqueeze(0) @ cov_x @ Q.T.unsqueeze(0)

            # Get covariance between x_m and z conditioned on x_o given the assumption on x_m
            cov_xm_z = ((cov_x + mean_x.unsqueeze(-1) @ X.unsqueeze(1)) @ Q.T
                        - mean_x.unsqueeze(-1) @ mean.unsqueeze(0) @ Q.T
                        - mean_x.unsqueeze(-1) @ mean_z.transpose(1, 2))
        elif self.posterior_type == PosteriorType.JOINT:
            # Get covariance between x_m and z conditioned on x_o
            cov_xm_z = compute_xm_z_covariance(M, self.fa_model.factor_loadings, cov_z)
        else:
            raise ValueError('Invalid posterior type!')

        # Compute updated mean
        mean_up = X.mean(dim=0)

        # Compute common matrices
        d = X - mean_up
        A = (cov_xm_z + d.unsqueeze(-1) @ mean_z.transpose(1, 2)).mean(dim=0)
        H = (cov_z + mean_z @ mean_z.transpose(1, 2)).mean(dim=0)

        # Compute updated factor loadings
        F_up = A @ torch.inverse(H)

        # Compute updated Psi
        V = (cov_x + d.unsqueeze(-1) @ d.unsqueeze(1)).mean(dim=0)
        Psi_up = V - 2 * (F_up @ A.T) + F_up @ H @ F_up.T
        # Select the diagonal only
        Psi_up = torch.diagonal(Psi_up)
        log_cov_up = torch.log(Psi_up)

        return mean_up, F_up, log_cov_up

    def training_step(self, batch, batch_idx):
        # Gradients not needed for EM
        with torch.no_grad():
            if self.hparams.data.total_miss > 0.:
                mean_up, F_up, log_cov_up = self.compute_updated_parameters_with_missing(batch)
            else:
                mean_up, F_up, log_cov_up = self.compute_updated_parameters_for_complete_data(batch)

            self.fa_model.mean.data = mean_up
            self.fa_model.factor_loadings.data = F_up
            self.fa_model.log_cov.data = log_cov_up

            # Compute the observed-data log-probability
            X, M = batch[:2]
            # log_marginal_prob does not fit in memory for FA-Frey, so instead just compute
            # the log-predictive quantity
            # if self.hparams.data.total_miss > 0.:
            #     log_probs = self.fa_model.log_marginal_prob(X, M)
            # else:
            log_probs = self.fa_model(X, M)
            log_prob = log_probs.mean()

        pbar = {
            'train_log_lik': log_prob.item()
            # 'train_entropy': entropy.item(),
        }
        output = OrderedDict({
            'loss': -log_prob,
            'progress_bar': pbar,
        })
        return output

    # Validation

    def validation_step(self, batch, batch_idx):
        X, M = batch[:2]

        # Compute the observed-data log-probability
        with torch.no_grad():
            # log_probs = self.fa_model.log_marginal_prob(X, M)
            log_probs = self.fa_model(X, M)
            log_prob = log_probs.mean()

        output = OrderedDict({
            'val_loss': -log_prob,
            'val_log_lik': log_prob.item()
        })
        return output
