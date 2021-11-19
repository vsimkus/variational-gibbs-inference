import numpy as np
import torch
import torch.distributions as distr

from cdi.common.posterior_computation import exact_gibbs_inference
from cdi.trainers.cdi import CDI


class PosteriorCDI(CDI):
    """
    Maximum likelihood estimation (MLE) model using
    cumulative data imputation (CDI) algorithm.
    Analytical posterior implementation.
    """

    # Training

    def compute_univariate_posteriors(self, batch, M_selected, sample_all=None):
        """
        Compute univariate posteriors for all values that are missing
        in M_selected.
        """
        X = batch[0]

        _, _, mean_x, cov_x = exact_gibbs_inference(
                                        X, M_selected.type_as(X),
                                        F=self.fa_model.factor_loadings,
                                        cov=torch.exp(self.fa_model.log_cov),
                                        mean=self.fa_model.mean)

        # Set variance and mean of observed variables to 0
        var = torch.diagonal(cov_x, dim1=-2, dim2=-1)
        var[M_selected] = 0.
        mean_x[M_selected] = 0.

        return mean_x, var

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

    def sample_missing_values(self, batch, M_selected, K, sample_all=None):
        """
        Sample K examples of the missing values for each x, indicated by
        M_selected. Used for approximating the expectation in ELBO.
        Args:
            batch:
                X (N, D): input batch
                M (N, D): missing data input batch binary mask,
                    1 - observed, 0 - missing.
                I (N): indices of the data
            M_selected (N, D): a subset of selected M, for which we want to get
                                the variational distribution parameters.
            K (int): number of samples for each missing value
        Returns:
            x_samples (K, N, D): K samples for all N*D missing values,
                0s for observed values
            entropy (N, D): analytic entropy for all N*D distributions,
                -inf for observed values
        """
        # Get posterior distribution of x
        mean_x, var = self.compute_univariate_posteriors(batch,
                                                         M_selected,
                                                         sample_all)

        # Sample from conditional univariate distributions
        x_samples = self.sample_univariate_gaussian(mean_x, var, K)

        # Entropy of the Gaussian distribution computed analytically
        entropy = 1/2 * torch.log((2*np.e*np.pi)*var)
        # entropy *= (~M_selected).float()

        # Detaching computation graph since parameters are "fixed"
        # when evaluating the expectation, so we don't update through
        # this channel.
        return x_samples.detach(), entropy.detach()

    def sample_imputation_values(self, batch, M_selected, sample_all=None):
        """
        Produces one sample for each missing value for imputation.
        Args:
            batch:
                X (N, D): input batch
                M (N, D): missing data input batch binary mask,
                    1 - observed, 0 - missing.
                I (N): indices of the data
            M_selected (N, D): a subset of selected M, for which we want to get
                                the variational distribution parameters.
        Returns:
            samples (N, D): 1 sample for each missing value
        """
        # Get posterior distribution of x
        mean_x, var = self.compute_univariate_posteriors(batch,
                                                         M_selected,
                                                         sample_all)

        if self.hparams.cdi.sample_imputation:
            # Sample from conditional univariate distributions
            x_samples = self.sample_univariate_gaussian(mean_x, var, K=1)
        else:
            # Otherwise impute with posterior means
            x_samples = mean_x

        # Detach graph, since we don't want to train the model through
        # this channel
        return x_samples.squeeze().detach()

    #
    # Test
    #
    @staticmethod
    def add_test_args(parent_parser):
        parser = super(PosteriorCDI, PosteriorCDI).add_test_args(parent_parser)
        parser._eval_type_arg.choices.append('eval_posterior')
        return parser

    def test_step(self, batch, batch_idx):
        # Make sure we're running the correct evaluation!
        if self.hparams.test.eval_type == 'eval_posterior':
            X, M = batch[:2]
            mean, var = self.compute_univariate_posteriors(
                        batch,
                        M_selected=torch.zeros_like(X, dtype=torch.bool),
                        sample_all=True)

            # Log missing data posteriors
            self.logger.accumulate_tensors('posterior_params',
                                           anal_post_mean=mean.cpu().detach(),
                                           anal_post_log_var=(torch.log(var)
                                                              .cpu().detach()),
                                           M=M.cpu().detach())
            return {}
        else:
            return super().test_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        suffix = ''
        if hasattr(self.hparams.test, 'output_suffix') and self.hparams.test.output_suffix is not None:
            suffix = f'_{self.hparams.test.output_suffix}'
        # Make sure we're running the correct evaluation!
        if self.hparams.test.eval_type == 'eval_posterior':
            # Save posterior parameters
            self.logger.save_accumulated_tensors('posterior_params', 'test'+suffix)

            return {}
        else:
            return super().test_epoch_end(outputs)
