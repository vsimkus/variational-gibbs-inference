import os
import sys
import time

import numpy as np
import torch
import torch.distributions as distr
import torch.optim as optim

import cdi.trainers.cdi as tcdi
from cdi.common.posterior_computation import (exact_gibbs_inference,
                                              exact_inference)
from cdi.models.variational_distribution import (GaussianVarDistr,
                                                 GaussianVarDistrFast,
                                                 SharedGaussianVarDistr,
                                                 SharedGaussianVarDistrSharedMeanVar,
                                                 SharedGaussianVarDistrPartSharedMeanVar,
                                                 FullySharedGaussianVarDistr,
                                                 SharedMixGaussianVarDistr)
from cdi.models.flow_variational_distribution import PiecewiseRationalQuadraticVarDistribution
from cdi.trainers.var_mle_pretraining import VarMLEPretraining
from cdi.util.utils import construct_experiment_name, find_best_model_epoch_from_fs
from cdi.util.stats_utils import load_statistics
from cdi.util.arg_utils import parse_bool
from cdi.util.analysis.eval import compute_univariate_gaussian_kl_div
import cdi.submodules.torch_reparametrised_mixture_distribution as rmd


def get_var_distribution_class_from_string(model):
    if model == 'individual':
        return GaussianVarDistr
    elif model == 'individual-f':
        return GaussianVarDistrFast
    elif model == 'shared':
        return SharedGaussianVarDistr
    elif model == 'shared-meanvar':
        return SharedGaussianVarDistrSharedMeanVar
    elif model == 'shared-partmeanvar':
        return SharedGaussianVarDistrPartSharedMeanVar
    elif model == 'fully-shared':
        return FullySharedGaussianVarDistr
    elif model == 'mog-shared':
        return SharedMixGaussianVarDistr
    elif model == 'rq-flow':
        return PiecewiseRationalQuadraticVarDistribution
    else:
        print((f'No such variational model `{model}`!'))
        sys.exit()


class VarCDI(tcdi.CDI):
    """
    Maximum likelihood estimation (MLE) model using
    cumulative data imputation (CDI) algorithm.
    Variational implementation.
    """
    def __init__(self, hparams, root=None, model=None, var_model=None):
        super(VarCDI, self).__init__(hparams, model=model)

        # Prepare variational model
        if var_model is None:
            VarDistr = get_var_distribution_class_from_string(
                            self.hparams.variational_model)
            self.variational_model = VarDistr(self.hparams)
            self.variational_model.reset_parameters()
        else:
            self.variational_model = var_model

        if hasattr(self.hparams, 'var_pretrained_model') and self.hparams.var_pretrained_model is not None:
            self.load_pretrained_model(root)

    def load_pretrained_model(self, root):
        seed_stamp = construct_experiment_name(self.hparams).split('/')[-1]
        # Get the best model epoch from FS
        last_model_path = os.path.join(
                            'trained_models',
                            self.hparams.exp_group,
                            self.hparams.var_pretrained_model,
                            seed_stamp,
                            'saved_models')
        if root is not None:
            last_model_path = os.path.join(root, last_model_path)
        chkpt_epoch = find_best_model_epoch_from_fs(last_model_path)
        # Get best model
        checkpoint = f'_ckpt_epoch_{chkpt_epoch}.ckpt'
        best_model_path = os.path.join(
                            'trained_models',
                            self.hparams.exp_group,
                            self.hparams.var_pretrained_model,
                            seed_stamp,
                            'saved_models',
                            checkpoint)
        if root is not None:
            best_model_path = os.path.join(root, best_model_path)
        print(f'Loading {best_model_path}')
        pretrained_model = VarMLEPretraining.load_from_checkpoint(best_model_path)
        # Save pretrained model hparams for initialisation
        self.hparams.pretrained_model_hparams = pretrained_model.hparams
        # Set current model to the pretrained model
        pretrained_model_new = VarMLEPretraining(self.hparams)
        pretrained_model_new.load_state_dict(pretrained_model.state_dict())
        self.variational_model = pretrained_model_new.variational_model

    @staticmethod
    def add_model_args(parent_parser, args=None):
        parser = super(VarCDI, VarCDI).add_model_args(parent_parser, args)
        parser.add_argument('--variational_model',
                            type=str, required=True,
                            help='Type of variational model.',
                            choices=['individual', 'individual-f',
                                     'shared', 'shared-meanvar',
                                     'shared-partmeanvar',
                                     'fully-shared',
                                     'mog-shared',
                                     'rq-flow'])
        parser.add_argument('--var_optim.optimiser',
                            type=str, required=True,
                            choices=['adam', 'amsgrad'])
        parser.add_argument('--var_optim.learning_rate',
                            type=float, required=True,
                            help=('The learning rate using in Adam '
                                  'optimiser for the var_model.'))
        parser.add_argument('--var_optim.weight_decay_coeff',
                            type=float, required=True,
                            help=('The weight decay used in Adam '
                                  'optimiser for the var_model.'))
        parser.add_argument('--var_optim.epsilon',
                            type=float, default=1e-8,
                            help=('Adam optimiser epsilon parameter.'))
        parser.add_argument('--var_optim.anneal_learning_rate',
                            type=parse_bool, default=False)
        parser.add_argument('--var_optim.anneal_steps',
                            type=int, default=None)
        parser.add_argument('--var_pretrained_model',
                            type=str, required=False,
                            help=('The name of the pretrained variational'
                                  ' model.'))
        parser.add_argument('--var_optim.grad_clip_val', type=float,
                            default=0.0,
                            help=('Variational model gradient norm clipping value.'))
        parser.add_argument('--var_optim.debug.error_if_nonfinite_norm', default=False,
                            type=parse_bool,
                            help=('Throws error if the norm of the gradients is not finite.'))
        parser.add_argument('--model_optim.grad_clip_val', type=float,
                            default=0.0,
                            help=('Model gradient norm clipping value.'))
        parser.add_argument('--model_optim.debug.error_if_nonfinite_norm', default=False,
                            type=parse_bool,
                            help=('Throws error if the norm of the gradients is not finite.'))

        parser.add_argument('--data.pre_imp_num_imputation_steps',
                            type=int,
                            help=('Number of Gibbs sampling updates in pretraining.'))
        parser.add_argument('--data.pre_imp_clip', type=parse_bool,
                            default=False, help=('Whether to clip Gibbs pre-imputation values to min/max.'))
        parser.add_argument('--data.pre_imp_reject', type=parse_bool,
                            default=False, help=('Whether to reject Gibbs pre-imputation values outside min/max.'))

        parser.add_argument('--cdi.conditional_type', type=str, default='gaussian',
                            choices=['gaussian', 'mog', 'cauchy', 'mix-cauchy', 'studentt'], help=('Univariate conditional types.'))

        # Debugging params
        parser.add_argument('--cdi.debug.log_imp_var_params',
                            type=parse_bool, default=False,
                            help=('DEBUG: Logs the variational parameters that'
                                  ' were used for imputation.'))
        parser.add_argument('--cdi.debug.log_var_kldivs',
                            type=parse_bool, default=False,
                            help=('DEBUG: Logs the kl-divergence with the '
                                  'current model for each dimension.'))

        # Add variational model parameters
        temp_args, _ = parser._parse_known_args(args)
        VarDistr = get_var_distribution_class_from_string(
                        temp_args.variational_model)
        parser = VarDistr.add_model_args(parser)

        return parser

    def initialise_dataset(self, hparams, dataset):
        metrics = None
        if (hparams.data.pre_imputation == 'systematic_gibbs_sampling'):
            init_start_time = time.time()

            # First impute the data with the same method as used
            # in the pretraining
            super().initialise_dataset(
                hparams.pretrained_model_hparams, dataset)

            print('Variational Gibbs sampling imputation.')
            pre_imp_clip = False
            pre_imp_reject = False
            if hasattr(self.hparams.data, 'pre_imp_clip'):
                pre_imp_clip = self.hparams.data.pre_imp_clip
            if hasattr(self.hparams.data, 'pre_imp_reject'):
                pre_imp_reject = self.hparams.data.pre_imp_reject

            self.impute_with_systematic_gibbs_samples(
                            dataset,
                            num_imputation_steps=hparams.data.pre_imp_num_imputation_steps,
                            clip_imp_values=pre_imp_clip,
                            reject_imp_values=pre_imp_reject)

            metrics = {
                'init_time': [time.time() - init_start_time],
                'cum_var_calls': [self.variational_model.cum_batch_size_called],
                'stage': ['initialise_dataset']
            }
        else:
            metrics = super().initialise_dataset(hparams, dataset)

        if ('pretrained_model_hparams' in hparams
                and hparams.pretrained_model_hparams is not None):
            # Load and add pretraining statistics
            seed_stamp = construct_experiment_name(hparams).split('/')[-1]
            pretrain_log_path = os.path.join(
                                'trained_models',
                                hparams.exp_group,
                                hparams.var_pretrained_model,
                                seed_stamp,
                                'logs')
            pre_stats = load_statistics(pretrain_log_path, 'summary.csv')

            if metrics is not None:
                metrics['init_time'].insert(0, np.sum(pre_stats['train_time']))
                if 'cum_var_calls' not in metrics:
                    metrics['cum_var_calls'] = [0]*len(metrics['init_time'])
                    metrics['cum_var_calls'][0] = np.sum(pre_stats['cum_var_calls'])
                else:
                    metrics['cum_var_calls'].insert(0, np.sum(pre_stats['cum_var_calls']))
                metrics['stage'].insert(0, 'pre_train')
            else:
                metrics = {
                    'init_time': [np.sum(pre_stats['train_time'])],
                    'cum_var_calls': [np.sum(pre_stats['cum_var_calls'])],
                    'stage': ['pre_train']
                }

        return metrics

    def configure_optimizers(self):
        optimiser = super().configure_optimizers()
        # Separate optimisers for the model and the variational
        # distribution since the hyperparameters might need to be
        # different.
        if self.hparams.var_optim.optimiser == 'adam':
            var_opt = optim.AdamW(
                    self.variational_model.parameters(),
                    amsgrad=False,
                    lr=self.hparams.var_optim.learning_rate,
                    weight_decay=self.hparams.var_optim.weight_decay_coeff,
                    # NOTE: Changing epsilon parameter to higher values 1e-4
                    # helps to resolve the training instability issues
                    eps=self.hparams.var_optim.epsilon)
        elif self.hparams.var_optim.optimiser == 'amsgrad':
            var_opt = optim.AdamW(
                    self.variational_model.parameters(),
                    amsgrad=True,
                    lr=self.hparams.var_optim.learning_rate,
                    weight_decay=self.hparams.var_optim.weight_decay_coeff,
                    eps=self.hparams.var_optim.epsilon)
        else:
            sys.exit('No such optimizer for the variational CDI!')

        if isinstance(optimiser, tuple):
            optimiser[0][0].add_optimisers(var_model_opt=var_opt)
            if self.hparams.var_optim.anneal_learning_rate:
                max_steps = (self.hparams.var_optim.anneal_steps
                             if hasattr(self.hparams.var_optim, 'anneal_steps') and self.hparams.var_optim.anneal_steps is not None
                             else self.hparams.max_epochs)
                optimiser[1].append(optim.lr_scheduler.CosineAnnealingLR(var_opt, max_steps, 0))
        else:
            optimiser.add_optimisers(var_model_opt=var_opt)

            if self.hparams.var_optim.anneal_learning_rate:
                max_steps = (self.hparams.var_optim.anneal_steps
                             if hasattr(self.hparams.var_optim, 'anneal_steps') and self.hparams.var_optim.anneal_steps is not None
                             else self.hparams.max_epochs)
                schedulers = [optim.lr_scheduler.CosineAnnealingLR(var_opt, max_steps, 0)]
                optimiser = (optimiser, schedulers)

        # self.optim = optimiser
        return optimiser

    # Training

    def compute_univariate_posteriors(self, batch, M_selected, sample_all=None):
        """
        Compute univariate posteriors for all values that are missing
        in M_selected.
        """
        X, M = batch[:2]
        # Cast type to X so can compute products
        M = M.type_as(X)
        params = self.variational_model(X, M, M_selected, sample_all)
        var = torch.exp(params[1])
        params = params[:1] + (var,) + params[2:]

        return params

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

    def rsample_univariate_mog(self, mean, var, unnorm_comp_logits, K, estimate_entropy=False):
        """
        Sample univariate MoG distribution using reparameterisation trick.
        Args:
            mean (float): N Gaussian means
            var (float): N Gaussian variances
            unnorm_comp_logits (float): N unnormalised mixture component logits
            K (int): number of samples for each Gaussian
        Returns:
            x_d (float of shape (K, N, D)): K samples from N*D
                univariate MoG distribution
        """
        std = var.pow(0.5)

        mixture = distr.Categorical(logits=unnorm_comp_logits)
        normal = rmd.StableNormal(loc=mean, scale=std)
        mog = rmd.ReparametrizedMixtureSameFamily(mixture_distribution=mixture,
                                                  component_distribution=normal)

        # # Sample by reparameterisation
        x_d = mog.rsample(sample_shape=(K,))

        if estimate_entropy:
            # No analytical solution for MoG entropy, so estimate
            entropy = -mog.log_prob(x_d).sum(dim=0)
            return x_d, entropy

        return x_d

    def sample_univariate_mog(self, mean, var, unnorm_comp_logits, K):
        """
        Sample univariate MoG distribution without reparameterisation trick.
        Args:
            mean (float): N Gaussian means
            var (float): N Gaussian variances
            unnorm_comp_logits (float): N unnormalised mixture component logits
            K (int): number of samples for each Gaussian
        Returns:
            x_d (float of shape (K, N, D)): K samples from N*D
                univariate MoG distribution
        """
        std = var.pow(0.5)

        mixture = distr.Categorical(logits=unnorm_comp_logits)
        normal = rmd.StableNormal(loc=mean, scale=std)
        mog = rmd.ReparametrizedMixtureSameFamily(mixture_distribution=mixture,
                                                  component_distribution=normal)

        # # Sample without reparametrisation is cheaper
        x_d = mog.sample(sample_shape=(K,))

        return x_d

    def rsample_univariate_mix_cauchy(self, median, var, unnorm_comp_logits, K, estimate_entropy=False):
        """
        Sample univariate Mixture of Cauchy distribution using reparameterisation trick.
        Args:
            median (float): N Cauchy medians
            var (float): N Cauchy variances
            unnorm_comp_logits (float): N unnormalised mixture component logits
            K (int): number of samples for each Gaussian
        Returns:
            x_d (float of shape (K, N, D)): K samples from N*D
                univariate MoC distribution
        """
        std = var.pow(0.5)

        mixture = distr.Categorical(logits=unnorm_comp_logits)
        cauchy = distr.Cauchy(loc=median, scale=std)
        moc = rmd.ReparametrizedMixtureSameFamily(mixture_distribution=mixture,
                                                  component_distribution=cauchy)

        # # Sample by reparameterisation
        x_d = moc.rsample(sample_shape=(K,))

        if estimate_entropy:
            # No analytical solution for MoG entropy, so estimate
            entropy = -moc.log_prob(x_d).sum(dim=0)
            return x_d, entropy

        return x_d

    def sample_univariate_mix_cauchy(self, median, var, unnorm_comp_logits, K):
        """
        Sample univariate Mixture of Cauchydistribution without reparameterisation trick.
        Args:
            median (float): N Cauchy medians
            var (float): N Cauchy variances
            unnorm_comp_logits (float): N unnormalised mixture component logits
            K (int): number of samples for each Gaussian
        Returns:
            x_d (float of shape (K, N, D)): K samples from N*D
                univariate MoC distribution
        """
        std = var.pow(0.5)

        mixture = distr.Categorical(logits=unnorm_comp_logits)
        cauchy = distr.Cauchy(loc=median, scale=std)
        moc = rmd.ReparametrizedMixtureSameFamily(mixture_distribution=mixture,
                                                  component_distribution=cauchy)

        # # Sample without reparametrisation is cheaper
        x_d = moc.sample(sample_shape=(K,))

        return x_d

    def rsample_univariate_studentt(self, mean, var, log_df, K, estimate_entropy=False):
        """
        Sample univariate Student's-T distribution using reparameterisation trick.
        Args:
            mean (float): N loc
            var (float): N var
            log_df (float): N log-degrees of freedom
            K (int): number of samples
        Returns:
            x_d (float of shape (K, N, D)): K samples from N*D
                univariate Student's-T distribution
        """
        # Reuse MoG parameters, i.e. the mixture component weight becomes degree-of-freedom
        std = var.pow(0.5).squeeze(-1)
        mean = mean.squeeze(-1)
        df = torch.exp(log_df).squeeze(-1)

        studentt = distr.StudentT(df=df, loc=mean, scale=std)

        # Sample by reparameterisation
        x_d = studentt.rsample(sample_shape=(K,))

        if estimate_entropy:
            # No analytical solution for MoG entropy, so estimate
            entropy = studentt.entropy()
            return x_d, entropy

        return x_d

    def sample_univariate_studentt(self, mean, var, log_df, K):
        """
        Sample univariate Student's-T distribution without reparameterisation trick.
        Args:
            mean (float): N loc
            var (float): N var
            log_df (float): N log-degrees of freedom
            K (int): number of samples
        Returns:
            x_d (float of shape (K, N, D)): K samples from N*D
                univariate Student's-T distribution
        """
        # Reuse MoG parameters, i.e. the mixture component weight becomes degree-of-freedom
        std = var.pow(0.5).squeeze(-1)
        mean = mean.squeeze(-1)
        df = torch.exp(log_df).squeeze(-1)

        studentt = distr.StudentT(df=df, loc=mean, scale=std)

        # Sample without reparametrisation
        x_d = studentt.sample(sample_shape=(K,))

        return x_d

    def sample_missing_values(self, batch, M_selected, K, sample_all):
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
        X, M, _ = batch

        if not isinstance(self.variational_model, (PiecewiseRationalQuadraticVarDistribution,)):
            params = self.compute_univariate_posteriors(batch,
                                                        M_selected,
                                                        sample_all=sample_all)

            # Compute mean and log-variance for each chosen missing value
            # given the other variables.
            if not hasattr(self.hparams.cdi, 'conditional_type') or self.hparams.cdi.conditional_type == 'gaussian':
                mean, var = params

                # Sample K values for each missing value
                x_samples = self.sample_univariate_gaussian(mean, var, K)

                # Entropy of the Gaussian distribution computed analytically
                entropy = 1/2 * torch.log((2*np.e*np.pi)*var)
                # Set entropies for observed variables to 0
                # entropy *= (~M_selected).float()
            elif self.hparams.cdi.conditional_type == 'mog':
                mean, var, unnorm_comp_logits = params
                # Sample K values for each missing value
                x_samples, entropy = self.rsample_univariate_mog(mean, var, unnorm_comp_logits, K, estimate_entropy=True)
            elif self.hparams.cdi.conditional_type == 'mix-cauchy':
                median, var, unnorm_comp_logits = params
                # Sample K values for each missing value
                x_samples, entropy = self.rsample_univariate_mix_cauchy(median, var, unnorm_comp_logits, K, estimate_entropy=True)
            elif self.hparams.cdi.conditional_type == 'studentt':
                loc, var, log_df = params
                # Sample K values for each missing value
                x_samples, entropy = self.rsample_univariate_studentt(loc, var, log_df, K, estimate_entropy=True)
        elif isinstance(self.variational_model, PiecewiseRationalQuadraticVarDistribution):
            x_samples, log_prob = self.variational_model.sample_and_log_prob(num_samples=K, context=X, M_selected=M_selected)
            # (B, K, *) -> (K, B, *)
            x_samples = x_samples.permute(1, 0, *((-1,)*(len(X.shape)-1)))
            log_prob = log_prob.permute(1, 0, *((-1,)*(len(X.shape)-1)))

            entropy = -log_prob.sum(dim=0)

        # Make sure the optimiser for var model runs
        self.optim.add_run_opt('var_model_opt')
        return x_samples, entropy

    def sample_imputation_values(self, batch, M_selected, sample_all):
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
        X, M, I = batch

        if not isinstance(self.variational_model, (PiecewiseRationalQuadraticVarDistribution,)):
            # Compute the posterior parameters for each missing variable
            params = self.compute_univariate_posteriors(batch,
                                                        M_selected,
                                                        sample_all=sample_all)

            if hasattr(self.hparams.cdi, 'debug') and hasattr(self.hparams.cdi.debug, 'log_imp_var_params') and self.hparams.cdi.debug.log_imp_var_params:
                self.logger.accumulate_tensors('var_params',
                                            mean=params[0].cpu().detach(),
                                            log_var=torch.log(params[1]).cpu().detach(),
                                            I=I.cpu().detach())

            if not hasattr(self.hparams.cdi, 'conditional_type') or self.hparams.cdi.conditional_type == 'gaussian':
                mean, var = params
                if self.hparams.cdi.sample_imputation:
                    # Sample 1 value for each missing value for imputation
                    x_samples = self.sample_univariate_gaussian(mean, var, K=1)
                else:
                    # Otherwise impute with posterior means
                    x_samples = mean
            elif self.hparams.cdi.conditional_type == 'mog':
                if self.hparams.cdi.sample_imputation:
                    mean, var, unnorm_comp_logits = params
                    # Sample 1 value for each missing value for imputation
                    x_samples = self.sample_univariate_mog(mean, var, unnorm_comp_logits, K=1)
                else:
                    raise NotImplementedError
            elif self.hparams.cdi.conditional_type == 'mix-cauchy':
                if self.hparams.cdi.sample_imputation:
                    median, var, unnorm_comp_logits = params
                    # Sample 1 value for each missing value for imputation
                    x_samples = self.sample_univariate_mix_cauchy(median, var, unnorm_comp_logits, K=1)
                else:
                    raise NotImplementedError
            elif self.hparams.cdi.conditional_type == 'studentt':
                if self.hparams.cdi.sample_imputation:
                    loc, var, log_df = params
                    # Sample 1 value for each missing value for imputation
                    x_samples = self.sample_univariate_studentt(loc, var, log_df, K=1)
                else:
                    raise NotImplementedError
        elif isinstance(self.variational_model, PiecewiseRationalQuadraticVarDistribution):
            if self.hparams.cdi.sample_imputation:
                x_samples = self.variational_model.sample(num_samples=1, context=X, M_selected=M_selected)
                x_samples = x_samples.permute(1, 0, *((-1,)*(len(X.shape)-1)))
            else:
                raise NotImplementedError

        return x_samples.detach().squeeze()

    # Hooks

    def on_after_backward(self):
        if hasattr(self.hparams.var_optim, 'grad_clip_val') and self.hparams.var_optim.grad_clip_val > 0:
            error_if_nonfinite_norm = (hasattr(self.hparams.var_optim, 'debug')
                                       and hasattr(self.hparams.var_optim.debug, 'error_if_nonfinite_norm')
                                       and self.hparams.var_optim.debug.error_if_nonfinite_norm)
            total_norm = torch.nn.utils.clip_grad_norm_(self.variational_model.parameters(),
                                                        self.hparams.var_optim.grad_clip_val,
                                                        norm_type=2.0)
                                                        #    error_if_nonfinite=error_if_nonfinite_norm)
            if error_if_nonfinite_norm and (total_norm.isnan() or total_norm.isinf()):
                raise RuntimeError('Non-finite norm encountered!')
        if hasattr(self.hparams.model_optim, 'grad_clip_val') and self.hparams.model_optim.grad_clip_val > 0:
            error_if_nonfinite_norm = (hasattr(self.hparams.model_optim, 'debug')
                                       and hasattr(self.hparams.model_optim.debug, 'error_if_nonfinite_norm')
                                       and self.hparams.model_optim.debug.error_if_nonfinite_norm)
            total_norm = torch.nn.utils.clip_grad_norm_(self.fa_model.parameters(),
                                                        self.hparams.model_optim.grad_clip_val,
                                                        norm_type=2.0)
                                                        #    error_if_nonfinite=error_if_nonfinite_norm)
            if error_if_nonfinite_norm and (total_norm.isnan() or total_norm.isinf()):
                raise RuntimeError('Non-finite norm encountered!')

    def on_epoch_start(self):
        super().on_epoch_start()
        self.variational_model.on_epoch_start()

    def on_epoch_end(self):
        super().on_epoch_end()
        # Save the accumulated imputation var. parameter tensors
        if hasattr(self.hparams.cdi, 'debug') and hasattr(self.hparams.cdi.debug, 'log_imp_var_params') and self.hparams.cdi.debug.log_imp_var_params:
            self.logger.save_accumulated_tensors('var_params', self.current_epoch)

    def training_epoch_end(self, outputs):
        results = super().training_epoch_end(outputs)

        # Add epoch-level stats
        # TODO: handle this in the Variational model class
        results['log']['cum_var_calls'] = self.variational_model.cum_batch_size_called if hasattr(self.variational_model, 'cum_batch_size_called') else 0

        if hasattr(self.hparams.cdi, 'debug') and hasattr(self.hparams.cdi.debug, 'log_var_kldivs') and self.hparams.cdi.debug.log_var_kldivs:
            with torch.no_grad():
                train_kldiv = self.compute_var_kl_div(self.train_dataloader())
                for i in range(train_kldiv.shape[0]):
                    results['log'][f'train_kldiv_{i}'] = train_kldiv[i].item()

                val_kldiv = self.compute_var_kl_div(self.val_dataloader())
                for i in range(val_kldiv.shape[0]):
                    results['log'][f'val_kldiv_{i}'] = val_kldiv[i].item()

        return results

    def compute_var_kl_div(self, dataloader):
        kldivs = []
        Ms = []
        if isinstance(dataloader, list):
            dataloader = dataloader[-1]

        for e, batch in enumerate(dataloader):
            # Transfer data to GPU
            if self.hparams.gpus is not None:
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            batch = self.transfer_batch_to_device(batch, device)

            # Compute parameters
            mean, var = self.compute_univariate_posteriors(batch[:3], batch[1], sample_all=None)

            # Compute analytical posterior
            _, _, mean_x, cov_x = exact_gibbs_inference(
                                    batch[0], batch[1].float(),
                                    F=self.fa_model.factor_loadings,
                                    cov=torch.exp(self.fa_model.log_cov),
                                    mean=self.fa_model.mean)
            # Set variance and mean of observed variables to 0
            var_x = torch.diagonal(cov_x, dim1=-2, dim2=-1)

            kldiv = compute_univariate_gaussian_kl_div(mean, torch.log(var),
                                                        mean_x, torch.log(var_x))
            kldiv[batch[1]] = 0
            Ms.append(batch[1])
            kldivs.append(kldiv)

        kldiv = torch.cat(kldivs)
        M = torch.cat(Ms)
        return kldiv.sum(dim=0) / M.sum(dim=0)

    #
    # Test
    #
    @staticmethod
    def add_test_args(parent_parser):
        parser = super(VarCDI, VarCDI).add_test_args(parent_parser)
        parser._eval_type_arg.choices.append('eval_posterior')
        parser._eval_type_arg.choices.append('eval_gibbs_sampling')
        parser.add_argument('--num_gibbs_passes', type=int,
                            help=('The number of gibbs passes used in '
                                  '`eval_gibbs_sampling`.'))
        parser.add_argument('--num_gibbs_chains', type=int,
                            help=('The number of gibbs chains for each '
                                  'sample.'))
        parser.add_argument('--sample_idx', type=int, nargs='+',
                            help=('Indices of samples used in '
                                  '`eval_gibbs_sampling`.'))
        parser._eval_type_arg.choices.append('eval_gibbs_sampling2')
        # parser.add_argument('--num_imputed_copies', type=int,
        #                     help=('number of imputation chains.'))
        parser.add_argument('--num_gibbs_steps', type=int,
                            help=('number of gibbs steps.'))
        parser.add_argument('--job_id', type=int, default=None)
        return parser

    def test_step(self, batch, batch_idx):
        # Make sure we're running the correct evaluation!
        if self.hparams.test.eval_type == 'eval_posterior':
            X, M = batch[:2]
            # Compute variational posterior
            mean, var = self.compute_univariate_posteriors(
                        batch,
                        M_selected=torch.zeros_like(X, dtype=torch.bool),
                        sample_all=True)
            # Compute analytical posterior
            _, _, mean_x, cov_x = exact_gibbs_inference(
                                X, torch.zeros_like(X),
                                F=self.fa_model.factor_loadings,
                                cov=torch.exp(self.fa_model.log_cov),
                                mean=self.fa_model.mean)
            # Set variance and mean of observed variables to 0
            var_x = torch.diagonal(cov_x, dim1=-2, dim2=-1)

            # Log missing data posteriors
            self.logger.accumulate_tensors(
                                    'posterior_params',
                                    var_post_mean=mean.cpu().detach(),
                                    var_post_log_var=(torch.log(var)
                                                      .cpu().detach()),
                                    anal_post_mean=mean_x.cpu().detach(),
                                    anal_post_log_var=(torch.log(var_x)
                                                       .cpu().detach()),
                                    M=M.cpu().detach())
            return {}
        elif self.hparams.test.eval_type == 'eval_gibbs_sampling':
            X, M, I = batch[:3]
            for idx in self.hparams.test.sample_idx:
                index_in_batch = torch.where(I == idx)[0]
                if index_in_batch.shape[0] > 0:
                    X_i = X[index_in_batch, :]
                    M_i = M[index_in_batch, :]
                    I_i = I[index_in_batch]

                    log_prob, _ = self.forward(X_i, M_i)

                    mu_x, sigma_x = exact_inference(
                                       X_i, M_i.type_as(X_i),
                                       F=self.fa_model.factor_loadings,
                                       cov=torch.exp(self.fa_model.log_cov),
                                       mean=self.fa_model.mean)[2:]
                    # Squeeze the batch dimension
                    mu_x, sigma_x = mu_x.squeeze(), sigma_x.squeeze()
                    missing_idx = torch.where(~M_i.squeeze())[0]
                    # Select the posterior dimensions only
                    mu_x = mu_x[missing_idx]
                    sigma_x = sigma_x[:, missing_idx]
                    sigma_x = sigma_x[missing_idx, :]

                    # Initialise chains
                    X_i_chains = X_i.repeat(self.hparams.test.num_gibbs_chains, 1)
                    M_i_chains = M_i.repeat(self.hparams.test.num_gibbs_chains, 1)
                    I_i_chains = I_i.repeat(self.hparams.test.num_gibbs_chains, 1)

                    # Sample the missing values
                    X_i_chains, M_i_chains, I_i_chains = self.systematic_gibbs_sampling(
                                (X_i_chains, M_i_chains, I_i_chains),
                                num_passes=self.hparams.test.num_gibbs_passes)

                    X_samples = X_i_chains[:, missing_idx]
                    sample_cov = np.cov(X_samples.numpy(), rowvar=False)
                    sample_mean = np.mean(X_samples.numpy(), axis=0)

                    # Log the samples
                    self.logger.log_tensors(
                        epoch='test',
                        logname=f'samples_{I[index_in_batch[0].numpy().item()]}',
                        X=X_i.squeeze(), M=M_i.squeeze(), I=I_i.squeeze(),
                        log_prob=log_prob, post_mean=mu_x, post_cov=sigma_x,
                        X_samples=X_i_chains,
                        sample_mean=sample_mean, sample_cov=sample_cov)

            return {}
        elif self.hparams.test.eval_type == 'eval_gibbs_sampling2':
            self.hparams.cdi.imputation_delay = -1
            self.impute_batch(batch, stage='test', l=-1,
                              num_imputation_steps=self.hparams.test.num_gibbs_steps)

            name = 'gibbs_samples' if self.hparams.test.job_id is None else f'gibbs_samples_{self.hparams.test.job_id }'
            self.logger.accumulate_tensors(
                name,
                X=batch[0].detach().cpu(),
                M=batch[1].detach().cpu(),
                orig_I=batch[3].detach().cpu())
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
        elif self.hparams.test.eval_type == 'eval_gibbs_sampling':
            return {}
        elif self.hparams.test.eval_type == 'eval_gibbs_sampling2':
            name = 'gibbs_samples' if self.hparams.test.job_id is None else f'gibbs_samples_{self.hparams.test.job_id }'
            self.logger.save_accumulated_tensors(name, 'test'+suffix)

            return {}
        else:
            return super().test_epoch_end(outputs)
