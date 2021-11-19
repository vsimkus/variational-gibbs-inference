import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.distributions as distr
import torch.optim as optim

import cdi.trainers.trainer_base as tb
from cdi.models.variational_distribution import (GaussianVarDistr,
                                                 GaussianVarDistrFast,
                                                 SharedGaussianVarDistr,
                                                 SharedGaussianVarDistrSharedMeanVar,
                                                 SharedGaussianVarDistrPartSharedMeanVar,
                                                 FullySharedGaussianVarDistr,
                                                 SharedMixGaussianVarDistr)
from cdi.models.flow_variational_distribution import PiecewiseRationalQuadraticVarDistribution
from cdi.util.arg_utils import parse_bool
from cdi.util.multi_optimiser import MultiOptimiser

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
        print(('No such variational model `{model}`!'))
        sys.exit()


class VarMLEPretraining(tb.TrainerBase):
    """
    MLE pretraining of variational models on observed data.
    """
    def __init__(self, hparams, var_model=None):
        super(VarMLEPretraining, self).__init__(hparams)

        # Prepare variational model
        if var_model is None:
            VarDistr = get_var_distribution_class_from_string(
                                self.hparams.variational_model)
            self.variational_model = VarDistr(self.hparams)
        else:
            self.variational_model = var_model

    @staticmethod
    def add_model_args(parent_parser, args=None):
        parser = super(VarMLEPretraining, VarMLEPretraining)\
            .add_model_args(parent_parser)
        parser.add_argument('--variational_model',
                            type=str, required=True,
                            help='Type of variational model.',
                            choices=['individual', 'individual-f',
                                     'shared', 'shared-meanvar',
                                     'shared-partmeanvar',
                                     'fully-shared',
                                     'mog-shared',
                                     'rq-flow'])
        parser.add_argument('--optim.optimiser',
                            type=str, required=True,
                            choices=['adam', 'amsgrad'])
        parser.add_argument('--optim.learning_rate',
                            type=float, required=True,
                            help=('The learning rate using in Adam '
                                  'optimiser for the var_model.'))
        parser.add_argument('--optim.weight_decay_coeff',
                            type=float, required=True,
                            help=('The weight decay used in Adam '
                                  'optimiser for the var_model.'))
        parser.add_argument('--optim.epsilon',
                            type=float, default=1e-8,
                            help=('Adam optimiser epsilon parameter.'))
        parser.add_argument('--optim.anneal_learning_rate',
                            type=parse_bool, default=False)

        parser.add_argument('--conditional_type', type=str, default='gaussian',
                            choices=['gaussian', 'mog', 'cauchy', 'mix-cauchy', 'studentt'], help=('Univariate conditional types.'))

        # Debugging params
        parser.add_argument('--debug.log_all_log_liks',
                            type=parse_bool, default=False,
                            help=('DEBUG: At each epoch logs log-likelihood'
                                  ' for each dimension.'))
        parser.add_argument('--debug.log_entropies',
                            type=parse_bool, default=False,
                            help=('DEBUG: At each epoch logs conditional entropies'
                                  ' for each dimension.'))

        # Add variational model parameters
        temp_args, _ = parser._parse_known_args(args)
        VarDistr = get_var_distribution_class_from_string(
                        temp_args.variational_model)
        parser = VarDistr.add_model_args(parser)

        return parser

    def configure_optimizers(self):
        # Separate optimisers for the model and the variational distribution
        # since the hyperparameters might need to be different.
        if self.hparams.optim.optimiser == 'adam':
            self.optim = MultiOptimiser(
                var_model_opt=optim.Adam(
                    self.variational_model.parameters(),
                    amsgrad=False,
                    lr=self.hparams.optim.learning_rate,
                    weight_decay=self.hparams.optim.weight_decay_coeff,
                    # NOTE: Changing epsilon parameter to higher values 1e-4
                    # helps to resolve the training instability issues
                    eps=self.hparams.optim.epsilon)
            )
        elif self.hparams.optim.optimiser == 'amsgrad':
            self.optim = MultiOptimiser(
                var_model_opt=optim.Adam(
                    self.variational_model.parameters(),
                    amsgrad=True,
                    lr=self.hparams.optim.learning_rate,
                    weight_decay=self.hparams.optim.weight_decay_coeff,
                    eps=self.hparams.optim.epsilon)
            )
        else:
            sys.exit('No such optimizer for the variational CDI!')

        if self.hparams.optim.anneal_learning_rate:
            schedulers = [optim.lr_scheduler.CosineAnnealingLR(self.optim.optimisers['var_model_opt'], self.hparams.max_epochs, 0)]
            return [self.optim], schedulers

        return self.optim

    def forward(self, X, M, M_selected):
        entropy = None
        if not isinstance(self.variational_model, (PiecewiseRationalQuadraticVarDistribution,)):
            params = self.variational_model(X, M, M_selected, sample_all=True)

            if not hasattr(self.hparams, 'conditional_type') or self.hparams.conditional_type == 'gaussian':
                mean, log_var = params
                model = distr.Normal(loc=mean, scale=torch.exp((1/2) * log_var))

                if hasattr(self.hparams, 'debug') and self.hparams.debug.log_entropies:
                    entropy = 1/2 * (np.log(2*np.e*np.pi) + log_var)
            elif self.hparams.conditional_type == 'cauchy':
                mean, log_var = params
                model = distr.Cauchy(loc=mean, scale=torch.exp((1/2) * log_var))

                if hasattr(self.hparams, 'debug') and  self.hparams.debug.log_entropies:
                    entropy = np.log(4*np.pi) + (1/2)*log_var
            elif self.hparams.conditional_type == 'mog':
                mean, log_var, unnorm_comp_logits = params
                std = torch.exp((1/2) * log_var)

                mixture = distr.Categorical(logits=unnorm_comp_logits)
                normal = rmd.StableNormal(loc=mean, scale=std)
                model = rmd.ReparametrizedMixtureSameFamily(mixture_distribution=mixture,
                                                            component_distribution=normal)

                if hasattr(self.hparams, 'debug') and self.hparams.debug.log_entropies:
                    raise NotImplementedError()
            elif self.hparams.conditional_type == 'mix-cauchy':
                mean, log_var, unnorm_comp_logits = params
                std = torch.exp((1/2) * log_var)

                mixture = distr.Categorical(logits=unnorm_comp_logits)
                normal = distr.Cauchy(loc=mean, scale=std)
                model = rmd.ReparametrizedMixtureSameFamily(mixture_distribution=mixture,
                                                            component_distribution=normal)

                if hasattr(self.hparams, 'debug') and self.hparams.debug.log_entropies:
                    raise NotImplementedError()
            elif self.hparams.conditional_type == 'studentt':
                # Reuse MoG parameters, i.e. the mixture component weight becomes degree-of-freedom
                mean, log_var, log_df = params
                mean = mean.squeeze(-1)
                log_var = log_var.squeeze(-1)
                log_df = log_df.squeeze(-1)
                model = distr.StudentT(df=torch.exp(log_df), loc=mean, scale=torch.exp((1/2) * log_var))

                if hasattr(self.hparams, 'debug') and self.hparams.debug.log_entropies:
                    entropy = model.entropy()

            log_prob = model.log_prob(X)
        elif isinstance(self.variational_model, PiecewiseRationalQuadraticVarDistribution):
            log_prob = self.variational_model.log_prob(X, context=X, M_selected=M_selected)

            if hasattr(self.hparams, 'debug') and self.hparams.debug.log_entropies:
                raise NotImplementedError()

        self.optim.add_run_opt('var_model_opt')
        return log_prob, entropy

    # Training

    def training_step(self, batch, batch_idx):
        """
        One iteration of MLE update on observed data
        batch:
            X (N, D): observable variables
            M (N, D): binary missingness mask.
            I (N,): indices of the X samples in the dataset
                (can used for imputation where necessary)
        """
        if self.num_imputed_copies_scheduler is not None:
            num_imputed_copies = self.num_imputed_copies_scheduler.get_value()
        elif isinstance(self.hparams.data.num_imputed_copies, list):
            num_imputed_copies = self.hparams.data.num_imputed_copies[0]
        else:  # BC
            num_imputed_copies = self.hparams.data.num_imputed_copies

        X, M, I, OI, incomp_mask = batch
        M_not = ~M
        if num_imputed_copies > 1:
            observed_and_full = M * ~incomp_mask[:, None]
            N_full = torch.sum(observed_and_full, dim=0)
            N_incomp = torch.sum(M ^ observed_and_full, dim=0)
            N_true_per_dim = N_full + torch.true_divide(N_incomp, num_imputed_copies)
            # Prevent division by zero
            N_true_per_dim[N_true_per_dim == 0] = 1

            log_probs, entropy = self.forward(X, M, M_selected=M_not)
            log_probs = log_probs*M

            log_probs[incomp_mask] /= num_imputed_copies
            log_probs = log_probs.sum(dim=0) / N_true_per_dim
            log_prob = log_probs.sum()

            if hasattr(self.hparams, 'debug') and self.hparams.debug.log_entropies:
                entropy[M_not] = 0.
                entropy[incomp_mask] /= num_imputed_copies
                entropy = entropy.sum(dim=0) / N_true_per_dim
        else:
            num_obs_per_dim = M.sum(dim=0)
            # Prevent division by zero
            num_obs_per_dim[num_obs_per_dim == 0] = 1

            # Compute log likelihood
            log_probs, entropy = self.forward(X, M, M_selected=M_not)
            log_probs = log_probs*M
            log_probs = log_probs.sum(dim=0) / num_obs_per_dim

            log_prob = log_probs.sum()

            if hasattr(self.hparams, 'debug') and self.hparams.debug.log_entropies:
                entropy[M_not] = 0.
                entropy = entropy.sum(dim=0) / num_obs_per_dim

        pbar = {
            'train_log_lik': log_prob.item()
        }
        if hasattr(self.hparams, 'debug') and self.hparams.debug.log_all_log_liks:
            for i in range(log_probs.shape[0]):
                pbar[f'train_log_lik_{i}'] = log_probs[i].item()
        if hasattr(self.hparams, 'debug') and self.hparams.debug.log_entropies:
            for i in range(log_probs.shape[0]):
                pbar[f'train_entropy_{i}'] = entropy[i].item()
        output = OrderedDict({
            'loss': -log_prob,
            'progress_bar': pbar,
        })
        return output

    def validation_step(self, batch, batch_idx):
        """
        One iteration of likelihood evaluation over the validation set.
        batch:
            X (N, D): observable variables
            M (N, D): binary missingness mask, used to determine the
                log-likelihood of the observed variables only.
            I (N,): indices of the X samples in the dataset
                (can used for imputation where necessary)
        """
        X, M = batch[:2]
        M_not = ~M
        num_obs_per_dim = M.sum(dim=0)
        # Prevent division by zero
        num_obs_per_dim[num_obs_per_dim == 0] = 1

        # Compute log-likelihood on fully visible data
        log_probs, entropy = self.forward(X, M, M_selected=M)
        log_probs[M_not] = 0.
        log_probs = log_probs.sum(dim=0) / num_obs_per_dim

        log_prob = log_probs.sum()

        if hasattr(self.hparams, 'debug') and self.hparams.debug.log_entropies:
            entropy[M_not] = 0.
            entropy = entropy.sum(dim=0) / num_obs_per_dim

        output = OrderedDict({
            'val_loss': -log_prob,
            'val_log_lik': log_prob.item()
        })
        if hasattr(self.hparams, 'debug') and self.hparams.debug.log_all_log_liks:
            for i in range(log_probs.shape[0]):
                output[f'val_log_lik_{i}'] = log_probs[i].item()
        if hasattr(self.hparams, 'debug') and self.hparams.debug.log_entropies:
            for i in range(log_probs.shape[0]):
                output[f'val_entropy_{i}'] = entropy[i].item()
        return output

    #
    # Hooks
    #

    def on_epoch_start(self):
        super().on_epoch_start()
        self.variational_model.on_epoch_start()

    def training_epoch_end(self, outputs):
        results = super().training_epoch_end(outputs)

        # Add epoch-level stats
        # TODO: handle this in the Variational model class
        results['log']['cum_var_calls'] = self.variational_model.cum_batch_size_called if hasattr(self.variational_model, 'cum_batch_size_called') else 0
        return results

    #
    # Test
    #
    @staticmethod
    def add_test_args(parent_parser):
        parser = super(VarMLEPretraining, VarMLEPretraining).add_test_args(parent_parser)
        # NOTE: Hacky way to store reference to this argument
        # So we can modify it later in sub-classes.
        parser._eval_type_arg = parser.add_argument(
                                        '--eval_type',
                                        type=str, required=True,
                                        choices=['eval_posterior'],
                                        help=('Available evaluations.'))

        parser.add_argument('--output_suffix', type=str,
                            help='Append a suffix to the end of the output.')

        return parser

    def test_step(self, batch, batch_idx):
        # Make sure we're running the correct evaluation!
        assert self.hparams.test.eval_type == 'eval_posterior',\
            f'eval_type=`{self.hparams.test.eval_type}` not supported!'

        # Make sure we're running the correct evaluation!
        X, M, _ = batch
        # Compute variational posterior
        mean, log_var = self.variational_model(
                                        X, M,
                                        torch.zeros_like(X, dtype=torch.bool),
                                        sample_all=True)

        # Log missing data posteriors
        self.logger.accumulate_tensors(
                                'posterior_params',
                                var_post_mean=mean.cpu().detach(),
                                var_post_log_var=(log_var.cpu().detach()),
                                M=M.cpu().detach())
        return {}

    def test_epoch_end(self, outputs):
        # Make sure we're running the correct evaluation!
        assert self.hparams.test.eval_type == 'eval_posterior',\
            f'eval_type=`{self.hparams.test.eval_type}` not supported!'

        suffix = ''
        if hasattr(self.hparams.test, 'output_suffix') and self.hparams.test.output_suffix is not None:
            suffix = f'_{self.hparams.test.output_suffix}'

        # Save posterior parameters
        self.logger.save_accumulated_tensors('posterior_params', 'test'+suffix)

        return {}

    # @classmethod
    # def load_from_checkpoint(
    #         cls,
    #         checkpoint_path: str,
    #         *args,
    #         map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
    #         hparams_file: Optional[str] = None,
    #         tags_csv: Optional[str] = None,  # backward compatible, todo: remove in v0.9.0
    #         **kwargs
    # ):
    #     model = super().load_from_checkpoint(checkpoint_path, *args, map_location, hparams_file, tags_csv, **kwargs)

    #     if not hasattr(model.hparams.config,
