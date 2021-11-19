import os
import shutil
from collections import OrderedDict

import torch
import torch.optim as optim

from cdi.trainers.trainer_base import TrainerBase
from cdi.trainers.complete_mle import CompleteMLE, get_mis_model_class_from_string
from cdi.trainers.variational_cdi import VarCDI
from cdi.trainers.mcimp import MCIMP
from cdi.util.multi_optimiser import MultiOptimiser
from cdi.util.utils import construct_experiment_name, set_random, find_best_model_epoch_from_fs
from cdi.util.arg_utils import parse_bool
from cdi.util.data.data_augmentation_dataset import collate_augmented_samples
from cdi.models.missingness_model import MultinomialLogRegMisModel
from cdi.models.factor_analysis import FactorAnalysis
from cdi.models.fc_vae import FC_VAE
from cdi.models.fc_partial_vae import FC_PartialVAE
from cdi.models.fc_hivae import FC_HIVAE
from cdi.models.fc_mcvae import FC_MCVAE, FC_MCVAE_BVGI, FC_MCVAE_BVGI_2G


class RefitEncoderOnTest(CompleteMLE):
    """
    Fit decoder and fit encoder on test data.
    """
    def __init__(self, hparams):
        super().__init__(hparams)

        # Construct path to trained model
        old_exp_name = construct_experiment_name(self.hparams)
        old_exp_name = '/'.join([self.hparams.model_name,
                                 old_exp_name.split('/', maxsplit=1)[1]])
        if self.hparams.model_checkpoint == 'best':
            ckpt = find_best_model_epoch_from_fs(os.path.join('trained_models', self.hparams.exp_group, old_exp_name, 'saved_models'))
            model_path = os.path.join('trained_models', self.hparams.exp_group,
                                      old_exp_name, 'saved_models',
                                      f'_ckpt_epoch_{ckpt}.ckpt')
        else:
            model_path = os.path.join('trained_models', self.hparams.exp_group,
                                      old_exp_name, 'saved_models',
                                      f'{self.hparams.model_checkpoint}.ckpt')

        # Load the model
        if not hasattr(self.hparams, 'trainer') or self.hparams.trainer == 'complete':
            model = CompleteMLE.load_from_checkpoint(model_path)
        elif self.hparams.trainer == 'vcdi':
            model = VarCDI.load_from_checkpoint(model_path)
        elif self.hparams.trainer == 'mcimp':
            model = MCIMP.load_from_checkpoint(model_path)
        self.fa_model = model.fa_model
        # Set parameters
        self.fa_model.set_hparams(self.hparams)
        # decoder freeze
        self.fa_model.freeze_decoder()

        if hasattr(self.hparams, 'reset_encoder') and self.hparams.reset_encoder:
            self.fa_model.reset_encoder()
        elif hasattr(self.hparams, 'reset_first_encoder_layer') and self.hparams.reset_first_encoder_layer:
            self.fa_model.reset_encoder_first_layer()

        if self.hparams.init_local_vi:
            self.fa_model.init_local_params(data_count=self.hparams.local_vi_data_count)

        if self.hparams.data.obs_zero_mean:
            # Copy the observed mean from training folder
            old_obs_mean_path = os.path.join(os.path.dirname(model_path), '../',
                                             'logs', 'tensors',
                                             'obs_mean_0.npz')
            new_obs_mean_path = os.path.join('trained_models',
                                             self.hparams.exp_group,
                                             construct_experiment_name(self.hparams),
                                             'logs', 'tensors',
                                             'obs_mean_0.npz')
            try:
                os.makedirs(os.path.dirname(new_obs_mean_path))
            except FileExistsError:
                pass
            try:
                shutil.copyfile(old_obs_mean_path, new_obs_mean_path)
            except shutil.SameFileError:
                pass

    @staticmethod
    def add_model_args(parent_parser, args=None):
        parser = TrainerBase.add_model_args(parent_parser)

        parser.add_argument('--trainer', type=str,
                            choices=['complete', 'vcdi', 'mcimp'],
                            help='Which trainer was used to fit the model.')
        parser.add_argument('--model_name', type=str,
                            help='Name of the trained model.')
        parser.add_argument('--model_checkpoint', type=str,
                            help='Checkpoint.')
        parser.add_argument('--test.test_seed',
                            type=int, required=True,
                            help=('The seed for the _test_ '
                                  'experiment.'))
        parser.add_argument('--test.init_missing_data',
                            type=parse_bool, default=False,
                            help=('Whether to impute the missing values.'))
        parser.add_argument('--test.num_imputed_copies', type=int, default=1,
                            help=('Number of imputation chains for test data.'))
        parser.add_argument('--test.generate_test_data', default=False,
                            type=parse_bool, help=('Whether to generate test data from the density model.'))
        parser.add_argument('--model_optim.learning_rate',
                            type=float, required=True,
                            help=('The learning rate using in Adam '
                                  'optimiser for the fa_model.'))
        parser.add_argument('--model_optim.weight_decay_coeff',
                            type=float, required=True,
                            help=('The weight decay used in Adam '
                                  'optimiser for the fa_model.'))
        parser.add_argument('--missingness_model',
                            type=str, required=False,
                            help='Type of missingness model.',
                            choices=['multinomial-log-regression'])
        parser.add_argument('--mis_model_optim.learning_rate',
                            type=float, required=False,
                            help=('The learning rate using in Adam '
                                  'optimiser for the mis_model.'))
        parser.add_argument('--mis_model_optim.weight_decay_coeff',
                            type=float, required=False,
                            help=('The weight decay used in Adam '
                                  'optimiser for the mis_model.'))

        parser.add_argument('--density_model', type=str, default='factor-analysis',
                            choices=['factor-analysis', 'fc-vae', 'fc-pvae', 'fc-hivae',
                                     'fc-mcvae', 'fc-mcvae-bvgi', 'fc-mcvae-bvgi2'])

        parser.add_argument('--init_local_vi', default=False,
                            type=parse_bool,
                            help=('Initialises local VI parameters.'))
        parser.add_argument('--local_vi_data_count', required=False,
                            type=int,
                            help=('Number of datapoints if local-vi is used.'))

        parser.add_argument('--reset_first_encoder_layer', default=False,
                            # Might be useful for the zero-masking models
                            type=parse_bool, help=('Whether to reset encoder input layers or not.'))
        parser.add_argument('--reset_encoder', default=False,
                            type=parse_bool, help=('Reset the whole encoder'))

        temp_args, _ = parser._parse_known_args(args)
        # For BC
        if (not hasattr(temp_args, 'density_model')) or temp_args.density_model == 'factor-analysis':
            # Add FA model parameters
            parser = FactorAnalysis.add_model_args(parser)
        elif temp_args.density_model == 'fc-vae':
            parser = FC_VAE.add_model_args(parser)
        elif temp_args.density_model == 'fc-pvae':
            parser = FC_PartialVAE.add_model_args(parser)
        elif temp_args.density_model == 'fc-hivae':
            parser = FC_HIVAE.add_model_args(parser)
        elif temp_args.density_model == 'fc-mcvae':
            parser = FC_MCVAE.add_model_args(parser)
        elif temp_args.density_model == 'fc-mcvae-bvgi':
            parser = FC_MCVAE_BVGI.add_model_args(parser)
        elif temp_args.density_model == 'fc-mcvae-bvgi2':
            parser = FC_MCVAE_BVGI_2G.add_model_args(parser)

        if 'missingness_model' in vars(temp_args):
            MisModel = get_mis_model_class_from_string(
                                temp_args.missingness_model)
            parser = MisModel.add_model_args(parser)

        return parser

    # Setup
    def configure_optimizers(self):
        self.optim = MultiOptimiser(
                fa_model_opt=optim.Adam(
                    self.fa_model.parameters(),
                    amsgrad=False,
                    lr=self.hparams.model_optim.learning_rate,
                    weight_decay=self.hparams.model_optim.weight_decay_coeff))

        if hasattr(self, 'mis_model'):
            self.optim.add_optimisers(
                mis_model_opt=optim.Adam(
                    self.mis_model.parameters(),
                    amsgrad=False,
                    lr=self.hparams.mis_model_optim.learning_rate,
                    weight_decay=self.hparams.mis_model_optim.weight_decay_coeff))

        return self.optim

    def setup(self, stage):
        out = super().setup(stage='fit')
        # We want to load the test data.
        out = super().setup(stage='test')

        if stage == 'fit':
            # Workaround for this particular model to know the patterns in the data
            # It is not necessary for stage==`test` because the details should be
            # persisted in the model
            if (hasattr(self, 'mis_model')
                    and isinstance(self.mis_model, MultinomialLogRegMisModel)):
                # NOTE: test-data
                self.mis_model.initialise(self.test_dataset.dataset.dataset.patterns)

        return out

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                                self.test_dataset,  # NOTE: test-data
                                batch_size=self.hparams.data.batch_size,
                                collate_fn=collate_augmented_samples,
                                num_workers=2,
                                shuffle=True)

    def on_train_start(self):
        # Called at the beginning of training before sanity check.
        # Set random seed before training
        set_random(self.hparams.test.test_seed)

    def forward(self, X, M, I=None):
        if self.training:
            self.optim.add_run_opt('fa_model_opt')
            if hasattr(self, 'mis_model'):
                self.optim.add_run_opt('mis_model_opt')
        fa_log_probs = self.fa_model(X, M, I)

        mis_log_probs = None
        if hasattr(self, 'mis_model'):
            mis_log_probs = self.mis_model(X, M)

        return fa_log_probs, mis_log_probs

    def training_step(self, batch, batch_idx):
        """
        One iteration of MLE update on complete(d) data
        batch:
            X (N, D): observable variables
            M (N, D): binary missingness mask.
            I (N,): indices of the X samples in the dataset
                (can used for imputation where necessary)
        """
        X, M, I, OI, incomp_mask = batch

        if self.num_imputed_copies_scheduler is not None:
            num_imputed_copies = self.num_imputed_copies_scheduler.get_value()
        elif isinstance(self.hparams.data.num_imputed_copies, list):
            num_imputed_copies = self.hparams.data.num_imputed_copies[0]
        else:  # BC
            num_imputed_copies = self.hparams.data.num_imputed_copies

        # Compute log likelihood
        fa_log_prob, mis_log_prob = self.forward(X, M, I)
        if mis_log_prob is not None:
            log_prob = fa_log_prob + mis_log_prob
        else:
            log_prob = fa_log_prob
        log_prob_std = log_prob.std(unbiased=True)
        # Divide the log_probabilities of incomplete samples
        # by the number of augmentations
        if num_imputed_copies > 1:
            log_prob[incomp_mask] /= num_imputed_copies

            full_mask = ~incomp_mask
            num_true_samples = full_mask.sum()
            num_true_samples = num_true_samples + torch.true_divide((X.shape[0] - num_true_samples), num_imputed_copies)
            log_prob = log_prob.sum() / num_true_samples
        else:
            log_prob = log_prob.mean()

        pbar = {
            'train_log_lik': log_prob.item(),
            'train_log_lik_std': log_prob_std.item()
        }
        output = OrderedDict({
            'loss': -log_prob,
            'progress_bar': pbar,
        })
        pbar['train_fa_log_prob'] = fa_log_prob.mean().item()
        if mis_log_prob is not None:
            pbar['train_mis_log_prob'] = mis_log_prob.mean().item()
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
        X, M, I = batch[:3]

        if (hasattr(self.hparams, 'validation_on_miss_only')  # for BC
                and self.hparams.validation_on_miss_only):
            mis_log_prob = None
            fa_log_prob = self.fa_model.log_marginal_prob(X, M)
            log_prob = fa_log_prob
        else:
            # Compute log-likelihood on fully visible data
            fa_log_prob, mis_log_prob = self.forward(X, M, I)
            if mis_log_prob is not None:
                log_prob = fa_log_prob + mis_log_prob
            else:
                log_prob = fa_log_prob
        log_prob_std = log_prob.std(unbiased=True)
        log_prob = log_prob.mean()

        output = OrderedDict({
            'val_loss': -log_prob,
            'val_log_lik': log_prob.item(),
            'val_log_lik_std': log_prob_std.item()
        })

        output['val_fa_log_prob'] = fa_log_prob.mean().item()
        if mis_log_prob is not None:
            output['val_mis_log_prob'] = mis_log_prob.mean().item()
        return output
