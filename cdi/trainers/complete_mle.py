import sys
from collections import OrderedDict

import torch
import torch.optim as optim

import cdi.trainers.trainer_base as tb
from cdi.models.factor_analysis import FactorAnalysis
from cdi.models.fc_vae import FC_VAE, CONV_VAE_bernoulli
from cdi.models.fc_partial_vae import FC_PartialVAE
from cdi.models.fc_hivae import FC_HIVAE
from cdi.models.fc_mcvae import FC_MCVAE, FC_MCVAE_2G, FC_MCVAE_BVGI, FC_MCVAE_BVGI_2G
from cdi.models.norm_flow import NormFlow
from cdi.models.missingness_model import MultinomialLogRegMisModel
from cdi.models.ensamble import Ensamble
from cdi.util.arg_utils import parse_bool
from cdi.util.multi_optimiser import MultiOptimiser


def get_mis_model_class_from_string(model):
    if model == 'multinomial-log-regression':
        return MultinomialLogRegMisModel
    else:
        print((f'No such missingness model `{model}`!'))
        sys.exit()


class CompleteMLE(tb.TrainerBase):
    """
    Maximum likelihood estimation (MLE) model
    with complete data.
    """
    def __init__(self, hparams, model=None):
        super(CompleteMLE, self).__init__(hparams)
        self.using_ensamble = False

        if hasattr(self.hparams, 'combine_imputations') and self.hparams.combine_imputations:
            assert hasattr(self.hparams.data, 'augment_complete') and self.hparams.data.augment_complete,\
                'augment_complete should be true when using combine_imputations!'

            assert len(self.hparams.data.num_imputed_copies) == 1,\
                ('The data should not use an scheduler imputation!')

        # Prepare model
        # For BC
        if model is None:
            Model = self.get_model_class_from_hparams()
            if hasattr(self.hparams, 'ensamble') and self.hparams.ensamble:
                self.fa_model = Ensamble(self.hparams, Model)

                assert len(self.hparams.data.num_imputed_copies) == 1,\
                    ('The data should not use an scheduler imputation!')

                assert hasattr(self.hparams.data, 'augment_complete') and self.hparams.data.augment_complete,\
                    ('augment_complete in the dataset should be true')

                self.using_ensamble = True
            else:
                self.fa_model = Model(self.hparams)
            self.fa_model.reset_parameters()
        else:
            self.fa_model = model

        if hasattr(self.hparams, 'mis_model'):
            MisModel = get_mis_model_class_from_string(
                                self.hparams.missingness_model)
            self.mis_model = MisModel(self.hparams)
            self.mis_model.reset_parameters()

    def get_model_class_from_hparams(self):
        if (not hasattr(self.hparams, 'density_model')) or self.hparams.density_model == 'factor-analysis':
            return FactorAnalysis
        elif self.hparams.density_model == 'fc-vae':
            return FC_VAE
        elif self.hparams.density_model == 'conv-vae-bernoulli':
            return CONV_VAE_bernoulli
        elif self.hparams.density_model == 'fc-pvae':
            return FC_PartialVAE
        elif self.hparams.density_model == 'fc-hivae':
            return FC_HIVAE
        elif self.hparams.density_model == 'fc-mcvae':
            return FC_MCVAE
        elif self.hparams.density_model == 'fc-mcvae2':
            return FC_MCVAE_2G
        elif self.hparams.density_model == 'fc-mcvae-bvgi':
            return FC_MCVAE_BVGI
        elif self.hparams.density_model == 'fc-mcvae-bvgi2':
            return FC_MCVAE_BVGI_2G
        elif self.hparams.density_model == 'flow':
            return NormFlow
        return None

    @staticmethod
    def add_model_args(parent_parser, args=None):
        parser = super(CompleteMLE, CompleteMLE).add_model_args(parent_parser)
        temp_args, _ = parser._parse_known_args(args)
        if temp_args.method not in ('expectation-maximisation'):
            parser.add_argument('--model_optim.learning_rate',
                                type=float, required=True,
                                help=('The learning rate using in Adam '
                                      'optimiser for the fa_model.'))
            parser.add_argument('--model_optim.weight_decay_coeff',
                                type=float, required=True,
                                help=('The weight decay used in Adam '
                                      'optimiser for the fa_model.'))
            parser.add_argument('--model_optim.optimiser',
                                type=str, choices=['adam', 'amsgrad'],
                                default='adam')
            parser.add_argument('--model_optim.anneal_learning_rate',
                                type=parse_bool, default=False)
            parser.add_argument('--model_optim.anneal_steps',
                                type=int, default=None)
        parser.add_argument('--model_enc_optim.learning_rate',
                            type=float, required=False,
                            help=('The learning rate using in Adam '
                                  'optimiser for the fa_models encoder.'))
        parser.add_argument('--model_enc_optim.weight_decay_coeff',
                            type=float, required=False,
                            help=('The weight decay used in Adam '
                                  'optimiser for the fa_model encoder.'))
        parser.add_argument('--model_enc_optim.optimiser',
                            type=str, choices=['adam', 'amsgrad'],
                            required=False)
        parser.add_argument('--model_enc_optim.anneal_learning_rate',
                            type=parse_bool, default=False)
        parser.add_argument('--model_enc_optim.anneal_steps',
                            type=int, default=None)
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
        parser.add_argument('--mis_model_optim.anneal_learning_rate',
                            type=parse_bool, default=False)
        parser.add_argument('--mis_model_optim.anneal_steps',
                            type=int, default=None)
        parser.add_argument('--validation_on_miss_only',
                            type=parse_bool, default=False,
                            help=('Evaluate using missing data marginal prob.'))
        parser.add_argument('--fix_generator', default=False,
                            help=('Whether to fix generator weights.'))

        parser.add_argument('--density_model', type=str, default='factor-analysis',
                            choices=['factor-analysis', 'fc-vae', 'conv-vae-bernoulli', 'fc-pvae',
                                     'fc-hivae', 'fc-mcvae', 'fc-mcvae2', 'fc-mcvae-bvgi', 'fc-mcvae-bvgi2',
                                     'flow'])
        parser.add_argument('--ensamble', type=parse_bool, default=False,
                            help=('Whether to fit a separate model on each imputed dataset.'))

        parser.add_argument('--combine_imputations', type=str,
                            choices=['avg', 'log-avg-exp'],
                            help='How to combine imputations (use with augment_complete).')

        temp_args, _ = parser._parse_known_args(args)
        # For BC
        if (not hasattr(temp_args, 'density_model')) or temp_args.density_model == 'factor-analysis':
            # Add FA model parameters
            parser = FactorAnalysis.add_model_args(parser)
        elif temp_args.density_model == 'fc-vae':
            parser = FC_VAE.add_model_args(parser)
        elif temp_args.density_model == 'conv-vae-bernoulli':
            parser = CONV_VAE_bernoulli.add_model_args(parser)
        elif temp_args.density_model == 'fc-pvae':
            parser = FC_PartialVAE.add_model_args(parser)
        elif temp_args.density_model == 'fc-hivae':
            parser = FC_HIVAE.add_model_args(parser)
        elif temp_args.density_model == 'fc-mcvae':
            parser = FC_MCVAE.add_model_args(parser)
        elif temp_args.density_model == 'fc-mcvae2':
            parser = FC_MCVAE_2G.add_model_args(parser)
        elif temp_args.density_model == 'fc-mcvae-bvgi':
            parser = FC_MCVAE_BVGI.add_model_args(parser)
        elif temp_args.density_model == 'fc-mcvae-bvgi2':
            parser = FC_MCVAE_BVGI_2G.add_model_args(parser)
        elif temp_args.density_model == 'flow':
            parser = NormFlow.add_model_args(parser)

        if 'missingness_model' in vars(temp_args):
            MisModel = get_mis_model_class_from_string(
                                temp_args.missingness_model)
            parser = MisModel.add_model_args(parser)

        return parser

    # Setup
    def configure_optimizers(self):
        schedulers = []
        if not hasattr(self.hparams, 'model_enc_optim') or self.hparams.model_enc_optim.optimiser is None:
            if (not hasattr(self.hparams.model_optim, 'optimiser')
                    or self.hparams.model_optim.optimiser == 'adam'):
                model_amsgrad = False
            elif self.hparams.model_optim.optimiser == 'amsgrad':
                model_amsgrad = True
            else:
                sys.exit('No such optimizer for the density model!')

            self.optim = MultiOptimiser(
                        fa_model_opt=optim.Adam(
                            self.fa_model.parameters(),
                            amsgrad=model_amsgrad,
                            lr=self.hparams.model_optim.learning_rate,
                            weight_decay=self.hparams.model_optim.weight_decay_coeff)
                        )

            if self.hparams.model_optim.anneal_learning_rate:
                max_steps = (self.hparams.model_optim.anneal_steps
                             if hasattr(self.hparams.model_optim, 'anneal_steps') and self.hparams.model_optim.anneal_steps is not None
                             else self.hparams.max_epochs)
                schedulers.append(optim.lr_scheduler.CosineAnnealingLR(self.optim.optimisers['fa_model_opt'], max_steps, 0))

        else:
            # In this case use a different optimiser for encoder/decoder of a VAE
            if self.hparams.model_optim.optimiser == 'adam':
                model_amsgrad = False
            elif self.hparams.model_optim.optimiser == 'amsgrad':
                model_amsgrad = True
            else:
                sys.exit('No such optimizer for the density model!')

            if self.hparams.model_enc_optim.optimiser == 'adam':
                model_enc_amsgrad = False
            elif self.hparams.model_enc_optim.optimiser == 'amsgrad':
                model_enc_amsgrad = True
            else:
                sys.exit('No such optimizer for the density model encoder!')

            self.optim = MultiOptimiser(
                        fa_model_opt=MultiOptimiser(
                            generator_opt=optim.Adam(
                                self.fa_model.generator_parameters(),
                                amsgrad=model_amsgrad,
                                lr=self.hparams.model_optim.learning_rate,
                                weight_decay=self.hparams.model_optim.weight_decay_coeff),
                            encoder_opt=optim.Adam(
                                self.fa_model.encoder_parameters(),
                                amsgrad=model_enc_amsgrad,
                                lr=self.hparams.model_enc_optim.learning_rate,
                                weight_decay=self.hparams.model_enc_optim.weight_decay_coeff),
                            use_run_list=False
                            )
                        )

            if self.hparams.model_optim.anneal_learning_rate:
                max_steps = (self.hparams.model_optim.anneal_steps
                             if hasattr(self.hparams.model_optim, 'anneal_steps') and self.hparams.model_optim.anneal_steps is not None
                             else self.hparams.max_epochs)
                schedulers.append(optim.lr_scheduler.CosineAnnealingLR(self.optim.optimisers['fa_model_opt'].optimisers['generator_opt'], max_steps, 0))
            if self.hparams.model_enc_optim.anneal_learning_rate:
                max_steps = (self.hparams.model_enc_optim.anneal_steps
                             if hasattr(self.hparams.model_enc_optim, 'anneal_steps') and self.hparams.model_enc_optim.anneal_steps is not None
                             else self.hparams.max_epochs)
                schedulers.append(optim.lr_scheduler.CosineAnnealingLR(self.optim.optimisers['fa_model_opt'].optimisers['encoder_opt'], max_steps, 0))

        if hasattr(self, 'mis_model'):
            self.optim.add_optimisers(
                mis_model_opt=optim.Adam(
                    self.mis_model.parameters(),
                    amsgrad=False,
                    lr=self.hparams.mis_model_optim.learning_rate,
                    weight_decay=self.hparams.mis_model_optim.weight_decay_coeff))

            if self.hparams.mis_model.anneal_learning_rate:
                max_steps = (self.hparams.mis_model.anneal_steps
                             if hasattr(self.hparams.mis_model, 'anneal_steps') and self.hparams.mis_model.anneal_steps is not None
                             else self.hparams.max_epochs)
                schedulers.append(optim.lr_scheduler.CosineAnnealingLR(self.optim.optimisers['mis_model_opt'], max_steps, 0))

        if len(schedulers) == 0:
            return self.optim
        else:
            return [self.optim], schedulers

    def setup(self, stage):
        out = super().setup(stage)

        if stage == 'fit':
            # Workaround for this particular model to know the patterns in the data
            # It is not necessary for stage==`test` because the details should be
            # persisted in the model
            if (hasattr(self, 'mis_model')
                    and isinstance(self.mis_model, MultinomialLogRegMisModel)):
                self.mis_model.initialise(self.train_dataset.dataset.dataset.patterns)

            if (hasattr(self.hparams, 'fix_generator') and self.hparams.fix_generator):
                self.fa_model.freeze_decoder()

            if hasattr(self.hparams, 'fix_encoder') and self.hparams.fix_encoder:
                self.fa_model.freeze_encoder()

        return out

    # Training

    def forward(self, X, M):
        if self.training:
            self.optim.add_run_opt('fa_model_opt')
            if hasattr(self, 'mis_model'):
                self.optim.add_run_opt('mis_model_opt')
        fa_log_probs = self.fa_model(X, M)

        mis_log_probs = None
        if hasattr(self, 'mis_model'):
            mis_log_probs = self.mis_model(X, M)

        return fa_log_probs, mis_log_probs

    # Training

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

        if self.using_ensamble:
            # TODO: didn't implement the mis_model loss
            fa_log_prob, mis_log_prob = self.forward(X, M)

            log_prob_std = fa_log_prob.std(unbiased=True)
            log_prob = fa_log_prob.mean()

        elif hasattr(self.hparams.data, 'augment_complete') and self.hparams.data.augment_complete:
            fa_log_prob, mis_log_prob = self.forward(X, M)
            if mis_log_prob is not None:
                log_prob = fa_log_prob + mis_log_prob
            else:
                log_prob = fa_log_prob
            log_prob_std = log_prob.std(unbiased=True)

            if not hasattr(self.hparams, 'combine_imputations') or self.hparams.combine_imputations == 'avg':
                log_prob = log_prob.reshape(-1, self.hparams.data.num_imputed_copies[-1])
                log_prob = log_prob.mean(dim=-1)

                log_prob = log_prob.mean()
            elif self.hparams.combine_imputations == 'log-avg-exp':
                log_prob = log_prob.reshape(-1, self.hparams.data.num_imputed_copies[-1])
                log_prob = (torch.logsumexp(log_prob, dim=-1)
                            - torch.log(torch.tensor(log_prob.shape[-1],
                                                     dtype=log_prob.dtype,
                                                     device=log_prob.device)))

                log_prob = log_prob.mean()

        else:
            if self.num_imputed_copies_scheduler is not None:
                num_imputed_copies = self.num_imputed_copies_scheduler.get_value()
            elif isinstance(self.hparams.data.num_imputed_copies, list):
                num_imputed_copies = self.hparams.data.num_imputed_copies[0]
            else:  # BC
                num_imputed_copies = self.hparams.data.num_imputed_copies

            # Compute log likelihood
            fa_log_prob, mis_log_prob = self.forward(X, M)
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
        X, M = batch[:2]

        if (hasattr(self.hparams, 'validation_on_miss_only')  # for BC
                and self.hparams.validation_on_miss_only):
            mis_log_prob = None
            fa_log_prob = self.fa_model.log_marginal_prob(X, M)
            log_prob = fa_log_prob
        else:
            # Compute log-likelihood on fully visible data
            fa_log_prob, mis_log_prob = self.forward(X, M)
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

    # Hooks

    def on_epoch_start(self):
        super().on_epoch_start()
        self.fa_model.on_epoch_start()
        if hasattr(self, 'mis_model'):
            self.mis_model.on_epoch_start()

    def training_epoch_end(self, outputs):
        results = super().training_epoch_end(outputs)

        # Add epoch-level stats
        # TODO: handle this in the model classes
        results['log']['cum_fa_calls'] = self.fa_model.cum_batch_size_called if hasattr(self.fa_model, 'cum_batch_size_called') else 0
        if hasattr(self, 'mis_model'):
            results['log']['cum_mis_model_calls'] = self.mis_model.cum_batch_size_called
        return results

    #
    # Test
    #

    @staticmethod
    def add_test_args(parent_parser):
        parser = super(CompleteMLE, CompleteMLE).add_test_args(parent_parser)
        # NOTE: Hacky way to store reference to this argument
        # So we can modify it later in sub-classes.
        parser._eval_type_arg = parser.add_argument(
                                        '--eval_type',
                                        type=str, required=True,
                                        choices=['eval_loglik',
                                                 'eval_loglik_vae',
                                                 'eval_vae_z_posterior',
                                                 'eval_generator_sensitivity',
                                                 'eval_sample_flow'],
                                        help=('Available evaluations.'))
        parser.add_argument('--num_z_samples', type=int,
                            help=('Number of z samples in VAE test-loglik evaluation.'))
        parser.add_argument('--num_x_to_use', type=int)
        parser.add_argument('--num_z_val_steps', type=int,
                            help='Increments in z value.')
        parser.add_argument('--num_samples', type=int,
                            help=('Number of samples to generate in eval_sample_flow.'))
        parser.add_argument('--gen_samples_filename', type=str, default='flow_samples',
                            help='Name for the generated samples file.')

        parser.add_argument('--output_suffix', type=str,
                            help='Append a suffix to the end of the output.')
        return parser

    def test_step(self, batch, batch_idx):
        # Make sure we're running the correct evaluation!
        assert self.hparams.test.eval_type in ('eval_loglik', 'eval_loglik_vae',
                                               'eval_vae_z_posterior', 'eval_generator_sensitivity',
                                               'eval_sample_flow'),\
            f'eval_type=`{self.hparams.test.eval_type}` not supported!'

        if self.hparams.test.eval_type == 'eval_loglik':
            X, M = batch[:2]

            data_log_probs, _ = self.forward(X, M)

            # Log data log-probabilities
            self.logger.accumulate_tensors('data_log_probs',
                                           data_log_probs=(data_log_probs
                                                           .cpu().detach()))

            return {'test_log_lik': data_log_probs.mean()}
        elif self.hparams.test.eval_type == 'eval_loglik_vae':
            X, M = batch[:2]

            data_log_probs = self.fa_model.test_loglik(X, M, self.hparams.test.num_z_samples)

            # Log data log-probabilities
            self.logger.accumulate_tensors('data_log_probs',
                                           data_log_probs=(data_log_probs
                                                           .cpu().detach()))

            return {'test_log_lik': data_log_probs.mean()}
        elif self.hparams.test.eval_type == 'eval_vae_z_posterior':
            X, M = batch[:2]

            Z_mean, Z_log_var = self.fa_model.test_z_posterior(X, M)

            # Log data log-probabilities
            self.logger.accumulate_tensors('z_posterior',
                                           z_mean=Z_mean.cpu().detach(),
                                           z_log_var=Z_log_var.cpu().detach())
        elif self.hparams.test.eval_type == 'eval_generator_sensitivity':
            X, M = batch[:2]

            Z_mean, Z_log_var = self.fa_model.test_z_posterior(X, M)

            Z_mean_max = torch.max(Z_mean, dim=0)[0]
            Z_mean_min = torch.min(Z_mean, dim=0)[0]

            if self.hparams.test.num_x_to_use != -1:
                perm = torch.randperm(X.shape[0])
                idx = perm[:self.hparams.test.num_x_to_use]
                X = X[idx, :]
                Z_mean = Z_mean[idx, :]
                Z_log_var = Z_log_var[idx, :]
            else:
                idx = torch.arange(X.shape[0])

            X_mean, X_log_var = self.fa_model.decode(Z_mean)

            self.logger.accumulate_tensors('generator_sensitivity',
                                           X=X.cpu().detach(),
                                           idx=idx.cpu().detach(),
                                           Z_mean=Z_mean.cpu().detach(),
                                           Z_log_var=Z_log_var.cpu().detach(),
                                           X_mean=X_mean.cpu().detach(),
                                           X_log_var=X_log_var.cpu().detach())

            for i in range(Z_mean.shape[-1]):
                Z_vals_i = torch.linspace(Z_mean_min[i].item(), Z_mean_max[i].item(), self.hparams.test.num_z_val_steps)
                Z_mean_i = Z_mean.unsqueeze(0).repeat(len(Z_vals_i), 1, 1)
                Z_mean_i[:, :, i] = Z_vals_i.unsqueeze(-1)

                X_mean_i, X_log_var_i = self.fa_model.decode(Z_mean_i)

                # Log data log-probabilities
                self.logger.accumulate_tensors('generator_sensitivity',
                                               **{
                                                    f'z_vals_{i}': Z_vals_i.cpu().detach(),
                                                    f'X_mean_{i}': X_mean_i.cpu().detach(),
                                                    f'X_log_var_{i}': X_log_var_i.cpu().detach(),
                                               })
        elif self.hparams.test.eval_type == 'eval_sample_flow':
            if batch_idx == 0:
                samples = self.fa_model.flow._sample(self.hparams.test.num_samples, context=None)

                # Log samples
                self.logger.accumulate_tensors(self.hparams.test.gen_samples_filename,
                                               samples=(samples.cpu().detach()))

    def test_epoch_end(self, outputs):
        suffix = ''
        if hasattr(self.hparams.test, 'output_suffix') and self.hparams.test.output_suffix is not None:
            suffix = f'_{self.hparams.test.output_suffix}'
        # Make sure we're running the correct evaluation!
        if self.hparams.test.eval_type in ('eval_loglik', 'eval_loglik_vae'):
            # Save log-probs
            self.logger.save_accumulated_tensors('data_log_probs', 'test'+suffix)
            test_log_lik_mean = torch.stack([x['test_log_lik']
                                            for x in outputs]).mean()
            result = {'test_loss': -test_log_lik_mean,
                      'progress_bar': {'test_log_lik': test_log_lik_mean}}
        elif self.hparams.test.eval_type == 'eval_vae_z_posterior':
            self.logger.save_accumulated_tensors('z_posterior', 'test'+suffix)
            result = {}
        elif self.hparams.test.eval_type == 'eval_generator_sensitivity':
            self.logger.save_accumulated_tensors('generator_sensitivity', 'test'+suffix)
            result = {}
        elif self.hparams.test.eval_type == 'eval_sample_flow':
            self.logger.save_accumulated_tensors(self.hparams.test.gen_samples_filename, 'test'+suffix)
            result = {}
        else:
            raise Exception(f'eval_type=`{self.hparams.test.eval_type}` not supported!')

        return result
