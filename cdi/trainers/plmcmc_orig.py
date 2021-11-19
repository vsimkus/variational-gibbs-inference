import time
from collections import OrderedDict
from enum import Enum

import numpy as np
import torch
from tqdm import tqdm

import cdi.trainers.complete_mle as cm
from cdi.util.arg_utils import parse_bool
from cdi.util.data.data_augmentation_dataset import (
    DataAugmentation, DataAugmentationWithScheduler, collate_augmented_samples)
from cdi.util.data.fully_missing_filter_dataset import FullyMissingDataFilter
from cdi.util.utils import EpochScheduler, EpochIntervalScheduler
from cdi.util.data.persistent_latents_dataset import LatentsDataset2


class ResentLatentsEnum(Enum):
    NO_RESET = 0
    USE_MODEL = 1
    USE_GAUSS = 2


class PLMCMC_orig(cm.CompleteMLE):
    """
    Projected Latent MCMC EM for normalising flows.
    Follows original implementation more closely.

    Canella et al. 2021 "Projected Latent Markov Chain Monte Carlo: Conditional Sampling of Normalizing Flows"
    """

    def __init__(self, hparams):
        super(PLMCMC_orig, self).__init__(hparams)

        assert not (self.hparams.plmcmc.mcmc_before_epoch and self.hparams.plmcmc.mcmc_during_epoch),\
            ('Cannot set both \'mcmc_before_epoch\' and \'mcmc_during_epoch\' to true.')

        self._init_schedulers()

    def _init_schedulers(self):
        # TODO: add PL-MCMC scheduler.
        self.num_imp_steps_schedule = EpochScheduler(
                    self,
                    self.hparams.plmcmc.num_imp_steps_schedule,
                    self.hparams.plmcmc.num_imp_steps_schedule_values)
        self.latent_reset_schedule = EpochScheduler(
                    self,
                    self.hparams.plmcmc.latent_reset_schedule,
                    [ResentLatentsEnum[v] for v in self.hparams.plmcmc.latent_reset_schedule_values])
        self.update_imputations_schedule = EpochIntervalScheduler(
                    self,
                    self.hparams.plmcmc.update_imputations_schedule_init_value,
                    self.hparams.plmcmc.update_imputations_schedule_main_value,
                    self.hparams.plmcmc.update_imputations_schedule_other_value,
                    self.hparams.plmcmc.update_imputations_schedule_start_epoch,
                    self.hparams.plmcmc.update_imputations_schedule_period)

    @staticmethod
    def add_model_args(parent_parser, args=None):
        parser = super(PLMCMC_orig, PLMCMC_orig).add_model_args(parent_parser, args)

        # PLMCMC args
        parser.add_argument('--plmcmc.mcmc_before_epoch', type=parse_bool,
                            required=True, help='Run MCMC before epoch on full data.')
        parser.add_argument('--plmcmc.mcmc_during_epoch', type=parse_bool,
                            required=True, help='Run MCMC during epoch on each batch separately.')
        parser.add_argument('--plmcmc.mcmc_batch_size', type=int,
                            help='Batch size to use in `mcmc_before_epoch` mode of mcmc.')

        parser.add_argument('--plmcmc.num_imp_steps_schedule',
                            type=int, nargs='+', required=True,
                            help=('A list of epochs when number of imputation'
                                  ' steps should be changed.'))
        parser.add_argument('--plmcmc.num_imp_steps_schedule_values',
                            type=int, nargs='+', required=True,
                            help=('A list of values that correspond to each'
                                  ' schedule\'s mode. The number of imputation'
                                  ' steps in each epoch.'))

        parser.add_argument('--plmcmc.latent_reset_schedule',
                            type=int, nargs='+', required=True,
                            help=('A list of epochs when latent reset mode should be changed.'))
        parser.add_argument('--plmcmc.latent_reset_schedule_values',
                            type=str, nargs='+', required=True,
                            help=('A list of values that correspond to each'
                                  ' schedule\'s mode. \'NO_RESET\', \'USE_MODEL\', \'USE_GAUSS\''))

        parser.add_argument('--plmcmc.update_imputations_schedule_init_value',
                            type=parse_bool, required=True,
                            help=('What to return before start_epoch.'))
        parser.add_argument('--plmcmc.update_imputations_schedule_main_value',
                            type=parse_bool, required=True,
                            help=('What to return when scheduled.'))
        parser.add_argument('--plmcmc.update_imputations_schedule_other_value',
                            type=parse_bool, required=True,
                            help=('What to return when not scheduled.'))
        parser.add_argument('--plmcmc.update_imputations_schedule_start_epoch',
                            type=int, required=True,
                            help=('When scheduling starts.'))
        parser.add_argument('--plmcmc.update_imputations_schedule_period',
                            type=int, required=True,
                            help=('Period of scheduling switches.'))

        parser.add_argument('--plmcmc.dim', type=int,
                            help=('Y dimensionality, should be the same as X.'))

        parser.add_argument('--plmcmc.resample_prop_prob', type=float,
                            required=True,
                            help=('Probability to use resampling proposal.'))
        parser.add_argument('--plmcmc.resample_prop_std', type=float,
                            required=True,
                            help=('Resampled point standard deviation using resampling proposal.'))
        parser.add_argument('--plmcmc.perturb_prop_std', type=float,
                            required=True,
                            help=('Perturbed point standard deviation using perturbing proposal.'))
        parser.add_argument('--plmcmc.perturb_std', type=float,
                            required=True,
                            help=('Noise added to the observed data in MCMC acceptance.'))
        parser.add_argument('--plmcmc.aux_dist_std', type=float,
                            required=True,
                            help=('Standard deviation of the auxilliary distribution q.'))
        parser.add_argument('--plmcmc.clamp_imputations', type=parse_bool,
                            required=True,
                            help=('Clamps the imputations to the observed data hypercube.'))

        # Debugging params
        # parser.add_argument('--cdi.debug.log_dataset',
        #                     type=parse_bool, default=False,
        #                     help=('DEBUG: Logs the dataset state at the end of'
        #                           ' the epoch.'))
        parser.add_argument('--plmcmc.debug.eval_incomplete',
                            type=parse_bool, default=False,
                            help=('In addition evaluates validation on incomplete data,'
                                  'runs a chain of validation imputations similar to training.'))
        return parser

    def init_plmcmc_data(self, dataset):
        X, Y, M, I, *_ = dataset[:]

        # TODO: scale and shift L if neccessary

        # NOTE: this piece of code is performed in the original implementation, however
        # this is redundant.
        # noise_proposals = self.fa_model.transform_to_noise(Y)
        # imputations = self.fa_model.transform_from_noise(noise_proposals)

        X = X*M + Y*(~M)

        dataset[I] = X
        # TODO: set Y if it is modified

    def setup(self, stage):
        super().setup(stage)

        if stage == 'fit':
            # Latents correspond to the $y$ in the paper.
            self.train_dataset = LatentsDataset2(self.train_dataset,
                                                 latent_dim=self.hparams.plmcmc.dim)

            X = self.train_dataset[:][0]
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)

            if isinstance(X_min, np.ndarray):
                X_min = torch.tensor(X_min)
                X_max = torch.tensor(X_max)
            else:
                X_min = X_min[0]
                X_max = X_max[0]

            self.X_min = X_min
            self.X_max = X_max

            self.init_plmcmc_data(self.train_dataset)

            if self.hparams.plmcmc.debug.eval_incomplete:
                # Remove fully-missing samples if required
                val_dataset = self.val_dataset
                if self.hparams.data.filter_fully_missing:
                    val_dataset = FullyMissingDataFilter(val_dataset)

                if self.num_imputed_copies_scheduler is None:
                    if isinstance(self.hparams.data.num_imputed_copies, list):
                        num_copies = self.hparams.data.num_imputed_copies[0]
                    else:
                        num_copies = self.hparams.data.num_imputed_copies
                    self.val_dataset_augmented = DataAugmentation(
                                                val_dataset,
                                                num_copies,
                                                augment_complete=hasattr(self.hparams.data, 'augment_complete') and self.hparams.data.augment_complete)
                else:
                    self.val_dataset_augmented = DataAugmentationWithScheduler(
                                                val_dataset,
                                                self.num_imputed_copies_scheduler,
                                                augment_complete=hasattr(self.hparams.data, 'augment_complete') and self.hparams.data.augment_complete)

                self.initialise_dataset(self.hparams, self.val_dataset_augmented)

                self.val_dataset_augmented = LatentsDataset2(self.val_dataset_augmented,
                                                            latent_dim=self.hparams.plmcmc.latent_dim)

                self.init_plmcmc_data(self.val_dataset_augmented)

    def val_dataloader(self):
        val_dataloader = super().val_dataloader()
        if self.hparams.plmcmc.debug.eval_incomplete:
            val_aug_dataloader = torch.utils.data.DataLoader(
                                self.val_dataset_augmented,
                                batch_size=self.hparams.data.batch_size,
                                collate_fn=collate_augmented_samples,
                                num_workers=2,
                                shuffle=False)
            return [val_dataloader, val_aug_dataloader]

        return [val_dataloader]

    # Training

    def sample_imputations(self, batch, num_steps, r_prob, r_std, p_std, q_std, perturb_std=0.01):
        X, Y, M, I, *_ = batch
        # M_not = ~M
        M = M.float()

        acceptances = 0
        tries = 0
        with torch.no_grad():
            bernoulli = torch.distributions.Bernoulli(probs=torch.tensor(1-r_prob).to(X.device))
            for _ in range(num_steps):
                # Choose which to resample completely
                # 1 - perturbation
                # 0 - resampling
                resample_mask = bernoulli.sample(sample_shape=[X.shape[0], 1]).float()
                # resample_mask_not = ~resample_mask

                inputs = X
                masks = M

                # f: X->Z
                original_proposals = self.fa_model.transform_to_noise_and_logabsdetJ(Y)[0]

                # g: Z->X
                # impute data
                projected_end = self.fa_model.transform_from_noise_and_logabsdetJ(original_proposals)[0].mul(1.0 - masks) + inputs.mul(masks)

                # Perturbation kernel
                # normal_sample_holder.normal_(mean=0.0, std=1.0)
                # normal_samples = torch.randn_like(original_proposals)
                new_proposals = (original_proposals + torch.randn_like(original_proposals).mul(p_std)).mul(resample_mask)

                # Completely resampled states
                new_proposals += torch.randn_like(original_proposals).mul(r_std).mul(1.0 - resample_mask)

                # g: Z->X
                # Get the proposal to the data space
                proposal = self.fa_model.transform_from_noise_and_logabsdetJ(new_proposals)[0]

                #proposal = torch.max(proposal, self.min)
                #proposal = torch.min(proposal, self.max)
                # perturbations.normal_(mean=0.0, std=1.0)
                current_perturbations = 0
                #bayes_mod = (1e6)*(((flow.g(original_proposals)-inputs + perturbations).mul(self.stdev.cuda()).mul(masks)**2).sum(dim=1)/2.0 - ((proposal-inputs + perturbations).mul(self.stdev.cuda()).mul(masks)**2).sum(dim=1)/2.0)
                # Why perturb the points? g: Z->X
                # Ignoring the perturbation, this corresponds to the log(q(y_o')/q(y_o))
                bayes_mod = (1e6)*(((self.fa_model.transform_from_noise_and_logabsdetJ(original_proposals)[0]-inputs + current_perturbations).mul(masks)**2).sum(dim=1)/2.0
                                   - ((proposal-inputs+current_perturbations).mul(masks)**2).sum(dim=1)/2.0)
                # Why perturb the observed data?
                proposal = proposal.mul(1.0 - masks) + (inputs+current_perturbations).mul(masks)
                projected_end = projected_end.mul(1.0 - masks) + (inputs+current_perturbations).mul(masks)
                #acceptance_prob = torch.exp(bayes_mod.unsqueeze(1) + flow.log_prob(proposal).unsqueeze(1) - (proposal**2).div(8).sum(dim=1).unsqueeze(1) - flow.log_prob(projected_end).unsqueeze(1) + (projected_end**2).div(8).sum(dim=1).unsqueeze(1) + ((new_proposals**2).sum(1)/2.0 - (original_proposals**2).sum(1)/2.0).unsqueeze(1).mul(1.0 - resample_mask)/sample_std**2)

                acceptance_prob = (bayes_mod.unsqueeze(1)
                                   # The log-likelihood term ratio
                                   + self.fa_model.log_prob(proposal).unsqueeze(1) - self.fa_model.log_prob(projected_end).unsqueeze(1)
                                   # The proposal ratio of the perturbed latents cancels out.
                                   # (eps2 - eps1)^2/sigma^2 - (eps1 - eps2)^2.sigma^2
                                   # The proposal log-ratio of the resampled latents
                                   + ((new_proposals**2).sum(1)/2.0 - (original_proposals**2).sum(1)/2.0).unsqueeze(1).mul(1.0 - resample_mask)/r_std**2
                                   # The log jacobian determinant terms is omitted because for the NICE model it cancels
                                   + self.fa_model.transform_from_noise_and_logabsdetJ(new_proposals)[1].unsqueeze(1) - self.fa_model.transform_from_noise_and_logabsdetJ(original_proposals)[1].unsqueeze(1)
                                   )

                acceptance_prob = torch.exp(torch.clamp(acceptance_prob, -25, 25))
                acceptance_samples = torch.rand_like(acceptance_prob)
                current_acceptances = (acceptance_samples < acceptance_prob).float()
                acceptances += float(current_acceptances.sum())
                tries += float(len(current_acceptances))
                # if sample_idx ==0:
                #     changing_indices += [affected_indices[index] for index in range(0,batch_length) if current_acceptances[index]]
                # Create the eventual new sample (accepted)
                original_proposals = new_proposals.mul(current_acceptances) + original_proposals.mul(1.0 - current_acceptances)

                # g:Z->X
                Y.data = (self.fa_model.transform_from_noise_and_logabsdetJ(original_proposals)[0]).data
                X.data = (self.fa_model.transform_from_noise_and_logabsdetJ(original_proposals)[0].mul(1.0 - masks) + inputs.mul(masks)).data

        return acceptances, tries

    def impute_batch(self, batch, stage, num_imputation_steps, latent_reset_mode, update_imputations):
        """
        Impute dataset at the start of each batch.
        batch:
            X (N, D): observable variables
            M (N, D): binary missingness mask.
            I (N,): indices of the X samples in the dataset
                (can used for imputation where necessary)
        """
        X, Y, M, I = batch[:4]

        with torch.no_grad():
            total_accepted = 0
            tries = 0
            if update_imputations:
                # Put the model into eval mode (so that drop-out does not affect MCMC)
                is_training = self.training
                if is_training:
                    self.eval()

                M_not = ~M
                if latent_reset_mode == ResentLatentsEnum.USE_MODEL:
                    # Completely resample *all* latents
                    Z = torch.empty_like(Y).normal_(mean=0.0, std=self.hparams.plmcmc.resample_prop_std)
                    Y.data = self.fa_model.transform_from_noise_and_logabsdetJ(Z)[0].data
                elif latent_reset_mode == ResentLatentsEnum.USE_GAUSS:
                    # TODO: scale and shift Y if necessary
                    Y.normal_(mean=0.0, std=1.0)
                # elif latent_reset_mode == ResentLatentsEnum.NO_RESET:
                #     # Do nothing
                #     pass

                # Update projected imputations
                X.data = (X*M + Y*M_not).data

                total_accepted, tries = self.sample_imputations(batch, num_imputation_steps,
                                                                r_prob=self.hparams.plmcmc.resample_prop_prob,
                                                                r_std=self.hparams.plmcmc.resample_prop_std,
                                                                p_std=self.hparams.plmcmc.perturb_prop_std,
                                                                q_std=self.hparams.plmcmc.aux_dist_std,
                                                                perturb_std=self.hparams.plmcmc.perturb_std)

                if self.hparams.plmcmc.clamp_imputations:
                    # torch.clamp_(X, min=self.X_min, max=self.X_max)
                    torch.min(X, self.X_max.to(X.device), out=X)
                    torch.max(X, self.X_min.to(X.device), out=X)

                if stage == 'train':
                    self.train_dataset[I.cpu()] = X.cpu()
                    self.train_dataset.set_latents(I.cpu(), Y.cpu())
                elif stage == 'val':
                    self.val_dataset_augmented[I.cpu()] = X.cpu()
                    self.val_dataset_augmented.set_latents(I.cpu(), Y.cpu())
                elif stage == 'test':
                    self.test_dataset[I.cpu()] = X.cpu()
                    self.test_dataset.set_latents(I.cpu(), Y.cpu())

                if is_training:
                    self.train()

        return {'train_imp_accepted': total_accepted,
                'train_imp_tries': tries}

    def update_step(self, batch):
        """
        One iteration of MLE update using CDI algorithm.
        batch:
            X (N, D): observable variables
            M (N, D): binary missingness mask.
            I (N,): indices of the X samples in the dataset
                (can used for imputation where necessary)
        """
        X, Y, M, I, OI, incomp_mask = batch
        # Compute the number of complete samples
        # As well as the true number of samples from the original dataset
        # N_incomp = incomp_mask.sum()
        # N_full = incomp_mask.shape[0] - N_incomp
        # N_true = N_full + (N_incomp / self.hparams.data.num_imputed_copies)

        if self.num_imputed_copies_scheduler is not None:
            num_imputed_copies = self.num_imputed_copies_scheduler.get_value()
        elif isinstance(self.hparams.data.num_imputed_copies, list):
            num_imputed_copies = self.hparams.data.num_imputed_copies[0]
        else:  # BC
            num_imputed_copies = self.hparams.data.num_imputed_copies

        # Get the unique original-indices (to establish total sample count)
        # Also get counts and inverse-index so that we can compute the
        # averages over incomplete-chains.
        if num_imputed_copies > 1:
            unique_oi, oi_inv_idx, oi_counts = torch.unique(
                                                    OI,
                                                    return_inverse=True,
                                                    return_counts=True)
            N_true = unique_oi.shape[0]
        else:
            N_true = X.shape[0]

        # Evaluate the CDI objective
        log_probs, _ = self.forward(X, M)

        # Divide the log_probs and entropies of incomplete samples
        # by the number of augmentations
        if num_imputed_copies > 1:
            log_probs /= oi_counts[oi_inv_idx]

        # Compute average log_prob and entropy.
        # Since the log_probs of augmented samples are now scaled down
        # We can compute the total average log_probability by dividing
        # total sum by the total *true* number of samples in the batch
        log_prob = log_probs.sum() / N_true

        # Compute loss and update parameters (by maximising log-probability
        # and entropy)
        loss = -log_prob

        pbar = {
            'train_log_lik': log_prob.item()
        }
        output = OrderedDict({
            'loss': loss,
            'progress_bar': pbar,
        })
        return output

    def training_step(self, batch, batch_idx):
        """
        Performs CDI update and imputation of missing values.
        """
        # Imputation
        if self.hparams.plmcmc.mcmc_during_epoch:
            imp_start_time = time.time()

            num_imputation_steps = self.num_imp_steps_schedule.get_value()
            latent_reset_mode = self.latent_reset_schedule.get_value()
            update_imputations = self.update_imputations_schedule.get_value()
            logs = self.impute_batch(batch, stage='train',
                                     num_imputation_steps=num_imputation_steps,
                                     latent_reset_mode=latent_reset_mode,
                                     update_imputations=update_imputations)

            if self.num_imputed_copies_scheduler is not None:
                prev_num_copies = self.num_imputed_copies_scheduler.get_value(self.current_epoch-1)
                curr_num_copies = self.num_imputed_copies_scheduler.get_value(self.current_epoch)
                # Check if the number of chains was increased, if so, we need to impute those copies
                if curr_num_copies > prev_num_copies:
                    X, Y, M, I = batch[:4]
                    mask = self.train_dataset.which_samples_are_new(prev_num_copies,
                                                                    curr_num_copies,
                                                                    indices=I.cpu().numpy())
                    mask = torch.tensor(mask, device=X.device)
                    X_new, Y_new, M_new, I_new = X[mask], Y[mask], M[mask], I[mask]
                    # Impute the new chains
                    self.impute_batch((X_new, Y_new, M_new, I_new), stage='train',
                                      num_imputation_steps=self.hparams.data.num_new_chain_imp_steps,
                                      latent_reset_mode=latent_reset_mode,
                                      update_imputations=update_imputations)
                    # Set the imputed values into the batch
                    X[mask], Y[mask], M[mask], I[mask] = X_new, Y_new, M_new, I_new

            imp_time = time.time() - imp_start_time
        else:
            logs = self.train_imp_log
            imp_time = self.train_imp_time
            self.train_imp_log = {}
            self.train_imp_time = 0.

        # Perturb data
        if self.hparams.plmcmc.perturb_std != 0:
            batch[0] += torch.randn_like(batch[0])*self.hparams.plmcmc.perturb_std

        # Update
        output = self.update_step(batch)
        output['progress_bar'].update(logs)
        output['progress_bar']['train_imp_time'] = imp_time
        return output

    # Validation

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        if dataset_idx == 0:
            if self.hparams.plmcmc.perturb_std != 0:
                batch[0] += torch.randn_like(batch[0])*self.hparams.plmcmc.perturb_std
            return super().validation_step(batch, batch_idx)
        elif dataset_idx == 1:
            with torch.autograd.no_grad():
                if self.hparams.plmcmc.mcmc_during_epoch:
                    num_imputation_steps = self.num_imp_steps_schedule.get_value()
                    latent_reset_mode = self.latent_reset_schedule.get_value()
                    update_imputations = self.update_imputations_schedule.get_value()
                    logs = self.impute_batch(batch, stage='val',
                                             num_imputation_steps=num_imputation_steps,
                                             latent_reset_mode=latent_reset_mode,
                                             update_imputations=update_imputations)
                else:
                    logs = self.val_imp_log
                    # imp_time = self.val_imp_time
                    self.val_imp_log = {}
                    self.val_imp_time = 0.

                if self.hparams.plmcmc.perturb_std != 0:
                    batch[0] += torch.randn_like(batch[0])*self.hparams.plmcmc.perturb_std

                output = self.update_step(batch)
                loss = output['loss']
                output['progress_bar'].update(logs)
                output = {k.replace('train', 'val'): v
                          for k, v in output['progress_bar'].items()}
                output['val_loss'] = loss
                return output

    # Hooks

    # def on_batch_start(self, batch):
    #     # Runs only for training batch!
    #     self.impute_batch(batch)

    def on_epoch_start(self):
        """Before train epoch"""
        super().on_epoch_start()

        if self.hparams.plmcmc.mcmc_before_epoch:
            imp_start_time = time.time()

            train_data = torch.utils.data.DataLoader(
                                    self.train_dataset,
                                    batch_size=self.hparams.plmcmc.mcmc_batch_size,
                                    collate_fn=collate_augmented_samples,
                                    num_workers=2,
                                    shuffle=False)

            num_imputation_steps = self.num_imp_steps_schedule.get_value()
            latent_reset_mode = self.latent_reset_schedule.get_value()
            update_imputations = self.update_imputations_schedule.get_value()
            if update_imputations:
                for batch in tqdm(train_data, desc='Performing MCMC on train data.'):
                    # Transfer data to GPU
                    batch = self.transfer_batch_to_device(batch, self.device)

                    logs = self.impute_batch(batch, stage='train',
                                            num_imputation_steps=num_imputation_steps,
                                            latent_reset_mode=latent_reset_mode,
                                            update_imputations=update_imputations)

                    if self.num_imputed_copies_scheduler is not None:
                        prev_num_copies = self.num_imputed_copies_scheduler.get_value(self.current_epoch-1)
                        curr_num_copies = self.num_imputed_copies_scheduler.get_value(self.current_epoch)
                        # Check if the number of chains was increased, if so, we need to impute those copies
                        if curr_num_copies > prev_num_copies:
                            X, Y, M, I = batch[:4]
                            mask = self.train_dataset.which_samples_are_new(prev_num_copies,
                                                                            curr_num_copies,
                                                                            indices=I.cpu().numpy())
                            mask = torch.tensor(mask, device=X.device)
                            X_new, Y_new, M_new, I_new = X[mask], Y[mask], M[mask], I[mask]
                            # Impute the new chains
                            self.impute_batch((X_new, Y_new, M_new, I_new), stage='train',
                                            num_imputation_steps=self.hparams.data.num_new_chain_imp_steps,
                                            latent_reset_mode=latent_reset_mode,
                                            update_imputations=update_imputations)
                            # Set the imputed values into the batch
                            # X[mask], Y[mask], M[mask], I[mask] = X_new, Y_new, M_new, I_new

                self.train_imp_log = logs
                self.train_imp_time = time.time() - imp_start_time
            else:
                self.train_imp_time = 0
                self.train_imp_log = {}
        else:
            self.train_imp_time = 0
            self.train_imp_log = {}


    def on_pre_performance_check(self):
        super().on_pre_performance_check()

        if self.hparams.plmcmc.mcmc_before_epoch and self.hparams.plmcmc.debug.eval_incomplete:
            imp_start_time = time.time()

            val_data = torch.utils.data.DataLoader(
                                    self.val_dataset_augmented,
                                    batch_size=self.hparams.plmcmc.mcmc_batch_size,
                                    collate_fn=collate_augmented_samples,
                                    num_workers=2,
                                    shuffle=False)

            num_imputation_steps = self.num_imp_steps_schedule.get_value()
            latent_reset_mode = self.latent_reset_schedule.get_value()
            update_imputations = self.update_imputations_schedule.get_value()
            if update_imputations:
                for batch in tqdm(val_data, desc='Performing MCMC on val data.'):
                    # Transfer data to GPU
                    batch = self.transfer_batch_to_device(batch, self.device)

                    logs = self.impute_batch(batch, stage='val',
                                            num_imputation_steps=num_imputation_steps,
                                            latent_reset_mode=latent_reset_mode,
                                            update_imputations=update_imputations)

                    if self.num_imputed_copies_scheduler is not None:
                        prev_num_copies = self.num_imputed_copies_scheduler.get_value(self.current_epoch-1)
                        curr_num_copies = self.num_imputed_copies_scheduler.get_value(self.current_epoch)
                        # Check if the number of chains was increased, if so, we need to impute those copies
                        if curr_num_copies > prev_num_copies:
                            X, Y, M, I = batch[:4]
                            mask = self.train_dataset.which_samples_are_new(prev_num_copies,
                                                                            curr_num_copies,
                                                                            indices=I.cpu().numpy())
                            mask = torch.tensor(mask, device=X.device)
                            X_new, Y_new, M_new, I_new = X[mask], Y[mask], M[mask], I[mask]
                            # Impute the new chains
                            self.impute_batch((X_new, Y_new, M_new, I_new), stage='val',
                                            num_imputation_steps=self.hparams.data.num_new_chain_imp_steps,
                                            latent_reset_mode=latent_reset_mode,
                                            update_imputations=update_imputations)
                            # Set the imputed values into the batch
                            # X[mask], Y[mask], M[mask], I[mask] = X_new, Y_new, M_new, I_new

                self.val_imp_log = logs
                self.val_imp_time = time.time() - imp_start_time
            else:
                self.val_imp_time = 0
                self.val_imp_log = {}
        else:
            self.val_imp_time = 0
            self.val_imp_log = {}

    def training_epoch_end(self, outputs):
        results = super().training_epoch_end(outputs)

        # We want the total time spent on imputation
        # instead of average
        imp_time_total = 0.
        for output in outputs:
            for key, value in output.items():
                if key != 'imp_time':
                    continue
                imp_time_total += value

        results['log']['imp_time'] = imp_time_total
        results['progress_bar']['imp_time'] = imp_time_total

        return results

    def validation_epoch_end(self, outputs):
        if self.hparams.plmcmc.debug.eval_incomplete:
            results = OrderedDict({
                'log': {},
                'progress_bar': {}
            })
            # Parse outputs for each val dataset
            # Change key for secondary datasets
            for i, output in enumerate(outputs):
                result = super().validation_epoch_end(output)
                if i == 0:
                    results['log'].update(result['log'])
                    results['progress_bar'].update(result['progress_bar'])
                elif i == 1:
                    results['log'].update({f'aug_{k}': v
                                           for k, v in result['log'].items()})
                    results['progress_bar'].update({f'aug_{k}': v
                                                    for k, v in result['progress_bar'].items()})

            return results
        else:
            return super().validation_epoch_end(outputs)

    # def on_epoch_end(self):
    #     if self.hparams.cdi.debug.log_dataset:
    #         with torch.no_grad():
    #             # Load training data, and compute its log-prob under the
    #             # current model
    #             for batch in self.train_dataloader():
    #                 # Transfer data to GPU
    #                 if self.hparams.gpus is not None:
    #                     device = torch.device('cuda')
    #                 else:
    #                     device = torch.device('cpu')
    #                 batch = self.transfer_batch_to_device(batch, device)

    #                 # Compute log-prob
    #                 P, _ = self.forward(batch[0], batch[1])
    #                 self.logger.accumulate_tensors('data',
    #                                                X=batch[0].cpu(),
    #                                                M=batch[1].cpu(),
    #                                                I=batch[2].cpu(),
    #                                                P=P.cpu())

    #             # Save the accumulated tensors
    #             self.logger.save_accumulated_tensors('data',
    #                                                  self.current_epoch)

