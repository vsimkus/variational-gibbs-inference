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


class PLMCMC(cm.CompleteMLE):
    """
    Projected Latent MCMC EM for normalising flows.

    Canella et al. 2021 "Projected Latent Markov Chain Monte Carlo: Conditional Sampling of Normalizing Flows"
    """

    def __init__(self, hparams):
        super(PLMCMC, self).__init__(hparams)

        assert not (self.hparams.plmcmc.mcmc_before_epoch and self.hparams.plmcmc.mcmc_during_epoch),\
            ('Cannot set both \'mcmc_before_epoch\' and \'mcmc_during_epoch\' to true.')

        self._init_schedulers()

    def _init_schedulers(self):
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
        parser = super(PLMCMC, PLMCMC).add_model_args(parent_parser, args)

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

        parser.add_argument('--plmcmc.remove_aux_term', type=parse_bool,
                            default=False,
                            help=('Removes the auxilliary distribution term from MH acceptance criterion.'))
        parser.add_argument('--plmcmc.remove_logabset_term', type=parse_bool,
                            default=False,
                            help=('Removes the logabsdet terms from the MH acceptance criterion. Not in their paper, but in their code.'))
        parser.add_argument('--plmcmc.approximate_kernel', type=parse_bool,
                            default=True,
                            help=('Approximates the mixture-kernel as in the original paper.'))

        parser.add_argument('--plmcmc.clamp_during_mcmc', type=parse_bool,
                            default=False,
                            help=('Clamp accepted values to observed range during mcmc.'))

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

    def sample_imputations(self, batch, num_steps, r_prob, r_std, p_std, q_std, perturb_std=0.01,
                           remove_aux_term=False, approximate_kernel=True, remove_logabset_term=False,
                           clamp_to_range=False):
        X, Y, M, I, *_ = batch
        M_not = ~M

        acceptances = 0
        tries = 0
        with torch.no_grad():
            bernoulli = torch.distributions.Bernoulli(probs=torch.tensor(1-r_prob).to(X.device))
            for _ in range(num_steps):
                # Choose which to resample completely
                # 1 - perturbation
                # 0 - resampling
                resample_mask = bernoulli.sample(sample_shape=[X.shape[0], 1]).bool()
                resample_mask_not = ~resample_mask

                noise, noise_logabsdet = self.fa_model.transform_to_noise_and_logabsdetJ(Y)
                # The logabsdet Jacobian of f^-1 is the negative of the same of f
                noise_logabsdet *= -1
                # The above seemed to be accurate only up to second decimal place...
                # _, noise_logabsdet = self.fa_model.transform_from_noise_and_logabsdetJ(noise)

                # Construct proposals
                # -> Perturbation kernel
                noise_proposals = (noise + torch.randn_like(noise)*p_std)*resample_mask
                # -> Completely resampled states
                noise_proposals += (torch.randn_like(noise)*r_std)*resample_mask_not

                # Transform the proposals to the data space
                Y_proposal, noise_proposal_logabsdet = self.fa_model.transform_from_noise_and_logabsdetJ(noise_proposals)

                # Perturb observed data
                # NOTE: This is not in the paper, but in their code.
                # TODO: Why perturb the observed points?
                if perturb_std == 0:
                    # Option to *not* perturb the data

                    if remove_aux_term:
                        bayes_mod = 0.
                    else:
                        # This corresponds to the log(q(y_o')/q(y_o))
                        bayes_mod = (((1/q_std)**2)/2)*((((Y-X)*M)**2).sum(dim=1)
                                                        - (((Y_proposal-X)*M)**2).sum(dim=1))

                    # Project the observed values onto the proposals
                    X_proposal = Y_proposal*M_not + X*M
                    X_proposal_logprob = self.fa_model.log_prob(X_proposal)
                    X_perturbed_obs = X
                    X_perturbed_obs_logprob = self.fa_model.log_prob(X_perturbed_obs)
                else:
                    perturbations = torch.randn_like(X)*perturb_std

                    if remove_aux_term:
                        bayes_mod = 0.
                    else:
                        # Ignoring the perturbation, this corresponds to the log(q(y_o')/q(y_o))
                        bayes_mod = (((1/q_std)**2)/2)*((((Y-X+perturbations)*M)**2).sum(dim=1)
                                                        - (((Y_proposal-X+perturbations)*M)**2).sum(dim=1))

                    # Project the observed values onto the proposals
                    # TODO: Why perturb the observed data?
                    X_proposal = Y_proposal*M_not + (X+perturbations)*M
                    X_proposal_logprob = self.fa_model.log_prob(X_proposal)
                    X_perturbed_obs = X + perturbations*M
                    X_perturbed_obs_logprob = self.fa_model.log_prob(X_perturbed_obs)

                if approximate_kernel:
                    # Use the approximation from the original paper, assumes r_std >> p_std
                    # Then the proposal ratio of the perturbed latents cancels out.
                    # (eps2 - eps1)^2/sigma^2 - (eps1 - eps2)^2/sigma^2
                    # The proposal log-ratio of the resampled latents
                    noise_transition_log_ratio = ((((noise_proposals**2).sum(1).unsqueeze(1) - (noise**2).sum(1).unsqueeze(1))*resample_mask_not)/(r_std**2)/2).squeeze(1)
                else:
                    d = noise_proposals.shape[-1]

                    ### Compute the mixture terms for the noise_proposal

                    # Compute the resampled probability
                    resamp_log_norm_const = -d/2*np.log(2*np.pi) - d*np.log(r_std)
                    noise_proposal_resampled_log_prob = resamp_log_norm_const - (noise_proposals**2).sum(dim=1)/(r_std**2)/2
                    # Add the log-probability of the component
                    noise_proposal_resampled_log_prob += np.log(r_prob)

                    # Compute the perturbed probability
                    perturb_log_norm_const = -d/2*np.log(2*np.pi) - d*np.log(p_std)
                    noise_proposal_perturbed_log_prob = perturb_log_norm_const - ((noise_proposals - noise)**2).sum(dim=1)/(p_std**2)/2
                    # Add the log-probability of the component
                    noise_proposal_perturbed_log_prob += np.log(1-r_prob)

                    # Compute the mixture log-probability for the noise_proposal using log-sum-exp-trick
                    noise_proposal_max = torch.max(noise_proposal_resampled_log_prob, noise_proposal_perturbed_log_prob)
                    noise_proposal_min = torch.min(noise_proposal_resampled_log_prob, noise_proposal_perturbed_log_prob)
                    noise_proposal_mixture_logprob = noise_proposal_max + torch.log(1 + torch.exp(noise_proposal_min - noise_proposal_max))

                    ### Compute the mixture terms for the noise

                    # Compute the resampled probability
                    noise_resampled_log_prob = resamp_log_norm_const - (noise**2).sum(dim=1)/(r_std**2)/2
                    # Add the log-probability of the component
                    noise_resampled_log_prob += np.log(r_prob)

                    # The perturbed probability if the same!
                    noise_perturbed_log_prob = noise_proposal_perturbed_log_prob

                    # Compute the mixture log-probability for the noise_proposal using log-sum-exp-trick
                    noise_max = torch.max(noise_resampled_log_prob, noise_perturbed_log_prob)
                    noise_min = torch.min(noise_resampled_log_prob, noise_perturbed_log_prob)
                    noise_mixture_logprob = noise_max + torch.log(1 + torch.exp(noise_min - noise_max))

                    noise_transition_log_ratio = noise_mixture_logprob - noise_proposal_mixture_logprob

                # The original implementation did not include the Jacobian log-ratio terms.
                # While for the NICE model this is ok, since the terms cancel, for GLOW or
                # spline flows, where the transformation parameters are conditional on the data
                # this is *not* ok. Added this flag to see the effect of not including this term.
                if remove_logabset_term:
                    noise_logabsdet_log_ratio = 0.
                else:
                    noise_logabsdet_log_ratio = noise_proposal_logabsdet - noise_logabsdet

                # Compute the proposal acceptance probability
                acceptance_prob = (bayes_mod
                                   # The log-likelihood term ratio
                                   + X_proposal_logprob - X_perturbed_obs_logprob
                                   # The transition proposal ratio
                                   + noise_transition_log_ratio
                                   # The log absolute jacobian determinant terms
                                   + noise_logabsdet_log_ratio
                                   ).unsqueeze(1)

                # Thresholding from the original code
                # Same as clamping to (-25, 0) due to the min(1, a_prob) operation in Metropolis-Hastings
                acceptance_prob = torch.exp(torch.clamp(acceptance_prob, -25, 25))

                acceptance_samples = torch.rand_like(acceptance_prob)
                accepted = acceptance_samples < acceptance_prob
                acceptances += float(accepted.sum())
                tries += float(len(accepted))

                if clamp_to_range:
                    torch.min(Y_proposal, self.X_max.to(X.device), out=Y_proposal)
                    torch.max(Y_proposal, self.X_min.to(X.device), out=Y_proposal)

                # Create the eventual new sample (accepted)
                # noise = noise_proposals*accepted + noise*(~accepted)
                Y.data = (Y_proposal*accepted + Y*(~accepted)).data
                X.data = (Y*M_not + X*M).data

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
                                                                perturb_std=self.hparams.plmcmc.perturb_std,
                                                                remove_aux_term=self.hparams.plmcmc.remove_aux_term,
                                                                approximate_kernel=self.hparams.plmcmc.approximate_kernel,
                                                                remove_logabset_term=self.hparams.plmcmc.remove_logabset_term,
                                                                clamp_to_range=self.hparams.plmcmc.clamp_during_mcmc)

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

