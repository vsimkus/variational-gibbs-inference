import math
import time
from collections import OrderedDict
from enum import Enum

import numpy as np
import torch

import cdi.trainers.complete_mle as cm
from cdi.util.arg_utils import parse_bool
from cdi.util.utils import EpochScheduler
from cdi.util.data.data_augmentation_dataset import DataAugmentation, \
                                                    collate_augmented_samples, \
                                                    DataAugmentationWithScheduler
from cdi.util.data.fully_missing_filter_dataset import FullyMissingDataFilter


class UpdateComponentsEnum(Enum):
    ALL = 0
    MISSING = 1


class CDI(cm.CompleteMLE):
    """
    Maximum likelihood estimation (MLE) using
    cumulative data imputation (CDI) algorithm.
    Base class for CDI implementations.
    """
    IMP_ACCEPT_LOG_PROB_THRESH = -500

    def __init__(self, hparams, model=None):
        super().__init__(hparams, model=model)

        self._init_schedulers()

    def _init_schedulers(self):
        self.update_components = \
            UpdateComponentsEnum[self.hparams.cdi.update_components.upper()]
        self.update_comp_schedule = EpochScheduler(
                    self,
                    self.hparams.cdi.update_comp_schedule,
                    self.hparams.cdi.update_comp_schedule_values)
        self.imputation_comp_schedule = EpochScheduler(
                    self,
                    self.hparams.cdi.imputation_comp_schedule,
                    self.hparams.cdi.imputation_comp_schedule_values)
        self.num_imp_steps_schedule = EpochScheduler(
                    self,
                    self.hparams.cdi.num_imp_steps_schedule,
                    self.hparams.cdi.num_imp_steps_schedule_values)
        self.imp_acceptance_check_schedule = EpochScheduler(
                    self,
                    self.hparams.cdi.imp_acceptance_check_schedule,
                    self.hparams.cdi.imp_acceptance_check_schedule_values)

    @staticmethod
    def add_model_args(parent_parser, args=None):
        parser = super(CDI, CDI).add_model_args(parent_parser, args)

        # CDI args
        parser.add_argument('--cdi.num_samples',
                            type=int, required=True,
                            help=('The number of samples from the '
                                  'posterior distribution to '
                                  'approximate the expectation.'))
        parser.add_argument('--cdi.update_components',
                            type=str, required=True, default="MISSING",
                            help=('Which components should be considered '
                                  'in update `ALL` or `MISSING`.'))
        parser.add_argument('--cdi.update_comp_schedule',
                            type=int, nargs='+', required=True,
                            help=('A list of epochs when component fractions '
                                  'should be changed.'))
        parser.add_argument('--cdi.update_comp_schedule_values',
                            type=float, nargs='+', required=True,
                            help=('A list of values that correspond to each'
                                  ' scheduled time. The values determine the'
                                  ' fraction of components per example that '
                                  'are used in the update.'))
        parser.add_argument('--cdi.sample_imputation',
                            type=parse_bool, default=True,
                            help=('Whether to impute with a posterior '
                                  'sample, otherwise imputes with '
                                  'posterior mean.'))
        parser.add_argument('--cdi.imp_acceptance_check_schedule',
                            type=int, nargs='+', required=True,
                            help=('A list of epochs when acceptance behaviour '
                                  'should be changed.'))
        parser.add_argument('--cdi.imp_acceptance_check_schedule_values',
                            type=parse_bool, nargs='+', required=True,
                            help=('Whether to perform an acceptance check '
                                  'before imputing a dataset with new '
                                  'samples.'))
        parser.add_argument('--cdi.imputation_delay',
                            type=int, required=True,
                            help=('The number of epochs to wait before '
                                  'starting to impute missing values '
                                  'in the dataset.'))
        parser.add_argument('--cdi.num_imp_steps_schedule',
                            type=int, nargs='+', required=True,
                            help=('A list of epochs when number of imputation'
                                  ' steps should be changed.'))
        parser.add_argument('--cdi.num_imp_steps_schedule_values',
                            type=int, nargs='+', required=True,
                            help=('A list of values that correspond to each'
                                  ' schedule\'s mode. The number of imputation'
                                  ' steps in each epoch.'))
        parser.add_argument('--cdi.imputation_comp_schedule',
                            type=int, nargs='+', required=True,
                            help=('A list of epochs when imputation mode '
                                  'should be changed.'))
        parser.add_argument('--cdi.imputation_comp_schedule_values',
                            type=float, nargs='+', required=True,
                            help=('A list of values that correspond to each'
                                  ' schedule\'s mode. The values determine the'
                                  ' fraction of components per example that '
                                  'are used in the update. -1 corresponds to '
                                  'exactly one per example.'))
        parser.add_argument('--cdi.entropy_coeff',
                            type=float, default=1.0,
                            help=('Coefficient for the entropy term in '
                                  'loss computation.'))
        parser.add_argument('--cdi.add_selected_to_mask',
                            type=parse_bool,
                            help=('Add the selected variable as observed '
                                  'to the missingness mask'))
        parser.add_argument('--cdi.train_ignore_nans', default=False,
                            type=parse_bool, help=('Whether nans and infs in the vcdi '
                                                   'loss should be ignored. Could be used '
                                                   'when evaluating validation with refitting.'))

        parser.add_argument('--cdi.split_computation', default=True,
                            type=parse_bool, help=('Whether CDI should split log-likelihood '
                                                   'computation between full/incomplete data-points.'))

        # Debugging params
        parser.add_argument('--cdi.debug.log_dataset',
                            type=parse_bool, default=False,
                            help=('DEBUG: Logs the dataset state at the end of'
                                  ' the epoch.'))
        # parser.add_argument('--cdi.debug.log_num_updates_per_dim',
        #                     type=parse_bool, default=False,
        #                     help=('DEBUG: Logs the number of updates per '
        #                           'dimension in each epoch.'))
        parser.add_argument('--cdi.debug.eval_incomplete',
                            type=parse_bool, default=False,
                            help=('In addition evaluates validation on incomplete data,'
                                  'runs a chain of validation imputations similar to training.'))
        parser.add_argument('--cdi.debug.eval_ignore_nans', default=False,
                            type=parse_bool, help=('Whether nans and infs in the incomp. '
                                                   'val loss should be ignored.'))
        parser.add_argument('--cdi.debug.eval_imp_clip', default=False,
                            type=parse_bool, help=('Whether to clip validation imputations to min/max.'))
        parser.add_argument('--cdi.debug.eval_imp_bound_reject', default=False,
                            type=parse_bool, help=('Whether to reject validation imputations that fall out of min/max.'))

        return parser

    def setup(self, stage):
        super().setup(stage)

        if stage == 'fit' and (hasattr(self.hparams.cdi, 'debug') and self.hparams.cdi.debug.eval_incomplete):
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

        if (hasattr(self.hparams.cdi, 'debug') and ((hasattr(self.hparams.cdi.debug, 'eval_imp_clip') and self.hparams.cdi.debug.eval_imp_clip)
                or (hasattr(self.hparams.cdi.debug, 'eval_imp_bound_reject') and self.hparams.cdi.debug.eval_imp_bound_reject))):
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

    def val_dataloader(self):
        val_dataloader = super().val_dataloader()
        if hasattr(self.hparams.cdi, 'debug') and self.hparams.cdi.debug.eval_incomplete:
            val_aug_dataloader = torch.utils.data.DataLoader(
                                self.val_dataset_augmented,
                                batch_size=self.hparams.data.batch_size,
                                collate_fn=collate_augmented_samples,
                                num_workers=2,
                                shuffle=False)
            return [val_dataloader, val_aug_dataloader]

        return [val_dataloader]

    # Training

    def forward(self, X, M, M_not_which=None):
        if self.training:
            self.optim.add_run_opt('fa_model_opt')
            if hasattr(self, 'mis_model'):
                self.optim.add_run_opt('mis_model_opt')

        if (hasattr(self.hparams.cdi, 'add_selected_to_mask')
                and self.hparams.cdi.add_selected_to_mask
                and M_not_which is not None):
            # "completed data" but also marginalise the extra variables, if the model allows
            # Should be used with VAE's "marginalise option"
            fa_log_probs = self.fa_model(X, M+M_not_which)
        else:
            fa_log_probs = self.fa_model(X, M)

        mis_log_probs = None
        if hasattr(self, 'mis_model'):
            mis_log_probs = self.mis_model(X, M)

        return fa_log_probs, mis_log_probs

    def select_fraction_components_per_example(self, X, l):
        """
        Selects a random subset of l*X.shape[-1] variables in each sample
        of X and outputs a binary mask.
        Args:
            X (N, D): used for shape.
            l (float): fraction per example to be selected.
                If -1, then select exactly 1.
        Returns:
            M_selected (N, D): A mask shape of X, but only a subset
                (l*X.shape[-1] per sample) of values are set to missing (0).
        """
        M_selected = torch.ones_like(X)
        if l == 1:
            return ~(M_selected.bool())
        elif l == 0:
            return M_selected.bool()

        # Find out how many variables we should select for each example
        num_l = 1 if l == -1 else math.ceil(X.shape[-1]*l)

        # Select num_l variables per example
        choices = torch.multinomial(M_selected, num_l, replacement=False)
        M_selected[torch.arange(M_selected.shape[0]).reshape(-1, 1),
                   choices] = 0.

        return M_selected.bool()

    def select_fraction_components_per_example_from_M(self, M, l):
        """
        Selects a subset of M in each example (l*M.shape[-1] per example).
        Args:
            M (N, D): missingness mask for the batch.
                1 - observed, 0 - missing.
            l (float): fraction per example to be selected.
                If -1, then select exactly 1.
        Returns:
            M_selected (N, D): A mask similar to M, but only a subset
                (l*M.shape[-1] per sample) of values are set to missing (0).
        """
        # If we are to select all missing components - return immediately
        if l == 1:
            return M
        elif l == 0:
            return torch.ones_like(M)

        # Get missing indices
        M_not = (~M).float()
        # Get mask for rows that are not fully observed
        m_num_missing = M_not.sum(dim=1)
        m_incomplete = m_num_missing > 0

        M_selected = torch.ones_like(M)
        if l == -1:
            # Sample 1 missing variable for each row from the available
            # missing values
            choices = torch.multinomial(M_not[m_incomplete, :], 1).flatten()
            # Set the sampled missing variable as missing
            M_selected[m_incomplete, choices] = False
        else:
            # Find out how many variables we should select for each example
            num_l = torch.ceil(m_num_missing*l).to(torch.long)[m_incomplete]

            # Uniform sampling among the missing
            M_selected_incomp = torch.rand(m_incomplete.sum(), M_not.shape[-1], device=M.device)*M_not[m_incomplete, :]

            # Get num_l-th largest in each row
            M_selected_sorted, _ = torch.sort(M_selected_incomp,
                                              dim=1,
                                              descending=True)
            k_th = M_selected_sorted[torch.arange(len(num_l), device=M.device), num_l-1]
            M_selected_incomp = M_selected_incomp < k_th[:, None]
            M_selected[m_incomplete, :] = M_selected_incomp

        return M_selected

    def generate_Gibbs_scan_order_from_M(self, M, T):
        """
        Generates a Gibbs sampling scan order using M of length T
        Args:
            M (N, D): missingness mask for the batch.
                1 - observed, 0 - missing.
            T (int): length of scan order
        Returns:
            scan_order (N, T): indices of scan order
        """
        # Get missing indices
        M_not = (~M).float()
        # Get mask for rows that are not fully observed
        m_num_missing = M_not.sum(dim=1)
        m_incomplete = m_num_missing > 0
        num_incomplete = m_incomplete.sum()

        repeat = int(torch.max(torch.ceil(torch.true_divide(T, m_num_missing[m_incomplete]))).item())
        # Uniform sampling among the missing dimensions
        score = (torch.rand(num_incomplete, repeat, M.shape[-1], device=M.device)
                 + torch.arange(repeat, device=M.device)[None, :, None])
        score *= M_not[m_incomplete, None, :]
        score = score.reshape(num_incomplete, -1)

        # Get T largest in each row
        _, scan_order = torch.sort(score, dim=1, descending=True)
        out = torch.full((M.shape[0], scan_order.shape[-1]), fill_value=-1, dtype=torch.long, device=M.device)
        out[m_incomplete, :] = scan_order % M.shape[-1]
        out = out[:, :T]

        return out, m_incomplete

    def compute_univariate_posteriors(self, batch, M_selected, sample_all):
        """
        Compute univariate posteriors for all values that are missing
        in M_selected
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def complete_all_dims(self, X, M, X_samples):
        """
        Completes the incomplete tensor X, with samples from X_samples
        according to missingness mask M.
        """
        M_not = ~M

        num_mis = M_not.sum(axis=1)
        x_ref_idx = torch.arange(X.shape[0], device=X.device)

        # Spread each missing mask such that each row has only one 1:
        # e.g. [[1, 0, 1] -> [[1, 0, 0],
        #       [0, 0, 1]]    [0, 0, 1],
        #                     [0, 0, 1]]
        eye = torch.eye(X.shape[-1], X.shape[-1],
                        device=X.device, dtype=torch.bool)
        eye = eye.unsqueeze(0).expand(X.shape[0], -1, -1)
        M_not_expanded = eye[M_not, :]

        # Repeat each data-point # of missing values that it has
        # NOTE: this removes rows that are fully observed (i.e. repeat 0 times)
        X = X.repeat_interleave(num_mis, dim=0)
        x_ref_idx = x_ref_idx.repeat_interleave(num_mis, dim=0)

        # Repeat each data-point # of imputation times
        X = X.unsqueeze(0).repeat(X_samples.shape[0], 1, 1)
        x_ref_idx = x_ref_idx.unsqueeze(0).expand(X_samples.shape[0], -1)
        M_not_expanded = M_not_expanded.unsqueeze(0).expand(X_samples.shape[0], -1, -1)
        M_not = M_not.unsqueeze(0).expand(X_samples.shape[0], -1, -1)

        # Set values
        # NOTE: this uses a HUGE amount of memory on both forward and backward
        # it would great to improve this, but not sure if it's possible.
        X[M_not_expanded] = X_samples[M_not]

        # Unroll
        X = X.reshape(-1, X.shape[-1])
        x_ref_idx = x_ref_idx.reshape(-1)
        M_not_expanded = M_not_expanded.reshape(-1, M_not_expanded.shape[-1])

        return X, x_ref_idx, M_not_expanded

    def sample_imp_acceptance_mask(self, X_imp, M, M_selected):
        """
        Check that the sampled imputations are not outliers under the
        current model, reject otherwise.
        Args:
            X_imp: Imputed data X.
            M_selected: Which values are treated "missing".
        Return:
            accept_mask: Array of length N, with 1s for samples that
                should be accepted.
        """
        # Evaluate log-probability of the new samples under the current model
        log_probs = self.forward(X_imp, M, ~M_selected)[0]

        # Hard threshold for rejecting very unlikely imputations
        return log_probs > self.IMP_ACCEPT_LOG_PROB_THRESH

    def impute_batch(self, batch, stage, l, num_imputation_steps):
        """
        TODO: clean this up
        Impute dataset at the start of each batch.
        batch:
            X (N, D): observable variables
            M (N, D): binary missingness mask.
            I (N,): indices of the X samples in the dataset
                (can used for imputation where necessary)
        """
        X, M, I = batch[:3]

        total_rejected = 0
        # Do not impute the dataset at the beginning when the
        # posterior/variational samples are poor
        if self.current_epoch >= self.hparams.cdi.imputation_delay:
            # l = self.imputation_comp_schedule.get_value()
            # num_imputation_steps = self.num_imp_steps_schedule.get_value()

            with torch.no_grad():
                if l == -1:
                    # Prepare scan order if we do proper Gibbs sampling
                    # (we can't really prepare the scan order with the block-Gibbs)
                    scan_order, incomp_mask = self.generate_Gibbs_scan_order_from_M(M, T=num_imputation_steps)
                    complete_mask = ~incomp_mask
                    # NOTE: here I set the scan order for complete data-points to zero,
                    # but we should not update anything for the complete data-points
                    scan_order[complete_mask, :] = 0

                for i in range(num_imputation_steps):
                    # Select values to be imputed in this step
                    if l != -1:
                        # If imputing in block-Gibbs scheme compute the M_selected in each iteration
                        M_selected = self.select_fraction_components_per_example_from_M(M, l)
                    else:
                        # Otherwise, use the precomputed scan order
                        M_selected = ~(torch.nn.functional.one_hot(scan_order[:, i], num_classes=M.shape[-1]).bool())
                        # Set to ones for fully observed
                        M_selected[complete_mask] = 1.

                    # # Select values to be imputed in this step
                    # M_selected = self.select_fraction_components_per_example_from_M(M, l)

                    X_imp = self.sample_imputation_values(
                                                    (X, M, I),
                                                    M_selected=M_selected,
                                                    sample_all=(l != -1))

                    # Clip validation imputation values to min/max
                    if stage == 'val' and hasattr(self.hparams.cdi.debug, 'eval_imp_clip') and self.hparams.cdi.debug.eval_imp_clip:
                        X_imp = torch.min(self.X_max.to(X), torch.max(self.X_min.to(X), X_imp))

                    # Reject imputation values that fall out of min/max box
                    if stage == 'val' and hasattr(self.hparams.cdi.debug, 'eval_imp_bound_reject') and self.hparams.cdi.debug.eval_imp_bound_reject:
                        M_selected = ~((self.X_min.to(X) <= X_imp) & (X_imp <= self.X_max.to(X)) & ~M_selected)

                    if self.imp_acceptance_check_schedule.get_value():
                        # Set observed values to their true values
                        X_imp[M_selected] = X[M_selected]

                        # Accept chain updates based on their log-probability under current model
                        imp_accept_mask = self.sample_imp_acceptance_mask(X_imp, M, M_selected)
                        # Set accepted samples
                        X[imp_accept_mask, :] = X_imp[imp_accept_mask, :]

                        # Compute stats
                        total_rejected = X.shape[0] - imp_accept_mask.sum()
                    else:
                        M_selected_not = ~M_selected
                        X[M_selected_not] = X_imp[M_selected_not]

                if stage == 'train':
                    self.train_dataset[I.cpu()] = X.cpu()
                elif stage == 'val':
                    self.val_dataset_augmented[I.cpu()] = X.cpu()
                elif stage == 'test':
                    self.test_dataset[I.cpu()] = X.cpu()

        logs = {
            'imp_rejected': total_rejected
        }
        return logs

    def compute_CDI_objective_terms(self, X, M, I, M_selected, l,
                                    set_full_lprob_to_zero=False):
        # Get number of missing values for each sample
        M_selected_not = ~M_selected
        L = M_selected_not.sum(dim=1)
        # Find incomplete sample mask
        # u - for update
        u_incomp_mask = L != 0
        u_full_mask = ~u_incomp_mask
        u_num_incomp = u_incomp_mask.sum()
        u_num_full = X.shape[0] - u_num_incomp
        # Sample #num_samples values for each missing value
        # and compute the entropies of all N*D posterior distributions
        if u_num_incomp > 0:
            X_samples, entropy = self.sample_missing_values(
                                                (X, M, I),
                                                M_selected=M_selected,
                                                K=self.hparams.cdi.num_samples,
                                                sample_all=(l != -1))

        # Compute log-prob term for each sample
        if l != -1:
            fa_log_probs = torch.zeros(X.shape[0]).type_as(X)
            mis_log_probs = None
            if hasattr(self, 'mis_model'):
                mis_log_probs = torch.zeros(X.shape[0]).type_as(X)
            # Check if there are incomplete samples,
            # If so, impute and then
            # compute avg. log-lik on completed samples
            if u_num_incomp > 0:
                # M_not_selected_completed has a 1 for the selected variable
                X_completed, x_ref_idx, M_not_selected_completed = self.complete_all_dims(
                                                        X, M_selected,
                                                        X_samples)
                M_incomp = M[x_ref_idx, :]
                # Compute log_prob of the filled-in samples
                # And take average over the K samples approximating
                # the expectation
                # log_prob = self.forward(X_completed, M_incomp) / X_samples.shape[0]
                fa_log_prob, mis_log_prob = self.forward(X_completed, M_incomp, M_not_selected_completed)
                fa_log_prob = fa_log_prob / X_samples.shape[0]
                # index_add can be source on non-determinism on a GPU
                fa_log_probs = fa_log_probs.index_add(0, x_ref_idx, fa_log_prob)
                if mis_log_prob is not None:
                    mis_log_prob = mis_log_prob / X_samples.shape[0]
                    mis_log_probs = mis_log_probs.index_add(0, x_ref_idx, mis_log_prob)

            # Evaluate on full samples
            if u_num_full > 0:
                if not set_full_lprob_to_zero:
                    X_full = X[u_full_mask, :]
                    M_full = M[u_full_mask, :]
                    # log_probs[u_full_mask] = self.forward(X_full, M_full)
                    fa_log_prob, mis_log_prob = self.forward(X_full, M_full)
                    fa_log_probs[u_full_mask] = fa_log_prob
                    if mis_log_prob is not None:
                        mis_log_probs[u_full_mask] = mis_log_prob
                else:
                    fa_log_probs[u_full_mask] = 0.
                    if mis_log_probs is not None:
                        mis_log_probs[u_full_mask] = 0.

            # Take average over all univariate ELBOs for the
            # log-likelihood term
            L[u_full_mask] = 1
            fa_log_probs /= L.float()
            if mis_log_probs is not None:
                mis_log_probs /= L.float()
        else:
            fa_log_probs = torch.zeros(X.shape[0]).type_as(X)
            mis_log_probs = None
            if hasattr(self, 'mis_model'):
                mis_log_probs = torch.zeros(X.shape[0]).type_as(X)
            if not hasattr(self.hparams.cdi, 'split_computation') or self.hparams.cdi.split_computation:
                # Impute and then compute avg. log-lik on completed samples
                if u_num_incomp > 0:
                    # Set observed values to the actual values
                    X_samples[:, M_selected] = X[M_selected]
                    X_samples_incomp = X_samples[:, u_incomp_mask, :]
                    X_samples_incomp_shape = X_samples_incomp.shape
                    X_samples_incomp = X_samples_incomp.reshape(
                                        -1, X_samples_incomp_shape[-1])
                    M_samples_incomp = (M[u_incomp_mask, :]
                                        .expand(self.hparams.cdi.num_samples, -1, -1))
                    M_samples_incomp = M_samples_incomp.reshape(
                                        -1, X_samples_incomp_shape[-1])
                    # log_prob = self.forward(X_samples_incomp, M_samples_incomp)
                    fa_log_prob, mis_log_prob = self.forward(X_samples_incomp, M_samples_incomp, ~M_selected)
                    # Take average over the K samples approximating the expectation
                    fa_log_probs[u_incomp_mask] = fa_log_prob.reshape(
                                        X_samples_incomp_shape[0], -1).mean(dim=0)
                    if mis_log_prob is not None:
                        mis_log_probs[u_incomp_mask] = mis_log_prob.reshape(
                                        X_samples_incomp_shape[0], -1).mean(dim=0)

                # Evaluate on full samples
                if u_num_full > 0:
                    if not set_full_lprob_to_zero:
                        X_full = X[u_full_mask, :]
                        M_full = M[u_full_mask, :]
                        # log_probs[u_full_mask] = self.forward(X_full, M_full)
                        fa_log_prob, mis_log_prob = self.forward(X_full, M_full)
                        fa_log_probs[u_full_mask] = fa_log_prob
                        if mis_log_prob is not None:
                            mis_log_probs[u_full_mask] = mis_log_prob
                    else:
                        fa_log_probs[u_full_mask] = 0.
                        if mis_log_probs is not None:
                            mis_log_probs[u_full_mask] = 0.
            else:
                # Set observed values to the actual values
                X_samples[:, M_selected] = X[M_selected]
                X_samples_shape = X_samples.shape
                X_samples = X_samples.reshape(-1, X_samples_shape[-1])
                M_samples = M.expand(self.hparams.cdi.num_samples, -1, -1)
                M_samples = M_samples.reshape(-1, X_samples_shape[-1])
                fa_log_prob, mis_log_prob = self.forward(X_samples, M_samples, ~M_selected)
                # Take average over the K samples approximating the expectation
                fa_log_probs = fa_log_prob.reshape(X_samples_shape[0], -1).mean(dim=0)
                if mis_log_prob is not None:
                    mis_log_probs = mis_log_prob.reshape(
                                    X_samples_shape[0], -1).mean(dim=0)

            L[u_full_mask] = 1

        # Compute entropy term for each sample
        if u_num_incomp > 0:
            # Compute the average entropy of missing variable posterior
            # distribution over the batch. Entropies of observed value
            # distributions are -inf, so replace them with 0s in the sum
            entropy[M_selected] = 0.
            # Compute average entropy for each sample (over all M_selected)
            entropy = entropy.sum(dim=1)
            entropy /= L  # Where L is the number of missing values in each sample
        else:
            entropy = torch.zeros(X.shape[0]).type_as(X)

        # if self.hparams.cdi.debug.log_num_updates_per_dim:
        if not hasattr(self, 'num_selected_per_dim'):
            self.num_updates_per_dim = (torch.zeros(X[0].shape, dtype=torch.int32)
                                        .unsqueeze(0))
        num_selected_per_dim = M_selected_not.sum(dim=0).cpu()*self.hparams.cdi.num_samples
        self.num_updates_per_dim += num_selected_per_dim

        return fa_log_probs, mis_log_probs, entropy

    def update_step(self, batch, only_var=False):
        """
        One iteration of MLE update using CDI algorithm.
        batch:
            X (N, D): observable variables
            M (N, D): binary missingness mask.
            I (N,): indices of the X samples in the dataset
                (can used for imputation where necessary)
        """
        X, M, I, OI, incomp_mask = batch
        # Compute the number of complete samples
        # As well as the true number of samples from the original dataset
        # N_incomp = incomp_mask.sum()
        # N_full = incomp_mask.shape[0] - N_incomp
        # N_true = N_full + (N_incomp / self.hparams.data.num_imputed_copies)

        if (hasattr(self.hparams.cdi, 'train_ignore_nans')
                and self.hparams.cdi.train_ignore_nans):
            # Because a nan in the input will set weights to nan, even if its
            # output is not used in the loss, we need to set the inputs to a
            # numerical value.
            X_nans = (~torch.isfinite(X)).any(dim=1)
            X[X_nans] = 0.

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

        # A milestone when to switch from the fraction of components in
        # an update
        l = self.update_comp_schedule.get_value()

        # Choose update components: from ALL or MISSING only
        if self.update_components == UpdateComponentsEnum.ALL:
            M_selected = self.select_fraction_components_per_example(X, l)
        else:  # self.update_components == UpdateComponentsEnum.MISSING:
            M_selected = self.select_fraction_components_per_example_from_M(M, l)
            # If we're updating only variational model, then the full samples
            # will be not be used.
            if only_var:
                N_full = (~incomp_mask).sum()
                N_true -= N_full

        # Evaluate the CDI objective
        fa_log_probs, mis_log_probs, entropy = self.compute_CDI_objective_terms(
                                        X, M, I,
                                        M_selected, l,
                                        # If we're only updating var model,
                                        # set full-sample (in M_selected)
                                        # probability to 0 without evaluating
                                        set_full_lprob_to_zero=only_var)

        nans = None
        if ((not self.training and self.hparams.cdi.debug.eval_ignore_nans)
                or (hasattr(self.hparams.cdi, 'train_ignore_nans') and self.hparams.cdi.train_ignore_nans)):
            # Which are nans or infs
            nans = ~torch.isfinite(fa_log_probs) | X_nans
            # Set nans to zero
            # NOTE: could use torch.nansum() below instead of this
            fa_log_probs[nans] = 0
            entropy[nans] = 0
            if mis_log_probs is not None:
                mis_log_probs[nans] = 0

            total_nans = nans.sum()
            # Remove nans from the count of samples (to take correct average)
            N_true -= total_nans

        # Divide the log_probs and entropies of incomplete samples
        # by the number of augmentations
        if num_imputed_copies > 1:
            if nans is not None:
                # Subtract one from the clone-count for each nan
                nans_idx = oi_inv_idx[nans]
                oi_counts.index_add_(0, nans_idx, torch.tensor([-1], device=X.device).expand(nans_idx.shape[0]))
                # if the count is 0 - avoid division by 0
                oi_counts[oi_counts == 0] = 1

            fa_log_probs /= oi_counts[oi_inv_idx]
            if mis_log_probs is not None:
                mis_log_probs /= oi_counts[oi_inv_idx]
            entropy /= oi_counts[oi_inv_idx]

        if mis_log_probs is not None:
            log_probs = fa_log_probs + mis_log_probs
        else:
            log_probs = fa_log_probs

        # Compute average log_prob and entropy.
        # Since the log_probs of augmented samples are now scaled down
        # We can compute the total average log_probability by dividing
        # total sum by the total *true* number of samples in the batch
        log_prob = log_probs.sum() / N_true
        entropy = entropy.sum() / N_true

        # Compute loss and update parameters (by maximising log-probability
        # and entropy)
        loss = -log_prob - self.hparams.cdi.entropy_coeff*entropy

        pbar = {
            'train_log_lik': log_prob.item(),
            'train_entropy': entropy.data.item()
        }
        output = OrderedDict({
            'loss': loss,
            'progress_bar': pbar,
        })
        pbar['train_fa_log_prob'] = (fa_log_probs.sum()/N_true).item()
        if nans is not None:
            pbar['train_total_nans'] = total_nans.item()
        if mis_log_probs is not None:
            pbar['train_mis_log_prob'] = (mis_log_probs.sum()/N_true).item()
        return output

    def training_step(self, batch, batch_idx):
        """
        Performs CDI update and imputation of missing values.
        """
        # Imputation
        imp_start_time = time.time()
        l = self.imputation_comp_schedule.get_value()
        num_imputation_steps = self.num_imp_steps_schedule.get_value()
        logs = self.impute_batch(batch, stage='train', l=l,
                                 num_imputation_steps=num_imputation_steps)

        if self.num_imputed_copies_scheduler is not None:
            prev_num_copies = self.num_imputed_copies_scheduler.get_value(self.current_epoch-1)
            curr_num_copies = self.num_imputed_copies_scheduler.get_value(self.current_epoch)
            # Check if the number of chains was increased, if so, we need to impute those copies
            if curr_num_copies > prev_num_copies:
                X, M, I = batch[:3]
                mask = self.train_dataset.which_samples_are_new(prev_num_copies,
                                                                curr_num_copies,
                                                                indices=I.cpu().numpy())
                mask = torch.tensor(mask, device=X.device)
                X_new, M_new, I_new = X[mask], M[mask], I[mask]
                # Impute the new chains
                self.impute_batch((X_new, M_new, I_new), stage='train', l=l,
                                  num_imputation_steps=self.hparams.data.num_new_chain_imp_steps)
                # Set the imputed values into the batch
                X[mask], M[mask], I[mask] = X_new, M_new, I_new

        imp_time = time.time() - imp_start_time

        # Update
        output = self.update_step(batch, only_var=False)
        output['progress_bar'].update(logs)
        output['progress_bar']['train_imp_time'] = imp_time
        return output

    # Validation

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        if dataset_idx == 0:
            return super().validation_step(batch, batch_idx)
        elif dataset_idx == 1:
            with torch.autograd.no_grad():
                l = self.imputation_comp_schedule.get_value()
                num_imputation_steps = self.num_imp_steps_schedule.get_value()
                self.impute_batch(batch, stage='val', l=l,
                                  num_imputation_steps=num_imputation_steps)

                if self.num_imputed_copies_scheduler is not None:
                    prev_num_copies = self.num_imputed_copies_scheduler.get_value(self.current_epoch-1)
                    curr_num_copies = self.num_imputed_copies_scheduler.get_value(self.current_epoch)
                    # Check if the number of chains was increased, if so, we need to impute those copies
                    if curr_num_copies > prev_num_copies:
                        X, M, I = batch[:3]
                        mask = self.val_dataset_augmented.which_samples_are_new(prev_num_copies,
                                                                        curr_num_copies,
                                                                        indices=I.numpy())
                        mask = torch.tensor(mask)
                        X_new, M_new, I_new = X[mask], M[mask], I[mask]
                        # Impute the new chains
                        self.impute_batch((X_new, M_new, I_new), stage='train', l=l,
                                          num_imputation_steps=self.hparams.data.num_new_chain_imp_steps)
                        # Set the imputed values into the batch
                        X[mask], M[mask], I[mask] = X_new, M_new, I_new

                output = self.update_step(batch, only_var=False)
                loss = output['loss']
                output = {k.replace('train', 'val'): v
                          for k, v in output['progress_bar'].items()}
                output['val_loss'] = loss
                return output

    # Hooks

    # def on_batch_start(self, batch):
    #     # Runs only for training batch!
    #     self.impute_batch(batch)

    def on_epoch_start(self):
        super().on_epoch_start()
        # if self.hparams.cdi.debug.log_num_updates_per_dim:
        if hasattr(self, 'num_updates_per_dim'):
            self.num_updates_per_dim = torch.zeros_like(
                                self.num_updates_per_dim)

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

        # Log the number of var model updates for each dimension
        if hasattr(self.hparams.cdi, 'debug') and hasattr(self.hparams.cdi.debug, 'log_num_updates_per_dim') and self.hparams.cdi.debug.log_num_updates_per_dim:
            self.logger.accumulate_tensors(
                            'num_var_updates',
                            num_updates=self.num_updates_per_dim,
                            curr_epoch=torch.tensor([self.current_epoch]))

        return results

    def validation_epoch_end(self, outputs):
        if hasattr(self.hparams.cdi, 'debug') and self.hparams.cdi.debug.eval_incomplete:
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

    def on_epoch_end(self):
        if hasattr(self.hparams.cdi, 'debug') and hasattr(self.hparams.cdi.debug, 'log_dataset') and self.hparams.cdi.debug.log_dataset:
            with torch.no_grad():
                # Load training data, and compute its log-prob under the
                # current model
                for batch in self.train_dataloader():
                    # Transfer data to GPU
                    if self.hparams.gpus is not None:
                        device = torch.device('cuda')
                    else:
                        device = torch.device('cpu')
                    batch = self.transfer_batch_to_device(batch, device)

                    # Compute log-prob
                    P, _ = self.forward(batch[0], batch[1])
                    self.logger.accumulate_tensors('data',
                                                   X=batch[0].cpu(),
                                                   M=batch[1].cpu(),
                                                   I=batch[2].cpu(),
                                                   P=P.cpu())

                # Save the accumulated tensors
                self.logger.save_accumulated_tensors('data',
                                                     self.current_epoch)

    # def on_after_backward(self):
    #     if self.current_epoch == 0:
    #         self.model_s = 0
    #         self.var_s = 0
    #         self.model_cache = {}
    #         self.var_cache = {}

    #     s_model = torch.tensor(0.)
    #     for i, p in enumerate(self.fa_model.parameters()):
    #         if self.current_epoch == 0:
    #             self.model_cache[i] = p.grad.data.flatten()
    #         else:
    #             grad = p.grad.data.flatten()
    #             s_model += self.model_cache[i] @ grad.T
    #             self.model_cache[i] = grad

    #     s_var = torch.tensor(0.)
    #     for i, p in enumerate(self.variational_model.parameters()):
    #         if self.current_epoch == 0:
    #             self.var_cache[i] = p.grad.data.flatten()
    #         else:
    #             grad = p.grad.data.flatten()
    #             s_var += self.var_cache[i] @ grad.T
    #             self.var_cache[i] = grad

    #     self.logger.log_metrics({
    #         's_model': s_model.item(),
    #         's_var': s_var.item()
    #     }, 0)

    #
    # Helpers
    #

    def systematic_gibbs_sampling(self, batch, num_passes):
        X, M, I = batch[:3]

        M_left = M.clone()
        while(M_left.sum() != M_left.numel()):
            M_selected = self.select_fraction_components_per_example_from_M(M_left, l=-1)
            M_selected_not = ~M_selected
            M_left = M_left + M_selected_not
            # Sample imputation value
            X_imp = self.sample_imputation_values(
                                            (X, M, I),
                                            M_selected,
                                            sample_all=False)
            # Impute batch
            X[M_selected_not] = X_imp[M_selected_not]

        return batch

    # def impute_with_systematic_gibbs_samples(self, dataset, num_passes=1):
    #     dataloader = torch.utils.data.DataLoader(
    #                             dataset,
    #                             batch_size=self.hparams.data.batch_size,
    #                             collate_fn=collate_augmented_samples,
    #                             num_workers=1,
    #                             shuffle=False)
    #     with torch.no_grad():
    #         for _ in range(num_passes):
    #             for batch in dataloader:
    #                 # Transfer data to GPU
    #                 if self.hparams.gpus is not None:
    #                     device = torch.device('cuda')
    #                 else:
    #                     device = torch.device('cpu')
    #                 batch = self.transfer_batch_to_device(batch, device)

    #                 # Run systematic Gibbs sampling
    #                 X, M, I = self.systematic_gibbs_sampling(batch, num_passes)[:3]

    #                 # Update data
    #                 dataset[I.cpu()] = X.cpu()

    def impute_with_systematic_gibbs_samples(self, dataset, num_imputation_steps, clip_imp_values=False, reject_imp_values=False):
        assert not (clip_imp_values and reject_imp_values), \
            'Clipping and rejecting cannot be used together!'

        dataloader = torch.utils.data.DataLoader(
                                dataset,
                                batch_size=self.hparams.data.batch_size,
                                collate_fn=collate_augmented_samples,
                                num_workers=1,
                                shuffle=False)

        var_model_before = self.variational_model
        if self.hparams.gpus is not None:
            device = torch.device('cuda')
            self.variational_model = self.variational_model.to(device)
        else:
            device = torch.device('cpu')

        print('Using device for pre-imputation:', device)

        if clip_imp_values or reject_imp_values:
            X = dataset[:][0]
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)

            if isinstance(X_min, np.ndarray):
                X_min = torch.tensor(X_min)
                X_max = torch.tensor(X_max)
            else:
                X_min = X_min[0]
                X_max = X_max[0]

            X_min = X_min.to(device)
            X_max = X_max.to(device)

        with torch.no_grad():
            for j, batch in enumerate(dataloader):
                print(f'Imputing batch {j}')

                # Transfer data to GPU
                batch = self.transfer_batch_to_device(batch, device)

                X, M, I = batch[:3]

                # Prepare scan order
                scan_order, incomp_mask = self.generate_Gibbs_scan_order_from_M(M, T=num_imputation_steps)
                complete_mask = ~incomp_mask
                # NOTE: here I set the scan order for complete data-points to zero,
                # but we should not update anything for the complete data-points
                scan_order[complete_mask, :] = 0

                for i in range(num_imputation_steps):
                    # Use the precomputed scan order
                    M_selected = ~(torch.nn.functional.one_hot(scan_order[:, i], num_classes=M.shape[-1]).bool())
                    # Set to ones for fully observed
                    M_selected[complete_mask] = 1.

                    X_imp = self.sample_imputation_values(
                                                    (X, M, I),
                                                    M_selected=M_selected,
                                                    sample_all=False)
                    if clip_imp_values:
                        X_imp = torch.min(X_max, torch.max(X_min, X_imp))

                    if reject_imp_values:
                        M_selected = ~((X_min <= X_imp) & (X_imp <= X_max) & ~M_selected)

                    # Don't impute if the imputation would be a nan
                    # imp_nans = ~torch.isfinite(X_imp)

                    M_selected_not = ~M_selected  # - imp_nans
                    X[M_selected_not] = X_imp[M_selected_not]

                dataset[I.cpu()] = X.cpu()

        self.variational_model = var_model_before
