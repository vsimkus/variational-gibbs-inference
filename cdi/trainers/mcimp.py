import copy
import os
import time
from collections import OrderedDict

import torch

import cdi.trainers.complete_mle as cm
from cdi.util.arg_utils import parse_bool
from cdi.util.data.data_augmentation_dataset import (
    DataAugmentation, DataAugmentationWithScheduler, collate_augmented_samples)
from cdi.util.data.fully_missing_filter_dataset import FullyMissingDataFilter
from cdi.util.utils import EpochScheduler
from cdi.util.data.persistent_latents_dataset import LatentsDataset


class MCIMP(cm.CompleteMLE):
    """
    Maximum likelihood estimation (MLE) using
    cumulative data imputation (CDI) algorithm.
    Base class for CDI implementations.
    """

    def __init__(self, hparams):
        super(MCIMP, self).__init__(hparams)

        self._init_schedulers()

        if hasattr(self.hparams.cdi, 'sample_ground_truth_model') and self.hparams.cdi.sample_ground_truth_model:
            f = os.path.join('trained_models', self.hparams.cdi.ground_truth_model_path)
            self.ground_model = cm.CompleteMLE.load_from_checkpoint(f).fa_model
            self.ground_model.freeze_decoder()
            self.ground_model.freeze_encoder()

    def _init_schedulers(self):
        self.num_imp_steps_schedule = EpochScheduler(
                    self,
                    self.hparams.cdi.num_imp_steps_schedule,
                    self.hparams.cdi.num_imp_steps_schedule_values)

    @staticmethod
    def add_model_args(parent_parser, args=None):
        parser = super(MCIMP, MCIMP).add_model_args(parent_parser, args)

        # CDI args
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
        parser.add_argument('--cdi.concat_latents', default=False,
                            type=parse_bool,
                            help=('Should concat latents to the X\'s in the dataset.'))
        parser.add_argument('--cdi.latent_dim', type=int,
                            help=('Latent dimensionality.'))

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
        # parser.add_argument('--cdi.debug.eval_imp_clip', default=False,
        #                     type=parse_bool, help=('Whether to clip validation imputations to min/max.'))
        # parser.add_argument('--cdi.debug.eval_imp_bound_reject', default=False,
        #                     type=parse_bool, help=('Whether to reject validation imputations that fall out of min/max.'))

        parser.add_argument('--cdi.sample_ground_truth_model', type=parse_bool,
                            default=False, help=('Whether use the ground truth model for sampling the imputation chain.'))
        parser.add_argument('--cdi.ground_truth_model_path', type=str,
                            default=None, help=('Path to the ground truth model weights.'))

        return parser

    def initialise_dataset(self, hparams, dataset):
        metrics = None
        if (hparams.data.pre_imputation == 'var_mcmc_using_model'):
            init_start_time = time.time()

            # First impute the data with the same method as used
            # in the pretraining
            temp_hparams = copy.deepcopy(hparams)
            temp_hparams.data.pre_imputation = temp_hparams.data.orig_pre_imputation
            super().initialise_dataset(temp_hparams, dataset)
            del temp_hparams

            # Skip the MCMC imputation here, and do it after concatenating the latents to the X's
            if not (hasattr(self.hparams.cdi, 'concat_latents') and self.hparams.cdi.concat_latents):
                print('VMCMC imputation using pretrained model.')

                self.pre_impute_using_model(dataset,
                                            num_imputation_steps=hparams.data.pre_imp_num_imputation_steps)

                metrics = {
                    'init_time': [time.time() - init_start_time],
                    'stage': ['initialise_dataset']
                }
        else:
            metrics = super().initialise_dataset(hparams, dataset)

        return metrics

    def setup(self, stage):
        super().setup(stage)

        if stage == 'fit':
            if hasattr(self.hparams.cdi, 'concat_latents') and self.hparams.cdi.concat_latents:
                self.train_dataset = LatentsDataset(self.train_dataset,
                                                    latent_dim=self.hparams.cdi.latent_dim)

                if (self.hparams.data.pre_imputation == 'var_mcmc_using_model'):
                    print('VMCMC imputation using pretrained model.')
                    self.pre_impute_using_model(self.train_dataset,
                                                num_imputation_steps=self.hparams.data.pre_imp_num_imputation_steps)

            if self.hparams.cdi.debug.eval_incomplete:
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

                if hasattr(self.hparams.cdi, 'concat_latents') and self.hparams.cdi.concat_latents:
                    self.val_dataset_augmented = LatentsDataset(self.val_dataset_augmented,
                                                                latent_dim=self.hparams.cdi.latent_dim)

                    if (self.hparams.data.pre_imputation == 'var_mcmc_using_model'):
                        print('VMCMC imputation using pretrained model.')
                        self.pre_impute_using_model(self.val_dataset_augmented,
                                                    num_imputation_steps=self.hparams.data.pre_imp_num_imputation_steps)

        # if ((hasattr(self.hparams.cdi.debug, 'eval_imp_clip') and self.hparams.cdi.debug.eval_imp_clip)
        #         or (hasattr(self.hparams.cdi.debug, 'eval_imp_bound_reject') and self.hparams.cdi.debug.eval_imp_bound_reject)):
        #     X = self.train_dataset[:][0]
        #     X_min = X.min(axis=0)
        #     X_max = X.max(axis=0)

        #     if isinstance(X_min, np.ndarray):
        #         X_min = torch.tensor(X_min)
        #         X_max = torch.tensor(X_max)
        #     else:
        #         X_min = X_min[0]
        #         X_max = X_max[0]

        #     self.X_min = X_min
        #     self.X_max = X_max

    def val_dataloader(self):
        val_dataloader = super().val_dataloader()
        if self.hparams.cdi.debug.eval_incomplete:
            val_aug_dataloader = torch.utils.data.DataLoader(
                                self.val_dataset_augmented,
                                batch_size=self.hparams.data.batch_size,
                                collate_fn=collate_augmented_samples,
                                num_workers=2,
                                shuffle=False)
            return [val_dataloader, val_aug_dataloader]

        return [val_dataloader]

    # Training

    def sample_missing_values(self, X, M):
        if hasattr(self.hparams.cdi, 'sample_ground_truth_model') and self.hparams.cdi.sample_ground_truth_model:
            # if self.hparams.data.obs_zero_mean:
            #     # The ground model is not zero-mean so have to add the mean and then subtract it.
            #     obs_mean = torch.tensor(self.obs_mean, device=X.device)
            #     X = X + obs_mean
            #     X_imp = self.ground_model.mcmc_sample_missing_values(X, M)
            #     return X_imp - obs_mean
            # else:
            return self.ground_model.mcmc_sample_missing_values(X, M)
        else:
            return self.fa_model.mcmc_sample_missing_values(X, M)

    def impute_batch(self, batch, stage, num_imputation_steps):
        """
        Impute dataset at the start of each batch.
        batch:
            X (N, D): observable variables
            M (N, D): binary missingness mask.
            I (N,): indices of the X samples in the dataset
                (can used for imputation where necessary)
        """
        X, M, I = batch[:3]
        M_not = ~M

        # Do not impute the dataset at the beginning when the
        # posterior/variational samples are poor
        if self.current_epoch >= self.hparams.cdi.imputation_delay:
            with torch.no_grad():
                for i in range(num_imputation_steps):
                    X_imp = self.sample_missing_values(X, M)
                    X[M_not] = X_imp[M_not]

                if stage == 'train':
                    self.train_dataset[I.cpu()] = X.cpu()
                elif stage == 'val':
                    self.val_dataset_augmented[I.cpu()] = X.cpu()
                elif stage == 'test':
                    self.test_dataset[I.cpu()] = X.cpu()

    def update_step(self, batch):
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
        imp_start_time = time.time()
        num_imputation_steps = self.num_imp_steps_schedule.get_value()
        logs = self.impute_batch(batch, stage='train',
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
                self.impute_batch((X_new, M_new, I_new), stage='train',
                                  num_imputation_steps=self.hparams.data.num_new_chain_imp_steps)
                # Set the imputed values into the batch
                X[mask], M[mask], I[mask] = X_new, M_new, I_new

        imp_time = time.time() - imp_start_time

        # Update
        output = self.update_step(batch)
        # output['progress_bar'].update(logs)
        output['progress_bar']['train_imp_time'] = imp_time
        return output

    # Validation

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        if dataset_idx == 0:
            return super().validation_step(batch, batch_idx)
        elif dataset_idx == 1:
            with torch.autograd.no_grad():
                num_imputation_steps = self.num_imp_steps_schedule.get_value()
                self.impute_batch(batch, stage='val',
                                  num_imputation_steps=num_imputation_steps)

                output = self.update_step(batch)
                loss = output['loss']
                output = {k.replace('train', 'val'): v
                          for k, v in output['progress_bar'].items()}
                output['val_loss'] = loss
                return output

    # Hooks

    # def on_batch_start(self, batch):
    #     # Runs only for training batch!
    #     self.impute_batch(batch)

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
        if self.hparams.cdi.debug.eval_incomplete:
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
        if self.hparams.cdi.debug.log_dataset:
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

    def pre_impute_using_model(self, dataset, num_imputation_steps):
        dataloader = torch.utils.data.DataLoader(
                                dataset,
                                batch_size=self.hparams.data.batch_size,
                                collate_fn=collate_augmented_samples,
                                num_workers=1,
                                shuffle=False)

        model_before = self.fa_model
        if hasattr(self, 'ground_model') and self.ground_model is not None:
            ground_model_before = self.ground_model
        if self.hparams.gpus is not None:
            device = torch.device('cuda')
            self.fa_model = self.fa_model.to(device)
            if hasattr(self, 'ground_model') and self.ground_model is not None:
                self.ground_model = self.ground_model.to(device)
        else:
            device = torch.device('cpu')

        print('Using device for pre-imputation:', device)

        with torch.no_grad():
            for j, batch in enumerate(dataloader):
                print(f'Imputing batch {j}')

                # Transfer data to GPU
                batch = self.transfer_batch_to_device(batch, device)
                X, M, I = batch[:3]
                M_not = ~M

                with torch.no_grad():
                    for i in range(num_imputation_steps):
                        X_imp = self.sample_missing_values(X, M)
                        X[M_not] = X_imp[M_not]

                    # Store imputation
                    dataset[I.cpu()] = X.cpu()

        self.fa_model = model_before
        if hasattr(self, 'ground_model') and self.ground_model is not None:
            self.ground_model = ground_model_before
