import os
import sys
import time
from collections import OrderedDict, defaultdict

import scipy.io as sio
import numpy as np
import pytorch_lightning as pl
import torch

from jsonargparse import ArgumentParser

from cdi.data.frey_faces import FreyFacesDataset
from cdi.data.toy_set import ToyDataset
from cdi.data.emnist import EMNIST
from cdi.data.uci_gas import GasDataset
from cdi.data.uci_power import PowerDataset
from cdi.data.uci_hepmass import HEPMASSDataset
from cdi.data.uci_miniboone import MiniBooNEDataset
from cdi.data.uci_generated import GenDataset
from cdi.trainers.regression_fn_trainer import RegressionTrainer
from cdi.util.data.data_augmentation_dataset import DataAugmentation, \
                                                    collate_augmented_samples, \
                                                    DataAugmentationWithScheduler
from cdi.util.data.missing_data_provider import MissingDataProvider
from cdi.util.data.block_missingness_dataset import BlockMissingnessData
from cdi.util.data.fully_missing_filter_dataset import FullyMissingDataFilter
from cdi.util.mutable_subset import MutableSubset
from cdi.util.utils import construct_experiment_name, \
                           set_random
from cdi.util.stats_utils import load_statistics
from cdi.util.arg_utils import parse_bool, args_required_length
from cdi.util.utils import EpochScheduler


class TrainerBase(pl.LightningModule):
    """
    Trainer base for all experiments.
    Handles data creation and preparation,
    and common arguments.
    """
    def __init__(self, hparams):
        super(TrainerBase, self).__init__()
        self.hparams = hparams

        # Set random seed before model initialisation
        set_random(self.hparams.model_seed)

        # For backward-compability
        if (isinstance(self.hparams.data.num_imputed_copies, list)
                and len(self.hparams.data.num_imputed_copies) != 0):
            self.num_imputed_copies_scheduler = EpochScheduler(self,
                                                               self.hparams.data.num_imputed_copies_schedule,
                                                               self.hparams.data.num_imputed_copies)
        else:
            self.num_imputed_copies_scheduler = None

        # For persisting generated missingness mechanisms
        if self.hparams.data.miss_type in ('MAR', 'MNAR', 'NMAR'):
            # If the model already knows the shapes - use them
            # Necessary for leading a pretrained model
            if hasattr(self.hparams.data, 'dataset_patterns_shape'):
                patterns_shape = self.hparams.data.dataset_patterns_shape
                rel_freqs_shape = self.hparams.data.dataset_pattern_rel_freqs_shape
                weights_shape = self.hparams.data.dataset_pattern_weights_shape
                balances_shape = self.hparams.data.dataset_pattern_balances_shape
            else:
                patterns_shape = (1, )
                rel_freqs_shape = (1, )
                weights_shape = (1, )
                balances_shape = (1, )

            self.dataset_patterns = torch.nn.Parameter(
                                        data=torch.empty(*patterns_shape,
                                                         dtype=torch.bool),
                                        requires_grad=False)
            self.dataset_pattern_rel_freqs = torch.nn.Parameter(
                                        data=torch.empty(*rel_freqs_shape,
                                                         dtype=torch.float),
                                        requires_grad=False)
            self.dataset_pattern_weights = torch.nn.Parameter(
                                        data=torch.empty(*weights_shape,
                                                         dtype=torch.float),
                                        requires_grad=False)
            self.dataset_pattern_balances = torch.nn.Parameter(
                                        data=torch.empty(*balances_shape,
                                                         dtype=torch.float),
                                        requires_grad=False)

    @staticmethod
    def add_model_args(parent_parser):
        # Add algorithm parameters
        parser = ArgumentParser(parser_mode='jsonnet',
                                parents=[parent_parser],
                                add_help=False)
        parser.add_argument('--model_seed',
                            type=int, default=20190508,
                            help=('Random seed for model initialisation '
                                  'and model training.'))
        parser.add_argument('--data_seeds',
                            type=int, nargs=3,  # action=args_required_length(3, 4),
                            default=[20200325, 20200406, 20200407],
                            help=('Random seed for data initialisation. '
                                  'First seed is for missingness mask '
                                  'generation, second - for train/val split,'
                                  ' third - pre-imputation.'))
                                  #'(optional) fourth - subsample training data seed.'))

        # Data args
        parser.add_argument('--data.dataset_root',
                            type=str, default='data',
                            help='Path to root of dataset.')
        parser.add_argument('--data.augment_complete',
                            default=False, type=parse_bool,
                            help=('Whether to augment complete or not.'))
        parser.add_argument('--data.num_imputed_copies',
                            type=int, nargs='+', default=[0],
                            help=('The number of copies to make '
                                  'of the incomplete samples'))
        parser.add_argument('--data.num_imputed_copies_schedule',
                            type=int, nargs='+', default=[0],
                            help=('A list of epochs when number of '
                                  'chains should be changed.'))
        parser.add_argument('--data.num_new_chain_imp_steps',
                            type=int, required=False,
                            help=('Number of new chain imputation chains, '
                                  'when a chain is added.'))
        parser.add_argument('--data.batch_size',
                            type=int, required=True,
                            help=('Batch size'))
        parser.add_argument('--data.dataset',
                            type=str, help='Dataset name.',
                            choices=['emnist_balanced',
                                     'fa_frey_processed',
                                     'fa_frey_large_processed',
                                     'fcvae_frey_processed',
                                     'fcvae_frey_trunc_processed',
                                     'fcvae_frey_large_processed',
                                     'fa_frey_trunc_processed',
                                     'frey_faces_processed',
                                     'frey_faces_not_processed',
                                     'toy_set',
                                     'toy_set2',
                                     'toy_set3',
                                     'uci_power', 'uci_gas', 'uci_hepmass', 'uci_miniboone',
                                     'uci_gas_gen'])
        parser.add_argument('--data.total_miss',
                            type=float, required=True,
                            help='Missingness fraction in dataset.')
        parser.add_argument('--data.miss_type',
                            type=str, required=True,
                            help='Missingness type in dataset.',
                            choices=['MCAR', 'MAR', 'MNAR'])
        parser.add_argument('--data.max_patterns',
                            type=int, required=False,
                            help=('Maximum number of patterns to generate for '
                                  '`MCAR`, `MAR`, `MNAR` mechanisms.'))
        parser.add_argument('--data.filter_fully_missing',
                            type=parse_bool, required=True,
                            help=('Whether fully-missing samples should '
                                  'be removed from the data. Removing '
                                  'MAR/MCAR data should be ok in any case,'
                                  ' but it is not ok in MNAR case. Also the '
                                  'baseline models should benefit from this.'))
        parser.add_argument('--data.pre_imputation',
                            type=str, default='mean',
                            choices=['mean',
                                     'empirical_distribution_samples',
                                     'true_values',
                                     'zero',
                                     'regression_prediction',
                                     'preimputed_data',
                                     'systematic_gibbs_sampling'],  # This one is for var_cdi only
                            help=('The imputation type used before '
                                  'training.'))
        parser.add_argument('--data.reg_model_config',
                            type=str,
                            help=('Path to regression model config,'
                                  ' used for regression prediction '
                                  'imputation'))
        parser.add_argument('--data.preimp_data_model_name',
                            type=str,
                            help=('The name of the model, whose imputed data'
                                  ' we are going to use as preimputation.'
                                  ' Needed for `preimputed_data`.'))
        parser.add_argument('--data.obs_zero_mean', type=parse_bool,
                            default=False,
                            help=('Whether to make observed data zero_mean.'))
        parser.add_argument('--data.train_on_val', type=parse_bool,
                            default=False, help=('Use validation data as training.'))
        parser.add_argument('--data.discard_fraction', type=float, required=False,
                            help=('Whether to subsample the training data and what fraction should be discarded.'))

        return parser

    # Setup
    @staticmethod
    def load_dataset(hparams, stage, path=None):
        test = (stage == 'test')

        if hparams.data.dataset == 'emnist_balanced':
            print('Using EMNIST balanced.')
            dataset = EMNIST(
                            root=hparams.data.dataset_root,
                            split='balanced',
                            train=not test)
        elif hparams.data.dataset == 'fa_frey_processed':
            print('Using preprocessed FA-Frey Faces.')
            dataset = FreyFacesDataset(
                            root=hparams.data.dataset_root,
                            preprocess=True,
                            generated='FA',
                            filename=path,
                            test=test)
        elif hparams.data.dataset == 'fa_frey_large_processed':
            print('Using preprocessed FA-Frey Faces Large.')
            dataset = FreyFacesDataset(
                            root=hparams.data.dataset_root,
                            preprocess=True,
                            generated='FA-large',
                            filename=path,
                            test=test)
        elif hparams.data.dataset == 'fcvae_frey_processed':
            print('Using preprocessed FCVAE-Frey Faces.')
            dataset = FreyFacesDataset(
                            root=hparams.data.dataset_root,
                            preprocess=True,
                            generated='FC-VAE',
                            filename=path,
                            test=test)
        elif hparams.data.dataset == 'fcvae_frey_trunc_processed':
            print('Using preprocessed truncated FCVAE-Frey Faces.')
            dataset = FreyFacesDataset(
                            root=hparams.data.dataset_root,
                            preprocess=True,
                            generated='FC-VAE',
                            truncate=True,
                            filename=path,
                            test=test)
        elif hparams.data.dataset == 'fcvae_frey_large_processed':
            print('Using preprocessed FCVAE-Frey Faces Large.')
            dataset = FreyFacesDataset(
                            root=hparams.data.dataset_root,
                            preprocess=True,
                            generated='FC-VAE-large',
                            filename=path,
                            test=test)
        elif hparams.data.dataset == 'fa_frey_trunc_processed':
            print('Using preprocessed truncated FA-Frey Faces.')
            dataset = FreyFacesDataset(
                            root=hparams.data.dataset_root,
                            preprocess=True,
                            truncate=True,
                            generated='FA',
                            filename=path,
                            test=test)
        elif hparams.data.dataset == 'frey_faces_processed':
            print('Using preprocessed Frey Faces.')
            dataset = FreyFacesDataset(
                            root=hparams.data.dataset_root,
                            preprocess=True,
                            generated=None,
                            filename=path,
                            test=test)
        elif hparams.data.dataset == 'frey_faces_not_processed':
            print('Using *not* preprocessed Frey Faces.')
            dataset = FreyFacesDataset(
                            root=hparams.data.dataset_root,
                            preprocess=False,
                            generated=None,
                            filename=path,
                            test=test)
        elif hparams.data.dataset == 'toy_set':
            dataset = ToyDataset(
                        root=hparams.data.dataset_root,
                        version=1,
                        filename=path,
                        test=test)
        elif hparams.data.dataset == 'toy_set2':
            dataset = ToyDataset(
                        root=hparams.data.dataset_root,
                        version=2,
                        filename=path,
                        test=test)
        elif hparams.data.dataset == 'toy_set3':
            dataset = ToyDataset(
                        root=hparams.data.dataset_root,
                        version=3,
                        filename=path,
                        test=test)
        elif hparams.data.dataset == 'uci_power':
            if stage != 'test':
                train_dataset = PowerDataset(
                            root=hparams.data.dataset_root,
                            split='train')
                val_dataset = PowerDataset(
                            root=hparams.data.dataset_root,
                            split='val')
                dataset = (train_dataset, val_dataset)
            else:
                dataset = PowerDataset(
                            root=hparams.data.dataset_root,
                            split='test')
        elif hparams.data.dataset == 'uci_gas':
            if stage != 'test':
                train_dataset = GasDataset(
                            root=hparams.data.dataset_root,
                            split='train')
                val_dataset = GasDataset(
                            root=hparams.data.dataset_root,
                            split='val')
                dataset = (train_dataset, val_dataset)
            else:
                dataset = GasDataset(
                            root=hparams.data.dataset_root,
                            split='test')
        elif hparams.data.dataset == 'uci_hepmass':
            if stage != 'test':
                train_dataset = HEPMASSDataset(
                            root=hparams.data.dataset_root,
                            split='train')
                val_dataset = HEPMASSDataset(
                            root=hparams.data.dataset_root,
                            split='val')
                dataset = (train_dataset, val_dataset)
            else:
                dataset = HEPMASSDataset(
                            root=hparams.data.dataset_root,
                            split='test')
        elif hparams.data.dataset == 'uci_miniboone':
            if stage != 'test':
                train_dataset = MiniBooNEDataset(
                            root=hparams.data.dataset_root,
                            split='train')
                val_dataset = MiniBooNEDataset(
                            root=hparams.data.dataset_root,
                            split='val')
                dataset = (train_dataset, val_dataset)
            else:
                dataset = MiniBooNEDataset(
                            root=hparams.data.dataset_root,
                            split='test')
        elif hparams.data.dataset == 'uci_gas_gen':
            if stage != 'test':
                train_dataset = GenDataset(
                                    root=hparams.data.dataset_root,
                                    data='gas',
                                    split='train')
                val_dataset = GenDataset(
                                    root=hparams.data.dataset_root,
                                    data='gas',
                                    split='val')
                dataset = (train_dataset, val_dataset)
            else:
                dataset = GenDataset(
                                    root=hparams.data.dataset_root,
                                    data='gas',
                                    split='test')
        else:
            print(('No such dataset available!',
                   f'`{hparams.data.dataset}`!'))
            sys.exit()

        return dataset

    @staticmethod
    def split_dataset(dataset, train_fraction=0.8):
        # Train/Val split
        dataset_size = len(dataset)
        indices = np.random.permutation(range(dataset_size))
        split = int(np.floor(train_fraction * dataset_size))
        train_indices, val_indices = indices[:split], indices[split:]
        train_dataset = MutableSubset(dataset, train_indices)
        val_dataset = MutableSubset(dataset, val_indices)

        return train_dataset, val_dataset

    def initialise_dataset(self, hparams, dataset):
        # TODO: refactor this method
        pre_metrics = None
        # Imputation method
        init_start_time = time.time()
        import cdi.common.imputation as imputation
        if hparams.data.pre_imputation == 'true_values':
            print('Using true values.')
        elif hparams.data.pre_imputation == 'mean':
            print('Performing mean imputation...')
            imputation.impute_with_mean(dataset)
        elif hparams.data.pre_imputation == 'empirical_distribution_samples':
            print('Imputation with samples from empirical distribution.')
            imputation.impute_with_empirical_distribution_sample(dataset)
        elif hparams.data.pre_imputation == 'zero':
            print('Imputation with zeros.')
            X, M, I = dataset.get_all_data()[:3]
            dataset[I] = X*M
        elif hparams.data.pre_imputation == 'preimputed_data':
            print('Setting data to the preimputed data.')
            seed_stamp = construct_experiment_name(hparams).split('/')[-1]
            preimputed_data_path = os.path.join(
                                        'trained_models',
                                        hparams.exp_group,
                                        hparams.data.preimp_data_model_name,
                                        seed_stamp,
                                        'logs',
                                        'tensors',
                                        'train_data_final.npz')
            print(f'Using data from {preimputed_data_path}.')
            data = np.load(preimputed_data_path, allow_pickle=True)

            X, M, I = dataset.get_all_data()[:3]

            if X.shape == data['X'].shape:
                assert (np.allclose(data['X']*data['M'], X*M)
                        and np.all(data['M'] == M)),\
                    ('The preimputed data does not match the current data!')

                # Set the data to the final values of another model's imputed data
                dataset[I] = data['X']

                X2, M2, _ = dataset.get_all_data()[:3]
                assert np.allclose(X2, data['X']) and np.allclose(M2, data['M'])
            else:
                # Workaround for the case where MICE saved the imputations in the old format
                X_load = np.zeros_like(X)
                M_load = np.ones_like(M, dtype=M.dtype)

                incomp_mask = (~M).any(axis=-1)
                incomp_mask_load = (~data['M']).any(axis=-1)

                X_load[incomp_mask] = data['X'][incomp_mask_load]
                M_load[incomp_mask] = data['M'][incomp_mask_load]
                X_load[~incomp_mask] = X[~incomp_mask]

                assert (np.allclose(X_load*M_load, X*M)
                        and np.all(M_load == M)),\
                    ('The preimputed data does not match the current data!')

                X_load_full = data['X'][~incomp_mask_load]

                assert np.allclose(X_load_full, X[~incomp_mask].reshape(X_load_full.shape[0], -1, X_load_full.shape[1],)[:, 0, :]),\
                    ('Load check failed. (Checking whether the fully-observed data is same)')

                # Set the data to the final values of another model's imputed data
                dataset[I] = X_load

                X2, M2, _ = dataset.get_all_data()[:3]
                assert np.allclose(X2, X_load) and np.allclose(M2, M_load)

            # Extract time metrics from the pre-trained model
            pre_model_log_path = os.path.join(
                                        'trained_models',
                                        hparams.exp_group,
                                        hparams.data.preimp_data_model_name,
                                        seed_stamp,
                                        'logs')
            pre_metrics = {
                'init_time': [],
                'stage': []
            }
            for d in os.listdir(pre_model_log_path):
                if d.startswith('summary'):
                    stats = load_statistics(pre_model_log_path, d)
                    pre_metrics['init_time'].append(np.sum(stats['train_time']))
                    pre_metrics['stage'].append('pre_train')

        elif hparams.data.pre_imputation == 'regression_prediction':
            print('Regression imputation.')
            reg_argparser = ArgumentParser(parser_mode='jsonnet',
                                           add_help=False,
                                           print_config=None)
            reg_argparser.add_argument('--experiment_name',
                                       type=str, required=True,
                                       help='Name of experiment.')
            reg_argparser.add_argument('--exp_group',
                                       type=str, required=True,
                                       help='Experiment group.')
            reg_argparser = TrainerBase.add_model_args(reg_argparser)
            reg_argparser = pl.Trainer.add_argparse_args(reg_argparser)
            reg_argparser = RegressionTrainer.add_regression_trainer_args(
                                                    reg_argparser)
            reg_args = reg_argparser.parse_path(
                                hparams.data.reg_model_config)

            # NOTE:
            # Set seed to same as for this model, since these seeds were
            # supposed to be used when training the regression model
            reg_args.data_seeds = hparams.data_seeds

            # Determine device for regression imputations
            if hparams.gpus is not None:
                device = torch.cuda.current_device()
            else:
                device = torch.device('cpu')
            imputation.impute_with_regression_predictions(
                                    dataset,
                                    reg_args,
                                    device,
                                    exp_group=reg_args.exp_group,
                                    repeat=1)

            # Extract time metrics from the pre-trained model
            exp_name = construct_experiment_name(reg_args)
            pre_model_log_path = os.path.join(
                                        'trained_models',
                                        reg_args.exp_group,
                                        exp_name,
                                        'logs')
            pre_metrics = {
                'init_time': [],
                'stage': []
            }
            for d in os.listdir(pre_model_log_path):
                if d.startswith('summary'):
                    stats = load_statistics(pre_model_log_path, d)
                    pre_metrics['init_time'].append(np.sum(stats['train_time']))
                    pre_metrics['stage'].append('pre_train')
        else:
            print(f'No such imputation method {hparams.data.pre_imputation =}!')
            sys.exit()

        if pre_metrics is not None:
            metrics = pre_metrics
            metrics['init_time'].append(time.time() - init_start_time)
            metrics['stage'].append('initialise_dataset')
        else:
            metrics = {
                'init_time': [time.time() - init_start_time],
                'stage': ['initialise_dataset']
            }

        return metrics

    def setup(self, stage):
        # Prepare the training and validation data
        if stage == 'fit':
            # Load the data
            set_random(seed=self.hparams.data_seeds[0])
            dataset = self.load_dataset(self.hparams, stage)

            if isinstance(dataset, tuple):
                train_dataset, val_dataset = dataset

                # Add missingness
                rng = torch.Generator()
                rng.manual_seed(self.hparams.data_seeds[0])
                train_dataset = MissingDataProvider(
                                    train_dataset,
                                    total_miss=self.hparams.data.total_miss,
                                    miss_type=self.hparams.data.miss_type,
                                    max_patterns=self.hparams.data.max_patterns if hasattr(self.hparams.data, 'max_patterns') else None,
                                    should_fit_to_data=True,
                                    rand_generator=rng)

                # If we generate random patterns then persist them so that we
                # can reuse them at test time
                if self.hparams.data.miss_type in ('MAR', 'MNAR', 'NMAR'):
                    self.dataset_patterns.data = dataset.patterns
                    self.dataset_pattern_rel_freqs.data = dataset.rel_freqs
                    self.dataset_pattern_weights.data = dataset.weights
                    self.dataset_pattern_balances.data = dataset.balances
                    self.hparams.data.dataset_patterns_shape = dataset.patterns.shape
                    self.hparams.data.dataset_pattern_rel_freqs_shape = dataset.rel_freqs.shape
                    self.hparams.data.dataset_pattern_weights_shape = dataset.weights.shape
                    self.hparams.data.dataset_pattern_balances_shape = dataset.balances.shape

                if self.hparams.data.miss_type in ('MAR', 'MNAR', 'NMAR'):
                    # Load the pattern from saved model
                    val_dataset = MissingDataProvider(
                                        val_dataset,
                                        total_miss=self.hparams.data.total_miss,
                                        miss_type='patterns',
                                        patterns=self.dataset_patterns,
                                        rel_freqs=self.dataset_pattern_rel_freqs,
                                        weights=self.dataset_pattern_weights,
                                        balances=self.dataset_pattern_balances,
                                        should_fit_to_data=False,
                                        rand_generator=rng)
                elif self.hparams.data.miss_type == 'MCAR':
                    val_dataset = MissingDataProvider(
                                        val_dataset,
                                        total_miss=self.hparams.data.total_miss,
                                        miss_type=self.hparams.data.miss_type,
                                        rand_generator=rng)

            else:
                # Add missingness
                rng = torch.Generator()
                rng.manual_seed(self.hparams.data_seeds[0])
                dataset = MissingDataProvider(
                                    dataset,
                                    total_miss=self.hparams.data.total_miss,
                                    miss_type=self.hparams.data.miss_type,
                                    max_patterns=self.hparams.data.max_patterns if hasattr(self.hparams.data, 'max_patterns') else None,
                                    should_fit_to_data=True,
                                    rand_generator=rng)

                # If we generate random patterns then persist them so that we
                # can reuse them at test time
                if self.hparams.data.miss_type in ('MAR', 'MNAR', 'NMAR'):
                    self.dataset_patterns.data = dataset.patterns
                    self.dataset_pattern_rel_freqs.data = dataset.rel_freqs
                    self.dataset_pattern_weights.data = dataset.weights
                    self.dataset_pattern_balances.data = dataset.balances
                    self.hparams.data.dataset_patterns_shape = dataset.patterns.shape
                    self.hparams.data.dataset_pattern_rel_freqs_shape = dataset.rel_freqs.shape
                    self.hparams.data.dataset_pattern_weights_shape = dataset.weights.shape
                    self.hparams.data.dataset_pattern_balances_shape = dataset.balances.shape

                # Split the data
                set_random(seed=self.hparams.data_seeds[1])
                train_dataset, val_dataset = TrainerBase.split_dataset(dataset)

            if hasattr(self.hparams.data, 'discard_fraction') and self.hparams.data.discard_fraction is not None:
                # assert len(self.hparams.data_seeds) > 3,\
                #     'The data seed for subsampling data should be provided!'
                # Re-use the missingness seed for this
                set_random(seed=self.hparams.data_seeds[0])

                # Sub-sample the training data
                train_dataset, _ = TrainerBase.split_dataset(train_dataset,
                                                             train_fraction=1-self.hparams.data.discard_fraction)
                val_dataset, _ = TrainerBase.split_dataset(val_dataset,
                                                           train_fraction=1-self.hparams.data.discard_fraction)

                print('Subsampled train dataset size:', len(train_dataset))
                print('Subsampled val dataset size', len(val_dataset))

            # Remove fully-missing samples if required
            # NOTE: we could do this before the split in MissingDataProvider
            # class, however, that would result in different splits for
            # different seeds and hence data leakage.
            if self.hparams.data.filter_fully_missing:
                train_dataset = FullyMissingDataFilter(train_dataset)
                # val_dataset = FullyMissingDataFilter(val_dataset)

            # Make training data zero-mean
            if hasattr(self.hparams.data, 'obs_zero_mean') and self.hparams.data.obs_zero_mean:
                X, M, I = train_dataset[:][:3]
                M = M.cpu().numpy()
                # Comp. observed mean
                self.obs_mean = (X * M).sum(axis=0)/M.sum(axis=0).astype(np.float32)
                # Make train data zero-mean
                train_dataset[I] = X - self.obs_mean
                # Make val. data zero-mean
                X, M, I = val_dataset[:][:3]
                val_dataset[I] = X - self.obs_mean

                # Save the mean vector
                if self.logger is not None:
                    self.logger.log_tensors(epoch=0, logname='obs_mean',
                                            obs_mean=self.obs_mean)

            if hasattr(self.hparams.data, 'train_on_val') and self.hparams.data.train_on_val:
                train_dataset = val_dataset
                if self.hparams.data.filter_fully_missing:
                    # Remove fully-missing
                    train_dataset = FullyMissingDataFilter(train_dataset)

            # Augment data by making several copies of the original
            # for multiple-imputation
            if self.num_imputed_copies_scheduler is None:
                if isinstance(self.hparams.data.num_imputed_copies, list):
                    num_copies = self.hparams.data.num_imputed_copies[0]
                else:
                    num_copies = self.hparams.data.num_imputed_copies
                self.train_dataset = DataAugmentation(
                                            train_dataset,
                                            num_copies,
                                            augment_complete=hasattr(self.hparams.data, 'augment_complete') and self.hparams.data.augment_complete)
            else:
                self.train_dataset = DataAugmentationWithScheduler(
                                            train_dataset,
                                            self.num_imputed_copies_scheduler,
                                            augment_complete=hasattr(self.hparams.data, 'augment_complete') and self.hparams.data.augment_complete)
            self.val_dataset = val_dataset

            # Initialise data
            set_random(seed=self.hparams.data_seeds[2])
            init_metrics = self.initialise_dataset(self.hparams,
                                                   self.train_dataset)
            # TrainerBase.initialise_dataset(self.hparams, self.val_dataset)
            if self.logger is not None and init_metrics is not None and not isinstance(self.logger, pl.loggers.TensorBoardLogger):
                self.logger.log_initialisation_metric(init_metrics)
        elif stage == 'test' and (hasattr(self.hparams, 'test')
                                  and (hasattr(self.hparams.test, 'use_train_data')
                                       and self.hparams.test.use_train_data)):
            # Use train data for testings
            self.setup(stage='fit')
            self.test_dataset = self.train_dataset
        # Prepare testing data
        elif stage == 'test':
            # Load test data
            dataset = self.load_dataset(self.hparams, stage)

            if self.hparams.test.generate_test_data:
                generated_data = self.fa_model.generate_data(self.hparams.test.generate_test_data_size,
                                                             seed=self.hparams.test.test_seed)

                exp = construct_experiment_name(self.hparams)
                path = os.path.join('trained_models',
                                    self.hparams.exp_group,
                                    exp, 'evaluations', 'generated_data.mat')
                if 'frey' in self.hparams.data.dataset:
                    if self.hparams.test.postprocess_generated_data:
                        generated_data = dataset.postprocess(generated_data)

                    data = {
                        "ff": generated_data.detach().numpy().T,
                        "seed": self.hparams.test.test_seed
                    }
                    sio.savemat(file_name=path, mdict=data)

                # Load generated test data
                dataset = self.load_dataset(self.hparams, stage,
                                                   os.path.join('../', path))

            # Add missingness
            rng = torch.Generator()
            rng.manual_seed(self.hparams.test.test_seed)
            if not hasattr(self.hparams.test, 'data') or self.hparams.test.data.miss_type is None:
                if self.hparams.data.miss_type in ('MAR', 'MNAR', 'NMAR'):
                    # Load the pattern from saved model
                    dataset = MissingDataProvider(
                                        dataset,
                                        total_miss=self.hparams.data.total_miss,
                                        miss_type='patterns',
                                        patterns=self.dataset_patterns,
                                        rel_freqs=self.dataset_pattern_rel_freqs,
                                        weights=self.dataset_pattern_weights,
                                        balances=self.dataset_pattern_balances,
                                        should_fit_to_data=False,
                                        rand_generator=rng)
                elif self.hparams.data.miss_type == 'MCAR':
                    dataset = MissingDataProvider(
                                        dataset,
                                        total_miss=self.hparams.data.total_miss,
                                        miss_type=self.hparams.data.miss_type,
                                        rand_generator=rng)
            elif hasattr(self.hparams.test, 'data') and self.hparams.test.data.miss_type in ('MAR', 'MNAR', 'NMAR'):
                # Load the pattern from saved model
                dataset = MissingDataProvider(
                                    dataset,
                                    total_miss=self.hparams.data.total_miss,
                                    miss_type='patterns',
                                    patterns=self.dataset_patterns,
                                    rel_freqs=self.dataset_pattern_rel_freqs,
                                    weights=self.dataset_pattern_weights,
                                    balances=self.dataset_pattern_balances,
                                    should_fit_to_data=False,
                                    rand_generator=rng)
            elif hasattr(self.hparams.test, 'data') and self.hparams.test.data.miss_type == 'MCAR':
                dataset = MissingDataProvider(
                                    dataset,
                                    total_miss=self.hparams.data.total_miss,
                                    miss_type=self.hparams.data.miss_type,
                                    rand_generator=rng)
            elif hasattr(self.hparams.test, 'data') and self.hparams.test.data.miss_type == 'block':
                dataset = BlockMissingnessData(dataset,
                                               width=self.hparams.test.data.width,
                                               fraction_incomplete=0.95,
                                               inverse=False,
                                               top_left=self.hparams.test.data.top_left,
                                               bottom_right=self.hparams.test.data.bottom_right)
            elif hasattr(self.hparams.test, 'data') and self.hparams.test.data.miss_type == 'block-inverse':
                dataset = BlockMissingnessData(dataset,
                                               width=self.hparams.test.data.width,
                                               fraction_incomplete=0.95,
                                               inverse=True,
                                               top_left=self.hparams.test.data.top_left,
                                               bottom_right=self.hparams.test.data.bottom_right)

            # # Remove fully-missing samples if required
            # if self.hparams.data.test.filter_fully_missing:
            #     dataset = FullyMissingDataFilter(dataset)

            # Make test data zero-mean
            if hasattr(self.hparams.data, 'obs_zero_mean') and self.hparams.data.obs_zero_mean:
                # Load the observed data mean from the training
                obs_mean_path = os.path.join('trained_models',
                                             self.hparams.exp_group,
                                             construct_experiment_name(self.hparams),
                                             'logs', 'tensors',
                                             'obs_mean_0.npz')
                self.obs_mean = np.load(obs_mean_path, allow_pickle=True)['obs_mean']
                X, M, I = dataset[:][:3]
                dataset[I] = X - self.obs_mean

            # Augmentation of test set
            dataset = DataAugmentation(dataset, self.hparams.test.num_imputed_copies,
                                       augment_complete=hasattr(self.hparams.data, 'augment_complete') and self.hparams.data.augment_complete)

            # Initialise data
            if self.hparams.test.init_missing_data:
                set_random(seed=self.hparams.test.test_seed)
                self.initialise_dataset(self.hparams, dataset)

            self.test_dataset = dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                                self.train_dataset,
                                batch_size=self.hparams.data.batch_size,
                                collate_fn=collate_augmented_samples,
                                num_workers=2,
                                shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                                self.val_dataset,
                                batch_size=self.hparams.data.batch_size,
                                # collate_fn=collate_augmented_samples,
                                num_workers=2,
                                shuffle=False)

    def forward(self, X):
        raise NotImplementedError

    # Training

    def get_progress_bar_dict(self):
        # Override default progress_bar to remove experiment version number
        progress_bar = super().get_progress_bar_dict()
        if 'v_num' in progress_bar:
            progress_bar.pop('v_num')
        return progress_bar

    # Hooks

    def on_train_start(self):
        # Called at the beginning of training before sanity check.
        # Set random seed before training
        set_random(self.hparams.model_seed)

    def on_epoch_start(self):
        # Called in the training loop at the very beginning of the epoch.
        # Set train epoch start time
        self.train_epoch_start_time = time.time()

    def training_epoch_end(self, outputs):
        # Called in the training loop at the end of the epoch
        results = OrderedDict({
            'log': {'train_time': time.time() - self.train_epoch_start_time},
            'progress_bar': {}
        })

        outputs_avg = defaultdict(float)
        for output in outputs:
            for key, value in output['callback_metrics'].items():
                if key == 'loss':
                    key = 'train_loss'
                outputs_avg[key] += value

        for key, sum_value in outputs_avg.items():
            # For some metrics we want aggregate sum, not average.
            if key in ('train_imp_time', 'train_imp_accepted', 'train_imp_tries'):
                continue
            outputs_avg[key] /= len(outputs)

        results['log'] = {**results['log'], **outputs_avg}
        results['progress_bar'] = {**results['progress_bar'], **outputs_avg}
        return results

    def on_sanity_check_start(self):
        self.val_epoch_start_time = time.time()

    def on_pre_performance_check(self):
        # Called at the very beginning of the validation loop.
        # Set validation epoch start time
        self.val_epoch_start_time = time.time()

    def validation_epoch_end(self, outputs):
        # Called in the training loop at the end of the epoch
        results = OrderedDict({
            'log': {'val_time': time.time() - self.val_epoch_start_time,
                    'curr_epoch': self.current_epoch},
            'progress_bar': {}
        })

        outputs_avg = defaultdict(float)
        for output in outputs:
            for key, value in output.items():
                if key == 'loss':
                    key = 'val_loss'
                outputs_avg[key] += value

        for key, sum_value in outputs_avg.items():
            # For some metrics we want aggregate sum, not average.
            if key in ('total_nans', 'val_imp_time', 'val_imp_accepted', 'val_imp_tries'):
                continue
            outputs_avg[key] /= len(outputs)

        results['log'] = {**results['log'], **outputs_avg}
        results['progress_bar'] = {**results['progress_bar'], **outputs_avg}

        return results

    #
    # Test
    #

    @staticmethod
    def add_test_args(parent_parser):
        parser = parent_parser
        parser.add_argument('--test_seed',
                            type=int, required=True,
                            help=('The seed for the _test_ '
                                  'experiment.'))
        parser.add_argument('--use_train_data', default=False,
                            type=parse_bool, help='Use train data for testing')
        parser.add_argument('--generate_test_data', default=False,
                            type=parse_bool, help=('Whether to generate test data from the density model.'))
        parser.add_argument('--generate_test_data_size', default=None,
                            type=int, help=('Size of generated test data.'))
        parser.add_argument('--postprocess_generated_data', default=True,
                            type=parse_bool, help=('Whether to apply any postprocessing to generated data.'))
        parser.add_argument('--init_missing_data',
                            type=parse_bool, default=False,
                            help=('Whether to impute the missing values.'))
        parser.add_argument('--batch_size', type=int,
                            default=None, help=('Defaults to batch size used in training.'))
        parser.add_argument('--num_imputed_copies', type=int, default=1,
                            help=('Number of imputation chains for test data.'))
        parser.add_argument('--data.miss_type',
                            type=str, required=False,
                            help='Missingness type in test dataset.',
                            choices=['MCAR', 'MAR', 'MNAR', 'block', 'block-inverse'])
        parser.add_argument('--data.width',
                            type=int, required=False,
                            help='Width of the data in pixels.')
        parser.add_argument('--data.top_left',
                            type=int, nargs=2, required=False,
                            help='Top-left corner of the missingess block.')
        parser.add_argument('--data.bottom_right',
                            type=int, nargs=2, required=False,
                            help='Bottom-right corner of the missingess block.')
        return parser

    def test_dataloader(self):
        # NOTE: Can also return a list of test dataloaders
        batch_size = self.hparams.data.batch_size
        if self.hparams.test.batch_size is not None:
            batch_size = self.hparams.test.batch_size
        return torch.utils.data.DataLoader(
                                self.test_dataset,
                                collate_fn=collate_augmented_samples,
                                batch_size=batch_size,
                                num_workers=1,
                                shuffle=False)

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_epoch_end(self, outputs):
        raise NotImplementedError


class TestSetRandomCallback(pl.Callback):
    def on_test_start(self, trainer, pl_module):
        # Set random seed before testing
        set_random(pl_module.hparams.test.test_seed)
