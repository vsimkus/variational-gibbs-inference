import os
import pprint
import time

import numpy as np
import torch
from jsonargparse import ActionConfigFile
from jsonargparse import ArgumentParser
# from pytorch_lightning import Trainer
# Enable MICE imputer
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from cdi.overrides.iterative_imputer import IterativeImputer
import cdi.trainers.trainer_base as tb
from cdi.overrides.trainer import Trainer
from cdi.util.arg_utils import convert_namespace
from cdi.util.data.data_augmentation_dataset import DataAugmentation
from cdi.util.fs_logger import FSLogger
from cdi.util.utils import (construct_experiment_name,
                            flatten_arg_namespace_to_dict)


def build_argparser():
    parser = ArgumentParser(parser_mode='jsonnet')
    parser = Trainer.add_argparse_args(parser)
    parser = ArgumentParser(parents=[parser],
                            parser_mode='jsonnet',
                            add_help=False)
    parser.add_argument('--output_root_dir',
                        type=str, default='.',
                        help='Root directory for outputs.')
    parser.add_argument('--config',
                        type=str, action=ActionConfigFile)
    parser.add_argument('--experiment_name',
                        type=str, required=True,
                        help='Name of experiment.')
    parser.add_argument('--exp_group',
                        type=str, required=True,
                        help='Experiment group.')

    # Add common arguments
    parser = tb.TrainerBase.add_model_args(parser)

    # Add MICE arguments
    parser.add_argument('--mice.max_iter',
                        type=int, required=True,
                        help=('Maximum number of iterations.'))
    parser.add_argument('--mice.imputation_order',
                        type=str, default='random',
                        help=('Order of imputations.'))
    parser.add_argument('--mice.imp_model.n_iter',
                        type=int, required=True,
                        help=('Max number of iterations for each estimator.'))
    parser.add_argument('--mice.imp_model.hparams',
                        type=float, nargs=4, default=[1e-6]*4,
                        help=('alpha1, alpha2, lambda1, lambda2'))
    parser.add_argument('--mice.verbose',
                        type=int, default=0, choices=[0, 1, 2],
                        help=('Verbosity of the outputs. Default (0) hides all outputs.'))

    return parser


def main(args):
    # Convert jsonargparse's SimpleNamespace to argparse.Namespace
    # which is required by pytorch_lightning
    args = convert_namespace(args)

    # Prepare logger
    root_dir = os.path.join(os.path.abspath(args.output_root_dir),
                            'trained_models',
                            args.exp_group,
                            construct_experiment_name(args))
    log_dir = os.path.join('file:/', root_dir, 'logs')
    logger = FSLogger(log_dir)

    # Log hyperparams
    logger.log_hyperparams(args)

    # Set the argument for imputed copies to 1
    # we leave the augmentation to this script instead
    num_imputations = args.data.num_imputed_copies[0]
    args.data.num_imputed_copies = [1]
    # Prepare data using the base trainer
    trainer_base = tb.TrainerBase(args)
    trainer_base.logger = logger
    trainer_base.setup(stage='fit')
    train_dataset = trainer_base.train_dataset
    # val_dataset = trainer_base.val_dataset

    X, M = train_dataset[:][:2]
    # Set missing X to nan
    X[~M] = np.nan

    train_start_time = time.time()
    X_imputed_all = []
    for i in range(1, num_imputations+1):
        imputer = IterativeImputer(
            estimator=BayesianRidge(
                n_iter=args.mice.imp_model.n_iter,
                alpha_1=args.mice.imp_model.hparams[0],
                alpha_2=args.mice.imp_model.hparams[1],
                lambda_1=args.mice.imp_model.hparams[2],
                lambda_2=args.mice.imp_model.hparams[3]),
            missing_values=np.nan,
            sample_posterior=True,
            max_iter=args.mice.max_iter,
            initial_strategy='mean',
            imputation_order='random',
            random_state=args.model_seed+i*3,  # To make sure seeds are different
            verbose=args.mice.verbose
        )

        X_imputed = imputer.fit_transform(X)
        X_imputed_all.append(X_imputed)
    train_time = time.time() - train_start_time

    print(f'Training finished in {train_time:.4f} seconds.')

    logger.log_metrics({
        'train_time': train_time,
        'curr_epoch': 1
        }, 'final')

    print('Saving completed data.')
    incomp_mask = M.sum(axis=1) < X.shape[1]
    # Keep the full matrix first for the first imputation
    # But only keep the incomplete samples from following imputations
    for i in range(1, len(X_imputed_all)):
        X_imputed_all[i] = X_imputed_all[i][incomp_mask, :]

    # Create dummy augmented data and set to MICE-imputed data
    train_dataset = DataAugmentation(train_dataset, num_imputations, augment_complete=args.data.augment_complete)
    train_dataset.aug_data = np.concatenate(X_imputed_all)

    # Save imputed data
    X_imp, M_imp, _ = train_dataset[:][:3]
    logger.log_tensors(epoch='final', logname='train_data',
                       X=torch.tensor(X_imp),
                       M=torch.tensor(M_imp))


if __name__ == '__main__':
    args = build_argparser().parse_args()
    print('Args:\n')
    pprint.pprint(flatten_arg_namespace_to_dict(args), width=1)

    # Train
    main(args)
