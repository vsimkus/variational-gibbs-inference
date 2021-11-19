
import logging as log
import os
import sys
import pprint

import torch
from jsonargparse import ActionConfigFile
from jsonargparse import ArgumentParser
# from pytorch_lightning import Trainer
from pytorch_lightning.profiler import AdvancedProfiler

from cdi.overrides.trainer import Trainer
from cdi.overrides.model_checkpoint import ModelCheckpoint
from cdi.trainers.complete_mle import CompleteMLE
from cdi.trainers.variational_inference import VI
from cdi.trainers.variational_cdi import VarCDI
from cdi.trainers.variational_cdi_em import VarCDIEM
from cdi.trainers.mcimp import MCIMP
from cdi.util.arg_utils import convert_namespace, parse_bool
from cdi.util.fs_logger import FSLogger
from cdi.util.utils import (construct_experiment_name,
                            flatten_arg_namespace_to_dict)
from cdi.util.print_progress_bar import PrintingProgressBar

log.root.setLevel(log.INFO)


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
    parser.add_argument('--save_top_k_models',
                        type=int, default=1,
                        help=('Number of top (validation) performance models'
                              ' to checkpoint.'))
    parser.add_argument('--save_weights_only',
                        type=parse_bool, default=True,
                        help=('Whether save optimizer state as well, or just '
                              'the weights.'))
    parser.add_argument('--save_period',
                        type=int, default=1,
                        help=('Period when to save weights.'))
    parser.add_argument('--save_custom_epochs',
                        type=int, nargs='*',
                        help='Epochs when to save a model')
    parser.add_argument('--save', default=False,
                        help=('Whether to save the weight snapshots.'))
    parser.add_argument('--autograd_detect_anomaly',
                        type=parse_bool, default=False,
                        help=('DEBUG: Whether torch autograd detect anomaly'
                              ' should be used.'))
    parser.add_argument('--method',
                        type=str, required=True,
                        choices=['complete-case', 'variational',
                                 'variational-inference',
                                 'variational-em', 'mcimp'],
                        help=('Which approximation method to be'
                              ' used in CDI/EM/MLE.'))

    parser.add_argument('--main_experiment_name',
                        type=str, required=True,
                        help='Name of experiment weights to be loaded.')
    parser.add_argument('--ckpts_to_load', type=str, required=True,
                        nargs='+', help=('Chkpt to load.'))
    parser.add_argument('--model_seed',
                        type=int, default=20190508,
                        help=('Random seed for model initialisation '
                              'and model training.'))
    parser.add_argument('--data_seeds',
                        type=int, nargs=3,
                        default=[20200325, 20200406, 20200407],
                        help=('Random seed for data initialisation. '
                              'First seed is for missingness mask '
                              'generation, second - for train/val split,'
                              ' third - pre-imputation.'))
    parser.add_argument('--fix_encoder', default=False,
                        help=('Whether to fix encoder weights.'))
    parser.add_argument('--reset_encoder', default=False,
                        help='Whether to reset encoder weights.')
    parser.add_argument('--reset_vcdi_var', required=False,
                        choices=['init', 'rand'],
                        help='Whether to reset vcdi var. model weights.')
    parser.add_argument('--replace_tqdm', type=parse_bool,
                        default=False, help='Replace tqdm with printing dumps.')
    parser.add_argument('--print_stat_frequency', type=int,
                        default=20, help='How often should we print stats with the Printing progress.')

    # Data overrides
    parser.add_argument('--data.pre_imputation',
                        type=str, required=False,
                        choices=['mean',
                                 'empirical_distribution_samples',
                                 'true_values',
                                 'regression_prediction',
                                 'preimputed_data',
                                 'systematic_gibbs_sampling',  # This one is for var_cdi only
                                 'var_mcmc_using_model'],  # Only for mcimp
                        help=('The imputation type used before training.'))
    parser.add_argument('--data.pre_imp_num_imputation_steps',
                        type=int, required=False,
                        help=('Number of Gibbs sampling updates in pretraining.'))
    parser.add_argument('--data.pre_imp_clip', type=parse_bool,
                        required=False, help=('Whether to clip Gibbs pre-imputation values to min/max.'))
    parser.add_argument('--data.pre_imp_reject', type=parse_bool,
                        required=False, help=('Whether to reject Gibbs pre-imputation values outside min/max.'))

    # CDI overrides
    parser.add_argument('--cdi.update_comp_schedule',
                        type=int, nargs='+', required=False,
                        help=('A list of epochs when component fractions '
                              'should be changed.'))
    parser.add_argument('--cdi.update_comp_schedule_values',
                        type=float, nargs='+', required=False,
                        help=('A list of values that correspond to each'
                              ' scheduled time. The values determine the'
                              ' fraction of components per example that '
                              'are used in the update.'))
    parser.add_argument('--cdi.imputation_delay',
                        type=int, required=False,
                        help=('The number of epochs to wait before '
                              'starting to impute missing values '
                              'in the dataset.'))
    parser.add_argument('--cdi.imp_acceptance_check_schedule',
                        type=int, nargs='+', required=False,
                        help=('A list of epochs when acceptance behaviour '
                              'should be changed.'))
    parser.add_argument('--cdi.imp_acceptance_check_schedule_values',
                        type=parse_bool, nargs='+', required=False,
                        help=('Whether to perform an acceptance check '
                              'before imputing a dataset with new '
                              'samples.'))
    parser.add_argument('--cdi.num_imp_steps_schedule',
                        type=int, nargs='+', required=False,
                        help=('A list of epochs when number of imputation'
                              ' steps should be changed.'))
    parser.add_argument('--cdi.num_imp_steps_schedule_values',
                        type=int, nargs='+', required=False,
                        help=('A list of values that correspond to each'
                              ' schedule\'s mode. The number of imputation'
                              ' steps in each epoch.'))
    parser.add_argument('--cdi.train_ignore_nans', required=False,
                        type=parse_bool, help=('Whether nans and infs in the vcdi '
                                               'loss should be ignored. Could be used '
                                               'when evaluating validation with refitting.'))
    parser.add_argument('--cdi.debug.eval_incomplete',
                        type=parse_bool, required=False,
                        help=('In addition evaluates validation on incomplete data,'
                              'runs a chain of validation imputations similar to training.'))

    # Model optim overrides
    parser.add_argument('--model_optim.optimiser',
                        type=str, choices=['adam', 'amsgrad'],
                        required=False)
    parser.add_argument('--model_optim.learning_rate',
                        type=float, required=False,
                        help=('The learning rate using in Adam '
                              'optimiser for the fa_model.'))

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
    log_dir = os.path.join('file:/', root_dir, f'logs_{args.ckpt_to_load}')
    logger = FSLogger(log_dir,
                      continue_from_checkpoint=args.resume_from_checkpoint)

    old_exp_name = construct_experiment_name(args)
    old_exp_name = '/'.join([args.main_experiment_name,
                             old_exp_name.split('/', maxsplit=1)[1]])
    main_root_dir = os.path.join('trained_models',
                                 args.exp_group,
                                 old_exp_name)
    ckpt = f'_ckpt_epoch_{args.ckpt_to_load}' if args.ckpt_to_load != 'last' else 'last'
    model_dir = os.path.join(main_root_dir, 'saved_models',
                             f'{ckpt}.ckpt')
    # Prepare CDI
    if args.method == 'variational':
        model = VarCDI.load_from_checkpoint(model_dir)
    elif args.method == 'variational-em':
        model = VarCDIEM.load_from_checkpoint(model_dir)
    elif args.method == 'complete-case':
        model = CompleteMLE.load_from_checkpoint(model_dir)
    elif args.method == 'variational-inference':
        model = VI.load_from_checkpoint(model_dir)
    elif args.method == 'mcimp':
        model = MCIMP.load_from_checkpoint(model_dir)
    else:
        print(('No such approximation '
               f'`{args.method}`!'))
        sys.exit()

    if hasattr(args, 'data') and args.data is not None:
        if hasattr(args.data, 'pre_imputation') and args.data.pre_imputation is not None:
            # Store the original pre-imputation method first for use in refitting on val data
            model.hparams.data.orig_pre_imputation = model.hparams.data.pre_imputation
            model.hparams.data.pre_imputation = args.data.pre_imputation
        if hasattr(args.data, 'pre_imp_num_imputation_steps') and args.data.pre_imp_num_imputation_steps is not None:
            model.hparams.data.pre_imp_num_imputation_steps = args.data.pre_imp_num_imputation_steps
        if hasattr(args.data, 'pre_imp_clip') and args.data.pre_imp_clip is not None:
            model.hparams.data.pre_imp_clip = args.data.pre_imp_clip
        if hasattr(args.data, 'pre_imp_reject') and args.data.pre_imp_reject is not None:
            model.hparams.data.pre_imp_reject = args.data.pre_imp_reject

    if hasattr(args, 'cdi') and args.cdi is not None:
        if hasattr(args.cdi, 'update_comp_schedule') and args.cdi.update_comp_schedule is not None:
            model.hparams.cdi.update_comp_schedule = args.cdi.update_comp_schedule
        if hasattr(args.cdi, 'update_comp_schedule_values') and args.cdi.update_comp_schedule_values is not None:
            model.hparams.cdi.update_comp_schedule_values = args.cdi.update_comp_schedule_values
        if hasattr(args.cdi, 'imputation_delay') and args.cdi.imputation_delay is not None:
            model.hparams.cdi.imputation_delay = args.cdi.imputation_delay
        if hasattr(args.cdi, 'imp_acceptance_check_schedule') and args.cdi.imp_acceptance_check_schedule is not None:
            model.hparams.cdi.imp_acceptance_check_schedule = args.cdi.imp_acceptance_check_schedule
        if hasattr(args.cdi, 'imp_acceptance_check_schedule_values') and args.cdi.imp_acceptance_check_schedule_values is not None:
            model.hparams.cdi.imp_acceptance_check_schedule_values = args.cdi.imp_acceptance_check_schedule_values
        if hasattr(args.cdi, 'num_imp_steps_schedule') and args.cdi.num_imp_steps_schedule is not None:
            model.hparams.cdi.num_imp_steps_schedule = args.cdi.num_imp_steps_schedule
        if hasattr(args.cdi, 'num_imp_steps_schedule_values') and args.cdi.num_imp_steps_schedule_values is not None:
            model.hparams.cdi.num_imp_steps_schedule_values = args.cdi.num_imp_steps_schedule_values
        if hasattr(args.cdi, 'train_ignore_nans') and args.cdi.train_ignore_nans is not None:
            model.hparams.cdi.train_ignore_nans = args.cdi.train_ignore_nans
        if hasattr(args.cdi, 'debug') and hasattr(args.cdi.debug, 'eval_incomplete') and args.cdi.debug.eval_incomplete is not None:
            model.hparams.cdi.debug.eval_incomplete = args.cdi.debug.eval_incomplete
        if isinstance(model, (VarCDI, VarCDIEM)):
            model._init_schedulers()

    if args.reset_vcdi_var is not None:
        if args.reset_vcdi_var == 'rand':
            model.variational_model.reset_parameters()
        elif args.reset_vcdi_var == 'init':
            model.load_pretrained_model(root=None)

    if hasattr(args, 'model_optim') and args.model_optim is not None:
        if hasattr(args.model_optim, 'optimiser') and args.model_optim.optimiser is not None:
            model.hparams.model_optim.optimiser = args.model_optim.optimiser
        if hasattr(args.model_optim, 'learning_rate') and args.model_optim.learning_rate is not None:
            model.hparams.model_optim.learning_rate = args.model_optim.learning_rate

    # Fix decoder and train on validation
    model.hparams.data.train_on_val = True
    model.hparams.fix_generator = True
    if args.fix_encoder:
        model.hparams.fix_encoder = True
    if args.reset_encoder:
        model.fa_model.reset_encoder()

    # Set the gpus
    model.hparams.gpus = args.gpus

    # Prepare profiler
    if hasattr(args, 'profiler') and args.profiler is not None:
        profiler_output = os.path.join(log_dir, 'profiler.out')
        profiler = AdvancedProfiler(
            output_filename=profiler_output
        )
    else:
        profiler = None

    # Prepare model saver
    if args.save:
        model_dir = os.path.join(root_dir, 'saved_models/file')
        model_save_cb = ModelCheckpoint(
            del_old_chpts=args.resume_from_checkpoint is None,
            ckpt_epochs=args.save_custom_epochs,
            filepath=model_dir,
            monitor='val_loss',
            verbose=True,
            save_top_k=args.save_top_k_models,
            mode='min',
            save_weights_only=args.save_weights_only,
            save_last=True,
            prefix='',
            period=args.save_period
        )
    else:
        model_save_cb = None

    callbacks = None
    if args.replace_tqdm:
        callbacks = [PrintingProgressBar(epoch_period=args.print_stat_frequency)]

    # Prepare trainer
    trainer = Trainer(gpus=args.gpus,
                      max_epochs=args.max_epochs,
                      checkpoint_callback=model_save_cb,
                      logger=logger,
                      resume_from_checkpoint=args.resume_from_checkpoint,
                      profiler=profiler,
                      track_grad_norm=int(args.track_grad_norm),
                      accumulate_grad_batches=args.accumulate_grad_batches,
                      log_gpu_memory=args.log_gpu_memory,
                      callbacks=callbacks)

    # Train
    torch.autograd.set_detect_anomaly(args.autograd_detect_anomaly)
    trainer.fit(model)


if __name__ == '__main__':
    args = build_argparser().parse_args()
    print('Args:\n')
    pprint.pprint(flatten_arg_namespace_to_dict(args), width=1)

    # experiment_name = args.experiment_name
    for ckpt in args.ckpts_to_load:
        print(f'Starting epoch: {ckpt}, out of ', args.ckpts_to_load)
        args.ckpt_to_load = ckpt
        # args.experiment_name = f'{experiment_name}_{ckpt}'
        # Train
        main(args)
