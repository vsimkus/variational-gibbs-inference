
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
from cdi.trainers.refit_encoder_on_test import RefitEncoderOnTest
from cdi.trainers.mc_expectation_maximisation import MCEM
from cdi.trainers.expectation_maximisation import EM
from cdi.trainers.variational_inference import VI
from cdi.trainers.posterior_cdi import PosteriorCDI
from cdi.trainers.var_mle_pretraining import VarMLEPretraining
from cdi.trainers.variational_cdi import VarCDI
from cdi.trainers.variational_cdi_em import VarCDIEM
from cdi.trainers.mcimp import MCIMP
from cdi.trainers.plmcmc import PLMCMC
from cdi.trainers.plmcmc_orig import PLMCMC_orig
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
    parser.add_argument('--autograd_detect_anomaly',
                        type=parse_bool, default=False,
                        help=('DEBUG: Whether torch autograd detect anomaly'
                              ' should be used.'))
    parser.add_argument('--method',
                        type=str, required=True,
                        choices=['complete-case', 'variational',
                                 'analytical', 'mcimp', 'expectation-maximisation',
                                 'mc-expectation-maximisation',
                                 'variational-inference',
                                 'var-pretraining', 'variational-em',
                                 'refit-encoder-on-test',
                                 'plmcmc', 'plmcmc-orig'],
                        help=('Which approximation method to be'
                              ' used in CDI/EM/MLE.'))
    parser.add_argument('--replace_tqdm', type=parse_bool,
                        default=False, help='Replace tqdm with printing dumps.')
    parser.add_argument('--print_stat_frequency', type=int,
                        default=20, help='How often should we print stats with the Printing progress.')
    parser.add_argument('--save', default=True,
                        help=('Whether to save the weight snapshots.'))

    temp_args, _ = parser._parse_known_args()
    if temp_args.method == 'variational':
        parser = VarCDI.add_model_args(parser)
    elif temp_args.method == 'variational-em':
        parser = VarCDIEM.add_model_args(parser)
    elif temp_args.method == 'analytical':
        parser = PosteriorCDI.add_model_args(parser)
    elif temp_args.method == 'mcimp':
        parser = MCIMP.add_model_args(parser)
    elif temp_args.method == 'plmcmc':
        parser = PLMCMC.add_model_args(parser)
    elif temp_args.method == 'plmcmc-orig':
        parser = PLMCMC_orig.add_model_args(parser)
    elif temp_args.method == 'complete-case':
        parser = CompleteMLE.add_model_args(parser)
    elif temp_args.method == 'refit-encoder-on-test':
        parser = RefitEncoderOnTest.add_model_args(parser)
    elif temp_args.method == 'mc-expectation-maximisation':
        parser = MCEM.add_model_args(parser)
    elif temp_args.method == 'expectation-maximisation':
        parser = EM.add_model_args(parser)
    elif temp_args.method == 'variational-inference':
        parser = VI.add_model_args(parser)
    elif temp_args.method == 'var-pretraining':
        parser = VarMLEPretraining.add_model_args(parser)
    else:
        print(('No such approximation'
               f'`{temp_args.method}`!'))
        sys.exit()

    return parser


def main(args):
    # Convert jsonargparse's SimpleNamespace to argparse.Namespace
    # which is required by pytorch_lightning
    args = convert_namespace(args)

    # Prepare CDI
    if args.method == 'variational':
        model = VarCDI(args)
    elif args.method == 'variational-em':
        model = VarCDIEM(args)
    elif args.method == 'analytical':
        model = PosteriorCDI(args)
    elif args.method == 'mcimp':
        model = MCIMP(args)
    elif args.method == 'plmcmc':
        model = PLMCMC(args)
    elif args.method == 'plmcmc-orig':
        model = PLMCMC_orig(args)
    elif args.method == 'complete-case':
        model = CompleteMLE(args)
    elif args.method == 'refit-encoder-on-test':
        model = RefitEncoderOnTest(args)
    elif args.method == 'mc-expectation-maximisation':
        model = MCEM(args)
    elif args.method == 'expectation-maximisation':
        model = EM(args)
    elif args.method == 'variational-inference':
        model = VI(args)
    elif args.method == 'var-pretraining':
        model = VarMLEPretraining(args)
    else:
        print(('No such approximation '
               f'`{args.method}`!'))
        sys.exit()

    # Prepare logger
    root_dir = os.path.join(os.path.abspath(args.output_root_dir),
                            'trained_models',
                            args.exp_group,
                            construct_experiment_name(args))
    log_dir = os.path.join('file:/', root_dir, 'logs')
    logger = FSLogger(log_dir,
                      continue_from_checkpoint=args.resume_from_checkpoint)

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
                      callbacks=callbacks,
                      gradient_clip_val=args.gradient_clip_val)

    # Train
    torch.autograd.set_detect_anomaly(args.autograd_detect_anomaly)
    trainer.fit(model)


if __name__ == '__main__':
    args = build_argparser().parse_args()
    print('Args:\n')
    pprint.pprint(flatten_arg_namespace_to_dict(args), width=1)

    # Train
    main(args)
