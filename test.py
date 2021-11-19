
import logging as log
import os
import pprint
import sys

from jsonargparse import ActionConfigFile
from jsonargparse import ArgumentParser
# from pytorch_lightning import Trainer

from cdi.overrides.trainer import Trainer
from cdi.trainers.trainer_base import TestSetRandomCallback
from cdi.trainers.complete_mle import CompleteMLE
from cdi.trainers.mc_expectation_maximisation import MCEM
from cdi.trainers.expectation_maximisation import EM
from cdi.trainers.posterior_cdi import PosteriorCDI
from cdi.trainers.var_mle_pretraining import VarMLEPretraining
from cdi.trainers.variational_cdi import VarCDI
from cdi.trainers.variational_cdi_em import VarCDIEM
from cdi.trainers.plmcmc import PLMCMC
from cdi.util.fs_logger import FSLogger
from cdi.util.arg_utils import parse_bool
from cdi.util.utils import (construct_experiment_name,
                            flatten_arg_namespace_to_dict,
                            find_best_model_epoch_from_fs)
from cdi.util.print_progress_bar import PrintingProgressBar


log.root.setLevel(log.INFO)


def build_argparser():
    parser = ArgumentParser(parser_mode='jsonnet')
    parser = Trainer.add_argparse_args(parser)
    parser = ArgumentParser(parents=[parser],
                            parser_mode='jsonnet',
                            add_help=False)
    parser.add_argument('--input_root_dir',
                        type=str, default='.',
                        help='Root directory for inputs.')
    parser.add_argument('--output_root_dir',
                        type=str, default='.',
                        help='Root directory for outputs.')
    parser.add_argument('--test_config',
                        type=str, action=ActionConfigFile)
    parser.add_argument('--experiment_name',
                        type=str, required=True,
                        help='Name of experiment.')
    parser.add_argument('--exp_group',
                        type=str, required=True,
                        help='Experiment group.')
    parser.add_argument('--method',
                        type=str, required=True,
                        help=('Name of the model to be used in the test run.'))
    parser.add_argument('--checkpoint',
                        type=str, default='latest',
                        help=('Checkpoint to load.'))
    parser.add_argument('--replace_tqdm', type=parse_bool,
                        default=False, help='Replace tqdm with printing dumps.')
    parser.add_argument('--print_stat_frequency', type=int,
                        default=20, help='How often should we print stats with the Printing progress.')

    temp_args, _ = parser._parse_known_args()
    if temp_args.method == 'variational':
        parser = VarCDI.add_test_args(parser)
    elif temp_args.method == 'variational-em':
        parser = VarCDIEM.add_test_args(parser)
    elif temp_args.method == 'analytical':
        parser = PosteriorCDI.add_test_args(parser)
    elif temp_args.method == 'plmcmc':
        parser = PLMCMC.add_test_args(parser)
    elif temp_args.method == 'complete-case':
        parser = CompleteMLE.add_test_args(parser)
    elif temp_args.method == 'mc-expectation-maximisation':
        parser = MCEM.add_test_args(parser)
    elif temp_args.method == 'expectation-maximisation':
        parser = EM.add_test_args(parser)
    elif temp_args.method == 'var-pretraining':
        parser = VarMLEPretraining.add_test_args(parser)
    else:
        print((f'No such method `{temp_args.method}`!'))
        sys.exit()

    # Used to find the relevant model
    parser.add_argument('--model_seed',
                        type=int, default=20190508,
                        help=('Used to construct the relevant model path.'))
    parser.add_argument('--data_seeds',
                        type=int, nargs=3,
                        default=[20200325, 20200406, 20200407],
                        help=('Used to construct the relevant model path.'))

    return parser


def main(args):
    model_path = os.path.join(args.input_root_dir,
                              'trained_models',
                              args.exp_group,
                              construct_experiment_name(args),
                              'saved_models')

    if args.checkpoint == 'best':
        chkpt_epoch = find_best_model_epoch_from_fs(model_path)
        checkpoint = f'_ckpt_epoch_{chkpt_epoch}'
    else:
        checkpoint = args.checkpoint

    model_path = os.path.join(model_path, f'{checkpoint}.ckpt')

    # Prepare CDI
    if args.method == 'variational':
        model = VarCDI.load_from_checkpoint(model_path)
    elif args.method == 'variational-em':
        model = VarCDIEM.load_from_checkpoint(model_path)
    elif args.method == 'analytical':
        model = PosteriorCDI.load_from_checkpoint(model_path)
    elif args.method == 'plmcmc':
        model = PLMCMC.load_from_checkpoint(model_path)
    elif args.method == 'complete-case':
        model = CompleteMLE.load_from_checkpoint(model_path)
    elif args.method == 'mc-expectation-maximisation':
        model = MCEM.load_from_checkpoint(model_path)
    elif args.method == 'expectation-maximisation':
        model = EM.load_from_checkpoint(model_path)
    elif args.method == 'var-pretraining':
        model = VarMLEPretraining.load_from_checkpoint(model_path)
    else:
        print((f'No such method `{args.method}`!'))
        sys.exit()

    # Set the experiment name (sometimes might be different due to copying models)
    model.hparams.experiment_name = args.experiment_name

    # Prepare logger
    root_dir = os.path.join(os.path.abspath(args.output_root_dir),
                            'trained_models',
                            args.exp_group,
                            construct_experiment_name(args))
    log_dir = os.path.join('file:/', root_dir, 'evaluations')

    logger = FSLogger(log_dir, continue_from_checkpoint=None)

    callbacks = [TestSetRandomCallback()]
    if args.replace_tqdm:
        callbacks = callbacks.append(PrintingProgressBar(epoch_period=args.print_stat_frequency))

    # Prepare trainer
    trainer = Trainer(gpus=args.gpus, logger=logger,
                      callbacks=callbacks)

    # Run test
    model.hparams.test = args
    trainer.test(model)


if __name__ == '__main__':
    args = build_argparser().parse_args()

    print('Args:\n')
    pprint.pprint(flatten_arg_namespace_to_dict(args), width=1)

    # Train
    main(args)
