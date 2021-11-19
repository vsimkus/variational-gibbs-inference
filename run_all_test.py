
import logging as log
import pprint
import subprocess

from jsonargparse import ActionConfigFile
from jsonargparse import ArgumentParser
# from pytorch_lightning import Trainer

from cdi.util.utils import flatten_arg_namespace_to_dict

log.root.setLevel(log.INFO)


def build_argparser():
    parser = ArgumentParser(parser_mode='jsonnet')
    parser.add_argument('--output_root_dir',
                        type=str, required=True,
                        help='Root directory for outputs.')
    parser.add_argument('--config',
                        type=str, action=ActionConfigFile)
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument('--exp_group',
                        type=str, required=True,
                        help='Experiment group.')
    parser.add_argument('--test_exp_group',
                        type=str, required=True,
                        help='Experiment test group.')
    parser.add_argument('--ckpts_to_load', nargs='+',
                        type=str, help='Which checkpoints to load.')

    # Used to find the relevant model
    parser.add_argument('--model_seed',
                        type=int, required=False,
                        help=('Used to construct the relevant model path.'))
    parser.add_argument('--data_seeds',
                        type=int, nargs=3, required=False,
                        help=('Used to construct the relevant model path.'))

    return parser


if __name__ == '__main__':
    args, unk_args = build_argparser()._parse_known_args()
    print('Args:\n')
    pprint.pprint(flatten_arg_namespace_to_dict(args), width=1)

    output_root_dir = args.output_root_dir if hasattr(args, 'output_root_dir') else '.'

    if args.model_seed is not None:
        unk_args.append(f'--model_seed={args.model_seed}')

    if args.data_seeds is not None:
        unk_args.append('--data_seeds')
        unk_args += [f'{ds}' for ds in args.data_seeds]

    for ckpt in args.ckpts_to_load:
        print(f'Starting epoch: {ckpt}, out of ', args.ckpts_to_load, flush=True)

        ckpta = ckpt
        ckpt = f'_ckpt_epoch_{ckpt}' if ckpt != 'last' else 'last'

        # Test
        cli_args = [f'--output_root_dir={output_root_dir}',
                    '--test_config', f'experiment_configs/{args.test_exp_group}/{args.test_config}',
                    f'--checkpoint={ckpt}',
                    f'--output_suffix={ckpta}'] + unk_args
        print(f'Executing test with args: {cli_args}', flush=True)
        subprocess.call(['python', 'test.py'] + cli_args)
