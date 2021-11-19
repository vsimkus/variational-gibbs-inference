
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
    parser.add_argument('--input_root_dir',
                        type=str,
                        help='Root directory for inputs for test. Defaults to output dir.')
    parser.add_argument('--config',
                        type=str, action=ActionConfigFile)
    parser.add_argument('--train_config', type=str, required=False)
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument('--exp_group',
                        type=str, required=True,
                        help='Experiment group.')
    parser.add_argument('--test_exp_group',
                        type=str, required=True,
                        help='Experiment test group.')
    parser.add_argument('--experiment_name',
                        type=str, required=True,
                        help='Name of experiment.')
    parser.add_argument('--ckpts_to_load', nargs='+',
                        type=str, help='Which checkpoints to load.')
    parser.add_argument('--save', default=False,
                        help=('Whether to save the weight snapshots.'))
    parser.add_argument('--skip_train', default=False,
                        help=('Should skip train or not'))

    return parser


if __name__ == '__main__':
    args, unk_args = build_argparser()._parse_known_args()
    print('Args:\n')
    pprint.pprint(flatten_arg_namespace_to_dict(args), width=1)

    output_root_dir = args.output_root_dir if hasattr(args, 'output_root_dir') else '.'
    input_root_dir = args.input_root_dir if hasattr(args, 'input_root_dir') else output_root_dir
    save = args.save if hasattr(args, 'save') else False
    skip_train = args.skip_train if hasattr(args, 'skip_train') else False

    assert not (args.train_config is None and not args.skip_train),\
        'train_config should be provided if _not_ skipping training!'

    experiment_name = args.experiment_name
    for ckpt in args.ckpts_to_load:
        print(f'Starting epoch: {ckpt}, out of ', args.ckpts_to_load, flush=True)

        exp = f'{experiment_name}_ckpt_{ckpt}'
        ckpt = f'_ckpt_epoch_{ckpt}' if ckpt != 'last' else 'last'

        if skip_train:
            print('Skipping training', flush=True)
        else:
            # Re-Train
            cli_args = [f'--output_root_dir={output_root_dir}',
                        '--config', f'experiment_configs/{args.exp_group}/{args.train_config}',
                        f'--experiment_name={exp}',
                        f'--model_checkpoint={ckpt}',
                        f'--save={save}'] + unk_args
            print(f'Executing train with args: {cli_args}', flush=True)
            subprocess.call(['python', 'train.py'] + cli_args)

        # Test
        cli_args = [f'--output_root_dir={output_root_dir}',
                    f'--input_root_dir={input_root_dir}',
                    '--test_config', f'experiment_configs/{args.test_exp_group}/{args.test_config}',
                    f'--experiment_name={exp}'] + unk_args
        print(f'Executing test with args: {cli_args}', flush=True)
        subprocess.call(['python', 'test.py'] + cli_args)
