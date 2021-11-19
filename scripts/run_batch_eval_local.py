#
# Helper script to run a batch of evaluations locally
#
import argparse
import os
import subprocess

from cdi.util.utils import flatten_arg_namespace_to_dict


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str, required=True,
                        help=('Name of the dataset, or the name of the first '
                              'subdirectory under `experiment_configs`.'))
    parser.add_argument('--eval_type',
                        type=str, required=True,
                        help=('Type of evaluation to be run. Matches the '
                              'subdirectory name of the config files.'))
    parser.add_argument('--model_file',
                        type=str,
                        help=('Path to the `models.txt` file, which contains '
                              'the names of the models for which to run the '
                              'evaluations.'))
    parser.add_argument('--models',
                        type=str, nargs='+',
                        help=('Names of the model for which to create the '
                              'evaluation configs.'))
    parser.add_argument('--groups',
                        type=int, nargs='+', required=True,
                        help=('Experiment groups to run i.e. [1..6]'))

    return parser


if __name__ == '__main__':
    args, unk_args = argparser().parse_known_args()
    assert (args.model_file is not None
            or args.models is not None),\
        'Either `model_file` or `models` arguments needs to be provided!'

    assert not (args.model_file is not None
                and args.models is not None),\
        'Only one of `model_file` or `models` arguments should be provided!'

    print('Args:\n', flatten_arg_namespace_to_dict(args))
    print('Test args:\n', unk_args)

    if args.model_file is not None:
        with open(args.model_file, 'r') as f:
            models = f.readlines()
            models = set(model.strip() for model in models if len(model.strip()) != 0)
    elif args.models is not None:
        models = args.models

    eval_conf_path = os.path.join('experiment_configs',
                                  args.dataset,
                                  args.eval_type)

    # Run evaluations
    for g in args.groups:
        for model in models:
            model_config_name = f'{model}.json'
            path = os.path.join(eval_conf_path, str(g), model_config_name)

            cli_args = ['--test_config', path] + unk_args
            print(f'Executing with args: {cli_args}')
            subprocess.call(['python', 'test.py'] + cli_args)
