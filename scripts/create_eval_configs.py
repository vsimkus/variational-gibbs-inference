#
# Helper script to create log-likelihood evaluation
# config files from learning-experiment configs
#
import argparse
import json
import os
import pprint

from cdi.util.utils import flatten_arg_namespace_to_dict


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str, required=True,
                        help=('Name of the dataset, or the name of the first '
                              'subdirectory under `experiment_configs`.'))
    parser.add_argument('--eval_type',
                        type=str, required=True)
    parser.add_argument('--checkpoint',
                        type=str, default='latest',
                        choices=['latest', 'best'])
    # One of the below options is necessary
    parser.add_argument('--model_file',
                        type=str,
                        help=('Path to the `models.txt` file, which contains '
                              'the names of the models for which to create the'
                              ' evaluation configs.'))
    parser.add_argument('--models',
                        type=str, nargs='+',
                        help=('Names of the model for which to create the '
                              'evaluation configs.'))
    parser.add_argument('--options', type=str,
                        help=('A dictionary in a string. Options to be added '
                              'to the eval configs.'))

    args = parser.parse_args()

    assert (args.model_file is not None
            or args.models is not None),\
        'Either `model_file` or `models` arguments needs to be provided!'

    assert not (args.model_file is not None
                and args.models is not None),\
        'Only one of `model_file` or `models` arguments should be provided!'

    return args


def load_config_as_dict(path):
    with open(path, 'r') as f:
        # Strip line comments before parsing json
        config = ''.join(line for line in f
                         if not line.lstrip().startswith('//'))
        config_json = json.loads(config)

        return config_json


def convert_learning_to_eval_config(config, eval_type,
                                    checkpoint, options=None):
    allowed_keys = set(['exp_group',
                        'experiment_name',
                        'model_seed',
                        'data_seeds',
                        # new
                        'eval_type'
                        'test_seed',
                        'method',
                        'checkpoint'])

    eval_config = {
        'test_seed': 20200428,
        'checkpoint': checkpoint,
        'eval_type': eval_type
    }
    # Filter config options
    for k, v in config.items():
        if k in allowed_keys:
            eval_config[k] = v

    # Add the extra options
    if options is not None:
        eval_config.update(options)

    return eval_config


def main(args):
    # Create a list of models
    if args.models is not None:
        models = set(args.models)
    elif args.model_file is not None:
        with open(args.model_file, 'r') as f:
            models = f.readlines()
            models = set(model.strip() for model in models
                         if len(model.strip()) != 0)

    options = None
    if args.options is not None:
        options = json.loads(args.options)

    root_dir = os.path.join('experiment_configs', args.dataset)
    learning_config_dir = os.path.join(root_dir, 'learning_experiments')
    for d in os.listdir(learning_config_dir):
        group_dir = os.path.join(learning_config_dir, d)

        # Each group should have _all_ models to convert
        models_pending = models.copy()
        for config_file in os.listdir(group_dir):
            # String .json extension and check if we should convert this
            model_name = os.path.splitext(config_file)[0]
            if model_name not in models_pending:
                continue
            # Load config
            config = load_config_as_dict(os.path.join(group_dir, config_file))

            # Convert config
            eval_config = convert_learning_to_eval_config(
                                                    config,
                                                    args.eval_type,
                                                    checkpoint=args.checkpoint,
                                                    options=options)

            # Store eval config
            eval_config_dir = os.path.join(root_dir, args.eval_type, d)
            if not os.path.exists(eval_config_dir):
                os.makedirs(eval_config_dir)
            eval_config_path = os.path.join(eval_config_dir, config_file)
            with open(eval_config_path, 'w+') as f_out:
                json.dump(eval_config, f_out, indent=4)

            # Remove from pending
            model_name = os.path.splitext(config_file)[0]
            models_pending.remove(model_name)

        if len(models_pending) != 0:
            print(f'WARNING: Some model configs were not found in '
                  f'{group_dir}: '
                  f'{models_pending}.')


if __name__ == '__main__':
    args = parse_arguments()
    print('Args:\n')
    pprint.pprint(flatten_arg_namespace_to_dict(args), width=1)

    main(args)
