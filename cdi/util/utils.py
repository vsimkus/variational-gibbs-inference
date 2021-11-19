import os
import os.path
import random
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch


def flatten_arg_namespace_to_dict(d, parent_key='', sep='.'):
    # Helper to flatten nested hyperparameter namespaces
    items = []
    for k, v in vars(d).items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, SimpleNamespace):
            items.extend(flatten_arg_namespace_to_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def set_random(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    # torch.cuda.manual_seed_all(seed=seed)  # Not needed since I only use 1 GPU.
    random.seed(seed)


def construct_experiment_name(args):
    return (f'{args.experiment_name}/'
            f'm{args.model_seed}_'
            f'd{"_".join(map(str, args.data_seeds))}')


def find_best_model_epoch_from_fs(path):
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    checkpoints = os.listdir(path)
    checkpoints.remove('last.ckpt')
    checkpoints = sorted(checkpoints, reverse=True)

    if len(checkpoints) > 1:
        raise Exception(f'WARNING: multiple checkpoints found: {checkpoints}.')

    checkpoint = os.path.splitext(checkpoints[0].split('_ckpt_epoch_')[1])[0]
    return checkpoint


class EpochScheduler(object):
    def __init__(self, model, epochs, values):
        """
        Uses the model to get the current epoch to choose a value.
        """
        assert (all(epochs[i] <= epochs[i + 1]
                    for i in range(len(epochs)-1))),\
            'Epochs not sorted.'

        assert epochs[0] == 0, \
            'Epochs should start at 0.'

        assert len(epochs) == len(values), \
            'Length of epochs and values should be the same.'

        self.model = model
        self.values = values
        self.schedule = defaultdict(lambda: values[-1])
        for i in range(len(epochs)-1):
            for j in range(epochs[i], epochs[i+1]):
                self.schedule[j] = values[i]

    def get_value(self, epoch=None):
        if epoch is not None:
            return self.schedule[epoch]

        return self.schedule[self.model.current_epoch]

    def get_max_value(self):
        return max(self.values)

    def get_last_value(self):
        return self.values[-1]


class EpochIntervalScheduler(object):
    def __init__(self, model, initial_value, main_value, other_value, start_epoch, period):
        """
        Uses the model to get the current epoch to choose a value.
        """
        assert (start_epoch >= 0),\
            'start_epoch cannot be negative.'

        self.model = model
        self.initial_value = initial_value
        self.main_value = main_value
        self.other_value = other_value
        self.start_epoch = start_epoch
        self.period = period

    def get_value(self, epoch=None):
        if epoch is None:
            epoch = self.model.current_epoch

        if epoch < self.start_epoch:
            return self.initial_value
        elif (epoch - self.start_epoch) % self.period == 0:
            return self.main_value
        else:
            return self.other_value
