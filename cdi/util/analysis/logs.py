import os
import re
from collections import defaultdict
from typing import Optional, List, Callable

import numpy as np


def collate_stats(all_stats):
    out_stats = defaultdict(lambda: defaultdict(list))
    for group, stats in all_stats.items():
        for exp, exp_stats in stats.items():
            out_stats[exp]['group'].append(group)
            for k, v in exp_stats.items():
                out_stats[exp][f'{k} mean'].append(np.mean(v))
                out_stats[exp][f'{k} std'].append(np.std(v))
    return out_stats


def trace_data_chains(dirname: str,
                      logname: str,
                      epochs: Optional[List[str]] = None,
                      indices: Optional[np.ndarray] = None,
                      ignore_tensors: List[str] = None,
                      transform_fn: Callable[[str, np.ndarray],
                                             np.ndarray] = None):
    regex = re.compile(rf'^{logname}_(\d+).npz$')
    if epochs is None:
        epochs = sorted([int(regex.search(d).group(1))
                        for d in os.listdir(dirname) if regex.match(d)])

    chains = defaultdict(list)
    for epoch in epochs:
        data = np.load(os.path.join(dirname, f'{logname}_{epoch}.npz'),
                       allow_pickle=True)
        chains['curr_epoch'].append(data['curr_epoch'])

        # Get index to create a sorting index
        I = data['I'].astype(np.long)
        sort_idx = I.argsort()
        for name, tensor in data.items():
            if (name != 'curr_epoch'
                and not (ignore_tensors is not None
                         and name in ignore_tensors)):
                sorted_tensor = tensor[sort_idx]
                # Select subset
                if indices is not None:
                    sorted_tensor = sorted_tensor[indices]
                # Apply transform
                if transform_fn is not None:
                    sorted_tensor = transform_fn(name, sorted_tensor)

                chains[name].append(sorted_tensor)

    for name, tensors in chains.items():
        chains[name] = np.stack(tensors)

    return chains
