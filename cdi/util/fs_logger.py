
import os
import shutil
from argparse import Namespace
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, Union

import numpy as np
import torch
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

import cdi.util.stats_utils as stats_utils


# TODO: Refactor
class FSLogger(LightningLoggerBase):
    def __init__(self, experiment_logs, continue_from_checkpoint=None):
        r"""
        Logs in filesystem
        """
        super().__init__()
        self.experiment_logs = experiment_logs
        self.continue_from_checkpoint = continue_from_checkpoint
        self.continue_logging = continue_from_checkpoint is not None

        self.stats = defaultdict(list)

        if not os.path.exists(self.experiment_logs):
            os.makedirs(self.experiment_logs)

        # If continuing - load previous stats
        if self.continue_logging:
            old_stats_path = os.path.join(os.path.dirname(self.continue_from_checkpoint),
                                          '../logs/summary.csv')
            new_stats_path = os.path.join(self.experiment_logs, 'summary.csv')
            try:
                shutil.copyfile(old_stats_path, new_stats_path)
            except shutil.SameFileError:
                pass
            self.stats = stats_utils.load_statistics(self.experiment_logs,
                                                     'summary.csv')

        # TODO: load tensors if continuing
        self.tensor_logs = os.path.join(self.experiment_logs, 'tensors')
        if not os.path.exists(self.tensor_logs):
            os.makedirs(self.tensor_logs)

        # Create a buffer for any tensors that are accumulated before logging
        self._default_tensor_buffers_val = lambda: defaultdict(list)
        self.tensor_buffers = defaultdict(self._default_tensor_buffers_val)

        if len(self.stats['curr_epoch']) == 0:
            self.start_epoch = 0
        else:
            self.start_epoch = self.stats['curr_epoch'][-1]+1

    @rank_zero_only
    def log_hyperparams(self,
                        params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)

        if len(params) > 0:
            stats_utils.save_hyperparams(self.experiment_logs,
                                         'hyperparams.csv',
                                         params)

    @rank_zero_only
    def log_initialisation_metric(self, metrics: Dict[str, float]):
        stats_utils.save_statistics(
                experiment_log_dir=self.experiment_logs,
                filename='init_summary.csv',
                stats_dict=metrics,
                current_epoch=0,
                save_full_dict=True)

    @rank_zero_only
    def log_metrics(self,
                    metrics: Dict[str, float],
                    step: int) -> None:
        # Don't log sanity check step
        if 'curr_epoch' in metrics and metrics['curr_epoch'] <= self.start_epoch:
            return

        for k, v in metrics.items():
            # PL logs after each step, just ignore those
            if k == 'epoch':
                continue
            # Prevent storing tensors in the stats dictionary
            # to avoid accumulation of the graph
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.stats[k].append(v)

        # `curr_epoch` key is only available in val metrics,
        # which means that train metrics have already been added
        if len(metrics) > 0 and 'curr_epoch' in metrics:
            stats_utils.save_statistics(
                    experiment_log_dir=self.experiment_logs,
                    filename='summary.csv',
                    stats_dict=self.stats,
                    current_epoch=metrics['curr_epoch']-1,
                    continue_from_mode=(self.continue_logging
                                        or metrics['curr_epoch'] > 2))

    @rank_zero_only
    def log_tensors(self,
                    epoch: int,
                    logname: str = 'tensors',
                    **kwargs) -> None:
        tensors = {
            "curr_epoch": epoch,
        }
        # Add tensors to a dictionary
        for k, v in kwargs.items():
            if not isinstance(v, np.ndarray):
                v = v.numpy()
            # To save space on large index tensors, we don't need int64
            if v.dtype == np.int64:
                v = v.astype(np.int32)

            tensors[k] = v

        # Store to file
        filepath = os.path.join(self.tensor_logs, f'{logname}_{epoch}.npz')
        np.savez_compressed(filepath, **tensors)

    @rank_zero_only
    def accumulate_tensors(self,
                           tensor_group: str,
                           **tensors) -> None:
        buffer = self.tensor_buffers[tensor_group]
        # Accumulate tensors in buffer
        for name, tensor in tensors.items():
            buffer[name].append(tensor)

    @rank_zero_only
    def save_accumulated_tensors(self, tensor_group: str, epoch: int) -> None:
        buffer = self.tensor_buffers[tensor_group]
        tensors = {
            name: torch.cat(tensors_list)
            for name, tensors_list in buffer.items()
        }
        self.log_tensors(epoch, logname=tensor_group, **tensors)

        # Reset the tensor_group
        self.tensor_buffers[tensor_group] = self._default_tensor_buffers_val()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # Store any buffers that weren't saved.
        for group, buffer in self.tensor_buffers.items():
            if len(buffer) != 0:
                self.save_accumulated_tensors(group, epoch='final')

    def _convert_params(self, params: Union[Dict[str, Any], Namespace]) -> Dict[str, Any]:
        # in case converting from namespace
        if isinstance(params, (Namespace, SimpleNamespace)):
            params = vars(params)

        if params is None:
            params = {}

        return params

    @property
    def experiment(self):
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.experiment_logs

    @property
    def version(self) -> str:
        return ""
