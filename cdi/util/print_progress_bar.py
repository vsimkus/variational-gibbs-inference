import sys

from pytorch_lightning.callbacks.progress import ProgressBarBase


class PrintingProgressBar(ProgressBarBase):
    def __init__(self, epoch_period=1):
        super().__init__()
        self.enable()

        self.epoch_period = epoch_period

    def enable(self):
        self.is_enabled = True

    def disable(self):
        self.is_enabled = False

    def on_batch_end(self, trainer, pl_module):
        super().on_batch_end(trainer, pl_module)

        final_batch = self.train_batch_idx == self.total_train_batches
        if self.is_enabled and final_batch and trainer.current_epoch % self.epoch_period == 0:
            sys.stdout.flush()
            stats = ", ".join(["=".join([key, str(val)]) for key, val in trainer.progress_bar_dict.items()])
            stats_out = f'Train epoch {trainer.current_epoch + 1}: {stats}'
            sys.stdout.write(f'{stats_out}\r')

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)

        final_batch = self.val_batch_idx == self.total_val_batches
        if self.is_enabled and final_batch and trainer.current_epoch % self.epoch_period == 0:
            sys.stdout.flush()
            stats = ", ".join(["=".join([key, str(val)]) for key, val in trainer.progress_bar_dict.items()])
            stats_out = f'Val epoch {trainer.current_epoch + 1}: {stats}'
            sys.stdout.write(f'{stats_out}\r')
