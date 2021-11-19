import os

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only


class ModelCheckpoint(ModelCheckpoint):
    """
    Adds another option to clear old checkpoints.
    """
    def __init__(self, del_old_chpts, ckpt_epochs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.del_old_chpts = del_old_chpts
        self.ckpt_epochs = ckpt_epochs

        if del_old_chpts:
            dir_name = os.path.split(kwargs['filepath'])[0]
            for f in os.listdir(dir_name):
                self._del_model(os.path.join(dir_name, f))

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)

        epoch = trainer.current_epoch
        if self.ckpt_epochs is not None and epoch in self.ckpt_epochs:
            filepath = os.path.join(self.dirpath,
                                    f'{self.prefix}_ckpt_epoch_{epoch}_cust.ckpt')
            self._save_model(filepath)
