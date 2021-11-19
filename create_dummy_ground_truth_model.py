
import logging as log
import os

import torch
# from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from cdi.overrides.trainer import Trainer
from cdi.trainers.posterior_cdi import PosteriorCDI
from cdi.util.arg_utils import convert_namespace
from cdi.util.utils import (construct_experiment_name,
                            flatten_arg_namespace_to_dict)
from train import build_argparser as train_argparser

log.root.setLevel(log.INFO)


# TODO: move this to helper scripts

def build_argparser():
    parser = train_argparser()

    return parser


def main(args):
    # Convert jsonargparse's SimpleNamespace to argparse.Namespace
    # which is required by pytorch_lightning
    args = convert_namespace(args)

    # Prepare model
    model = PosteriorCDI(args)

    # Load ground truth data
    dataset = PosteriorCDI.load_dataset(args, stage='train')
    F = dataset.meta_data['F']
    mean = dataset.meta_data['c']
    cov = dataset.meta_data['Psi']

    # Set the FA model to ground truth
    model.fa_model.set_parameters(F=torch.from_numpy(F).squeeze(),
                                  mean=torch.from_numpy(mean).squeeze(),
                                  log_cov=torch.log(torch.from_numpy(cov).squeeze()))

    # Prepare path
    root_dir = os.path.join(os.path.abspath(args.output_root_dir),
                            'trained_models',
                            args.exp_group,
                            construct_experiment_name(args))
    model_path = os.path.join(root_dir, 'saved_models/last.ckpt')

    # Reuse PL trainer for storing the model
    mc = ModelCheckpoint(
        filepath=model_path,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    trainer = Trainer(checkpoint_callback=mc)
    # These hacks will break with PL upgrades
    trainer.optimizers = []
    trainer.model = model

    # Save the model
    trainer.save_checkpoint(model_path)


if __name__ == '__main__':
    args = build_argparser().parse_args()
    print('Args:\n', flatten_arg_namespace_to_dict(args))

    # Create model
    main(args)
