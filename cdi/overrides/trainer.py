from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.utilities import parsing


class Trainer(pl.Trainer):
    """
    Overrides the default Trainer add_argparse_args to fix issues with jsonargparse
    Overriden for PL v 0.8.1, overrides indicated by `NOTE` below
    """
    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        r"""Extends existing argparse by default `Trainer` attributes.

        Args:
            parent_parser:
                The custom cli arguments parser, which will be extended by
                the Trainer default arguments.

        Only arguments of the allowed types (str, float, int, bool) will
        extend the `parent_parser`.

        Examples:
            >>> import argparse
            >>> import pprint
            >>> parser = argparse.ArgumentParser()
            >>> parser = Trainer.add_argparse_args(parser)
            >>> args = parser.parse_args([])
            >>> pprint.pprint(vars(args))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            {...
             'check_val_every_n_epoch': 1,
             'checkpoint_callback': True,
             'default_root_dir': None,
             'deterministic': False,
             'distributed_backend': None,
             'early_stop_callback': False,
             ...
             'logger': True,
             'max_epochs': 1000,
             'max_steps': None,
             'min_epochs': 1,
             'min_steps': None,
             ...
             'profiler': None,
             'progress_bar_refresh_rate': 1,
             ...}

        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False, )

        blacklist = ['kwargs']
        depr_arg_names = cls.get_deprecated_arg_names() + blacklist

        allowed_types = (str, float, int, bool)

        # TODO: get "help" from docstring :)
        for arg, arg_types, arg_default in (at for at in cls.get_init_arguments_and_types()
                                            if at[0] not in depr_arg_names):
            arg_types = [at for at in allowed_types if at in arg_types]
            if not arg_types:
                # skip argument with not supported type
                continue
            arg_kwargs = {}
            if bool in arg_types:
                arg_kwargs.update(nargs="?")
                # if the only arg type is bool
                if len(arg_types) == 1:
                    # redefine the type for ArgParser needed
                    def use_type(x):
                        # NOTE: Fix one:
                        return x if isinstance(x, bool) else bool(parsing.str_to_bool(x))
                else:
                    # filter out the bool as we need to use more general
                    use_type = [at for at in arg_types if at is not bool][0]
            else:
                use_type = arg_types[0]

            if arg == 'gpus':
                use_type = str
                # NOTE: Fix two:
                arg_default = None

            parser.add_argument(
                f'--{arg}',
                dest=arg,
                default=arg_default,
                type=use_type,
                help='autogenerated by pl.Trainer',
                **arg_kwargs,
            )

        return parser
