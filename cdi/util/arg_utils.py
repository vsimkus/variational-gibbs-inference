import argparse
from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict


def parse_bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def args_required_length(nmin, nmax):
    """
    When we need nargs to be in range between `nmin` and `nmax`.
    From: https://stackoverflow.com/questions/4194948/python-argparse-is-there-a-way-to-specify-a-range-in-nargs
    """
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin <= len(values) <= nmax:
                msg = 'argument "{f}" requires between {nmin} and {nmax} arguments'.format(
                    f=self.dest, nmin=nmin, nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength


def convert_namespace(cfg: Dict[str, Any]) -> Namespace:
    """Converts a nested SimpleNamespace to nested argparse.Namespace
    Args:
        cfg (dict): The configuration to process.

    Returns:
        argparse.Namespace: The nested configuration namespace.
    """
    cfg = deepcopy(cfg)

    def convert_namespace(cfg):
        cfg = dict(vars(cfg))
        for k, v in cfg.items():
            if isinstance(v, SimpleNamespace):
                cfg[k] = convert_namespace(v)
            elif isinstance(v, list):
                for nn, vv in enumerate(v):
                    if isinstance(vv, SimpleNamespace):
                        cfg[k][nn] = convert_namespace(vv)
        return Namespace(**cfg)
    return convert_namespace(cfg)


def dict_to_namespace(d):
    x = Namespace()
    _ = [setattr(x, k, dict_to_namespace(v)) if isinstance(v, dict) else setattr(x, k, v) for k, v in d.items() ]
    return x
