from typing import Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.types import _size


"""
Copied from v1.8
"""


class Unflatten(nn.Module):
    r"""
    Unflattens a tensor dim expanding it to a desired shape. For use with :class:`~nn.Sequential`.

    * :attr:`dim` specifies the dimension of the input tensor to be unflattened, and it can
      be either `int` or `str` when `Tensor` or `NamedTensor` is used, respectively.

    * :attr:`unflattened_size` is the new shape of the unflattened dimension of the tensor and it can be
      a `tuple` of ints or a `list` of ints or `torch.Size` for `Tensor` input;  a `NamedShape`
      (tuple of `(name, size)` tuples) for `NamedTensor` input.

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`

    Args:
        dim (Union[int, str]): Dimension to be unflattened
        unflattened_size (Union[torch.Size, Tuple, List, NamedShape]): New shape of the unflattened dimension

    Examples:
        >>> input = torch.randn(2, 50)
        >>> # With tuple of ints
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, (2, 5, 5))
        >>> )
        >>> output = m(input)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
        >>> # With torch.Size
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, torch.Size([2, 5, 5]))
        >>> )
        >>> output = m(input)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
        >>> # With namedshape (tuple of tuples)
        >>> input = torch.randn(2, 50, names=('N', 'features'))
        >>> unflatten = nn.Unflatten('features', (('C', 2), ('H', 5), ('W', 5)))
        >>> output = unflatten(input)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
    """
    NamedShape = Tuple[Tuple[str, int]]

    __constants__ = ['dim', 'unflattened_size']
    dim: Union[int, str]
    unflattened_size: Union[_size, NamedShape]

    def __init__(self, dim: Union[int, str], unflattened_size: Union[_size, NamedShape]) -> None:
        super(Unflatten, self).__init__()

        if isinstance(dim, int):
            self._require_tuple_int(unflattened_size)
        elif isinstance(dim, str):
            self._require_tuple_tuple(unflattened_size)
        else:
            raise TypeError("invalid argument type for dim parameter")

        self.dim = dim
        self.unflattened_size = unflattened_size

    def _require_tuple_tuple(self, input):
        if (isinstance(input, tuple)):
            for idx, elem in enumerate(input):
                if not isinstance(elem, tuple):
                    raise TypeError("unflattened_size must be tuple of tuples, " +
                                    "but found element of type {} at pos {}".format(type(elem).__name__, idx))
            return
        raise TypeError("unflattened_size must be a tuple of tuples, " +
                        "but found type {}".format(type(input).__name__))

    def _require_tuple_int(self, input):
        if (isinstance(input, (tuple, list))):
            for idx, elem in enumerate(input):
                if not isinstance(elem, int):
                    raise TypeError("unflattened_size must be tuple of ints, " +
                                    "but found element of type {} at pos {}".format(type(elem).__name__, idx))
            return
        raise TypeError("unflattened_size must be a tuple of ints, but found type {}".format(type(input).__name__))

    def forward(self, input: Tensor) -> Tensor:
        # NOTE: override
        return input.reshape(-1, *self.unflattened_size)
