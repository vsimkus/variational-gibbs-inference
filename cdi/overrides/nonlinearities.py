import numpy as np
import torch
from torch import nn

from nflows.transforms import splines
from nflows.transforms.nonlinearities import _share_across_batch
from nflows.transforms.base import Transform
from nflows.utils import torchutils

from cdi.overrides.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
    rational_quadratic_spline
)


class PiecewiseRationalQuadraticCDF(Transform):
    def __init__(
        self,
        shape,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        identity_init=False,
        min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE,
        check_discriminant=True,
    ):
        super().__init__()

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        self.tail_bound = tail_bound
        self.tails = tails

        self.check_discriminant = check_discriminant

        if isinstance(shape, int):
            shape = (shape,)
        if identity_init:
            self.unnormalized_widths = nn.Parameter(torch.zeros(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.zeros(*shape, num_bins))

            constant = np.log(np.exp(1 - min_derivative) - 1)
            num_derivatives = (
                (num_bins - 1) if self.tails == "linear" else (num_bins + 1)
            )
            self.unnormalized_derivatives = nn.Parameter(
                constant * torch.ones(*shape, num_derivatives)
            )
        else:
            self.unnormalized_widths = nn.Parameter(torch.rand(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.rand(*shape, num_bins))

            num_derivatives = (
                (num_bins - 1) if self.tails == "linear" else (num_bins + 1)
            )
            self.unnormalized_derivatives = nn.Parameter(
                torch.rand(*shape, num_derivatives)
            )

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths = _share_across_batch(self.unnormalized_widths, batch_size)
        unnormalized_heights = _share_across_batch(
            self.unnormalized_heights, batch_size
        )
        unnormalized_derivatives = _share_across_batch(
            self.unnormalized_derivatives, batch_size
        )

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            check_discriminant=self.check_discriminant,
            **spline_kwargs
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)
