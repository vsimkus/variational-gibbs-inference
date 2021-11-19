import nflows
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as distr
import torch.nn as nn
import torch.nn.functional as F
from nflows.transforms import Transform
from nflows.nn.nets.resnet import ResidualBlock as nflows_ResidualBlock
import nflows.utils as nflows_utils
from torch.nn import init

from cdi.util.arg_utils import parse_bool
from cdi.layers.channel_linear import LinearWithChannels, LinearWithSharedChannels
from cdi.overrides.rational_quadratic import (
    DEFAULT_MIN_BIN_HEIGHT, DEFAULT_MIN_BIN_WIDTH, DEFAULT_MIN_DERIVATIVE,
    rational_quadratic_spline, unconstrained_rational_quadratic_spline)


class ElementwiseTransform(Transform):
    """Transforms each input variable with an invertible elementwise transformation.

    Based on the AutoregressiveTransform from:
    https://github.com/bayesiains/nflows/blob/HEAD/nflows/transforms/autoregressive.py#L24-L61
    """

    def __init__(self, net):
        super(ElementwiseTransform, self).__init__()
        self.net = net

    def forward(self, inputs, context, sparse_args):
        elementwise_params = self.net(context, sparse_args)
        outputs, logabsdet = self._elementwise_forward(inputs, elementwise_params)
        return outputs, logabsdet

    def inverse(self, inputs, context, sparse_args):
        elementwise_params = self.net(context, sparse_args)
        outputs, logabsdet = self._elementwise_inverse(inputs, elementwise_params)
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, inputs, elementwise_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, inputs, elementwise_params):
        raise NotImplementedError()


class ContextResidualNet(nflows.nn.nets.ResidualNet):
    """
    In the elementwise-transformation setting there are only context inputs.
    """
    def forward(self, context, sparse_args=None):
        temps = self.initial_layer(context)
        for block in self.blocks:
            temps = block(temps)
        outputs = self.final_layer(temps)
        return outputs


class ChannelResidualBlock(nn.Module):
    """A channel-wise residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        channel_dim,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
        reduce_mem=False,
        shared=False
    ):
        super().__init__()
        self.activation = activation
        self.reduce_mem = reduce_mem

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        Layer = LinearWithSharedChannels if shared else LinearWithChannels
        self.linear_layers = nn.ModuleList(
            [Layer(in_features=features,
                   out_features=features,
                   channels=channel_dim,
                   reduce_mem=reduce_mem,
                   mask=None)
             for _ in range(2)]
        )

        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, sparse_args):
        M_channel = None
        channel_sparse_idx = None
        if self.reduce_mem:
            M_channel, channel_sparse_idx = sparse_args

        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps, M_channel, channel_sparse_idx)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps, M_channel, channel_sparse_idx)

        return inputs + temps


class ContextChannelResidualNet(nn.Module):
    """A channel-wise residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        channel_dim,
        hidden_features,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        reduce_mem=False
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.reduce_mem = reduce_mem

        self.initial_layer = LinearWithChannels(in_features=in_features,
                                                out_features=hidden_features,
                                                channels=channel_dim,
                                                reduce_mem=reduce_mem)
        self.blocks = nn.ModuleList(
            [
                ChannelResidualBlock(
                    features=hidden_features,
                    channel_dim=channel_dim,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    reduce_mem=reduce_mem
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = LinearWithChannels(in_features=hidden_features,
                                              out_features=out_features,
                                              channels=channel_dim,
                                              reduce_mem=reduce_mem)

    def forward(self, context, sparse_args):
        M_channel = None
        channel_sparse_idx = None
        if self.reduce_mem:
            M_channel, channel_sparse_idx = sparse_args

        temps = self.initial_layer(context, M_channel, channel_sparse_idx)
        for block in self.blocks:
            temps = block(temps, sparse_args)
        outputs = self.final_layer(temps, M_channel, channel_sparse_idx)
        return outputs


class PiecewiseRationalQuadraticElementwiseTransform(ElementwiseTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        num_blocks=2,
        # use_residual_blocks=True,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
        reduce_mem=False,
        network='resnet'
    ):
        self.features = features
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        if network == 'resnet':
            net = ContextResidualNet(
                # We set the input features as the context
                in_features=context_features,
                out_features=features*self._output_dim_multiplier(),
                hidden_features=hidden_features,
                # No context.
                context_features=None,
                num_blocks=num_blocks,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            )
        elif network == 'channel-resnet':
            net = ContextChannelResidualNet(
                # We set the input features as the context
                in_features=context_features,
                out_features=self._output_dim_multiplier(),
                channel_dim=features,
                hidden_features=hidden_features,
                num_blocks=num_blocks,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
                reduce_mem=reduce_mem
            )

        super().__init__(net)

    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, elementwise_params, inverse=False):
        # batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = elementwise_params.view(
            -1, self.features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.net.hidden_features)
            unnormalized_heights /= np.sqrt(self.net.hidden_features)

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, logabsdet

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)


class ElementwiseLinearTransform(Transform):
    """Elementwise linear transforms."""

    def __init__(self, features):
        if not nflows_utils.is_positive_int(features):
            raise TypeError('Number of features must be a positive integer.')
        super().__init__()

        self.features = features
        self.bias = nn.Parameter(torch.zeros(features))
        self.log_weight = nn.Parameter(torch.zeros(features))

    def logabsdet(self):
        return self.log_weight

    def forward(self, inputs, context=None, sparse_args=None):
        outputs = inputs * torch.exp(self.log_weight) + self.bias
        logabsdet = self.logabsdet().unsqueeze(0).expand(inputs.shape[0], -1)

        return outputs, logabsdet

    def inverse(self, inputs, context=None, sparse_args=None):
        outputs = (inputs - self.bias) * torch.exp(-self.log_weight)
        logabsdet = -self.logabsdet().unsqueeze(0).expand(inputs.shape[0], -1)

        return outputs, logabsdet


class CompositeTransform(Transform):
    """Composes several transforms into one, in the order they are given."""

    def __init__(self, transforms, batch_dims=1):
        """Constructor.

        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__()
        self._transforms = nn.ModuleList(transforms)
        self.batch_dims = batch_dims

    @staticmethod
    def _cascade(inputs, funcs, context, sparse_args, *, batch_dims=1):
        batch_size = inputs.shape[:batch_dims]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(*batch_size)
        for func in funcs:
            outputs, logabsdet = func(outputs, context, sparse_args)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs, context=None, sparse_args=None):
        funcs = self._transforms
        return self._cascade(inputs, funcs, context, sparse_args, batch_dims=self.batch_dims)

    def inverse(self, inputs, context=None, sparse_args=None):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context, sparse_args, batch_dims=self.batch_dims)


class StandardNormal(nflows.distributions.Distribution):
    """A univariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super().__init__()
        self._shape = torch.Size(shape)

        self.normal = distr.Normal(0, 1)
        self.register_buffer("dummy",
                             torch.tensor(0.5, dtype=torch.float64),
                             persistent=False)

    def log_prob(self, inputs, context=None, sparse_args=None):
        return super().log_prob(inputs, context)

    def _log_prob(self, inputs, context, sparse_args=None):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        return self.normal.log_prob(inputs)

    def sample(self, num_samples, context=None, sparse_args=None, batch_size=None):
        return super().sample(num_samples, context, batch_size)

    def _sample(self, num_samples, context, sparse_args=None):
        if context is None:
            return torch.randn(num_samples, *self._shape, device=self.dummy.device)
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]

            return torch.randn(context_size, num_samples, *self._shape, device=self.dummy.device)

    def sample_and_log_prob(self, num_samples, context=None, sparse_args=None):
        return super().sample_and_log_prob(num_samples, context)


class ConditionalNormal(nflows.distributions.Distribution):
    """A univariate Normal whose parameters are functions of a context."""

    def __init__(self, shape, context_encoder=None, standardise_outliers=False, clamp_log_std=False):
        """Constructor.

        Args:
            shape: list, tuple or torch.Size, the shape of the input variables.
            context_encoder: callable or None, encodes the context to the distribution parameters.
                If None, defaults to the identity function.
            standardise_outliers: Boolean, if true, sets outliers to standard normal.
            clamp_log_std: Boolean, if True clamps the log of std to avoid excessive values.
        """
        super().__init__()
        self._shape = torch.Size(shape)
        self.clamp_log_std = clamp_log_std
        self.standardise_outliers = standardise_outliers
        if context_encoder is None:
            self._context_encoder = lambda x: x
        else:
            self._context_encoder = context_encoder

        assert not (standardise_outliers and clamp_log_std),\
            'Only one of standardise_outliers or clamp_log_std can be set true.'

    def _compute_params(self, context, sparse_args):
        """Compute the means and log stds form the context."""
        if context is None:
            raise ValueError("Context can't be None.")

        if isinstance(self._context_encoder, (LinearWithChannels, LinearWithSharedChannels)):
            M_channel = None
            channel_sparse_idx = None
            if self._context_encoder.reduce_mem:
                M_channel, channel_sparse_idx = sparse_args
            params = self._context_encoder(context, M_channel, channel_sparse_idx)
        elif isinstance(self._context_encoder, nn.Linear):
            params = self._context_encoder(context)

        if params.shape[-1] % 2 != 0:
            raise RuntimeError(
                "The context encoder must return a tensor whose last dimension is even."
            )
        if params.shape[0] != context.shape[0]:
            raise RuntimeError(
                "The batch dimension of the parameters is inconsistent with the input."
            )

        split = params.shape[-1] // 2
        means = params[..., :split].reshape(params.shape[0], *self._shape)
        log_stds = params[..., split:].reshape(params.shape[0], *self._shape)
        if self.standardise_outliers:
            mask = (log_stds < -10) | (log_stds > 10)
            log_stds[mask] = 0
            means[mask] = 0
        if self.clamp_log_std:
            log_stds = torch.clamp(log_stds, min=-10, max=10)
        return means, log_stds

    def log_prob(self, inputs, context=None, sparse_args=None):
        """Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """
        inputs = torch.as_tensor(inputs)
        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of context items."
                )
        return self._log_prob(inputs, context, sparse_args)

    def _log_prob(self, inputs, context, sparse_args):
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )

        # Compute parameters.
        means, log_stds = self._compute_params(context, sparse_args)
        assert means.shape == inputs.shape and log_stds.shape == inputs.shape

        # Compute log prob.
        normal = distr.Normal(loc=means, scale=torch.exp(log_stds))
        log_prob = normal.log_prob(inputs)
        return log_prob

    def sample(self, num_samples, context=None, sparse_args=None, batch_size=None):
        """Generates samples from the distribution. Samples can be generated in batches.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.

        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given.
        """
        if not nflows.utils.typechecks.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        if context is not None:
            context = torch.as_tensor(context)

        if batch_size is None:
            return self._sample(num_samples, context, sparse_args)

        else:
            if not nflows.utils.typechecks.is_positive_int(batch_size):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context, sparse_args) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context, sparse_args))
            return torch.cat(samples, dim=0)

    def _sample(self, num_samples, context, sparse_args):
        # Compute parameters.
        means, log_stds = self._compute_params(context, sparse_args)

        # Generate samples.
        normal = distr.Normal(loc=means, scale=torch.exp(log_stds))
        # shape=(B, K, *)
        return normal.rsample(sample_shape=(num_samples, )).permute(1, 0, *((-1,)*(len(means.shape)-1)))

    def sample_and_log_prob(self, num_samples, context=None, sparse_args=None):
        """Generates samples from the distribution together with their log probability.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.

        Returns:
            A tuple of:
                * A Tensor containing the samples, with shape [num_samples, ...] if context is None,
                  or [context_size, num_samples, ...] if context is given.
                * A Tensor containing the log probabilities of the samples, with shape
                  [num_samples, ...] if context is None, or [context_size, num_samples, ...] if
                  context is given.
        """
        samples = self.sample(num_samples, context=context, sparse_args=sparse_args)

        if context is not None:
            # Merge the context dimension with sample dimension in order to call log_prob.
            samples = nflows.utils.torchutils.merge_leading_dims(samples, num_dims=2)
            context = nflows.utils.torchutils.repeat_rows(context, num_reps=num_samples)
            assert samples.shape[0] == context.shape[0]

        log_prob = self.log_prob(samples, context=context, sparse_args=sparse_args)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = nflows.utils.torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            log_prob = nflows.utils.torchutils.split_leading_dim(log_prob, shape=[-1, num_samples])
        return samples, log_prob


class ConditionalStudentT(nflows.distributions.Distribution):
    """A univariate Student's-T whose parameters are functions of a context."""

    def __init__(self, shape, context_encoder=None):
        """Constructor.

        Args:
            shape: list, tuple or torch.Size, the shape of the input variables.
            context_encoder: callable or None, encodes the context to the distribution parameters.
                If None, defaults to the identity function.
        """
        super().__init__()
        self._shape = torch.Size(shape)
        if context_encoder is None:
            self._context_encoder = lambda x: x
        else:
            self._context_encoder = context_encoder

    def _compute_params(self, context, sparse_args):
        """Compute the locs and log-scales and log-degrees-of-freedom form the context."""
        if context is None:
            raise ValueError("Context can't be None.")

        if isinstance(self._context_encoder, (LinearWithChannels, LinearWithSharedChannels)):
            M_channel = None
            channel_sparse_idx = None
            if self._context_encoder.reduce_mem:
                M_channel, channel_sparse_idx = sparse_args
            params = self._context_encoder(context, M_channel, channel_sparse_idx)
        elif isinstance(self._context_encoder, nn.Linear):
            params = self._context_encoder(context)

        if params.shape[-1] % 3 != 0:
            raise RuntimeError(
                "The context encoder must return a tensor whose last dimension is divisable by 3."
            )
        if params.shape[0] != context.shape[0]:
            raise RuntimeError(
                "The batch dimension of the parameters is inconsistent with the input."
            )

        split = params.shape[-1] // 3
        locs = params[..., :split].reshape(params.shape[0], *self._shape)
        log_scales = params[..., split:split*2].reshape(params.shape[0], *self._shape)
        log_df = params[..., split*2:].reshape(params.shape[0], *self._shape)
        return locs, log_scales, log_df

    def log_prob(self, inputs, context=None, sparse_args=None):
        """Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """
        inputs = torch.as_tensor(inputs)
        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of context items."
                )
        return self._log_prob(inputs, context, sparse_args)

    def _log_prob(self, inputs, context, sparse_args):
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )

        # Compute parameters.
        locs, log_scales, log_df = self._compute_params(context, sparse_args)
        assert locs.shape == inputs.shape and log_scales.shape == inputs.shape and log_df.shape == inputs.shape

        # Compute log prob.
        studentt = distr.StudentT(df=torch.exp(log_df), loc=locs, scale=torch.exp(log_scales))
        log_prob = studentt.log_prob(inputs)
        return log_prob

    def sample(self, num_samples, context=None, sparse_args=None, batch_size=None):
        """Generates samples from the distribution. Samples can be generated in batches.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.

        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given.
        """
        if not nflows.utils.typechecks.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        if context is not None:
            context = torch.as_tensor(context)

        if batch_size is None:
            return self._sample(num_samples, context, sparse_args)

        else:
            if not nflows.utils.typechecks.is_positive_int(batch_size):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context, sparse_args) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context, sparse_args))
            return torch.cat(samples, dim=0)

    def _sample(self, num_samples, context, sparse_args):
        # Compute parameters.
        locs, log_scales, log_df = self._compute_params(context, sparse_args)

        # Generate samples.
        studentt = distr.StudentT(df=torch.exp(log_df), loc=locs, scale=torch.exp(log_scales))
        # shape=(B, K, *)
        return studentt.rsample(sample_shape=(num_samples, )).permute(1, 0, *((-1,)*(len(locs.shape)-1)))

    def sample_and_log_prob(self, num_samples, context=None, sparse_args=None):
        """Generates samples from the distribution together with their log probability.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.

        Returns:
            A tuple of:
                * A Tensor containing the samples, with shape [num_samples, ...] if context is None,
                  or [context_size, num_samples, ...] if context is given.
                * A Tensor containing the log probabilities of the samples, with shape
                  [num_samples, ...] if context is None, or [context_size, num_samples, ...] if
                  context is given.
        """
        samples = self.sample(num_samples, context=context, sparse_args=sparse_args)

        if context is not None:
            # Merge the context dimension with sample dimension in order to call log_prob.
            samples = nflows.utils.torchutils.merge_leading_dims(samples, num_dims=2)
            context = nflows.utils.torchutils.repeat_rows(context, num_reps=num_samples)
            assert samples.shape[0] == context.shape[0]

        log_prob = self.log_prob(samples, context=context, sparse_args=sparse_args)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = nflows.utils.torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            log_prob = nflows.utils.torchutils.split_leading_dim(log_prob, shape=[-1, num_samples])
        return samples, log_prob


class ResidualInputsEncoder(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 num_blocks,
                 out_features,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        super(ResidualInputsEncoder, self).__init__()
        self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                nflows_ResidualBlock(
                    features=hidden_features,
                    context_features=None,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    zero_initialization=False)
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, inputs, sparse_args=None):
        temps = self.initial_layer(inputs)
        for block in self.blocks:
            temps = block(temps)
        outputs = self.final_layer(temps)
        return outputs


class ChannelResidualInputsEncoder(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 channel_dim,
                 num_blocks,
                 out_features,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 reduce_mem=False,
                 mask_inputs=False):
        super(ChannelResidualInputsEncoder, self).__init__()
        self.reduce_mem = reduce_mem
        mask = None
        if mask_inputs:
            # Prepare input weight mask, zeros on diag, elsewhere 1
            mask = (torch.ones(channel_dim, channel_dim) - torch.eye(channel_dim, channel_dim))
            mask = mask.unsqueeze(-2)
        self.initial_layer = LinearWithSharedChannels(in_features,
                                                      hidden_features,
                                                      mask=mask,
                                                      channels=channel_dim,
                                                      reduce_mem=reduce_mem)
        self.blocks = nn.ModuleList(
            [
                ChannelResidualBlock(
                    features=hidden_features,
                    channel_dim=channel_dim,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    zero_initialization=False,
                    reduce_mem=reduce_mem,
                    shared=True)
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = LinearWithSharedChannels(hidden_features,
                                                    out_features,
                                                    mask=None,
                                                    channels=channel_dim,
                                                    reduce_mem=reduce_mem)

    def forward(self, inputs, sparse_args):
        M_channel = None
        channel_sparse_idx = None
        if self.reduce_mem:
            M_channel, channel_sparse_idx = sparse_args

        inputs = inputs.unsqueeze(1)
        temps = self.initial_layer(inputs, M_channel, channel_sparse_idx)
        for block in self.blocks:
            temps = block(temps, sparse_args)
        outputs = self.final_layer(temps, M_channel, channel_sparse_idx)
        return outputs


class Flow(nflows.flows.Flow):
    """Base class for all flow objects."""

    def _log_prob(self, inputs, context, sparse_args):
        if not isinstance(self._embedding_net, nn.Identity):
            embedded_context = self._embedding_net(context, sparse_args)
        else:
            embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context, sparse_args=sparse_args)
        log_prob = self._distribution.log_prob(noise, context=embedded_context, sparse_args=sparse_args)
        return log_prob + logabsdet

    def _sample(self, num_samples, context, sparse_args):
        if not isinstance(self._embedding_net, nn.Identity):
            embedded_context = self._embedding_net(context, sparse_args)
        else:
            embedded_context = self._embedding_net(context)
        noise = self._distribution.sample(num_samples, context=embedded_context, sparse_args=sparse_args)

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = nflows.utils.torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = nflows.utils.torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, _ = self._transform.inverse(noise, context=embedded_context, sparse_args=sparse_args)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = nflows.utils.torchutils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def log_prob(self, inputs, context=None, sparse_args=None):
        """Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """
        inputs = torch.as_tensor(inputs)
        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of context items."
                )
        return self._log_prob(inputs, context, sparse_args)

    def sample_and_log_prob(self, num_samples, context=None, sparse_args=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        if not isinstance(self._embedding_net, nn.Identity):
            embedded_context = self._embedding_net(context, sparse_args)
        else:
            embedded_context = self._embedding_net(context)
        noise, log_prob = self._distribution.sample_and_log_prob(
            num_samples, context=embedded_context, sparse_args=sparse_args
        )

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = nflows.utils.torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = nflows.utils.torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, logabsdet = self._transform.inverse(noise, context=embedded_context, sparse_args=sparse_args)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = nflows.utils.torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = nflows.utils.torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None, sparse_args=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        if not isinstance(self._embedding_net, nn.Identity):
            embedded_context = self._embedding_net(context, sparse_args)
        else:
            embedded_context = self._embedding_net(context)
        noise, _ = self._transform(inputs, context=embedded_context, sparse_args=sparse_args)
        return noise


class InverseTransform(nflows.transforms.InverseTransform):
    """Creates a transform that is the inverse of a given transform."""

    def forward(self, inputs, context=None, sparse_args=None):
        return self._transform.inverse(inputs, context, sparse_args)

    def inverse(self, inputs, context=None, sparse_args=None):
        return self._transform(inputs, context, sparse_args)


class PiecewiseRationalQuadraticVarDistribution(pl.LightningModule):
    def __init__(self, hparams):
        super(PiecewiseRationalQuadraticVarDistribution, self).__init__()

        self.hparams = hparams.var_model

        if self.hparams.activation == 'relu':
            activation = F.relu
        elif self.hparams.activation == 'lrelu':
            activation = F.leaky_relu
        elif self.hparams.activation == 'sigmoid':
            activation = F.sigmoid
        else:
            raise ValueError(f'activation={self.hparams.activation} is not implemented')

        if self.hparams.network == 'resnet':
            self.inputs_encoder = ResidualInputsEncoder(
                in_features=self.hparams.input_dim,
                hidden_features=self.hparams.encoder_hidden_features,
                num_blocks=self.hparams.encoder_num_blocks,
                out_features=self.hparams.context_intermediate_features,
                activation=activation,
                dropout_probability=self.hparams.dropout_probability,
                use_batch_norm=self.hparams.use_batch_norm
            )
            context_encoder = nn.Linear(self.hparams.context_intermediate_features, 2*self.hparams.input_dim)
        elif self.hparams.network == 'channel-resnet':
            self.inputs_encoder = ChannelResidualInputsEncoder(
                in_features=self.hparams.input_dim,
                hidden_features=self.hparams.encoder_hidden_features,
                channel_dim=self.hparams.input_dim,
                num_blocks=self.hparams.encoder_num_blocks,
                out_features=self.hparams.context_intermediate_features,
                activation=activation,
                dropout_probability=self.hparams.dropout_probability,
                use_batch_norm=self.hparams.use_batch_norm,
                reduce_mem=self.hparams.reduce_mem,
                mask_inputs=self.hparams.mask_inputs
            )
            context_encoder = LinearWithChannels(self.hparams.context_intermediate_features, 2,
                                                 channels=self.hparams.input_dim,
                                                 mask=None,
                                                 reduce_mem=self.hparams.reduce_mem)

        if not hasattr(self.hparams, 'base_distribution') or self.hparams.base_distribution == 'cond-gaussian':
            base_clamp_log_std = hasattr(self.hparams, 'base_clamp_log_std') and self.hparams.base_clamp_log_std
            base_standardise_outliers = hasattr(self.hparams, 'base_standardise_outliers') and self.hparams.base_standardise_outliers
            distribution = ConditionalNormal(
                shape=[self.hparams.input_dim],
                context_encoder=context_encoder,
                clamp_log_std=base_clamp_log_std,
                standardise_outliers=base_standardise_outliers
            )
        elif self.hparams.base_distribution == 'cond-studentt':
            context_encoder = nn.Linear(self.hparams.context_intermediate_features, 3*self.hparams.input_dim)
            distribution = ConditionalStudentT(
                shape=[self.hparams.input_dim],
                context_encoder=context_encoder
            )
        elif self.hparams.base_distribution == 'gaussian':
            distribution = StandardNormal((self.hparams.input_dim,))

        transform = CompositeTransform([
            CompositeTransform([
                ElementwiseLinearTransform(features=self.hparams.input_dim),
                PiecewiseRationalQuadraticElementwiseTransform(
                    features=self.hparams.input_dim,
                    hidden_features=self.hparams.hidden_features,
                    context_features=self.hparams.context_intermediate_features,
                    num_bins=self.hparams.num_bins,
                    tails='linear',
                    tail_bound=self.hparams.tail_bound,
                    num_blocks=self.hparams.num_transform_blocks,
                    # use_residual_blocks=True,
                    activation=activation,
                    dropout_probability=self.hparams.dropout_probability,
                    use_batch_norm=self.hparams.use_batch_norm,
                    reduce_mem=self.hparams.reduce_mem,
                    network=self.hparams.network
                )
            ], batch_dims=2)
            for _ in range(self.hparams.num_flow_steps)
        ], batch_dims=2)
        transform = CompositeTransform([
            transform,
            ElementwiseLinearTransform(features=self.hparams.input_dim)
        ], batch_dims=2)
        self.approximate_posterior = Flow(InverseTransform(transform), distribution)

        self.cum_batch_size_called = 0

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--var_model.input_dim',
                            type=int, required=True,
                            help='Dimensionality of input.')
        parser.add_argument('--var_model.encoder_hidden_features',
                            type=int, required=True,
                            help=('Dimensionality of input encoder hidden features.'))
        parser.add_argument('--var_model.encoder_num_blocks',
                            type=int, required=True,
                            help=('Number of resblocks in encoder.'))
        parser.add_argument('--var_model.context_intermediate_features',
                            type=int, required=True,
                            help=('Intermediate features.'))
        parser.add_argument('--var_model.activation',
                            type=str, required=True,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid', 'relu'])
        parser.add_argument('--var_model.dropout_probability',
                            type=float, default=0.0,
                            help=('Dropout probabiliy.'))
        parser.add_argument('--var_model.use_batch_norm', type=parse_bool, default=False,
                            help='Whether to use batch norm.')
        parser.add_argument('--var_model.hidden_features',
                            type=int, required=True,
                            help=('Dimensionality of transformation networks\' hidden features.'))
        parser.add_argument('--var_model.num_flow_steps', type=int, default=10,
                            help='Number of blocks to use in flow.')
        parser.add_argument('--var_model.tail_bound', type=float, default=3,
                            help='Box is on [-bound, bound]^2')
        parser.add_argument('--var_model.num_bins', type=int, default=8,
                            help='Number of bins to use for piecewise transforms.')
        parser.add_argument('--var_model.num_transform_blocks', type=int, default=2,
                            help='Number of blocks to use in coupling/autoregressive nets.')

        parser.add_argument('--var_model.base_distribution', type=str,
                            default='cond-gaussian',
                            choices=['cond-gaussian', 'cond-studentt', 'gaussian'],
                            help='Choice of base distribution.')
        parser.add_argument('--var_model.base_clamp_log_std', type=parse_bool,
                            default=False,
                            help=('Clamps the log_std of the base distribution to a reasonable range.'))
        parser.add_argument('--var_model.base_standardise_outliers', type=parse_bool,
                            default=False,
                            help=('Standardise distributions that fall outside of a reasonable range.'))

        parser.add_argument('--var_model.network',
                            type=str, default='resnet',
                            choices=['resnet', 'channel-resnet'],
                            help='Different choices for the parameter network.')
        parser.add_argument('--var_model.mask_inputs',
                            type=parse_bool, default=True,
                            help=('Whether to set the input to zero to enforse q(x_j | x_o, x_m\\j)'))
        parser.add_argument('--var_model.reduce_mem',
                            type=parse_bool, default=True,
                            help=('Whether should try to reduce memory usage.'))

        return parser

    def forward(self):
        pass

    def sample_and_log_prob(self, num_samples, context, M_selected):
        M_channel = None
        channel_sparse_idx = None
        if self.hparams.reduce_mem:
            M_channel = ~M_selected.T
            channel_sparse_idx = M_channel.nonzero(as_tuple=False)

        sparse_args = (M_channel, channel_sparse_idx)

        posterior_context = self.inputs_encoder(context, sparse_args)
        samples, log_prob = self.approximate_posterior.sample_and_log_prob(
            num_samples,
            context=posterior_context,
            sparse_args=sparse_args
        )

        return samples, log_prob

    def sample(self, num_samples, context, M_selected):
        M_channel = None
        channel_sparse_idx = None
        if self.hparams.reduce_mem:
            M_channel = ~M_selected.T
            channel_sparse_idx = M_channel.nonzero(as_tuple=False)

        sparse_args = (M_channel, channel_sparse_idx)

        posterior_context = self.inputs_encoder(context, sparse_args)
        samples = self.approximate_posterior._sample(
            num_samples,
            context=posterior_context,
            sparse_args=sparse_args
        )

        return samples

    def log_prob(self, inputs, context, M_selected):
        M_channel = None
        channel_sparse_idx = None
        if self.hparams.reduce_mem:
            M_channel = ~M_selected.T
            channel_sparse_idx = M_channel.nonzero(as_tuple=False)

        sparse_args = (M_channel, channel_sparse_idx)

        posterior_context = self.inputs_encoder(context, sparse_args)
        log_prob = self.approximate_posterior.log_prob(inputs, context=posterior_context, sparse_args=sparse_args)
        return log_prob

    def on_epoch_start(self):
        self.cum_batch_size_called = 0

    def reset_parameters(self):
        pass
