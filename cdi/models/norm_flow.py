import nflows
import pytorch_lightning as pl
import torch
import torch.distributions as distr
import torch.nn.functional as F
from nflows import distributions, flows, transforms
from nflows.nn.nets import ResidualNet

from cdi.overrides.autoregressive import \
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from cdi.overrides.coupling import PiecewiseRationalQuadraticCouplingTransform
from cdi.util.arg_utils import parse_bool

"""
Adapted from https://github.com/bayesiains/nsf/
"""


def create_linear_transform(hparams):
    if hparams.linear_transform_type == 'permutation':
        return transforms.RandomPermutation(features=hparams.dim)
    elif hparams.linear_transform_type == 'lu':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=hparams.dim),
            transforms.LULinear(hparams.dim, identity_init=True)
        ])
    elif hparams.linear_transform_type == 'svd':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=hparams.dim),
            transforms.SVDLinear(hparams.dim, num_householder=10, identity_init=True)
        ])
    else:
        raise ValueError


def create_base_transform(hparams, i):
    if hparams.base_transform_type == 'affine-coupling':
        return transforms.AffineCouplingTransform(
            mask=nflows.utils.create_alternating_binary_mask(hparams.dim, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hparams.hidden_features,
                context_features=None,
                num_blocks=hparams.num_transform_blocks,
                activation=F.relu,
                dropout_probability=hparams.dropout_probability,
                use_batch_norm=hparams.use_batch_norm
            )
        )
    elif hparams.base_transform_type == 'rq-coupling':
        return PiecewiseRationalQuadraticCouplingTransform(
            mask=nflows.utils.create_alternating_binary_mask(hparams.dim, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hparams.hidden_features,
                context_features=None,
                num_blocks=hparams.num_transform_blocks,
                activation=F.relu,
                dropout_probability=hparams.dropout_probability,
                use_batch_norm=hparams.use_batch_norm
            ),
            num_bins=hparams.num_bins,
            tails='linear',
            tail_bound=hparams.tail_bound,
            apply_unconditional_transform=hparams.apply_unconditional_transform,
            check_discriminant=(hasattr(hparams, 'check_discriminant') and hparams.check_discriminant) or (not hasattr(hparams, 'check_discriminant'))
        )
    elif hparams.base_transform_type == 'affine-autoregressive':
        return transforms.MaskedAffineAutoregressiveTransform(
            features=hparams.dim,
            hidden_features=hparams.hidden_features,
            context_features=None,
            num_blocks=hparams.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=hparams.dropout_probability,
            use_batch_norm=hparams.use_batch_norm
        )
    elif hparams.base_transform_type == 'rq-autoregressive':
        return MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=hparams.dim,
            hidden_features=hparams.hidden_features,
            context_features=None,
            num_bins=hparams.num_bins,
            tails='linear',
            tail_bound=hparams.tail_bound,
            num_blocks=hparams.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=hparams.dropout_probability,
            use_batch_norm=hparams.use_batch_norm,
            check_discriminant=(hasattr(hparams, 'check_discriminant') and hparams.check_discriminant) or (not hasattr(hparams, 'check_discriminant'))
        )
    else:
        raise ValueError


def create_transform(hparams):
    transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            create_linear_transform(hparams),
            create_base_transform(hparams, i)
        ]) for i in range(hparams.num_flow_steps)
    ] + [
        create_linear_transform(hparams)
    ])
    return transform


class StandardStudentT(nflows.distributions.Distribution):
    """A independent Student's-T with zero loc and unit scale, and one degree of freedom."""

    def __init__(self, shape):
        super().__init__()
        self._shape = torch.Size(shape)

        self.studentt = distr.StudentT(df=1, loc=0, scale=1)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        log_probs = self.studentt.log_prob(inputs)
        return nflows.utils.torchutils.sum_except_batch(log_probs, num_batch_dims=1)

    def _sample(self, num_samples, context):
        if context is None:
            return self.studentt.sample(sample_shape=(num_samples, *self.shape_)).to(self.device)
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            return self.studentt.sample(sample_shape=(context_size, num_samples, *self._shape)).to(device=self.device)


class NormFlow(pl.LightningModule):
    def __init__(self, args):
        super(NormFlow, self).__init__()
        self.hparams = args.flow

        if not hasattr(self.hparams, 'base_distribution') or self.hparams.base_distribution == 'gaussian':
            distribution = distributions.StandardNormal((self.hparams.dim,))
        elif self.hparams.base_distribution == 'studentt':
            distribution = StandardStudentT((self.hparams.dim,))
        else:
            raise ValueError(f'base_distribution={self.hparams.base_distribution} is invalid.')

        transform = create_transform(self.hparams)
        self.flow = flows.Flow(transform, distribution)

        self.cum_batch_size_called = 0

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--flow.dim', type=int, required=True,
                            help='Dimensionality of the flow.')
        parser.add_argument('--flow.base_transform_type', type=str, default='rq-autoregressive',
                            choices=['affine-coupling', 'rq-coupling',
                                     'affine-autoregressive', 'rq-autoregressive'],
                            help='Type of transform to use between linear layers.')
        parser.add_argument('--flow.linear_transform_type', type=str, default='lu',
                            choices=['permutation', 'lu', 'svd'],
                            help='Type of linear transform to use.')
        parser.add_argument('--flow.num_flow_steps', type=int, default=10,
                            help='Number of blocks to use in flow.')
        parser.add_argument('--flow.hidden_features', type=int, default=256,
                            help='Number of hidden features to use in coupling/autoregressive nets.')
        parser.add_argument('--flow.tail_bound', type=float, default=3,
                            help='Box is on [-bound, bound]^2')
        parser.add_argument('--flow.num_bins', type=int, default=8,
                            help='Number of bins to use for piecewise transforms.')
        parser.add_argument('--flow.num_transform_blocks', type=int, default=2,
                            help='Number of blocks to use in coupling/autoregressive nets.')
        parser.add_argument('--flow.use_batch_norm', type=parse_bool, default=False,
                            help='Whether to use batch norm in coupling/autoregressive nets.')
        parser.add_argument('--flow.dropout_probability', type=float, default=0.25,
                            help='Dropout probability for coupling/autoregressive nets.')
        parser.add_argument('--flow.apply_unconditional_transform', type=parse_bool, default=True,
                            help='Whether to unconditionally transform \'identity\' '
                                 'features in coupling layer.')

        parser.add_argument('--flow.base_distribution', type=str,
                            choices=['gaussian', 'studentt'],
                            default='gaussian',
                            help='Base distribution type.')

        parser.add_argument('--flow.check_discriminant', type=parse_bool,
                            default=True, required=False,
                            help=('Whether to check the descriminant in rational-quadratic flow inverses. '
                                  'Disabling migh provide more stable MCMC sampling for PLMCMC.'))

        return parser

    def forward(self, X, M):
        # Track training calls
        if self.training:
            self.cum_batch_size_called += X.shape[0]

        return self.flow.log_prob(X)

    def log_prob(self, X):
        return self.flow.log_prob(X)

    def transform_to_noise_and_logabsdetJ(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.
        And gives the log-abs-determinant of the Jacobian of the transformation.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
            A `Tensor` of shape [batch_size, ...], the log-absolute-determinant of the Jacobian of the transformation.
        """
        return self.flow._transform(inputs, context=self.flow._embedding_net(context))

    def transform_from_noise_and_logabsdetJ(self, inputs, context=None):
        """Transforms given noise into data. Useful for goodness-of-fit checking.
        And gives the log-abs-determinant of the Jacobian of the transformation.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the samples.
            A `Tensor` of shape [batch_size, ...], the log-absolute-determinant of the Jacobian of the transformation.
        """
        return self.flow._transform.inverse(inputs, context=self.flow._embedding_net(context))

    def reset_parameters(self):
        # TODO
        pass

    def on_epoch_start(self):
        self.cum_batch_size_called = 0
