import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributions as distr

from cdi.layers.channel_linear import LinearWithChannels, LinearWithSharedChannels
from cdi.layers.masked_linear import MaskedLinear, MaskedLinearPerExample
from cdi.layers.sequential_with_multi_args import SequentialWithMultiInputs
from cdi.util.arg_utils import parse_bool


class UnivariateGaussianVarDistr(pl.LightningModule):
    """
    Univariate Gaussian variational distribution parametric model.
    Predicts Gaussian parameters (mean, log-variance) of d-th variable given
    the others.
    """
    def __init__(self, hparams):
        super(UnivariateGaussianVarDistr, self).__init__()
        self.hparams = hparams.var_model

        assert self.hparams.activation in ('lrelu', 'sigmoid', 'relu'), \
            'Activation not supported!'

        hidden_dims = [self.hparams.input_dim] + self.hparams.hidden_dims

        mean_layers = []
        log_var_layers = []
        for i in range(len(hidden_dims)-1):
            mean_layers.append(nn.Linear(in_features=hidden_dims[i],
                                         out_features=hidden_dims[i+1]))
            log_var_layers.append(nn.Linear(in_features=hidden_dims[i],
                                            out_features=hidden_dims[i+1]))

            if self.hparams.activation == 'sigmoid':
                mean_layers.append(nn.Sigmoid())
                log_var_layers.append(nn.Sigmoid())
            elif self.hparams.activation == 'lrelu':
                mean_layers.append(nn.LeakyReLU())
                log_var_layers.append(nn.LeakyReLU())
            elif self.hparams.activation == 'relu':
                mean_layers.append(nn.ReLU())
                log_var_layers.append(nn.ReLU())

        # Add final output layers
        mean_layers.append(nn.Linear(in_features=hidden_dims[-1],
                                     out_features=1))
        log_var_layers.append(nn.Linear(in_features=hidden_dims[-1],
                                        out_features=1))

        self.mean = nn.Sequential(*mean_layers)
        self.log_var = nn.Sequential(*log_var_layers)

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--var_model.input_dim',
                            type=int, required=True,
                            help='Dimensionality of input.')
        parser.add_argument('--var_model.hidden_dims',
                            type=int, nargs='+', required=True,
                            help='Dimensionalities of hidden layers.')
        parser.add_argument('--var_model.activation',
                            type=str, required=True,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid', 'relu'])
        return parser

    def forward(self, X):
        """
        Compute Gaussian mean and log-variance of the missing variable given
        the observed X.
        Args:
            X (N, D-1): observed variables (without the missing variable)
        Returns:
            mean (float): Gaussian mean
            log_var (float): Gaussian log-variance
        """
        mean = self.mean(X)
        log_var = self.log_var(X)

        return mean, log_var

    def reset_parameters(self):
        for layer in self.mean:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()

        for layer in self.log_var:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()


class GaussianVarDistr(pl.LightningModule):
    """
    Gaussian variational distribution parameter model.
    Wraps the univariate conditional distributions for each variable
    given all other variables.
    """
    def __init__(self, hparams):
        super(GaussianVarDistr, self).__init__()
        self.hparams = hparams.var_model

        self.init()

        self.cum_batch_size_called = 0

    def init(self):
        # Create variational distribution for each visible variable
        self.distr_dict = nn.ModuleDict()
        for i in range(0, self.hparams.num_univariate_q):
            self.distr_dict['distr_{}'.format(i)] = \
                UnivariateGaussianVarDistr(self.hparams)

    @staticmethod
    def add_model_args(parent_parser):
        parser = UnivariateGaussianVarDistr.add_model_args(parent_parser)
        parser.add_argument('--var_model.input_missing_vectors',
                            type=parse_bool, default=False,
                            help=('Whether the binary missing vectors '
                                  'should be passed to the variational '
                                  'models.'))
        parser.add_argument('--var_model.num_univariate_q',
                            type=int, required=True,
                            help=('Number of univariate distributions.'))
        return parser

    def forward(self, X, M, M_selected, sample_all=None):
        """
        Compute Gaussian mean and log-variance of the missing variables given
        the observed X.
        Args:
            X (N, D): observable variables (with the missing variable)
            M (N, D): the missingness boolean mask. 1 - observed, 0 - missing.
            M_selected (N, D): a subset of M, for which we want to compute the
                variational distribution parameters.
        Return:
            means (float): variational Gaussian mean of the missing variable
                for each sample and each dimension, 0 if the value is not
                missing at the current sample & dimension.
            log_vars (float): variational Gaussian log-variance of the missing
                variable for each sample and dimension, -inf if the value is
                not missing at the current sample & dimension. So that the
                variance is 0 (exp(-inf)=0).
        """
        # Set mean to 0 and log_var to -inf,
        # since this corresponds to 0 variance.
        # Thus for observed variables, the sampled values will be 0.
        means = torch.zeros_like(X)
        log_vars = torch.full_like(X, float('-inf'))

        if self.hparams.input_missing_vectors:
            M = M.type_as(X)

        # Compute variational distribution parameters
        M_selected_not = ~M_selected
        for d in range(self.hparams.num_univariate_q):
            # Find all samples which have a missing variable at d-th dimension
            X_indices = M_selected_not[:, d].nonzero(as_tuple=False).squeeze(-1)
            if X_indices.shape[0] == 0:
                continue

            # Select samples from X where d-th feature is missing
            X_d = X[X_indices, :]
            # Remove d-th feature from X
            feature_mask = torch.arange(X.shape[-1])
            feature_mask = feature_mask[feature_mask != d]
            X_d = X_d[:, feature_mask]

            if self.hparams.input_missing_vectors:
                # M_d = M[X_indices[:, None], feature_mask]
                X_d = torch.cat((X_d, M[X_indices, :]), dim=1)

            # Compute Gaussian parameters
            mean, log_var = self.distr_dict['distr_{}'.format(d)](X_d)

            means[X_indices, d] = mean.squeeze()
            log_vars[X_indices, d] = log_var.squeeze()

            # Track training calls
            if self.training:
                self.cum_batch_size_called += X_d.shape[0]

        return means, log_vars

    def reset_parameters(self):
        for i in range(self.hparams.num_univariate_q):
            self.distr_dict['distr_{}'.format(i)].reset_parameters()

    def on_epoch_start(self):
        self.cum_batch_size_called = 0


class GaussianVarDistrFast(pl.LightningModule):
    """
    Gaussian variational distribution parameter model.
    Fast individual networks.
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams.var_model

        self.init()

        self.cum_batch_size_called = 0

    def init(self):
        assert self.hparams.activation in ('lrelu', 'sigmoid', 'relu'), \
            'Activation not supported!'

        # Input dimensionality is the #vars.
        if self.hparams.input_missing_vectors:
            # *2 if missingness mask is passed to the models
            input_dim = self.hparams.input_dim*2
        else:
            input_dim = self.hparams.input_dim

        # Add input and output dimensions to the list
        hidden_dims = [input_dim] + self.hparams.hidden_dims

        mean_layers = []
        log_var_layers = []
        for i in range(len(hidden_dims)-1):
            mask = None
            if i == 0:
                # Prepare input weight mask, zeros on diag, elsewhere 1
                mask = (torch.ones(self.hparams.input_dim, self.hparams.input_dim)
                        - torch.eye(self.hparams.input_dim, self.hparams.input_dim))
                if self.hparams.input_missing_vectors:
                    # No masking for miss-mask
                    mask = torch.cat([mask, torch.ones(self.hparams.input_dim, self.hparams.input_dim)], dim=-1)
                mask = mask.unsqueeze(-2)

            mean_layers.append(LinearWithChannels(in_features=hidden_dims[i],
                                                  out_features=hidden_dims[i+1],
                                                  channels=self.hparams.input_dim,
                                                  mask=mask))
            log_var_layers.append(LinearWithChannels(in_features=hidden_dims[i],
                                                     out_features=hidden_dims[i+1],
                                                     channels=self.hparams.input_dim,
                                                     mask=mask))

            if self.hparams.activation == 'sigmoid':
                mean_layers.append(nn.Sigmoid())
                log_var_layers.append(nn.Sigmoid())
            elif self.hparams.activation == 'lrelu':
                mean_layers.append(nn.LeakyReLU())
                log_var_layers.append(nn.LeakyReLU())
            elif self.hparams.activation == 'relu':
                mean_layers.append(nn.ReLU())
                log_var_layers.append(nn.ReLU())

        # Add final output layers
        mean_layers.append(LinearWithChannels(in_features=hidden_dims[-1],
                                              out_features=1,
                                              channels=self.hparams.input_dim))
        log_var_layers.append(LinearWithChannels(in_features=hidden_dims[-1],
                                                 out_features=1,
                                                 channels=self.hparams.input_dim))

        self.mean = SequentialWithMultiInputs(*mean_layers)
        self.log_var = SequentialWithMultiInputs(*log_var_layers)

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--var_model.input_dim',
                            type=int, required=True,
                            help='Dimensionality of input.')
        parser.add_argument('--var_model.hidden_dims',
                            type=int, nargs='+', required=True,
                            help=('Dimensionalities of layers.'))
        parser.add_argument('--var_model.activation',
                            type=str, required=True,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid', 'relu'])
        parser.add_argument('--var_model.input_missing_vectors',
                            type=parse_bool, default=False,
                            help=('Whether the binary missing vectors '
                                  'should be passed to the variational '
                                  'models.'))
        return parser

    def forward(self, X, M, M_selected, sample_all=None):
        if self.hparams.input_missing_vectors:
            X = torch.cat((X, M), dim=-1)

        M_channel = ~M_selected.T
        channel_sparse_idx = M_channel.nonzero(as_tuple=False)

        means = self.mean(X.unsqueeze(1), M_channel, channel_sparse_idx).squeeze(2)
        log_vars = self.log_var(X.unsqueeze(1), M_channel, channel_sparse_idx).squeeze(2)

        # Track training calls
        if self.training:
            self.cum_batch_size_called += X.shape[0]

        return means, log_vars

    def reset_parameters(self):
        for layer in self.mean:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()

        for layer in self.log_var:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()

    def on_epoch_start(self):
        self.cum_batch_size_called = 0


class SharedGaussianVarDistr(pl.LightningModule):
    """
    Gaussian variational distribution parameter model.
    Fast individual networks.
    Part-shared across distributions.
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams.var_model

        self.init()

        self.cum_batch_size_called = 0

    def init(self):
        assert self.hparams.activation in ('lrelu', 'sigmoid', 'relu'), \
            'Activation not supported!'

        # Input dimensionality is the #vars.
        if self.hparams.input_missing_vectors:
            # *2 if missingness mask is passed to the models
            input_dim = self.hparams.input_dim*2
        else:
            input_dim = self.hparams.input_dim

        # Add input and output dimensions to the list
        hidden_dims = [input_dim] + self.hparams.body_hidden_dims + self.hparams.head_hidden_dims + [1]

        mean_layers = []
        log_var_layers = []
        reduce_mem = self.hparams.reduce_mem if hasattr(self.hparams, 'reduce_mem') else True
        for i in range(len(hidden_dims)-1):
            mask = None
            if i == 0 and (not hasattr(self.hparams, 'mask_inputs') or self.hparams.mask_inputs):
                # Prepare input weight mask, zeros on diag, elsewhere 1
                mask = (torch.ones(self.hparams.input_dim, self.hparams.input_dim)
                        - torch.eye(self.hparams.input_dim, self.hparams.input_dim))
                if self.hparams.input_missing_vectors:
                    # No masking for miss-mask
                    mask = torch.cat([mask, torch.ones(self.hparams.input_dim, self.hparams.input_dim)], dim=-1)
                mask = mask.unsqueeze(-2)

            if i < len(self.hparams.body_hidden_dims):
                Layer = LinearWithSharedChannels
            else:
                Layer = LinearWithChannels

            mean_layers.append(Layer(in_features=hidden_dims[i],
                                     out_features=hidden_dims[i+1],
                                     channels=self.hparams.input_dim,
                                     mask=mask,
                                     reduce_mem=reduce_mem))
            log_var_layers.append(Layer(in_features=hidden_dims[i],
                                        out_features=hidden_dims[i+1],
                                        channels=self.hparams.input_dim,
                                        mask=mask,
                                        reduce_mem=reduce_mem))

            if i != len(hidden_dims)-2:
                if self.hparams.activation == 'sigmoid':
                    mean_layers.append(nn.Sigmoid())
                    log_var_layers.append(nn.Sigmoid())
                elif self.hparams.activation == 'lrelu':
                    mean_layers.append(nn.LeakyReLU())
                    log_var_layers.append(nn.LeakyReLU())
                elif self.hparams.activation == 'relu':
                    mean_layers.append(nn.ReLU())
                    log_var_layers.append(nn.ReLU())

        self.mean = SequentialWithMultiInputs(*mean_layers)
        self.log_var = SequentialWithMultiInputs(*log_var_layers)

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--var_model.input_dim',
                            type=int, required=True,
                            help='Dimensionality of input.')
        parser.add_argument('--var_model.body_hidden_dims',
                            type=int, nargs='+', required=True,
                            help=('Dimensionalities of shared '
                                  'hidden layers.'))
        parser.add_argument('--var_model.head_hidden_dims',
                            type=int, nargs='+', required=True,
                            help=('Dimensionalities of hidden layers '
                                  'of head networks.'))
        parser.add_argument('--var_model.activation',
                            type=str, required=True,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid', 'relu'])
        parser.add_argument('--var_model.input_missing_vectors',
                            type=parse_bool, default=False,
                            help=('Whether the binary missing vectors '
                                  'should be passed to the variational '
                                  'models.'))
        parser.add_argument('--var_model.mask_inputs',
                            type=parse_bool, default=True,
                            help=('Whether to set the input to zero to enforse q(x_j | x_o, x_m\\j)'))
        parser.add_argument('--var_model.reduce_mem',
                            type=parse_bool, default=True,
                            help=('Whether should try to reduce memory usage.'))
        return parser

    def forward(self, X, M, M_selected, sample_all=None):
        if self.hparams.input_missing_vectors:
            X = torch.cat((X, M), dim=-1)

        reduce_mem = self.hparams.reduce_mem if hasattr(self.hparams, 'reduce_mem') else True

        M_channel = None
        channel_sparse_idx = None
        if reduce_mem:
            M_channel = ~M_selected.T
            channel_sparse_idx = M_channel.nonzero(as_tuple=False)

        means = self.mean(X.unsqueeze(1), M_channel, channel_sparse_idx).squeeze(2)
        log_vars = self.log_var(X.unsqueeze(1), M_channel, channel_sparse_idx).squeeze(2)

        # Track training calls
        if self.training:
            self.cum_batch_size_called += X.shape[0]

        return means, log_vars

    def reset_parameters(self):
        for layer in self.mean:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()

        for layer in self.log_var:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()

    def on_epoch_start(self):
        self.cum_batch_size_called = 0


class SharedMixGaussianVarDistr(pl.LightningModule):
    """
    MoG variational distribution parameter model.
    Fast individual networks.
    Part-shared across distributions.
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams.var_model

        self.init()

        self.cum_batch_size_called = 0

    def init(self):
        assert self.hparams.activation in ('lrelu', 'sigmoid', 'relu'), \
            'Activation not supported!'

        # Input dimensionality is the #vars.
        if self.hparams.input_missing_vectors:
            # *2 if missingness mask is passed to the models
            input_dim = self.hparams.input_dim*2
        else:
            input_dim = self.hparams.input_dim

        # Add input and output dimensions to the list
        hidden_dims = [input_dim] + self.hparams.body_hidden_dims + self.hparams.head_hidden_dims + [self.hparams.num_components]

        mean_layers = []
        log_var_layers = []
        comp_logit_layers = []
        reduce_mem = self.hparams.reduce_mem if hasattr(self.hparams, 'reduce_mem') else True
        for i in range(len(hidden_dims)-1):
            mask = None
            if i == 0 and (not hasattr(self.hparams, 'mask_inputs') or self.hparams.mask_inputs):
                # Prepare input weight mask, zeros on diag, elsewhere 1
                mask = (torch.ones(self.hparams.input_dim, self.hparams.input_dim)
                        - torch.eye(self.hparams.input_dim, self.hparams.input_dim))
                if self.hparams.input_missing_vectors:
                    # No masking for miss-mask
                    mask = torch.cat([mask, torch.ones(self.hparams.input_dim, self.hparams.input_dim)], dim=-1)
                mask = mask.unsqueeze(-2)

            if i < len(self.hparams.body_hidden_dims):
                Layer = LinearWithSharedChannels
            else:
                Layer = LinearWithChannels

            mean_layers.append(Layer(in_features=hidden_dims[i],
                                     out_features=hidden_dims[i+1],
                                     channels=self.hparams.input_dim,
                                     mask=mask,
                                     reduce_mem=reduce_mem))
            log_var_layers.append(Layer(in_features=hidden_dims[i],
                                        out_features=hidden_dims[i+1],
                                        channels=self.hparams.input_dim,
                                        mask=mask,
                                        reduce_mem=reduce_mem))
            comp_logit_layers.append(Layer(in_features=hidden_dims[i],
                                           out_features=hidden_dims[i+1],
                                           channels=self.hparams.input_dim,
                                           mask=mask,
                                           reduce_mem=reduce_mem))

            if i != len(hidden_dims)-2:
                if self.hparams.activation == 'sigmoid':
                    mean_layers.append(nn.Sigmoid())
                    log_var_layers.append(nn.Sigmoid())
                    comp_logit_layers.append(nn.Sigmoid())
                elif self.hparams.activation == 'lrelu':
                    mean_layers.append(nn.LeakyReLU())
                    log_var_layers.append(nn.LeakyReLU())
                    comp_logit_layers.append(nn.LeakyReLU())
                elif self.hparams.activation == 'relu':
                    mean_layers.append(nn.ReLU())
                    log_var_layers.append(nn.ReLU())
                    comp_logit_layers.append(nn.ReLU())

        self.mean = SequentialWithMultiInputs(*mean_layers)
        self.log_var = SequentialWithMultiInputs(*log_var_layers)
        self.comp_logit = SequentialWithMultiInputs(*comp_logit_layers)

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--var_model.input_dim',
                            type=int, required=True,
                            help='Dimensionality of input.')
        parser.add_argument('--var_model.num_components', type=int,
                            required=True, help=('Number of components in the mixture.'))
        parser.add_argument('--var_model.body_hidden_dims',
                            type=int, nargs='+', required=True,
                            help=('Dimensionalities of shared '
                                  'hidden layers.'))
        parser.add_argument('--var_model.head_hidden_dims',
                            type=int, nargs='+', required=True,
                            help=('Dimensionalities of hidden layers '
                                  'of head networks.'))
        parser.add_argument('--var_model.activation',
                            type=str, required=True,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid', 'relu'])
        parser.add_argument('--var_model.input_missing_vectors',
                            type=parse_bool, default=False,
                            help=('Whether the binary missing vectors '
                                  'should be passed to the variational '
                                  'models.'))
        parser.add_argument('--var_model.mask_inputs',
                            type=parse_bool, default=True,
                            help=('Whether to set the input to zero to enforse q(x_j | x_o, x_m\\j)'))
        parser.add_argument('--var_model.reduce_mem',
                            type=parse_bool, default=True,
                            help=('Whether should try to reduce memory usage.'))

        parser.add_argument('--var_model.clamp_log_std', type=parse_bool,
                            default=False,
                            help=('Clamps the log_std to reasonable values.'))

        return parser

    def forward(self, X, M, M_selected, sample_all=None):
        if self.hparams.input_missing_vectors:
            X = torch.cat((X, M), dim=-1)

        reduce_mem = self.hparams.reduce_mem if hasattr(self.hparams, 'reduce_mem') else True

        M_channel = None
        channel_sparse_idx = None
        if reduce_mem:
            M_channel = ~M_selected.T
            channel_sparse_idx = M_channel.nonzero(as_tuple=False)

        means = self.mean(X.unsqueeze(1), M_channel, channel_sparse_idx)
        log_vars = self.log_var(X.unsqueeze(1), M_channel, channel_sparse_idx)
        comp_logits = self.comp_logit(X.unsqueeze(1), M_channel, channel_sparse_idx)

        if hasattr(self.hparams, 'clamp_log_std') and self.hparams.clamp_log_std:
            log_vars = torch.clamp(log_vars, -10*2, 10*2)

        # Track training calls
        if self.training:
            self.cum_batch_size_called += X.shape[0]

        return means, log_vars, comp_logits

    def reset_parameters(self):
        for layer in self.mean:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()

        for layer in self.log_var:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()

        for layer in self.comp_logit:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()

    def on_epoch_start(self):
        self.cum_batch_size_called = 0


class SharedGaussianVarDistrPartSharedMeanVar(pl.LightningModule):
    """
    Gaussian variational distribution parameter model.
    Fast individual networks.
    Part-shared across distributions.
    Also, part-shared the body is shared for mean and log-var computation.
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams.var_model

        self.init()

        self.cum_batch_size_called = 0

    def init(self):
        assert self.hparams.activation in ('lrelu', 'sigmoid', 'relu'), \
            'Activation not supported!'

        # Input dimensionality is the #vars.
        if self.hparams.input_missing_vectors:
            # *2 if missingness mask is passed to the models
            input_dim = self.hparams.input_dim*2
        else:
            input_dim = self.hparams.input_dim

        # Add input and output dimensions to the list
        body_hidden_dims = [input_dim] + self.hparams.body_hidden_dims
        heads_hidden_dims = [body_hidden_dims[-1]] + self.hparams.head_hidden_dims + [1]

        body_layers = []
        reduce_mem = self.hparams.reduce_mem if hasattr(self.hparams, 'reduce_mem') else True
        for i in range(len(body_hidden_dims)-1):
            mask = None
            if i == 0 and (not hasattr(self.hparams, 'mask_inputs') or self.hparams.mask_inputs):
                # Prepare input weight mask, zeros on diag, elsewhere 1
                mask = (torch.ones(self.hparams.input_dim, self.hparams.input_dim)
                        - torch.eye(self.hparams.input_dim, self.hparams.input_dim))
                if self.hparams.input_missing_vectors:
                    # No masking for miss-mask
                    mask = torch.cat([mask, torch.ones(self.hparams.input_dim, self.hparams.input_dim)], dim=-1)
                mask = mask.unsqueeze(-2)

            body_layers.append(LinearWithSharedChannels(in_features=body_hidden_dims[i],
                                                        out_features=body_hidden_dims[i+1],
                                                        channels=self.hparams.input_dim,
                                                        mask=mask,
                                                        reduce_mem=reduce_mem))

            if self.hparams.activation == 'sigmoid':
                body_layers.append(nn.Sigmoid())
            elif self.hparams.activation == 'lrelu':
                body_layers.append(nn.LeakyReLU())
            elif self.hparams.activation == 'relu':
                body_layers.append(nn.ReLU())

        mean_layers = []
        log_var_layers = []
        for i in range(len(heads_hidden_dims)-1):
            mask = None
            if (i == 0
                    and (not hasattr(self.hparams, 'mask_inputs') or self.hparams.mask_inputs)
                    and len(body_layers) == 0):
                # Prepare input weight mask, zeros on diag, elsewhere 1
                mask = (torch.ones(self.hparams.input_dim, self.hparams.input_dim)
                        - torch.eye(self.hparams.input_dim, self.hparams.input_dim))
                if self.hparams.input_missing_vectors:
                    # No masking for miss-mask
                    mask = torch.cat([mask, torch.ones(self.hparams.input_dim, self.hparams.input_dim)], dim=-1)
                mask = mask.unsqueeze(-2)

            Layer = LinearWithChannels
            mean_layers.append(Layer(in_features=heads_hidden_dims[i],
                                     out_features=heads_hidden_dims[i+1],
                                     channels=self.hparams.input_dim,
                                     mask=mask,
                                     reduce_mem=reduce_mem))
            log_var_layers.append(Layer(in_features=heads_hidden_dims[i],
                                        out_features=heads_hidden_dims[i+1],
                                        channels=self.hparams.input_dim,
                                        mask=mask,
                                        reduce_mem=reduce_mem))

            if i != len(heads_hidden_dims)-2:
                if self.hparams.activation == 'sigmoid':
                    mean_layers.append(nn.Sigmoid())
                    log_var_layers.append(nn.Sigmoid())
                elif self.hparams.activation == 'lrelu':
                    mean_layers.append(nn.LeakyReLU())
                    log_var_layers.append(nn.LeakyReLU())
                elif self.hparams.activation == 'relu':
                    mean_layers.append(nn.ReLU())
                    log_var_layers.append(nn.ReLU())

        self.body = SequentialWithMultiInputs(*body_layers)
        self.mean = SequentialWithMultiInputs(*mean_layers)
        self.log_var = SequentialWithMultiInputs(*log_var_layers)

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--var_model.input_dim',
                            type=int, required=True,
                            help='Dimensionality of input.')
        parser.add_argument('--var_model.body_hidden_dims',
                            type=int, nargs='+', required=True,
                            help=('Dimensionalities of shared '
                                  'hidden layers.'))
        parser.add_argument('--var_model.head_hidden_dims',
                            type=int, nargs='+', required=True,
                            help=('Dimensionalities of hidden layers '
                                  'of head networks.'))
        parser.add_argument('--var_model.activation',
                            type=str, required=True,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid', 'relu'])
        parser.add_argument('--var_model.input_missing_vectors',
                            type=parse_bool, default=False,
                            help=('Whether the binary missing vectors '
                                  'should be passed to the variational '
                                  'models.'))
        parser.add_argument('--var_model.mask_inputs',
                            type=parse_bool, default=True,
                            help=('Whether to set the input to zero to enforse q(x_j | x_o, x_m\\j)'))
        parser.add_argument('--var_model.reduce_mem',
                            type=parse_bool, default=True,
                            help=('Whether should try to reduce memory usage.'))
        return parser

    def forward(self, X, M, M_selected, sample_all=None):
        if self.hparams.input_missing_vectors:
            X = torch.cat((X, M), dim=-1)

        reduce_mem = self.hparams.reduce_mem if hasattr(self.hparams, 'reduce_mem') else True
        M_channel = None
        channel_sparse_idx = None
        if reduce_mem:
            M_channel = ~M_selected.T
            channel_sparse_idx = M_channel.nonzero(as_tuple=False)

        intermediate = self.body(X.unsqueeze(1), M_channel, channel_sparse_idx)
        means = self.mean(intermediate, M_channel, channel_sparse_idx).squeeze(2)
        log_vars = self.log_var(intermediate, M_channel, channel_sparse_idx).squeeze(2)

        # Track training calls
        if self.training:
            self.cum_batch_size_called += X.shape[0]

        return means, log_vars

    def reset_parameters(self):
        for layer in self.body:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()

        for layer in self.mean:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()

        for layer in self.log_var:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()

    def on_epoch_start(self):
        self.cum_batch_size_called = 0


class SharedGaussianVarDistrSharedMeanVar(pl.LightningModule):
    """
    Gaussian variational distribution parameter model.
    Parallelised individual networks.
    Part-shared across distributions.
    Same network for mean and log-var.
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams.var_model

        self.init()

        self.cum_batch_size_called = 0

    def init(self):
        assert self.hparams.activation in ('lrelu', 'sigmoid', 'relu'), \
            'Activation not supported!'

        # Input dimensionality is the #vars.
        if self.hparams.input_missing_vectors:
            # *2 if missingness mask is passed to the models
            input_dim = self.hparams.input_dim*2
        else:
            input_dim = self.hparams.input_dim

        # Add input and output dimensions to the list
        # 2 outputs: mean and log-var
        hidden_dims = [input_dim] + self.hparams.body_hidden_dims + self.hparams.head_hidden_dims + [2]

        hidden_layers = []
        reduce_mem = self.hparams.reduce_mem if hasattr(self.hparams, 'reduce_mem') else True
        for i in range(len(hidden_dims)-1):
            mask = None
            if i == 0 and (not hasattr(self.hparams, 'mask_inputs') or self.hparams.mask_inputs):
                # Prepare input weight mask, zeros on diag, elsewhere 1
                mask = (torch.ones(self.hparams.input_dim, self.hparams.input_dim)
                        - torch.eye(self.hparams.input_dim, self.hparams.input_dim))
                if self.hparams.input_missing_vectors:
                    # No masking for miss-mask
                    mask = torch.cat([mask, torch.ones(self.hparams.input_dim, self.hparams.input_dim)], dim=-1)
                mask = mask.unsqueeze(-2)

            if i < len(self.hparams.body_hidden_dims):
                Layer = LinearWithSharedChannels
            else:
                Layer = LinearWithChannels

            hidden_layers.append(Layer(in_features=hidden_dims[i],
                                       out_features=hidden_dims[i+1],
                                       channels=self.hparams.input_dim,
                                       mask=mask,
                                       reduce_mem=reduce_mem))

            if i != len(hidden_dims)-2:
                if self.hparams.activation == 'sigmoid':
                    hidden_layers.append(nn.Sigmoid())
                elif self.hparams.activation == 'lrelu':
                    hidden_layers.append(nn.LeakyReLU())
                elif self.hparams.activation == 'relu':
                    hidden_layers.append(nn.ReLU())

        self.net = SequentialWithMultiInputs(*hidden_layers)

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--var_model.input_dim',
                            type=int, required=True,
                            help='Dimensionality of input.')
        parser.add_argument('--var_model.body_hidden_dims',
                            type=int, nargs='+', required=True,
                            help=('Dimensionalities of shared '
                                  'hidden layers.'))
        parser.add_argument('--var_model.head_hidden_dims',
                            type=int, nargs='+', required=True,
                            help=('Dimensionalities of hidden layers '
                                  'of head networks.'))
        parser.add_argument('--var_model.activation',
                            type=str, required=True,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid', 'relu'])
        parser.add_argument('--var_model.input_missing_vectors',
                            type=parse_bool, default=False,
                            help=('Whether the binary missing vectors '
                                  'should be passed to the variational '
                                  'models.'))
        parser.add_argument('--var_model.mask_inputs',
                            type=parse_bool, default=True,
                            help=('Whether to set the input to zero to enforse q(x_j | x_o, x_m\\j)'))
        parser.add_argument('--var_model.reduce_mem',
                            type=parse_bool, default=True,
                            help=('Whether should try to reduce memory usage.'))
        return parser

    def forward(self, X, M, M_selected, sample_all=None):
        if self.hparams.input_missing_vectors:
            X = torch.cat((X, M), dim=-1)

        reduce_mem = self.hparams.reduce_mem if hasattr(self.hparams, 'reduce_mem') else True
        M_channel = None
        channel_sparse_idx = None
        if reduce_mem:
            M_channel = ~M_selected.T
            channel_sparse_idx = M_channel.nonzero(as_tuple=False)

        outputs = self.net(X.unsqueeze(1), M_channel, channel_sparse_idx)
        means = outputs[:, :, 0]
        log_vars = outputs[:, :, 1]

        # Track training calls
        if self.training:
            self.cum_batch_size_called += X.shape[0]

        return means, log_vars

    def reset_parameters(self):
        for layer in self.net:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()

    def on_epoch_start(self):
        self.cum_batch_size_called = 0


class FullySharedGaussianVarDistr(pl.LightningModule):
    """
    Gaussian variational distribution parameter model that uses a shared model
    for all univariate gaussian distributions.
    Also shared model for mean and log-var parameter predictions.
    """
    def __init__(self, hparams):
        super(FullySharedGaussianVarDistr, self).__init__()
        self.hparams = hparams.var_model

        self.init()

        self.cum_batch_size_called = 0

    def init(self):
        assert self.hparams.activation in ('lrelu', 'sigmoid'), \
            'Activation not supported!'

        if self.hparams.input_missing_vectors == 'missing':
            self.input_missing_vectors = True
            self.input_selected_missing_vectors = False
        elif self.hparams.input_missing_vectors == 'selected_missing':
            self.input_missing_vectors = False
            self.input_selected_missing_vectors = True
        else:
            self.input_missing_vectors = False
            self.input_selected_missing_vectors = False

        assert not (self.input_missing_vectors
                    and self.input_selected_missing_vectors), \
            'Cannot input both M and M_selected!'

        # Input dimensionality is the #vars.
        if self.input_missing_vectors or self.input_selected_missing_vectors:
            # *2 if missingness mask is passed to the models
            input_len = self.hparams.input_dim*2
        else:
            input_len = self.hparams.input_dim

        output_len = self.hparams.input_dim

        # Add input and output dimensions to the list
        hidden_layer_dims = [input_len] + self.hparams.hidden_layer_dims

        # Create variational distribution model
        hidden_layers = []
        for i in range(0, len(hidden_layer_dims)-1):
            hidden_layers.append(
                        nn.Linear(in_features=hidden_layer_dims[i],
                                  out_features=hidden_layer_dims[i+1]))

            if self.hparams.activation == 'sigmoid':
                hidden_layers.append(nn.Sigmoid())
            elif self.hparams.activation == 'lrelu':
                hidden_layers.append(nn.LeakyReLU())

        self.net = nn.Sequential(*hidden_layers)

        self.mean = nn.Linear(in_features=hidden_layer_dims[-1],
                              out_features=output_len)
        self.log_var = nn.Linear(in_features=hidden_layer_dims[-1],
                                 out_features=output_len)

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--var_model.input_dim',
                            type=int, required=True,
                            help='Dimensionality of input.')
        parser.add_argument('--var_model.hidden_layer_dims',
                            type=int, nargs='+', required=True,
                            help=('Dimensionalities of shared '
                                  'hidden layers.'))
        parser.add_argument('--var_model.activation',
                            type=str, required=True,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid'])
        parser.add_argument('--var_model.input_missing_vectors',
                            type=str, default=None,
                            choices=['missing', 'missing_selected'],
                            help=('Which missing matrix to input if any'
                                  '. \'missing\', \'missing_selected\','
                                  ' or None.'))
        return parser

    def forward(self, X, M, M_selected, sample_all=None):
        """
        Compute Gaussian mean and log-variance of the missing variables given
        the observed X.
        Args:
            X (N, D): observable variables (with the missing variable)
            M (N, D): the missingness boolean mask. 1 - observed, 0 - missing.
            M_selected (N, D): a subset of M, for which we want to compute the
                variational distribution parameters.
        Return:
            means (float): variational Gaussian mean of the missing variable
                for each sample and each dimension, 0 if the value is not
                missing at the current sample & dimension.
            log_vars (float): variational Gaussian log-variance of the missing
                variable for each sample and dimension, -inf if the value is
                not missing at the current sample & dimension. So that the
                variance is 0 (exp(-inf)=0).
        """
        if self.input_selected_missing_vectors:
            # Cast type to X so can concat
            M = M_selected.type_as(X)
            X = torch.cat((X, M), dim=1)
        elif self.input_missing_vectors:
            # Cast type to X so can concat
            X = torch.cat((X, M), dim=1)

        intermediate = self.net(X)
        mean = self.mean(intermediate)
        log_var = self.log_var(intermediate)

        # Track training calls
        if self.training:
            self.cum_batch_size_called += X.shape[0]

        return mean, log_var

    def reset_parameters(self):
        for layer in self.net:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.reset_parameters()

        self.mean.reset_parameters()
        self.log_var.reset_parameters()

    def on_epoch_start(self):
        self.cum_batch_size_called = 0


class GaussVarMADE(pl.LightningModule):
    """
    Implements Masked AutoEncoder for Density Estimation model, by Germain et al. 2015
    https://arxiv.org/abs/1502.03509

    Uses Gaussian conditional distributions.

    Roughly based on re-implementation of Andrej Karpathy https://github.com/karpathy/pytorch-made
    """
    def __init__(self, hparams):
        """
        hparams.var_model (Namespace):
            input_dim: integer; number of inputs
            hidden_dims sizes: a list of integers; number of units in hidden layers
            num_masks: can be used to train ensemble over orderings/connections
            activation: LReLU or Sigmoid
            resample_masks_every: rpochs in integer, how often to to resample masks
        """

        super().__init__()
        self.hparams = hparams.var_model

        assert self.hparams.activation in ('lrelu', 'sigmoid', 'relu'), \
            'Activation not supported!'

        hidden_dims = self.hparams.hidden_dims
        mean_layers = []
        log_var_layers = []
        if self.hparams.activation == 'sigmoid':
            mean_layers.append(nn.Sigmoid())
            log_var_layers.append(nn.Sigmoid())
        elif self.hparams.activation == 'lrelu':
            mean_layers.append(nn.LeakyReLU())
            log_var_layers.append(nn.LeakyReLU())
        elif self.hparams.activation == 'relu':
            mean_layers.append(nn.ReLU())
            log_var_layers.append(nn.ReLU())

        for i in range(len(hidden_dims)-1):
            mean_layers.append(MaskedLinear(in_features=hidden_dims[i],
                                            out_features=hidden_dims[i+1]))
            log_var_layers.append(MaskedLinear(in_features=hidden_dims[i],
                                               out_features=hidden_dims[i+1]))

            if self.hparams.activation == 'sigmoid':
                mean_layers.append(nn.Sigmoid())
                log_var_layers.append(nn.Sigmoid())
            elif self.hparams.activation == 'lrelu':
                mean_layers.append(nn.LeakyReLU())
                log_var_layers.append(nn.LeakyReLU())
            elif self.hparams.activation == 'relu':
                mean_layers.append(nn.ReLU())
                log_var_layers.append(nn.ReLU())

        self.mean_mid = nn.Sequential(*mean_layers)
        self.log_var_mid = nn.Sequential(*log_var_layers)

        # Add input/output layers
        self.mean_in = MaskedLinearPerExample(in_features=self.hparams.input_dim,
                                              out_features=hidden_dims[0])
        self.log_var_in = MaskedLinearPerExample(in_features=self.hparams.input_dim,
                                                 out_features=hidden_dims[0])
        self.mean_out = MaskedLinearPerExample(in_features=hidden_dims[-1],
                                               out_features=self.hparams.input_dim)
        self.log_var_out = MaskedLinearPerExample(in_features=hidden_dims[-1],
                                                  out_features=self.hparams.input_dim)

        # seeds for orders/connectivities of the model ensemble
        self.seed = 0  # for cycling through num_masks orderings

        self.m = []
        self.update_masks()  # builds the initial self.m connectivity
        # NOTE: we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

        self.cum_batch_size_called = 0

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--var_model.input_dim',
                            type=int, required=True,
                            help='Dimensionality of input.')
        parser.add_argument('--var_model.hidden_dims',
                            type=int, nargs='+', required=True,
                            help='Dimensionalities of hidden layers.')
        parser.add_argument('--var_model.activation',
                            type=str, required=True,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid', 'relu'])
        parser.add_argument('--var_model.num_masks',
                            type=int, default=1,
                            help=('Can be used to train ensemble over '
                                  'orderings/connections'))
        parser.add_argument('--var_model.resample_masks_every',
                            type=int, required=True,
                            help=('Every N epochs resample masks'))
        # TODO:
        parser.add_argument('--var_model.input_missing_vectors',
                            type=str, default=False,
                            help=('Which missing matrix to input if any'
                                  '. \'missing\', \'missing_selected\','
                                  ' or None.'))
        return parser

    def update_masks(self):
        # only a single seed, skip for efficiency
        if self.m and self.hparams.num_masks == 1:
            return
        L = len(self.hparams.hidden_dims)

        # fetch the next seed and construct a random stream
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        self.seed = (self.seed + 1) % self.hparams.num_masks

        # Construct connectivity matricess
        self.m = []
        self.m.append(torch.randint(self.hparams.input_dim-1,
                                    size=(self.hparams.hidden_dims[0], ),
                                    generator=rng))
        for l in range(1, L):
            self.m.append(torch.randint(self.m[l-1].min(),
                                        self.hparams.input_dim-1,
                                        size=(self.hparams.hidden_dims[l], ),
                                        generator=rng))

        # construct the mask matrices
        masks = [self.m[l-1][:, None] <= self.m[l][None, :] for l in range(1, L)]
        # masks.append(self.m[L-1][:, None] < self.m[-1][None, :])

        # set the masks in all MaskedLinear layers
        mean_layers = [l for l in self.mean_mid if isinstance(l, MaskedLinear)]
        log_var_layers = [l for l in self.log_var_mid if isinstance(l, MaskedLinear)]
        for ml, vl, m in zip(mean_layers, log_var_layers, masks):
            ml.set_mask(m)
            vl.set_mask(m)

    def create_rand_order_based_on_missingness(self, M):
        # fetch the next seed and construct a random stream
        rng = torch.Generator(device=M.device)
        rng.manual_seed(self.seed)

        # Make sure that the scores for the observed values are lower than missing
        scores_miss = torch.rand(M.shape, device=M.device, dtype=torch.float, generator=rng)*~M
        scores_obs = (-torch.rand(M.shape, device=M.device, dtype=torch.float, generator=rng))*M
        scores = scores_miss + scores_obs

        # Sorting the random scores gives us a random order then ensures
        # that variables of interest always depend on all other variables
        sorted_idxs = torch.argsort(scores, dim=1, descending=False)
        temp = torch.arange(0, M.shape[-1], device=M.device).expand(M.shape[0], -1)
        order = torch.zeros_like(M, dtype=torch.long).scatter(dim=1,
                                                              index=sorted_idxs,
                                                              src=temp)

        return order

    def set_per_example_masks(self, M):
        self.order = self.create_rand_order_based_on_missingness(M)

        # Get input masks for each example
        input_example_masks = self.order[:, :, None] <= self.m[0][None, :].to(M.device)

        # Get output masks for each example
        output_example_masks = self.m[-1][:, None].to(M.device) < self.order[:, None, :]

        # Set the masks to the input layer
        self.mean_in.set_masks(input_example_masks)
        self.log_var_in.set_masks(input_example_masks)

        # Set the masks to the output layer
        self.mean_out.set_masks(output_example_masks)
        self.log_var_out.set_masks(output_example_masks)

    def forward(self, X, K, idx_mask):
        mean = self.mean_in(X, K, idx_mask)
        mean = self.mean_mid(mean)
        mean = self.mean_out(mean, K, idx_mask)

        log_var = self.log_var_in(X, K, idx_mask)
        log_var = self.log_var_mid(log_var)
        log_var = self.log_var_out(log_var, K, idx_mask)

        # Track training calls
        if self.training:
            self.cum_batch_size_called += X.shape[0]

        return mean, log_var

    def sample(self, X, M, K):
        X = X.clone().expand(K, -1, -1)
        log_probs = torch.zeros_like(X)
        # Final masks are chosen for each batch to ensure that variables of
        # interest depend upon each other, and always on the other variables
        self.set_per_example_masks(M)

        # Get order of variables of interest, the rank of uninteresting variables
        # is set to greater than the max, so shall go last (we will ignore those)
        order = self.order.clone()
        order[M] = M.shape[-1]+1
        _, sorted_idx = torch.sort(order, dim=-1, descending=False)

        left = M.clone()
        i = 0
        all_idx = torch.arange(order.shape[0])
        while left.sum() != M.numel():
            idx = sorted_idx[:, i]
            order_idx = order[all_idx, idx]
            # Don't compute autoregress if we've reached the uninteresting variables
            idx_mask = order_idx < M.shape[-1]+1

            # Compute conditional distributions
            mean, log_var = self(X[:, idx_mask, :].reshape(-1, X.shape[-1]),
                                 K, idx_mask)
            normal = distr.Normal(loc=mean, scale=torch.exp(1/2 * log_var))

            # Sample
            samples = normal.rsample()

            # Evaluate sample log_prob and store for relevant samples (by idx)
            sample_log_prob = normal.log_prob(samples).reshape(K, -1, X.shape[-1])
            M_temp = torch.zeros_like(M)
            M_temp[torch.arange(M.shape[0]), idx] = 1
            M_temp[~idx_mask, :] = 0
            log_probs[:, M_temp] += sample_log_prob[:, M_temp[idx_mask, :]]

            # Store relevant samples (by idx)
            samples = samples.reshape(K, -1, X.shape[-1])
            X[:, M_temp] = samples[:, M_temp[idx_mask, :]]
            X = X.clone()

            # remove these values from pending
            left[all_idx, idx] = 1
            i += 1

        # Compute total log-probability of each sample
        log_probs = log_probs.sum(dim=-1)

        return X, log_probs

    def reset_parameters(self):
        self.mean_in.reset_parameters()
        self.mean_out.reset_parameters()
        for layer in self.mean_mid:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()

        self.log_var_in.reset_parameters()
        self.log_var_out.reset_parameters()
        for layer in self.log_var_mid:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, nn.ReLU)):
                layer.reset_parameters()

    def on_epoch_start(self):
        self.cum_batch_size_called = 0

        if self.current_epoch % self.hparams.resample_masks_every == 0:
            self.update_masks()
