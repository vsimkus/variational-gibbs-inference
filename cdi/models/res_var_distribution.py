import torch
import pytorch_lightning as pl

from cdi.layers.channel_linear import LinearWithChannels, LinearWithSharedChannels


class ChannelResidualBlock(torch.nn.Module):
    def __init__(self, features, channel_dim, activation=torch.nn.functional.relu, dropout_probability=0.0,
                 zero_initialization=True, reduce_mem=False, shared=False):
        super().__init__()
        self.activation = activation
        self.reduce_mem = reduce_mem

        Layer = LinearWithSharedChannels if shared else LinearWithChannels
        self.linear_layers = torch.nn.ModuleList(
            [Layer(in_features=features,
                   out_features=features,
                   channels=channel_dim,
                   reduce_mem=reduce_mem,
                   mask=None)
             for _ in range(2)]
        )

        self.dropout = torch.nn.Dropout(p=dropout_probability)
        if zero_initialization:
            torch.nn.init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, sparse_args=None):
        M_channel = None
        channel_sparse_idx = None
        if self.reduce_mem:
            M_channel, channel_sparse_idx = sparse_args

        temps = inputs
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps, M_channel, channel_sparse_idx)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps, M_channel, channel_sparse_idx)

        return inputs + temps

    def reset_parameters(self):
        for l in self.linear_layers:
            l.reset_parameters()


class ResidualMixtureOfGaussiansVarConditionals(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams.var_model

        self.init_nn()
        self.reset_parameters()

    def init_nn(self):
        # Create network layers
        mean_layers = []
        log_var_layers = []
        comp_logit_layers = []
        # In the first layer of the j-th conditional, mask the
        # j-th input with a zero.
        # Prepare input weight mask, zeros on diag, elsewhere 1
        mask = (torch.ones(self.hparams.input_dim, self.hparams.input_dim)
                - torch.eye(self.hparams.input_dim, self.hparams.input_dim))
        mask = mask.unsqueeze(-2)
        # Using a custom LinearWithChannels layer, which is analogous to Linear.
        # Allows parallel computation of all d conditionals for each input:
        # each "channel" corresponds to different conditional.
        mean_layers.append(LinearWithChannels(in_features=self.hparams.input_dim,
                                              out_features=self.hparams.hidden_features,
                                              channels=self.hparams.input_dim,  # d conditionals
                                              mask=mask,
                                              reduce_mem=False))
        log_var_layers.append(LinearWithChannels(in_features=self.hparams.input_dim,
                                                 out_features=self.hparams.hidden_features,
                                                 channels=self.hparams.input_dim,  # d conditionals
                                                 mask=mask,
                                                 reduce_mem=False))
        comp_logit_layers.append(LinearWithChannels(in_features=self.hparams.input_dim,
                                                    out_features=self.hparams.hidden_features,
                                                    channels=self.hparams.input_dim,  # d conditionals
                                                    mask=mask,
                                                    reduce_mem=False))

        for i in range(self.hparams.num_blocks):
            mean_layers.append(ChannelResidualBlock(
                features=self.hparams.hidden_features,
                channel_dim=self.hparams.input_dim,
                activation=torch.nn.functional.relu,
                dropout_probability=0.,
                reduce_mem=False
            ))
            log_var_layers.append(ChannelResidualBlock(
                features=self.hparams.hidden_features,
                channel_dim=self.hparams.input_dim,
                activation=torch.nn.functional.relu,
                dropout_probability=0.,
                reduce_mem=False
            ))
            comp_logit_layers.append(ChannelResidualBlock(
                features=self.hparams.hidden_features,
                channel_dim=self.hparams.input_dim,
                activation=torch.nn.functional.relu,
                dropout_probability=0.,
                reduce_mem=False
            ))

        # Add final output layers
        mean_layers.append(LinearWithChannels(in_features=self.hparams.hidden_features,
                                              # one output for each conditional component
                                              out_features=self.hparams.num_components,
                                              channels=self.hparams.input_dim,  # d conditionals
                                              reduce_mem=False))
        log_var_layers.append(LinearWithChannels(in_features=self.hparams.hidden_features,
                                                 # one output for each conditional component
                                                 out_features=self.hparams.num_components,
                                                 channels=self.hparams.input_dim,  # d conditionals
                                                 reduce_mem=False))
        comp_logit_layers.append(LinearWithChannels(in_features=self.hparams.hidden_features,
                                                    # one output for each conditional component
                                                    out_features=self.hparams.num_components,
                                                    channels=self.hparams.input_dim,  # d conditionals
                                                    reduce_mem=False))

        self.mean = torch.nn.Sequential(*mean_layers)
        self.log_var = torch.nn.Sequential(*log_var_layers)
        self.comp_logits = torch.nn.Sequential(*comp_logit_layers)

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--var_model.num_components',
                            type=int, required=True,
                            help='Number of mixture components.')
        parser.add_argument('--var_model.input_dim',
                            type=int, required=True,
                            help='Dimensionality of input.')
        parser.add_argument('--var_model.hidden_features',
                            type=int, required=True,
                            help=('Dimensionalities of hidden layers.'))
        parser.add_argument('--var_model.num_blocks',
                            type=int, required=True,
                            help=('Number of residual blocks.'))

        return parser

    # The rest of the args are not used, but are there for backwards compatibility
    def forward(self, X, M=None, *args, **kwargs):
        means = self.mean(X.unsqueeze(1)).squeeze(2)
        log_vars = self.log_var(X.unsqueeze(1)).squeeze(2)
        comp_logits = self.comp_logits(X.unsqueeze(1)).squeeze(2)

        return means, log_vars, comp_logits

    def reset_parameters(self):
        for nn in [self.mean, self.log_var, self.comp_logits]:
            for layer in nn:
                if not isinstance(layer, (torch.nn.LeakyReLU, torch.nn.Sigmoid, torch.nn.ReLU)):
                    layer.reset_parameters()
