import torch
import torch.nn as nn
import torch.nn.functional as F

from cdi.models.fc_vae import FC_VAE
from cdi.util.arg_utils import parse_bool


class FC_PartialVAE(FC_VAE):
    """
    Fully-connected PartialVAE with PointNet+
    "EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE" by Ma et al. (2019)
    """

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--fc_vae_model.num_z_samples', type=int,
                            default=1, help='Number of latent samples.')
        parser.add_argument('--fc_vae_model.input_dim', type=int,
                            required=True, help='Image dimension.')
        parser.add_argument('--fc_vae_model.decoder_hidden_dims', type=int,
                            nargs='+', required=True,
                            help='Decoder layer dimensionalities')
        parser.add_argument('--fc_vae_model.activation',
                            type=str, required=True,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid'])
        parser.add_argument('--fc_vae_model.bound',
                            type=str, required=True,
                            choices=['vae', 'miwae'],
                            help='Which bound to use.')
        parser.add_argument('--fc_vae_model.marginalise',
                            type=parse_bool, required=False,
                            help=('For VAE bound, marginalise the generator or not.'))
        parser.add_argument('--fc_vae_model.marginalise_val',
                            type=parse_bool, required=False,
                            help=('For VAE bound, marginalise the generator or not. (val only)'))

        parser.add_argument('--fc_vae_model.pos_emb_dim', type=int,
                            help=('Size of position embedding.'))
        parser.add_argument('--fc_vae_model.encoder_shared_net_layers', type=int,
                            nargs='+', help='Number of hidden layers in the shared net.')
        parser.add_argument('--fc_vae_model.encoder_hidden_dims', type=int,
                            nargs='+', required=True,
                            help='Encoder layer dimensionalities')
        parser.add_argument('--fc_vae_model.encoder_activation',
                            type=str, required=True,
                            help='Encoder activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid'])
        parser.add_argument('--fc_vae_model.encoder_residuals', default=False,
                            type=parse_bool, help=('Whether to use residual connections in the encoder or not.'))
        parser.add_argument('--fc_vae_model.aggregation', default='sum',
                            choices=['sum', 'avg'], type=str,
                            help='What type of invariante aggregation should be used.')
        parser.add_argument('--fc_vae_model.no_aggregation_activation', default=False,
                            type=parse_bool, help='If true, then no activation after aggregation.')

        return parser

    def initialise(self):
        assert self.hparams.activation in ('lrelu', 'sigmoid'), \
            'Activation not supported!'

        activation = self.hparams.activation if self.hparams.activation != 'lrelu' else 'leaky_relu'
        pos_emb = torch.empty(self.hparams.input_dim, self.hparams.pos_emb_dim, dtype=torch.float)
        nn.init.kaiming_uniform_(pos_emb, a=0.1, nonlinearity=activation)

        self.pos_emb = torch.nn.Parameter(pos_emb, requires_grad=True)

        b = torch.randn(self.hparams.input_dim, dtype=torch.float)
        self.b = torch.nn.Parameter(b, requires_grad=True)

        # Add input dimensions to the list
        hidden_dims = [self.hparams.pos_emb_dim+2] + self.hparams.encoder_shared_net_layers

        shared_encoder = []
        for i in range(len(hidden_dims)-1):
            if self.hparams.encoder_activation == 'sigmoid':
                activation = F.sigmoid
            elif self.hparams.encoder_activation == 'lrelu':
                activation = F.leaky_relu

            shared_encoder.append(FCLayer(hidden_dims[i], hidden_dims[i+1],
                                          residual=self.hparams.encoder_residuals,
                                          activation=activation))

        self.shared_encoder = nn.Sequential(*shared_encoder)

        # Add input dimensions to the list
        hidden_dims = [self.hparams.encoder_shared_net_layers[-1]] + self.hparams.encoder_hidden_dims

        encoder = []
        for i in range(len(hidden_dims)-2):
            if self.hparams.encoder_activation == 'sigmoid':
                activation = F.sigmoid
            elif self.hparams.encoder_activation == 'lrelu':
                activation = F.leaky_relu

            encoder.append(FCLayer(hidden_dims[i], hidden_dims[i+1],
                                   residual=self.hparams.encoder_residuals,
                                   activation=activation))

        self.encoder = nn.Sequential(*encoder)
        self.enc_mean = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        self.enc_log_var = nn.Linear(hidden_dims[-2], hidden_dims[-1])

        # Add output dimensions to the list
        hidden_dims = self.hparams.decoder_hidden_dims + [self.hparams.input_dim]

        decoder = []
        for i in range(len(hidden_dims)-2):
            decoder.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

            if self.hparams.activation == 'sigmoid':
                decoder.append(nn.Sigmoid())
            elif self.hparams.activation == 'lrelu':
                decoder.append(nn.LeakyReLU())

        self.decoder = nn.Sequential(*decoder)
        self.dec_mean = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        self.dec_log_var = nn.Linear(hidden_dims[-2], hidden_dims[-1])

    def encode(self, X, M=None):
        X_orig = X
        X = X.unsqueeze(-1).expand(-1, -1, self.pos_emb.shape[-1]) * self.pos_emb.unsqueeze(0).expand(X.shape[0], -1, -1)

        # Add the original X and a bias as per their implementation (this is not mentioned in their paper)
        X = torch.cat([X_orig.unsqueeze(-1), X, self.b.unsqueeze(0).unsqueeze(-1).expand(X_orig.shape[0], -1, -1)], dim=-1)

        X_emb = self.shared_encoder(X)

        if M is not None:
            X_emb = X_emb * M.unsqueeze(-1).expand(-1, -1, X_emb.shape[-1])

        if not hasattr(self.hparams, 'aggregation') or self.hparams.aggregation == 'sum':
            X_emb = torch.sum(X_emb, axis=-2)
        elif self.hparams.aggregation == 'avg':
            X_emb = torch.sum(X_emb, axis=-2)
            # Compute average
            if M is not None:
                M_summed = M.sum(axis=-1, keepdim=True)
                # Avoid division by zero
                M_summed[M_summed == 0] = 1
                X_emb = X_emb / M_summed

        if not hasattr(self.hparams, 'no_aggregation_activation') or not self.hparams.no_aggregation_activation:
            if self.hparams.encoder_activation == 'sigmoid':
                X_emb = F.sigmoid(X_emb)
            elif self.hparams.encoder_activation == 'lrelu':
                X_emb = F.leaky_relu(X_emb)

        Z_prime = self.encoder(X_emb)
        Z_mean = self.enc_mean(Z_prime)
        Z_log_var = self.enc_log_var(Z_prime)

        return Z_mean, Z_log_var

    def reset_encoder(self):
        activation = self.hparams.activation if self.hparams.activation != 'lrelu' else 'leaky_relu'
        pos_emb = torch.empty(self.hparams.input_dim, self.hparams.pos_emb_dim, dtype=torch.float)
        nn.init.kaiming_uniform_(pos_emb, a=0.1, nonlinearity=activation)

        self.pos_emb = torch.nn.Parameter(pos_emb, requires_grad=True)

        b = torch.randn(self.hparams.input_dim, dtype=torch.float)
        self.b = torch.nn.Parameter(b, requires_grad=True)

        for layer in self.shared_encoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.reset_parameters()
        for layer in self.encoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.reset_parameters()
        self.enc_mean.reset_parameters()
        self.enc_log_var.reset_parameters()

    # Hacky way to separate the parameters so I can use separate optimisers
    # for both parts of the model
    def generator_parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if 'dec' in name:
                yield param

    def encoder_parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if 'dec' not in name:
                yield param


class FCLayer(nn.Linear):
    def __init__(self, *args, residual=False, activation=F.relu, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual = residual
        self.activation = activation

        if self.in_features != (self.out_features):
            self.residual = False

    def forward(self, X):
        X_in = X  # for residual connection

        # Apply linear transform
        X = super().forward(X)

        if self.activation:
            X = self.activation(X)

        if self.residual:
            X = X_in + X  # residual connection

        return X
