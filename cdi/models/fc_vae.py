import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as distr
import torch.nn as nn

from cdi.util.arg_utils import parse_bool
from cdi.layers.unflatten import Unflatten


class FC_VAE(pl.LightningModule):
    """
    Fully-connected VAE
    """
    def __init__(self, args):
        super(FC_VAE, self).__init__()
        self.hparams = args.fc_vae_model

        self.initialise()
        self.cum_batch_size_called = 0

    def set_hparams(self, hparams):
        self.hparams = hparams.fc_vae_model

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--fc_vae_model.num_z_samples', type=int,
                            default=1, help='Number of latent samples.')
        parser.add_argument('--fc_vae_model.input_dim', type=int,
                            required=True, help='Image dimension.')
        parser.add_argument('--fc_vae_model.encoder_hidden_dims', type=int,
                            nargs='+', required=True,
                            help='Encoder layer dimensionalities')
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
                            help=('For VAE bound, marginalise the generator or not. (train only)'))
        parser.add_argument('--fc_vae_model.marginalise_val',
                            type=parse_bool, required=False,
                            help=('For VAE bound, marginalise the generator or not. (val only)'))

        parser.add_argument('--fc_vae_model.mask_mis_with_zero',
                            type=parse_bool, default=False,
                            help=('If true, masks missing encoder inputs with 0.'))

        parser.add_argument('--fc_vae_model.local_vi', type=parse_bool,
                            default=False, help=('Whether to use local VI.'))
        return parser

    def initialise(self):
        assert self.hparams.activation in ('lrelu', 'sigmoid'), \
            'Activation not supported!'

        # Add input dimensions to the list
        hidden_dims = [self.hparams.input_dim] + self.hparams.encoder_hidden_dims

        encoder = []
        for i in range(len(hidden_dims)-2):
            encoder.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

            if self.hparams.activation == 'sigmoid':
                encoder.append(nn.Sigmoid())
            elif self.hparams.activation == 'lrelu':
                encoder.append(nn.LeakyReLU())

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
        Z_prime = self.encoder(X)
        Z_mean = self.enc_mean(Z_prime)
        Z_log_var = self.enc_log_var(Z_prime)

        return Z_mean, Z_log_var

    def decode(self, Z):
        X_tilde = self.decoder(Z)
        X_tilde_mean = self.dec_mean(X_tilde)
        X_tilde_log_var = self.dec_log_var(X_tilde)

        return X_tilde_mean, X_tilde_log_var

    def vae_bound(self, X, X_tilde_mean, X_tilde_log_var, Z_mean, Z_log_var, M):
        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))

        log_prob = X_norm.log_prob(X)
        if ((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
                or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val)):
            log_prob = log_prob * M
        log_prob = log_prob.sum(dim=-1).mean(dim=0)

        # Instead of computing entropy and log-prob of Z,
        # Compute analytical -KL(q(z|x) || p(z)) term here
        KL_neg = (1/2 * (1 + Z_log_var - torch.exp(Z_log_var) - Z_mean**2)).sum(dim=-1)

        # Return lower-bound on the marginal log-probability
        return log_prob + KL_neg

    def miwae_bound(self, X, X_tilde_mean, X_tilde_log_var, Z, Z_mean, Z_log_var, M):
        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))
        X_log_prob = X_norm.log_prob(X) * M
        X_log_prob = X_log_prob.sum(dim=-1)

        # Z prior is standard normal
        Z_log_prob = (-1/2*torch.log(torch.tensor(2*np.pi, device=X.device)) - 1/2 * Z**2).sum(dim=-1)

        # entropy term of Z posterior
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z_neg_ent = Z_norm.log_prob(Z).sum(dim=-1)

        return (torch.logsumexp(X_log_prob + Z_log_prob - Z_neg_ent, dim=0)
                - torch.log(torch.tensor(Z.shape[0], dtype=torch.float, device=X.device)))

    def forward(self, X, M, I=None):
        # if self.training:
        if self.hparams.bound == 'miwae' or (hasattr(self.hparams, 'mask_mis_with_zero') and self.hparams.mask_mis_with_zero):
            # Set missing inputs to zero
            X = X * M

        if hasattr(self.hparams, 'local_vi') and self.hparams.local_vi:
            Z_mean, Z_log_var = self.encode_local(I)
        else:
            Z_mean, Z_log_var = self.encode(X, M)

        # Sample latent variables
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z = Z_norm.rsample(sample_shape=(self.hparams.num_z_samples,))
        Z = Z.reshape(-1, Z.shape[-1])

        X_tilde_mean, X_tilde_log_var = self.decode(Z)
        X_tilde_mean = X_tilde_mean.reshape(self.hparams.num_z_samples, *X.shape)
        X_tilde_log_var = X_tilde_log_var.reshape(self.hparams.num_z_samples, *X.shape)

        if not hasattr(self.hparams, 'bound') or self.hparams.bound == 'vae':  # BC
            return self.vae_bound(X, X_tilde_mean, X_tilde_log_var, Z_mean, Z_log_var, M)
        elif self.hparams.bound == 'miwae':
            Z = Z.reshape(self.hparams.num_z_samples, -1, Z.shape[-1])
            return self.miwae_bound(X, X_tilde_mean, X_tilde_log_var, Z, Z_mean, Z_log_var, M)
        else:
            raise Exception('Invalid bound!')

    def mcmc_sample_missing_values(self, X, M):
        Z_mean, Z_log_var = self.encode(X, M)
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z = Z_norm.rsample()

        X_tilde_mean, X_tilde_log_var = self.decode(Z)
        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))
        X_tilde = X_norm.rsample()

        return X_tilde

    def reset_parameters(self):
        self.reset_encoder()
        self.reset_decoder()

    def reset_encoder(self):
        for layer in self.encoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.reset_parameters()
        self.enc_mean.reset_parameters()
        self.enc_log_var.reset_parameters()

    def reset_encoder_first_layer(self):
        self.encoder[0].reset_parameters()

    def reset_decoder(self):
        for layer in self.decoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.reset_parameters()

        self.dec_mean.reset_parameters()
        self.dec_log_var.reset_parameters()

    # Test

    def freeze_decoder(self):
        for layer in self.decoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.requires_grad_(False)
        self.dec_mean.requires_grad_(False)
        self.dec_log_var.requires_grad_(False)

    def freeze_encoder(self):
        for layer in self.encoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.requires_grad_(False)
        self.enc_mean.requires_grad_(False)
        self.enc_log_var.requires_grad_(False)

    def test_loglik(self, X, M, num_z_samples):
        # X = X*M
        # Z_norm = distr.Normal(loc=torch.zeros(self.hparams.encoder_hidden_dims[-1], device=X.device),
        #                       scale=torch.ones(self.hparams.encoder_hidden_dims[-1], device=X.device))
        # Z = Z_norm.sample(sample_shape=(num_z_samples, X.shape[0]))
        # Z = Z.reshape(-1, Z.shape[-1])

        # X_tilde_mean, X_tilde_log_var = self.decode(Z)
        # X_tilde_mean = X_tilde_mean.reshape(num_z_samples, *X.shape)
        # X_tilde_log_var = X_tilde_log_var.reshape(num_z_samples, *X.shape)
        # X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))

        # # log_probs = (X_norm.log_prob(X)*M).sum(dim=-1)
        # log_probs = X_norm.log_prob(X).sum(dim=-1)
        # log_probs = torch.logsumexp(log_probs, dim=0)
        # log_probs -= torch.log(torch.tensor(num_z_samples,
        #                                     dtype=torch.float,
        #                                     device=X.device))

        self.hparams.bound = 'miwae'
        self.hparams.num_z_samples = num_z_samples

        log_probs = self.forward(X, M)

        return log_probs

    def test_z_posterior(self, X, M):
        if self.hparams.bound == 'miwae' or (hasattr(self.hparams, 'mask_mis_with_zero') and self.hparams.mask_mis_with_zero):
            # Set missing inputs to zero
            X = X * M

        if hasattr(self.hparams, 'local_vi') and self.hparams.local_vi:
            Z_mean, Z_log_var = self.encode_local(I)
        else:
            Z_mean, Z_log_var = self.encode(X, M)

        return Z_mean, Z_log_var

    # Local-VI

    def init_local_params(self, data_count):
        # Initialise to the prior
        self.q_local_mean = torch.nn.Parameter(
                                data=torch.zeros(
                                        data_count,
                                        self.hparams.encoder_hidden_dims[-1],
                                        dtype=torch.float),
                                requires_grad=True)
        self.q_local_log_var = torch.nn.Parameter(
                                data=torch.zeros(
                                        data_count,
                                        self.hparams.encoder_hidden_dims[-1],
                                        dtype=torch.float),
                                requires_grad=True)

    def encode_local(self, I):
        # Select mean and log_var by index
        Z_mean = self.q_local_mean[I, :]
        Z_log_var = self.q_local_log_var[I, :]

        return Z_mean, Z_log_var


class CONV_VAE_bernoulli(pl.LightningModule):
    """
    Convolutional VAE
    """
    def __init__(self, args):
        super(CONV_VAE_bernoulli, self).__init__()
        self.hparams = args.conv_vae_model

        self.initialise()
        self.cum_batch_size_called = 0

    def set_hparams(self, hparams):
        self.hparams = hparams.conv_vae_model

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--conv_vae_model.num_z_samples', type=int,
                            default=1, help='Number of latent samples.')
        # parser.add_argument('--conv_vae_model.input_dim', type=int,
        #                     required=True, help='Image dimension.')
        # parser.add_argument('--conv_vae_model.encoder_hidden_dims', type=int,
        #                     nargs='+', required=True,
        #                     help='Encoder layer dimensionalities')
        # parser.add_argument('--conv_vae_model.decoder_hidden_dims', type=int,
        #                     nargs='+', required=True,
        #                     help='Decoder layer dimensionalities')
        parser.add_argument('--conv_vae_model.img_shape', nargs=2,
                            type=int, help=('Shape of the image (tuple).'))
        parser.add_argument('--conv_vae_model.encoder_channels', type=int,
                            nargs='+', help=('Channels.'))
        parser.add_argument('--conv_vae_model.encoder_kernel_size', type=int,
                            nargs='+', help=('Kernels.'))
        parser.add_argument('--conv_vae_model.encoder_strides', type=int,
                            nargs='+', help=('Strides.'))
        parser.add_argument('--conv_vae_model.z_dim', type=int,
                            help=('Dimension of the latents.'))

        parser.add_argument('--conv_vae_model.activation',
                            type=str, required=True,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid'])
        parser.add_argument('--conv_vae_model.bound',
                            type=str, required=True,
                            choices=['vae', 'miwae'],
                            help='Which bound to use.')
        parser.add_argument('--conv_vae_model.marginalise',
                            type=parse_bool, required=False,
                            help=('For VAE bound, marginalise the generator or not. (train only)'))
        parser.add_argument('--conv_vae_model.marginalise_val',
                            type=parse_bool, required=False,
                            help=('For VAE bound, marginalise the generator or not. (val only)'))

        parser.add_argument('--conv_vae_model.mask_mis_with_zero',
                            type=parse_bool, default=False,
                            help=('If true, masks missing encoder inputs with 0.'))

        return parser

    def initialise(self):
        assert self.hparams.activation in ('lrelu', 'sigmoid'), \
            'Activation not supported!'

        assert len(self.hparams.encoder_channels) == len(self.hparams.encoder_kernel_size), \
            'Parameter lists should be of the same length!'

        with torch.no_grad():

            X = torch.randn(2, 1, *self.hparams.img_shape)  # (B, C, H, W)
            X_in = X.reshape(2, -1)

            encoder = []
            encoder.append(Unflatten(X.reshape(2, -1).shape[-1], (1, *self.hparams.img_shape)))
            for i in range(len(self.hparams.encoder_channels)):
                layer = nn.Conv2d(in_channels=X.shape[1],
                                  out_channels=self.hparams.encoder_channels[i],
                                  kernel_size=self.hparams.encoder_kernel_size[i],
                                  stride=self.hparams.encoder_strides[i])
                encoder.append(layer)

                X = layer(X)

                if self.hparams.activation == 'sigmoid':
                    encoder.append(nn.Sigmoid())
                elif self.hparams.activation == 'lrelu':
                    encoder.append(nn.LeakyReLU())

                # print(X.shape)

            encoder.append(nn.Flatten())
            X_shape_before_flatten = X.shape
            X = encoder[-1](X)
            X_shape_after_flatten = X.shape

            encoder.append(nn.Linear(in_features=X.shape[-1],
                                     out_features=self.hparams.z_dim))
            X = encoder[-1](X)
            if self.hparams.activation == 'sigmoid':
                encoder.append(nn.Sigmoid())
            elif self.hparams.activation == 'lrelu':
                encoder.append(nn.LeakyReLU())

            self.encoder = nn.Sequential(*encoder)

            self.enc_mean = nn.Linear(X.shape[-1], X.shape[-1])
            self.enc_log_var = nn.Linear(X.shape[-1], X.shape[-1])

            Z_mean = self.enc_mean(X)
            Z_log_var = self.enc_log_var(X)

            decoder = []

            decoder.append(nn.Linear(in_features=Z_mean.shape[-1],
                                     out_features=self.hparams.z_dim))

            X = decoder[-1](Z_mean)

            if self.hparams.activation == 'sigmoid':
                decoder.append(nn.Sigmoid())
            elif self.hparams.activation == 'lrelu':
                decoder.append(nn.LeakyReLU())

            decoder.append(nn.Linear(in_features=X.shape[-1],
                                     out_features=X_shape_after_flatten[-1]))

            X = decoder[-1](X)

            decoder.append(Unflatten(X.shape[-1], X_shape_before_flatten[1:]))

            # breakpoint()
            X = decoder[-1](X)

            for i in range(len(self.hparams.encoder_channels)):
                if self.hparams.activation == 'sigmoid':
                    decoder.append(nn.Sigmoid())
                elif self.hparams.activation == 'lrelu':
                    decoder.append(nn.LeakyReLU())

                layer = nn.ConvTranspose2d(in_channels=X.shape[1],
                                           out_channels=self.hparams.encoder_channels[-(i+1)],
                                           kernel_size=self.hparams.encoder_kernel_size[-(i+1)],
                                           stride=self.hparams.encoder_strides[-(i+1)])

                decoder.append(layer)
                X = layer(X)
                # print(X.shape)

            if self.hparams.activation == 'sigmoid':
                decoder.append(nn.Sigmoid())
            elif self.hparams.activation == 'lrelu':
                decoder.append(nn.LeakyReLU())

            decoder.append(nn.Flatten())
            X = decoder[-1](X)

            decoder.append(nn.Linear(X.shape[-1], X_in.shape[-1]))
            X = decoder[-1](X)

            decoder.append(nn.Sigmoid())

            self.decoder = nn.Sequential(*decoder)

    def encode(self, X, M=None):
        Z_prime = self.encoder(X)
        Z_mean = self.enc_mean(Z_prime)
        Z_log_var = self.enc_log_var(Z_prime)

        return Z_mean, Z_log_var

    def decode(self, Z):
        X_tilde = self.decoder(Z)
        return X_tilde

    def vae_bound(self, X, X_tilde_prob, Z_mean, Z_log_var, M):
        X_bin = distr.Binomial(probs=X_tilde_prob)

        log_prob = X_bin.log_prob(X)
        if ((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
                or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val)):
            log_prob = log_prob * M
        log_prob = log_prob.sum(dim=-1).mean(dim=0)

        # Instead of computing entropy and log-prob of Z,
        # Compute analytical -KL(q(z|x) || p(z)) term here
        KL_neg = (1/2 * (1 + Z_log_var - torch.exp(Z_log_var) - Z_mean**2)).sum(dim=-1)

        # Return lower-bound on the marginal log-probability
        return log_prob + KL_neg

    def miwae_bound(self, X, X_tilde_prob, Z, Z_mean, Z_log_var, M):
        X_bin = distr.Binomial(probs=X_tilde_prob)
        X_log_prob = X_bin.log_prob(X) * M
        X_log_prob = X_log_prob.sum(dim=-1)

        # Z prior is standard normal
        Z_log_prob = (-1/2*torch.log(torch.tensor(2*np.pi, device=X.device)) - 1/2 * Z**2).sum(dim=-1)

        # entropy term of Z posterior
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z_neg_ent = Z_norm.log_prob(Z).sum(dim=-1)

        return (torch.logsumexp(X_log_prob + Z_log_prob - Z_neg_ent, dim=0)
                - torch.log(torch.tensor(Z.shape[0], dtype=torch.float, device=X.device)))

    def forward(self, X, M, I=None):
        # if self.training:
        if self.hparams.bound == 'miwae' or (hasattr(self.hparams, 'mask_mis_with_zero') and self.hparams.mask_mis_with_zero):
            # Set missing inputs to zero
            X = X * M

        Z_mean, Z_log_var = self.encode(X, M)

        # Sample latent variables
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z = Z_norm.rsample(sample_shape=(self.hparams.num_z_samples,))
        Z = Z.reshape(-1, Z.shape[-1])

        X_tilde_prob = self.decode(Z)
        X_tilde_prob = X_tilde_prob.reshape(self.hparams.num_z_samples, *X.shape)

        if not hasattr(self.hparams, 'bound') or self.hparams.bound == 'vae':  # BC
            return self.vae_bound(X, X_tilde_prob, Z_mean, Z_log_var, M)
        elif self.hparams.bound == 'miwae':
            Z = Z.reshape(self.hparams.num_z_samples, -1, Z.shape[-1])
            return self.miwae_bound(X, X_tilde_prob, Z, Z_mean, Z_log_var, M)
        else:
            raise Exception('Invalid bound!')

    def reset_parameters(self):
        self.reset_encoder()
        self.reset_decoder()

    def reset_encoder(self):
        for layer in self.encoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, Unflatten, nn.Flatten)):
                layer.reset_parameters()
        self.enc_mean.reset_parameters()
        self.enc_log_var.reset_parameters()

    def reset_encoder_first_layer(self):
        self.encoder[0].reset_parameters()

    def reset_decoder(self):
        for layer in self.decoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid, Unflatten, nn.Flatten)):
                layer.reset_parameters()

        # self.dec_mean.reset_parameters()
        # self.dec_log_var.reset_parameters()
        # self.dec_final.reset_parameters()

    def freeze_decoder(self):
        for layer in self.decoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.requires_grad_(False)
        # self.dec_mean.requires_grad_(False)
        # self.dec_log_var.requires_grad_(False)
        # self.dec_final.requires_grad_(False)
