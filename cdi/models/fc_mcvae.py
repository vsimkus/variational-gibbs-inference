import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as distr
import torch.nn as nn

from cdi.util.arg_utils import parse_bool


class FC_MCVAE(pl.LightningModule):
    """
    Fully-connected VAE
    """
    def __init__(self, args):
        super(FC_MCVAE, self).__init__()
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
                            type=str, default='vae',
                            choices=['vae', 'miwae'],
                            help='Which bound to use.')
        parser.add_argument('--fc_vae_model.marginalise',
                            type=parse_bool, required=False,
                            help=('For VAE bound, marginalise the generator or not. (train only)'))
        parser.add_argument('--fc_vae_model.marginalise_val',
                            type=parse_bool, required=False,
                            help=('For VAE bound, marginalise the generator or not. (val only)'))

        parser.add_argument('--fc_vae_model.sample_imps', type=parse_bool, default=False)
        parser.add_argument('--fc_vae_model.detach_imps', type=parse_bool, default=True)
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
        if (((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
                    or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val))
                or not self.hparams.sample_imps):
            X_imputed = X
        elif self.hparams.sample_imps:
            X_imputed = X_norm.rsample()
            if self.hparams.detach_imps:
                X_imputed = X_imputed.detach()  # Detach imputations
            X_imputed[:, M] = X[M]
        else:
            raise Exception('Invalid options!')

        log_prob = X_norm.log_prob(X_imputed)
        if ((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
                or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val)):
            log_prob = log_prob * M
        log_prob = log_prob.sum(dim=-1).mean(dim=0)

        # Compute the imputation entropy and also detach
        if (((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
                    or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val))
                or not self.hparams.sample_imps):
            imp_ent = 0.
        elif self.hparams.sample_imps:
            imp_ent = torch.tensor(1/2 * np.log(2*np.pi) + 1/2, device=X.device) + 1/2*X_tilde_log_var
            if self.hparams.detach_imps:
                imp_ent = imp_ent.detach()
            imp_ent = imp_ent * ~M
            imp_ent = imp_ent.sum(dim=-1).mean(dim=0)

        # Instead of computing entropy and log-prob of Z,
        # Compute analytical -KL(q(z|x) || p(z)) term here
        KL_neg = (1/2 * (1 + Z_log_var - torch.exp(Z_log_var) - Z_mean**2)).sum(dim=-1)

        # Return lower-bound on the marginal log-probability
        # print(imp_ent.mean())
        return log_prob + KL_neg + imp_ent

    def miwae_bound(self, X, X_tilde_mean, X_tilde_log_var, Z, Z_mean, Z_log_var, M):
        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))
        if (((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
                    or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val))
                or not self.hparams.sample_imps):
            X_imputed = X
        elif self.hparams.sample_imps:
            X_imputed = X_norm.rsample()
            if self.hparams.detach_imps:
                X_imputed = X_imputed.detach()  # Detach imputations
            X_imputed[:, M] = X[M]
        else:
            raise Exception('Invalid options!')

        X_log_prob = X_norm.log_prob(X_imputed)
        if ((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
                or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val)):
            X_log_prob = X_log_prob * M
        X_log_prob = X_log_prob.sum(dim=-1).mean(dim=0)

        # Compute the imputation entropy and also detach
        if (((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
                    or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val))
                or not self.hparams.sample_imps):
            imp_ent = 0.
        elif self.hparams.sample_imps:
            imp_ent = torch.tensor(1/2 * np.log(2*np.pi) + 1/2, device=X.device) + 1/2*X_tilde_log_var
            if self.hparams.detach_imps:
                imp_ent = imp_ent.detach()
            imp_ent = imp_ent * ~M
            imp_ent = imp_ent.sum(dim=-1).mean(dim=0)

        # Z prior is standard normal
        Z_log_prob = (-1/2*torch.log(torch.tensor(2*np.pi, device=X.device)) - 1/2 * Z**2).sum(dim=-1)

        # entropy term of Z posterior
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z_neg_ent = Z_norm.log_prob(Z).sum(dim=-1)

        return (torch.logsumexp(X_log_prob + Z_log_prob - Z_neg_ent + imp_ent, dim=0)
                - torch.log(torch.tensor(Z.shape[0], dtype=torch.float, device=X.device)))

    def forward(self, X, M, I=None):
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

    def test_loglik(self, X, M, num_z_samples):
        # self.hparams.bound = 'miwae'
        # self.hparams.num_z_samples = num_z_samples

        # log_probs = self.forward(X, M)

        # return log_probs

        Z_mean, Z_log_var = self.encode(X, M)

        # Sample latent variables
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z = Z_norm.rsample(sample_shape=(num_z_samples,))
        Z = Z.reshape(-1, Z.shape[-1])

        X_tilde_mean, X_tilde_log_var = self.decode(Z)
        X_tilde_mean = X_tilde_mean.reshape(num_z_samples, *X.shape)
        X_tilde_log_var = X_tilde_log_var.reshape(num_z_samples, *X.shape)

        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))
        X_log_prob = X_norm.log_prob(X)  # * M
        X_log_prob = X_log_prob.sum(dim=-1)

        # Z prior is standard normal
        Z = Z.reshape(num_z_samples, -1, Z.shape[-1])
        Z_log_prob = (-1/2*torch.log(torch.tensor(2*np.pi, device=X.device)) - 1/2 * Z**2).sum(dim=-1)

        # entropy term of Z posterior
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z_neg_ent = Z_norm.log_prob(Z).sum(dim=-1)

        return (torch.logsumexp(X_log_prob + Z_log_prob - Z_neg_ent, dim=0)
                - torch.log(torch.tensor(Z.shape[0], dtype=torch.float, device=X.device)))

    # def test_z_posterior(self, X, M):
    #     if self.hparams.bound == 'miwae' or (hasattr(self.hparams, 'mask_mis_with_zero') and self.hparams.mask_mis_with_zero):
    #         # Set missing inputs to zero
    #         X = X * M

    #     if hasattr(self.hparams, 'local_vi') and self.hparams.local_vi:
    #         Z_mean, Z_log_var = self.encode_local(I)
    #     else:
    #         Z_mean, Z_log_var = self.encode(X, M)

    #     return Z_mean, Z_log_var


class FC_MCVAE_2G(pl.LightningModule):
    """
    Fully-connected VAE
    """
    def __init__(self, args):
        super(FC_MCVAE_2G, self).__init__()
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
        # parser.add_argument('--fc_vae_model.bound',
        #                     type=str, required=True,
        #                     choices=['vae', 'miwae'],
        #                     help='Which bound to use.')
        parser.add_argument('--fc_vae_model.marginalise',
                            type=parse_bool, required=False,
                            help=('For VAE bound, marginalise the generator or not. (train only)'))
        parser.add_argument('--fc_vae_model.marginalise_val',
                            type=parse_bool, required=False,
                            help=('For VAE bound, marginalise the generator or not. (val only)'))
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

        decoder2 = []
        for i in range(len(hidden_dims)-2):
            decoder2.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

            if self.hparams.activation == 'sigmoid':
                decoder2.append(nn.Sigmoid())
            elif self.hparams.activation == 'lrelu':
                decoder2.append(nn.LeakyReLU())

        self.decoder2 = nn.Sequential(*decoder2)
        self.dec_mean2 = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        self.dec_log_var2 = nn.Linear(hidden_dims[-2], hidden_dims[-1])

    def encode(self, X, M=None):
        Z_prime = self.encoder(X)
        Z_mean = self.enc_mean(Z_prime)
        Z_log_var = self.enc_log_var(Z_prime)

        return Z_mean, Z_log_var

    def decode_mis(self, Z):
        X_tilde = self.decoder2(Z)
        X_tilde_mean = self.dec_mean2(X_tilde)
        X_tilde_log_var = self.dec_log_var2(X_tilde)

        return X_tilde_mean, X_tilde_log_var

    def decode(self, Z):
        X_tilde = self.decoder(Z)
        X_tilde_mean = self.dec_mean(X_tilde)
        X_tilde_log_var = self.dec_log_var(X_tilde)

        return X_tilde_mean, X_tilde_log_var

    def vae_bound(self, X, X_tilde_mean, X_tilde_log_var, X_tilde_mean2, X_tilde_log_var2, Z_mean, Z_log_var, M):
        X_norm2 = distr.Normal(loc=X_tilde_mean2, scale=torch.exp(X_tilde_log_var2/2))
        X_imputed = X_norm2.rsample()
        X_imputed[:, M] = X[M]
        # M_not = ~M

        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))
        log_prob = X_norm.log_prob(X_imputed)
        if ((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
                or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val)):
            log_prob = log_prob * M
        log_prob = log_prob.sum(dim=-1).mean(dim=0)

        # Compute the imputation entropy
        imp_ent = torch.tensor(1/2 * np.log(2*np.pi) + 1/2, device=X.device) + 1/2*X_tilde_log_var2
        imp_ent = imp_ent * ~M
        imp_ent = imp_ent.sum(dim=-1).mean(dim=0)

        # Instead of computing entropy and log-prob of Z,
        # Compute analytical -KL(q(z|x) || p(z)) term here
        KL_neg = (1/2 * (1 + Z_log_var - torch.exp(Z_log_var) - Z_mean**2)).sum(dim=-1)

        # Return lower-bound on the marginal log-probability
        # print(imp_ent.mean())
        return log_prob + KL_neg + imp_ent

    def forward(self, X, M, I=None):
        Z_mean, Z_log_var = self.encode(X, M)

        # Sample latent variables
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z = Z_norm.rsample(sample_shape=(self.hparams.num_z_samples,))
        Z = Z.reshape(-1, Z.shape[-1])

        X_tilde_mean, X_tilde_log_var = self.decode(Z)
        X_tilde_mean = X_tilde_mean.reshape(self.hparams.num_z_samples, *X.shape)
        X_tilde_log_var = X_tilde_log_var.reshape(self.hparams.num_z_samples, *X.shape)

        X_tilde_mean2, X_tilde_log_var2 = self.decode_mis(Z)
        X_tilde_mean2 = X_tilde_mean2.reshape(self.hparams.num_z_samples, *X.shape)
        X_tilde_log_var2 = X_tilde_log_var2.reshape(self.hparams.num_z_samples, *X.shape)

        return self.vae_bound(X, X_tilde_mean, X_tilde_log_var, X_tilde_mean2, X_tilde_log_var2, Z_mean, Z_log_var, M)

    def mcmc_sample_missing_values(self, X, M):
        Z_mean, Z_log_var = self.encode(X, M)
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z = Z_norm.rsample()

        X_tilde_mean, X_tilde_log_var = self.decode_mis(Z)
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

        for layer in self.decoder2:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.reset_parameters()
        self.dec_mean2.reset_parameters()
        self.dec_log_var2.reset_parameters()

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

    def test_loglik(self, X, M, num_z_samples):
        # self.hparams.bound = 'miwae'
        # self.hparams.num_z_samples = num_z_samples

        # log_probs = self.forward(X, M)

        # return log_probs

        Z_mean, Z_log_var = self.encode(X, M)

        # Sample latent variables
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z = Z_norm.rsample(sample_shape=(num_z_samples,))
        Z = Z.reshape(-1, Z.shape[-1])

        X_tilde_mean, X_tilde_log_var = self.decode(Z)
        X_tilde_mean = X_tilde_mean.reshape(num_z_samples, *X.shape)
        X_tilde_log_var = X_tilde_log_var.reshape(num_z_samples, *X.shape)

        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))
        X_log_prob = X_norm.log_prob(X)  # * M
        X_log_prob = X_log_prob.sum(dim=-1)

        # Z prior is standard normal
        Z = Z.reshape(num_z_samples, -1, Z.shape[-1])
        Z_log_prob = (-1/2*torch.log(torch.tensor(2*np.pi, device=X.device)) - 1/2 * Z**2).sum(dim=-1)

        # entropy term of Z posterior
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z_neg_ent = Z_norm.log_prob(Z).sum(dim=-1)

        return (torch.logsumexp(X_log_prob + Z_log_prob - Z_neg_ent, dim=0)
                - torch.log(torch.tensor(Z.shape[0], dtype=torch.float, device=X.device)))

    # def test_z_posterior(self, X, M):
    #     if self.hparams.bound == 'miwae' or (hasattr(self.hparams, 'mask_mis_with_zero') and self.hparams.mask_mis_with_zero):
    #         # Set missing inputs to zero
    #         X = X * M

    #     if hasattr(self.hparams, 'local_vi') and self.hparams.local_vi:
    #         Z_mean, Z_log_var = self.encode_local(I)
    #     else:
    #         Z_mean, Z_log_var = self.encode(X, M)

    #     return Z_mean, Z_log_var


class FC_MCVAE_BVGI(pl.LightningModule):
    """
    Fully-connected VAE for block-gibbs VGI
    """
    def __init__(self, args):
        super(FC_MCVAE_BVGI, self).__init__()
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
        parser.add_argument('--fc_vae_model.marginalise',
                            type=parse_bool, required=False,
                            help=('For VAE bound, marginalise the generator or not. (train only)'))
        parser.add_argument('--fc_vae_model.marginalise_val',
                            type=parse_bool, required=False,
                            help=('For VAE bound, marginalise the generator or not. (val only)'))

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

    def standard_elbo(self, X, M):
        Z_mean, Z_log_var = self.encode(X, M)

        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z = Z_norm.rsample(sample_shape=(self.hparams.num_z_samples,))
        Z = Z.reshape(-1, Z.shape[-1])

        X_tilde_mean, X_tilde_log_var = self.decode(Z)
        X_tilde_mean = X_tilde_mean.reshape(self.hparams.num_z_samples, *X.shape)
        X_tilde_log_var = X_tilde_log_var.reshape(self.hparams.num_z_samples, *X.shape)

        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))
        # if ((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
        #         or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val)):
        #     X_imputed = X
        # else:
        #     X_imputed = X_norm.rsample()
        #     X_imputed = X_imputed.detach()  # Detach imputations
        #     X_imputed[:, M] = X[M]

        log_prob = X_norm.log_prob(X)
        # if ((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
        #         or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val)):
        #     log_prob = log_prob * M
        log_prob = log_prob * M
        log_prob = log_prob.sum(dim=-1).mean(dim=0)

        # Compute the imputation entropy and also detach
        # if (((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
        #             or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val))
        #         or not self.hparams.sample_imps):
        #     imp_ent = 0.
        # elif self.hparams.sample_imps:
        #     imp_ent = torch.tensor(1/2 * np.log(2*np.pi) + 1/2, device=X.device) + 1/2*X_tilde_log_var
        #     if self.hparams.detach_imps:
        #         imp_ent = imp_ent.detach()
        #     imp_ent = imp_ent * ~M
        #     imp_ent = imp_ent.sum(dim=-1).mean(dim=0)

        # Instead of computing entropy and log-prob of Z,
        # Compute analytical -KL(q(z|x) || p(z)) term here
        KL_neg = (1/2 * (1 + Z_log_var - torch.exp(Z_log_var) - Z_mean**2)).sum(dim=-1)

        # Return lower-bound on the marginal log-probability
        # print(imp_ent.mean())
        return log_prob + KL_neg  #+ imp_ent

    def forward(self, X, M, I=None):
        if X.shape[-1] == self.hparams.input_dim:  # and not self.training and :
            return self.standard_elbo(X, M)

        # Split X to observables and latents
        X, Z = X[:, :self.hparams.input_dim], X[:, self.hparams.input_dim:]
        M = M[:, :self.hparams.input_dim]

        Z_mean, Z_log_var = self.encode(X, M)

        # Sample latent variables
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z_updated = Z_norm.rsample(sample_shape=(self.hparams.num_z_samples,))
        Z_updated = Z_updated.reshape(-1, Z_updated.shape[-1])

        # Decode updated latents
        X_tilde_mean_up, X_tilde_log_var_up = self.decode(Z_updated)
        X_tilde_mean_up = X_tilde_mean_up.reshape(self.hparams.num_z_samples, *X.shape)
        X_tilde_log_var_up = X_tilde_log_var_up.reshape(self.hparams.num_z_samples, *X.shape)

        # Decode previous latents
        X_tilde_mean_prev, X_tilde_log_var_prev = self.decode(Z)

        # Sample updated imputations
        X_norm = distr.Normal(loc=X_tilde_mean_prev, scale=torch.exp(X_tilde_log_var_prev/2))
        if ((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
                or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val)):
            X_updated = X
        else:
            X_updated = X_norm.rsample(sample_shape=(self.hparams.num_z_samples,))
            # Detach graph for the imputations
            X_updated = X_updated.detach()
            X_updated[:, M] = X[M]
            # X_updated = X_updated.reshape(-1, X_updated.shape[-1])

        # Computing VGI objective

        # Updated Z's, prev X's
        X_norm_up = distr.Normal(loc=X_tilde_mean_up, scale=torch.exp(X_tilde_log_var_up/2))
        X_prev_log_prob = X_norm_up.log_prob(X)  # TODO: add marginalisation
        X_prev_log_prob = X_prev_log_prob.sum(dim=-1)
        # Compute analytical -KL(q(z|x) || p(z)) term here
        KL_neg = (1/2 * (1 + Z_log_var - torch.exp(Z_log_var) - Z_mean**2)).sum(dim=-1)

        VGI1 = (X_prev_log_prob + KL_neg).mean(dim=0)

        # Updated X's, prev Z's
        X_up_log_prob = X_norm.log_prob(X_updated)  # TODO: add marginalisation
        X_up_log_prob = X_up_log_prob.sum(dim=-1)
        # Compute log_prob of z's log p(z)
        Z_log_prob = (-1/2*torch.log(torch.tensor(2*np.pi, device=X.device)) - 1/2 * Z**2).sum(dim=-1)
        # Compute entropy of imputation distribution p(xm)
        imp_ent = torch.tensor(1/2 * np.log(2*np.pi) + 1/2, device=X.device) + 1/2*X_tilde_log_var_prev
        # Detach graph for the imputation entropy
        imp_ent = imp_ent.detach()
        imp_ent = imp_ent * ~M
        imp_ent = imp_ent.sum(dim=-1)

        VGI2 = (X_up_log_prob + Z_log_prob + imp_ent).mean(dim=0)

        return 1/2 * (VGI1 + VGI2)

    def mcmc_sample_missing_values(self, X, M):
        X, _ = X[:, :self.hparams.input_dim], X[:, self.hparams.input_dim:]

        Z_mean, Z_log_var = self.encode(X, M)
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z = Z_norm.rsample()

        X_tilde_mean, X_tilde_log_var = self.decode(Z)
        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))
        X_tilde = X_norm.rsample()

        return torch.cat([X_tilde, Z], dim=-1)

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

    def test_loglik(self, X, M, num_z_samples):
        # self.hparams.bound = 'miwae'
        # self.hparams.num_z_samples = num_z_samples

        # log_probs = self.forward(X, M)

        # return log_probs

        Z_mean, Z_log_var = self.encode(X, M)

        # Sample latent variables
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z = Z_norm.rsample(sample_shape=(num_z_samples,))
        Z = Z.reshape(-1, Z.shape[-1])

        X_tilde_mean, X_tilde_log_var = self.decode(Z)
        X_tilde_mean = X_tilde_mean.reshape(num_z_samples, *X.shape)
        X_tilde_log_var = X_tilde_log_var.reshape(num_z_samples, *X.shape)

        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))
        X_log_prob = X_norm.log_prob(X)  # * M
        X_log_prob = X_log_prob.sum(dim=-1)

        # Z prior is standard normal
        Z = Z.reshape(num_z_samples, -1, Z.shape[-1])
        Z_log_prob = (-1/2*torch.log(torch.tensor(2*np.pi, device=X.device)) - 1/2 * Z**2).sum(dim=-1)

        # entropy term of Z posterior
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z_neg_ent = Z_norm.log_prob(Z).sum(dim=-1)

        return (torch.logsumexp(X_log_prob + Z_log_prob - Z_neg_ent, dim=0)
                - torch.log(torch.tensor(Z.shape[0], dtype=torch.float, device=X.device)))

    # def test_z_posterior(self, X, M):
    #     if self.hparams.bound == 'miwae' or (hasattr(self.hparams, 'mask_mis_with_zero') and self.hparams.mask_mis_with_zero):
    #         # Set missing inputs to zero
    #         X = X * M

    #     if hasattr(self.hparams, 'local_vi') and self.hparams.local_vi:
    #         Z_mean, Z_log_var = self.encode_local(I)
    #     else:
    #         Z_mean, Z_log_var = self.encode(X, M)

    #     return Z_mean, Z_log_var


class FC_MCVAE_BVGI_2G(pl.LightningModule):
    """
    Fully-connected VAE, separate imputation generator
    """
    def __init__(self, args):
        super(FC_MCVAE_BVGI_2G, self).__init__()
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
        parser.add_argument('--fc_vae_model.var_decoder_hidden_dims', type=int,
                            nargs='+', required=False,
                            help='Variational decoder layer dimensionalities, defaults to same as the generative decoder')
        parser.add_argument('--fc_vae_model.activation',
                            type=str, required=True,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid'])
        parser.add_argument('--fc_vae_model.var_dec_activation',
                            type=str, required=False,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid'])
        parser.add_argument('--fc_vae_model.marginalise',
                            type=parse_bool, required=False,
                            help=('For VAE bound, marginalise the generator or not. (train only)'))
        parser.add_argument('--fc_vae_model.marginalise_val',
                            type=parse_bool, required=False,
                            help=('For VAE bound, marginalise the generator or not. (val only)'))

        parser.add_argument('--fc_vae_model.var_decoder_concat_x', type=parse_bool,
                            required=False, help=('Whether to concatenate z and x in the input to the variational decoder.'))
        parser.add_argument('--fc_vae_model.var_decoder_mask_xm', type=parse_bool,
                            required=False, help=('Whether to mask xm with zeros the input to the variational decoder.'))

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

        if hasattr(self.hparams, 'var_decoder_hidden_dims') and self.hparams.var_decoder_hidden_dims is not None:
            hidden_dims = self.hparams.var_decoder_hidden_dims + [self.hparams.input_dim]

        if hasattr(self.hparams, 'var_dec_activation') and self.hparams.var_dec_activation is not None:
            activation = self.hparams.var_dec_activation
        else:
            activation = self.hparams.activation

        decoder2 = []
        for i in range(len(hidden_dims)-2):
            decoder2.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

            if activation == 'sigmoid':
                decoder2.append(nn.Sigmoid())
            elif activation == 'lrelu':
                decoder2.append(nn.LeakyReLU())

        self.decoder2 = nn.Sequential(*decoder2)
        self.dec_mean2 = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        self.dec_log_var2 = nn.Linear(hidden_dims[-2], hidden_dims[-1])

    def encode(self, X, M=None):
        Z_prime = self.encoder(X)
        Z_mean = self.enc_mean(Z_prime)
        Z_log_var = self.enc_log_var(Z_prime)

        return Z_mean, Z_log_var

    def decode_mis(self, Z):
        X_tilde = self.decoder2(Z)
        X_tilde_mean = self.dec_mean2(X_tilde)
        X_tilde_log_var = self.dec_log_var2(X_tilde)

        return X_tilde_mean, X_tilde_log_var

    def decode(self, Z):
        X_tilde = self.decoder(Z)
        X_tilde_mean = self.dec_mean(X_tilde)
        X_tilde_log_var = self.dec_log_var(X_tilde)

        return X_tilde_mean, X_tilde_log_var

    def standard_elbo(self, X, M):
        Z_mean, Z_log_var = self.encode(X, M)

        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z = Z_norm.rsample(sample_shape=(self.hparams.num_z_samples,))
        Z = Z.reshape(-1, Z.shape[-1])

        X_tilde_mean, X_tilde_log_var = self.decode(Z)
        X_tilde_mean = X_tilde_mean.reshape(self.hparams.num_z_samples, *X.shape)
        X_tilde_log_var = X_tilde_log_var.reshape(self.hparams.num_z_samples, *X.shape)

        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))
        # if ((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
        #         or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val)):
        #     X_imputed = X
        # else:
        #     X_imputed = X_norm.rsample()
        #     X_imputed = X_imputed.detach()  # Detach imputations
        #     X_imputed[:, M] = X[M]

        log_prob = X_norm.log_prob(X)
        # if ((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
        #         or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val)):
        #     log_prob = log_prob * M
        log_prob = log_prob * M
        log_prob = log_prob.sum(dim=-1).mean(dim=0)

        # Compute the imputation entropy and also detach
        # if (((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
        #             or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val))
        #         or not self.hparams.sample_imps):
        #     imp_ent = 0.
        # elif self.hparams.sample_imps:
        #     imp_ent = torch.tensor(1/2 * np.log(2*np.pi) + 1/2, device=X.device) + 1/2*X_tilde_log_var
        #     if self.hparams.detach_imps:
        #         imp_ent = imp_ent.detach()
        #     imp_ent = imp_ent * ~M
        #     imp_ent = imp_ent.sum(dim=-1).mean(dim=0)

        # Instead of computing entropy and log-prob of Z,
        # Compute analytical -KL(q(z|x) || p(z)) term here
        KL_neg = (1/2 * (1 + Z_log_var - torch.exp(Z_log_var) - Z_mean**2)).sum(dim=-1)

        # Return lower-bound on the marginal log-probability
        # print(imp_ent.mean())
        return log_prob + KL_neg  #+ imp_ent

    def forward(self, X, M, I=None):
        if X.shape[-1] == self.hparams.input_dim:  # and not self.training and :
            return self.standard_elbo(X, M)

        # Split X to observables and latents
        X, Z = X[:, :self.hparams.input_dim], X[:, self.hparams.input_dim:]
        M = M[:, :self.hparams.input_dim]

        Z_mean, Z_log_var = self.encode(X, M)

        # Sample latent variables
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z_updated = Z_norm.rsample(sample_shape=(self.hparams.num_z_samples,))
        Z_updated = Z_updated.reshape(-1, Z_updated.shape[-1])

        # Decode updated latents
        X_tilde_mean_up, X_tilde_log_var_up = self.decode(Z_updated)
        X_tilde_mean_up = X_tilde_mean_up.reshape(self.hparams.num_z_samples, *X.shape)
        X_tilde_log_var_up = X_tilde_log_var_up.reshape(self.hparams.num_z_samples, *X.shape)

        # Decode previous latents
        X_tilde_mean_prev, X_tilde_log_var_prev = self.decode(Z)

        # Decode with variational decoder
        XZ = Z
        if hasattr(self.hparams, 'var_decoder_concat_x') and self.hparams.var_decoder_concat_x:
            X_dec = X
            if hasattr(self.hparams, 'var_decoder_mask_xm') and self.hparams.var_decoder_mask_xm:
                X_dec = X_dec * M
            XZ = torch.cat([X_dec, Z], dim=-1)

        X_tilde_mean_prev_var, X_tilde_log_var_prev_var = self.decode_mis(XZ)

        # Sample updated imputations
        X_var = distr.Normal(loc=X_tilde_mean_prev_var, scale=torch.exp(X_tilde_log_var_prev_var/2))
        if ((self.training and hasattr(self.hparams, 'marginalise') and self.hparams.marginalise)
                or (not self.training and hasattr(self.hparams, 'marginalise_val') and self.hparams.marginalise_val)):
            X_updated = X
        else:
            X_updated = X_var.rsample(sample_shape=(self.hparams.num_z_samples,))
            X_updated[:, M] = X[M]

        # Computing VGI objective

        # Updated Z's, prev X's
        X_norm_up = distr.Normal(loc=X_tilde_mean_up, scale=torch.exp(X_tilde_log_var_up/2))
        X_prev_log_prob = X_norm_up.log_prob(X)  # TODO: add marginalisation
        X_prev_log_prob = X_prev_log_prob.sum(dim=-1)
        # Compute analytical -KL(q(z|x) || p(z)) term here
        KL_neg = (1/2 * (1 + Z_log_var - torch.exp(Z_log_var) - Z_mean**2)).sum(dim=-1)

        VGI1 = (X_prev_log_prob + KL_neg).mean(dim=0)

        # Updated X's, prev Z's
        X_norm = distr.Normal(loc=X_tilde_mean_prev, scale=torch.exp(X_tilde_log_var_prev/2))
        X_up_log_prob = X_norm.log_prob(X_updated)  # TODO: add marginalisation
        X_up_log_prob = X_up_log_prob.sum(dim=-1)
        # Compute log_prob of z's log p(z)
        Z_log_prob = (-1/2*torch.log(torch.tensor(2*np.pi, device=X.device)) - 1/2 * Z**2).sum(dim=-1)
        # Compute entropy of imputation distribution p(xm)
        imp_ent = torch.tensor(1/2 * np.log(2*np.pi) + 1/2, device=X.device) + 1/2*X_tilde_log_var_prev_var
        imp_ent = imp_ent * ~M
        imp_ent = imp_ent.sum(dim=-1)

        VGI2 = (X_up_log_prob + Z_log_prob + imp_ent).mean(dim=0)

        return 1/2 * (VGI1 + VGI2)

    def mcmc_sample_missing_values(self, X, M):
        X, _ = X[:, :self.hparams.input_dim], X[:, self.hparams.input_dim:]
        M = M[:, :self.hparams.input_dim]

        Z_mean, Z_log_var = self.encode(X, M)
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z = Z_norm.rsample()

        # Decode with variational decoder
        XZ = Z
        if hasattr(self.hparams, 'var_decoder_concat_x') and self.hparams.var_decoder_concat_x:
            X_dec = X
            if hasattr(self.hparams, 'var_decoder_mask_xm') and self.hparams.var_decoder_mask_xm:
                X_dec = X_dec * M
            XZ = torch.cat([X_dec, Z], dim=-1)
        X_tilde_mean, X_tilde_log_var = self.decode_mis(XZ)

        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))
        X_tilde = X_norm.rsample()

        return torch.cat([X_tilde, Z], dim=-1)

    def reset_parameters(self):
        self.reset_encoder()
        self.reset_decoder()

    def reset_encoder(self):
        for layer in self.encoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.reset_parameters()
        self.enc_mean.reset_parameters()
        self.enc_log_var.reset_parameters()

        for layer in self.decoder2:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.reset_parameters()
        self.dec_mean2.reset_parameters()
        self.dec_log_var2.reset_parameters()

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

    def test_loglik(self, X, M, num_z_samples):
        # self.hparams.bound = 'miwae'
        # self.hparams.num_z_samples = num_z_samples

        # log_probs = self.forward(X, M)

        # return log_probs

        Z_mean, Z_log_var = self.encode(X, M)

        # Sample latent variables
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z = Z_norm.rsample(sample_shape=(num_z_samples,))
        Z = Z.reshape(-1, Z.shape[-1])

        X_tilde_mean, X_tilde_log_var = self.decode(Z)
        X_tilde_mean = X_tilde_mean.reshape(num_z_samples, *X.shape)
        X_tilde_log_var = X_tilde_log_var.reshape(num_z_samples, *X.shape)

        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))
        X_log_prob = X_norm.log_prob(X)  # * M
        X_log_prob = X_log_prob.sum(dim=-1)

        # Z prior is standard normal
        Z = Z.reshape(num_z_samples, -1, Z.shape[-1])
        Z_log_prob = (-1/2*torch.log(torch.tensor(2*np.pi, device=X.device)) - 1/2 * Z**2).sum(dim=-1)

        # entropy term of Z posterior
        Z_norm = distr.Normal(loc=Z_mean, scale=torch.exp(Z_log_var/2))
        Z_neg_ent = Z_norm.log_prob(Z).sum(dim=-1)

        return (torch.logsumexp(X_log_prob + Z_log_prob - Z_neg_ent, dim=0)
                - torch.log(torch.tensor(Z.shape[0], dtype=torch.float, device=X.device)))

    # def test_z_posterior(self, X, M):
    #     if self.hparams.bound == 'miwae' or (hasattr(self.hparams, 'mask_mis_with_zero') and self.hparams.mask_mis_with_zero):
    #         # Set missing inputs to zero
    #         X = X * M

    #     if hasattr(self.hparams, 'local_vi') and self.hparams.local_vi:
    #         Z_mean, Z_log_var = self.encode_local(I)
    #     else:
    #         Z_mean, Z_log_var = self.encode(X, M)

    #     return Z_mean, Z_log_var
