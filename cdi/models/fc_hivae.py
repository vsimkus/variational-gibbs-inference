import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as distr
import torch.nn as nn
import torch.nn.functional as F

from cdi.util.arg_utils import parse_bool


class FC_HIVAE(pl.LightningModule):
    """
    Fully-connected HIVAE
    Nazabal, et. al. (2018) "Handling Incomplete Heterogeneous Data using VAEs"
    """
    def __init__(self, args):
        super(FC_HIVAE, self).__init__()
        self.hparams = args.fc_hivae_model

        self.initialise()
        self.cum_batch_size_called = 0

    def set_hparams(self, hparams):
        self.hparams = hparams.fc_hivae_model

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--fc_hivae_model.num_prior_components', type=int,
                            required=True, help=('Number of prior components.'))
        parser.add_argument('--fc_hivae_model.num_z_samples', type=int,
                            default=1, help='Number of latent samples.')
        parser.add_argument('--fc_hivae_model.input_dim', type=int,
                            required=True, help='Image dimension.')
        parser.add_argument('--fc_hivae_model.s_encoder_hidden_dims', type=int,
                            nargs='+', required=True,
                            help='Encoder layer dimensionalities for s')
        parser.add_argument('--fc_hivae_model.s_tau_param', type=float,
                            required=True, help='Gumbel-Softmax temperature.')
        parser.add_argument('--fc_hivae_model.z_encoder_hidden_dims', type=int,
                            nargs='+', required=True,
                            help='Encoder layer dimensionalities for z')
        parser.add_argument('--fc_hivae_model.shared_decoder_hidden_dims', type=int,
                            nargs='+', required=True,
                            help='Shared decoder layer dimensionalities')
        parser.add_argument('--fc_hivae_model.individual_decoder_hidden_dims', type=int,
                            nargs='+', required=True,
                            help='Individual decoder layer dimensionalities')
        parser.add_argument('--fc_hivae_model.activation',
                            type=str, required=True,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid'])

        parser.add_argument('--fc_hivae_model.local_vi', type=parse_bool,
                            default=False, help=('Whether to use local VI.'))

        return parser

    def initialise(self):
        assert self.hparams.activation in ('lrelu', 'sigmoid'), \
            'Activation not supported!'

        #
        # s-encoder
        #

        # Add input dimensions to the list
        hidden_dims = [self.hparams.input_dim] + self.hparams.s_encoder_hidden_dims

        encoder = []
        for i in range(len(hidden_dims)-1):
            encoder.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

            if i < len(hidden_dims)-2:
                if self.hparams.activation == 'sigmoid':
                    encoder.append(nn.Sigmoid())
                elif self.hparams.activation == 'lrelu':
                    encoder.append(nn.LeakyReLU())

        self.s_encoder = nn.Sequential(*encoder)

        #
        # z-encoder
        #

        # Add input dimensions to the list
        hidden_dims = ([self.hparams.input_dim + self.hparams.s_encoder_hidden_dims[-1]]
                       + self.hparams.z_encoder_hidden_dims)

        encoder = []
        for i in range(len(hidden_dims)-2):
            encoder.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

            if self.hparams.activation == 'sigmoid':
                encoder.append(nn.Sigmoid())
            elif self.hparams.activation == 'lrelu':
                encoder.append(nn.LeakyReLU())

        self.z_encoder = nn.Sequential(*encoder)

        self.z_enc_mean = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        self.z_enc_log_var = nn.Linear(hidden_dims[-2], hidden_dims[-1])

        #
        # Model for p(z | s)
        #
        self.z_prior_mean = nn.Linear(self.hparams.s_encoder_hidden_dims[-1],
                                      self.hparams.shared_decoder_hidden_dims[0])

        #
        # Shared decoder part
        #

        # Add output dimensions to the list
        hidden_dims = self.hparams.shared_decoder_hidden_dims

        decoder = []
        for i in range(len(hidden_dims)-1):
            decoder.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

            if self.hparams.activation == 'sigmoid':
                decoder.append(nn.Sigmoid())
            elif self.hparams.activation == 'lrelu':
                decoder.append(nn.LeakyReLU())

        self.shared_decoder = nn.Sequential(*decoder)

        #
        # Individual decoders
        #
        # Add output dimensions to the list
        hidden_dims = [self.hparams.shared_decoder_hidden_dims[-1] + self.hparams.s_encoder_hidden_dims[-1]] + self.hparams.individual_decoder_hidden_dims
        hidden_dims += [self.hparams.input_dim]

        # decoder = []
        # for i in range(len(hidden_dims)-1):
        #     decoder.append(LinearWithChannels(hidden_dims[i],
        #                                       hidden_dims[i+1],
        #                                       channels=self.hparams.input_dim))

        #     if self.hparams.activation == 'sigmoid':
        #         decoder.append(nn.Sigmoid())
        #     elif self.hparams.activation == 'lrelu':
        #         decoder.append(nn.LeakyReLU())

        # self.ind_decoder = SequentialWithMultiInputs(*decoder)

        # self.ind_dec_mean = LinearWithChannels(hidden_dims[-1], 1,
        #                                        channels=self.hparams.input_dim)
        # self.ind_dec_log_var = LinearWithChannels(hidden_dims[-1], 1,
        #                                           channels=self.hparams.input_dim)

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

    def encode(self, XS):
        Z_prime = self.z_encoder(XS)
        Z_mean = self.z_enc_mean(Z_prime)
        Z_log_var = self.z_enc_log_var(Z_prime)

        return Z_mean, Z_log_var

    def decode(self, Z, S):
        Y = self.shared_decoder(Z)

        Y = torch.cat([Y, S], dim=-1)

        # M_channel = ~M.T
        # channel_sparse_idx = M_channel.nonzero(as_tuple=False)
        # X_tilde_prime = self.ind_decoder(Y.unsqueeze(1),
        #                                  M_channel,
        #                                  channel_sparse_idx)

        # X_tilde_mean = self.ind_dec_mean(X_tilde_prime,
        #                                  M_channel,
        #                                  channel_sparse_idx)
        # X_tilde_mean = X_tilde_mean.squeeze(2)
        # X_tilde_log_var = self.ind_dec_log_var(X_tilde_prime,
        #                                        M_channel,
        #                                        channel_sparse_idx)
        # X_tilde_log_var = X_tilde_log_var.squeeze(2)

        X_tilde_prime = self.decoder(Y)
        X_tilde_mean = self.dec_mean(X_tilde_prime)
        X_tilde_log_var = self.dec_log_var(X_tilde_prime)

        return X_tilde_mean, X_tilde_log_var

    def forward(self, X, M, I=None):
        # if self.training:
        # Set missing inputs to zero
        X = X * M

        #
        # Inference part
        #

        if hasattr(self.hparams, 'local_vi') and self.hparams.local_vi:
            S_logit_orig, S, Z_mean_q, Z_log_var_q = self.encode_local(I, self.hparams.num_z_samples)
        else:
            # Encode X to get S
            S_logit = self.s_encoder(X)
            S_logit_orig = S_logit
            # Sample num_z_samples of S
            S_logit = S_logit.unsqueeze(0).expand(self.hparams.num_z_samples, -1, -1)
            # Gumbel-softmax trick to sample S
            # NOTE: could also try hard=True instead
            S = F.gumbel_softmax(S_logit, tau=self.hparams.s_tau_param, hard=False)
            S = S.reshape(-1, S.shape[-1])

            # Encode X and S to get Z
            # XS = torch.cat([X, S], dim=-1)
            XS = torch.cat([X.unsqueeze(0).expand(self.hparams.num_z_samples, -1, -1).reshape(-1, X.shape[-1]), S], dim=-1)
            Z_mean_q, Z_log_var_q = self.encode(XS)

        # Sample latent variables Z
        Z_norm = distr.Normal(loc=Z_mean_q, scale=torch.exp(Z_log_var_q/2))
        # Z = Z_norm.rsample(sample_shape=(self.hparams.num_z_samples,))
        # Z = Z.reshape(-1, Z.shape[-1])
        Z = Z_norm.rsample()

        #
        # Generative part
        #

        # p(z | s)
        Z_mean_p = self.z_prior_mean(S)
        Z_log_var_p = torch.zeros_like(Z_mean_p)

        X_tilde_mean, X_tilde_log_var = self.decode(Z, S)

        X_tilde_mean = X_tilde_mean.reshape(self.hparams.num_z_samples, *X.shape)
        X_tilde_log_var = X_tilde_log_var.reshape(self.hparams.num_z_samples, *X.shape)

        Z_mean_q = Z_mean_q.reshape(self.hparams.num_z_samples, X.shape[0], Z_mean_q.shape[-1])
        Z_log_var_q = Z_log_var_q.reshape(self.hparams.num_z_samples, X.shape[0], Z_log_var_q.shape[-1])
        Z_mean_p = Z_mean_p.reshape(self.hparams.num_z_samples, X.shape[0], Z_mean_p.shape[-1])
        Z_log_var_p = Z_log_var_p.reshape(self.hparams.num_z_samples, X.shape[0], Z_log_var_p.shape[-1])

        # Compute cost

        # KL(q(s | x) || p(s))
        # This is just the Multinomial entropy, or Multinomial logistic loss + constant for the prior
        # KL_S = (-F.cross_entropy(S_logit, torch.softmax(S_logit, dim=-1)) + torch.log(S_logit.shape[-1])
        #         ).sum(dim=-1)
        log_pi = torch.log_softmax(S_logit_orig, dim=-1)
        KL_S = ((torch.exp(log_pi)*log_pi).sum(dim=-1)
                + torch.log(torch.tensor(S_logit_orig.shape[-1], dtype=torch.float, device=X.device)))

        # KL(q(z | x, s) || p(z | s))
        KL_Z = (Z_log_var_p - Z_log_var_q
                + (torch.exp(Z_log_var_q) + (Z_mean_q - Z_mean_p)**2)/(2*torch.exp(Z_log_var_p))
                - 1/2
                ).sum(dim=-1)
        KL_Z = KL_Z.mean(dim=0)

        # Log-likelihood
        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))
        log_prob = X_norm.log_prob(X) * M
        # if self.training:
        #     log_prob *= M
        # else:
        #     # When not training, eval on complete data
        #     pass

        log_prob = log_prob.sum(dim=-1).mean(dim=0)

        # Return lower-bound on the marginal log-probability
        return log_prob - KL_Z - KL_S

    def reset_parameters(self):
        self.reset_encoder()
        self.reset_decoder()

    def reset_encoder(self):
        for layer in self.s_encoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.reset_parameters()
        for layer in self.z_encoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.reset_parameters()
        self.z_enc_mean.reset_parameters()
        self.z_enc_log_var.reset_parameters()

    def reset_encoder_first_layer(self):
        self.s_encoder[0].reset_parameters()
        self.z_encoder[0].reset_parameters()

    def reset_decoder(self):
        self.z_prior_mean.reset_parameters()

        for layer in self.shared_decoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.reset_parameters()

        # for layer in self.ind_decoder:
        #     if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
        #         layer.reset_parameters()

        # self.ind_dec_mean.reset_parameters()
        # self.ind_dec_log_var.reset_parameters()

        for layer in self.decoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.reset_parameters()

        self.dec_mean.reset_parameters()
        self.dec_log_var.reset_parameters()

    # Test

    def freeze_decoder(self):
        self.z_prior_mean.requires_grad_(False)

        for layer in self.shared_decoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.requires_grad_(False)

        for layer in self.decoder:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.requires_grad_(False)

        self.dec_mean.requires_grad_(False)
        self.dec_log_var.requires_grad_(False)

    def test_loglik(self, X, M, num_z_samples):
        # S = torch.randint(high=self.hparams.s_encoder_hidden_dims[-1],
        #                   size=(num_z_samples,),
        #                   device=X.device)
        # S_onehot = torch.zeros(num_z_samples, self.hparams.s_encoder_hidden_dims[-1],
        #                        device=X.device)
        # S_onehot.scatter_(dim=1, index=S.unsqueeze(-1), value=1)

        # Z_mean_p = self.z_prior_mean(S_onehot)

        # Z_norm = distr.Normal(loc=Z_mean_p,
        #                       scale=torch.ones_like(Z_mean_p))
        # Z = Z_norm.sample(sample_shape=(X.shape[0],))
        # Z = Z.transpose(0, 1).reshape(-1, Z.shape[-1])
        # S_onehot = S_onehot.unsqueeze(1).expand(-1, X.shape[0], -1).reshape(-1, S_onehot.shape[-1])

        # X_tilde_mean, X_tilde_log_var = self.decode(Z, S_onehot)
        # X_tilde_mean = X_tilde_mean.reshape(num_z_samples, *X.shape)
        # X_tilde_log_var = X_tilde_log_var.reshape(num_z_samples, *X.shape)
        # X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))

        # log_probs = X_norm.log_prob(X).sum(dim=-1)
        # log_probs = log_probs.mean(dim=0)


        # Set missing inputs to zero
        X = X * M

        #
        # Inference part
        #

        if hasattr(self.hparams, 'local_vi') and self.hparams.local_vi:
            S_logit_orig, S, Z_mean_q, Z_log_var_q = self.encode_local(I, self.hparams.num_z_samples)
            S_logit = S_logit_orig.unsqueeze(0).expand(num_z_samples, -1, -1)
        else:
            # Encode X to get S
            S_logit = self.s_encoder(X)  # (N, D_s)
            # Gumbel-softmax trick to sample S
            # NOTE: could also try hard=True instead
            S_logit = S_logit.unsqueeze(0).expand(num_z_samples, -1, -1)
            S = F.gumbel_softmax(S_logit, tau=self.hparams.s_tau_param, hard=False)  # (num_z, N, D_s)
            S = S.reshape(-1, S.shape[-1])

            # Encode X and S to get Z
            XS = torch.cat([X.unsqueeze(0).expand(num_z_samples, -1, -1).reshape(-1, X.shape[-1]), S], dim=-1)
            Z_mean_q, Z_log_var_q = self.encode(XS)

        # Sample latent variables Z
        Z_norm = distr.Normal(loc=Z_mean_q, scale=torch.exp(Z_log_var_q/2))
        Z = Z_norm.rsample()

        #
        # Generative part
        #

        # p(z | s)
        Z_mean_p = self.z_prior_mean(S)
        # Z_log_var_p = torch.zeros_like(Z_mean_p)

        X_tilde_mean, X_tilde_log_var = self.decode(Z, S)

        X_tilde_mean = X_tilde_mean.reshape(num_z_samples, *X.shape)
        X_tilde_log_var = X_tilde_log_var.reshape(num_z_samples, *X.shape)

        # Compute marginal log-prob

        # value of q(s | x) and p(s) (const)
        log_pi = torch.log_softmax(S_logit, dim=-1)
        # ent_qs = -(torch.exp(log_pi)*log_pi).sum(dim=-1)
        log_qs = (S.reshape(*S_logit.shape)*log_pi).sum(dim=-1)
        log_ps = torch.log(torch.tensor(S_logit.shape[-1], dtype=torch.float, device=X.device))

        # value of q(z | x, s) and p(z | s)
        log_qz = Z_norm.log_prob(Z).sum(dim=-1).reshape(num_z_samples, -1)
        log_pz = (-1/2*torch.log(torch.tensor(2*np.pi, device=X.device)) - 1/2 * (Z - Z_mean_p)**2).sum(dim=-1)
        log_pz = log_pz.reshape(num_z_samples, -1)

        # Log-likelihood
        X_norm = distr.Normal(loc=X_tilde_mean, scale=torch.exp(X_tilde_log_var/2))
        X_log_prob = X_norm.log_prob(X) * M
        X_log_prob = X_log_prob.sum(dim=-1)

        return (torch.logsumexp(X_log_prob + log_pz + log_ps - log_qs - log_qz, dim=0)
                - torch.log(torch.tensor(num_z_samples, dtype=torch.float, device=X.device)))

    # Local-VI

    def init_local_params(self, data_count):
        # Initialise to the prior
        self.q_local_s = torch.nn.Parameter(
                                data=torch.log(torch.ones(
                                        data_count,
                                        self.hparams.s_encoder_hidden_dims[-1],
                                        dtype=torch.float) / self.hparams.s_encoder_hidden_dims[-1]),
                                requires_grad=True)

        prior_means = self.z_prior_mean.weight.T + self.z_prior_mean.bias
        self.q_local_mean = torch.nn.Parameter(
                                data=prior_means.unsqueeze(0).repeat(data_count, 1, 1),
                                requires_grad=True)
        self.q_local_log_var = torch.nn.Parameter(
                                data=torch.zeros(
                                        data_count,
                                        self.hparams.s_encoder_hidden_dims[-1],
                                        self.hparams.z_encoder_hidden_dims[-1],
                                        dtype=torch.float),
                                requires_grad=True)

    def encode_local(self, I, num_z_samples):
        # Sample S
        S_logit = self.q_local_s[I, :]
        S_logit_orig = S_logit
        # Sample num_z_samples of S
        S_logit = S_logit.unsqueeze(0).expand(num_z_samples, -1, -1)
        # Gumbel-softmax trick to sample S
        # NOTE: could also try hard=True instead
        S = F.gumbel_softmax(S_logit, tau=self.hparams.s_tau_param, hard=True)
        S = S.reshape(-1, S.shape[-1])

        q_means = self.q_local_mean[I, :, :].unsqueeze(0).expand(num_z_samples, -1, -1, -1)
        q_means = q_means.reshape(-1, q_means.shape[-2], q_means.shape[-1])
        Z_mean = (S.unsqueeze(-2) @ q_means).squeeze(-2)
        q_logvars = self.q_local_log_var[I, :, :].unsqueeze(0).expand(num_z_samples, -1, -1, -1)
        q_logvars = q_logvars.reshape(-1, q_logvars.shape[-2], q_logvars.shape[-1])
        Z_log_var = (S.unsqueeze(-2) @ q_logvars).squeeze(-2)

        return S_logit_orig, S, Z_mean, Z_log_var
