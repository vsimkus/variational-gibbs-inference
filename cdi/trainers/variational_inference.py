import sys

import torch.optim as optim

from cdi.models.variational_distribution import GaussVarMADE
from cdi.trainers.mc_expectation_maximisation import MCEM


def get_var_distribution_class_from_string(model):
    if model == 'made':
        return GaussVarMADE
    else:
        print((f'No such variational model `{model}`!'))
        sys.exit()


class VI(MCEM):
    """
    Standard Variational Inference estimation.
    """
    def __init__(self, hparams):
        super().__init__(hparams)

        # Prepare variational model
        VarDistr = get_var_distribution_class_from_string(
                        self.hparams.variational_model)
        self.variational_model = VarDistr(self.hparams)

        self.variational_model.reset_parameters()

    @staticmethod
    def add_model_args(parent_parser, args=None):
        parser = super(VI, VI).add_model_args(parent_parser)

        # VI args
        parser.add_argument('--variational_model',
                            type=str, required=True,
                            help='Type of variational model.',
                            choices=['made'])
        parser.add_argument('--var_optim.optimiser',
                            type=str, required=True,
                            choices=['adam', 'amsgrad'])
        parser.add_argument('--var_optim.learning_rate',
                            type=float, required=True,
                            help=('The learning rate using in Adam '
                                  'optimiser for the var_model.'))
        parser.add_argument('--var_optim.weight_decay_coeff',
                            type=float, required=True,
                            help=('The weight decay used in Adam '
                                  'optimiser for the var_model.'))
        parser.add_argument('--var_optim.epsilon',
                            type=float, default=1e-8,
                            help=('Adam optimiser epsilon parameter.'))

        # Add variational model parameters
        temp_args, _ = parser._parse_known_args(args)
        VarDistr = get_var_distribution_class_from_string(
                        temp_args.variational_model)
        parser = VarDistr.add_model_args(parser)

        return parser

    def configure_optimizers(self):
        optimiser = super().configure_optimizers()
        # Separate optimisers for the model and the variational
        # distribution since the hyperparameters might need to be
        # different.
        if self.hparams.var_optim.optimiser == 'adam':
            var_opt = optim.Adam(
                    self.variational_model.parameters(),
                    amsgrad=False,
                    lr=self.hparams.var_optim.learning_rate,
                    weight_decay=self.hparams.var_optim.weight_decay_coeff,
                    eps=self.hparams.var_optim.epsilon)
        elif self.hparams.var_optim.optimiser == 'amsgrad':
            var_opt = optim.Adam(
                    self.variational_model.parameters(),
                    amsgrad=True,
                    lr=self.hparams.var_optim.learning_rate,
                    weight_decay=self.hparams.var_optim.weight_decay_coeff,
                    eps=self.hparams.var_optim.epsilon)
        else:
            sys.exit('No such optimizer for the variational CDI!')

        if isinstance(optimiser, tuple):
            optimiser[0][0].add_optimisers(var_model_opt=var_opt)
        else:
            optimiser.add_optimisers(var_model_opt=var_opt)

        # self.optim = optimiser
        return optimiser

    def sample_missing_values(self, batch, K):
        """
        Sample K examples of the missing values for each x, indicated by M.
        Used for approximating the expectation in ELBO.
        Args:
            batch:
                X (N, D): input batch
                M (N, D): missing data input batch binary mask,
                    1 - observed, 0 - missing.
                I (N): indices of the data
            K (int): number of samples for each missing value
        Returns:
            x_samples (K, N, D): K posterior samples
            entropy (N): entropy, -inf for observed values
        """
        X, M, _ = batch

        X_samples, log_probs = self.variational_model.sample(X, M, K)
        X_samples = X_samples.reshape(K, *(X.shape))
        # Compute sample-avg entropy
        entropy = -log_probs.mean(dim=0)

        # Make sure the optimiser for var model runs
        self.optim.add_run_opt('var_model_opt')
        return X_samples, entropy

    def on_epoch_start(self):
        super().on_epoch_start()
        self.variational_model.on_epoch_start()

    def training_epoch_end(self, outputs):
        results = super().training_epoch_end(outputs)

        # Add epoch-level stats
        # TODO: handle this in the Variational model class
        results['log']['cum_var_calls'] = self.variational_model.cum_batch_size_called

        return results
