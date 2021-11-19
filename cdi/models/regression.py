import pytorch_lightning as pl
import torch
import torch.nn as nn

from jsonargparse import ArgumentParser


class UnivariateRegression(pl.LightningModule):
    """
    Regression function used in baseline that predicts the d-th variable
    assuming the other variables are observed.
    """
    def __init__(self, args):
        super(UnivariateRegression, self).__init__()
        self.args = args.reg_model

        assert not (self.args.activation != 'lrelu'
                    and self.args.activation != 'sigmoid'), \
            'Activation not supported!'

        # Add input dimension to the list
        hidden_dims = [self.args.input_dim] + self.args.hidden_layer_dims

        # Create hidden layers
        hidden_layers = []
        for i in range(0, len(hidden_dims)-1):
            hidden_layers.append(nn.Linear(in_features=hidden_dims[i],
                                           out_features=hidden_dims[i+1]))

            if self.args.activation == 'sigmoid':
                hidden_layers.append(nn.Sigmoid())
            elif self.args.activation == 'lrelu':
                hidden_layers.append(nn.LeakyReLU())

        # One final output layer
        hidden_layers.append(nn.Linear(in_features=hidden_dims[-1],
                                       out_features=1))

        self.layers = nn.Sequential(*hidden_layers)

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parser_mode='jsonnet',
                                parents=[parent_parser],
                                add_help=False)
        parser.add_argument('--reg_model.input_dim',
                            type=int, required=True,
                            help='Dimensionality of input.')
        parser.add_argument('--reg_model.hidden_layer_dims',
                            type=int, nargs='+', required=True,
                            help='Dimensionalities of hidden layers.')
        parser.add_argument('--reg_model.activation',
                            type=str, required=True,
                            help='Activation: lrelu or sigmoid.',
                            choices=['lrelu', 'sigmoid'])
        return parser

    def forward(self, X, M):
        """
        Predict a variable given the others.
        Args:
            X (N, D-1): observed variables (without the missing variable)
            M (N, D-1): missingness mask, 0 - missing, 1 - observed, or None
        Returns:
            d (float): d-th variable prediction given all other variables.
        """
        # Add missing vector to the inputs
        if M is not None:
            # Cast type to X so can concat
            M = M.type_as(X)
            X = torch.cat((X, M), dim=1)

        # Predict the target variable
        return self.layers(X)

    def reset_parameters(self):
        for layer in self.layers:
            if not isinstance(layer, (nn.LeakyReLU, nn.Sigmoid)):
                layer.reset_parameters()
