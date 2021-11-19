import pytorch_lightning as pl
import torch
import torch.nn as nn


class MultinomialLogRegMisModel(pl.LightningModule):
    """
    A simple missingness model that performs a Multinomial
    Logistic Regression over _known_ missingness patterns.
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams.mis_model

        # If the model already knows the shapes - use them
        # Necessary for leading a pretrained model
        if hasattr(self.hparams, 'weights_shape'):
            weights_shape = self.hparams.weights_shape
            balances_shape = self.hparams.balances_shape
            # Also reload class map
            self.class_to_idx = self.hparams.class_to_idx
        else:
            weights_shape = (1, 1, )
            balances_shape = (1, )

        self.weights = torch.nn.Parameter(data=torch.empty(
                                                  *weights_shape,
                                                  dtype=torch.float
                                          ),
                                          requires_grad=True)
        self.balances = torch.nn.Parameter(data=torch.empty(
                                                  *balances_shape,
                                                  dtype=torch.float
                                           ),
                                           requires_grad=True)

        self.cum_batch_size_called = 0

    def initialise(self, patterns):
        self.weights.data = torch.randn(self.hparams.data_dim,
                                        # +1 for the fully-observed pattern
                                        patterns.shape[0]+1,
                                        dtype=torch.float)
        self.balances.data = torch.randn(
                                         # +1 for the fully-observed pattern
                                         patterns.shape[0] + 1,
                                         dtype=torch.float)

        classes = bool2long(patterns)
        self.class_to_idx = {clazz.item(): i for i, clazz in enumerate(classes)}

        # Add pattern for fully-observed (all ones)
        full_pattern = torch.ones(patterns.shape[-1], dtype=torch.bool)
        full_class = bool2long(full_pattern).item()
        if full_class not in self.class_to_idx:
            self.class_to_idx[full_class] = len(self.class_to_idx)

        # Save shapes and class map so that the model can be reloaded
        self.hparams.weights_shape = self.weights.shape
        self.hparams.balances_shape = self.balances.shape
        self.hparams.class_to_idx = self.class_to_idx

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--mis_model.data_dim', type=int,
                            help='Observed dimensionality.')
        return parser

    def forward(self, X, M):
        # Track training calls
        if self.training:
            self.cum_batch_size_called += X.shape[0]

        # Calculate class scores
        scores = X @ self.weights + self.balances

        # Convert miss masks to longs and then to class indices
        target_class = bool2long(M)
        target_indices = torch.empty(M.shape[0], dtype=torch.long)
        # Slow but Pytorch doesn't have a solution for such mapping
        # Could improve a little by embedding this in C
        for clazz, idx in self.class_to_idx.items():
            target_indices[target_class == clazz] = idx

        loss = nn.CrossEntropyLoss(reduction='none')
        return -loss(scores, target_indices)

    def reset_parameters(self):
        pass

    def on_epoch_start(self):
        self.cum_batch_size_called = 0


def bool2long(bool_tensor):
    """
    Coverts a boolean tensor to long tensor, the final dimension is squashed.
    """
    exp_vec = 2**torch.arange(bool_tensor.shape[-1], dtype=torch.long)
    return bool_tensor.long() @ (torch.flip(exp_vec, (-1, )))
