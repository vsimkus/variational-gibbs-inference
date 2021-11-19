import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from nflows.transforms.linear import Linear


class LULinear(Linear):
    """A linear transform where we parameterize the LU decomposition of the weights."""

    def __init__(self, features, using_cache=False, identity_init=True, eps=1e-3):
        super().__init__(features, using_cache)

        self.eps = eps

        self.register_buffer("lower_indices",
                             torch.tril_indices(features, features, offset=-1),
                             persistent=False)
        self.register_buffer("upper_indices",
                             torch.triu_indices(features, features, offset=1),
                             persistent=False)
        self.register_buffer("diag_indices",
                             torch.arange(features),
                             persistent=False)

        n_triangular_entries = ((features - 1) * features) // 2

        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.unconstrained_upper_diag = nn.Parameter(torch.zeros(features))

        self._initialize(identity_init)

    def _initialize(self, identity_init):
        init.zeros_(self.bias)

        if identity_init:
            init.zeros_(self.lower_entries)
            init.zeros_(self.upper_entries)
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_upper_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.lower_entries, -stdv, stdv)
            init.uniform_(self.upper_entries, -stdv, stdv)
            init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)

    def _create_lower_upper(self):
        lower = self.lower_entries.new_zeros(self.features, self.features)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        # The diagonal of L is taken to be all-ones without loss of generality.
        lower[self.diag_indices, self.diag_indices] = 1.0

        upper = self.upper_entries.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices, self.diag_indices] = self.upper_diag

        return lower, upper

    def forward_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self._create_lower_upper()

        # If the L/U matrix dimensions are smaller than the batch size of
        # inputs then it is more efficient to first multiply the L and U
        if inputs.shape[0] > lower.shape[0]:
            outputs = F.linear(inputs, lower @ upper, self.bias)
        else:
            outputs = F.linear(inputs, upper)
            outputs = F.linear(outputs, lower, self.bias)

        logabsdet = self.logabsdet() * inputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self._create_lower_upper()
        outputs = inputs - self.bias
        outputs, _ = torch.triangular_solve(
            outputs.t(), lower, upper=False, unitriangular=True
        )
        outputs, _ = torch.triangular_solve(
            outputs, upper, upper=True, unitriangular=False
        )
        outputs = outputs.t()

        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * inputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    def weight(self):
        """Cost:
            weight = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        return lower @ upper

    def weight_inverse(self):
        """Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        identity = torch.eye(self.features, self.features)
        lower_inverse, _ = torch.triangular_solve(
            identity, lower, upper=False, unitriangular=True
        )
        weight_inverse, _ = torch.triangular_solve(
            lower_inverse, upper, upper=True, unitriangular=False
        )
        return weight_inverse

    @property
    def upper_diag(self):
        return F.softplus(self.unconstrained_upper_diag) + self.eps

    def logabsdet(self):
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(torch.log(self.upper_diag))
