import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights

    Based on implementation by Andrej Karpathy https://github.com/karpathy/pytorch-made
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(mask.T)

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MaskedLinearPerExample(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(1, out_features, in_features))

    def set_masks(self, mask):
        self.mask.data = torch.transpose(mask, 2, 1)

    def forward(self, X, K, idx_mask):
        mask = self.mask[idx_mask, :, :].expand(K, -1, -1, -1)
        mask = mask.reshape(-1, *self.mask.shape[-2:])
        return ((mask * self.weight) @ X.unsqueeze(-1)).squeeze(-1) + self.bias
