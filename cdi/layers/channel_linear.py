import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearWithChannels(nn.Module):
    def __init__(self, in_features, out_features, channels, mask=None, bias=True, reduce_mem=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels

        self.mask = mask
        self.reduce_mem = reduce_mem

        self.weight = nn.Parameter(torch.Tensor(channels, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(channels, 1, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Use default Linear layer initialisation - Kaiming init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, channel_sample_mask=None, channel_sample_sparse_idx=None):
        # Change from (B, C, D) to (C, B, D)
        x = x.transpose(0, 1)

        weight = self.weight
        # Mask weights
        if self.mask is not None:
            weight = weight * self.mask.to(x.device)

        if x.shape[0] != weight.shape[0] and x.shape[0] == 1:
            x = x.expand(weight.shape[0], -1, -1)
            ret = torch.bmm(x, weight.transpose(-1, -2))
        elif self.reduce_mem:
            assert channel_sample_mask is not None and channel_sample_sparse_idx is not None,\
                'The `channel_sample_mask` and `channel_sample_sparse_idx` should not be None, when memory optimisation is used!'
            ret = MatrixMultiplicationStoreSparseInput.apply(x, weight.transpose(-1, -2),
                                                             channel_sample_mask,
                                                             channel_sample_sparse_idx)
        else:
            ret = torch.bmm(x, weight.transpose(-1, -2))

        # ret = torch.bmm(x, weight.transpose(-1, -2))
        # ret = MatrixMultiplicationStoreSparseInput.apply(x, weight.transpose(-1, -2), ., .)
        if self.bias is not None:
            ret = ret + self.bias

        # Change back to (B, C, D)
        return ret.transpose(0, 1)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, channels={self.channels}, mask={self.mask is not None}, bias={self.bias is not None}, reduce_mem={self.reduce_mem}"


class LinearWithSharedChannels(nn.Linear):
    """
    Basically just a linear model with 1 channel but allows
    masking some weights in some channels, so results in a channeled
    layer with shared weights.
    """
    def __init__(self, in_features, out_features, channels, mask, bias=True, reduce_mem=True):
        super().__init__(in_features, out_features, bias=bias)

        self.channels = channels
        self.mask = mask
        self.reduce_mem = reduce_mem

    def forward(self, x, channel_sample_mask, channel_sample_sparse_idx):
        # Change from (B, C, D) to (C, B, D)
        x = x.transpose(0, 1)

        # Repeat and mask weights
        if self.mask is not None:
            weight = self.mask.to(x.device) * self.weight.unsqueeze(0).expand(self.channels, -1, -1)
        else:
            weight = self.weight.unsqueeze(0)

        if x.shape[0] == weight.shape[0] and x.shape[0] == 1:
            # ret = torch.bmm(x, weight.transpose(-1, -2))
            # Just run regular linear forward
            return F.linear(x.squeeze(0), self.weight, self.bias).unsqueeze(1)
        elif x.shape[0] != weight.shape[0] and x.shape[0] == 1:
            x = x.expand(weight.shape[0], -1, -1)
            ret = torch.bmm(x, weight.transpose(-1, -2))
        elif x.shape[0] != weight.shape[0] and weight.shape[0] == 1 and self.reduce_mem:
            weight = weight.expand(x.shape[0], -1, -1)
            ret = MatrixMultiplicationStoreSparseInput.apply(x, weight.transpose(-1, -2),
                                                             channel_sample_mask,
                                                             channel_sample_sparse_idx)
        elif x.shape[0] != weight.shape[0] and weight.shape[0] == 1 and not self.reduce_mem:
            weight = weight.expand(x.shape[0], -1, -1)
            ret = torch.bmm(x, weight.transpose(-1, -2))
        elif self.reduce_mem:
            ret = MatrixMultiplicationStoreSparseInput.apply(x, weight.transpose(-1, -2),
                                                             channel_sample_mask,
                                                             channel_sample_sparse_idx)
        else:
            ret = torch.bmm(x, weight.transpose(-1, -2))

        # ret = torch.bmm(x, weight.transpose(-1, -2))
        # ret = MatrixMultiplicationStoreSparseInput.apply(x, weight.transpose(-1, -2), ., .)
        if self.bias is not None and not (x.shape[0] == weight.shape[0] and x.shape[0] == 1):
            ret = ret + self.bias

        # Change back to (B, C, D)
        return ret.transpose(0, 1)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, channels={self.channels}, mask={self.mask is not None}, bias={self.bias is not None}, reduce_mem={self.reduce_mem}"


class MatrixMultiplicationStoreSparseInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, mask, mask_nonzero):
        with torch.no_grad():
            ret = torch.bmm(X, weight)

            # Store sparse matrix
            X_sparse = torch.sparse_coo_tensor(mask_nonzero.t(), X[mask], size=X.shape, device=X.device)

            # If input doesn't need gradient, then don't store weight matrix
            if not ctx.needs_input_grad[0]:
                ctx.save_for_backward(X_sparse)
            else:
                ctx.save_for_backward(X_sparse, weight)

        # Return dense matrix
        return ret

    @staticmethod
    def backward(ctx, grad_ret):
        grad_X, grad_weights = None, None

        # Retrieve stored tensors
        if not ctx.needs_input_grad[0]:
            X_sparse = ctx.saved_tensors
        else:
            X_sparse, weight = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_X = torch.bmm(grad_ret, weight.transpose(-1, -2))

        if ctx.needs_input_grad[1]:
            X = X_sparse.to_dense()
            grad_weights = torch.bmm(X.transpose(-1, -2), grad_ret)

        return grad_X, grad_weights, None, None
