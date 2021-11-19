import torch
import torch.testing as tt

from cdi.layers.channel_linear import (LinearWithChannels,
                                       LinearWithSharedChannels)


def test_linear_with_channels_output():
    C, D, K = 3, 2, 4
    layer = LinearWithChannels(D, K, C, bias=True)

    X = torch.tensor([[1, 2],
                      [3, 4],
                      [5, 6],
                      [7, 8],
                      [9, 10]], dtype=torch.float)

    weights = torch.tensor([[[1, 1, 1, 1],
                             [2, 2, 2, 2]],
                            [[3, 3, 3, 3],
                             [4, 4, 4, 4]],
                            [[5, 5, 5, 5],
                             [6, 6, 6, 6]]], dtype=torch.float).transpose(-1, -2)

    bias = torch.tensor([[[1, 1, 1, 1]],
                         [[2, 2, 2, 2]],
                         [[3, 3, 3, 3]]], dtype=torch.float)

    # Set layer weights and biases
    layer.weight.data = weights
    layer.bias.data = bias

    expected_out = torch.tensor([[[5, 5, 5, 5],
                                  [11, 11, 11, 11],
                                  [17, 17, 17, 17]],
                                 [[11, 11, 11, 11],
                                  [25, 25, 25, 25],
                                  [39, 39, 39, 39]],
                                 [[17, 17, 17, 17],
                                  [39, 39, 39, 39],
                                  [61, 61, 61, 61]],
                                 [[23, 23, 23, 23],
                                  [53, 53, 53, 53],
                                  [83, 83, 83, 83]],
                                 [[29, 29, 29, 29],
                                  [67, 67, 67, 67],
                                  [105, 105, 105, 105]]], dtype=torch.float)
    expected_out = expected_out + bias.transpose(0, 1)

    out = layer(X.unsqueeze(1))

    tt.assert_allclose(out, expected_out, msg="Incorrect output!")


def test_linear_with_channels_with_mask():
    # Set channels to number of dimensions (for our variational network use-case)
    C = D = 3
    K = 4

    X = torch.tensor([[1, 2, 3],
                      [3, 4, 5],
                      [5, 6, 7],
                      [7, 8, 9],
                      [9, 10, 11]], dtype=torch.float)

    weights = torch.tensor([[[1, 1, 1, 1],
                             [2, 2, 2, 2],
                             [3, 3, 3, 3]],
                            [[3, 3, 3, 3],
                             [4, 4, 4, 4],
                             [5, 5, 5, 5]],
                            [[5, 5, 5, 5],
                             [6, 6, 6, 6],
                             [7, 7, 7, 7]]], dtype=torch.float).transpose(-1, -2)

    bias = torch.tensor([[[1, 1, 1, 1]],
                         [[2, 2, 2, 2]],
                         [[3, 3, 3, 3]]], dtype=torch.float)

    # We want the mask to be diagonal on inputs, equivalent to masking some inputs
    mask = (torch.ones(C,D) - torch.eye(C, D)).unsqueeze(-2)

    layer = LinearWithChannels(D, K, C, mask=mask, bias=True)

    # Set layer weights and biases
    layer.weight.data = weights
    layer.bias.data = bias

    masked_weights = torch.tensor([[[0, 0, 0, 0],
                                    [2, 2, 2, 2],
                                    [3, 3, 3, 3]],
                                   [[3, 3, 3, 3],
                                    [0, 0, 0, 0],
                                    [5, 5, 5, 5]],
                                   [[5, 5, 5, 5],
                                    [6, 6, 6, 6],
                                    [0, 0, 0, 0]]], dtype=torch.float).transpose(-1, -2)

    expected_out = (X.unsqueeze(1).transpose(0, 1) @ masked_weights.transpose(-1, -2)).transpose(0, 1)
    expected_out = expected_out + bias.transpose(0, 1)

    out = layer(X.unsqueeze(1))

    tt.assert_allclose(out, expected_out, msg="Incorrect output for masked weights!")

