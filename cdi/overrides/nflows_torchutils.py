import torch


def searchsorted(bin_locations, inputs, eps=1e-6):
    # searchsorted is not differentiable
    bin_locations = bin_locations.detach()
    inputs = inputs.detach()

    bin_locations[..., -1] += eps
    return (torch.searchsorted(bin_locations, inputs[..., None], right=True) - 1)[..., 0]
