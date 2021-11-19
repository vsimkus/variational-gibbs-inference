import torch.nn as nn


class SequentialWithMultiInputs(nn.Sequential):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def forward(self, x, *args):
        for module in self:
            if not isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid)):
                x = module(x, *args)
            else:
                x = module(x)
        return x
